#!/usr/bin/env python3
"""Delay-aware predictive feedback control for delayed sensing.

The feedback-ablation analysis showed that 60-120 s delays erode robust
feasibility. This script tests whether a lightweight state predictor can
recover part of that lost feasibility by extrapolating delayed measurements
forward before selecting the irradiation command.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_feedback_control import (
    CONTROL_DT_S,
    MAX_PROTOCOL_S,
    REFERENCE_INTENSITY_W_CM2,
    REFERENCE_PROTOCOL_S,
)
from run_feedback_ablation_noise_delay import (
    choose_ablation_intensity,
    noisy_measurement,
    simulate_ablation_case,
)
from run_first_model import (
    EvidenceDefaults,
    ModelConfig,
    build_pennes_backward_euler_tridiagonal,
    nanoparticle_profile,
    optical_fluence_profile,
    relative_gans_absorption,
    solve_tridiagonal,
)
from run_robust_chance_optimization import sample_uncertainty, success_flags, wilson_lower_bound


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "predictive_delay_control"
FIG = ROOT / "figures" / "predictive_delay_control"

N_DRAWS = 80
SEED = 20260709


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def delay_scenarios() -> list[dict]:
    return [
        {"scenario": "delay_60s", "noise_fraction": 0.00, "delay_s": 60},
        {"scenario": "delay_120s", "noise_fraction": 0.00, "delay_s": 120},
        {"scenario": "noise_10pct_delay_60s", "noise_fraction": 0.10, "delay_s": 60},
        {"scenario": "noise_10pct_delay_120s", "noise_fraction": 0.10, "delay_s": 120},
    ]


def predictive_policies() -> list[dict]:
    return [
        {
            "controller": "predictive_linear_full",
            "signals": ("temperature", "oxygen", "ros"),
            "max_intensity_w_cm2": 1.30,
            "prediction_gain": 1.00,
            "lookahead_s": 0.0,
            "thermal_margin_c": 0.0,
            "estimator_window": 2,
        },
        {
            "controller": "predictive_guarded_full",
            "signals": ("temperature", "oxygen", "ros"),
            "max_intensity_w_cm2": 1.15,
            "prediction_gain": 1.00,
            "lookahead_s": 30.0,
            "thermal_margin_c": 1.0,
            "estimator_window": 2,
        },
        {
            "controller": "predictive_temp_ros",
            "signals": ("temperature", "ros"),
            "max_intensity_w_cm2": 1.15,
            "prediction_gain": 1.00,
            "lookahead_s": 30.0,
            "thermal_margin_c": 1.0,
            "estimator_window": 2,
        },
        {
            "controller": "predictive_smoothed_full",
            "signals": ("temperature", "oxygen", "ros"),
            "max_intensity_w_cm2": 1.30,
            "prediction_gain": 0.85,
            "lookahead_s": 0.0,
            "thermal_margin_c": 0.35,
            "estimator_window": 6,
        },
        {
            "controller": "predictive_thermal_floor_full",
            "signals": ("temperature", "oxygen", "ros"),
            "max_intensity_w_cm2": 1.30,
            "prediction_gain": 0.85,
            "lookahead_s": 0.0,
            "thermal_margin_c": 0.35,
            "estimator_window": 6,
            "thermal_floor_warm_fraction": 0.48,
            "thermal_floor_temp_c": 48.6,
            "thermal_floor_fraction": 0.58,
            "maintenance_intensity_w_cm2": 0.22,
            "warm_target": 0.56,
        },
        {
            "controller": "predictive_reduced_mpc_full",
            "signals": ("temperature", "oxygen", "ros"),
            "max_intensity_w_cm2": 1.30,
            "prediction_gain": 0.85,
            "lookahead_s": 0.0,
            "thermal_margin_c": 0.35,
            "estimator_window": 6,
            "mpc_horizon_steps": 4,
            "mpc_target_temp_c": 47.4,
            "mpc_target_warm_fraction": 0.56,
            "mpc_target_ros_fraction": 0.88,
            "mpc_energy_weight": 0.045,
            "mpc_overheat_weight": 4.8,
            "mpc_oxygen_weight": 0.9,
            "mpc_thermal_weight": 1.8,
            "mpc_ros_weight": 1.2,
        },
    ]


def full_feedback_policy() -> dict:
    return {
        "controller": "delayed_full_feedback",
        "signals": ("temperature", "oxygen", "ros"),
        "max_intensity_w_cm2": 1.30,
    }


def state_at_time(history: list[dict], state_time_s: float) -> dict:
    if not history:
        raise ValueError("empty state history")
    index = int(round(state_time_s / CONTROL_DT_S))
    index = min(max(index, 0), len(history) - 1)
    return history[index]


def estimate_current_state(
    measurements: list[dict],
    current_time_s: float,
    delay_s: float,
    policy: dict,
) -> dict:
    latest = measurements[-1]
    estimated = {
        "max_temperature_c": latest["max_temperature_c"],
        "warm_fraction": latest["warm_fraction"],
        "mean_oxygen_um": latest["mean_oxygen_um"],
        "ros_threshold_fraction": latest["ros_threshold_fraction"],
    }
    window = int(policy.get("estimator_window", 2))
    windowed = measurements[-window:]
    if len(windowed) < 2:
        return estimated

    horizon = max(current_time_s - latest["state_time_s"], 0.0)
    horizon = float(policy["prediction_gain"]) * horizon + float(policy["lookahead_s"])
    times = np.array([row["state_time_s"] for row in windowed], dtype=float)
    times = times - times[-1]
    slope_limits = {
        "max_temperature_c": (-0.045, 0.060),
        "warm_fraction": (-0.006, 0.006),
        "mean_oxygen_um": (-0.18, 0.18),
        "ros_threshold_fraction": (0.0, 0.006),
    }
    for key in ("max_temperature_c", "warm_fraction", "mean_oxygen_um", "ros_threshold_fraction"):
        values = np.array([row[key] for row in windowed], dtype=float)
        if len(np.unique(times)) < 2:
            slope = 0.0
        else:
            slope = float(np.polyfit(times, values, deg=1)[0])
        lo, hi = slope_limits[key]
        slope = float(np.clip(slope, lo, hi))
        estimated[key] = latest[key] + slope * horizon

    estimated["max_temperature_c"] += float(policy["thermal_margin_c"])
    estimated["warm_fraction"] = float(np.clip(estimated["warm_fraction"], 0.0, 1.0))
    estimated["mean_oxygen_um"] = float(np.clip(estimated["mean_oxygen_um"], 0.5, 40.0))
    estimated["ros_threshold_fraction"] = float(np.clip(estimated["ros_threshold_fraction"], 0.0, 1.0))
    return estimated


def choose_predictive_intensity(policy: dict, estimated: dict, delay_s: float) -> float:
    temp = float(estimated["max_temperature_c"])
    warm = float(estimated["warm_fraction"])
    ros = float(estimated["ros_threshold_fraction"])
    oxygen = float(estimated["mean_oxygen_um"])

    if policy["controller"] == "predictive_reduced_mpc_full":
        return choose_reduced_mpc_intensity(policy, temp, warm, ros, oxygen, delay_s)

    if policy["controller"].startswith("predictive_thermal_floor"):
        long_delay_temp = float(policy.get("long_delay_safety_c", 51.0))
        mid_delay_temp = float(policy.get("mid_delay_safety_c", 51.5))
        long_delay_ros = float(policy.get("long_delay_ros_stop", 0.78))
        mid_delay_ros = float(policy.get("mid_delay_ros_stop", 0.80))
        long_delay_maintenance = float(policy.get("long_delay_maintenance_i", 0.18))
        mid_delay_maintenance = float(policy.get("mid_delay_maintenance_i", 0.20))
        if delay_s >= 120.0 and temp >= long_delay_temp:
            return 0.0 if ros >= long_delay_ros else long_delay_maintenance
        if delay_s >= 60.0 and temp >= mid_delay_temp:
            return 0.0 if ros >= mid_delay_ros else mid_delay_maintenance
        if oxygen < 9.0 and ros < 0.75 and "oxygen" in policy["signals"]:
            return 0.0
        thermal_floor = float(policy.get("thermal_floor_warm_fraction", 0.48))
        floor_temp = float(policy.get("thermal_floor_temp_c", 48.6))
        if warm < thermal_floor and temp < floor_temp:
            floor_intensity = float(policy.get("thermal_floor_fraction", 0.58)) * float(policy["max_intensity_w_cm2"])
            if oxygen < 12.0 and ros < 0.80:
                floor_intensity *= float(policy.get("oxygen_limited_floor_scale", 0.55))
            return float(np.clip(floor_intensity, 0.22, float(policy["max_intensity_w_cm2"])))
        if ros >= float(policy.get("ros_stop_after_floor", 0.82)) and warm >= thermal_floor:
            return 0.0 if temp >= 45.5 else float(policy.get("maintenance_intensity_w_cm2", 0.22))
        return choose_ablation_intensity(policy, estimated)

    if policy["controller"] == "predictive_smoothed_full":
        if delay_s >= 120.0 and temp >= 51.0:
            return 0.0 if ros >= 0.75 else 0.18
        if delay_s >= 60.0 and temp >= 51.5:
            return 0.0 if ros >= 0.80 else 0.20
        return choose_ablation_intensity(policy, estimated)

    if delay_s >= 120.0 and temp >= 49.0:
        return 0.0 if ros >= 0.65 else 0.12
    if delay_s >= 60.0 and temp >= 50.0:
        return 0.0 if ros >= 0.70 else 0.15
    if oxygen < 9.0 and ros < 0.75 and "oxygen" in policy["signals"]:
        return 0.0
    return choose_ablation_intensity(policy, estimated)


def reduced_state_step(policy: dict, state: tuple[float, float, float, float], intensity: float, delay_s: float) -> tuple[float, float, float, float]:
    temp, warm, ros, oxygen = state
    max_i = float(policy["max_intensity_w_cm2"])
    i_norm = intensity / max(max_i, 1e-9)
    delay_load = np.clip(delay_s / 180.0, 0.0, 1.0)

    temp_next = temp + 1.20 * i_norm - 0.18 * max(temp - 37.0, 0.0)
    over_warm = max(temp_next - 50.0, 0.0)
    under_warm = max(43.0 - temp_next, 0.0)
    warm_next = warm + 0.085 * np.exp(-0.20 * under_warm) * i_norm - 0.075 * over_warm - 0.020 * (1.0 - i_norm)
    oxygen_factor = oxygen / (oxygen + 10.0)
    ros_next = ros + 0.070 * i_norm * oxygen_factor * (1.0 - ros)
    oxygen_next = oxygen - 1.05 * i_norm * (1.0 - ros) + 0.55 * (1.0 - oxygen / 22.0)

    temp_next += 0.15 * delay_load * i_norm
    return (
        float(np.clip(temp_next, 37.0, 58.0)),
        float(np.clip(warm_next, 0.0, 1.0)),
        float(np.clip(ros_next, 0.0, 1.0)),
        float(np.clip(oxygen_next, 0.5, 40.0)),
    )


def reduced_mpc_stage_cost(policy: dict, state: tuple[float, float, float, float], intensity: float) -> float:
    temp, warm, ros, oxygen = state
    max_i = float(policy["max_intensity_w_cm2"])
    target_temp = float(policy.get("mpc_target_temp_c", 47.4))
    target_warm = float(policy.get("mpc_target_warm_fraction", 0.56))
    target_ros = float(policy.get("mpc_target_ros_fraction", 0.88))
    thermal_weight = float(policy.get("mpc_thermal_weight", 1.8))
    ros_weight = float(policy.get("mpc_ros_weight", 1.2))
    overheat_weight = float(policy.get("mpc_overheat_weight", 4.8))
    oxygen_weight = float(policy.get("mpc_oxygen_weight", 0.9))
    energy_weight = float(policy.get("mpc_energy_weight", 0.045))

    thermal_deficit = max(target_temp - temp, 0.0) / 6.0 + max(target_warm - warm, 0.0)
    ros_deficit = max(target_ros - ros, 0.0)
    overheat = max(temp - 51.0, 0.0) / 4.0 + max(temp - 55.0, 0.0)
    oxygen_deficit = max(10.0 - oxygen, 0.0) / 10.0
    energy = (intensity / max(max_i, 1e-9)) ** 2
    return float(
        thermal_weight * thermal_deficit**2
        + ros_weight * ros_deficit**2
        + overheat_weight * overheat**2
        + oxygen_weight * oxygen_deficit**2
        + energy_weight * energy
    )


def choose_reduced_mpc_intensity(policy: dict, temp: float, warm: float, ros: float, oxygen: float, delay_s: float) -> float:
    max_i = float(policy["max_intensity_w_cm2"])
    if temp >= 52.2:
        return 0.0 if ros >= 0.78 else min(0.14, max_i)
    if oxygen < 8.0 and ros < 0.82:
        return 0.0
    grid = np.array([0.0, 0.12, 0.22, 0.35, 0.50, 0.68, 0.85, 1.0]) * max_i
    horizon = int(policy.get("mpc_horizon_steps", 4))
    best_i = 0.0
    best_cost = float("inf")
    for intensity in grid:
        state = (temp, warm, ros, oxygen)
        cost = 0.0
        for h in range(horizon):
            state = reduced_state_step(policy, state, float(intensity), delay_s)
            cost += (0.88**h) * reduced_mpc_stage_cost(policy, state, float(intensity))
        if cost < best_cost:
            best_cost = cost
            best_i = float(intensity)
    return best_i


def simulate_predictive_case(
    policy: dict,
    scenario: dict,
    draw: pd.Series,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    ev = EvidenceDefaults()
    cfg = replace(
        ModelConfig(),
        spheroid_radius_m=float(draw["radius_um"]) * 1e-6,
        photosensitizer_um_at_reference_phi=float(draw["photosensitizer_um"]),
        dt_s=CONTROL_DT_S,
    )
    radius = cfg.spheroid_radius_m
    r = np.linspace(0.0, radius, cfg.radial_nodes)
    dr = r[1] - r[0]
    absorption = relative_gans_absorption(808.0, 70.0, 18.0, ev.medium_refractive_index)
    np_profile = nanoparticle_profile(r, radius, cfg.nanoparticle_surface_bias)
    fluence_unit = optical_fluence_profile(r, radius, 808.0, 1.0)
    q_opt_unit = cfg.optical_heat_coeff_w_m3 * float(draw["volume_fraction"]) * absorption * (fluence_unit / 1e4) * np_profile
    perfusion_loss_s = (
        float(draw["perfusion_s"])
        * cfg.blood_density_kg_m3
        * cfg.blood_cp_j_kg_k
        / (cfg.tissue_density_kg_m3 * cfg.tissue_cp_j_kg_k)
    )
    alpha = cfg.tissue_k_w_m_k / (cfg.tissue_density_kg_m3 * cfg.tissue_cp_j_kg_k)
    lower, diag, upper = build_pennes_backward_euler_tridiagonal(
        r, dr, CONTROL_DT_S, alpha, perfusion_loss_s, cfg.external_cooling_s
    )

    rise = np.zeros_like(r)
    temperature = cfg.body_temperature_c + rise
    oxygen_um = ev.oxygen_initial_um * (0.45 + 0.55 * (r / radius) ** 1.6)
    oxygen_um = np.maximum(oxygen_um, 2.0)
    s0_um = cfg.photosensitizer_um_at_reference_phi * (
        float(draw["volume_fraction"]) / cfg.reference_volume_fraction
    ) * np_profile
    ros_dose_um = np.zeros_like(r)
    fluence_unit_mw_cm2 = fluence_unit * 0.1

    delay_s = float(scenario["delay_s"])
    state_history: list[dict] = []
    measurement_history: list[dict] = []
    trace_rows = []
    delivered_energy = 0.0
    light_on_time = 0.0
    max_command = 0.0
    n_steps = int(round(MAX_PROTOCOL_S / CONTROL_DT_S))
    ros_substeps = int(round(CONTROL_DT_S / cfg.ros_dt_s))

    for step in range(n_steps):
        current_time_s = step * CONTROL_DT_S
        ros_dose_mm = ros_dose_um / 1000.0
        true_state = {
            "state_time_s": current_time_s,
            "max_temperature_c": float(temperature.max()),
            "warm_fraction": float(np.mean((temperature >= 43.0) & (temperature <= 50.0))),
            "mean_oxygen_um": float(np.mean(oxygen_um)),
            "ros_threshold_fraction": float(np.mean(ros_dose_mm >= ev.singlet_oxygen_threshold_mm)),
        }
        state_history.append(true_state)

        delayed_time_s = max(0.0, current_time_s - delay_s)
        delayed_state = state_at_time(state_history, delayed_time_s)
        measured = noisy_measurement(delayed_state, scenario, rng)
        measured["state_time_s"] = delayed_time_s
        measurement_history.append(measured)
        estimated = estimate_current_state(measurement_history, current_time_s, delay_s, policy)
        intensity = choose_predictive_intensity(policy, estimated, delay_s)

        max_command = max(max_command, intensity)
        if intensity > 1e-6:
            light_on_time += CONTROL_DT_S
            delivered_energy += intensity * CONTROL_DT_S

        source_k_s = intensity * q_opt_unit / (cfg.tissue_density_kg_m3 * cfg.tissue_cp_j_kg_k)
        rise = solve_tridiagonal(lower, diag, upper, rise + CONTROL_DT_S * source_k_s)
        temperature = cfg.body_temperature_c + rise

        for _ in range(ros_substeps):
            if intensity > 1e-6:
                oxygen_factor = oxygen_um / (oxygen_um + ev.oxygen_beta_um)
                generation_um_s = ev.xi_cm2_mw_s * (intensity * fluence_unit_mw_cm2) * s0_um * oxygen_factor
                ros_dose_um += cfg.ros_dt_s * generation_um_s
                oxygen_um -= cfg.ros_dt_s * generation_um_s
                s0_um -= cfg.ros_dt_s * ev.sigma_per_um * generation_um_s * s0_um / (s0_um + ev.oxygen_delta_um)
                s0_um = np.maximum(s0_um, 0.0)
            oxygen_um += cfg.ros_dt_s * ev.oxygen_perfusion_um_s * (1.0 - oxygen_um / ev.oxygen_initial_um)
            oxygen_um = np.clip(oxygen_um, 0.5, ev.oxygen_initial_um)

        if step % 20 == 0 or step == n_steps - 1:
            trace_rows.append(
                {
                    "controller": policy["controller"],
                    "scenario": scenario["scenario"],
                    "uncertainty_id": int(draw["uncertainty_id"]),
                    "time_s": current_time_s + CONTROL_DT_S,
                    "intensity_w_cm2": intensity,
                    "estimated_max_temperature_c": float(estimated["max_temperature_c"]),
                    "max_temperature_c": float(temperature.max()),
                    "mean_oxygen_um": float(np.mean(oxygen_um)),
                    "ros_threshold_fraction": float(np.mean((ros_dose_um / 1000.0) >= ev.singlet_oxygen_threshold_mm)),
                    "delivered_energy_j_cm2": delivered_energy,
                }
            )

    ros_dose_mm = ros_dose_um / 1000.0
    result = {
        "controller": policy["controller"],
        "scenario": scenario["scenario"],
        "noise_fraction": float(scenario["noise_fraction"]),
        "delay_s": delay_s,
        "uncertainty_id": int(draw["uncertainty_id"]),
        "protocol_duration_s": MAX_PROTOCOL_S,
        "light_on_time_s": light_on_time,
        "delivered_energy_j_cm2": delivered_energy,
        "mean_intensity_w_cm2": delivered_energy / MAX_PROTOCOL_S,
        "max_commanded_intensity_w_cm2": max_command,
        "max_temperature_c": float(temperature.max()),
        "warm_fraction": float(np.mean((temperature >= 43.0) & (temperature <= 50.0))),
        "overheat_fraction": float(np.mean(temperature > 55.0)),
        "ros_threshold_fraction": float(np.mean(ros_dose_mm >= ev.singlet_oxygen_threshold_mm)),
        "hypoxic_failure_fraction": float(np.mean((oxygen_um < 5.0) & (ros_dose_mm < ev.singlet_oxygen_threshold_mm))),
    }
    result["thermal_window_score"] = result["warm_fraction"] - 1.5 * result["overheat_fraction"]
    result["ros_score"] = result["ros_threshold_fraction"] - 0.5 * result["hypoxic_failure_fraction"]
    result["dual_modality_score"] = (
        result["warm_fraction"] * (0.25 + 0.75 * result["ros_threshold_fraction"])
        - 1.5 * result["overheat_fraction"]
        - 0.5 * result["hypoxic_failure_fraction"]
    )
    result["protocol_burden"] = MAX_PROTOCOL_S / REFERENCE_PROTOCOL_S + max_command / REFERENCE_INTENSITY_W_CM2
    result.update(success_flags(result))
    return result, trace_rows


def summarize(outcomes: pd.DataFrame) -> pd.DataFrame:
    grouped = outcomes.groupby(["controller", "scenario"], as_index=False)
    summary = grouped.agg(
        n=("chance_success", "size"),
        chance_success_probability=("chance_success", "mean"),
        strict_safe_dual_probability=("strict_safe_dual", "mean"),
        mean_positive_dual_score=("dual_modality_score", lambda x: float(np.maximum(x, 0.0).mean())),
        median_energy_j_cm2=("delivered_energy_j_cm2", "median"),
        median_tmax_c=("max_temperature_c", "median"),
        q90_tmax_c=("max_temperature_c", lambda x: float(np.quantile(x, 0.90))),
        overheat_failure_probability=("overheat_failure", "mean"),
        ros_failure_probability=("ros_failure", "mean"),
        thermal_failure_probability=("thermal_failure", "mean"),
    )
    lower = outcomes.groupby(["controller", "scenario"])["chance_success"].apply(wilson_lower_bound)
    summary["chance_success_wilson95_low"] = [
        lower.loc[(row.controller, row.scenario)] for row in summary.itertuples(index=False)
    ]
    return summary


def plot_predictive_summary(summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    scenario_order = [s["scenario"] for s in delay_scenarios()]
    scenario_labels = {
        "delay_60s": "60 s delay",
        "delay_120s": "120 s delay",
        "noise_10pct_delay_60s": "10% noise\n+ 60 s delay",
        "noise_10pct_delay_120s": "10% noise\n+ 120 s delay",
    }
    controller_order = [
        "delayed_full_feedback",
        "predictive_linear_full",
        "predictive_smoothed_full",
        "predictive_guarded_full",
        "predictive_temp_ros",
    ]
    controller_labels = {
        "delayed_full_feedback": "delayed\nfeedback",
        "predictive_linear_full": "linear\npredictor",
        "predictive_smoothed_full": "smoothed\npredictor",
        "predictive_guarded_full": "guarded\npredictor",
        "predictive_temp_ros": "temp + ROS\npredictor",
    }
    colors = {
        "delayed_full_feedback": "#6b6b6b",
        "predictive_linear_full": "#4c78a8",
        "predictive_smoothed_full": "#2f7f4f",
        "predictive_guarded_full": "#d69a31",
        "predictive_temp_ros": "#8a5fbf",
    }

    plt.rcParams.update(
        {
            "font.size": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    pivot = summary.pivot_table(index="controller", columns="scenario", values="chance_success_probability").loc[
        controller_order, scenario_order
    ]
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels([scenario_labels[s] for s in scenario_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(controller_order)))
    ax.set_yticklabels([controller_labels[c] for c in controller_order])
    ax.set_title("A. Delay-aware success")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, pad=0.01)

    ax = axes[0, 1]
    pivot_low = summary.pivot_table(index="controller", columns="scenario", values="chance_success_wilson95_low").loc[
        controller_order, scenario_order
    ]
    im = ax.imshow(pivot_low.to_numpy(), vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels([scenario_labels[s] for s in scenario_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(controller_order)))
    ax.set_yticklabels([])
    ax.set_title("B. Wilson lower bound")
    for i in range(pivot_low.shape[0]):
        for j in range(pivot_low.shape[1]):
            ax.text(j, i, f"{pivot_low.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, pad=0.01)

    ax = axes[0, 2]
    for controller in controller_order:
        rows = summary[summary["controller"] == controller].set_index("scenario").loc[scenario_order]
        ax.plot(
            np.arange(len(scenario_order)),
            rows["chance_success_probability"],
            marker="o",
            linewidth=1.6,
            color=colors[controller],
            label=controller_labels[controller].replace("\n", " "),
        )
    ax.axhline(0.70, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels([scenario_labels[s] for s in scenario_order], rotation=25, ha="right")
    ax.set_ylabel("Chance of success")
    ax.set_title("C. Delay/noise response")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    selected = summary[summary["scenario"] == "noise_10pct_delay_120s"].set_index("controller").loc[controller_order]
    y = np.arange(len(selected))
    ax.barh(y - 0.18, selected["overheat_failure_probability"], height=0.36, color="#b04a5a", label="overheat")
    ax.barh(y + 0.18, selected["thermal_failure_probability"], height=0.36, color="#d69a31", label="thermal")
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels([controller_labels[c].replace("\n", " ") for c in controller_order])
    ax.invert_yaxis()
    ax.set_xlabel("Failure probability")
    ax.set_title("D. Failure modes at longest delay")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.scatter(
        summary["median_energy_j_cm2"],
        summary["chance_success_probability"],
        c=summary["chance_success_wilson95_low"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=52,
        alpha=0.9,
        linewidth=0,
    )
    ax.set_xlabel("Median delivered energy (J cm$^{-2}$)")
    ax.set_ylabel("Chance of success")
    ax.set_title("E. Success-energy tradeoff")
    fig.colorbar(ax.collections[0], ax=ax, pad=0.01, label="Wilson low")

    ax = axes[1, 2]
    if not traces.empty:
        scenario = "delay_60s"
        trace_candidates = traces[(traces["scenario"] == scenario) & (traces["uncertainty_id"] == 24)]
        for controller in controller_order:
            trace = trace_candidates[trace_candidates["controller"] == controller]
            if trace.empty:
                continue
            ax.plot(trace["time_s"], trace["max_temperature_c"], color=colors[controller], linewidth=1.4, label=controller_labels[controller].replace("\n", " "))
        ax.axhspan(43.0, 50.0, color="#2f7f4f", alpha=0.08)
        ax.axhline(55.0, color="#b04a5a", linestyle="--", linewidth=0.9)
        ax.legend(frameon=False, fontsize=6)
        ax.set_ylim(40.0, 56.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Maximum temperature (C)")
    ax.set_title("F. Representative 60 s-delay trajectory")

    for suffix in ("svg", "png"):
        fig.savefig(FIG / f"figure_s10_predictive_delay_control.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    uncertainty = sample_uncertainty(n=N_DRAWS, seed=SEED)
    baseline_policy = full_feedback_policy()
    policies = predictive_policies()
    scenarios = delay_scenarios()
    outcomes = []
    traces = []

    for i, draw in uncertainty.iterrows():
        if i and i % 10 == 0:
            print(f"processed {i}/{N_DRAWS} uncertainty draws")
        for scenario in scenarios:
            baseline, baseline_traces = simulate_ablation_case(baseline_policy, scenario, draw, rng)
            baseline["controller"] = "delayed_full_feedback"
            outcomes.append(baseline)
            traces.extend(baseline_traces)
            for policy in policies:
                result, trace_rows = simulate_predictive_case(policy, scenario, draw, rng)
                outcomes.append(result)
                traces.extend(trace_rows)

    outcomes_df = pd.DataFrame(outcomes)
    traces_df = pd.DataFrame(traces)
    summary = summarize(outcomes_df)
    uncertainty.to_csv(OUT / "predictive_delay_uncertainty_draws.csv", index=False)
    outcomes_df.to_csv(OUT / "predictive_delay_outcomes.csv", index=False)
    traces_df.to_csv(OUT / "predictive_delay_traces.csv", index=False)
    summary.to_csv(OUT / "predictive_delay_summary.csv", index=False)
    metadata = {
        "n_uncertainty_draws": N_DRAWS,
        "seed": SEED,
        "scenarios": scenarios,
        "baseline_policy": baseline_policy,
        "predictive_policies": policies,
        "predictor": "finite-difference extrapolation of delayed measurements with delay-dependent thermal guard",
    }
    (OUT / "predictive_delay_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_predictive_summary(summary, traces_df)
    print("Wrote predictive delay-control outputs:")
    print(f"- {OUT / 'predictive_delay_summary.csv'}")
    print(f"- {FIG / 'figure_s10_predictive_delay_control.svg'}")
    print(summary.sort_values(["scenario", "chance_success_probability"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
