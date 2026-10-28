#!/usr/bin/env python3
"""Feedback-signal ablation and sensor noise/delay robustness analysis."""

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
    simulate_fixed_case,
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
OUT = ROOT / "data" / "feedback_ablation"
FIG = ROOT / "figures" / "feedback_ablation"

N_DRAWS = 60
SEED = 20260708


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def controller_catalog() -> list[dict]:
    return [
        {"controller": "temperature_only", "signals": ("temperature",), "max_intensity_w_cm2": 1.10},
        {"controller": "oxygen_only", "signals": ("oxygen",), "max_intensity_w_cm2": 1.10},
        {"controller": "ros_only", "signals": ("ros",), "max_intensity_w_cm2": 1.10},
        {"controller": "temperature_oxygen", "signals": ("temperature", "oxygen"), "max_intensity_w_cm2": 1.30},
        {"controller": "temperature_ros", "signals": ("temperature", "ros"), "max_intensity_w_cm2": 1.30},
        {"controller": "temperature_oxygen_ros", "signals": ("temperature", "oxygen", "ros"), "max_intensity_w_cm2": 1.30},
    ]


def sensing_scenarios() -> list[dict]:
    return [
        {"scenario": "ideal", "noise_fraction": 0.00, "delay_s": 0},
        {"scenario": "noise_5pct", "noise_fraction": 0.05, "delay_s": 0},
        {"scenario": "noise_10pct", "noise_fraction": 0.10, "delay_s": 0},
        {"scenario": "delay_30s", "noise_fraction": 0.00, "delay_s": 30},
        {"scenario": "delay_60s", "noise_fraction": 0.00, "delay_s": 60},
        {"scenario": "delay_120s", "noise_fraction": 0.00, "delay_s": 120},
        {"scenario": "noise_10pct_delay_60s", "noise_fraction": 0.10, "delay_s": 60},
    ]


def noisy_measurement(state: dict, scenario: dict, rng: np.random.Generator) -> dict:
    noise = float(scenario["noise_fraction"])
    measured = dict(state)
    if noise <= 0:
        return measured
    measured["max_temperature_c"] = state["max_temperature_c"] + rng.normal(0.0, noise * 5.0)
    measured["warm_fraction"] = float(np.clip(state["warm_fraction"] + rng.normal(0.0, noise * 0.20), 0.0, 1.0))
    measured["mean_oxygen_um"] = float(np.clip(state["mean_oxygen_um"] * (1.0 + rng.normal(0.0, noise)), 0.5, 40.0))
    measured["ros_threshold_fraction"] = float(
        np.clip(state["ros_threshold_fraction"] + rng.normal(0.0, noise * 0.20), 0.0, 1.0)
    )
    return measured


def choose_ablation_intensity(policy: dict, measured: dict) -> float:
    signals = set(policy["signals"])
    max_i = float(policy["max_intensity_w_cm2"])
    safety_stop_c = float(policy.get("safety_stop_c", 52.0))
    oxygen_floor_um = float(policy.get("oxygen_floor_um", 10.0))
    oxygen_low_um = float(policy.get("oxygen_low_um", 14.0))
    thermal_hold_c = float(policy.get("thermal_hold_c", 49.0))
    thermal_target_c = float(policy.get("thermal_target_c", 45.5))
    ros_completion = float(policy.get("ros_completion", 0.80))
    warm_target = float(policy.get("warm_target", 0.50))
    maintenance_i = float(policy.get("maintenance_intensity_w_cm2", 0.20))
    oxygen_limited_fraction = float(policy.get("oxygen_limited_fraction", 0.25))
    warmup_fraction = float(policy.get("warmup_fraction", 0.65))
    ros_progress_fraction = float(policy.get("ros_progress_fraction", 0.40))
    temp = float(measured["max_temperature_c"])
    warm = float(measured["warm_fraction"])
    oxygen = float(measured["mean_oxygen_um"])
    ros = float(measured["ros_threshold_fraction"])

    has_temp = "temperature" in signals
    has_oxygen = "oxygen" in signals
    has_ros = "ros" in signals

    if has_temp and temp >= safety_stop_c:
        return 0.0
    if has_oxygen and oxygen < oxygen_floor_um and (not has_ros or ros < ros_completion):
        return 0.0

    if has_temp and temp >= thermal_hold_c:
        if has_ros and ros >= ros_completion:
            return 0.0
        return maintenance_i

    if has_ros and ros >= ros_completion:
        if has_temp:
            return maintenance_i if temp < thermal_target_c else 0.0
        return 0.0

    if has_temp and temp < thermal_target_c:
        if has_oxygen and oxygen < oxygen_low_um:
            return oxygen_limited_fraction * max_i
        return max_i

    if has_temp and warm < warm_target:
        return warmup_fraction * max_i

    if has_oxygen and oxygen < oxygen_low_um:
        return maintenance_i

    if has_ros:
        return ros_progress_fraction * max_i

    return maintenance_i


def simulate_ablation_case(policy: dict, scenario: dict, draw: pd.Series, rng: np.random.Generator) -> tuple[dict, list[dict]]:
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

    delay_steps = int(round(float(scenario["delay_s"]) / CONTROL_DT_S))
    state_history: list[dict] = []
    trace_rows = []
    delivered_energy = 0.0
    light_on_time = 0.0
    max_command = 0.0
    n_steps = int(round(MAX_PROTOCOL_S / CONTROL_DT_S))
    ros_substeps = int(round(CONTROL_DT_S / cfg.ros_dt_s))

    for step in range(n_steps):
        ros_dose_mm = ros_dose_um / 1000.0
        true_state = {
            "max_temperature_c": float(temperature.max()),
            "warm_fraction": float(np.mean((temperature >= 43.0) & (temperature <= 50.0))),
            "mean_oxygen_um": float(np.mean(oxygen_um)),
            "ros_threshold_fraction": float(np.mean(ros_dose_mm >= ev.singlet_oxygen_threshold_mm)),
        }
        state_history.append(true_state)
        delayed_state = state_history[max(0, len(state_history) - 1 - delay_steps)]
        measured = noisy_measurement(delayed_state, scenario, rng)
        intensity = choose_ablation_intensity(policy, measured)
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
                    "time_s": (step + 1) * CONTROL_DT_S,
                    "intensity_w_cm2": intensity,
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
        "delay_s": float(scenario["delay_s"]),
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


def plot_summary(summary: pd.DataFrame) -> None:
    controller_order = [c["controller"] for c in controller_catalog()]
    scenario_order = [s["scenario"] for s in sensing_scenarios()]
    scenario_labels = {
        "ideal": "ideal",
        "noise_5pct": "5% noise",
        "noise_10pct": "10% noise",
        "delay_30s": "30 s delay",
        "delay_60s": "60 s delay",
        "delay_120s": "120 s delay",
        "noise_10pct_delay_60s": "10% noise\n+ 60 s delay",
    }
    controller_labels = {
        "temperature_only": "temperature",
        "oxygen_only": "oxygen",
        "ros_only": "ROS",
        "temperature_oxygen": "temperature\n+ oxygen",
        "temperature_ros": "temperature\n+ ROS",
        "temperature_oxygen_ros": "temperature\n+ oxygen\n+ ROS",
    }
    fixed = summary[summary["controller"] == "fixed_best_utility"]
    feedback = summary[summary["controller"].isin(controller_order)]

    plt.rcParams.update(
        {
            "font.size": 8.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), constrained_layout=True)

    ax = axes[0, 0]
    pivot = feedback.pivot_table(index="controller", columns="scenario", values="chance_success_probability").loc[
        controller_order, scenario_order
    ]
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels([scenario_labels[s] for s in scenario_order], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(controller_order)))
    ax.set_yticklabels([controller_labels[c] for c in controller_order])
    ax.set_title("A. Sensing-structure success")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, pad=0.01)

    ax = axes[0, 1]
    pivot_low = feedback.pivot_table(index="controller", columns="scenario", values="chance_success_wilson95_low").loc[
        controller_order, scenario_order
    ]
    im = ax.imshow(pivot_low.to_numpy(), vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels([scenario_labels[s] for s in scenario_order], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(controller_order)))
    ax.set_yticklabels([])
    ax.set_title("B. Wilson lower bound")
    for i in range(pivot_low.shape[0]):
        for j in range(pivot_low.shape[1]):
            ax.text(j, i, f"{pivot_low.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, pad=0.01)

    ax = axes[0, 2]
    ideal = feedback[feedback["scenario"] == "ideal"].set_index("controller").loc[controller_order]
    y = np.arange(len(ideal))
    ax.barh(y, ideal["chance_success_probability"], color="#2f7fb8", label="mean")
    ax.barh(y, ideal["chance_success_wilson95_low"], color="#164f7a", label="Wilson low")
    fixed_val = float(fixed[fixed["scenario"] == "ideal"]["chance_success_probability"].iloc[0])
    ax.axvline(fixed_val, color="#666666", linestyle="--", linewidth=1.0, label="fixed utility")
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels([controller_labels[c].replace("\n+ ", " + ") for c in controller_order])
    ax.invert_yaxis()
    ax.set_xlabel("Chance of success")
    ax.set_title("C. Ideal-sensing ablation")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    best = "temperature_oxygen_ros"
    best_rows = feedback[feedback["controller"] == best].set_index("scenario").loc[scenario_order]
    ax.plot(np.arange(len(scenario_order)), best_rows["chance_success_probability"], marker="o", color="#2f7f4f")
    ax.plot(np.arange(len(scenario_order)), best_rows["chance_success_wilson95_low"], marker="s", color="#164f7a")
    ax.axhline(0.70, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels([scenario_labels[s] for s in scenario_order], rotation=30, ha="right")
    ax.set_ylabel("Chance of success")
    ax.set_title("D. Delay sensitivity of full feedback")

    ax = axes[1, 1]
    selected = feedback[(feedback["scenario"] == "noise_10pct_delay_60s")].set_index("controller").loc[controller_order]
    y = np.arange(len(selected))
    ax.barh(y - 0.18, selected["overheat_failure_probability"], height=0.36, color="#b04a5a", label="overheat")
    ax.barh(y + 0.18, selected["ros_failure_probability"], height=0.36, color="#4d83b8", label="ROS")
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels([controller_labels[c].replace("\n+ ", " + ") for c in controller_order])
    ax.invert_yaxis()
    ax.set_xlabel("Failure probability")
    ax.set_title("E. Failure modes with noise + delay")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    ax.scatter(
        feedback["median_energy_j_cm2"],
        feedback["chance_success_probability"],
        c=feedback["chance_success_wilson95_low"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=45,
        alpha=0.88,
        linewidth=0,
    )
    ax.set_xlabel("Median delivered energy (J cm$^{-2}$)")
    ax.set_ylabel("Chance of success")
    ax.set_title("F. Success-energy tradeoff")
    fig.colorbar(ax.collections[0], ax=ax, pad=0.01, label="Wilson low")

    for suffix in ("svg", "png"):
        fig.savefig(FIG / f"figure_s9_feedback_ablation_noise_delay.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    uncertainty = sample_uncertainty(n=N_DRAWS, seed=SEED)
    policies = controller_catalog()
    scenarios = sensing_scenarios()
    fixed_policy = {"controller": "fixed_best_utility", "duty_cycle": 0.45, "intensity_w_cm2": 0.40}
    outcomes = []
    traces = []
    for i, draw in uncertainty.iterrows():
        if i and i % 10 == 0:
            print(f"processed {i}/{N_DRAWS} uncertainty draws")
        for scenario in scenarios:
            fixed = simulate_fixed_case(fixed_policy, draw)
            fixed["scenario"] = scenario["scenario"]
            fixed["noise_fraction"] = scenario["noise_fraction"]
            fixed["delay_s"] = scenario["delay_s"]
            outcomes.append(fixed)
            for policy in policies:
                result, trace_rows = simulate_ablation_case(policy, scenario, draw, rng)
                outcomes.append(result)
                traces.extend(trace_rows)

    outcomes_df = pd.DataFrame(outcomes)
    traces_df = pd.DataFrame(traces)
    summary = summarize(outcomes_df)
    uncertainty.to_csv(OUT / "feedback_ablation_uncertainty_draws.csv", index=False)
    outcomes_df.to_csv(OUT / "feedback_ablation_outcomes.csv", index=False)
    traces_df.to_csv(OUT / "feedback_ablation_traces.csv", index=False)
    summary.to_csv(OUT / "feedback_ablation_summary.csv", index=False)
    metadata = {
        "n_uncertainty_draws": N_DRAWS,
        "seed": SEED,
        "controllers": policies,
        "sensing_scenarios": scenarios,
        "fixed_comparator": fixed_policy,
    }
    (OUT / "feedback_ablation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_summary(summary)
    print("Wrote feedback ablation outputs:")
    print(f"- {OUT / 'feedback_ablation_summary.csv'}")
    print(f"- {FIG / 'figure_s9_feedback_ablation_noise_delay.svg'}")
    print(summary.sort_values(["scenario", "chance_success_probability"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
