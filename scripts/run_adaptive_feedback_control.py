#!/usr/bin/env python3
"""Adaptive feedback-control protocols for pulsed nanoparticle therapy.

Fixed duty-cycle schedules were not globally robust under broad biological
uncertainty. This script tests whether simple state-feedback controllers can
recover feasibility by adjusting light intensity from simulated temperature,
oxygen, and ROS state variables during treatment.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_first_model import (
    EvidenceDefaults,
    ModelConfig,
    build_pennes_backward_euler_tridiagonal,
    gans_peak_nm,
    nanoparticle_profile,
    optical_fluence_profile,
    relative_gans_absorption,
    solve_tridiagonal,
    simulate_case,
)
from run_robust_chance_optimization import sample_uncertainty, success_flags, wilson_lower_bound


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "adaptive_control"
FIG = ROOT / "figures" / "adaptive_control"

CONTROL_DT_S = 10.0
MAX_PROTOCOL_S = 3000.0
REFERENCE_INTENSITY_W_CM2 = 1.6
REFERENCE_PROTOCOL_S = 3000.0


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def controller_catalog() -> list[dict]:
    return [
        {
            "controller": "thermal_guard",
            "max_intensity_w_cm2": 1.10,
            "thermal_target_c": 45.0,
            "thermal_hold_c": 48.5,
            "safety_stop_c": 51.5,
            "oxygen_floor_um": 7.0,
            "ros_completion": 0.80,
            "maintenance_intensity_w_cm2": 0.18,
        },
        {
            "controller": "oxygen_thermal_feedback",
            "max_intensity_w_cm2": 1.30,
            "thermal_target_c": 45.5,
            "thermal_hold_c": 49.0,
            "safety_stop_c": 52.0,
            "oxygen_floor_um": 10.0,
            "ros_completion": 0.80,
            "maintenance_intensity_w_cm2": 0.20,
        },
        {
            "controller": "safety_first_feedback",
            "max_intensity_w_cm2": 0.90,
            "thermal_target_c": 44.5,
            "thermal_hold_c": 47.5,
            "safety_stop_c": 50.5,
            "oxygen_floor_um": 12.0,
            "ros_completion": 0.70,
            "maintenance_intensity_w_cm2": 0.12,
        },
    ]


def fixed_protocol_catalog() -> list[dict]:
    return [
        {"controller": "fixed_safety_first", "duty_cycle": 0.10, "intensity_w_cm2": 0.90},
        {"controller": "fixed_best_utility", "duty_cycle": 0.45, "intensity_w_cm2": 0.40},
        {"controller": "fixed_frontier", "duty_cycle": 0.125, "intensity_w_cm2": 1.50},
    ]


def choose_intensity(
    policy: dict,
    max_temp_c: float,
    warm_fraction: float,
    ros_fraction: float,
    mean_oxygen_um: float,
) -> float:
    if max_temp_c >= policy["safety_stop_c"]:
        return 0.0
    if mean_oxygen_um < policy["oxygen_floor_um"] and ros_fraction < policy["ros_completion"]:
        return 0.0
    if max_temp_c >= policy["thermal_hold_c"]:
        return policy["maintenance_intensity_w_cm2"] if ros_fraction < policy["ros_completion"] else 0.0
    if ros_fraction >= policy["ros_completion"] and warm_fraction >= 0.50:
        return policy["maintenance_intensity_w_cm2"] if max_temp_c < policy["thermal_target_c"] else 0.0
    if max_temp_c < policy["thermal_target_c"] and ros_fraction < 0.50:
        return policy["max_intensity_w_cm2"]
    if warm_fraction < 0.50 and max_temp_c < policy["thermal_target_c"]:
        return 0.65 * policy["max_intensity_w_cm2"]
    if ros_fraction < policy["ros_completion"]:
        return 0.40 * policy["max_intensity_w_cm2"]
    return policy["maintenance_intensity_w_cm2"]


def simulate_adaptive_case(policy: dict, draw: pd.Series) -> dict:
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
    peak_nm = gans_peak_nm(70.0, 18.0, ev.medium_refractive_index)
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
        r,
        dr,
        CONTROL_DT_S,
        alpha,
        perfusion_loss_s,
        cfg.external_cooling_s,
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

    rows = []
    delivered_energy = 0.0
    light_on_time = 0.0
    max_command = 0.0
    n_steps = int(round(MAX_PROTOCOL_S / CONTROL_DT_S))
    ros_substeps = int(round(CONTROL_DT_S / cfg.ros_dt_s))

    for step in range(n_steps):
        time_s = step * CONTROL_DT_S
        ros_dose_mm = ros_dose_um / 1000.0
        warm_fraction = float(np.mean((temperature >= 43.0) & (temperature <= 50.0)))
        ros_fraction = float(np.mean(ros_dose_mm >= ev.singlet_oxygen_threshold_mm))
        max_temp = float(temperature.max())
        mean_oxygen = float(np.mean(oxygen_um))
        intensity = choose_intensity(policy, max_temp, warm_fraction, ros_fraction, mean_oxygen)
        max_command = max(max_command, intensity)
        if intensity > 0.0:
            light_on_time += CONTROL_DT_S
            delivered_energy += intensity * CONTROL_DT_S

        source_k_s = intensity * q_opt_unit / (cfg.tissue_density_kg_m3 * cfg.tissue_cp_j_kg_k)
        rise = solve_tridiagonal(lower, diag, upper, rise + CONTROL_DT_S * source_k_s)
        temperature = cfg.body_temperature_c + rise

        for _ in range(ros_substeps):
            if intensity > 0.0:
                oxygen_factor = oxygen_um / (oxygen_um + ev.oxygen_beta_um)
                generation_um_s = ev.xi_cm2_mw_s * (intensity * fluence_unit_mw_cm2) * s0_um * oxygen_factor
                ros_dose_um += cfg.ros_dt_s * generation_um_s
                oxygen_um -= cfg.ros_dt_s * generation_um_s
                s0_um -= cfg.ros_dt_s * ev.sigma_per_um * generation_um_s * s0_um / (s0_um + ev.oxygen_delta_um)
                s0_um = np.maximum(s0_um, 0.0)
            oxygen_um += cfg.ros_dt_s * ev.oxygen_perfusion_um_s * (1.0 - oxygen_um / ev.oxygen_initial_um)
            oxygen_um = np.clip(oxygen_um, 0.5, ev.oxygen_initial_um)

        if step % 10 == 0 or step == n_steps - 1:
            rows.append(
                {
                    "time_s": time_s + CONTROL_DT_S,
                    "controller": policy["controller"],
                    "uncertainty_id": int(draw["uncertainty_id"]),
                    "intensity_w_cm2": intensity,
                    "max_temperature_c": float(temperature.max()),
                    "mean_oxygen_um": float(np.mean(oxygen_um)),
                    "warm_fraction": float(np.mean((temperature >= 43.0) & (temperature <= 50.0))),
                    "ros_threshold_fraction": float(np.mean((ros_dose_um / 1000.0) >= ev.singlet_oxygen_threshold_mm)),
                    "delivered_energy_j_cm2": delivered_energy,
                }
            )

    ros_dose_mm = ros_dose_um / 1000.0
    result = {
        "controller": policy["controller"],
        "uncertainty_id": int(draw["uncertainty_id"]),
        "perfusion_s": float(draw["perfusion_s"]),
        "radius_um": float(draw["radius_um"]),
        "photosensitizer_um": float(draw["photosensitizer_um"]),
        "volume_fraction": float(draw["volume_fraction"]),
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
    return result, rows


def simulate_fixed_case(policy: dict, draw: pd.Series) -> dict:
    ev = EvidenceDefaults()
    cfg = replace(
        ModelConfig(),
        spheroid_radius_m=float(draw["radius_um"]) * 1e-6,
        photosensitizer_um_at_reference_phi=float(draw["photosensitizer_um"]),
    )
    result = simulate_case(
        wavelength_nm=808.0,
        length_nm=70.0,
        width_nm=18.0,
        volume_fraction=float(draw["volume_fraction"]),
        sar_mw_per_kg=0.0,
        intensity_w_cm2=float(policy["intensity_w_cm2"]),
        perfusion_s=float(draw["perfusion_s"]),
        duty_cycle=float(policy["duty_cycle"]),
        cfg=cfg,
        ev=ev,
    )
    row = {
        "controller": policy["controller"],
        "uncertainty_id": int(draw["uncertainty_id"]),
        "perfusion_s": float(draw["perfusion_s"]),
        "radius_um": float(draw["radius_um"]),
        "photosensitizer_um": float(draw["photosensitizer_um"]),
        "volume_fraction": float(draw["volume_fraction"]),
        "protocol_duration_s": float(result["protocol_duration_s"]),
        "light_on_time_s": float(result["protocol_duration_s"] * policy["duty_cycle"]),
        "delivered_energy_j_cm2": float(result["protocol_duration_s"] * policy["duty_cycle"] * policy["intensity_w_cm2"]),
        "mean_intensity_w_cm2": float(policy["duty_cycle"] * policy["intensity_w_cm2"]),
        "max_commanded_intensity_w_cm2": float(policy["intensity_w_cm2"]),
        "max_temperature_c": float(result["max_temperature_c"]),
        "warm_fraction": float(result["warm_fraction"]),
        "overheat_fraction": float(result["overheat_fraction"]),
        "ros_threshold_fraction": float(result["ros_threshold_fraction"]),
        "hypoxic_failure_fraction": float(result["hypoxic_failure_fraction"]),
        "thermal_window_score": float(result["thermal_window_score"]),
        "ros_score": float(result["ros_score"]),
        "dual_modality_score": float(result["dual_modality_score"]),
        "protocol_burden": float(result["protocol_duration_s"] / REFERENCE_PROTOCOL_S + policy["intensity_w_cm2"] / REFERENCE_INTENSITY_W_CM2),
    }
    row.update(success_flags(row))
    return row


def summarize(outcomes: pd.DataFrame) -> pd.DataFrame:
    grouped = outcomes.groupby("controller", as_index=False)
    summary = grouped.agg(
        n=("chance_success", "size"),
        chance_success_probability=("chance_success", "mean"),
        strict_safe_dual_probability=("strict_safe_dual", "mean"),
        mean_positive_dual_score=("dual_modality_score", lambda x: float(np.maximum(x, 0.0).mean())),
        median_dual_score=("dual_modality_score", "median"),
        median_energy_j_cm2=("delivered_energy_j_cm2", "median"),
        median_light_on_time_s=("light_on_time_s", "median"),
        median_tmax_c=("max_temperature_c", "median"),
        q90_tmax_c=("max_temperature_c", lambda x: float(np.quantile(x, 0.90))),
        overheat_failure_probability=("overheat_failure", "mean"),
        ros_failure_probability=("ros_failure", "mean"),
        thermal_failure_probability=("thermal_failure", "mean"),
        median_protocol_burden=("protocol_burden", "median"),
    )
    lower = outcomes.groupby("controller")["chance_success"].apply(wilson_lower_bound)
    summary["chance_success_wilson95_low"] = summary["controller"].map(lower)
    summary["adaptive_utility"] = (
        summary["chance_success_wilson95_low"]
        * summary["mean_positive_dual_score"]
        / ((summary["median_protocol_burden"] + 1e-12) * (1.0 + summary["overheat_failure_probability"]))
    )
    return summary.sort_values("adaptive_utility", ascending=False)


def plot_adaptive_figure(outcomes: pd.DataFrame, summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    ordered = summary.iloc[::-1]
    labels = ordered["controller"].str.replace("_", "\n", regex=False)
    y = np.arange(len(ordered))
    ax.barh(y, ordered["chance_success_probability"], color="#2878b5", label="mean")
    ax.barh(y, ordered["chance_success_wilson95_low"], color="#164f7a", label="95% low")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Chance of success")
    ax.set_title("A. Feedback versus fixed protocols")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    ax.scatter(
        outcomes["delivered_energy_j_cm2"],
        outcomes["dual_modality_score"],
        c=outcomes["chance_success"].astype(int),
        cmap="viridis",
        s=34,
        alpha=0.78,
        linewidth=0,
    )
    ax.set_xlabel("Delivered energy (J cm$^{-2}$)")
    ax.set_ylabel("Dual-window score")
    ax.set_title("B. Outcome distribution")

    ax = axes[0, 2]
    width = 0.36
    x = np.arange(len(summary))
    ax.bar(x - width / 2, summary["overheat_failure_probability"], width=width, color="#b04a5a", label="overheat")
    ax.bar(x + width / 2, summary["ros_failure_probability"], width=width, color="#4d83b8", label="ROS")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["controller"].str.replace("_", "\n", regex=False), rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Failure probability")
    ax.set_title("C. Failure modes")
    ax.legend(frameon=False)

    best_controller = str(summary.iloc[0]["controller"])
    best_trace_id = int(
        outcomes[(outcomes["controller"] == best_controller) & (outcomes["chance_success"])]
        ["uncertainty_id"]
        .head(1)
        .fillna(outcomes[outcomes["controller"] == best_controller]["uncertainty_id"].iloc[0])
        .iloc[0]
    )
    trace = traces[(traces["controller"] == best_controller) & (traces["uncertainty_id"] == best_trace_id)]

    ax = axes[1, 0]
    ax.plot(trace["time_s"], trace["max_temperature_c"], color="#b04a5a", label="Tmax")
    ax.axhspan(43, 50, color="#b04a5a", alpha=0.12)
    ax.axhline(55, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("D. Adaptive thermal trajectory")

    ax = axes[1, 1]
    ax.plot(trace["time_s"], trace["ros_threshold_fraction"], color="#2f7f4f", label="ROS coverage")
    ax.plot(trace["time_s"], trace["mean_oxygen_um"] / 40.0, color="#4d83b8", label="mean O2 / 40")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction or normalized oxygen")
    ax.set_title("E. Oxygen and ROS feedback state")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    ax.step(trace["time_s"], trace["intensity_w_cm2"], where="post", color="#5a4b9b")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Commanded intensity (W cm$^{-2}$)")
    ax.set_title("F. Adaptive control action")

    for suffix in ("svg", "png"):
        fig.savefig(FIG / f"figure_s7_adaptive_feedback_control.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    uncertainty = sample_uncertainty()
    outcomes: list[dict] = []
    traces: list[dict] = []

    for _, draw in uncertainty.iterrows():
        for policy in controller_catalog():
            result, trace_rows = simulate_adaptive_case(policy, draw)
            outcomes.append(result)
            traces.extend(trace_rows)
        for policy in fixed_protocol_catalog():
            outcomes.append(simulate_fixed_case(policy, draw))

    outcomes_df = pd.DataFrame(outcomes)
    traces_df = pd.DataFrame(traces)
    summary = summarize(outcomes_df)

    outcomes_df.to_csv(OUT / "adaptive_control_outcomes.csv", index=False)
    traces_df.to_csv(OUT / "adaptive_control_traces.csv", index=False)
    summary.to_csv(OUT / "adaptive_control_summary.csv", index=False)
    metadata = {
        "n_uncertainty_draws": int(len(uncertainty)),
        "n_adaptive_controllers": int(len(controller_catalog())),
        "n_fixed_comparators": int(len(fixed_protocol_catalog())),
        "max_protocol_s": MAX_PROTOCOL_S,
        "control_dt_s": CONTROL_DT_S,
        "success_definition": {
            "warm_fraction": ">= 0.50",
            "ros_threshold_fraction": ">= 0.50",
            "hypoxic_failure_fraction": "<= 0.35",
            "overheat_fraction": "<= 0.05",
            "max_temperature_c": "<= 55.0",
        },
        "best_controller": summary.iloc[0].to_dict(),
        "controller_catalog": controller_catalog(),
        "fixed_protocol_catalog": fixed_protocol_catalog(),
    }
    (OUT / "adaptive_control_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_adaptive_figure(outcomes_df, summary, traces_df)

    best = summary.iloc[0]
    print("Wrote adaptive feedback-control outputs:")
    print(f"- {OUT / 'adaptive_control_summary.csv'}")
    print(f"- {OUT / 'adaptive_control_outcomes.csv'}")
    print(f"- {FIG / 'figure_s7_adaptive_feedback_control.svg'}")
    print(
        "Best controller: "
        f"{best['controller']}, "
        f"Psuccess={best['chance_success_probability']:.3f}, "
        f"Wilson95low={best['chance_success_wilson95_low']:.3f}, "
        f"overheat={best['overheat_failure_probability']:.3f}, "
        f"ROS fail={best['ros_failure_probability']:.3f}"
    )


if __name__ == "__main__":
    main()
