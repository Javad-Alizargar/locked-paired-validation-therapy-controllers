from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_adaptive_feedback_control import simulate_fixed_case  # noqa: E402
from run_feedback_ablation_noise_delay import simulate_ablation_case, summarize  # noqa: E402
from run_predictive_delay_control import predictive_policies, simulate_predictive_case  # noqa: E402
from run_robust_chance_optimization import sample_uncertainty  # noqa: E402


NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data" / "information_atlas"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

SEED = 20260710
N_DRAWS = 8
EPS = 1e-12
REFERENCE_RADIUS_UM = 500.0
REFERENCE_PERFUSION_S = 0.008
REFERENCE_PHOTOSENSITIZER_UM = 1.0
REFERENCE_VOLUME_FRACTION = 0.003
TISSUE_ALPHA_M2_S = 1.45e-7


def atlas_scenarios() -> list[dict]:
    rows = []
    for delay_s in [0, 30, 60, 90, 120, 180]:
        for noise_fraction in [0.0, 0.10]:
            rows.append(
                {
                    "scenario": f"delay_{delay_s}s_noise_{int(noise_fraction * 100)}pct",
                    "delay_s": float(delay_s),
                    "noise_fraction": float(noise_fraction),
                }
            )
    return rows


def atlas_policies() -> tuple[list[dict], list[dict], dict]:
    feedback = [
        {
            "controller": "full_feedback",
            "signals": ("temperature", "oxygen", "ros"),
            "max_intensity_w_cm2": 1.30,
        },
        {
            "controller": "temperature_ros_feedback",
            "signals": ("temperature", "ros"),
            "max_intensity_w_cm2": 1.30,
        },
    ]
    predictive = [
        {**p, "controller": "predictive_smoothed_full"}
        for p in predictive_policies()
        if p["controller"] == "predictive_smoothed_full"
    ]
    fixed = {"controller": "fixed_reference", "duty_cycle": 0.125, "intensity_w_cm2": 1.10}
    return feedback, predictive, fixed


def geometry_normalized_policy(draw: pd.Series, scenario: dict) -> dict:
    """Controller scaled by geometry, diffusion, perfusion recovery, and latency."""
    radius_um = float(draw["radius_um"])
    perfusion_s = float(draw["perfusion_s"])
    radius_ratio = radius_um / REFERENCE_RADIUS_UM
    recovery_ratio = REFERENCE_PERFUSION_S / (perfusion_s + EPS)
    radius_m = radius_um * 1e-6
    reference_tau_s = (REFERENCE_RADIUS_UM * 1e-6) ** 2 / TISSUE_ALPHA_M2_S
    thermal_tau_s = radius_m**2 / TISSUE_ALPHA_M2_S
    diffusion_ratio = thermal_tau_s / (reference_tau_s + EPS)
    delay_diffusion_ratio = float(scenario["delay_s"]) / (thermal_tau_s + EPS)

    geometry_load = np.clip(radius_ratio * np.sqrt(recovery_ratio), 0.75, 1.65)
    delay_load = np.clip(np.log10(1.0 + delay_diffusion_ratio) / 2.0, 0.0, 0.85)
    control_load = np.clip(0.45 * (geometry_load - 1.0) + 0.30 * (diffusion_ratio - 1.0) + 0.25 * delay_load, -0.25, 0.65)
    positive_load = max(control_load, 0.0)

    max_intensity = 1.30 / (1.0 + 0.12 * positive_load)
    max_intensity = float(np.clip(max_intensity, 1.12, 1.30))
    thermal_shift = float(np.clip(1.4 * positive_load, 0.0, 1.8))
    oxygen_shift = float(np.clip(2.4 * positive_load, 0.0, 3.5))
    ros_completion = float(np.clip(0.80 - 0.035 * positive_load, 0.74, 0.80))

    return {
        "controller": "geometry_normalized_full_feedback",
        "signals": ("temperature", "oxygen", "ros"),
        "max_intensity_w_cm2": max_intensity,
        "safety_stop_c": 52.0 - 0.6 * thermal_shift,
        "thermal_hold_c": 49.0 - thermal_shift,
        "thermal_target_c": 45.5 - 0.5 * thermal_shift,
        "oxygen_floor_um": 10.0 + oxygen_shift,
        "oxygen_low_um": 14.0 + oxygen_shift,
        "ros_completion": ros_completion,
        "warm_target": float(np.clip(0.50 + 0.07 * positive_load, 0.50, 0.61)),
        "maintenance_intensity_w_cm2": float(np.clip(0.20 * max_intensity / 1.30, 0.12, 0.20)),
        "oxygen_limited_fraction": float(np.clip(0.25 / (1.0 + 0.55 * positive_load), 0.15, 0.25)),
        "warmup_fraction": float(np.clip(0.65 / (1.0 + 0.16 * positive_load), 0.52, 0.65)),
        "ros_progress_fraction": float(np.clip(0.40 / (1.0 + 0.10 * positive_load), 0.34, 0.40)),
        "geometry_load": float(geometry_load),
        "diffusion_ratio": float(diffusion_ratio),
        "delay_diffusion_ratio_policy": float(delay_diffusion_ratio),
        "control_load": float(control_load),
    }


def geometry_dose_capacity_policy(draw: pd.Series, scenario: dict) -> dict:
    """Controller scaled by geometry and photochemical dose capacity."""
    radius_um = float(draw["radius_um"])
    perfusion_s = float(draw["perfusion_s"])
    ps_um = float(draw["photosensitizer_um"])
    volume_fraction = float(draw["volume_fraction"])

    radius_ratio = radius_um / REFERENCE_RADIUS_UM
    recovery_ratio = REFERENCE_PERFUSION_S / (perfusion_s + EPS)
    radius_m = radius_um * 1e-6
    reference_tau_s = (REFERENCE_RADIUS_UM * 1e-6) ** 2 / TISSUE_ALPHA_M2_S
    thermal_tau_s = radius_m**2 / TISSUE_ALPHA_M2_S
    diffusion_ratio = thermal_tau_s / (reference_tau_s + EPS)
    delay_diffusion_ratio = float(scenario["delay_s"]) / (thermal_tau_s + EPS)
    delay_load = np.clip(np.log10(1.0 + delay_diffusion_ratio) / 2.0, 0.0, 0.85)

    geometry_load = np.clip(radius_ratio * np.sqrt(recovery_ratio), 0.75, 1.65)
    geometry_control_load = np.clip(
        0.42 * (geometry_load - 1.0) + 0.28 * (diffusion_ratio - 1.0) + 0.22 * delay_load,
        -0.25,
        0.65,
    )
    positive_geometry_load = max(float(geometry_control_load), 0.0)

    dose_capacity = (
        (ps_um / REFERENCE_PHOTOSENSITIZER_UM)
        * (volume_fraction / REFERENCE_VOLUME_FRACTION)
        * np.sqrt(perfusion_s / REFERENCE_PERFUSION_S)
        / (radius_ratio**1.15 + EPS)
    )
    dose_capacity = float(np.clip(dose_capacity, 0.04, 4.0))
    dose_deficit = float(np.clip(np.log10(1.0 / dose_capacity), 0.0, 1.2))

    max_intensity = 1.30 / (1.0 + 0.08 * positive_geometry_load + 0.08 * dose_deficit)
    max_intensity = float(np.clip(max_intensity, 1.08, 1.30))
    thermal_shift = float(np.clip(1.1 * positive_geometry_load + 0.35 * dose_deficit, 0.0, 1.6))
    oxygen_shift = float(np.clip(1.8 * positive_geometry_load + 0.45 * dose_deficit, 0.0, 3.2))
    ros_completion = float(np.clip(0.80 - 0.020 * positive_geometry_load - 0.12 * dose_deficit, 0.60, 0.80))

    return {
        "controller": "geometry_dose_capacity_full_feedback",
        "signals": ("temperature", "oxygen", "ros"),
        "max_intensity_w_cm2": max_intensity,
        "safety_stop_c": 52.0 - 0.45 * thermal_shift,
        "thermal_hold_c": 49.0 - thermal_shift,
        "thermal_target_c": 45.5 - 0.10 * positive_geometry_load,
        "oxygen_floor_um": 10.0 + oxygen_shift,
        "oxygen_low_um": 14.0 + oxygen_shift,
        "ros_completion": ros_completion,
        "warm_target": float(np.clip(0.50 + 0.04 * positive_geometry_load, 0.50, 0.58)),
        "maintenance_intensity_w_cm2": float(np.clip(0.20 * max_intensity / 1.30, 0.16, 0.20)),
        "oxygen_limited_fraction": float(np.clip(0.25 / (1.0 + 0.30 * positive_geometry_load), 0.18, 0.25)),
        "warmup_fraction": float(np.clip(0.65 / (1.0 + 0.06 * positive_geometry_load), 0.58, 0.65)),
        "ros_progress_fraction": float(np.clip(0.40 / (1.0 + 0.05 * positive_geometry_load), 0.36, 0.40)),
        "geometry_load": float(geometry_load),
        "diffusion_ratio": float(diffusion_ratio),
        "delay_diffusion_ratio_policy": float(delay_diffusion_ratio),
        "control_load": float(geometry_control_load),
        "dose_capacity_number": dose_capacity,
        "dose_deficit": dose_deficit,
    }


def make_model_family_draws(base: pd.DataFrame) -> pd.DataFrame:
    families = []
    transforms = {
        "baseline": lambda d: d,
        "slow_recovery": lambda d: d.assign(perfusion_s=np.clip(d["perfusion_s"] * 0.55, 0.0015, 0.030)),
        "large_geometry": lambda d: d.assign(radius_um=np.clip(d["radius_um"] * 1.22, 300, 900)),
        "low_photosensitizer": lambda d: d.assign(photosensitizer_um=np.clip(d["photosensitizer_um"] * 0.45, 0.02, 2.0)),
    }
    for family, transform in transforms.items():
        d = transform(base.copy())
        d["model_family"] = family
        families.append(d)
    draws = pd.concat(families, ignore_index=True)
    draws["atlas_uncertainty_id"] = np.arange(len(draws))
    return draws


def add_draw_metadata(result: dict, draw: pd.Series, scenario: dict, controller_class: str) -> dict:
    result = dict(result)
    result["model_family"] = draw["model_family"]
    result["atlas_uncertainty_id"] = int(draw["atlas_uncertainty_id"])
    result["perfusion_s"] = float(draw["perfusion_s"])
    result["radius_um"] = float(draw["radius_um"])
    result["photosensitizer_um"] = float(draw["photosensitizer_um"])
    result["volume_fraction"] = float(draw["volume_fraction"])
    result["scenario"] = scenario["scenario"]
    result["delay_s"] = float(scenario["delay_s"])
    result["noise_fraction"] = float(scenario["noise_fraction"])
    result["controller_class"] = controller_class
    return result


def add_trace_metadata(rows: list[dict], draw: pd.Series, controller_class: str) -> list[dict]:
    out = []
    for row in rows:
        r = dict(row)
        r["model_family"] = draw["model_family"]
        r["atlas_uncertainty_id"] = int(draw["atlas_uncertainty_id"])
        r["controller_class"] = controller_class
        out.append(r)
    return out


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(vals) + 1)
    uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0) + EPS
    x_train = np.nan_to_num((x_train - mu) / sd)
    x_test = np.nan_to_num((x_test - mu) / sd)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]
    y_train = y_train.astype(float)

    def loss(w: np.ndarray) -> float:
        z = x_train @ w
        n_pos = max(y_train.sum(), 1.0)
        n_neg = max((1 - y_train).sum(), 1.0)
        weights = np.where(y_train > 0, len(y_train) / (2 * n_pos), len(y_train) / (2 * n_neg))
        return float(np.sum(weights * (np.logaddexp(0, z) - y_train * z)) + 0.05 * np.sum(w[1:] ** 2))

    result = minimize(loss, np.zeros(x_train.shape[1]), method="BFGS")
    z = np.clip(x_test @ result.x, -40, 40)
    return 1 / (1 + np.exp(-z))


def add_reduced_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["average_intensity_w_cm2"] = out["delivered_energy_j_cm2"] / (out["protocol_duration_s"] + EPS)
    out["duty_proxy"] = out["light_on_time_s"] / (out["protocol_duration_s"] + EPS)
    out["recovery_time_s"] = 1 / (out["perfusion_s"] + EPS)
    out["radius_m"] = out["radius_um"] * 1e-6
    out["thermal_diffusion_time_s"] = out["radius_m"] ** 2 / TISSUE_ALPHA_M2_S
    out["perfusion_diffusion_ratio"] = out["recovery_time_s"] / (out["thermal_diffusion_time_s"] + EPS)
    out["delay_diffusion_ratio"] = out["delay_s"] / (out["thermal_diffusion_time_s"] + EPS)
    out["radius_recovery_number"] = out["radius_um"] ** 2 * out["perfusion_s"]
    out["dose_capacity_number"] = (
        (out["photosensitizer_um"] / REFERENCE_PHOTOSENSITIZER_UM)
        * (out["volume_fraction"] / REFERENCE_VOLUME_FRACTION)
        * np.sqrt(out["perfusion_s"] / REFERENCE_PERFUSION_S)
        / ((out["radius_um"] / REFERENCE_RADIUS_UM) ** 1.15 + EPS)
    )
    out["dose_capacity_number"] = out["dose_capacity_number"].clip(0.04, 4.0)
    out["dose_deficit"] = np.log10(1.0 / out["dose_capacity_number"]).clip(lower=0.0, upper=1.2)
    out["drive_number"] = (out["average_intensity_w_cm2"] * out["volume_fraction"] * out["radius_um"] ** 2) / (
        out["perfusion_s"] + EPS
    )
    out["reaction_number"] = (out["average_intensity_w_cm2"] * out["photosensitizer_um"]) / (
        out["perfusion_s"] + EPS
    )
    out["reaction_drive_balance"] = out["reaction_number"] / (out["drive_number"] + EPS)
    out["delay_recovery_ratio"] = out["delay_s"] / (out["recovery_time_s"] + EPS)
    if "trace_information_quality" not in out:
        out["trace_information_quality"] = np.nan
    if "trace_reconstruction_error" not in out:
        out["trace_reconstruction_error"] = np.nan
    out["nominal_observation_quality"] = np.exp(-out["delay_s"] / 120.0) / (1 + 6.0 * out["noise_fraction"] ** 2)
    out["observation_quality"] = out["trace_information_quality"].fillna(out["nominal_observation_quality"])
    out.loc[out["controller_class"] == "fixed", "observation_quality"] = 0.0
    for col in ["drive_number", "reaction_number", "reaction_drive_balance", "recovery_time_s"]:
        out[f"log_{col}"] = np.log10(out[col] + EPS)
    for col in ["thermal_diffusion_time_s", "perfusion_diffusion_ratio", "radius_recovery_number", "dose_capacity_number"]:
        out[f"log_{col}"] = np.log10(out[col] + EPS)
    return out


def trace_information_metrics(traces: pd.DataFrame, scenarios: list[dict]) -> pd.DataFrame:
    if traces.empty:
        return pd.DataFrame()
    scenario_map = {row["scenario"]: row for row in scenarios}
    state_cols = ["max_temperature_c", "mean_oxygen_um", "ros_threshold_fraction"]
    rows = []
    group_cols = ["model_family", "atlas_uncertainty_id", "controller", "scenario", "controller_class"]
    for key, g in traces.groupby(group_cols, dropna=False):
        g = g.sort_values("time_s").reset_index(drop=True)
        scenario = scenario_map.get(key[3], {"delay_s": 0.0, "noise_fraction": 0.0})
        delay_s = float(scenario["delay_s"])
        noise = float(scenario["noise_fraction"])
        if len(g) < 4:
            continue
        dt = float(np.nanmedian(np.diff(g["time_s"])))
        lag = int(round(delay_s / max(dt, EPS)))
        true = g[state_cols].to_numpy(float)
        if lag <= 0:
            observed = true.copy()
        else:
            observed = np.vstack([np.repeat(true[[0]], min(lag, len(true)), axis=0), true[:-lag]])[: len(true)]
        if noise > 0:
            scales = np.array([5.0, np.nanstd(true[:, 1]) + EPS, 0.20])
            observed = observed + noise * scales * 0.5
        scale = np.nanstd(true, axis=0) + EPS
        err = np.sqrt(np.nanmean(((true - observed) / scale) ** 2))
        quality = float(np.exp(-0.5 * err**2))
        corr_vals = []
        for j in range(true.shape[1]):
            if np.nanstd(true[:, j]) > EPS and np.nanstd(observed[:, j]) > EPS:
                corr_vals.append(float(np.corrcoef(true[:, j], observed[:, j])[0, 1]))
        corr = float(np.nanmean(corr_vals)) if corr_vals else 1.0
        rows.append(
            {
                "model_family": key[0],
                "atlas_uncertainty_id": key[1],
                "controller": key[2],
                "scenario": key[3],
                "controller_class": key[4],
                "trace_reconstruction_error": float(err),
                "trace_information_quality": quality,
                "trace_state_correlation": corr,
                "trace_dt_s": dt,
            }
        )
    return pd.DataFrame(rows)


def validate_holdout(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "log_drive_number",
        "log_reaction_number",
        "log_reaction_drive_balance",
        "log_recovery_time_s",
        "log_thermal_diffusion_time_s",
        "log_perfusion_diffusion_ratio",
        "log_radius_recovery_number",
        "log_dose_capacity_number",
        "dose_deficit",
        "duty_proxy",
        "delay_recovery_ratio",
        "delay_diffusion_ratio",
        "noise_fraction",
        "observation_quality",
        "trace_reconstruction_error",
    ]
    df = df.copy()
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows = []
    x_all = df[features].to_numpy(float)
    y_all = df["chance_success"].astype(int).to_numpy()
    pred = logistic_predict(x_all, y_all, x_all)
    rows.append(
        {
            "heldout_model_family": "apparent_all",
            "n_train": len(df),
            "n_test": len(df),
            "success_rate_test": float(y_all.mean()),
            "auc": auc_score(y_all, pred),
        }
    )
    for family in sorted(df["model_family"].unique()):
        train = df["model_family"] != family
        test = df["model_family"] == family
        pred = logistic_predict(df.loc[train, features].to_numpy(float), df.loc[train, "chance_success"].astype(int).to_numpy(), df.loc[test, features].to_numpy(float))
        y = df.loc[test, "chance_success"].astype(int).to_numpy()
        rows.append(
            {
                "heldout_model_family": family,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "success_rate_test": float(y.mean()),
                "auc": auc_score(y, pred),
            }
        )
    return pd.DataFrame(rows)


def make_summary_figure(outcomes: pd.DataFrame, validation: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    dyn = outcomes[outcomes["controller_class"] != "fixed"]
    delay_summary = (
        dyn.groupby(["controller", "delay_s", "noise_fraction"], as_index=False)
        .agg(success=("chance_success", "mean"))
        .sort_values("delay_s")
    )
    for (controller, noise), g in delay_summary.groupby(["controller", "noise_fraction"]):
        style = "-" if noise == 0 else "--"
        ax.plot(g["delay_s"], g["success"], marker="o", linestyle=style, label=f"{controller}, noise={noise:.2f}")
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Success probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("A. Matched delay/noise grid")
    ax.legend(frameon=False, fontsize=6, ncol=1)

    ax = axes[0, 1]
    h = validation[validation["heldout_model_family"] != "apparent_all"]
    ax.barh(h["heldout_model_family"], h["auc"], color="#6c5b7b")
    ax.axvline(0.80, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.85, color="#c44e52", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Held-out AUC")
    ax.set_title("B. Held-out model-family transfer")

    ax = axes[1, 0]
    family = outcomes.groupby(["model_family", "controller_class"], as_index=False).agg(success=("chance_success", "mean"))
    pivot = family.pivot(index="model_family", columns="controller_class", values="success").fillna(0)
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("C. Model-family success structure")
    fig.colorbar(im, ax=ax, label="Success probability")

    ax = axes[1, 1]
    info = outcomes[
        (outcomes["controller_class"] != "fixed")
        & outcomes["trace_information_quality"].notna()
    ].copy()
    info["delay_plot_s"] = info["delay_s"] + 8.0 * info["noise_fraction"]
    ok = info["chance_success"] == 1
    ax.scatter(info.loc[~ok, "delay_plot_s"], info.loc[~ok, "trace_information_quality"], s=10, alpha=0.18, color="black", label="failure")
    ax.scatter(info.loc[ok, "delay_plot_s"], info.loc[ok, "trace_information_quality"], s=12, alpha=0.35, color="#2ca25f", label="success")
    med = info.groupby("delay_s", as_index=False).agg(trace_information_quality=("trace_information_quality", "median"))
    ax.plot(med["delay_s"], med["trace_information_quality"], color="#2166ac", linewidth=1.6, marker="o", markersize=3, label="median")
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Trace information quality")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("D. Trace-derived information loss")
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_verdict(outcomes: pd.DataFrame, validation: pd.DataFrame) -> None:
    holdout = validation[validation["heldout_model_family"] != "apparent_all"]
    median_auc = float(holdout["auc"].median())
    min_auc = float(holdout["auc"].min())
    apparent_auc = float(validation.loc[validation["heldout_model_family"] == "apparent_all", "auc"].iloc[0])
    pass_pilot = median_auc >= 0.80 and min_auc >= 0.70
    lines = [
        "# Step 2 Information Atlas Pilot Result",
        "",
        f"Atlas rows: {len(outcomes)}.",
        f"Overall success probability: {outcomes['chance_success'].mean():.3f}.",
        f"Apparent AUC: {apparent_auc:.3f}.",
        f"Median held-out model-family AUC: {median_auc:.3f}.",
        f"Minimum held-out model-family AUC: {min_auc:.3f}.",
        "",
        "## Theory revision implemented",
        "",
        "The revised pilot adds dimensionless geometry/recovery variables: thermal diffusion time, perfusion-diffusion ratio, delay-diffusion ratio, and a radius-recovery number.",
        "It also replaces the earlier delay/noise-only information proxy with a trace-derived metric that compares true simulated state trajectories with delayed noisy observations of temperature, oxygen, and ROS-threshold occupancy.",
        "",
        "## Verdict",
        "",
        "PASS pilot atlas design; scale to a larger simulation atlas." if pass_pilot else "BORDERLINE/FAIL; atlas design needs revision before scaling.",
        "",
        "## Reviewer interpretation",
        "",
        "This pilot tests matched delay/noise scenarios across explicit model-family perturbations. It is a design test, not a final Nature Computational Science result.",
        "A final manuscript would require a larger atlas, stricter family holdouts, symbolic/sparse boundary discovery, and calibration analysis.",
    ]
    (OUT_NOTES / "step2_information_atlas_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_geometry_controller_result(outcomes: pd.DataFrame) -> None:
    if "geometry_normalized_full_feedback" not in set(outcomes["controller"]):
        return
    comparison = outcomes.pivot_table(index="model_family", columns="controller", values="chance_success", aggfunc="mean")
    preferred = [
        "full_feedback",
        "geometry_normalized_full_feedback",
        "geometry_dose_capacity_full_feedback",
        "temperature_ros_feedback",
        "predictive_smoothed_full",
        "fixed_reference",
    ]
    comparison = comparison[[col for col in preferred if col in comparison.columns]].copy()
    comparison["geometry_minus_full_feedback"] = (
        comparison["geometry_normalized_full_feedback"] - comparison["full_feedback"]
    )
    if "geometry_dose_capacity_full_feedback" in comparison:
        comparison["geometry_dose_minus_full_feedback"] = (
            comparison["geometry_dose_capacity_full_feedback"] - comparison["full_feedback"]
        )
    comparison.to_csv(OUT_DATA / "geometry_normalized_controller_comparison.csv")
    large_delta = float(comparison.loc["large_geometry", "geometry_minus_full_feedback"])
    low_ps_delta = float(comparison.loc["low_photosensitizer", "geometry_minus_full_feedback"])
    has_capacity = "geometry_dose_minus_full_feedback" in comparison
    large_capacity_delta = float(comparison.loc["large_geometry", "geometry_dose_minus_full_feedback"]) if has_capacity else float("nan")
    low_ps_capacity_delta = float(comparison.loc["low_photosensitizer", "geometry_dose_minus_full_feedback"]) if has_capacity else float("nan")
    lines = [
        "# Geometry-normalized controller pilot",
        "",
        "This analysis tests whether a controller scaled by spheroid radius, thermal diffusion time, perfusion recovery, and delay-diffusion load improves the atlas failure mode observed for the large-geometry holdout.",
        "",
        f"Rows analyzed: {len(outcomes)}.",
        "",
        "## Success probability by model family",
        "",
        "```",
        comparison.round(3).to_string(),
        "```",
        "",
        "## Strict interpretation",
        "",
        f"Large-geometry success changed by {large_delta:+.3f} relative to ordinary full feedback.",
        f"Low-photosensitizer success changed by {low_ps_delta:+.3f}, showing that the present geometry-only policy can under-dose weak photochemical cases.",
        f"Geometry-plus-dose success changed by {large_capacity_delta:+.3f} in large geometry and {low_ps_capacity_delta:+.3f} in low photosensitizer relative to ordinary full feedback.",
        "",
        "A Nature Computational Science-level claim requires the geometry-plus-dose controller to improve the weak holdouts without sacrificing baseline transfer. If it does not, the result should be treated as a diagnostic failure that motivates a learned or optimized controller rather than a hand-designed law.",
    ]
    (OUT_NOTES / "geometry_normalized_controller_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    base = sample_uncertainty(n=N_DRAWS, seed=SEED)
    draws = make_model_family_draws(base)
    scenarios = atlas_scenarios()
    feedback_policies, predictive_policy_list, fixed_policy = atlas_policies()

    outcomes: list[dict] = []
    traces: list[dict] = []
    total = len(draws) * len(scenarios)
    processed = 0
    for _, draw in draws.iterrows():
        for scenario in scenarios:
            processed += 1
            if processed % 40 == 0:
                print(f"processed {processed}/{total} draw-scenarios")

            fixed = simulate_fixed_case(fixed_policy, draw)
            outcomes.append(add_draw_metadata(fixed, draw, scenario, "fixed"))

            for policy in feedback_policies:
                result, trace_rows = simulate_ablation_case(policy, scenario, draw, rng)
                outcomes.append(add_draw_metadata(result, draw, scenario, "feedback"))
                traces.extend(add_trace_metadata(trace_rows, draw, "feedback"))

            geometry_policy = geometry_normalized_policy(draw, scenario)
            result, trace_rows = simulate_ablation_case(geometry_policy, scenario, draw, rng)
            result.update(
                {
                    "geometry_load": geometry_policy["geometry_load"],
                    "diffusion_ratio": geometry_policy["diffusion_ratio"],
                    "delay_diffusion_ratio_policy": geometry_policy["delay_diffusion_ratio_policy"],
                    "control_load": geometry_policy["control_load"],
                }
            )
            outcomes.append(add_draw_metadata(result, draw, scenario, "geometry_feedback"))
            traces.extend(add_trace_metadata(trace_rows, draw, "geometry_feedback"))

            capacity_policy = geometry_dose_capacity_policy(draw, scenario)
            result, trace_rows = simulate_ablation_case(capacity_policy, scenario, draw, rng)
            result.update(
                {
                    "geometry_load": capacity_policy["geometry_load"],
                    "diffusion_ratio": capacity_policy["diffusion_ratio"],
                    "delay_diffusion_ratio_policy": capacity_policy["delay_diffusion_ratio_policy"],
                    "control_load": capacity_policy["control_load"],
                    "dose_capacity_number": capacity_policy["dose_capacity_number"],
                    "dose_deficit": capacity_policy["dose_deficit"],
                }
            )
            outcomes.append(add_draw_metadata(result, draw, scenario, "geometry_dose_feedback"))
            traces.extend(add_trace_metadata(trace_rows, draw, "geometry_dose_feedback"))

            for policy in predictive_policy_list:
                result, trace_rows = simulate_predictive_case(policy, scenario, draw, rng)
                outcomes.append(add_draw_metadata(result, draw, scenario, "predictive"))
                traces.extend(add_trace_metadata(trace_rows, draw, "predictive"))

    traces_df = pd.DataFrame(traces)
    trace_info = trace_information_metrics(traces_df, scenarios)
    outcomes_raw = pd.DataFrame(outcomes)
    if not trace_info.empty:
        outcomes_raw = outcomes_raw.merge(
            trace_info,
            on=["model_family", "atlas_uncertainty_id", "controller", "scenario", "controller_class"],
            how="left",
        )
    outcomes_df = add_reduced_features(outcomes_raw)
    summary = summarize(outcomes_df)
    validation = validate_holdout(outcomes_df)

    draws.to_csv(OUT_DATA / "information_atlas_uncertainty_draws.csv", index=False)
    outcomes_df.to_csv(OUT_DATA / "information_atlas_outcomes.csv", index=False)
    traces_df.to_csv(OUT_DATA / "information_atlas_traces.csv", index=False)
    trace_info.to_csv(OUT_DATA / "information_atlas_trace_information.csv", index=False)
    summary.to_csv(OUT_DATA / "information_atlas_summary.csv", index=False)
    validation.to_csv(OUT_DATA / "information_atlas_validation.csv", index=False)
    metadata = {
        "seed": SEED,
        "n_base_draws": N_DRAWS,
        "model_families": sorted(draws["model_family"].unique()),
        "scenarios": scenarios,
        "feedback_policies": feedback_policies,
        "predictive_policies": predictive_policy_list,
        "fixed_policy": fixed_policy,
    }
    (OUT_DATA / "information_atlas_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_summary_figure(outcomes_df, validation, OUT_FIG / "figure_2_information_atlas_pilot.png")
    write_verdict(outcomes_df, validation)
    write_geometry_controller_result(outcomes_df)

    print("Wrote information atlas pilot outputs")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
