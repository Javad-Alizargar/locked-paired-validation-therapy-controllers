#!/usr/bin/env python3
"""Chance-constrained robust protocol optimization under biological uncertainty.

The nominal duty/intensity maps answer: which protocols work for one baseline
microenvironment?  This script asks a reviewer-facing robustness question:
which protocols keep working when perfusion, spheroid radius, photosensitizer
loading, and nanoparticle loading vary simultaneously?
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_first_model import EvidenceDefaults, ModelConfig, simulate_case


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "robust_optimization"
FIG = ROOT / "figures" / "robust_optimization"

N_UNCERTAINTY = 40
SEED = 20260706
CHANCE_THRESHOLD = 0.70
FRONTIER_LOWER_BOUND = 0.20
REFERENCE_PROTOCOL_S = 3000.0
REFERENCE_INTENSITY_W_CM2 = 1.6


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    samples = np.zeros((n, dims))
    for j in range(dims):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.random(n)) / n
    return samples


def sample_uncertainty(n: int = N_UNCERTAINTY, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lhs = latin_hypercube(n, 4, rng)
    rows = []
    for i, row in enumerate(lhs):
        perfusion = 10 ** (math.log10(0.003) + row[0] * (math.log10(0.030) - math.log10(0.003)))
        radius_um = 300.0 + row[1] * 500.0
        photosensitizer_um = 10 ** (math.log10(0.05) + row[2] * (math.log10(2.0) - math.log10(0.05)))
        volume_fraction = 0.0015 + row[3] * 0.0030
        rows.append(
            {
                "uncertainty_id": i,
                "perfusion_s": float(perfusion),
                "radius_um": float(radius_um),
                "photosensitizer_um": float(photosensitizer_um),
                "volume_fraction": float(volume_fraction),
            }
        )
    return pd.DataFrame(rows)


def candidate_protocols() -> pd.DataFrame:
    duties = np.array([0.08, 0.10, 0.125, 0.16, 0.20, 0.275, 0.35, 0.45, 0.55, 0.65])
    intensities = np.array([0.20, 0.244, 0.30, 0.40, 0.55, 0.70, 0.90, 1.10, 1.30, 1.50])
    rows = []
    protocol_id = 0
    for duty in duties:
        for intensity in intensities:
            rows.append(
                {
                    "protocol_id": protocol_id,
                    "duty_cycle": float(duty),
                    "intensity_w_cm2": float(intensity),
                    "protocol_duration_s": float(300.0 / duty),
                    "light_on_time_s": 300.0,
                    "delivered_energy_j_cm2": float(300.0 * intensity),
                    "protocol_burden": float((300.0 / duty) / REFERENCE_PROTOCOL_S + intensity / REFERENCE_INTENSITY_W_CM2),
                }
            )
            protocol_id += 1
    return pd.DataFrame(rows)


def success_flags(result: dict) -> dict:
    warm = result["warm_fraction"] >= 0.50
    ros = result["ros_threshold_fraction"] >= 0.50
    low_hypoxia = result["hypoxic_failure_fraction"] <= 0.35
    safe_heat = result["overheat_fraction"] <= 0.05 and result["max_temperature_c"] <= 55.0
    strict_safe_dual = result["dual_modality_score"] >= 0.95 and result["overheat_fraction"] <= 0.0
    return {
        "chance_success": bool(warm and ros and low_hypoxia and safe_heat),
        "strict_safe_dual": bool(strict_safe_dual),
        "thermal_failure": bool(not warm and safe_heat),
        "ros_failure": bool(not ros),
        "hypoxic_failure": bool(not low_hypoxia),
        "overheat_failure": bool(not safe_heat),
    }


def run_simulation_grid(protocols: pd.DataFrame, uncertainty: pd.DataFrame) -> pd.DataFrame:
    ev = EvidenceDefaults()
    cfg = ModelConfig()
    rows: list[dict] = []
    for _, protocol in protocols.iterrows():
        for _, draw in uncertainty.iterrows():
            cfg2 = replace(
                cfg,
                spheroid_radius_m=float(draw["radius_um"]) * 1e-6,
                photosensitizer_um_at_reference_phi=float(draw["photosensitizer_um"]),
            )
            result = simulate_case(
                wavelength_nm=808.0,
                length_nm=70.0,
                width_nm=18.0,
                volume_fraction=float(draw["volume_fraction"]),
                sar_mw_per_kg=0.0,
                intensity_w_cm2=float(protocol["intensity_w_cm2"]),
                perfusion_s=float(draw["perfusion_s"]),
                duty_cycle=float(protocol["duty_cycle"]),
                cfg=cfg2,
                ev=ev,
            )
            flags = success_flags(result)
            row = {
                "protocol_id": int(protocol["protocol_id"]),
                "uncertainty_id": int(draw["uncertainty_id"]),
                "duty_cycle": float(protocol["duty_cycle"]),
                "intensity_w_cm2": float(protocol["intensity_w_cm2"]),
                "protocol_duration_s": float(result["protocol_duration_s"]),
                "delivered_energy_j_cm2": float(protocol["delivered_energy_j_cm2"]),
                "protocol_burden": float(protocol["protocol_burden"]),
                "perfusion_s": float(draw["perfusion_s"]),
                "radius_um": float(draw["radius_um"]),
                "photosensitizer_um": float(draw["photosensitizer_um"]),
                "volume_fraction": float(draw["volume_fraction"]),
                "max_temperature_c": float(result["max_temperature_c"]),
                "warm_fraction": float(result["warm_fraction"]),
                "overheat_fraction": float(result["overheat_fraction"]),
                "ros_threshold_fraction": float(result["ros_threshold_fraction"]),
                "hypoxic_failure_fraction": float(result["hypoxic_failure_fraction"]),
                "dual_modality_score": float(result["dual_modality_score"]),
            }
            row.update(flags)
            rows.append(row)
    return pd.DataFrame(rows)


def wilson_lower_bound(successes: pd.Series, z: float = 1.96) -> float:
    n = int(successes.count())
    if n == 0:
        return math.nan
    phat = float(successes.mean())
    denom = 1.0 + z**2 / n
    center = phat + z**2 / (2.0 * n)
    spread = z * math.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n)
    return float((center - spread) / denom)


def summarize_protocols(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("protocol_id", as_index=False)
    summary = grouped.agg(
        duty_cycle=("duty_cycle", "first"),
        intensity_w_cm2=("intensity_w_cm2", "first"),
        protocol_duration_s=("protocol_duration_s", "first"),
        delivered_energy_j_cm2=("delivered_energy_j_cm2", "first"),
        protocol_burden=("protocol_burden", "first"),
        chance_success_probability=("chance_success", "mean"),
        strict_safe_dual_probability=("strict_safe_dual", "mean"),
        mean_dual_score=("dual_modality_score", "mean"),
        mean_positive_dual_score=("dual_modality_score", lambda x: float(np.maximum(x, 0.0).mean())),
        median_dual_score=("dual_modality_score", "median"),
        q10_dual_score=("dual_modality_score", lambda x: float(np.quantile(x, 0.10))),
        median_tmax_c=("max_temperature_c", "median"),
        q90_tmax_c=("max_temperature_c", lambda x: float(np.quantile(x, 0.90))),
        mean_ros_fraction=("ros_threshold_fraction", "mean"),
        thermal_failure_probability=("thermal_failure", "mean"),
        ros_failure_probability=("ros_failure", "mean"),
        hypoxic_failure_probability=("hypoxic_failure", "mean"),
        overheat_failure_probability=("overheat_failure", "mean"),
    )
    lower = rows.groupby("protocol_id")["chance_success"].apply(wilson_lower_bound)
    summary["chance_success_wilson95_low"] = summary["protocol_id"].map(lower)
    summary["chance_constraint_met"] = summary["chance_success_wilson95_low"] >= CHANCE_THRESHOLD
    summary["prototype_robust_frontier"] = summary["chance_success_wilson95_low"] >= FRONTIER_LOWER_BOUND
    summary["robust_utility"] = (
        summary["chance_success_wilson95_low"]
        * summary["mean_positive_dual_score"]
        / ((summary["protocol_burden"] + 1.0e-12) * (1.0 + summary["overheat_failure_probability"]))
    )
    summary["risk_adjusted_score"] = (
        summary["chance_success_probability"]
        + summary["chance_success_wilson95_low"]
        + summary["q10_dual_score"].clip(lower=0.0)
        - 0.15 * summary["overheat_failure_probability"]
        - 0.10 * summary["protocol_burden"]
    )
    return summary.sort_values(["prototype_robust_frontier", "chance_success_probability", "robust_utility"], ascending=False)


def bootstrap_best_protocol(summary: pd.DataFrame, rows: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    protocol_ids = summary["protocol_id"].to_numpy()
    counts = {int(pid): 0 for pid in protocol_ids}
    grouped = {int(pid): frame for pid, frame in rows.groupby("protocol_id")}
    for _ in range(n_boot):
        boot_scores = []
        for pid in protocol_ids:
            frame = grouped[int(pid)]
            sample = frame.sample(n=len(frame), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
            p = float(sample["chance_success"].mean())
            mean_dual = float(sample["dual_modality_score"].mean())
            burden = float(frame["protocol_burden"].iloc[0])
            boot_scores.append((p * mean_dual / (burden + 1.0e-12), int(pid)))
        winner = max(boot_scores)[1]
        counts[winner] += 1
    out = summary[["protocol_id", "duty_cycle", "intensity_w_cm2", "robust_utility"]].copy()
    out["bootstrap_best_frequency"] = out["protocol_id"].map(lambda pid: counts[int(pid)] / n_boot)
    return out.sort_values("bootstrap_best_frequency", ascending=False)


def factor_stress_summary(rows: pd.DataFrame) -> pd.DataFrame:
    top_ids = (
        rows.groupby("protocol_id")["chance_success"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    top = rows[rows["protocol_id"].isin(top_ids)].copy()
    stress_rows = []
    for factor in ["perfusion_s", "radius_um", "photosensitizer_um", "volume_fraction"]:
        x = np.log10(top[factor]) if factor in {"perfusion_s", "photosensitizer_um"} else top[factor]
        q25 = float(np.quantile(x, 0.25))
        q75 = float(np.quantile(x, 0.75))
        low = top[x <= q25]
        high = top[x >= q75]
        stress_rows.append(
            {
                "factor": factor,
                "success_low_quartile": float(low["chance_success"].mean()),
                "success_high_quartile": float(high["chance_success"].mean()),
                "high_minus_low": float(high["chance_success"].mean() - low["chance_success"].mean()),
                "overheat_low_quartile": float(low["overheat_failure"].mean()),
                "overheat_high_quartile": float(high["overheat_failure"].mean()),
                "ros_failure_low_quartile": float(low["ros_failure"].mean()),
                "ros_failure_high_quartile": float(high["ros_failure"].mean()),
            }
        )
    return pd.DataFrame(stress_rows).sort_values("high_minus_low")


def plot_robust_figure(summary: pd.DataFrame, rows: pd.DataFrame, stress: pd.DataFrame) -> None:
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
    pivot = summary.pivot_table(
        index="duty_cycle",
        columns="intensity_w_cm2",
        values="chance_success_probability",
        aggfunc="max",
    ).sort_index()
    mesh = ax.pcolormesh(pivot.columns, pivot.index, pivot.to_numpy(), cmap="viridis", shading="auto", vmin=0, vmax=1)
    ax.contour(
        pivot.columns,
        pivot.index,
        pivot.to_numpy(),
        levels=[FRONTIER_LOWER_BOUND],
        colors="white",
        linewidths=1.2,
    )
    ax.set_xlabel("Intensity (W cm$^{-2}$)")
    ax.set_ylabel("Duty cycle")
    ax.set_title("A. Chance-constrained success probability")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("P(success)")

    ax = axes[0, 1]
    colors = np.where(summary["prototype_robust_frontier"], "#2f7f4f", "#b8b8b8")
    ax.scatter(
        summary["protocol_burden"],
        summary["chance_success_wilson95_low"],
        c=colors,
        s=34 + 90 * summary["mean_dual_score"].clip(lower=0.0),
        alpha=0.85,
        linewidth=0,
    )
    ax.axhline(CHANCE_THRESHOLD, color="#333333", linestyle="--", linewidth=0.9)
    ax.axhline(FRONTIER_LOWER_BOUND, color="#2f7f4f", linestyle=":", linewidth=1.1)
    ax.set_xlabel("Protocol burden")
    ax.set_ylabel("Wilson 95% lower bound")
    ax.set_title("B. Strict chance constraint and robust frontier")

    ax = axes[0, 2]
    top = summary.sort_values("robust_utility", ascending=False).head(8).iloc[::-1]
    labels = [f"d={d:.2f}, I={i:.2f}" for d, i in zip(top["duty_cycle"], top["intensity_w_cm2"])]
    ax.barh(np.arange(len(top)), top["chance_success_probability"], color="#2878b5", label="mean")
    ax.barh(np.arange(len(top)), top["chance_success_wilson95_low"], color="#164f7a", label="95% low")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Chance of success")
    ax.set_title("C. Top robust protocols")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    ax.scatter(
        summary["delivered_energy_j_cm2"],
        summary["mean_dual_score"],
        c=summary["chance_success_probability"],
        cmap="magma",
        s=40,
        alpha=0.85,
        linewidth=0,
    )
    ax.set_xlabel("Delivered energy (J cm$^{-2}$)")
    ax.set_ylabel("Mean dual score under uncertainty")
    ax.set_title("D. Robust benefit versus optical energy")
    cbar = fig.colorbar(ax.collections[0], ax=ax, pad=0.01)
    cbar.set_label("P(success)")

    ax = axes[1, 1]
    best_id = int(summary.sort_values("robust_utility", ascending=False).iloc[0]["protocol_id"])
    best_rows = rows[rows["protocol_id"] == best_id]
    ax.hist(best_rows["dual_modality_score"], bins=np.linspace(-2, 1, 25), color="#6f7fbf", alpha=0.86)
    ax.axvline(best_rows["dual_modality_score"].median(), color="#222222", linewidth=1.2)
    ax.set_xlabel("Dual-window score")
    ax.set_ylabel("Uncertainty draws")
    ax.set_title("E. Score distribution for best robust protocol")

    ax = axes[1, 2]
    factors = stress["factor"].str.replace("_", " ", regex=False)
    ax.barh(np.arange(len(stress)), stress["high_minus_low"], color="#9b4d96")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(np.arange(len(stress)))
    ax.set_yticklabels(factors)
    ax.set_xlabel("Success(high quartile) - success(low quartile)")
    ax.set_title("F. Stress-test factor direction")

    for suffix in ("svg", "png"):
        fig.savefig(FIG / f"figure_s6_robust_chance_optimization.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    protocols = candidate_protocols()
    uncertainty = sample_uncertainty()
    rows = run_simulation_grid(protocols, uncertainty)
    summary = summarize_protocols(rows)
    boot = bootstrap_best_protocol(summary, rows)
    stress = factor_stress_summary(rows)

    protocols.to_csv(OUT / "robust_candidate_protocols.csv", index=False)
    uncertainty.to_csv(OUT / "robust_uncertainty_draws.csv", index=False)
    rows.to_csv(OUT / "robust_protocol_uncertainty_outcomes.csv", index=False)
    summary.to_csv(OUT / "robust_protocol_summary.csv", index=False)
    boot.to_csv(OUT / "robust_bootstrap_winner_frequency.csv", index=False)
    stress.to_csv(OUT / "robust_factor_stress_summary.csv", index=False)

    best = summary.sort_values("robust_utility", ascending=False).iloc[0]
    best_chance = summary[summary["chance_constraint_met"]].sort_values("robust_utility", ascending=False)
    best_frontier = summary[summary["prototype_robust_frontier"]].sort_values(
        ["chance_success_probability", "robust_utility"], ascending=False
    )
    best_safety_first = summary[summary["overheat_failure_probability"] <= 0.05].sort_values(
        ["chance_success_probability", "robust_utility"], ascending=False
    )
    metadata = {
        "seed": SEED,
        "n_candidate_protocols": int(len(protocols)),
        "n_uncertainty_draws": int(len(uncertainty)),
        "n_simulations": int(len(rows)),
        "chance_threshold": CHANCE_THRESHOLD,
        "prototype_frontier_lower_bound": FRONTIER_LOWER_BOUND,
        "success_definition": {
            "warm_fraction": ">= 0.50",
            "ros_threshold_fraction": ">= 0.50",
            "hypoxic_failure_fraction": "<= 0.35",
            "overheat_fraction": "<= 0.05",
            "max_temperature_c": "<= 55.0",
        },
        "uncertainty_ranges": {
            "perfusion_s": "log-uniform 0.003-0.030",
            "radius_um": "uniform 300-800",
            "photosensitizer_um": "log-uniform 0.05-2.0",
            "volume_fraction": "uniform 0.0015-0.0045",
        },
        "best_robust_protocol": best.to_dict(),
        "n_protocols_meeting_chance_constraint": int(summary["chance_constraint_met"].sum()),
        "n_protocols_on_prototype_robust_frontier": int(summary["prototype_robust_frontier"].sum()),
        "best_chance_constrained_protocol": None if best_chance.empty else best_chance.iloc[0].to_dict(),
        "best_prototype_frontier_protocol": None if best_frontier.empty else best_frontier.iloc[0].to_dict(),
        "best_safety_first_protocol": None if best_safety_first.empty else best_safety_first.iloc[0].to_dict(),
    }
    (OUT / "robust_chance_optimization_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_robust_figure(summary, rows, stress)

    print("Wrote robust chance-constrained optimization outputs:")
    print(f"- {OUT / 'robust_protocol_summary.csv'}")
    print(f"- {OUT / 'robust_protocol_uncertainty_outcomes.csv'}")
    print(f"- {FIG / 'figure_s6_robust_chance_optimization.svg'}")
    print(
        "Best robust protocol: "
        f"duty={best['duty_cycle']:.3f}, "
        f"I={best['intensity_w_cm2']:.3f} W/cm2, "
        f"Psuccess={best['chance_success_probability']:.3f}, "
        f"Wilson95low={best['chance_success_wilson95_low']:.3f}, "
        f"utility={best['robust_utility']:.3f}"
    )
    print(f"Strict chance constraint met by {int(summary['chance_constraint_met'].sum())} protocols.")
    print(f"Prototype robust frontier contains {int(summary['prototype_robust_frontier'].sum())} protocols.")


if __name__ == "__main__":
    main()
