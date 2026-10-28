from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
NCS_SCRIPT_DIR = ROOT / "nature_computational_science" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(NCS_SCRIPT_DIR))

from run_feedback_ablation_noise_delay import simulate_ablation_case  # noqa: E402
from run_robust_chance_optimization import sample_uncertainty  # noqa: E402
from run_step2_information_atlas import (  # noqa: E402
    EPS,
    REFERENCE_PERFUSION_S,
    REFERENCE_PHOTOSENSITIZER_UM,
    REFERENCE_RADIUS_UM,
    REFERENCE_VOLUME_FRACTION,
    TISSUE_ALPHA_M2_S,
    add_draw_metadata,
    atlas_scenarios,
    geometry_dose_capacity_policy,
    make_model_family_draws,
)


NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data" / "minimax_policy"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

SEED = 20260711
N_OPT_DRAWS = 4
N_VALIDATE_DRAWS = 8
N_RANDOM_CANDIDATES = 28
N_VALIDATE_TOP = 4


def selected_scenarios() -> list[dict]:
    return [
        row
        for row in atlas_scenarios()
        if row["delay_s"] in {0.0, 90.0, 180.0} and row["noise_fraction"] in {0.0, 0.10}
    ]


def lhs_candidates(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bounds = {
        "intensity_geom": (0.00, 0.16),
        "intensity_dose": (0.00, 0.16),
        "thermal_geom": (0.60, 1.50),
        "thermal_dose": (0.00, 0.45),
        "oxygen_geom": (0.60, 2.60),
        "oxygen_dose": (0.00, 0.85),
        "ros_geom": (0.00, 0.04),
        "ros_dose": (0.00, 0.16),
        "warm_geom": (0.00, 0.07),
        "maintenance_floor": (0.14, 0.20),
        "oxygen_limited_floor": (0.16, 0.24),
        "warmup_floor": (0.54, 0.64),
    }
    rows = []
    for i in range(n):
        row = {"candidate_id": f"random_{i:02d}"}
        for key, (lo, hi) in bounds.items():
            row[key] = float(lo + (hi - lo) * rng.random())
        rows.append(row)
    defaults = {
        "candidate_id": "current_geometry_dose",
        "intensity_geom": 0.08,
        "intensity_dose": 0.08,
        "thermal_geom": 1.10,
        "thermal_dose": 0.35,
        "oxygen_geom": 1.80,
        "oxygen_dose": 0.45,
        "ros_geom": 0.020,
        "ros_dose": 0.120,
        "warm_geom": 0.040,
        "maintenance_floor": 0.16,
        "oxygen_limited_floor": 0.18,
        "warmup_floor": 0.58,
    }
    conservative = {
        **defaults,
        "candidate_id": "geometry_safe_reference",
        "intensity_geom": 0.10,
        "intensity_dose": 0.04,
        "thermal_geom": 1.30,
        "thermal_dose": 0.15,
        "oxygen_geom": 2.20,
        "oxygen_dose": 0.20,
        "ros_dose": 0.06,
        "warmup_floor": 0.60,
    }
    dose_preserving = {
        **defaults,
        "candidate_id": "dose_preserving_reference",
        "intensity_geom": 0.04,
        "intensity_dose": 0.02,
        "thermal_geom": 0.85,
        "thermal_dose": 0.08,
        "oxygen_geom": 1.20,
        "oxygen_dose": 0.10,
        "ros_dose": 0.04,
        "warm_geom": 0.02,
        "maintenance_floor": 0.18,
        "oxygen_limited_floor": 0.22,
        "warmup_floor": 0.62,
    }
    return pd.concat([pd.DataFrame([defaults, conservative, dose_preserving]), pd.DataFrame(rows)], ignore_index=True)


def optimized_policy(draw: pd.Series, scenario: dict, params: pd.Series, candidate_id: str) -> dict:
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

    max_intensity = 1.30 / (
        1.0
        + float(params["intensity_geom"]) * positive_geometry_load
        + float(params["intensity_dose"]) * dose_deficit
    )
    max_intensity = float(np.clip(max_intensity, 1.04, 1.30))
    thermal_shift = float(
        np.clip(float(params["thermal_geom"]) * positive_geometry_load + float(params["thermal_dose"]) * dose_deficit, 0.0, 2.0)
    )
    oxygen_shift = float(
        np.clip(float(params["oxygen_geom"]) * positive_geometry_load + float(params["oxygen_dose"]) * dose_deficit, 0.0, 3.5)
    )
    ros_completion = float(
        np.clip(
            0.80 - float(params["ros_geom"]) * positive_geometry_load - float(params["ros_dose"]) * dose_deficit,
            0.58,
            0.80,
        )
    )

    return {
        "controller": f"optimized_minimax_{candidate_id}",
        "signals": ("temperature", "oxygen", "ros"),
        "max_intensity_w_cm2": max_intensity,
        "safety_stop_c": 52.0 - 0.45 * thermal_shift,
        "thermal_hold_c": 49.0 - thermal_shift,
        "thermal_target_c": 45.5 - 0.10 * positive_geometry_load,
        "oxygen_floor_um": 10.0 + oxygen_shift,
        "oxygen_low_um": 14.0 + oxygen_shift,
        "ros_completion": ros_completion,
        "warm_target": float(np.clip(0.50 + float(params["warm_geom"]) * positive_geometry_load, 0.50, 0.58)),
        "maintenance_intensity_w_cm2": float(np.clip(0.20 * max_intensity / 1.30, float(params["maintenance_floor"]), 0.20)),
        "oxygen_limited_fraction": float(np.clip(0.25 / (1.0 + 0.30 * positive_geometry_load), float(params["oxygen_limited_floor"]), 0.25)),
        "warmup_fraction": float(np.clip(0.65 / (1.0 + 0.06 * positive_geometry_load), float(params["warmup_floor"]), 0.65)),
        "ros_progress_fraction": float(np.clip(0.40 / (1.0 + 0.05 * positive_geometry_load), 0.36, 0.40)),
        "geometry_load": float(geometry_load),
        "dose_capacity_number": dose_capacity,
        "dose_deficit": dose_deficit,
    }


def common_rng(draw: pd.Series, scenario_index: int, candidate_index: int = 0) -> np.random.Generator:
    seed = SEED + int(draw["atlas_uncertainty_id"]) * 1009 + scenario_index * 9176 + candidate_index * 0
    return np.random.default_rng(seed)


def evaluate_policy(
    draws: pd.DataFrame,
    scenarios: list[dict],
    params: pd.Series,
    candidate_id: str,
    candidate_index: int,
    include_reference: bool = False,
) -> pd.DataFrame:
    rows = []
    for _, draw in draws.iterrows():
        for scenario_index, scenario in enumerate(scenarios):
            policy = optimized_policy(draw, scenario, params, candidate_id)
            result, _ = simulate_ablation_case(policy, scenario, draw, common_rng(draw, scenario_index, candidate_index))
            result.update({key: float(params[key]) for key in params.index if key != "candidate_id"})
            result["candidate_id"] = candidate_id
            result["policy_family"] = "optimized_minimax"
            result["geometry_load"] = policy["geometry_load"]
            result["dose_capacity_number"] = policy["dose_capacity_number"]
            result["dose_deficit"] = policy["dose_deficit"]
            rows.append(add_draw_metadata(result, draw, scenario, "optimized_minimax"))
            if include_reference and candidate_id == "best":
                ref = geometry_dose_capacity_policy(draw, scenario)
                ref_result, _ = simulate_ablation_case(ref, scenario, draw, common_rng(draw, scenario_index, candidate_index))
                ref_result["candidate_id"] = "current_geometry_dose"
                ref_result["policy_family"] = "current_geometry_dose"
                rows.append(add_draw_metadata(ref_result, draw, scenario, "geometry_dose_feedback"))
    return pd.DataFrame(rows)


def summarize_candidates(outcomes: pd.DataFrame) -> pd.DataFrame:
    family = outcomes.groupby(["candidate_id", "model_family"], as_index=False).agg(
        success=("chance_success", "mean"),
        overheat=("overheat_fraction", "mean"),
        warm=("warm_fraction", "mean"),
        ros=("ros_threshold_fraction", "mean"),
        mean_intensity=("mean_intensity_w_cm2", "mean"),
    )
    rows = []
    for candidate_id, g in family.groupby("candidate_id"):
        rows.append(
            {
                "candidate_id": candidate_id,
                "min_family_success": float(g["success"].min()),
                "mean_family_success": float(g["success"].mean()),
                "std_family_success": float(g["success"].std(ddof=0)),
                "mean_overheat": float(g["overheat"].mean()),
                "min_warm": float(g["warm"].min()),
                "min_ros": float(g["ros"].min()),
                "minimax_score": float(g["success"].min() - 0.15 * g["success"].std(ddof=0) - 0.25 * g["overheat"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["minimax_score", "min_family_success", "mean_family_success"], ascending=False)


def make_figure(validation: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    ax = axes[0]
    pivot = validation.pivot_table(index="model_family", columns="candidate_id", values="chance_success", aggfunc="mean")
    validated_scores = pivot.min(axis=0).sort_values(ascending=False)
    validated_best = str(validated_scores.index[0])
    keep = [col for col in [validated_best, "current_geometry_dose", "best"] if col in pivot.columns]
    keep = list(dict.fromkeys(keep))
    pivot[keep].plot(kind="bar", ax=ax, color=["#2f7f4f", "#8a6fba", "#bdbdbd"][: len(keep)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success probability")
    ax.set_title("A. Full-grid validation")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    best = validation[validation["candidate_id"] == validated_best]
    ax.scatter(best["dose_capacity_number"], best["chance_success"].astype(int), s=16, alpha=0.35, c=best["delay_s"], cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel("Dose-capacity number")
    ax.set_ylabel("Success")
    ax.set_title("B. Optimized policy outcomes")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)

    opt_draws = make_model_family_draws(sample_uncertainty(n=N_OPT_DRAWS, seed=SEED))
    opt_scenarios = selected_scenarios()
    candidates = lhs_candidates(N_RANDOM_CANDIDATES, SEED)
    candidates.to_csv(OUT_DATA / "minimax_policy_candidates.csv", index=False)

    opt_rows = []
    for candidate_index, params in candidates.iterrows():
        candidate_id = str(params["candidate_id"])
        opt_rows.append(evaluate_policy(opt_draws, opt_scenarios, params, candidate_id, candidate_index))
        print(f"evaluated {candidate_index + 1}/{len(candidates)} candidates: {candidate_id}", flush=True)
    optimization = pd.concat(opt_rows, ignore_index=True)
    candidate_summary = summarize_candidates(optimization)
    optimization.to_csv(OUT_DATA / "minimax_policy_optimization_outcomes.csv", index=False)
    candidate_summary.to_csv(OUT_DATA / "minimax_policy_candidate_summary.csv", index=False)

    top_ids = candidate_summary["candidate_id"].head(N_VALIDATE_TOP).tolist()
    validation_draws = make_model_family_draws(sample_uncertainty(n=N_VALIDATE_DRAWS, seed=20260710))
    validation_scenarios = atlas_scenarios()
    validation_rows = []
    for rank, candidate_id in enumerate(top_ids):
        params = candidates[candidates["candidate_id"] == candidate_id].iloc[0]
        label = "best" if rank == 0 else f"rank_{rank + 1}_{candidate_id}"
        validation_rows.append(evaluate_policy(validation_draws, validation_scenarios, params, label, rank))
    best_params = candidates[candidates["candidate_id"] == top_ids[0]].iloc[0]
    validation_rows.append(evaluate_policy(validation_draws, validation_scenarios, best_params, "best", 0, include_reference=True))
    validation = pd.concat(validation_rows, ignore_index=True).drop_duplicates(
        subset=["candidate_id", "model_family", "atlas_uncertainty_id", "scenario", "controller"]
    )
    validation.to_csv(OUT_DATA / "minimax_policy_validation_outcomes.csv", index=False)
    validation_summary = validation.groupby(["candidate_id", "model_family"], as_index=False).agg(
        success=("chance_success", "mean"),
        overheat=("overheat_fraction", "mean"),
        warm=("warm_fraction", "mean"),
        ros=("ros_threshold_fraction", "mean"),
        mean_intensity=("mean_intensity_w_cm2", "mean"),
    )
    validation_summary.to_csv(OUT_DATA / "minimax_policy_validation_summary.csv", index=False)
    make_figure(validation, OUT_FIG / "figure_3_minimax_policy_optimization.png")

    validation_pivot = validation_summary.pivot_table(index="model_family", columns="candidate_id", values="success")
    validation_scores = validation_pivot.min(axis=0).sort_values(ascending=False)
    validated_best_id = str(validation_scores.index[0])
    screening_best_family = validation_summary[validation_summary["candidate_id"] == "best"]
    validated_best_family = validation_summary[validation_summary["candidate_id"] == validated_best_id]
    current_family = validation_summary[validation_summary["candidate_id"] == "current_geometry_dose"]
    lines = [
        "# Step 3 Minimax Policy Optimization",
        "",
        f"Optimization candidates: {len(candidates)}.",
        f"Optimization draw-scenarios per candidate: {len(opt_draws) * len(opt_scenarios)}.",
        f"Validated top candidates on full pilot draw-scenarios: {len(validation_draws) * len(validation_scenarios)}.",
        "",
        "## Best Candidate Parameters",
        "",
        "```json",
        json.dumps({k: best_params[k].item() if hasattr(best_params[k], "item") else best_params[k] for k in candidates.columns}, indent=2),
        "```",
        "",
        "## Full-Grid Validation",
        "",
        "```",
        validation_pivot.round(3).to_string(),
        "```",
        "",
        "Worst-family validation success:",
        "",
        "```",
        validation_scores.round(3).to_string(),
        "```",
        "",
        "## Strict Interpretation",
        "",
        f"Screening-best optimized worst-family success: {screening_best_family['success'].min():.3f}.",
        f"Validation-selected optimized policy: {validated_best_id}.",
        f"Validation-selected worst-family success: {validated_best_family['success'].min():.3f}.",
        f"Current hand-designed geometry-dose worst-family success: {current_family['success'].min():.3f}.",
        "This is still a pilot-scale result. A manuscript-level claim requires a larger atlas and held-out validation after the policy class is frozen.",
    ]
    (OUT_NOTES / "step3_minimax_policy_optimization_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote minimax policy optimization outputs")
    print(candidate_summary.head(8).to_string(index=False))
    print(validation_summary.pivot_table(index="model_family", columns="candidate_id", values="success").round(3).to_string())


if __name__ == "__main__":
    main()
