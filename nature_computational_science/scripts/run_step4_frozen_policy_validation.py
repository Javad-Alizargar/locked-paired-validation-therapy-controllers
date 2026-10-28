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

from run_adaptive_feedback_control import simulate_fixed_case  # noqa: E402
from run_feedback_ablation_noise_delay import simulate_ablation_case  # noqa: E402
from run_predictive_delay_control import predictive_policies, simulate_predictive_case  # noqa: E402
from run_robust_chance_optimization import sample_uncertainty  # noqa: E402
from run_step2_information_atlas import (  # noqa: E402
    add_draw_metadata,
    add_reduced_features,
    atlas_scenarios,
    geometry_dose_capacity_policy,
    make_model_family_draws,
    validate_holdout,
)
from run_step3_minimax_policy_optimization import optimized_policy  # noqa: E402


NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data" / "frozen_policy_validation"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

SEED = 20260712
N_DRAWS = 16
FROZEN_POLICY_ID = "random_27"
FROZEN_PARAMS = {
    "candidate_id": FROZEN_POLICY_ID,
    "intensity_geom": 0.030646,
    "intensity_dose": 0.067669,
    "thermal_geom": 1.241952,
    "thermal_dose": 0.346905,
    "oxygen_geom": 2.386116,
    "oxygen_dose": 0.349697,
    "ros_geom": 0.007055,
    "ros_dose": 0.092386,
    "warm_geom": 0.060132,
    "maintenance_floor": 0.183027,
    "oxygen_limited_floor": 0.169704,
    "warmup_floor": 0.561871,
}


def common_rng(draw: pd.Series, scenario_index: int, controller_index: int) -> np.random.Generator:
    seed = SEED + int(draw["atlas_uncertainty_id"]) * 1009 + scenario_index * 9176 + controller_index * 131
    return np.random.default_rng(seed)


def frozen_policy(draw: pd.Series, scenario: dict) -> dict:
    params = pd.Series(FROZEN_PARAMS)
    policy = optimized_policy(draw, scenario, params, FROZEN_POLICY_ID)
    policy["controller"] = "frozen_minimax_geometry_dose"
    return policy


def run_validation() -> pd.DataFrame:
    draws = make_model_family_draws(sample_uncertainty(n=N_DRAWS, seed=SEED))
    scenarios = atlas_scenarios()
    fixed_policy = {"controller": "fixed_reference", "duty_cycle": 0.125, "intensity_w_cm2": 1.10}
    full_feedback = {
        "controller": "full_feedback",
        "signals": ("temperature", "oxygen", "ros"),
        "max_intensity_w_cm2": 1.30,
    }
    predictive = [
        {**p, "controller": "predictive_smoothed_full"}
        for p in predictive_policies()
        if p["controller"] == "predictive_smoothed_full"
    ][0]

    outcomes = []
    total = len(draws) * len(scenarios)
    processed = 0
    for _, draw in draws.iterrows():
        fixed = simulate_fixed_case(fixed_policy, draw)
        for scenario_index, scenario in enumerate(scenarios):
            fixed_row = dict(fixed)
            fixed_row["scenario"] = scenario["scenario"]
            fixed_row["delay_s"] = float(scenario["delay_s"])
            fixed_row["noise_fraction"] = float(scenario["noise_fraction"])
            outcomes.append(add_draw_metadata(fixed_row, draw, scenario, "fixed"))

            policy = full_feedback
            result, _ = simulate_ablation_case(policy, scenario, draw, common_rng(draw, scenario_index, 1))
            outcomes.append(add_draw_metadata(result, draw, scenario, "feedback"))

            policy = geometry_dose_capacity_policy(draw, scenario)
            result, _ = simulate_ablation_case(policy, scenario, draw, common_rng(draw, scenario_index, 2))
            outcomes.append(add_draw_metadata(result, draw, scenario, "geometry_dose_feedback"))

            policy = frozen_policy(draw, scenario)
            result, _ = simulate_ablation_case(policy, scenario, draw, common_rng(draw, scenario_index, 3))
            result.update({key: float(value) for key, value in FROZEN_PARAMS.items() if key != "candidate_id"})
            result["candidate_id"] = FROZEN_POLICY_ID
            result["policy_family"] = "frozen_minimax"
            outcomes.append(add_draw_metadata(result, draw, scenario, "frozen_minimax"))

            result, _ = simulate_predictive_case(predictive, scenario, draw, common_rng(draw, scenario_index, 4))
            outcomes.append(add_draw_metadata(result, draw, scenario, "predictive"))

            processed += 1
            if processed % 80 == 0:
                print(f"processed {processed}/{total} draw-scenarios", flush=True)

    return pd.DataFrame(outcomes)


def make_figure(outcomes: pd.DataFrame, validation: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    pivot = outcomes.pivot_table(index="model_family", columns="controller", values="chance_success", aggfunc="mean")
    keep = [
        "frozen_minimax_geometry_dose",
        "full_feedback",
        "geometry_dose_capacity_full_feedback",
        "predictive_smoothed_full",
        "fixed_reference",
    ]
    pivot[[col for col in keep if col in pivot.columns]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success probability")
    ax.set_title("A. Independent family validation")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[0, 1]
    h = validation[validation["heldout_model_family"] != "apparent_all"]
    ax.barh(h["heldout_model_family"], h["auc"], color="#2f7f4f")
    ax.axvline(0.70, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.80, color="#c44e52", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Held-out AUC")
    ax.set_title("B. Reduced-law transfer")

    ax = axes[1, 0]
    f = outcomes[outcomes["controller"] == "frozen_minimax_geometry_dose"]
    delay = f.groupby(["delay_s", "noise_fraction"], as_index=False).agg(success=("chance_success", "mean"))
    for noise, g in delay.groupby("noise_fraction"):
        ax.plot(g["delay_s"], g["success"], marker="o", label=f"noise={noise:.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Success probability")
    ax.set_title("C. Frozen policy delay/noise response")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.scatter(f["dose_capacity_number"], f["chance_success"].astype(int), c=f["delay_s"], cmap="viridis", s=14, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xlabel("Dose-capacity number")
    ax.set_ylabel("Success")
    ax.set_title("D. Frozen policy outcomes")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)

    outcomes = run_validation()
    outcomes.to_csv(OUT_DATA / "frozen_policy_validation_outcomes.csv", index=False)
    reduced = add_reduced_features(outcomes)
    reduced.to_csv(OUT_DATA / "frozen_policy_validation_reduced_features.csv", index=False)
    summary = outcomes.groupby(["controller", "model_family"], as_index=False).agg(
        success=("chance_success", "mean"),
        strict_safe_dual=("strict_safe_dual", "mean"),
        overheat=("overheat_fraction", "mean"),
        warm=("warm_fraction", "mean"),
        ros=("ros_threshold_fraction", "mean"),
        mean_intensity=("mean_intensity_w_cm2", "mean"),
    )
    summary.to_csv(OUT_DATA / "frozen_policy_validation_summary.csv", index=False)
    validation = validate_holdout(reduced)
    validation.to_csv(OUT_DATA / "frozen_policy_validation_holdout_auc.csv", index=False)
    metadata = {
        "seed": SEED,
        "n_draws": N_DRAWS,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "frozen_params": FROZEN_PARAMS,
        "note": "Independent validation only; no retuning or policy selection is performed in this script.",
    }
    (OUT_DATA / "frozen_policy_validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figure(reduced, validation, OUT_FIG / "figure_4_frozen_policy_validation.png")

    pivot = summary.pivot_table(index="model_family", columns="controller", values="success")
    frozen = summary[summary["controller"] == "frozen_minimax_geometry_dose"]
    full = summary[summary["controller"] == "full_feedback"]
    frozen_worst = float(frozen["success"].min())
    full_worst = float(full["success"].min())
    auc_holdout = validation[validation["heldout_model_family"] != "apparent_all"]
    lines = [
        "# Step 4 Frozen Policy Independent Validation",
        "",
        f"Independent seed: {SEED}.",
        f"Uncertainty draws per model family: {N_DRAWS}.",
        f"Total outcome rows: {len(outcomes)}.",
        "No retuning or policy selection was performed.",
        "",
        "## Frozen Policy Parameters",
        "",
        "```json",
        json.dumps(FROZEN_PARAMS, indent=2),
        "```",
        "",
        "## Success Probability",
        "",
        "```",
        pivot.round(3).to_string(),
        "```",
        "",
        "## Held-Out Reduced-Law AUC",
        "",
        "```",
        validation.round(3).to_string(index=False),
        "```",
        "",
        "## Strict Interpretation",
        "",
        f"Frozen optimized worst-family success: {frozen_worst:.3f}.",
        f"Full-feedback worst-family success: {full_worst:.3f}.",
        f"Median held-out family AUC: {auc_holdout['auc'].median():.3f}.",
        f"Minimum held-out family AUC: {auc_holdout['auc'].min():.3f}.",
        "This is the first independent validation of the frozen policy class. It is not yet a final manuscript-scale atlas.",
    ]
    (OUT_NOTES / "step4_frozen_policy_validation_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote frozen policy validation outputs")
    print(pivot.round(3).to_string())
    print(validation.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
