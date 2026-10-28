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
)
from run_step3_minimax_policy_optimization import optimized_policy  # noqa: E402
from run_step4_frozen_policy_validation import FROZEN_PARAMS, FROZEN_POLICY_ID  # noqa: E402
from run_step5_symbolic_boundary_discovery import auc_score  # noqa: E402


NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data" / "external_boundary_validation"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

SEED = 20260713
N_DRAWS = 16
EPS = 1e-12

BOUNDARY = {
    "intercept": -0.129,
    "terms": [
        {"feature": "observation_quality", "coef": 0.735, "mean": 0.447, "sd": 0.317},
        {"feature": "log_reaction_number", "coef": 0.643, "mean": 0.726, "sd": 0.436},
        {"feature": "duty_proxy*log_dose_capacity_number", "coef": 0.612, "mean": -0.321, "sd": 0.368},
        {"feature": "log_thermal_diffusion_time_s", "coef": 0.702, "mean": 0.330, "sd": 0.255},
    ],
    "source": "step5_symbolic_boundary_discovery_result.md",
}


def common_rng(draw: pd.Series, scenario_index: int, controller_index: int) -> np.random.Generator:
    seed = SEED + int(draw["atlas_uncertainty_id"]) * 1009 + scenario_index * 9176 + controller_index * 131
    return np.random.default_rng(seed)


def frozen_policy(draw: pd.Series, scenario: dict) -> dict:
    params = pd.Series(FROZEN_PARAMS)
    policy = optimized_policy(draw, scenario, params, FROZEN_POLICY_ID)
    policy["controller"] = "frozen_minimax_geometry_dose"
    return policy


def run_external_atlas() -> pd.DataFrame:
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

            result, _ = simulate_ablation_case(full_feedback, scenario, draw, common_rng(draw, scenario_index, 1))
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


def apply_frozen_boundary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["duty_proxy*log_dose_capacity_number"] = out["duty_proxy"] * out["log_dose_capacity_number"]
    score = np.full(len(out), BOUNDARY["intercept"], dtype=float)
    for term in BOUNDARY["terms"]:
        z = (out[term["feature"]].astype(float) - term["mean"]) / (term["sd"] + EPS)
        score += term["coef"] * z
    out["frozen_boundary_score"] = score
    out["frozen_boundary_probability"] = 1 / (1 + np.exp(-np.clip(score, -40, 40)))
    return out


def boundary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, g in df.groupby("model_family"):
        y = g["chance_success"].astype(int).to_numpy()
        p = g["frozen_boundary_probability"].to_numpy(float)
        rows.append(
            {
                "model_family": family,
                "n": len(g),
                "success_rate": float(y.mean()),
                "auc": auc_score(y, p),
                "brier": float(np.mean((p - y) ** 2)),
                "mean_predicted_probability": float(np.mean(p)),
                "mean_boundary_score": float(np.mean(g["frozen_boundary_score"])),
            }
        )
    y = df["chance_success"].astype(int).to_numpy()
    p = df["frozen_boundary_probability"].to_numpy(float)
    rows.append(
        {
            "model_family": "all",
            "n": len(df),
            "success_rate": float(y.mean()),
            "auc": auc_score(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "mean_predicted_probability": float(np.mean(p)),
            "mean_boundary_score": float(np.mean(df["frozen_boundary_score"])),
        }
    )
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    h = metrics[metrics["model_family"] != "all"]
    ax.barh(h["model_family"], h["auc"], color="#2f7f4f")
    ax.axvline(0.70, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.75, color="#c44e52", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC, no refit")
    ax.set_title("A. External boundary validation")

    ax = axes[0, 1]
    ok = df["chance_success"].astype(bool)
    ax.scatter(df.loc[~ok, "frozen_boundary_score"], df.loc[~ok, "dose_capacity_number"], s=10, alpha=0.18, color="black", label="failure")
    ax.scatter(df.loc[ok, "frozen_boundary_score"], df.loc[ok, "dose_capacity_number"], s=10, alpha=0.28, color="#2ca25f", label="success")
    ax.axvline(0, color="#c44e52", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Frozen boundary score")
    ax.set_ylabel("Dose-capacity number")
    ax.set_title("B. External score separation")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    bins = np.linspace(df["frozen_boundary_score"].quantile(0.02), df["frozen_boundary_score"].quantile(0.98), 9)
    cal = df.groupby(["model_family", pd.cut(df["frozen_boundary_score"], bins=bins)], observed=False)["chance_success"].mean().reset_index()
    for family, g in cal.groupby("model_family"):
        mids = [interval.mid for interval in g["frozen_boundary_score"]]
        ax.plot(mids, g["chance_success"], marker="o", label=family)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frozen boundary score bin")
    ax.set_ylabel("Observed success probability")
    ax.set_title("C. External calibration by family")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    pivot = df.pivot_table(index="model_family", columns="controller", values="chance_success", aggfunc="mean")
    keep = ["predictive_smoothed_full", "frozen_minimax_geometry_dose", "full_feedback", "fixed_reference"]
    pivot[[col for col in keep if col in pivot.columns]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success probability")
    ax.set_title("D. Controller outcomes in external atlas")
    ax.legend(frameon=False, fontsize=6)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    outcomes = run_external_atlas()
    outcomes.to_csv(OUT_DATA / "external_boundary_validation_outcomes.csv", index=False)
    reduced = add_reduced_features(outcomes)
    reduced = apply_frozen_boundary(reduced)
    reduced.to_csv(OUT_DATA / "external_boundary_validation_reduced_scored.csv", index=False)
    metrics = boundary_metrics(reduced)
    metrics.to_csv(OUT_DATA / "external_boundary_validation_metrics.csv", index=False)
    summary = reduced.groupby(["controller", "model_family"], as_index=False).agg(
        success=("chance_success", "mean"),
        overheat=("overheat_fraction", "mean"),
        warm=("warm_fraction", "mean"),
        ros=("ros_threshold_fraction", "mean"),
        mean_boundary_probability=("frozen_boundary_probability", "mean"),
    )
    summary.to_csv(OUT_DATA / "external_boundary_validation_controller_summary.csv", index=False)
    metadata = {
        "seed": SEED,
        "n_draws": N_DRAWS,
        "boundary": BOUNDARY,
        "note": "Exact frozen Step 5 boundary; no refitting, feature selection, or rescaling on this external atlas.",
    }
    (OUT_DATA / "external_boundary_validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figure(reduced, metrics, OUT_FIG / "figure_6_external_boundary_validation.png")

    fam = metrics[metrics["model_family"] != "all"]
    min_auc = float(fam["auc"].min())
    median_auc = float(fam["auc"].median())
    decision = (
        "CONTINUE: the exact four-term boundary externally validates above the 0.70 minimum-AUC gate."
        if min_auc >= 0.70
        else "PIVOT: the exact four-term boundary does not externally validate above the 0.70 minimum-AUC gate."
    )
    lines = [
        "# Step 6 External Validation of Frozen Four-Term Boundary",
        "",
        f"External seed: {SEED}.",
        f"Uncertainty draws per model family: {N_DRAWS}.",
        f"Total outcome rows: {len(reduced)}.",
        "No refitting, feature selection, or rescaling was performed.",
        "",
        "## Frozen Boundary",
        "",
        "```",
        "logit(P_success) = -0.129 + 0.735 z(observation_quality) + 0.643 z(log_reaction_number) + 0.612 z(duty_proxy*log_dose_capacity_number) + 0.702 z(log_thermal_diffusion_time_s)",
        "```",
        "",
        "## External Validation Metrics",
        "",
        "```",
        metrics.round(3).to_string(index=False),
        "```",
        "",
        "## Step-Gate Decision",
        "",
        f"External median family AUC: {median_auc:.3f}.",
        f"External minimum family AUC: {min_auc:.3f}.",
        decision,
    ]
    (OUT_NOTES / "step6_external_boundary_validation_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote external boundary validation outputs")
    print(metrics.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
