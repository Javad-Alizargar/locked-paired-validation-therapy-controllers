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
from run_predictive_delay_control import predictive_policies, simulate_predictive_case  # noqa: E402
from run_robust_chance_optimization import sample_uncertainty  # noqa: E402
from run_step2_information_atlas import add_draw_metadata, add_reduced_features, atlas_scenarios, make_model_family_draws  # noqa: E402
from run_step12_low_class_failure_mode_decomposition import (  # noqa: E402
    CONTINUOUS_MECHANISMS,
    MECHANISM_FLAGS,
    clustered_ci,
    delay_bin,
    failure_summary,
    mechanism_delta_summary,
    rescue_profile,
    success_summary,
)


NCS = ROOT / "nature_computational_science"
REFERENCE_SCORED = NCS / "data" / "predictive_control_reclassification" / "multiseed_outofseed_scored_controllers.csv"
OUT_DATA = NCS / "data" / "locked_large_validation"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

VALIDATION_SEEDS = [20260720, 20260721, 20260722]
N_DRAWS_PER_SEED = 32
EPS = 1e-12

LOCKED_BOUNDARY = {
    "intercept": -0.016,
    "terms": [
        {
            "feature": "log_dose_capacity_number",
            "coef": 1.406,
            "mean": -0.625,
            "sd": 0.549,
        },
        {
            "feature": "log_reaction_number*observation_quality",
            "coef": 0.930,
            "mean": 0.319,
            "sd": 0.307,
        },
        {
            "feature": "log_reaction_drive_balance",
            "coef": -1.482,
            "mean": -3.530,
            "sd": 0.576,
        },
    ],
    "source": "Step 7 interpretation-only combined equation; used here as a locked scoring rule.",
}

PAIR_KEYS = [
    "validation_seed",
    "model_family",
    "atlas_uncertainty_id",
    "scenario",
    "delay_s",
    "noise_fraction",
]


def common_rng(seed: int, draw: pd.Series, scenario_index: int, controller_index: int) -> np.random.Generator:
    value = seed + int(draw["atlas_uncertainty_id"]) * 1009 + scenario_index * 9176 + controller_index * 131
    return np.random.default_rng(value)


def locked_class_thresholds(reference_path: Path) -> tuple[float, float]:
    reference = pd.read_csv(reference_path, usecols=["predicted_probability"])
    q33, q67 = reference["predicted_probability"].quantile([0.33, 0.67]).to_numpy(float)
    return float(q33), float(q67)


def apply_locked_boundary(df: pd.DataFrame, q33: float, q67: float) -> pd.DataFrame:
    out = df.copy()
    if "log_reaction_number*observation_quality" not in out.columns:
        out["log_reaction_number*observation_quality"] = out["log_reaction_number"] * out["observation_quality"]
    score = np.full(len(out), LOCKED_BOUNDARY["intercept"], dtype=float)
    for term in LOCKED_BOUNDARY["terms"]:
        z = (out[term["feature"]].astype(float) - term["mean"]) / (term["sd"] + EPS)
        score += term["coef"] * z
    out["locked_boundary_score"] = score
    out["predicted_probability"] = 1 / (1 + np.exp(-np.clip(score, -40, 40)))
    out["controllability_class"] = pd.cut(
        out["predicted_probability"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["low", "transitional", "high"],
    )
    out["class_index"] = out["controllability_class"].map({"low": 0, "transitional": 1, "high": 2}).astype(int)
    out["chance_success_int"] = out["chance_success"].astype(int)
    return out


def run_seed(seed: int) -> pd.DataFrame:
    draws = make_model_family_draws(sample_uncertainty(n=N_DRAWS_PER_SEED, seed=seed))
    scenarios = atlas_scenarios()
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

    outcomes: list[dict] = []
    total = len(draws) * len(scenarios)
    processed = 0
    for _, draw in draws.iterrows():
        for scenario_index, scenario in enumerate(scenarios):
            result, _ = simulate_ablation_case(full_feedback, scenario, draw, common_rng(seed, draw, scenario_index, 1))
            outcomes.append(add_draw_metadata(result, draw, scenario, "feedback"))

            result, _ = simulate_predictive_case(predictive, scenario, draw, common_rng(seed, draw, scenario_index, 2))
            outcomes.append(add_draw_metadata(result, draw, scenario, "predictive"))

            processed += 1
            if processed % 192 == 0:
                print(f"seed {seed}: processed {processed}/{total} draw-scenarios", flush=True)
    out = pd.DataFrame(outcomes)
    out["validation_seed"] = f"seed_{seed}"
    return out


def paired_table(scored: pd.DataFrame) -> pd.DataFrame:
    keep = (
        PAIR_KEYS
        + [
            "controller",
            "chance_success_int",
            "predicted_probability",
            "controllability_class",
            "class_index",
        ]
        + MECHANISM_FLAGS
        + CONTINUOUS_MECHANISMS
    )
    a = scored.loc[scored["controller"] == "full_feedback", keep].copy()
    b = scored.loc[scored["controller"] == "predictive_smoothed_full", keep].copy()
    a = a.rename(columns={col: f"from_{col}" for col in keep if col not in PAIR_KEYS})
    b = b.rename(columns={col: f"to_{col}" for col in keep if col not in PAIR_KEYS})
    paired = a.merge(b, on=PAIR_KEYS, how="inner", validate="one_to_one")
    paired["success_uplift"] = paired["to_chance_success_int"] - paired["from_chance_success_int"]
    paired["rescued_success"] = (paired["from_chance_success_int"] == 0) & (paired["to_chance_success_int"] == 1)
    paired["lost_success"] = (paired["from_chance_success_int"] == 1) & (paired["to_chance_success_int"] == 0)
    paired["class_shift"] = paired["to_class_index"] - paired["from_class_index"]
    paired["probability_shift"] = paired["to_predicted_probability"] - paired["from_predicted_probability"]
    paired["delay_bin"] = paired["delay_s"].map(delay_bin)
    paired["cluster_id"] = (
        paired["validation_seed"].astype(str)
        + "|"
        + paired["model_family"].astype(str)
        + "|"
        + paired["atlas_uncertainty_id"].astype(str)
    )
    for flag in MECHANISM_FLAGS:
        paired[f"{flag}_resolved"] = paired[f"from_{flag}"].astype(bool) & ~paired[f"to_{flag}"].astype(bool)
        paired[f"{flag}_induced"] = ~paired[f"from_{flag}"].astype(bool) & paired[f"to_{flag}"].astype(bool)
        paired[f"{flag}_net_resolution"] = paired[f"{flag}_resolved"].astype(int) - paired[f"{flag}_induced"].astype(int)
        paired[f"{flag}_rate_delta"] = paired[f"to_{flag}"].astype(int) - paired[f"from_{flag}"].astype(int)
    for col in CONTINUOUS_MECHANISMS:
        paired[f"{col}_delta"] = paired[f"to_{col}"] - paired[f"from_{col}"]
    return paired


def seed_family_endpoint_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    low = paired[paired["from_controllability_class"].astype(str) == "low"].copy()
    for (seed, family), g in low.groupby(["validation_seed", "model_family"]):
        ros_col = "ros_failure_rate_delta"
        energy_col = "delivered_energy_j_cm2_delta"
        ros_lo, ros_hi = clustered_ci(g, ros_col)
        uplift_lo, uplift_hi = clustered_ci(g, "success_uplift")
        rows.append(
            {
                "validation_seed": seed,
                "model_family": family,
                "n": len(g),
                "from_success_rate": float(g["from_chance_success_int"].mean()),
                "to_success_rate": float(g["to_chance_success_int"].mean()),
                "success_uplift": float(g["success_uplift"].mean()),
                "success_uplift_ci_low": uplift_lo,
                "success_uplift_ci_high": uplift_hi,
                "rescued_success_rate": float(g["rescued_success"].mean()),
                "lost_success_rate": float(g["lost_success"].mean()),
                "rescued_to_lost_ratio": float(g["rescued_success"].mean() / g["lost_success"].mean()) if g["lost_success"].mean() > 0 else float("inf"),
                "ros_failure_delta": float(g[ros_col].mean()),
                "ros_failure_delta_ci_low": ros_lo,
                "ros_failure_delta_ci_high": ros_hi,
                "delivered_energy_delta": float(g[energy_col].mean()),
                "warm_fraction_delta": float(g["warm_fraction_delta"].mean()),
                "ros_threshold_fraction_delta": float(g["ros_threshold_fraction_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def gate_decision(success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame, seed_family: pd.DataFrame) -> tuple[str, list[str]]:
    low = success[success["stratum"] == "from_low"].iloc[0]
    ros = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "ros_failure")].iloc[0]
    energy = deltas[(deltas["stratum"] == "from_low") & (deltas["mechanism_variable"] == "delivered_energy_j_cm2")].iloc[0]
    warm = deltas[(deltas["stratum"] == "from_low") & (deltas["mechanism_variable"] == "warm_fraction")].iloc[0]
    ros_frac = deltas[(deltas["stratum"] == "from_low") & (deltas["mechanism_variable"] == "ros_threshold_fraction")].iloc[0]
    positive_seed_families = int((seed_family["success_uplift"] > 0).sum())
    total_seed_families = int(len(seed_family))
    checks = [
        f"Low-class success uplift = {low['success_uplift']:.3f} [{low['success_uplift_ci_low']:.3f}, {low['success_uplift_ci_high']:.3f}].",
        f"Low-class rescued/lost ratio = {low['rescued_to_lost_ratio']:.2f}.",
        f"Low-class ROS failure delta = {ros['failure_rate_delta']:.3f} [{ros['failure_rate_delta_ci_low']:.3f}, {ros['failure_rate_delta_ci_high']:.3f}].",
        f"Low-class warm-fraction delta = {warm['mean_delta']:.3f} [{warm['mean_delta_ci_low']:.3f}, {warm['mean_delta_ci_high']:.3f}].",
        f"Low-class ROS-threshold-fraction delta = {ros_frac['mean_delta']:.3f} [{ros_frac['mean_delta_ci_low']:.3f}, {ros_frac['mean_delta_ci_high']:.3f}].",
        f"Low-class delivered-energy delta = {energy['mean_delta']:.2f} [{energy['mean_delta_ci_low']:.2f}, {energy['mean_delta_ci_high']:.2f}] J cm^-2.",
        f"Positive seed-family low-class uplift strata = {positive_seed_families}/{total_seed_families}.",
    ]
    passes = (
        low["success_uplift"] >= 0.08
        and low["success_uplift_ci_low"] > 0
        and low["rescued_to_lost_ratio"] >= 2.0
        and ros["failure_rate_delta"] < 0
        and ros["failure_rate_delta_ci_high"] < 0
        and warm["mean_delta"] > 0
        and ros_frac["mean_delta"] > 0
        and energy["mean_delta_ci_high"] <= 30.0
        and positive_seed_families >= int(np.ceil(0.75 * total_seed_families))
    )
    if passes:
        decision = (
            "CONFIRMED: the larger locked atlas repeats the low-controllability predictive-rescue mechanism. "
            "This materially strengthens the Nature Computational Science case."
        )
    else:
        decision = (
            "NOT CONFIRMED: the larger locked atlas does not fully repeat the pre-declared rescue mechanism. "
            "The project should pivot or add a stronger controller/model before targeting Nature Computational Science."
        )
    return decision, checks


def make_figure(success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame, seed_family: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    plot = success[success["stratum"].isin(["from_low", "from_transitional", "from_high"])].copy()
    labels = plot["stratum"].str.replace("from_", "", regex=False)
    y = plot["success_uplift"]
    yerr = np.vstack([y - plot["success_uplift_ci_low"], plot["success_uplift_ci_high"] - y])
    ax.bar(labels, y, color=["#b45f4d", "#d59b43", "#347f68"])
    ax.errorbar(labels, y, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Paired success uplift")
    ax.set_title("A. Locked validation by class")

    ax = axes[0, 1]
    low_fail = failures[failures["stratum"] == "from_low"].copy()
    labels = low_fail["failure_mode"].str.replace("_failure", "", regex=False).str.upper()
    y = -low_fail["failure_rate_delta"]
    yerr = np.vstack([y - (-low_fail["failure_rate_delta_ci_high"]), (-low_fail["failure_rate_delta_ci_low"]) - y])
    ax.bar(labels, y, color="#4b7fb8")
    ax.errorbar(labels, y, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Failure-rate reduction")
    ax.set_title("B. Low-class failure mechanism")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 2]
    low_delay = success[success["stratum"].str.startswith("low_delay_")].copy()
    low_delay["delay_label"] = low_delay["stratum"].str.replace("low_delay_", "", regex=False)
    order = ["0 s", "1-30 s", "31-60 s", ">60 s"]
    low_delay["delay_label"] = pd.Categorical(low_delay["delay_label"], categories=order, ordered=True)
    low_delay = low_delay.sort_values("delay_label")
    y = low_delay["success_uplift"]
    yerr = np.vstack([y - low_delay["success_uplift_ci_low"], low_delay["success_uplift_ci_high"] - y])
    ax.plot(low_delay["delay_label"].astype(str), y, marker="o", color="#6d4c91")
    ax.errorbar(low_delay["delay_label"].astype(str), y, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Low-class success uplift")
    ax.set_title("C. Delay dependence")

    ax = axes[1, 0]
    low_delta = deltas[deltas["stratum"] == "from_low"].set_index("mechanism_variable")
    show = ["warm_fraction", "ros_threshold_fraction", "hypoxic_failure_fraction", "overheat_fraction"]
    labels = ["warm\nfraction", "ROS\nfraction", "hypoxic\nfraction", "overheat\nfraction"]
    y = 100 * low_delta.loc[show, "mean_delta"]
    yerr = np.vstack(
        [
            y - 100 * low_delta.loc[show, "mean_delta_ci_low"],
            100 * low_delta.loc[show, "mean_delta_ci_high"] - y,
        ]
    )
    ax.bar(labels, y, color=["#347f68", "#347f68", "#b45f4d", "#b45f4d"])
    ax.errorbar(labels, y, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Predictive - full (percentage points)")
    ax.set_title("D. State shifts")

    ax = axes[1, 1]
    plot = seed_family.sort_values("success_uplift")
    labels = plot["validation_seed"].str.replace("seed_", "", regex=False) + "\n" + plot["model_family"]
    colors = np.where(plot["success_uplift"] > 0, "#4b7fb8", "#b45f4d")
    ax.barh(np.arange(len(plot)), plot["success_uplift"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(plot)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Low-class success uplift")
    ax.set_title("E. Seed-family consistency")

    ax = axes[1, 2]
    ax.scatter(seed_family["ros_failure_delta"], seed_family["success_uplift"], s=45, color="#347f68", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ROS failure delta")
    ax.set_ylabel("Low-class success uplift")
    ax.set_title("F. Mechanism-outcome link")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)

    q33, q67 = locked_class_thresholds(REFERENCE_SCORED)
    outcomes = pd.concat([run_seed(seed) for seed in VALIDATION_SEEDS], ignore_index=True)
    outcomes.to_csv(OUT_DATA / "locked_large_validation_outcomes.csv", index=False)
    reduced = add_reduced_features(outcomes)
    scored = apply_locked_boundary(reduced, q33, q67)
    scored.to_csv(OUT_DATA / "locked_large_validation_scored.csv", index=False)

    paired = paired_table(scored)
    success = success_summary(paired)
    failures = failure_summary(paired)
    deltas = mechanism_delta_summary(paired)
    profiles = rescue_profile(paired)
    seed_family = seed_family_endpoint_summary(paired)

    paired.to_csv(OUT_DATA / "locked_large_validation_paired_failure_modes.csv", index=False)
    success.to_csv(OUT_DATA / "locked_large_validation_success_uplift.csv", index=False)
    failures.to_csv(OUT_DATA / "locked_large_validation_failure_modes.csv", index=False)
    deltas.to_csv(OUT_DATA / "locked_large_validation_mechanism_deltas.csv", index=False)
    profiles.to_csv(OUT_DATA / "locked_large_validation_rescue_profiles.csv", index=False)
    seed_family.to_csv(OUT_DATA / "locked_large_validation_seed_family_endpoints.csv", index=False)

    decision, checks = gate_decision(success, failures, deltas, seed_family)
    make_figure(success, failures, deltas, seed_family, OUT_FIG / "figure_13_locked_large_validation.png")

    metadata = {
        "validation_seeds": VALIDATION_SEEDS,
        "n_draws_per_seed": N_DRAWS_PER_SEED,
        "n_outcome_rows": int(len(scored)),
        "n_paired_rows": int(len(paired)),
        "controllers": ["full_feedback", "predictive_smoothed_full"],
        "locked_boundary": LOCKED_BOUNDARY,
        "locked_class_thresholds_from_step8_reference": {"q33": q33, "q67": q67},
        "predeclared_endpoints": [
            "low-class success uplift",
            "low-class rescued/lost ratio",
            "low-class ROS failure-rate reduction",
            "low-class warm-fraction shift",
            "low-class ROS-threshold-fraction shift",
            "low-class delivered-energy delta",
            "seed-family consistency",
        ],
        "note": "No retuning, feature selection, policy optimization, or threshold fitting is performed on this locked validation atlas.",
    }
    (OUT_DATA / "locked_large_validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    note = [
        "# Step 13 Locked Larger Validation Atlas",
        "",
        "Question: does the Step 12 low-controllability predictive-rescue mechanism repeat in a larger independent atlas with no retuning?",
        "",
        "## Locked Protocol",
        "",
        f"Validation seeds: {VALIDATION_SEEDS}.",
        f"Uncertainty draws per seed before family transforms: {N_DRAWS_PER_SEED}.",
        f"Outcome rows: {len(scored)}.",
        f"Paired full-feedback versus predictive-feedback rows: {len(paired)}.",
        "Controllability classes use the locked Step 7 three-term score and Step 8 reference quantile thresholds. No new model is fitted to this validation atlas.",
        "",
        "## Pre-Declared Endpoint Summary",
        "",
        "```",
        success.round(3).to_string(index=False),
        "```",
        "",
        "## Low-Class Failure Modes",
        "",
        "```",
        failures[failures["stratum"] == "from_low"].round(3).to_string(index=False),
        "```",
        "",
        "## Low-Class Mechanistic State Shifts",
        "",
        "```",
        deltas[deltas["stratum"] == "from_low"].round(3).to_string(index=False),
        "```",
        "",
        "## Seed-Family Endpoint Check",
        "",
        "```",
        seed_family.round(3).to_string(index=False),
        "```",
        "",
        "## Strict Gate Checks",
        "",
        *[f"- {line}" for line in checks],
        "",
        "## Decision",
        "",
        decision,
    ]
    (OUT_NOTES / "step13_locked_large_validation_result.md").write_text("\n".join(note) + "\n", encoding="utf-8")

    print("Wrote locked larger validation atlas outputs")
    print(success.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
