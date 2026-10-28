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
from run_step13_locked_large_validation_atlas import (  # noqa: E402
    LOCKED_BOUNDARY,
    N_DRAWS_PER_SEED,
    REFERENCE_SCORED,
    VALIDATION_SEEDS,
    apply_locked_boundary,
    locked_class_thresholds,
)


NCS = ROOT / "nature_computational_science"
STEP13_SCORED = NCS / "data" / "locked_large_validation" / "locked_large_validation_scored.csv"
STEP13_SUCCESS = NCS / "data" / "locked_large_validation" / "locked_large_validation_success_uplift.csv"
OUT_DATA = NCS / "data" / "thermal_floor_predictive_validation"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

PAIR_KEYS = [
    "validation_seed",
    "model_family",
    "atlas_uncertainty_id",
    "scenario",
    "delay_s",
    "noise_fraction",
]

THERMAL_FLOOR_CONTROLLER = "predictive_thermal_floor_full"


def common_rng(seed: int, draw: pd.Series, scenario_index: int, controller_index: int) -> np.random.Generator:
    value = seed + int(draw["atlas_uncertainty_id"]) * 1009 + scenario_index * 9176 + controller_index * 131
    return np.random.default_rng(value)


def thermal_floor_policy() -> dict:
    return [
        {**p, "controller": THERMAL_FLOOR_CONTROLLER}
        for p in predictive_policies()
        if p["controller"] == THERMAL_FLOOR_CONTROLLER
    ][0]


def run_thermal_floor_seed(seed: int) -> pd.DataFrame:
    draws = make_model_family_draws(sample_uncertainty(n=N_DRAWS_PER_SEED, seed=seed))
    scenarios = atlas_scenarios()
    policy = thermal_floor_policy()
    outcomes = []
    total = len(draws) * len(scenarios)
    processed = 0
    for _, draw in draws.iterrows():
        for scenario_index, scenario in enumerate(scenarios):
            result, _ = simulate_predictive_case(policy, scenario, draw, common_rng(seed, draw, scenario_index, 5))
            outcomes.append(add_draw_metadata(result, draw, scenario, "predictive_thermal_floor"))
            processed += 1
            if processed % 192 == 0:
                print(f"thermal-floor seed {seed}: processed {processed}/{total} draw-scenarios", flush=True)
    out = pd.DataFrame(outcomes)
    out["validation_seed"] = f"seed_{seed}"
    return out


def pair_controllers(scored: pd.DataFrame, from_controller: str, to_controller: str) -> pd.DataFrame:
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
    a = scored.loc[scored["controller"] == from_controller, keep].copy()
    b = scored.loc[scored["controller"] == to_controller, keep].copy()
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
        uplift_lo, uplift_hi = clustered_ci(g, "success_uplift")
        thermal_lo, thermal_hi = clustered_ci(g, "thermal_failure_rate_delta")
        ros_lo, ros_hi = clustered_ci(g, "ros_failure_rate_delta")
        rows.append(
            {
                "validation_seed": seed,
                "model_family": family,
                "n": len(g),
                "success_uplift": float(g["success_uplift"].mean()),
                "success_uplift_ci_low": uplift_lo,
                "success_uplift_ci_high": uplift_hi,
                "rescued_to_lost_ratio": float(g["rescued_success"].mean() / g["lost_success"].mean()) if g["lost_success"].mean() > 0 else float("inf"),
                "thermal_failure_delta": float(g["thermal_failure_rate_delta"].mean()),
                "thermal_failure_delta_ci_low": thermal_lo,
                "thermal_failure_delta_ci_high": thermal_hi,
                "ros_failure_delta": float(g["ros_failure_rate_delta"].mean()),
                "ros_failure_delta_ci_low": ros_lo,
                "ros_failure_delta_ci_high": ros_hi,
                "overheat_failure_delta": float(g["overheat_failure_rate_delta"].mean()),
                "delivered_energy_delta": float(g["delivered_energy_j_cm2_delta"].mean()),
                "warm_fraction_delta": float(g["warm_fraction_delta"].mean()),
                "ros_threshold_fraction_delta": float(g["ros_threshold_fraction_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_to_smoothed(thermal_success: pd.DataFrame) -> pd.DataFrame:
    smoothed = pd.read_csv(STEP13_SUCCESS)
    smoothed = smoothed[["stratum", "success_uplift", "success_uplift_ci_low", "success_uplift_ci_high", "rescued_to_lost_ratio"]].copy()
    smoothed["controller"] = "predictive_smoothed_full"
    thermal = thermal_success[["stratum", "success_uplift", "success_uplift_ci_low", "success_uplift_ci_high", "rescued_to_lost_ratio"]].copy()
    thermal["controller"] = THERMAL_FLOOR_CONTROLLER
    return pd.concat([smoothed, thermal], ignore_index=True)


def gate_decision(success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame, seed_family: pd.DataFrame) -> tuple[str, list[str]]:
    low = success[success["stratum"] == "from_low"].iloc[0]
    thermal = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "thermal_failure")].iloc[0]
    ros = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "ros_failure")].iloc[0]
    overheat = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "overheat_failure")].iloc[0]
    energy = deltas[(deltas["stratum"] == "from_low") & (deltas["mechanism_variable"] == "delivered_energy_j_cm2")].iloc[0]
    positive_seed_families = int((seed_family["success_uplift"] > 0).sum())
    total_seed_families = int(len(seed_family))
    checks = [
        f"Low-class success uplift = {low['success_uplift']:.3f} [{low['success_uplift_ci_low']:.3f}, {low['success_uplift_ci_high']:.3f}].",
        f"Low-class rescued/lost ratio = {low['rescued_to_lost_ratio']:.2f}.",
        f"Low-class thermal failure delta = {thermal['failure_rate_delta']:.3f} [{thermal['failure_rate_delta_ci_low']:.3f}, {thermal['failure_rate_delta_ci_high']:.3f}].",
        f"Low-class ROS failure delta = {ros['failure_rate_delta']:.3f} [{ros['failure_rate_delta_ci_low']:.3f}, {ros['failure_rate_delta_ci_high']:.3f}].",
        f"Low-class overheat failure delta = {overheat['failure_rate_delta']:.3f} [{overheat['failure_rate_delta_ci_low']:.3f}, {overheat['failure_rate_delta_ci_high']:.3f}].",
        f"Low-class delivered-energy delta = {energy['mean_delta']:.2f} [{energy['mean_delta_ci_low']:.2f}, {energy['mean_delta_ci_high']:.2f}] J cm^-2.",
        f"Positive seed-family low-class uplift strata = {positive_seed_families}/{total_seed_families}.",
    ]
    passes = (
        low["success_uplift"] >= 0.08
        and low["success_uplift_ci_low"] > 0
        and low["rescued_to_lost_ratio"] >= 2.0
        and thermal["failure_rate_delta"] <= 0.0
        and ros["failure_rate_delta"] <= 0.0
        and overheat["failure_rate_delta"] <= 0.02
        and energy["mean_delta_ci_high"] <= 60.0
        and positive_seed_families >= int(np.ceil(0.75 * total_seed_families))
    )
    if passes:
        decision = (
            "CONFIRMED: thermal-floor predictive control repairs the Step 13 thermal-underdose failure while "
            "preserving ROS/overheat protection strongly enough for the next high-impact manuscript gate."
        )
    else:
        decision = (
            "NOT CONFIRMED: the thermal-floor controller does not yet repair the Step 13 failure mode strongly enough. "
            "The next pivot should be formal MPC or direct policy optimization against the pre-declared endpoints."
        )
    return decision, checks


def make_figure(
    success: pd.DataFrame,
    failures: pd.DataFrame,
    deltas: pd.DataFrame,
    seed_family: pd.DataFrame,
    controller_comparison: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    plot = controller_comparison[
        controller_comparison["stratum"].isin(["from_low", "from_transitional", "from_high"])
    ].copy()
    xlabels = ["low", "trans.", "high"]
    x = np.arange(3)
    width = 0.36
    for offset, controller, color in [
        (-width / 2, "predictive_smoothed_full", "#9aa0a6"),
        (width / 2, THERMAL_FLOOR_CONTROLLER, "#347f68"),
    ]:
        g = plot[plot["controller"] == controller].set_index("stratum").loc[
            ["from_low", "from_transitional", "from_high"]
        ]
        ax.bar(x + offset, g["success_uplift"], width=width, color=color, label=controller.replace("predictive_", ""))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Paired success uplift")
    ax.set_title("A. Thermal-floor versus smoothed")
    ax.legend(frameon=False, fontsize=7)

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
    ax.set_ylabel("Thermal-floor - full (percentage points)")
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
    ax.scatter(seed_family["thermal_failure_delta"], seed_family["success_uplift"], s=45, color="#347f68", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Thermal failure delta")
    ax.set_ylabel("Low-class success uplift")
    ax.set_title("F. Thermal repair link")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)

    q33, q67 = locked_class_thresholds(REFERENCE_SCORED)
    baseline_scored = pd.read_csv(STEP13_SCORED)
    baseline_scored = baseline_scored[baseline_scored["controller"].isin(["full_feedback", "predictive_smoothed_full"])].copy()

    thermal_outcomes = pd.concat([run_thermal_floor_seed(seed) for seed in VALIDATION_SEEDS], ignore_index=True)
    thermal_outcomes.to_csv(OUT_DATA / "thermal_floor_predictive_outcomes.csv", index=False)
    thermal_scored = apply_locked_boundary(add_reduced_features(thermal_outcomes), q33, q67)
    thermal_scored.to_csv(OUT_DATA / "thermal_floor_predictive_scored.csv", index=False)

    combined = pd.concat([baseline_scored, thermal_scored], ignore_index=True, sort=False)
    combined.to_csv(OUT_DATA / "thermal_floor_combined_locked_scored.csv", index=False)

    paired = pair_controllers(combined, "full_feedback", THERMAL_FLOOR_CONTROLLER)
    success = success_summary(paired)
    failures = failure_summary(paired)
    deltas = mechanism_delta_summary(paired)
    profiles = rescue_profile(paired)
    seed_family = seed_family_endpoint_summary(paired)
    controller_comparison = compare_to_smoothed(success)
    decision, checks = gate_decision(success, failures, deltas, seed_family)

    paired.to_csv(OUT_DATA / "thermal_floor_paired_failure_modes.csv", index=False)
    success.to_csv(OUT_DATA / "thermal_floor_success_uplift.csv", index=False)
    failures.to_csv(OUT_DATA / "thermal_floor_failure_modes.csv", index=False)
    deltas.to_csv(OUT_DATA / "thermal_floor_mechanism_deltas.csv", index=False)
    profiles.to_csv(OUT_DATA / "thermal_floor_rescue_profiles.csv", index=False)
    seed_family.to_csv(OUT_DATA / "thermal_floor_seed_family_endpoints.csv", index=False)
    controller_comparison.to_csv(OUT_DATA / "thermal_floor_vs_smoothed_success_comparison.csv", index=False)

    make_figure(success, failures, deltas, seed_family, controller_comparison, OUT_FIG / "figure_14_thermal_floor_predictive_validation.png")

    metadata = {
        "validation_seeds": VALIDATION_SEEDS,
        "n_draws_per_seed": N_DRAWS_PER_SEED,
        "thermal_floor_policy": thermal_floor_policy(),
        "locked_boundary": LOCKED_BOUNDARY,
        "locked_class_thresholds_from_step8_reference": {"q33": q33, "q67": q67},
        "predeclared_endpoint_inherited_from_step13": [
            "low-class success uplift",
            "low-class rescued/lost ratio",
            "low-class thermal failure delta",
            "low-class ROS failure-rate reduction",
            "low-class overheat failure-rate change",
            "low-class delivered-energy delta",
            "seed-family consistency",
        ],
        "note": "Controller form is a mechanistic repair of the Step 13 thermal-underdose failure. No retuning, feature selection, or threshold fitting is performed on the validation atlas.",
    }
    (OUT_DATA / "thermal_floor_predictive_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    note = [
        "# Step 14 Thermal-Floor Predictive Validation",
        "",
        "Question: can a mechanistic thermal-floor predictive controller preserve the ROS/overheat protection observed in Step 13 while eliminating induced thermal underdose?",
        "",
        "## Locked Protocol",
        "",
        f"Validation seeds: {VALIDATION_SEEDS}.",
        f"Uncertainty draws per seed before family transforms: {N_DRAWS_PER_SEED}.",
        f"Paired full-feedback versus thermal-floor predictive rows: {len(paired)}.",
        "The full-feedback baseline and class thresholds are inherited from Step 13. The controller is not tuned on this validation output.",
        "",
        "## Thermal-Floor Policy",
        "",
        "```json",
        json.dumps(thermal_floor_policy(), indent=2),
        "```",
        "",
        "## Success-Uplift Summary",
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
    (OUT_NOTES / "step14_thermal_floor_predictive_validation_result.md").write_text("\n".join(note) + "\n", encoding="utf-8")

    print("Wrote thermal-floor predictive validation outputs")
    print(success.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
