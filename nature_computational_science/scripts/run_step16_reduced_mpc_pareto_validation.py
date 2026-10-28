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
    success_summary,
)
from run_step13_locked_large_validation_atlas import (  # noqa: E402
    LOCKED_BOUNDARY,
    REFERENCE_SCORED,
    VALIDATION_SEEDS,
    apply_locked_boundary,
    locked_class_thresholds,
)


NCS = ROOT / "nature_computational_science"
STEP13_SCORED = NCS / "data" / "locked_large_validation" / "locked_large_validation_scored.csv"
OUT_DATA = NCS / "data" / "reduced_mpc_pareto_validation"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

MPC_CONTROLLER = "predictive_reduced_mpc_full"
PAIR_KEYS = ["validation_seed", "model_family", "atlas_uncertainty_id", "scenario", "delay_s", "noise_fraction"]


def common_rng(seed: int, draw: pd.Series, scenario_index: int, controller_index: int) -> np.random.Generator:
    value = seed + int(draw["atlas_uncertainty_id"]) * 1009 + scenario_index * 9176 + controller_index * 131
    return np.random.default_rng(value)


def mpc_policy() -> dict:
    return [p for p in predictive_policies() if p["controller"] == MPC_CONTROLLER][0]


def run_mpc_seed(seed: int) -> pd.DataFrame:
    draws = make_model_family_draws(sample_uncertainty(n=32, seed=seed))
    scenarios = atlas_scenarios()
    policy = mpc_policy()
    outcomes = []
    total = len(draws) * len(scenarios)
    processed = 0
    for _, draw in draws.iterrows():
        for scenario_index, scenario in enumerate(scenarios):
            result, _ = simulate_predictive_case(policy, scenario, draw, common_rng(seed, draw, scenario_index, 6))
            row = add_draw_metadata(result, draw, scenario, "reduced_mpc")
            row["validation_seed"] = f"seed_{seed}"
            outcomes.append(row)
            processed += 1
            if processed % 192 == 0:
                print(f"reduced-MPC seed {seed}: {processed}/{total}", flush=True)
    return pd.DataFrame(outcomes)


def pair_controllers(scored: pd.DataFrame) -> pd.DataFrame:
    keep = PAIR_KEYS + ["controller", "chance_success_int", "predicted_probability", "controllability_class", "class_index"] + MECHANISM_FLAGS + CONTINUOUS_MECHANISMS
    a = scored.loc[scored["controller"] == "full_feedback", keep].copy()
    b = scored.loc[scored["controller"] == MPC_CONTROLLER, keep].copy()
    a = a.rename(columns={col: f"from_{col}" for col in keep if col not in PAIR_KEYS})
    b = b.rename(columns={col: f"to_{col}" for col in keep if col not in PAIR_KEYS})
    paired = a.merge(b, on=PAIR_KEYS, how="inner", validate="one_to_one")
    paired["success_uplift"] = paired["to_chance_success_int"] - paired["from_chance_success_int"]
    paired["rescued_success"] = (paired["from_chance_success_int"] == 0) & (paired["to_chance_success_int"] == 1)
    paired["lost_success"] = (paired["from_chance_success_int"] == 1) & (paired["to_chance_success_int"] == 0)
    paired["class_shift"] = paired["to_class_index"] - paired["from_class_index"]
    paired["probability_shift"] = paired["to_predicted_probability"] - paired["from_predicted_probability"]
    paired["delay_bin"] = paired["delay_s"].map(delay_bin)
    paired["cluster_id"] = paired["validation_seed"].astype(str) + "|" + paired["model_family"].astype(str) + "|" + paired["atlas_uncertainty_id"].astype(str)
    for flag in MECHANISM_FLAGS:
        paired[f"{flag}_resolved"] = paired[f"from_{flag}"].astype(bool) & ~paired[f"to_{flag}"].astype(bool)
        paired[f"{flag}_induced"] = ~paired[f"from_{flag}"].astype(bool) & paired[f"to_{flag}"].astype(bool)
        paired[f"{flag}_net_resolution"] = paired[f"{flag}_resolved"].astype(int) - paired[f"{flag}_induced"].astype(int)
        paired[f"{flag}_rate_delta"] = paired[f"to_{flag}"].astype(int) - paired[f"from_{flag}"].astype(int)
    for col in CONTINUOUS_MECHANISMS:
        paired[f"{col}_delta"] = paired[f"to_{col}"] - paired[f"from_{col}"]
    return paired


def seed_family_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    low = paired[paired["from_controllability_class"].astype(str) == "low"].copy()
    for (seed, family), g in low.groupby(["validation_seed", "model_family"]):
        lo, hi = clustered_ci(g, "success_uplift")
        rows.append(
            {
                "validation_seed": seed,
                "model_family": family,
                "n": len(g),
                "success_uplift": float(g["success_uplift"].mean()),
                "success_uplift_ci_low": lo,
                "success_uplift_ci_high": hi,
                "rescued_to_lost_ratio": float(g["rescued_success"].mean() / g["lost_success"].mean()) if g["lost_success"].mean() > 0 else float("inf"),
                "thermal_failure_delta": float(g["thermal_failure_rate_delta"].mean()),
                "ros_failure_delta": float(g["ros_failure_rate_delta"].mean()),
                "overheat_failure_delta": float(g["overheat_failure_rate_delta"].mean()),
                "energy_delta": float(g["delivered_energy_j_cm2_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def gate_decision(success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame, seed_family: pd.DataFrame) -> tuple[str, list[str]]:
    low = success[success["stratum"] == "from_low"].iloc[0]
    high = success[success["stratum"] == "from_high"].iloc[0]
    thermal = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "thermal_failure")].iloc[0]
    ros = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "ros_failure")].iloc[0]
    overheat = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "overheat_failure")].iloc[0]
    energy = deltas[(deltas["stratum"] == "from_low") & (deltas["mechanism_variable"] == "delivered_energy_j_cm2")].iloc[0]
    positive = int((seed_family["success_uplift"] > 0).sum())
    checks = [
        f"Low uplift = {low['success_uplift']:.3f} [{low['success_uplift_ci_low']:.3f}, {low['success_uplift_ci_high']:.3f}].",
        f"High uplift = {high['success_uplift']:.3f} [{high['success_uplift_ci_low']:.3f}, {high['success_uplift_ci_high']:.3f}].",
        f"Thermal failure delta = {thermal['failure_rate_delta']:.3f} [{thermal['failure_rate_delta_ci_low']:.3f}, {thermal['failure_rate_delta_ci_high']:.3f}].",
        f"ROS failure delta = {ros['failure_rate_delta']:.3f} [{ros['failure_rate_delta_ci_low']:.3f}, {ros['failure_rate_delta_ci_high']:.3f}].",
        f"Overheat failure delta = {overheat['failure_rate_delta']:.3f} [{overheat['failure_rate_delta_ci_low']:.3f}, {overheat['failure_rate_delta_ci_high']:.3f}].",
        f"Energy delta = {energy['mean_delta']:.2f} [{energy['mean_delta_ci_low']:.2f}, {energy['mean_delta_ci_high']:.2f}] J cm^-2.",
        f"Positive seed-family low-class strata = {positive}/{len(seed_family)}.",
    ]
    passes = (
        low["success_uplift"] >= 0.08
        and low["success_uplift_ci_low"] > 0
        and high["success_uplift"] > -0.04
        and thermal["failure_rate_delta"] <= 0
        and ros["failure_rate_delta"] <= 0
        and overheat["failure_rate_delta"] <= 0.03
        and energy["mean_delta_ci_high"] <= 50
        and positive >= 9
    )
    decision = (
        "CONFIRMED: reduced-order MPC passes the locked tradeoff gate."
        if passes
        else "NOT CONFIRMED: reduced-order MPC does not pass the locked tradeoff gate; this supports a structural Pareto limitation unless full MPC changes the frontier."
    )
    return decision, checks


def make_figure(success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame, seed_family: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.8), constrained_layout=True)
    ax = axes[0, 0]
    plot = success[success["stratum"].isin(["from_low", "from_transitional", "from_high"])]
    ax.bar(plot["stratum"].str.replace("from_", "", regex=False), plot["success_uplift"], color=["#b45f4d", "#d59b43", "#347f68"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Success uplift")
    ax.set_title("A. Reduced-MPC validation")

    ax = axes[0, 1]
    low_fail = failures[failures["stratum"] == "from_low"]
    labels = low_fail["failure_mode"].str.replace("_failure", "", regex=False).str.upper()
    ax.bar(labels, -low_fail["failure_rate_delta"], color="#4b7fb8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Failure-rate reduction")
    ax.set_title("B. Low-class failure modes")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 2]
    low_delay = success[success["stratum"].str.startswith("low_delay_")].copy()
    low_delay["delay_label"] = low_delay["stratum"].str.replace("low_delay_", "", regex=False)
    order = ["0 s", "1-30 s", "31-60 s", ">60 s"]
    low_delay["delay_label"] = pd.Categorical(low_delay["delay_label"], categories=order, ordered=True)
    low_delay = low_delay.sort_values("delay_label")
    ax.plot(low_delay["delay_label"].astype(str), low_delay["success_uplift"], marker="o", color="#6d4c91")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Low-class uplift")
    ax.set_title("C. Delay dependence")

    ax = axes[1, 0]
    low_delta = deltas[deltas["stratum"] == "from_low"].set_index("mechanism_variable")
    show = ["warm_fraction", "ros_threshold_fraction", "overheat_fraction", "delivered_energy_j_cm2"]
    ax.bar(["warm", "ROS", "overheat", "energy"], low_delta.loc[show, "mean_delta"], color=["#347f68", "#347f68", "#b45f4d", "#777777"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("D. State shifts")

    ax = axes[1, 1]
    sf = seed_family.sort_values("success_uplift")
    labels = sf["validation_seed"].str.replace("seed_", "", regex=False) + "\n" + sf["model_family"]
    ax.barh(np.arange(len(sf)), sf["success_uplift"], color=np.where(sf["success_uplift"] > 0, "#4b7fb8", "#b45f4d"))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(sf)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Low-class uplift")
    ax.set_title("E. Seed-family consistency")

    ax = axes[1, 2]
    ax.scatter(seed_family["overheat_failure_delta"], seed_family["success_uplift"], color="#347f68", s=45)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Overheat delta")
    ax.set_ylabel("Low-class uplift")
    ax.set_title("F. Pareto tension")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    q33, q67 = locked_class_thresholds(REFERENCE_SCORED)
    baseline = pd.read_csv(STEP13_SCORED)
    baseline = baseline[baseline["controller"] == "full_feedback"].copy()
    outcomes = pd.concat([run_mpc_seed(seed) for seed in VALIDATION_SEEDS], ignore_index=True)
    outcomes.to_csv(OUT_DATA / "reduced_mpc_outcomes.csv", index=False)
    scored = apply_locked_boundary(add_reduced_features(outcomes), q33, q67)
    scored.to_csv(OUT_DATA / "reduced_mpc_scored.csv", index=False)
    combined = pd.concat([baseline, scored], ignore_index=True, sort=False)
    paired = pair_controllers(combined)
    success = success_summary(paired)
    failures = failure_summary(paired)
    deltas = mechanism_delta_summary(paired)
    seed_family = seed_family_summary(paired)
    decision, checks = gate_decision(success, failures, deltas, seed_family)

    paired.to_csv(OUT_DATA / "reduced_mpc_paired_failure_modes.csv", index=False)
    success.to_csv(OUT_DATA / "reduced_mpc_success_uplift.csv", index=False)
    failures.to_csv(OUT_DATA / "reduced_mpc_failure_modes.csv", index=False)
    deltas.to_csv(OUT_DATA / "reduced_mpc_mechanism_deltas.csv", index=False)
    seed_family.to_csv(OUT_DATA / "reduced_mpc_seed_family_endpoints.csv", index=False)
    (OUT_DATA / "reduced_mpc_metadata.json").write_text(
        json.dumps({"policy": mpc_policy(), "locked_boundary": LOCKED_BOUNDARY, "decision": decision}, indent=2),
        encoding="utf-8",
    )
    make_figure(success, failures, deltas, seed_family, OUT_FIG / "figure_16_reduced_mpc_pareto_validation.png")

    note = [
        "# Step 16 Reduced-Order MPC Pareto Validation",
        "",
        "Question: does a receding-horizon reduced-order MPC benchmark move the system beyond the heuristic policy tradeoff?",
        "",
        "## Policy",
        "",
        "```json",
        json.dumps(mpc_policy(), indent=2),
        "```",
        "",
        "## Locked Validation Summary",
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
        "## Low-Class State Shifts",
        "",
        "```",
        deltas[deltas["stratum"] == "from_low"].round(3).to_string(index=False),
        "```",
        "",
        "## Gate Checks",
        "",
        *[f"- {line}" for line in checks],
        "",
        "## Decision",
        "",
        decision,
    ]
    (OUT_NOTES / "step16_reduced_mpc_pareto_validation_result.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print("Wrote reduced-MPC validation outputs")
    print(success.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
