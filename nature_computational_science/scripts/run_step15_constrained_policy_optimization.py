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
from run_predictive_delay_control import simulate_predictive_case  # noqa: E402
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
STEP13_SCORED_PATH = NCS / "data" / "locked_large_validation" / "locked_large_validation_scored.csv"
OUT_DATA = NCS / "data" / "constrained_policy_optimization"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

TRAIN_SEEDS = [20260730, 20260731]
N_TRAIN_DRAWS = 8
N_CANDIDATES = 20
PAIR_KEYS = ["validation_seed", "model_family", "atlas_uncertainty_id", "scenario", "delay_s", "noise_fraction"]
OPT_CONTROLLER = "predictive_thermal_floor_optimized"


def common_rng(seed: int, draw: pd.Series, scenario_index: int, controller_index: int, candidate_index: int = 0) -> np.random.Generator:
    value = seed + int(draw["atlas_uncertainty_id"]) * 1009 + scenario_index * 9176 + controller_index * 131 + candidate_index * 65537
    return np.random.default_rng(value)


def candidate_policies(n: int = N_CANDIDATES, seed: int = 20260715) -> list[dict]:
    rng = np.random.default_rng(seed)
    candidates = [
        {
            "candidate_id": "baseline_thermal_floor",
            "controller": OPT_CONTROLLER,
            "signals": ("temperature", "oxygen", "ros"),
            "max_intensity_w_cm2": 1.30,
            "prediction_gain": 0.85,
            "lookahead_s": 0.0,
            "thermal_margin_c": 0.35,
            "estimator_window": 6,
            "thermal_floor_warm_fraction": 0.48,
            "thermal_floor_temp_c": 48.6,
            "thermal_floor_fraction": 0.58,
            "maintenance_intensity_w_cm2": 0.22,
            "warm_target": 0.56,
            "long_delay_safety_c": 51.0,
            "mid_delay_safety_c": 51.5,
            "long_delay_ros_stop": 0.78,
            "mid_delay_ros_stop": 0.80,
            "long_delay_maintenance_i": 0.18,
            "mid_delay_maintenance_i": 0.20,
            "oxygen_limited_floor_scale": 0.55,
            "ros_stop_after_floor": 0.82,
        }
    ]
    for i in range(n - 1):
        candidates.append(
            {
                "candidate_id": f"random_{i:02d}",
                "controller": OPT_CONTROLLER,
                "signals": ("temperature", "oxygen", "ros"),
                "max_intensity_w_cm2": float(rng.uniform(1.12, 1.30)),
                "prediction_gain": float(rng.uniform(0.65, 0.95)),
                "lookahead_s": 0.0,
                "thermal_margin_c": float(rng.uniform(0.20, 0.55)),
                "estimator_window": 6,
                "thermal_floor_warm_fraction": float(rng.uniform(0.40, 0.54)),
                "thermal_floor_temp_c": float(rng.uniform(47.2, 49.2)),
                "thermal_floor_fraction": float(rng.uniform(0.36, 0.62)),
                "maintenance_intensity_w_cm2": float(rng.uniform(0.16, 0.24)),
                "warm_target": float(rng.uniform(0.50, 0.58)),
                "long_delay_safety_c": float(rng.uniform(50.2, 51.2)),
                "mid_delay_safety_c": float(rng.uniform(50.7, 51.8)),
                "long_delay_ros_stop": float(rng.uniform(0.74, 0.86)),
                "mid_delay_ros_stop": float(rng.uniform(0.76, 0.88)),
                "long_delay_maintenance_i": float(rng.uniform(0.10, 0.20)),
                "mid_delay_maintenance_i": float(rng.uniform(0.12, 0.24)),
                "oxygen_limited_floor_scale": float(rng.uniform(0.35, 0.65)),
                "ros_stop_after_floor": float(rng.uniform(0.78, 0.90)),
            }
        )
    return candidates


def run_full_feedback_baseline(seeds: list[int], n_draws: int) -> pd.DataFrame:
    full_feedback = {"controller": "full_feedback", "signals": ("temperature", "oxygen", "ros"), "max_intensity_w_cm2": 1.30}
    outcomes = []
    for seed in seeds:
        draws = make_model_family_draws(sample_uncertainty(n=n_draws, seed=seed))
        scenarios = atlas_scenarios()
        total = len(draws) * len(scenarios)
        processed = 0
        for _, draw in draws.iterrows():
            for scenario_index, scenario in enumerate(scenarios):
                result, _ = simulate_ablation_case(full_feedback, scenario, draw, common_rng(seed, draw, scenario_index, 1))
                row = add_draw_metadata(result, draw, scenario, "feedback")
                row["validation_seed"] = f"seed_{seed}"
                outcomes.append(row)
                processed += 1
                if processed % 192 == 0:
                    print(f"training baseline seed {seed}: {processed}/{total}", flush=True)
    return pd.DataFrame(outcomes)


def run_policy_on_seeds(policy: dict, seeds: list[int], n_draws: int, candidate_index: int) -> pd.DataFrame:
    outcomes = []
    for seed in seeds:
        draws = make_model_family_draws(sample_uncertainty(n=n_draws, seed=seed))
        scenarios = atlas_scenarios()
        total = len(draws) * len(scenarios)
        processed = 0
        for _, draw in draws.iterrows():
            for scenario_index, scenario in enumerate(scenarios):
                result, _ = simulate_predictive_case(policy, scenario, draw, common_rng(seed, draw, scenario_index, 5, candidate_index))
                row = add_draw_metadata(result, draw, scenario, "optimized_predictive")
                row["validation_seed"] = f"seed_{seed}"
                row["candidate_id"] = policy["candidate_id"]
                outcomes.append(row)
                processed += 1
                if processed % 192 == 0:
                    print(f"{policy['candidate_id']} seed {seed}: {processed}/{total}", flush=True)
    return pd.DataFrame(outcomes)


def pair_controllers(scored: pd.DataFrame, to_controller: str = OPT_CONTROLLER) -> pd.DataFrame:
    keep = PAIR_KEYS + ["controller", "chance_success_int", "predicted_probability", "controllability_class", "class_index", "candidate_id"] + MECHANISM_FLAGS + CONTINUOUS_MECHANISMS
    for col in ["candidate_id"]:
        if col not in scored.columns:
            scored[col] = ""
    a = scored.loc[scored["controller"] == "full_feedback", keep].copy()
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
    paired["cluster_id"] = paired["validation_seed"].astype(str) + "|" + paired["model_family"].astype(str) + "|" + paired["atlas_uncertainty_id"].astype(str)
    for flag in MECHANISM_FLAGS:
        paired[f"{flag}_resolved"] = paired[f"from_{flag}"].astype(bool) & ~paired[f"to_{flag}"].astype(bool)
        paired[f"{flag}_induced"] = ~paired[f"from_{flag}"].astype(bool) & paired[f"to_{flag}"].astype(bool)
        paired[f"{flag}_net_resolution"] = paired[f"{flag}_resolved"].astype(int) - paired[f"{flag}_induced"].astype(int)
        paired[f"{flag}_rate_delta"] = paired[f"to_{flag}"].astype(int) - paired[f"from_{flag}"].astype(int)
    for col in CONTINUOUS_MECHANISMS:
        paired[f"{col}_delta"] = paired[f"to_{col}"] - paired[f"from_{col}"]
    return paired


def endpoint_metrics(paired: pd.DataFrame) -> dict:
    low = paired[paired["from_controllability_class"].astype(str) == "low"]
    trans = paired[paired["from_controllability_class"].astype(str) == "transitional"]
    high = paired[paired["from_controllability_class"].astype(str) == "high"]
    low_loss = float(low["lost_success"].mean())
    low_rescue = float(low["rescued_success"].mean())
    return {
        "n_low": len(low),
        "all_uplift": float(paired["success_uplift"].mean()),
        "low_uplift": float(low["success_uplift"].mean()),
        "transitional_uplift": float(trans["success_uplift"].mean()) if len(trans) else np.nan,
        "high_uplift": float(high["success_uplift"].mean()) if len(high) else np.nan,
        "rescued_to_lost_ratio": float(low_rescue / low_loss) if low_loss > 0 else float("inf"),
        "thermal_failure_delta": float(low["thermal_failure_rate_delta"].mean()),
        "ros_failure_delta": float(low["ros_failure_rate_delta"].mean()),
        "overheat_failure_delta": float(low["overheat_failure_rate_delta"].mean()),
        "energy_delta": float(low["delivered_energy_j_cm2_delta"].mean()),
    }


def objective(m: dict) -> float:
    return (
        m["low_uplift"]
        + 0.45 * min(m["thermal_failure_delta"], 0.0) * -1
        + 0.35 * min(m["ros_failure_delta"], 0.0) * -1
        - 0.95 * max(m["overheat_failure_delta"], 0.0)
        - 0.0018 * max(m["energy_delta"], 0.0)
        - 0.60 * max(-m["high_uplift"], 0.0)
        - 0.35 * max(-m["transitional_uplift"], 0.0)
    )


def evaluate_training_candidates(q33: float, q67: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline = run_full_feedback_baseline(TRAIN_SEEDS, N_TRAIN_DRAWS)
    baseline_scored = apply_locked_boundary(add_reduced_features(baseline), q33, q67)
    all_pairs = []
    rows = []
    policies = candidate_policies()
    for idx, policy in enumerate(policies):
        outcomes = run_policy_on_seeds(policy, TRAIN_SEEDS, N_TRAIN_DRAWS, idx)
        scored = apply_locked_boundary(add_reduced_features(outcomes), q33, q67)
        combined = pd.concat([baseline_scored, scored], ignore_index=True, sort=False)
        paired = pair_controllers(combined)
        paired["candidate_id"] = policy["candidate_id"]
        all_pairs.append(paired)
        metrics = endpoint_metrics(paired)
        rows.append({**metrics, "candidate_id": policy["candidate_id"], "objective": objective(metrics), "policy": json.dumps(policy, sort_keys=True)})
    table = pd.DataFrame(rows).sort_values("objective", ascending=False)
    best_policy = [p for p in policies if p["candidate_id"] == table.iloc[0]["candidate_id"]][0]
    return table, pd.concat(all_pairs, ignore_index=True), best_policy


def validate_policy(best_policy: dict, q33: float, q67: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(STEP13_SCORED_PATH)
    baseline = baseline[baseline["controller"] == "full_feedback"].copy()
    rows = []
    for idx, seed in enumerate(VALIDATION_SEEDS):
        rows.append(run_policy_on_seeds(best_policy, [seed], 32, idx))
    outcomes = pd.concat(rows, ignore_index=True)
    scored = apply_locked_boundary(add_reduced_features(outcomes), q33, q67)
    combined = pd.concat([baseline, scored], ignore_index=True, sort=False)
    paired = pair_controllers(combined)
    success = success_summary(paired)
    failures = failure_summary(paired)
    deltas = mechanism_delta_summary(paired)
    return paired, success, failures, deltas


def gate_decision(success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame) -> tuple[str, list[str]]:
    low = success[success["stratum"] == "from_low"].iloc[0]
    high = success[success["stratum"] == "from_high"].iloc[0]
    thermal = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "thermal_failure")].iloc[0]
    ros = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "ros_failure")].iloc[0]
    overheat = failures[(failures["stratum"] == "from_low") & (failures["failure_mode"] == "overheat_failure")].iloc[0]
    energy = deltas[(deltas["stratum"] == "from_low") & (deltas["mechanism_variable"] == "delivered_energy_j_cm2")].iloc[0]
    checks = [
        f"Low uplift = {low['success_uplift']:.3f} [{low['success_uplift_ci_low']:.3f}, {low['success_uplift_ci_high']:.3f}].",
        f"High-class uplift = {high['success_uplift']:.3f} [{high['success_uplift_ci_low']:.3f}, {high['success_uplift_ci_high']:.3f}].",
        f"Thermal failure delta = {thermal['failure_rate_delta']:.3f}.",
        f"ROS failure delta = {ros['failure_rate_delta']:.3f}.",
        f"Overheat failure delta = {overheat['failure_rate_delta']:.3f}.",
        f"Energy delta = {energy['mean_delta']:.2f} J cm^-2.",
    ]
    passes = (
        low["success_uplift"] >= 0.08
        and low["success_uplift_ci_low"] > 0
        and high["success_uplift"] > -0.04
        and thermal["failure_rate_delta"] <= 0
        and ros["failure_rate_delta"] <= 0
        and overheat["failure_rate_delta"] <= 0.03
        and energy["mean_delta_ci_high"] <= 50
    )
    decision = (
        "CONFIRMED: constrained optimization found a policy satisfying the pre-declared rescue/safety tradeoff."
        if passes
        else "NOT CONFIRMED: constrained optimization improved the search discipline but did not yet satisfy the validation gate."
    )
    return decision, checks


def make_figure(training: pd.DataFrame, success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(training["overheat_failure_delta"], training["low_uplift"], c=training["objective"], cmap="viridis", s=45)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Training low-class overheat delta")
    ax.set_ylabel("Training low-class uplift")
    ax.set_title("A. Constrained search")

    ax = axes[0, 1]
    plot = success[success["stratum"].isin(["from_low", "from_transitional", "from_high"])]
    ax.bar(plot["stratum"].str.replace("from_", "", regex=False), plot["success_uplift"], color=["#b45f4d", "#d59b43", "#347f68"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Validation success uplift")
    ax.set_title("B. Locked validation")

    ax = axes[1, 0]
    low_fail = failures[failures["stratum"] == "from_low"]
    labels = low_fail["failure_mode"].str.replace("_failure", "", regex=False).str.upper()
    ax.bar(labels, -low_fail["failure_rate_delta"], color="#4b7fb8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Failure-rate reduction")
    ax.set_title("C. Low-class failure modes")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    low_delta = deltas[deltas["stratum"] == "from_low"].set_index("mechanism_variable")
    show = ["warm_fraction", "ros_threshold_fraction", "overheat_fraction", "delivered_energy_j_cm2"]
    ax.bar(["warm", "ROS", "overheat", "energy"], low_delta.loc[show, "mean_delta"], color=["#347f68", "#347f68", "#b45f4d", "#777777"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("D. Validation state shifts")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    q33, q67 = locked_class_thresholds(REFERENCE_SCORED)
    training, training_pairs, best_policy = evaluate_training_candidates(q33, q67)
    paired, success, failures, deltas = validate_policy(best_policy, q33, q67)
    decision, checks = gate_decision(success, failures, deltas)

    training.to_csv(OUT_DATA / "training_candidate_objectives.csv", index=False)
    training_pairs.to_csv(OUT_DATA / "training_candidate_pairs.csv", index=False)
    paired.to_csv(OUT_DATA / "optimized_policy_locked_validation_pairs.csv", index=False)
    success.to_csv(OUT_DATA / "optimized_policy_success_uplift.csv", index=False)
    failures.to_csv(OUT_DATA / "optimized_policy_failure_modes.csv", index=False)
    deltas.to_csv(OUT_DATA / "optimized_policy_mechanism_deltas.csv", index=False)
    (OUT_DATA / "optimized_policy_metadata.json").write_text(
        json.dumps(
            {
                "train_seeds": TRAIN_SEEDS,
                "validation_seeds": VALIDATION_SEEDS,
                "n_train_draws": N_TRAIN_DRAWS,
                "n_candidates": N_CANDIDATES,
                "best_policy": best_policy,
                "locked_boundary": LOCKED_BOUNDARY,
                "decision": decision,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_figure(training, success, failures, deltas, OUT_FIG / "figure_15_constrained_policy_optimization.png")

    note = [
        "# Step 15 Constrained Policy Optimization",
        "",
        "Question: can a policy selected by a pre-declared constrained objective solve the thermal-underdose versus overheat tradeoff better than hand-designed rules?",
        "",
        "## Method",
        "",
        f"Training seeds: {TRAIN_SEEDS}; validation seeds: {VALIDATION_SEEDS}.",
        f"Training draws per seed before family transforms: {N_TRAIN_DRAWS}.",
        f"Candidate policies screened: {N_CANDIDATES}.",
        "The selected policy is validated on the locked Step 13 seeds against the inherited full-feedback baseline.",
        "",
        "## Best Policy",
        "",
        "```json",
        json.dumps(best_policy, indent=2),
        "```",
        "",
        "## Training Candidate Ranking",
        "",
        "```",
        training.head(10).round(3).to_string(index=False),
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
        "## Gate Checks",
        "",
        *[f"- {line}" for line in checks],
        "",
        "## Decision",
        "",
        decision,
    ]
    (OUT_NOTES / "step15_constrained_policy_optimization_result.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print("Wrote constrained policy optimization outputs")
    print(training.head(5).round(3).to_string(index=False))
    print(success.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
