from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NCS = ROOT / "nature_computational_science"
IN_DATA = NCS / "data" / "predictive_control_reclassification" / "multiseed_outofseed_scored_controllers.csv"
OUT_DATA = NCS / "data" / "failure_mode_decomposition"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

FROM_CONTROLLER = "full_feedback"
TO_CONTROLLER = "predictive_smoothed_full"
CLASS_ORDER = ["low", "transitional", "high"]

KEYS = [
    "validation_seed",
    "model_family",
    "atlas_uncertainty_id",
    "scenario",
    "delay_s",
    "noise_fraction",
]

MECHANISM_FLAGS = [
    "thermal_failure",
    "ros_failure",
    "hypoxic_failure",
    "overheat_failure",
]

CONTINUOUS_MECHANISMS = [
    "max_temperature_c",
    "warm_fraction",
    "overheat_fraction",
    "ros_threshold_fraction",
    "hypoxic_failure_fraction",
    "delivered_energy_j_cm2",
    "mean_intensity_w_cm2",
    "light_on_time_s",
    "protocol_burden",
    "observation_quality",
    "delay_diffusion_ratio",
    "dose_capacity_number",
    "reaction_number",
    "reaction_drive_balance",
]


def delay_bin(delay_s: float) -> str:
    if delay_s <= 0:
        return "0 s"
    if delay_s <= 30:
        return "1-30 s"
    if delay_s <= 60:
        return "31-60 s"
    return ">60 s"


def paired_controller_table(scored: pd.DataFrame) -> pd.DataFrame:
    keep = (
        KEYS
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
    missing = [col for col in keep if col not in scored.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    a = scored.loc[scored["controller"] == FROM_CONTROLLER, keep].copy()
    b = scored.loc[scored["controller"] == TO_CONTROLLER, keep].copy()
    a = a.rename(columns={col: f"from_{col}" for col in keep if col not in KEYS})
    b = b.rename(columns={col: f"to_{col}" for col in keep if col not in KEYS})
    paired = a.merge(b, on=KEYS, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError("No paired rows found for full feedback versus predictive feedback.")

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


def clustered_ci(df: pd.DataFrame, value_col: str, n_boot: int = 1200, seed: int = 20260718) -> tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    grouped = df.groupby("cluster_id", observed=False)[value_col].agg(["sum", "count"]).reset_index()
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    n_clusters = len(grouped)
    vals = []
    for _ in range(n_boot):
        sampled = rng.integers(0, n_clusters, n_clusters)
        vals.append(float(sums[sampled].sum() / counts[sampled].sum()))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def summarize_success(df: pd.DataFrame, label: str) -> dict:
    lo, hi = clustered_ci(df, "success_uplift")
    rescued = float(df["rescued_success"].mean())
    lost = float(df["lost_success"].mean())
    return {
        "stratum": label,
        "n": len(df),
        "clusters": df["cluster_id"].nunique(),
        "from_success_rate": float(df["from_chance_success_int"].mean()),
        "to_success_rate": float(df["to_chance_success_int"].mean()),
        "success_uplift": float(df["success_uplift"].mean()),
        "success_uplift_ci_low": lo,
        "success_uplift_ci_high": hi,
        "rescued_success_rate": rescued,
        "lost_success_rate": lost,
        "rescued_to_lost_ratio": float(rescued / lost) if lost > 0 else float("inf"),
        "mean_class_shift": float(df["class_shift"].mean()),
        "mean_probability_shift": float(df["probability_shift"].mean()),
    }


def success_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_success(paired, "all")]
    for cls in CLASS_ORDER:
        g = paired[paired["from_controllability_class"].astype(str) == cls]
        rows.append(summarize_success(g, f"from_{cls}"))
    low = paired[paired["from_controllability_class"].astype(str) == "low"]
    for seed, g in low.groupby("validation_seed"):
        rows.append(summarize_success(g, f"low_seed_{seed}"))
    for family, g in low.groupby("model_family"):
        rows.append(summarize_success(g, f"low_family_{family}"))
    for dbin, g in low.groupby("delay_bin", sort=False):
        rows.append(summarize_success(g, f"low_delay_{dbin}"))
    return pd.DataFrame(rows)


def failure_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strata = {"all": paired, "from_low": paired[paired["from_controllability_class"].astype(str) == "low"]}
    low = strata["from_low"]
    for seed, g in low.groupby("validation_seed"):
        strata[f"low_seed_{seed}"] = g
    for family, g in low.groupby("model_family"):
        strata[f"low_family_{family}"] = g
    for dbin, g in low.groupby("delay_bin", sort=False):
        strata[f"low_delay_{dbin}"] = g

    for stratum, df in strata.items():
        for flag in MECHANISM_FLAGS:
            delta_col = f"{flag}_rate_delta"
            net_col = f"{flag}_net_resolution"
            lo, hi = clustered_ci(df, delta_col)
            net_lo, net_hi = clustered_ci(df, net_col)
            from_rate = float(df[f"from_{flag}"].astype(int).mean())
            to_rate = float(df[f"to_{flag}"].astype(int).mean())
            rows.append(
                {
                    "stratum": stratum,
                    "failure_mode": flag,
                    "n": len(df),
                    "from_failure_rate": from_rate,
                    "to_failure_rate": to_rate,
                    "failure_rate_delta": float(df[delta_col].mean()),
                    "failure_rate_delta_ci_low": lo,
                    "failure_rate_delta_ci_high": hi,
                    "resolved_rate": float(df[f"{flag}_resolved"].mean()),
                    "induced_rate": float(df[f"{flag}_induced"].mean()),
                    "net_resolution_rate": float(df[net_col].mean()),
                    "net_resolution_ci_low": net_lo,
                    "net_resolution_ci_high": net_hi,
                    "relative_failure_reduction": float((from_rate - to_rate) / from_rate) if from_rate > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def mechanism_delta_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strata = {"all": paired, "from_low": paired[paired["from_controllability_class"].astype(str) == "low"]}
    for col in CONTINUOUS_MECHANISMS:
        for stratum, df in strata.items():
            delta_col = f"{col}_delta"
            lo, hi = clustered_ci(df, delta_col)
            rows.append(
                {
                    "stratum": stratum,
                    "mechanism_variable": col,
                    "n": len(df),
                    "from_mean": float(df[f"from_{col}"].mean()),
                    "to_mean": float(df[f"to_{col}"].mean()),
                    "mean_delta": float(df[delta_col].mean()),
                    "mean_delta_ci_low": lo,
                    "mean_delta_ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def rescue_profile(paired: pd.DataFrame) -> pd.DataFrame:
    low = paired[paired["from_controllability_class"].astype(str) == "low"].copy()
    rows = []
    for label, mask in {
        "rescued": low["rescued_success"],
        "lost": low["lost_success"],
        "unchanged_failure": (low["from_chance_success_int"] == 0) & (low["to_chance_success_int"] == 0),
        "unchanged_success": (low["from_chance_success_int"] == 1) & (low["to_chance_success_int"] == 1),
    }.items():
        g = low[mask]
        row = {"outcome_transition": label, "n": len(g)}
        for flag in MECHANISM_FLAGS:
            row[f"from_{flag}_rate"] = float(g[f"from_{flag}"].astype(int).mean()) if len(g) else float("nan")
            row[f"to_{flag}_rate"] = float(g[f"to_{flag}"].astype(int).mean()) if len(g) else float("nan")
        for col in ["max_temperature_c", "warm_fraction", "ros_threshold_fraction", "hypoxic_failure_fraction", "delivered_energy_j_cm2"]:
            row[f"from_{col}_mean"] = float(g[f"from_{col}"].mean()) if len(g) else float("nan")
            row[f"to_{col}_mean"] = float(g[f"to_{col}"].mean()) if len(g) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame, paired: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    plot = success[success["stratum"].isin(["from_low", "from_transitional", "from_high"])].copy()
    labels = plot["stratum"].str.replace("from_", "", regex=False)
    y = plot["success_uplift"]
    yerr = np.vstack([y - plot["success_uplift_ci_low"], plot["success_uplift_ci_high"] - y])
    ax.bar(labels, y, color=["#b45f4d", "#d59b43", "#347f68"], alpha=0.9)
    ax.errorbar(labels, y, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Paired success uplift")
    ax.set_title("A. Predictive rescue by class")

    ax = axes[0, 1]
    low_fail = failures[failures["stratum"] == "from_low"].copy()
    labels = low_fail["failure_mode"].str.replace("_failure", "", regex=False).str.upper()
    y = -low_fail["failure_rate_delta"]
    yerr = np.vstack(
        [
            y - (-low_fail["failure_rate_delta_ci_high"]),
            (-low_fail["failure_rate_delta_ci_low"]) - y,
        ]
    )
    ax.bar(labels, y, color="#4b7fb8")
    ax.errorbar(labels, y, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
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
    y = low_delay["success_uplift"]
    yerr = np.vstack([y - low_delay["success_uplift_ci_low"], low_delay["success_uplift_ci_high"] - y])
    ax.plot(low_delay["delay_label"].astype(str), y, marker="o", color="#6d4c91")
    ax.errorbar(low_delay["delay_label"].astype(str), y, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Low-class success uplift")
    ax.set_title("C. Delay-stratified rescue")

    ax = axes[1, 0]
    low_delta = deltas[deltas["stratum"] == "from_low"].copy()
    show = [
        "warm_fraction",
        "ros_threshold_fraction",
        "hypoxic_failure_fraction",
        "overheat_fraction",
    ]
    low_delta = low_delta[low_delta["mechanism_variable"].isin(show)]
    labels = [
        "warm\nfraction",
        "ROS\nfraction",
        "hypoxic\nfraction",
        "overheat\nfraction",
    ]
    low_delta = low_delta.set_index("mechanism_variable").loc[show].reset_index()
    colors = ["#347f68", "#347f68", "#b45f4d", "#b45f4d"]
    y = 100 * low_delta["mean_delta"]
    yerr = np.vstack(
        [
            y - 100 * low_delta["mean_delta_ci_low"],
            100 * low_delta["mean_delta_ci_high"] - y,
        ]
    )
    ax.bar(labels, y, color=colors)
    ax.errorbar(labels, y, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Predictive - full (percentage points)")
    ax.set_title("D. Mechanistic state shifts")

    ax = axes[1, 1]
    low = paired[paired["from_controllability_class"].astype(str) == "low"]
    bins = np.linspace(0, 1, 16)
    ax.hist(low.loc[low["rescued_success"], "from_predicted_probability"], bins=bins, alpha=0.75, label="rescued", color="#347f68")
    ax.hist(low.loc[low["lost_success"], "from_predicted_probability"], bins=bins, alpha=0.75, label="lost", color="#b45f4d")
    ax.set_xlabel("Initial controllability probability")
    ax.set_ylabel("Count")
    ax.set_title("E. Rescue/loss starting positions")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    low_family = success[success["stratum"].str.startswith("low_family_")].copy()
    low_family["family"] = low_family["stratum"].str.replace("low_family_", "", regex=False)
    low_family = low_family.sort_values("success_uplift")
    y_pos = np.arange(len(low_family))
    x = low_family["success_uplift"].to_numpy(float)
    xerr = np.vstack(
        [
            x - low_family["success_uplift_ci_low"].to_numpy(float),
            low_family["success_uplift_ci_high"].to_numpy(float) - x,
        ]
    )
    ax.barh(y_pos, x, color="#4b7fb8")
    ax.errorbar(x, y_pos, xerr=xerr, fmt="none", color="black", capsize=3, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(low_family["family"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Low-class success uplift")
    ax.set_title("F. Family consistency")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def gate_decision(success: pd.DataFrame, failures: pd.DataFrame, deltas: pd.DataFrame) -> tuple[str, list[str]]:
    low = success[success["stratum"] == "from_low"].iloc[0]
    low_family = success[success["stratum"].str.startswith("low_family_")]
    low_seed = success[success["stratum"].str.startswith("low_seed_")]
    energy = deltas[(deltas["stratum"] == "from_low") & (deltas["mechanism_variable"] == "delivered_energy_j_cm2")].iloc[0]
    low_fail = failures[failures["stratum"] == "from_low"].copy()
    dominant = low_fail.sort_values("relative_failure_reduction", ascending=False).iloc[0]

    checks = [
        f"Low-class success uplift = {low['success_uplift']:.3f} [{low['success_uplift_ci_low']:.3f}, {low['success_uplift_ci_high']:.3f}].",
        f"Low-class rescued/lost ratio = {low['rescued_to_lost_ratio']:.2f}.",
        f"Lowest low-family uplift = {low_family['success_uplift'].min():.3f}.",
        f"Lowest low-seed uplift = {low_seed['success_uplift'].min():.3f}.",
        f"Delivered-energy delta in low class = {energy['mean_delta']:.2f} J cm^-2 [{energy['mean_delta_ci_low']:.2f}, {energy['mean_delta_ci_high']:.2f}].",
        f"Dominant failure reduction = {dominant['failure_mode']} with relative reduction {dominant['relative_failure_reduction']:.3f}.",
    ]

    passes = (
        low["success_uplift"] >= 0.10
        and low["success_uplift_ci_low"] > 0
        and low["rescued_to_lost_ratio"] >= 2.0
        and low_family["success_uplift"].min() > 0
        and low_seed["success_uplift"].min() > 0
        and energy["mean_delta"] <= 25.0
        and low_fail["relative_failure_reduction"].max() >= 0.20
    )
    if passes:
        decision = (
            "CONTINUE TOWARD HIGH-IMPACT MANUSCRIPT: predictive feedback shows a seed- and family-consistent "
            "low-controllability rescue with a mechanistic failure-mode reduction and no large energy penalty."
        )
    else:
        decision = (
            "PIVOT OR STRENGTHEN BEFORE NCS: predictive feedback may still be useful, but the current evidence does "
            "not yet meet the strict mechanistic-rescue threshold needed for a Nature Computational Science-level claim."
        )
    return decision, checks


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(IN_DATA)
    paired = paired_controller_table(scored)
    success = success_summary(paired)
    failures = failure_summary(paired)
    deltas = mechanism_delta_summary(paired)
    profiles = rescue_profile(paired)

    paired.to_csv(OUT_DATA / "paired_full_to_predictive_failure_modes.csv", index=False)
    success.to_csv(OUT_DATA / "success_uplift_by_stratum.csv", index=False)
    failures.to_csv(OUT_DATA / "failure_mode_resolution_by_stratum.csv", index=False)
    deltas.to_csv(OUT_DATA / "mechanism_delta_by_stratum.csv", index=False)
    profiles.to_csv(OUT_DATA / "low_class_rescue_transition_profiles.csv", index=False)

    make_figure(success, failures, deltas, paired, OUT_FIG / "figure_12_failure_mode_decomposition.png")
    decision, checks = gate_decision(success, failures, deltas)

    metadata = {
        "input": str(IN_DATA.relative_to(ROOT)),
        "comparison": f"{FROM_CONTROLLER}_to_{TO_CONTROLLER}",
        "method": "Paired counterfactual failure-mode decomposition on matched seed, model family, uncertainty draw, delay/noise scenario.",
        "bootstrap": "Cluster bootstrap by validation seed, model family, and biological uncertainty draw.",
        "strict_gate": [
            "low-class success uplift >= 0.10",
            "bootstrap lower confidence bound > 0",
            "rescued/lost ratio >= 2",
            "positive uplift in every low-class seed and family",
            "low-class delivered-energy increase <= 25 J cm^-2",
            "at least one low-class failure mode reduced by >= 20%",
        ],
    }
    (OUT_DATA / "failure_mode_decomposition_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    low_success = success[success["stratum"] == "from_low"].iloc[0]
    low_fail = failures[failures["stratum"] == "from_low"].copy()
    low_delta = deltas[deltas["stratum"] == "from_low"].copy()
    note = [
        "# Step 12 Low-Controllability Failure-Mode Decomposition",
        "",
        "Question: does predictive feedback rescue low-controllability cases through a specific physical mechanism, or is the Step 8 uplift only a statistical artifact?",
        "",
        "## Method",
        "",
        "The analysis pairs full feedback and predictive feedback on the same validation seed, model family, uncertainty draw, delay/noise scenario, and biological parameters. It then decomposes paired success changes into thermal, ROS/product, hypoxic/resource, and overheating failure modes. Confidence intervals use a cluster bootstrap over validation seed, model family, and biological uncertainty draw.",
        "",
        "## Success-Uplift Summary",
        "",
        "```",
        success.round(3).to_string(index=False),
        "```",
        "",
        "## Low-Class Failure-Mode Summary",
        "",
        "```",
        low_fail.round(3).to_string(index=False),
        "```",
        "",
        "## Low-Class Mechanistic State Shifts",
        "",
        "```",
        low_delta.round(3).to_string(index=False),
        "```",
        "",
        "## Strict Step-Gate Checks",
        "",
        *[f"- {line}" for line in checks],
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Interpretation",
        "",
        (
            f"The low-controllability class changes from {low_success['from_success_rate']:.3f} to "
            f"{low_success['to_success_rate']:.3f} success under predictive feedback. The key question for the "
            "manuscript is whether this improvement is both mechanistically localizable and reproducible across seeds and families."
        ),
    ]
    (OUT_NOTES / "step12_failure_mode_decomposition_result.md").write_text("\n".join(note) + "\n", encoding="utf-8")

    print("Wrote Step 12 failure-mode decomposition outputs")
    print(success.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
