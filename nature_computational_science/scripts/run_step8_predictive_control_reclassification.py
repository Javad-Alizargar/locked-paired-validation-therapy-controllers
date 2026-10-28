from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_step7_multiseed_controllability_classes import (
    cross_seed_metrics,
    load_data,
)


ROOT = Path(__file__).resolve().parents[2]
NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data" / "predictive_control_reclassification"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

FEATURES = [
    "log_dose_capacity_number",
    "log_reaction_number*observation_quality",
    "log_reaction_drive_balance",
]

CLASS_LABELS = ["low", "transitional", "high"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASS_LABELS)}


def add_outofseed_predictions(df: pd.DataFrame) -> pd.DataFrame:
    _, predictions = cross_seed_metrics(df, FEATURES, "multiseed_boundary")
    keys = ["validation_seed", "model_family", "controller", "chance_success_int"]
    pred = predictions.copy()
    pred["_row_order"] = pred.groupby(keys).cumcount()
    base = df.copy()
    base["_row_order"] = base.groupby(keys).cumcount()
    out = base.merge(
        pred[keys + ["_row_order", "predicted_probability", "trained_on_seed"]],
        on=keys + ["_row_order"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["_row_order"])
    q = out["predicted_probability"].quantile([0.33, 0.67]).to_numpy()
    out["controllability_class"] = pd.cut(
        out["predicted_probability"],
        bins=[-np.inf, q[0], q[1], np.inf],
        labels=CLASS_LABELS,
    )
    out["class_index"] = out["controllability_class"].map(CLASS_TO_INT).astype(int)
    return out


def paired_table(df: pd.DataFrame, from_controller: str, to_controller: str) -> pd.DataFrame:
    keys = ["validation_seed", "model_family", "atlas_uncertainty_id", "scenario", "delay_s", "noise_fraction"]
    keep = keys + [
        "controller",
        "chance_success_int",
        "predicted_probability",
        "controllability_class",
        "class_index",
        "dose_capacity_number",
        "observation_quality",
        "log_reaction_number",
        "log_reaction_drive_balance",
    ]
    a = df[df["controller"] == from_controller][keep].copy()
    b = df[df["controller"] == to_controller][keep].copy()
    a = a.rename(columns={col: f"from_{col}" for col in keep if col not in keys})
    b = b.rename(columns={col: f"to_{col}" for col in keep if col not in keys})
    paired = a.merge(b, on=keys, how="inner", validate="one_to_one")
    paired["class_shift"] = paired["to_class_index"] - paired["from_class_index"]
    paired["probability_shift"] = paired["to_predicted_probability"] - paired["from_predicted_probability"]
    paired["success_uplift"] = paired["to_chance_success_int"] - paired["from_chance_success_int"]
    paired["up_class"] = paired["class_shift"] > 0
    paired["down_class"] = paired["class_shift"] < 0
    paired["rescued_success"] = (paired["from_chance_success_int"] == 0) & (paired["to_chance_success_int"] == 1)
    paired["lost_success"] = (paired["from_chance_success_int"] == 1) & (paired["to_chance_success_int"] == 0)
    paired["comparison"] = f"{from_controller}_to_{to_controller}"
    return paired


def bootstrap_ci(paired: pd.DataFrame, value_col: str, n_boot: int = 1000, seed: int = 20260714) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = paired[["validation_seed", "atlas_uncertainty_id"]].drop_duplicates().reset_index(drop=True)
    vals = []
    for _ in range(n_boot):
        sampled = clusters.iloc[rng.integers(0, len(clusters), len(clusters))]
        sampled = sampled.assign(_boot=np.arange(len(sampled)))
        merged = sampled.merge(paired, on=["validation_seed", "atlas_uncertainty_id"], how="left")
        vals.append(float(merged[value_col].mean()))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def summarize_pair(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall = {
        "comparison": paired["comparison"].iloc[0],
        "stratum": "all",
        "n": len(paired),
        "mean_class_shift": float(paired["class_shift"].mean()),
        "p_up_class": float(paired["up_class"].mean()),
        "p_down_class": float(paired["down_class"].mean()),
        "mean_probability_shift": float(paired["probability_shift"].mean()),
        "mean_success_uplift": float(paired["success_uplift"].mean()),
        "rescued_success_rate": float(paired["rescued_success"].mean()),
        "lost_success_rate": float(paired["lost_success"].mean()),
    }
    lo, hi = bootstrap_ci(paired, "success_uplift")
    overall["success_uplift_ci_low"] = lo
    overall["success_uplift_ci_high"] = hi
    rows.append(overall)
    for cls in CLASS_LABELS:
        g = paired[paired["from_controllability_class"].astype(str) == cls]
        if g.empty:
            continue
        row = {
            "comparison": paired["comparison"].iloc[0],
            "stratum": f"from_{cls}",
            "n": len(g),
            "mean_class_shift": float(g["class_shift"].mean()),
            "p_up_class": float(g["up_class"].mean()),
            "p_down_class": float(g["down_class"].mean()),
            "mean_probability_shift": float(g["probability_shift"].mean()),
            "mean_success_uplift": float(g["success_uplift"].mean()),
            "rescued_success_rate": float(g["rescued_success"].mean()),
            "lost_success_rate": float(g["lost_success"].mean()),
        }
        lo, hi = bootstrap_ci(g, "success_uplift")
        row["success_uplift_ci_low"] = lo
        row["success_uplift_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def transition_matrix(paired: pd.DataFrame) -> pd.DataFrame:
    mat = pd.crosstab(
        paired["from_controllability_class"],
        paired["to_controllability_class"],
        normalize="index",
    ).reindex(index=CLASS_LABELS, columns=CLASS_LABELS).fillna(0.0)
    return mat.reset_index().rename(columns={"from_controllability_class": "from_class"})


def make_figure(paired_predictive: pd.DataFrame, paired_frozen: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    mat = pd.crosstab(
        paired_predictive["from_controllability_class"],
        paired_predictive["to_controllability_class"],
        normalize="index",
    ).reindex(index=CLASS_LABELS, columns=CLASS_LABELS).fillna(0.0)
    im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_LABELS, rotation=20)
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predictive-control class")
    ax.set_ylabel("Full-feedback class")
    ax.set_title("A. Predictive reclassification")
    fig.colorbar(im, ax=ax, label="Row fraction")

    ax = axes[0, 1]
    plot = summary[(summary["comparison"].str.contains("predictive")) & (summary["stratum"] != "all")]
    ax.bar(plot["stratum"].str.replace("from_", "", regex=False), plot["mean_success_uplift"], color="#2f7f4f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Paired success uplift")
    ax.set_title("B. Uplift by starting class")

    ax = axes[1, 0]
    for label, paired, color in [
        ("predictive", paired_predictive, "#2f7f4f"),
        ("frozen minimax", paired_frozen, "#6c5b7b"),
    ]:
        ax.scatter(
            paired["from_predicted_probability"],
            paired["to_predicted_probability"],
            s=10,
            alpha=0.20,
            label=label,
            color=color,
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Full-feedback controllability probability")
    ax.set_ylabel("Comparator controllability probability")
    ax.set_title("C. Paired probability shift")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    overall = summary[summary["stratum"] == "all"].copy()
    labels = overall["comparison"].str.replace("full_feedback_to_", "", regex=False)
    ax.bar(labels, overall["mean_class_shift"], color=["#2f7f4f", "#6c5b7b"][: len(overall)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean class shift")
    ax.set_title("D. Net controllability reclassification")
    ax.tick_params(axis="x", rotation=20)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    scored = add_outofseed_predictions(df)
    scored.to_csv(OUT_DATA / "multiseed_outofseed_scored_controllers.csv", index=False)

    pair_predictive = paired_table(scored, "full_feedback", "predictive_smoothed_full")
    pair_frozen = paired_table(scored, "full_feedback", "frozen_minimax_geometry_dose")
    pair_predictive.to_csv(OUT_DATA / "paired_full_to_predictive.csv", index=False)
    pair_frozen.to_csv(OUT_DATA / "paired_full_to_frozen_minimax.csv", index=False)

    summary = pd.concat([summarize_pair(pair_predictive), summarize_pair(pair_frozen)], ignore_index=True)
    summary.to_csv(OUT_DATA / "predictive_control_reclassification_summary.csv", index=False)
    transition_matrix(pair_predictive).to_csv(OUT_DATA / "transition_matrix_full_to_predictive.csv", index=False)
    transition_matrix(pair_frozen).to_csv(OUT_DATA / "transition_matrix_full_to_frozen_minimax.csv", index=False)

    metadata = {
        "features": FEATURES,
        "method": "Paired in-silico counterfactual reclassification using leave-one-seed-out controllability probabilities.",
        "comparisons": ["full_feedback_to_predictive_smoothed_full", "full_feedback_to_frozen_minimax_geometry_dose"],
        "bootstrap": "Cluster bootstrap by validation seed and uncertainty draw for paired success uplift.",
    }
    (OUT_DATA / "predictive_control_reclassification_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figure(pair_predictive, pair_frozen, summary, OUT_FIG / "figure_8_predictive_control_reclassification.png")

    pred_overall = summary[(summary["comparison"] == "full_feedback_to_predictive_smoothed_full") & (summary["stratum"] == "all")].iloc[0]
    low_trans = summary[
        (summary["comparison"] == "full_feedback_to_predictive_smoothed_full")
        & (summary["stratum"].isin(["from_low", "from_transitional"]))
    ]
    decision = (
        "CONTINUE: predictive control produces upward controllability reclassification with non-negative paired success uplift."
        if pred_overall["mean_class_shift"] > 0 and pred_overall["mean_success_uplift"] >= 0
        else "PIVOT: predictive control does not consistently move cases upward or improve paired success."
    )
    lines = [
        "# Step 8 Predictive-Control Reclassification",
        "",
        "Question: does predictive control move matched cases from lower to higher seed-stable controllability classes?",
        "",
        "## Method",
        "",
        "Each controller is paired on the same validation seed, model family, uncertainty draw, delay/noise scenario, and biological parameters. Controllability probabilities are out-of-seed predictions from the Step 7 reduced boundary. We compare full feedback against predictive feedback and frozen minimax feedback using paired reclassification, paired success uplift, and cluster bootstrap intervals.",
        "",
        "## Summary",
        "",
        "```",
        summary.round(3).to_string(index=False),
        "```",
        "",
        "## Step-Gate Decision",
        "",
        f"Predictive mean class shift: {pred_overall['mean_class_shift']:.3f}.",
        f"Predictive paired success uplift: {pred_overall['mean_success_uplift']:.3f} [{pred_overall['success_uplift_ci_low']:.3f}, {pred_overall['success_uplift_ci_high']:.3f}].",
        f"Low/transitional mean class shift: {low_trans['mean_class_shift'].mean():.3f}.",
        f"Low/transitional paired success uplift: {low_trans['mean_success_uplift'].mean():.3f}.",
        decision,
        "",
        "Interpretation: a positive class shift without meaningful success uplift would support mechanism/stratification but not therapeutic superiority. A positive class shift with paired success uplift supports predictive-control recovery as a central result.",
    ]
    (OUT_NOTES / "step8_predictive_control_reclassification_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote predictive-control reclassification outputs")
    print(summary.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
