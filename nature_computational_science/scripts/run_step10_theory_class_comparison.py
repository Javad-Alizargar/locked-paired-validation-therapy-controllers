from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
NCS = ROOT / "nature_computational_science"
IN_FILES = {
    "seed_20260712": NCS / "data" / "frozen_policy_validation" / "frozen_policy_validation_reduced_features.csv",
    "seed_20260713": NCS / "data" / "external_boundary_validation" / "external_boundary_validation_reduced_scored.csv",
}
OUT_DATA = NCS / "data" / "theory_class_comparison"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"
EPS = 1e-12


BASE_TERMS = [
    "log_dose_capacity_number",
    "log_reaction_number",
    "log_reaction_drive_balance",
    "log_thermal_diffusion_time_s",
    "log_perfusion_diffusion_ratio",
    "dose_deficit",
    "delay_diffusion_ratio",
    "observation_quality",
    "duty_proxy",
]

THEORY_FEATURES = {
    "current_three_term": [
        "log_dose_capacity_number",
        "log_reaction_number*observation_quality",
        "log_reaction_drive_balance",
    ],
    "information_recovery": [
        "observation_quality",
        "delay_diffusion_ratio",
        "log_perfusion_diffusion_ratio",
        "observation_quality*log_perfusion_diffusion_ratio",
        "dose_deficit",
    ],
    "dose_geometry_information": [
        "log_dose_capacity_number",
        "log_reaction_number*observation_quality",
        "log_reaction_drive_balance",
        "log_thermal_diffusion_time_s*observation_quality",
        "dose_deficit*delay_diffusion_ratio",
    ],
    "controller_augmented": [
        "log_dose_capacity_number",
        "log_reaction_number*observation_quality",
        "log_reaction_drive_balance",
        "controller_predictive",
        "controller_fixed",
        "controller_frozen_minimax",
        "controller_geometry_dose",
        "controller_predictive*observation_quality",
        "controller_predictive*dose_deficit",
    ],
    "failure_mode_theory": [
        "log_dose_capacity_number",
        "log_reaction_number",
        "log_reaction_drive_balance",
        "dose_deficit",
        "delay_diffusion_ratio",
        "duty_proxy*log_dose_capacity_number",
        "observation_quality*delay_diffusion_ratio",
    ],
}


INTERACTIONS = [
    ("log_reaction_number", "observation_quality"),
    ("log_thermal_diffusion_time_s", "observation_quality"),
    ("observation_quality", "log_perfusion_diffusion_ratio"),
    ("dose_deficit", "delay_diffusion_ratio"),
    ("duty_proxy", "log_dose_capacity_number"),
    ("observation_quality", "delay_diffusion_ratio"),
    ("controller_predictive", "observation_quality"),
    ("controller_predictive", "dose_deficit"),
]


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(vals) + 1)
    _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic_fit(x_train: np.ndarray, y_train: np.ndarray, l2: float = 0.04) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0) + EPS
    x = np.nan_to_num((x_train - mu) / sd)
    x = np.c_[np.ones(len(x)), x]
    y = y_train.astype(float)
    n_pos = max(y.sum(), 1.0)
    n_neg = max((1 - y).sum(), 1.0)
    sample_weights = np.where(y > 0, len(y) / (2 * n_pos), len(y) / (2 * n_neg))

    def loss(w: np.ndarray) -> float:
        z = np.clip(x @ w, -40, 40)
        return float(np.sum(sample_weights * (np.logaddexp(0, z) - y * z)) + l2 * np.sum(w[1:] ** 2))

    def grad(w: np.ndarray) -> np.ndarray:
        z = np.clip(x @ w, -40, 40)
        p = 1 / (1 + np.exp(-z))
        g = x.T @ (sample_weights * (p - y))
        g[1:] += 2 * l2 * w[1:]
        return g

    result = minimize(loss, np.zeros(x.shape[1]), jac=grad, method="L-BFGS-B", options={"maxiter": 250})
    return result.x, mu, sd


def logistic_predict(x_test: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = np.nan_to_num((x_test - mu) / (sd + EPS))
    x = np.c_[np.ones(len(x)), x]
    z = np.clip(x @ w, -40, 40)
    return 1 / (1 + np.exp(-z))


def load_data() -> pd.DataFrame:
    rows = []
    for seed, path in IN_FILES.items():
        df = pd.read_csv(path)
        df["validation_seed"] = seed
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["controller_predictive"] = (df["controller"] == "predictive_smoothed_full").astype(float)
    df["controller_fixed"] = (df["controller"] == "fixed_reference").astype(float)
    df["controller_frozen_minimax"] = (df["controller"] == "frozen_minimax_geometry_dose").astype(float)
    df["controller_geometry_dose"] = (df["controller"] == "geometry_dose_capacity_full_feedback").astype(float)
    for a, b in INTERACTIONS:
        df[f"{a}*{b}"] = df[a] * df[b]
    all_terms = sorted(set(BASE_TERMS + [x for v in THEORY_FEATURES.values() for x in v] + [f"{a}*{b}" for a, b in INTERACTIONS]))
    df[all_terms] = df[all_terms].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["chance_success_int"] = df["chance_success"].astype(int)
    return df


def evaluate_grouped(df: pd.DataFrame, features: list[str], model_name: str, group_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []
    preds = []
    if group_mode == "leave_seed_out":
        groups = [(seed, df["validation_seed"] != seed, df["validation_seed"] == seed) for seed in sorted(df["validation_seed"].unique())]
    elif group_mode == "leave_seed_family_out":
        groups = []
        for seed in sorted(df["validation_seed"].unique()):
            for family in sorted(df["model_family"].unique()):
                test = (df["validation_seed"] == seed) & (df["model_family"] == family)
                train = ~test
                groups.append((f"{seed}/{family}", train, test))
    elif group_mode == "leave_seed_controller_out":
        groups = []
        for seed in sorted(df["validation_seed"].unique()):
            for controller in sorted(df["controller"].unique()):
                test = (df["validation_seed"] == seed) & (df["controller"] == controller)
                train = ~test
                groups.append((f"{seed}/{controller}", train, test))
    else:
        raise ValueError(group_mode)
    for group, train, test in groups:
        if test.sum() == 0:
            continue
        w, mu, sd = logistic_fit(df.loc[train, features].to_numpy(float), df.loc[train, "chance_success_int"].to_numpy(int))
        p = logistic_predict(df.loc[test, features].to_numpy(float), w, mu, sd)
        y = df.loc[test, "chance_success_int"].to_numpy(int)
        details.append(
            {
                "model": model_name,
                "group_mode": group_mode,
                "heldout_group": group,
                "n": int(test.sum()),
                "auc": auc_score(y, p),
                "success_rate": float(y.mean()),
                "n_features": len(features),
                "features": " + ".join(features),
            }
        )
        pred = df.loc[test, ["validation_seed", "model_family", "controller", "chance_success_int"]].copy()
        pred["heldout_group"] = group
        pred["model"] = model_name
        pred["predicted_probability"] = p
        preds.append(pred)
    return pd.DataFrame(details), pd.concat(preds, ignore_index=True)


def compare_theories(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    pred_rows = []
    for model_name, features in THEORY_FEATURES.items():
        for mode in ["leave_seed_out", "leave_seed_family_out", "leave_seed_controller_out"]:
            detail, pred = evaluate_grouped(df, features, model_name, mode)
            detail_rows.append(detail)
            pred_rows.append(pred)
    return pd.concat(detail_rows, ignore_index=True), pd.concat(pred_rows, ignore_index=True)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    return details.groupby(["model", "group_mode"], as_index=False).agg(
        median_auc=("auc", "median"),
        min_auc=("auc", "min"),
        mean_auc=("auc", "mean"),
        n_groups=("auc", "size"),
        n_features=("n_features", "first"),
    ).sort_values(["group_mode", "min_auc", "median_auc"], ascending=[True, False, False])


def make_figure(summary_df: pd.DataFrame, details: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    seed = summary_df[summary_df["group_mode"] == "leave_seed_out"].sort_values("min_auc")
    ax.barh(seed["model"], seed["min_auc"], color="#2f7f4f")
    ax.axvline(0.70, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Minimum AUC")
    ax.set_title("A. Leave-seed-out robustness")

    ax = axes[0, 1]
    fam = summary_df[summary_df["group_mode"] == "leave_seed_family_out"].sort_values("min_auc")
    ax.barh(fam["model"], fam["min_auc"], color="#6c5b7b")
    ax.axvline(0.65, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Minimum AUC")
    ax.set_title("B. Leave-seed-family-out robustness")

    ax = axes[1, 0]
    ctrl = summary_df[summary_df["group_mode"] == "leave_seed_controller_out"].sort_values("min_auc")
    ax.barh(ctrl["model"], ctrl["min_auc"], color="#2878b5")
    ax.axvline(0.65, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Minimum AUC")
    ax.set_title("C. Leave-seed-controller-out robustness")

    ax = axes[1, 1]
    best_name = seed.sort_values(["min_auc", "median_auc"], ascending=False)["model"].iloc[0]
    best = details[(details["model"] == best_name) & (details["group_mode"] == "leave_seed_family_out")]
    ax.hist(best["auc"].dropna(), bins=np.linspace(0.5, 1.0, 12), color="#2f7f4f", alpha=0.8)
    ax.axvline(best["auc"].median(), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("AUC")
    ax.set_ylabel("Held-out groups")
    ax.set_title(f"D. {best_name} group AUC distribution")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    details, preds = compare_theories(df)
    summary_df = summarize(details)
    details.to_csv(OUT_DATA / "theory_class_grouped_validation_detail.csv", index=False)
    preds.to_csv(OUT_DATA / "theory_class_predictions.csv", index=False)
    summary_df.to_csv(OUT_DATA / "theory_class_grouped_validation_summary.csv", index=False)
    metadata = {
        "input_files": {k: str(v) for k, v in IN_FILES.items()},
        "theory_features": THEORY_FEATURES,
        "validation_modes": ["leave_seed_out", "leave_seed_family_out", "leave_seed_controller_out"],
        "interpretation": "Grouped validation compares interpretable theory classes rather than optimizing one equation on one split.",
    }
    (OUT_DATA / "theory_class_comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figure(summary_df, details, OUT_FIG / "figure_10_theory_class_comparison.png")

    seed_summary = summary_df[summary_df["group_mode"] == "leave_seed_out"].sort_values(["min_auc", "median_auc"], ascending=False)
    family_summary = summary_df[summary_df["group_mode"] == "leave_seed_family_out"].sort_values(["min_auc", "median_auc"], ascending=False)
    controller_summary = summary_df[summary_df["group_mode"] == "leave_seed_controller_out"].sort_values(["min_auc", "median_auc"], ascending=False)
    best = seed_summary.iloc[0]
    decision = (
        "CONTINUE: a richer theory class improves grouped AUC and remains interpretable."
        if best["min_auc"] >= 0.72 and family_summary.iloc[0]["min_auc"] >= 0.65
        else "PIVOT: richer theory classes improve some splits but are not yet robust enough for a high-impact law claim."
    )
    lines = [
        "# Step 10 Theory-Class Comparison",
        "",
        "Question: can richer but still interpretable theory classes improve AUC without turning into a black-box leaderboard?",
        "",
        "## Summary",
        "",
        "```",
        summary_df.round(3).to_string(index=False),
        "```",
        "",
        "## Best Models by Validation Mode",
        "",
        f"Best leave-seed-out: {seed_summary.iloc[0]['model']} with min AUC {seed_summary.iloc[0]['min_auc']:.3f}.",
        f"Best leave-seed-family-out: {family_summary.iloc[0]['model']} with min AUC {family_summary.iloc[0]['min_auc']:.3f}.",
        f"Best leave-seed-controller-out: {controller_summary.iloc[0]['model']} with min AUC {controller_summary.iloc[0]['min_auc']:.3f}.",
        "",
        "## Step-Gate Decision",
        "",
        decision,
        "",
        "Interpretation: acceptable high-impact methods here are grouped validation, interpretable interaction classes, paired uplift, and external transfer tests. We should avoid adding complexity unless it improves grouped validation and explains a mechanism.",
    ]
    (OUT_NOTES / "step10_theory_class_comparison_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote theory-class comparison outputs")
    print(summary_df.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
