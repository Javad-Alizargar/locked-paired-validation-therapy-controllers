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
IN_DATA = NCS / "data" / "frozen_policy_validation" / "frozen_policy_validation_reduced_features.csv"
OUT_DATA = NCS / "data" / "symbolic_boundary"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"
EPS = 1e-12


BASE_FEATURES = [
    "log_drive_number",
    "log_reaction_number",
    "log_reaction_drive_balance",
    "log_recovery_time_s",
    "log_thermal_diffusion_time_s",
    "log_perfusion_diffusion_ratio",
    "log_radius_recovery_number",
    "log_dose_capacity_number",
    "dose_deficit",
    "delay_recovery_ratio",
    "delay_diffusion_ratio",
    "noise_fraction",
    "observation_quality",
    "duty_proxy",
]

INTERACTION_PAIRS = [
    ("log_dose_capacity_number", "delay_diffusion_ratio"),
    ("log_dose_capacity_number", "log_perfusion_diffusion_ratio"),
    ("dose_deficit", "delay_diffusion_ratio"),
    ("dose_deficit", "log_recovery_time_s"),
    ("log_reaction_drive_balance", "delay_diffusion_ratio"),
    ("observation_quality", "delay_diffusion_ratio"),
    ("duty_proxy", "log_dose_capacity_number"),
    ("duty_proxy", "log_perfusion_diffusion_ratio"),
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


def logistic_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, l2: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0) + EPS
    xtr = np.nan_to_num((x_train - mu) / sd)
    xte = np.nan_to_num((x_test - mu) / sd)
    xtr = np.c_[np.ones(len(xtr)), xtr]
    xte = np.c_[np.ones(len(xte)), xte]
    y = y_train.astype(float)

    n_pos = max(y.sum(), 1.0)
    n_neg = max((1 - y).sum(), 1.0)
    sample_weights = np.where(y > 0, len(y) / (2 * n_pos), len(y) / (2 * n_neg))

    def loss(w: np.ndarray) -> float:
        z = np.clip(xtr @ w, -40, 40)
        return float(np.sum(sample_weights * (np.logaddexp(0, z) - y * z)) + l2 * np.sum(w[1:] ** 2))

    def grad(w: np.ndarray) -> np.ndarray:
        z = np.clip(xtr @ w, -40, 40)
        p = 1 / (1 + np.exp(-z))
        g = xtr.T @ (sample_weights * (p - y))
        g[1:] += 2 * l2 * w[1:]
        return g

    result = minimize(loss, np.zeros(xtr.shape[1]), jac=grad, method="L-BFGS-B", options={"maxiter": 250})
    z = np.clip(xte @ result.x, -40, 40)
    return 1 / (1 + np.exp(-z)), result.x, mu, sd


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for a, b in INTERACTION_PAIRS:
        out[f"{a}*{b}"] = out[a] * out[b]
    return out


def clean_data() -> pd.DataFrame:
    df = pd.read_csv(IN_DATA)
    df = add_interactions(df)
    feature_cols = BASE_FEATURES + [f"{a}*{b}" for a, b in INTERACTION_PAIRS]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["chance_success_int"] = df["chance_success"].astype(int)
    return df


def evaluate_features(df: pd.DataFrame, features: list[str], model_name: str = "sparse_logistic") -> dict:
    rows = []
    for family in sorted(df["model_family"].unique()):
        train = df["model_family"] != family
        test = df["model_family"] == family
        pred, _, _, _ = logistic_fit_predict(
            df.loc[train, features].to_numpy(float),
            df.loc[train, "chance_success_int"].to_numpy(int),
            df.loc[test, features].to_numpy(float),
        )
        y = df.loc[test, "chance_success_int"].to_numpy(int)
        rows.append(
            {
                "model": model_name,
                "heldout_model_family": family,
                "n_features": len(features),
                "features": " + ".join(features),
                "auc": auc_score(y, pred),
                "success_rate": float(y.mean()),
            }
        )
    pred_all, weights, mu, sd = logistic_fit_predict(
        df[features].to_numpy(float),
        df["chance_success_int"].to_numpy(int),
        df[features].to_numpy(float),
    )
    y_all = df["chance_success_int"].to_numpy(int)
    rows.append(
        {
            "model": model_name,
            "heldout_model_family": "apparent_all",
            "n_features": len(features),
            "features": " + ".join(features),
            "auc": auc_score(y_all, pred_all),
            "success_rate": float(y_all.mean()),
        }
    )
    result = pd.DataFrame(rows)
    holdout = result[result["heldout_model_family"] != "apparent_all"]
    return {
        "model": model_name,
        "features": features,
        "median_holdout_auc": float(holdout["auc"].median()),
        "min_holdout_auc": float(holdout["auc"].min()),
        "apparent_auc": float(result.loc[result["heldout_model_family"] == "apparent_all", "auc"].iloc[0]),
        "detail": result,
        "weights": weights,
        "mu": mu,
        "sd": sd,
    }


def search_sparse_boundaries(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    all_features = BASE_FEATURES + [f"{a}*{b}" for a, b in INTERACTION_PAIRS]
    candidates = []
    screened = []
    for feature in all_features:
        res = evaluate_features(df, [feature])
        screened.append(
            {
                "feature": feature,
                "median_holdout_auc": res["median_holdout_auc"],
                "min_holdout_auc": res["min_holdout_auc"],
                "score": res["min_holdout_auc"] + 0.25 * res["median_holdout_auc"],
            }
        )
    screened_df = pd.DataFrame(screened).sort_values(["score", "min_holdout_auc"], ascending=False)
    shortlist = screened_df["feature"].head(12).tolist()
    required = ["log_dose_capacity_number", "delay_diffusion_ratio", "log_perfusion_diffusion_ratio", "observation_quality"]
    shortlist = list(dict.fromkeys(required + shortlist))
    for k in [2, 3, 4]:
        for features in itertools.combinations(shortlist, k):
            res = evaluate_features(df, list(features))
            score = res["min_holdout_auc"] + 0.25 * res["median_holdout_auc"] - 0.01 * k
            candidates.append(
                {
                    "features": " + ".join(features),
                    "n_features": k,
                    "median_holdout_auc": res["median_holdout_auc"],
                    "min_holdout_auc": res["min_holdout_auc"],
                    "apparent_auc": res["apparent_auc"],
                    "selection_score": score,
                }
            )
    table = pd.DataFrame(candidates).sort_values(
        ["selection_score", "min_holdout_auc", "median_holdout_auc"], ascending=False
    )
    table["screened_shortlist"] = ", ".join(shortlist)
    best_features = table.iloc[0]["features"].split(" + ")
    best = evaluate_features(df, best_features, "best_sparse_boundary")
    return table, best


def fit_black_boxes(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    manual_full = evaluate_features(df, features, "full_logistic_manual")["detail"]
    manual_full = manual_full[manual_full["heldout_model_family"] != "apparent_all"].copy()
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        manual_full["sklearn_status"] = f"sklearn unavailable: {exc}"
        return manual_full

    models = {
        "full_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=8, random_state=20260713, class_weight="balanced_subsample"),
        "gradient_boosting": GradientBoostingClassifier(random_state=20260713, max_depth=2, learning_rate=0.04, n_estimators=220),
    }
    rows = manual_full.to_dict("records")
    for name, model in models.items():
        for family in sorted(df["model_family"].unique()):
            train = df["model_family"] != family
            test = df["model_family"] == family
            model.fit(df.loc[train, features], df.loc[train, "chance_success_int"])
            pred = model.predict_proba(df.loc[test, features])[:, 1]
            y = df.loc[test, "chance_success_int"].to_numpy(int)
            rows.append(
                {
                    "model": name,
                    "heldout_model_family": family,
                    "auc": auc_score(y, pred),
                    "success_rate": float(y.mean()),
                    "n_features": len(features),
                    "features": "all_dimensionless",
                }
            )
    return pd.DataFrame(rows)


def equation_text(best: dict) -> str:
    weights = best["weights"]
    features = best["features"]
    mu = best["mu"]
    sd = best["sd"]
    terms = [f"{weights[0]:+.3f}"]
    for coef, name, m, s in zip(weights[1:], features, mu, sd):
        terms.append(f"{coef:+.3f}*z({name}; mean={m:.3g}, sd={s:.3g})")
    return "logit(P_success) = " + " ".join(terms)


def make_figure(best_detail: pd.DataFrame, blackbox: pd.DataFrame, df: pd.DataFrame, best: dict, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    combined = pd.concat([best_detail[best_detail["heldout_model_family"] != "apparent_all"], blackbox], ignore_index=True)
    pivot = combined.pivot_table(index="heldout_model_family", columns="model", values="auc")
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.75, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out AUC")
    ax.set_title("A. Sparse law vs black-box models")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[0, 1]
    features = best["features"]
    pred, weights, mu, sd = logistic_fit_predict(
        df[features].to_numpy(float),
        df["chance_success_int"].to_numpy(int),
        df[features].to_numpy(float),
    )
    df_plot = df.copy()
    df_plot["boundary_score"] = np.log(pred / (1 - pred + EPS))
    ok = df_plot["chance_success_int"] == 1
    ax.scatter(df_plot.loc[~ok, "boundary_score"], df_plot.loc[~ok, "dose_capacity_number"], s=10, alpha=0.22, color="black", label="failure")
    ax.scatter(df_plot.loc[ok, "boundary_score"], df_plot.loc[ok, "dose_capacity_number"], s=10, alpha=0.28, color="#2ca25f", label="success")
    ax.set_yscale("log")
    ax.axvline(0, color="#c44e52", linestyle="--", linewidth=1)
    ax.set_xlabel("Sparse boundary score")
    ax.set_ylabel("Dose-capacity number")
    ax.set_title("B. Boundary separation")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    fam = df_plot.groupby(["model_family", pd.cut(df_plot["boundary_score"], bins=np.linspace(df_plot["boundary_score"].quantile(0.02), df_plot["boundary_score"].quantile(0.98), 9))], observed=False)["chance_success_int"].mean().reset_index()
    for family, g in fam.groupby("model_family"):
        mids = [interval.mid for interval in g["boundary_score"]]
        ax.plot(mids, g["chance_success_int"], marker="o", label=family)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sparse boundary score bin")
    ax.set_ylabel("Observed success probability")
    ax.set_title("C. Calibration by family")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    coefs = pd.Series(weights[1:], index=features).sort_values()
    ax.barh(coefs.index, coefs.values, color=np.where(coefs.values > 0, "#2f7f4f", "#c44e52"))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("D. Compact equation terms")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    df = clean_data()
    search_table, best = search_sparse_boundaries(df)
    search_table.to_csv(OUT_DATA / "sparse_boundary_search.csv", index=False)
    best["detail"].to_csv(OUT_DATA / "best_sparse_boundary_holdout.csv", index=False)
    all_dimensionless = BASE_FEATURES + [f"{a}*{b}" for a, b in INTERACTION_PAIRS]
    blackbox = fit_black_boxes(df, all_dimensionless)
    blackbox.to_csv(OUT_DATA / "blackbox_boundary_comparison.csv", index=False)
    eq = equation_text(best)
    metadata = {
        "input": str(IN_DATA),
        "base_features": BASE_FEATURES,
        "interaction_pairs": INTERACTION_PAIRS,
        "best_features": best["features"],
        "equation": eq,
    }
    (OUT_DATA / "symbolic_boundary_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figure(best["detail"], blackbox, df, best, OUT_FIG / "figure_5_symbolic_boundary_discovery.png")

    bb_summary = blackbox.groupby("model", as_index=False).agg(
        median_holdout_auc=("auc", "median"),
        min_holdout_auc=("auc", "min"),
    ).sort_values(["min_holdout_auc", "median_holdout_auc"], ascending=False)
    sparse_holdout = best["detail"][best["detail"]["heldout_model_family"] != "apparent_all"]
    best_sparse_min = float(sparse_holdout["auc"].min())
    best_sparse_median = float(sparse_holdout["auc"].median())
    best_black_min = float(bb_summary["min_holdout_auc"].max()) if not bb_summary.empty and "min_holdout_auc" in bb_summary else float("nan")
    verdict = (
        "CONTINUE: compact boundary is competitive with black-box models."
        if best_sparse_min >= 0.72 and best_sparse_min >= best_black_min - 0.05
        else "PIVOT: compact boundary is informative but not yet strong enough as a standalone law."
    )
    lines = [
        "# Step 5 Symbolic Boundary Discovery",
        "",
        f"Rows analyzed: {len(df)} from the independent frozen-policy validation atlas.",
        "",
        "## Best Compact Boundary",
        "",
        f"Features: {', '.join(best['features'])}.",
        "",
        "```",
        eq,
        "```",
        "",
        "## Sparse Boundary Hold-Out AUC",
        "",
        "```",
        best["detail"].round(3).to_string(index=False),
        "```",
        "",
        "## Black-Box Comparator AUC",
        "",
        "```",
        bb_summary.round(3).to_string(index=False),
        "```",
        "",
        "## Step-Gate Decision",
        "",
        f"Sparse boundary median held-out AUC: {best_sparse_median:.3f}.",
        f"Sparse boundary minimum held-out AUC: {best_sparse_min:.3f}.",
        f"Best black-box minimum held-out AUC: {best_black_min:.3f}.",
        verdict,
        "",
        "Interpretation: this step supports a reduced controllability-boundary story only if the compact equation transfers across held-out families nearly as well as black-box models. Otherwise, the next pivot should be predictive-control recovery plus a richer boundary class.",
    ]
    (OUT_NOTES / "step5_symbolic_boundary_discovery_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote symbolic boundary discovery outputs")
    print(lines[-7:])


if __name__ == "__main__":
    main()
