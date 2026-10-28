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
OUT_DATA = NCS / "data" / "multiseed_controllability"
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
    ("log_reaction_number", "observation_quality"),
    ("log_thermal_diffusion_time_s", "observation_quality"),
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


def logistic_fit(x_train: np.ndarray, y_train: np.ndarray, l2: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    for seed_label, path in IN_FILES.items():
        df = pd.read_csv(path)
        df["validation_seed"] = seed_label
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    for a, b in INTERACTION_PAIRS:
        df[f"{a}*{b}"] = df[a] * df[b]
    features = BASE_FEATURES + [f"{a}*{b}" for a, b in INTERACTION_PAIRS]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["chance_success_int"] = df["chance_success"].astype(int)
    return df


def cross_seed_metrics(df: pd.DataFrame, features: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    pred_rows = []
    for train_seed in sorted(df["validation_seed"].unique()):
        test_seed = [s for s in sorted(df["validation_seed"].unique()) if s != train_seed][0]
        train = df["validation_seed"] == train_seed
        test = df["validation_seed"] == test_seed
        w, mu, sd = logistic_fit(df.loc[train, features].to_numpy(float), df.loc[train, "chance_success_int"].to_numpy(int))
        pred = logistic_predict(df.loc[test, features].to_numpy(float), w, mu, sd)
        test_df = df.loc[test, ["validation_seed", "model_family", "controller", "chance_success_int"]].copy()
        test_df["predicted_probability"] = pred
        test_df["model"] = model_name
        test_df["trained_on_seed"] = train_seed
        pred_rows.append(test_df)
        y = df.loc[test, "chance_success_int"].to_numpy(int)
        detail_rows.append(
            {
                "model": model_name,
                "trained_on_seed": train_seed,
                "tested_on_seed": test_seed,
                "model_family": "all",
                "auc": auc_score(y, pred),
                "success_rate": float(y.mean()),
                "n": int(test.sum()),
                "features": " + ".join(features),
            }
        )
        for family, g in test_df.groupby("model_family"):
            detail_rows.append(
                {
                    "model": model_name,
                    "trained_on_seed": train_seed,
                    "tested_on_seed": test_seed,
                    "model_family": family,
                    "auc": auc_score(g["chance_success_int"].to_numpy(int), g["predicted_probability"].to_numpy(float)),
                    "success_rate": float(g["chance_success_int"].mean()),
                    "n": len(g),
                    "features": " + ".join(features),
                }
            )
    return pd.DataFrame(detail_rows), pd.concat(pred_rows, ignore_index=True)


def score_detail(detail: pd.DataFrame, n_features: int) -> dict:
    fam = detail[detail["model_family"] != "all"]
    return {
        "median_seed_family_auc": float(fam["auc"].median()),
        "min_seed_family_auc": float(fam["auc"].min()),
        "mean_seed_family_auc": float(fam["auc"].mean()),
        "all_seed_auc": float(detail[detail["model_family"] == "all"]["auc"].mean()),
        "selection_score": float(fam["auc"].min() + 0.25 * fam["auc"].median() - 0.012 * n_features),
    }


def search_boundaries(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    all_features = BASE_FEATURES + [f"{a}*{b}" for a, b in INTERACTION_PAIRS]
    screened = []
    for feature in all_features:
        detail, _ = cross_seed_metrics(df, [feature], "single_feature")
        s = score_detail(detail, 1)
        screened.append({"feature": feature, **s})
    screened_df = pd.DataFrame(screened).sort_values(["selection_score", "min_seed_family_auc"], ascending=False)
    required = [
        "observation_quality",
        "log_reaction_number",
        "log_thermal_diffusion_time_s",
        "log_dose_capacity_number",
        "delay_diffusion_ratio",
    ]
    shortlist = list(dict.fromkeys(required + screened_df["feature"].head(12).tolist()))
    rows = []
    best = None
    for k in [2, 3, 4, 5]:
        for features in itertools.combinations(shortlist, k):
            detail, predictions = cross_seed_metrics(df, list(features), "candidate_sparse")
            s = score_detail(detail, k)
            row = {
                "features": " + ".join(features),
                "n_features": k,
                "screened_shortlist": ", ".join(shortlist),
                **s,
            }
            rows.append(row)
            if best is None or row["selection_score"] > best["row"]["selection_score"]:
                best = {"row": row, "detail": detail, "predictions": predictions, "features": list(features)}
    table = pd.DataFrame(rows).sort_values(["selection_score", "min_seed_family_auc", "median_seed_family_auc"], ascending=False)
    return table, best


def controllability_classes(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = predictions.copy()
    q = out["predicted_probability"].quantile([0.33, 0.67]).to_numpy()
    out["controllability_class"] = pd.cut(
        out["predicted_probability"],
        bins=[-np.inf, q[0], q[1], np.inf],
        labels=["low", "transitional", "high"],
    )
    classes = out.groupby(["validation_seed", "model_family", "controllability_class"], observed=False).agg(
        n=("chance_success_int", "size"),
        observed_success=("chance_success_int", "mean"),
        mean_predicted=("predicted_probability", "mean"),
    ).reset_index()
    global_classes = out.groupby("controllability_class", observed=False).agg(
        n=("chance_success_int", "size"),
        observed_success=("chance_success_int", "mean"),
        mean_predicted=("predicted_probability", "mean"),
    ).reset_index()
    return classes, global_classes


def fit_combined_equation(df: pd.DataFrame, features: list[str]) -> dict:
    w, mu, sd = logistic_fit(df[features].to_numpy(float), df["chance_success_int"].to_numpy(int))
    terms = [f"{w[0]:+.3f}"]
    for coef, name, m, s in zip(w[1:], features, mu, sd):
        terms.append(f"{coef:+.3f}*z({name}; mean={m:.3g}, sd={s:.3g})")
    return {
        "weights": w.tolist(),
        "means": mu.tolist(),
        "scales": sd.tolist(),
        "equation": "logit(P_success) = " + " ".join(terms),
    }


def make_figure(best_detail: pd.DataFrame, predictions: pd.DataFrame, classes: pd.DataFrame, global_classes: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    h = best_detail[best_detail["model_family"] != "all"].copy()
    h["group"] = h["tested_on_seed"] + " / " + h["model_family"]
    ax.barh(h["group"], h["auc"], color="#2f7f4f")
    ax.axvline(0.70, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.75, color="#c44e52", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC")
    ax.set_title("A. Cross-seed/family validation")

    ax = axes[0, 1]
    ok = predictions["chance_success_int"] == 1
    ax.scatter(predictions.loc[~ok, "predicted_probability"], predictions.loc[~ok, "model_family"], s=12, alpha=0.20, color="black", label="failure")
    ax.scatter(predictions.loc[ok, "predicted_probability"], predictions.loc[ok, "model_family"], s=12, alpha=0.26, color="#2ca25f", label="success")
    ax.set_xlabel("Out-of-seed predicted probability")
    ax.set_title("B. Out-of-seed separation")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    piv = global_classes.set_index("controllability_class")
    ax.bar(piv.index.astype(str), piv["observed_success"], color=["#c44e52", "#ddcc77", "#2f7f4f"])
    ax.plot(piv.index.astype(str), piv["mean_predicted"], color="black", marker="o", label="mean predicted")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success probability")
    ax.set_title("C. Seed-stable controllability classes")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    class_pivot = classes.pivot_table(index="model_family", columns="controllability_class", values="observed_success", aggfunc="mean", observed=False)
    im = ax.imshow(class_pivot[["low", "transitional", "high"]].to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low", "transitional", "high"], rotation=20)
    ax.set_yticks(range(len(class_pivot.index)))
    ax.set_yticklabels(class_pivot.index)
    ax.set_title("D. Class success by family")
    fig.colorbar(im, ax=ax, label="Observed success")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    search_table, best = search_boundaries(df)
    search_table.to_csv(OUT_DATA / "multiseed_sparse_boundary_search.csv", index=False)
    best_detail = best["detail"]
    predictions = best["predictions"]
    best_detail.to_csv(OUT_DATA / "best_multiseed_boundary_validation.csv", index=False)
    predictions.to_csv(OUT_DATA / "best_multiseed_boundary_outofseed_predictions.csv", index=False)
    classes, global_classes = controllability_classes(predictions)
    classes.to_csv(OUT_DATA / "controllability_classes_by_seed_family.csv", index=False)
    global_classes.to_csv(OUT_DATA / "controllability_classes_global.csv", index=False)
    combined_eq = fit_combined_equation(df, best["features"])
    metadata = {
        "input_files": {k: str(v) for k, v in IN_FILES.items()},
        "features": best["features"],
        "combined_equation": combined_eq,
        "validation": "Leave-one-seed-out selection and validation; each prediction is out-of-seed.",
    }
    (OUT_DATA / "multiseed_controllability_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figure(best_detail, predictions, classes, global_classes, OUT_FIG / "figure_7_multiseed_controllability_classes.png")

    scores = score_detail(best_detail, len(best["features"]))
    decision = (
        "CONTINUE: multi-seed boundary produces seed-stable controllability classes."
        if scores["min_seed_family_auc"] >= 0.68 and global_classes["observed_success"].is_monotonic_increasing
        else "PIVOT: multi-seed boundary is not stable enough; move to predictive-control mechanisms or a richer model class."
    )
    lines = [
        "# Step 7 Multi-Seed Controllability Classes",
        "",
        f"Rows analyzed: {len(df)} from {len(IN_FILES)} independent atlas seeds.",
        "Candidate boundaries were selected using leave-one-seed-out validation from the start.",
        "",
        "## Best Seed-Stable Boundary",
        "",
        f"Features: {', '.join(best['features'])}.",
        "",
        "Combined-fit equation for interpretation only:",
        "",
        "```",
        combined_eq["equation"],
        "```",
        "",
        "## Cross-Seed Validation",
        "",
        "```",
        best_detail.round(3).to_string(index=False),
        "```",
        "",
        "## Global Controllability Classes",
        "",
        "```",
        global_classes.round(3).to_string(index=False),
        "```",
        "",
        "## Step-Gate Decision",
        "",
        f"Minimum seed-family AUC: {scores['min_seed_family_auc']:.3f}.",
        f"Median seed-family AUC: {scores['median_seed_family_auc']:.3f}.",
        decision,
    ]
    (OUT_NOTES / "step7_multiseed_controllability_classes_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote multi-seed controllability outputs")
    print(best_detail.round(3).to_string(index=False))
    print(global_classes.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
