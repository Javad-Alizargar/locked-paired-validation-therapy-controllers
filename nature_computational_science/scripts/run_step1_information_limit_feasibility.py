from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

EPS = 1e-12
RNG = np.random.default_rng(20260704)


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    keep = np.isfinite(s)
    y = y[keep]
    s = s[keep]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(vals) + 1)
    uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inv, weights=ranks)
        ranks = (sums / counts)[inv]
    rank_pos = ranks[: len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0) + EPS
    x_train = np.nan_to_num((x_train - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num((x_test - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]

    def loss(w: np.ndarray) -> float:
        z = x_train @ w
        n_pos = max(y_train.sum(), 1.0)
        n_neg = max((1 - y_train).sum(), 1.0)
        weights = np.where(y_train > 0, len(y_train) / (2 * n_pos), len(y_train) / (2 * n_neg))
        return float(np.sum(weights * (np.logaddexp(0, z) - y_train * z)) + 0.05 * np.sum(w[1:] ** 2))

    result = minimize(loss, np.zeros(x_train.shape[1]), method="BFGS")
    z = x_test @ result.x
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def brier_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return float(np.mean((y - p) ** 2))


def calibration_slope(y_true: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(pred, dtype=float), 1e-5, 1 - 1e-5)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    pred2 = logistic_fit_predict(logit, y, logit)
    lp = np.log(np.clip(pred2, 1e-5, 1 - 1e-5) / np.clip(1 - pred2, 1e-5, 1 - 1e-5))
    return float(np.polyfit(logit.ravel(), lp, 1)[0])


def load_with_uncertainty(path: str, family: str, uncertainty_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(ROOT / path)
    df["source_family"] = family
    if uncertainty_path:
        u = pd.read_csv(ROOT / uncertainty_path)
        fill_cols = ["perfusion_s", "radius_um", "photosensitizer_um", "volume_fraction"]
        df = df.merge(u, on="uncertainty_id", how="left", suffixes=("", "_draw"))
        for col in fill_cols:
            draw = f"{col}_draw"
            if draw in df:
                if col not in df:
                    df[col] = df[draw]
                else:
                    df[col] = df[col].fillna(df[draw])
                df = df.drop(columns=[draw])
    return df


def build_outcome_table() -> pd.DataFrame:
    frames = [
        load_with_uncertainty(
            "data/robust_optimization/robust_protocol_uncertainty_outcomes.csv",
            "fixed_protocol_robust",
            "data/robust_optimization/robust_uncertainty_draws.csv",
        ),
        load_with_uncertainty(
            "data/adaptive_control/adaptive_control_outcomes.csv",
            "adaptive_initial",
            "data/robust_optimization/robust_uncertainty_draws.csv",
        ),
        load_with_uncertainty(
            "data/final_adaptive_robustness/final_adaptive_robustness_outcomes.csv",
            "adaptive_locked",
            "data/final_adaptive_robustness/final_uncertainty_draws.csv",
        ),
        load_with_uncertainty(
            "data/feedback_ablation/feedback_ablation_outcomes.csv",
            "feedback_ablation",
            "data/feedback_ablation/feedback_ablation_uncertainty_draws.csv",
        ),
        load_with_uncertainty(
            "data/predictive_delay_control/predictive_delay_outcomes.csv",
            "predictive_delay",
            "data/predictive_delay_control/predictive_delay_uncertainty_draws.csv",
        ),
    ]
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "scenario" not in df:
        df["scenario"] = "nominal"
    df["scenario"] = df["scenario"].fillna("nominal")
    df["noise_fraction"] = df.get("noise_fraction", 0.0).fillna(0.0)
    df["delay_s"] = df.get("delay_s", 0.0).fillna(0.0)
    df["chance_success"] = df["chance_success"].astype(bool).astype(int)
    return df


def trace_information_table() -> pd.DataFrame:
    trace_specs = [
        ("data/adaptive_control/adaptive_control_traces.csv", "adaptive_initial"),
        ("data/final_adaptive_robustness/final_adaptive_robustness_traces.csv", "adaptive_locked"),
        ("data/feedback_ablation/feedback_ablation_traces.csv", "feedback_ablation"),
        ("data/predictive_delay_control/predictive_delay_traces.csv", "predictive_delay"),
    ]
    rows: list[dict[str, float | str | int]] = []
    for rel, family in trace_specs:
        path = ROOT / rel
        if not path.exists():
            continue
        tr = pd.read_csv(path)
        tr["source_family"] = family
        if "scenario" not in tr:
            tr["scenario"] = "nominal"
        group_cols = ["source_family", "controller", "scenario", "uncertainty_id"]
        for key, g in tr.groupby(group_cols, dropna=False):
            family_i, controller, scenario, uncertainty_id = key
            g = g.sort_values("time_s")
            state_cols = [c for c in ["max_temperature_c", "mean_oxygen_um", "ros_threshold_fraction"] if c in g]
            state = g[state_cols].to_numpy(dtype=float)
            if len(g) < 3 or state.size == 0:
                continue
            scaled = (state - np.nanmean(state, axis=0)) / (np.nanstd(state, axis=0) + EPS)
            step_change = np.sqrt(np.nanmean(np.diff(scaled, axis=0) ** 2, axis=1))
            state_volatility = float(np.nanmedian(step_change))
            command = g["intensity_w_cm2"].to_numpy(dtype=float)
            command_variability = float(np.nanstd(command) / (np.nanmean(command) + EPS))
            rows.append(
                {
                    "source_family": family_i,
                    "controller": controller,
                    "scenario": scenario,
                    "uncertainty_id": uncertainty_id,
                    "state_volatility": state_volatility,
                    "command_variability": command_variability,
                    "trace_points": len(g),
                    "trace_dt_median_s": float(np.nanmedian(np.diff(g["time_s"].to_numpy(dtype=float)))),
                }
            )
    return pd.DataFrame(rows)


def add_reduced_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in [
        ("perfusion_s", 0.009),
        ("radius_um", 450.0),
        ("photosensitizer_um", 1.0),
        ("volume_fraction", 0.003),
        ("protocol_duration_s", 3000.0),
        ("delivered_energy_j_cm2", 0.0),
        ("light_on_time_s", 0.0),
        ("mean_intensity_w_cm2", 0.0),
        ("max_commanded_intensity_w_cm2", 0.0),
    ]:
        if col not in out:
            out[col] = default
        out[col] = out[col].fillna(default)

    out["duty_proxy"] = np.clip(out["light_on_time_s"] / (out["protocol_duration_s"] + EPS), 0, 1)
    out["average_intensity_w_cm2"] = out["delivered_energy_j_cm2"] / (out["protocol_duration_s"] + EPS)
    out["recovery_time_s"] = 1 / (out["perfusion_s"] + EPS)
    out["drive_number"] = (out["average_intensity_w_cm2"] * out["volume_fraction"] * out["radius_um"] ** 2) / (
        out["perfusion_s"] + EPS
    )
    out["reaction_number"] = (out["average_intensity_w_cm2"] * out["photosensitizer_um"]) / (out["perfusion_s"] + EPS)
    out["energy_loading_number"] = (out["delivered_energy_j_cm2"] * out["volume_fraction"]) / (
        out["radius_um"] * out["perfusion_s"] + EPS
    )
    out["reaction_drive_balance"] = out["reaction_number"] / (out["drive_number"] + EPS)
    out["delay_recovery_ratio"] = out["delay_s"] / (out["recovery_time_s"] + EPS)
    out["delay_protocol_ratio"] = out["delay_s"] / (out["protocol_duration_s"] + EPS)
    volatility = out.get("state_volatility", pd.Series(0.25, index=out.index)).fillna(0.25)
    delayed_distortion = volatility * out["delay_s"] / (out.get("trace_dt_median_s", pd.Series(100.0, index=out.index)).fillna(100.0) + EPS)
    noise_distortion = out["noise_fraction"].fillna(0.0) * 2.0
    out["observation_distortion_proxy"] = delayed_distortion + noise_distortion
    out["information_proxy"] = np.exp(-0.5 * out["observation_distortion_proxy"] ** 2) / (1 + out["noise_fraction"] ** 2)
    out.loc[out["source_family"].str.contains("fixed", na=False), "information_proxy"] = 0.0
    out["log_drive_number"] = np.log10(out["drive_number"] + EPS)
    out["log_reaction_number"] = np.log10(out["reaction_number"] + EPS)
    out["log_energy_loading_number"] = np.log10(out["energy_loading_number"] + EPS)
    out["log_reaction_drive_balance"] = np.log10(out["reaction_drive_balance"] + EPS)
    out["log_recovery_time_s"] = np.log10(out["recovery_time_s"] + EPS)
    return out


def validation_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    x = df[features].to_numpy(dtype=float)
    y = df["chance_success"].to_numpy(dtype=int)
    pred_all = logistic_fit_predict(x, y, x)
    rows.append(
        {
            "heldout_family": "apparent_all_data",
            "n_train": len(df),
            "n_test": len(df),
            "success_rate_test": float(np.mean(y)),
            "auc": auc_score(y, pred_all),
            "brier": brier_score(y, pred_all),
            "calibration_slope": calibration_slope(y, pred_all),
        }
    )
    for family in sorted(df["source_family"].unique()):
        train = df["source_family"] != family
        test = df["source_family"] == family
        if train.sum() < 20 or test.sum() < 20:
            continue
        pred = logistic_fit_predict(df.loc[train, features].to_numpy(float), df.loc[train, "chance_success"].to_numpy(int), df.loc[test, features].to_numpy(float))
        y_test = df.loc[test, "chance_success"].to_numpy(int)
        rows.append(
            {
                "heldout_family": family,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "success_rate_test": float(np.mean(y_test)),
                "auc": auc_score(y_test, pred),
                "brier": brier_score(y_test, pred),
                "calibration_slope": calibration_slope(y_test, pred),
            }
        )
    return pd.DataFrame(rows)


def information_summary(df: pd.DataFrame) -> pd.DataFrame:
    is_fixed_controller = df["controller"].fillna("").str.startswith("fixed")
    dynamic = df[(~df["source_family"].str.contains("fixed", na=False)) & (~is_fixed_controller)].copy()
    rows = []
    for cols in [["source_family"], ["controller"], ["source_family", "scenario"], ["delay_s", "noise_fraction"]]:
        grouped = dynamic.groupby(cols, dropna=False)
        for key, g in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            rec = {col: val for col, val in zip(cols, key)}
            rec.update(
                {
                    "n": len(g),
                    "success_probability": float(g["chance_success"].mean()),
                    "mean_information_proxy": float(g["information_proxy"].mean()),
                    "median_information_proxy": float(g["information_proxy"].median()),
                    "mean_observation_distortion_proxy": float(g["observation_distortion_proxy"].mean()),
                    "median_delay_s": float(g["delay_s"].median()),
                    "median_noise_fraction": float(g["noise_fraction"].median()),
                }
            )
            rows.append(rec)
    rho, p = spearmanr(dynamic["information_proxy"], dynamic["chance_success"])
    rows.append(
        {
            "source_family": "all_dynamic_spearman",
            "n": len(dynamic),
            "success_probability": float(dynamic["chance_success"].mean()),
            "mean_information_proxy": float(dynamic["information_proxy"].mean()),
            "spearman_rho_information_success": float(rho),
            "spearman_p_information_success": float(p),
        }
    )
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, val: pd.DataFrame, info: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)

    is_fixed_controller = df["controller"].fillna("").str.startswith("fixed")
    dynamic = df[(~df["source_family"].str.contains("fixed", na=False)) & (~is_fixed_controller)].copy()
    dynamic["info_bin"] = pd.qcut(dynamic["information_proxy"].rank(method="first"), q=6, duplicates="drop")
    binned = dynamic.groupby("info_bin", observed=True).agg(
        information_proxy=("information_proxy", "mean"),
        success=("chance_success", "mean"),
        n=("chance_success", "size"),
    )
    ax = axes[0, 0]
    ax.plot(binned["information_proxy"], binned["success"], marker="o", color="#1f6f8b")
    for _, row in binned.iterrows():
        ax.text(row["information_proxy"], row["success"] + 0.025, f"n={int(row['n'])}", ha="center", fontsize=7)
    ax.set_xlabel("Observation information proxy")
    ax.set_ylabel("Success probability")
    ax.set_title("A. Existing data do not yet collapse by information proxy")
    ax.set_ylim(-0.05, 1.05)

    ax = axes[0, 1]
    plot_val = val[val["heldout_family"] != "apparent_all_data"].copy()
    ax.barh(plot_val["heldout_family"], plot_val["auc"], color="#7a5195")
    ax.axvline(0.80, color="black", linestyle="--", linewidth=1, label="Step-1 gate")
    ax.axvline(0.85, color="#cc503e", linestyle=":", linewidth=1, label="NCS target")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Held-out AUC")
    ax.set_title("B. Family holdout tests")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    fail = df["chance_success"] == 0
    ok = df["chance_success"] == 1
    ax.scatter(df.loc[fail, "log_drive_number"], df.loc[fail, "log_reaction_drive_balance"], s=10, alpha=0.18, color="#444444", label="failure")
    ax.scatter(df.loc[ok, "log_drive_number"], df.loc[ok, "log_reaction_drive_balance"], s=12, alpha=0.35, color="#2ca25f", label="success")
    ax.set_xlabel("log10 drive number")
    ax.set_ylabel("log10 reaction/drive balance")
    ax.set_title("C. Reduced variables separate phase-window outcomes")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    delay_summary = dynamic.groupby(["delay_s", "noise_fraction"], dropna=False).agg(
        success=("chance_success", "mean"),
        info=("information_proxy", "mean"),
        n=("chance_success", "size"),
    ).reset_index()
    sc = ax.scatter(delay_summary["delay_s"], delay_summary["success"], c=delay_summary["info"], s=30 + delay_summary["n"] / 2, cmap="viridis", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Success probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("D. Delay/noise regimes define information loss")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Information proxy")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_verdict(df: pd.DataFrame, val: pd.DataFrame, info: pd.DataFrame, path: Path) -> None:
    holdout = val[val["heldout_family"] != "apparent_all_data"]
    median_auc = float(holdout["auc"].median())
    min_auc = float(holdout["auc"].min())
    apparent = float(val.loc[val["heldout_family"] == "apparent_all_data", "auc"].iloc[0])
    spearman_row = info[info.get("source_family", "") == "all_dynamic_spearman"]
    rho = float(spearman_row["spearman_rho_information_success"].iloc[0]) if len(spearman_row) else float("nan")
    p = float(spearman_row["spearman_p_information_success"].iloc[0]) if len(spearman_row) else float("nan")
    pass_gate = median_auc >= 0.80 and rho > 0 and p < 0.01
    lines = [
        "# Step 1 Feasibility Result",
        "",
        f"Rows analyzed: {len(df)}.",
        f"Overall success rate: {df['chance_success'].mean():.3f}.",
        f"Apparent all-data AUC: {apparent:.3f}.",
        f"Median family-holdout AUC: {median_auc:.3f}.",
        f"Minimum family-holdout AUC: {min_auc:.3f}.",
        f"Spearman information-success association among dynamic controllers: rho = {rho:.3f}, p = {p:.3g}.",
        "",
        "## Verdict",
        "",
        "PASS for Step 1 feasibility." if pass_gate else "FAIL or BORDERLINE for Step 1 feasibility.",
        "",
        "## Interpretation",
        "",
    ]
    if pass_gate:
        lines += [
            "Existing simulations contain a measurable reduced information-control signal. This is sufficient to justify building a larger simulation atlas.",
            "This does not yet justify a Nature Computational Science submission. It justifies the next analysis phase.",
        ]
    else:
        lines += [
            "The existing simulations do not yet show a strong enough reduced information-control signal for a Nature Computational Science manuscript.",
            "Before building a broad manuscript, improve the simulation design or narrow the claim.",
        ]
    lines += [
        "",
        "## Files generated",
        "",
        "- `nature_computational_science/data/reduced_control_features.csv`",
        "- `nature_computational_science/data/information_limit_summary.csv`",
        "- `nature_computational_science/data/information_limit_validation.csv`",
        "- `nature_computational_science/figures/figure_3_information_limit_feasibility.png`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)

    outcomes = build_outcome_table()
    traces = trace_information_table()
    merge_cols = ["source_family", "controller", "scenario", "uncertainty_id"]
    df = outcomes.merge(traces, on=merge_cols, how="left")
    df = add_reduced_features(df)

    features = [
        "log_drive_number",
        "log_reaction_number",
        "log_energy_loading_number",
        "log_reaction_drive_balance",
        "log_recovery_time_s",
        "duty_proxy",
        "delay_protocol_ratio",
        "noise_fraction",
        "information_proxy",
        "command_variability",
    ]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    val = validation_table(df, features)
    info = information_summary(df)

    df.to_csv(OUT_DATA / "reduced_control_features.csv", index=False)
    info.to_csv(OUT_DATA / "information_limit_summary.csv", index=False)
    val.to_csv(OUT_DATA / "information_limit_validation.csv", index=False)
    make_figure(df, val, info, OUT_FIG / "figure_3_information_limit_feasibility.png")
    write_verdict(df, val, info, OUT_NOTES / "step1_feasibility_result.md")

    print("Wrote", OUT_DATA / "reduced_control_features.csv")
    print("Wrote", OUT_DATA / "information_limit_summary.csv")
    print("Wrote", OUT_DATA / "information_limit_validation.csv")
    print("Wrote", OUT_FIG / "figure_3_information_limit_feasibility.png")
    print("Wrote", OUT_NOTES / "step1_feasibility_result.md")
    print(val.to_string(index=False))


if __name__ == "__main__":
    main()
