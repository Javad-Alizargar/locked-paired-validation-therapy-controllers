"""
PeerJ Computer Science re-analysis of the locked validation atlas.

Design principles
-----------------
1. Every reported number is computed here from committed trial-level data.
2. Inference respects the cluster structure (384 clusters = seed × family × draw,
   12 scenarios each, balanced).
3. Multiplicity is controlled with Holm over locked families.
4. Effect sizes are reported with 95% cluster-bootstrap intervals and MDE at 80% power.
5. Strata are described for what they are: cross-fitted *predicted baseline success*
   terciles from a frozen equation trained on different seeds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config

B_BOOT = 10_000
B_SIGN = 20_000
SEED = 20261028
Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143

STRATA = ["low", "transitional", "high"]
STRATUM_LABEL = {
    "all": "all",
    "low": "predicted-low",
    "transitional": "predicted-intermediate",
    "high": "predicted-high",
}

# Mechanism deltas present in the paired file. Each is a per-pair indicator difference.
MECH_COLS = {
    "thermal_failure": "thermal_failure_rate_delta",
    "ros_failure": "ros_failure_rate_delta",
    "hypoxic_failure": "hypoxic_failure_rate_delta",
    "overheat_failure": "overheat_failure_rate_delta",
    "energy_j_cm2": "delivered_energy_j_cm2_delta",
    "max_temperature_c": "max_temperature_c_delta",
    "warm_fraction": "warm_fraction_delta",
    "ros_threshold_fraction": "ros_threshold_fraction_delta",
}

ENDPOINT_DEFINITIONS = [
    ("warm", "warm_fraction ≥ 0.50", "fraction of the protocol window with 43 ≤ T ≤ 50 °C"),
    ("ros", "ros_threshold_fraction ≥ 0.50", "fraction of the window with ROS dose at or above the singlet-oxygen threshold"),
    ("hypoxia_ok", "hypoxic_failure_fraction ≤ 0.35", "fraction of the window with O2 < 5 µM AND ROS below threshold"),
    ("safe_heat", "overheat_fraction ≤ 0.05 and max_T ≤ 55 °C", "no thermal injury"),
    ("success", "warm AND ros AND hypoxia_ok AND safe_heat", "all four criteria simultaneously"),
    ("thermal_failure", "NOT warm AND safe_heat", "insufficient heating without thermal injury"),
    ("ros_failure", "NOT ros", "ROS dose below threshold"),
    ("overheat_failure", "NOT safe_heat", "thermal injury"),
]


# ----------------------------------------------------------------------------- io

def load_paired(base: Path) -> pd.DataFrame:
    p = base / "nature_computational_science" / "data" / "controller_pareto_synthesis" / "controller_family_locked_pairs.csv"
    df = pd.read_csv(p, low_memory=False)
    df = df.rename(columns={c: c for c in df.columns})
    # explicit contract checks
    assert df["success_uplift"].isin([-1, 0, 1]).all(), "success_uplift must be a signed indicator difference"
    assert np.array_equal(
        df["success_uplift"].to_numpy(int),
        df["to_chance_success_int"].to_numpy(int) - df["from_chance_success_int"].to_numpy(int),
    ), "success_uplift must equal to - from"
    assert np.array_equal(
        df["success_uplift"].to_numpy(int),
        df["rescued_success"].astype(int).to_numpy() - df["lost_success"].astype(int).to_numpy(),
    ), "success_uplift must equal rescued - lost"
    # verify mechanism deltas are indicator differences
    for name, flag in [
        ("thermal_failure", "thermal_failure"),
        ("ros_failure", "ros_failure"),
        ("hypoxic_failure", "hypoxic_failure"),
        ("overheat_failure", "overheat_failure"),
    ]:
        col = MECH_COLS[name]
        recomputed = df[f"to_{flag}"].astype(int) - df[f"from_{flag}"].astype(int)
        assert np.allclose(recomputed, df[col]), f"{col} is not to_flag - from_flag"
    # balanced clusters
    sizes = df.groupby(["controller_family", "cluster_id"], observed=True).size()
    assert sizes.nunique() == 1, "clusters are not balanced"
    return df


# ----------------------------------------------------------------------------- inference

@dataclass
class Effect:
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    p: float
    mde80: float

    def as_dict(self, prefix: str) -> dict:
        return {
            f"{prefix}": self.estimate,
            f"{prefix}_ci_low": self.ci_low,
            f"{prefix}_ci_high": self.ci_high,
            f"{prefix}_se": self.se,
            f"{prefix}_p": self.p,
            f"{prefix}_mde80": self.mde80,
        }


def _cluster_arrays(df: pd.DataFrame, col: str):
    g = df.groupby("cluster_id", observed=True)[col].agg(["sum", "count"])
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def cluster_bootstrap(df: pd.DataFrame, col: str, idx: np.ndarray) -> tuple[float, float, float, float]:
    """Ratio-of-means cluster bootstrap. `idx` is a pre-drawn (B, n_clusters) index array
    shared across columns so that intervals for paired quantities are coherent."""
    sums, counts = _cluster_arrays(df, col)
    est = sums.sum() / counts.sum()
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(est), float(lo), float(hi), float(boot.std(ddof=1))


def cluster_signflip_p(df: pd.DataFrame, col: str, signs: np.ndarray) -> float:
    """Cluster-level sign-flip randomization test for a zero mean paired difference.

    The statistic is the ratio of cluster sums to cluster counts, i.e. exactly the
    estimand reported by `cluster_bootstrap`. Clusters are unbalanced once the data
    are stratified, so flipping *cluster means* would test a different quantity.
    """
    sums, counts = _cluster_arrays(df, col)
    obs = sums.sum() / counts.sum()
    if obs == 0.0:
        return 1.0
    perm = (signs * sums).sum(axis=1) / counts.sum()
    return float((1 + np.sum(np.abs(perm) >= abs(obs))) / (len(perm) + 1))


def effects_for(df: pd.DataFrame, col: str, idx: np.ndarray, signs: np.ndarray) -> Effect:
    est, lo, hi, se = cluster_bootstrap(df, col, idx)
    p = cluster_signflip_p(df, col, signs)
    return Effect(est, lo, hi, se, p, (Z_975 + Z_80) * se)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial on discordant pairs)."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def design_effect(df: pd.DataFrame, col: str) -> float:
    """Ratio of cluster-robust variance to simple random-sampling variance."""
    d = df.groupby("cluster_id", observed=True)[col].mean().to_numpy(float)
    n = len(df)
    c = len(d)
    var_cluster = d.var(ddof=1) / c
    var_srs = df[col].var(ddof=1) / n
    return float(var_cluster / var_srs) if var_srs > 0 else float("nan")


# ----------------------------------------------------------------------------- primary table

def build_primary(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for controller in sorted(df["controller_family"].unique()):
        cdf = df[df["controller_family"] == controller]
        for stratum in ["all"] + STRATA:
            sub = cdf if stratum == "all" else cdf[cdf["from_controllability_class"].astype(str) == stratum]
            n_clusters = sub["cluster_id"].nunique()
            idx = rng.integers(0, n_clusters, size=(B_BOOT, n_clusters))
            signs = rng.integers(0, 2, size=(B_SIGN, n_clusters), dtype=np.int8) * 2 - 1

            row = {
                "controller": controller,
                "stratum": stratum,
                "stratum_label": STRATUM_LABEL[stratum],
                "n_pairs": len(sub),
                "n_clusters": n_clusters,
                "baseline_success_rate": float(sub["from_chance_success_int"].mean()),
                "controller_success_rate": float(sub["to_chance_success_int"].mean()),
                "rescued": int(sub["rescued_success"].sum()),
                "lost": int(sub["lost_success"].sum()),
                "design_effect_success": design_effect(sub, "success_uplift"),
            }
            e = effects_for(sub, "success_uplift", idx, signs)
            row.update(e.as_dict("delta_success"))
            row["mcnemar_unclustered_p"] = mcnemar_exact(int(sub["rescued_success"].sum()), int(sub["lost_success"].sum()))
            for name, col in MECH_COLS.items():
                row.update(effects_for(sub, col, idx, signs).as_dict(f"delta_{name}"))
            rows.append(row)
    out = pd.DataFrame(rows)

    # Holm over the locked primary family: success, 4 controllers × 3 strata (12 tests)
    primary = out[out["stratum"] != "all"].copy()
    primary["delta_success_p_holm"] = holm(primary["delta_success_p"].tolist())
    out = out.merge(
        primary[["controller", "stratum", "delta_success_p_holm"]],
        on=["controller", "stratum"], how="left",
    )
    # Holm over the mechanism family: 5 endpoints x 4 controllers at the all-stratum level (20 tests)
    mech_names = ["thermal_failure", "ros_failure", "hypoxic_failure", "overheat_failure", "energy_j_cm2"]
    alllev = out[out["stratum"] == "all"]
    for name in mech_names:
        pcol = f"delta_{name}_p"
        vals = alllev[pcol].tolist()
        adj = holm(vals)
        mapping = dict(zip(alllev["controller"], adj))
        out[f"delta_{name}_p_holm"] = out.apply(
            lambda r, c=pcol, m=mapping: m.get(r["controller"]) if r["stratum"] == "all" else np.nan, axis=1
        )
    return out


# ----------------------------------------------------------------------------- specification curve

def _scenario_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all_scenarios": pd.Series(True, index=df.index),
        "nominal (0 s delay, 0% noise)": (df["delay_s"] == 0) & (df["noise_fraction"] == 0),
        "delayed (>=120 s)": df["delay_s"] >= 120,
        "noisy (10%)": df["noise_fraction"] == 0.1,
    }


def build_specification_curve(df: pd.DataFrame) -> pd.DataFrame:
    """How the headline contrast depends on analyst choices.

    Varies: seed subset x scenario subset x estimator x stratum.
    The primary specification is: all seeds, all scenarios, paired cluster bootstrap, all stratum.
    """
    seeds = sorted(df["validation_seed"].unique())
    rows = []
    for controller in sorted(df["controller_family"].unique()):
        cdf = df[df["controller_family"] == controller]
        masks = _scenario_masks(cdf)
        for seed_choice, seed_subset in [
            ("all seeds", seeds),
            *[(s.replace("seed_", ""), [s]) for s in seeds],
        ]:
            for scen_name, scen_mask in masks.items():
                for stratum in ["all"] + STRATA:
                    sub = cdf[cdf["validation_seed"].isin(seed_subset) & scen_mask]
                    if stratum != "all":
                        sub = sub[sub["from_controllability_class"].astype(str) == stratum]
                    if sub["cluster_id"].nunique() < 5:
                        continue
                    # paired, cluster-robust
                    n_clusters = sub["cluster_id"].nunique()
                    idx = np.random.default_rng(SEED).integers(0, n_clusters, size=(2000, n_clusters))
                    est, lo, hi, se = cluster_bootstrap(sub, "success_uplift", idx)
                    # naive: unpaired difference of proportions, no clustering
                    p_from = sub["from_chance_success_int"].mean()
                    p_to = sub["to_chance_success_int"].mean()
                    naive = p_to - p_from
                    se_naive = np.sqrt(p_to * (1 - p_to) / len(sub) + p_from * (1 - p_from) / len(sub))
                    rows.append({
                        "controller": controller,
                        "seed_subset": seed_choice,
                        "scenarios": scen_name,
                        "stratum": stratum,
                        "estimator": "paired cluster bootstrap",
                        "delta": est, "ci_low": lo, "ci_high": hi,
                        "significant": bool(lo > 0 or hi < 0),
                    })
                    rows.append({
                        "controller": controller,
                        "seed_subset": seed_choice,
                        "scenarios": scen_name,
                        "stratum": stratum,
                        "estimator": "naive unpaired",
                        "delta": naive,
                        "ci_low": naive - Z_975 * se_naive,
                        "ci_high": naive + Z_975 * se_naive,
                        "significant": bool(abs(naive) > Z_975 * se_naive),
                    })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Pareto

PARETO_OBJECTIVES = [
    ("delta_success", "max"),
    ("delta_thermal_failure", "min"),
    ("delta_ros_failure", "min"),
    ("delta_overheat_failure", "min"),
    ("delta_energy_j_cm2", "min"),
]


def build_pareto(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    alllev = primary[primary["stratum"] == "all"].set_index("controller")
    controllers = list(alllev.index)
    mat = np.column_stack([
        alllev["delta_success"].to_numpy(float),
        alllev["delta_thermal_failure"].to_numpy(float),
        alllev["delta_ros_failure"].to_numpy(float),
        alllev["delta_overheat_failure"].to_numpy(float),
        alllev["delta_energy_j_cm2"].to_numpy(float),
    ])
    # orient so larger is better
    oriented = mat.copy()
    for j, (_, sense) in enumerate(PARETO_OBJECTIVES):
        if sense == "min":
            oriented[:, j] = -oriented[:, j]

    rows = []
    for i, a in enumerate(controllers):
        for j, b in enumerate(controllers):
            if i == j:
                continue
            ge = oriented[i] >= oriented[j]
            gt = oriented[i] > oriented[j]
            if ge.all() and gt.any():
                rows.append({"controller": a, "dominates": b})
    dom = pd.DataFrame(rows, columns=["controller", "dominates"])

    summary = []
    for i, a in enumerate(controllers):
        summary.append({
            "controller": a,
            "delta_success": mat[i, 0],
            "delta_thermal_failure": mat[i, 1],
            "delta_ros_failure": mat[i, 2],
            "delta_overheat_failure": mat[i, 3],
            "delta_energy_j_cm2": mat[i, 4],
            "dominates_n": int((dom["controller"] == a).sum()),
            "dominated_by_n": int((dom["dominates"] == a).sum()),
            "pareto_optimal": bool((dom["dominates"] == a).sum() == 0),
        })
    return dom, pd.DataFrame(summary)


def build_weight_sensitivity(pareto_summary: pd.DataFrame, n: int = 200_000) -> pd.DataFrame:
    """Rank stability under random objective weights.

    Objectives are min-max normalised across controllers, then combined with
    Dirichlet(1,...,1) weights. This replaces the single arbitrary scalar utility.
    """
    ctrls = pareto_summary["controller"].tolist()
    M = pareto_summary[[c for c, _ in PARETO_OBJECTIVES]].to_numpy(float)
    orient = np.array([1.0 if s == "max" else -1.0 for _, s in PARETO_OBJECTIVES])
    V = M * orient
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    rng_range = np.where(hi - lo == 0, 1.0, hi - lo)
    N = (V - lo) / rng_range  # 0..1, 1 = best
    rng = np.random.default_rng(SEED)
    W = rng.dirichlet(np.ones(len(PARETO_OBJECTIVES)), size=n)
    scores = W @ N.T  # n x controllers
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1
    rows = []
    for k, c in enumerate(ctrls):
        rows.append({
            "controller": c,
            "p_rank1": float((ranks[:, k] == 1).mean()),
            "p_rank_last": float((ranks[:, k] == len(ctrls)).mean()),
            "mean_rank": float(ranks[:, k].mean()),
            "p_top2": float((ranks[:, k] <= 2).mean()),
        })
    return pd.DataFrame(rows).sort_values("p_rank1", ascending=False)


# ----------------------------------------------------------------------------- strata audit

def build_strata_audit(base: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    def auc(y, s):
        y = np.asarray(y, int); s = np.asarray(s, float)
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        order = np.argsort(np.concatenate([pos, neg]))
        ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
        return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    cdf = df[df["controller_family"] == "Constrained optimized"]
    rows = []
    for (seed, fam), g in cdf.groupby(["validation_seed", "model_family"], observed=True):
        rows.append({
            "seed": seed, "model_family": fam, "n": len(g),
            "auc": auc(g["from_chance_success_int"], g["from_predicted_probability"]),
            "observed_success": float(g["from_chance_success_int"].mean()),
            "mean_predicted": float(g["from_predicted_probability"].mean()),
        })
    detail = pd.DataFrame(rows)
    detail["scope"] = "locked validation (seeds 20260720-22)"

    ext = pd.read_csv(base / "nature_computational_science" / "data" / "external_boundary_validation" / "external_boundary_validation_metrics.csv")
    ext_rows = pd.DataFrame({
        "seed": "seed_20260713",
        "model_family": ext["model_family"],
        "n": ext["n"],
        "auc": ext["auc"],
        "observed_success": ext["success_rate"],
        "mean_predicted": ext["mean_predicted_probability"],
        "scope": "external held-out (different seed)",
    })
    detail = pd.concat([detail, ext_rows], ignore_index=True)

    meta = json.loads((base / "nature_computational_science" / "data" / "multiseed_controllability" / "multiseed_controllability_metadata.json").read_text())
    q33, q67 = (cdf["from_predicted_probability"].quantile([0.33, 0.67]).to_numpy(float))
    facts = {
        "frozen_equation": meta["combined_equation"]["equation"],
        "features": meta["features"],
        "thresholds_from_reference_population": {
            "q33": float(pd.read_csv(base / "nature_computational_science" / "data" / "predictive_control_reclassification" / "multiseed_outofseed_scored_controllers.csv", usecols=["predicted_probability"])["predicted_probability"].quantile(0.33)),
            "q67": float(pd.read_csv(base / "nature_computational_science" / "data" / "predictive_control_reclassification" / "multiseed_outofseed_scored_controllers.csv", usecols=["predicted_probability"])["predicted_probability"].quantile(0.67)),
        },
        "thresholds_implied_in_validation_population": {"q33": float(q33), "q67": float(q67)},
        "validation_auc_overall": auc(cdf["from_chance_success_int"], cdf["from_predicted_probability"]),
        "validation_auc_min_seed_family": float(detail[detail.scope.str.startswith("locked")]["auc"].min()),
        "external_auc_overall": float(ext[ext.model_family == "all"]["auc"].iloc[0]),
        "external_brier_overall": float(ext[ext.model_family == "all"]["brier"].iloc[0]),
        "stratum_counts": {k: int(v) for k, v in cdf["from_controllability_class"].astype(str).value_counts().items()},
        "observed_success_by_stratum": {
            k: float(v) for k, v in cdf.groupby(cdf["from_controllability_class"].astype(str), observed=True)["from_chance_success_int"].mean().items()
        },
    }
    return detail, facts


# ----------------------------------------------------------------------------- main

def run(base: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_paired(base)

    primary = build_primary(df)
    primary.to_csv(out_dir / "primary_endpoints.csv", index=False)

    spec = build_specification_curve(df)
    spec.to_csv(out_dir / "specification_curve.csv", index=False)

    dom, pareto = build_pareto(primary)
    dom.to_csv(out_dir / "pareto_dominance_edges.csv", index=False)
    pareto.to_csv(out_dir / "pareto_summary.csv", index=False)

    weights = build_weight_sensitivity(pareto)
    weights.to_csv(out_dir / "weight_sensitivity.csv", index=False)

    strata_detail, strata_facts = build_strata_audit(base, df)
    strata_detail.to_csv(out_dir / "strata_audit.csv", index=False)

    endpoints = pd.DataFrame(ENDPOINT_DEFINITIONS, columns=["criterion", "rule", "meaning"])
    endpoints.to_csv(out_dir / "endpoint_definitions.csv", index=False)

    # specification-curve stability on the headline subset only, so that the number
    # quoted in the manuscript is the one plotted in Figure 4.
    headline = spec[(spec["estimator"] == "paired cluster bootstrap") & (spec["stratum"] == "all")]
    stability = (
        headline.groupby(["controller"], observed=True)["delta"]
        .agg(fraction_positive=lambda s: float((s > 0).mean()), n_specifications="size",
             delta_min="min", delta_max="max")
        .reset_index()
    )
    stability.to_csv(out_dir / "specification_stability.csv", index=False)

    summary = {
        "n_pairs_per_controller": int(len(df) / df["controller_family"].nunique()),
        "n_controllers": int(df["controller_family"].nunique()),
        "n_clusters": int(df["cluster_id"].nunique()),
        "rows_per_cluster": int(len(df) / df["controller_family"].nunique() / df["cluster_id"].nunique()),
        "bootstrap_replicates": B_BOOT,
        "signflip_replicates": B_SIGN,
        "random_seed": SEED,
        "strata": strata_facts,
        "primary_all_stratum": primary[primary["stratum"] == "all"][
            ["controller", "delta_success", "delta_success_ci_low", "delta_success_ci_high",
             "delta_success_p", "delta_success_mde80"]
        ].round(6).to_dict(orient="records"),
        "specification_sign_agreement": stability.round(4).to_dict(orient="records"),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
