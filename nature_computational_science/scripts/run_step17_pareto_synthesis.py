from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_step12_low_class_failure_mode_decomposition import clustered_ci


ROOT = Path(__file__).resolve().parents[2]
NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data" / "controller_pareto_synthesis"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"

CONTROLLERS = {
    "Smoothed predictive": NCS / "data" / "locked_large_validation" / "locked_large_validation_paired_failure_modes.csv",
    "Thermal-floor predictive": NCS
    / "data"
    / "thermal_floor_predictive_validation"
    / "thermal_floor_paired_failure_modes.csv",
    "Constrained optimized": NCS
    / "data"
    / "constrained_policy_optimization"
    / "optimized_policy_locked_validation_pairs.csv",
    "Reduced-order MPC": NCS
    / "data"
    / "reduced_mpc_pareto_validation"
    / "reduced_mpc_paired_failure_modes.csv",
}

FAILURE_COLS = [
    "thermal_failure_rate_delta",
    "ros_failure_rate_delta",
    "overheat_failure_rate_delta",
]


def load_pairs() -> pd.DataFrame:
    rows = []
    for controller, path in CONTROLLERS.items():
        df = pd.read_csv(path)
        df["controller_family"] = controller
        rows.append(df)
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["cluster_id"] = (
        out["validation_seed"].astype(str)
        + "|"
        + out["model_family"].astype(str)
        + "|"
        + out["atlas_uncertainty_id"].astype(str)
    )
    out["delay_bin"] = pd.cut(
        out["delay_s"],
        bins=[-np.inf, 0, 30, 60, np.inf],
        labels=["0 s", "1-30 s", "31-60 s", ">60 s"],
    )
    return out


def mean_ci(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    lo, hi = clustered_ci(df, col)
    return float(df[col].mean()), lo, hi


def summarize_stratum(df: pd.DataFrame, controller: str, stratum: str, group: pd.DataFrame) -> dict:
    low_loss = float(group["lost_success"].mean())
    low_rescue = float(group["rescued_success"].mean())
    success, success_lo, success_hi = mean_ci(group, "success_uplift")
    energy, energy_lo, energy_hi = mean_ci(group, "delivered_energy_j_cm2_delta")
    row = {
        "controller_family": controller,
        "stratum": stratum,
        "n": len(group),
        "clusters": group["cluster_id"].nunique(),
        "success_uplift": success,
        "success_uplift_ci_low": success_lo,
        "success_uplift_ci_high": success_hi,
        "rescued_to_lost_ratio": float(low_rescue / low_loss) if low_loss > 0 else float("inf"),
        "energy_delta": energy,
        "energy_delta_ci_low": energy_lo,
        "energy_delta_ci_high": energy_hi,
        "thermal_failure_delta": float(group["thermal_failure_rate_delta"].mean()),
        "ros_failure_delta": float(group["ros_failure_rate_delta"].mean()),
        "overheat_failure_delta": float(group["overheat_failure_rate_delta"].mean()),
        "warm_fraction_delta": float(group["warm_fraction_delta"].mean()),
        "ros_threshold_fraction_delta": float(group["ros_threshold_fraction_delta"].mean()),
        "max_temperature_delta": float(group["max_temperature_c_delta"].mean()),
    }
    row["safety_penalty"] = max(row["overheat_failure_delta"], 0.0) + max(row["energy_delta"], 0.0) / 500.0
    row["mechanism_gain"] = -min(row["thermal_failure_delta"], 0.0) - min(row["ros_failure_delta"], 0.0)
    row["scalar_utility"] = (
        row["success_uplift"]
        + 0.35 * row["mechanism_gain"]
        - 0.80 * max(row["overheat_failure_delta"], 0.0)
        - 0.0015 * max(row["energy_delta"], 0.0)
        - 0.50 * max(-row["success_uplift"], 0.0)
    )
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for controller, cdf in df.groupby("controller_family"):
        rows.append(summarize_stratum(df, controller, "all", cdf))
        for cls in ["low", "transitional", "high"]:
            g = cdf[cdf["from_controllability_class"].astype(str) == cls]
            rows.append(summarize_stratum(df, controller, f"from_{cls}", g))
        low = cdf[cdf["from_controllability_class"].astype(str) == "low"]
        for delay, g in low.groupby("delay_bin", observed=False):
            if len(g):
                rows.append(summarize_stratum(df, controller, f"low_delay_{delay}", g))
    return pd.DataFrame(rows)


def pareto_flags(summary: pd.DataFrame) -> pd.DataFrame:
    low = summary[summary["stratum"] == "from_low"].copy()
    low["dominated"] = False
    # Maximize success uplift and mechanism gain; minimize overheat and positive energy.
    for idx, row in low.iterrows():
        others = low[low.index != idx]
        better_or_equal = (
            (others["success_uplift"] >= row["success_uplift"])
            & (others["mechanism_gain"] >= row["mechanism_gain"])
            & (others["overheat_failure_delta"] <= row["overheat_failure_delta"])
            & (others["energy_delta"] <= row["energy_delta"])
        )
        strictly_better = (
            (others["success_uplift"] > row["success_uplift"])
            | (others["mechanism_gain"] > row["mechanism_gain"])
            | (others["overheat_failure_delta"] < row["overheat_failure_delta"])
            | (others["energy_delta"] < row["energy_delta"])
        )
        low.loc[idx, "dominated"] = bool((better_or_equal & strictly_better).any())
    return summary.merge(low[["controller_family", "dominated"]], on="controller_family", how="left")


def make_figure(summary: pd.DataFrame, pairs: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0), constrained_layout=True)
    colors = {
        "Smoothed predictive": "#8d99ae",
        "Thermal-floor predictive": "#2a9d8f",
        "Constrained optimized": "#457b9d",
        "Reduced-order MPC": "#c44536",
    }
    low = summary[summary["stratum"] == "from_low"].set_index("controller_family")
    order = list(CONTROLLERS.keys())

    ax = axes[0, 0]
    x = np.arange(len(order))
    y = low.loc[order, "success_uplift"]
    yerr = np.vstack([y - low.loc[order, "success_uplift_ci_low"], low.loc[order, "success_uplift_ci_high"] - y])
    ax.bar(x, y, color=[colors[o] for o in order])
    ax.errorbar(x, y, yerr=yerr, fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["smoothed", "floor", "optimized", "reduced\nMPC"], rotation=20)
    ax.set_ylabel("Low-class success uplift")
    ax.set_title("A. Locked validation")

    ax = axes[0, 1]
    for controller in order:
        row = low.loc[controller]
        ax.scatter(row["overheat_failure_delta"], row["success_uplift"], s=120, color=colors[controller], label=controller)
        ax.text(row["overheat_failure_delta"], row["success_uplift"], " " + controller.split()[0], va="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Low-class overheat-failure delta")
    ax.set_ylabel("Low-class success uplift")
    ax.set_title("B. Safety-success tradeoff")

    ax = axes[0, 2]
    width = 0.24
    summary_failure_cols = ["thermal_failure_delta", "ros_failure_delta", "overheat_failure_delta"]
    for j, col in enumerate(summary_failure_cols):
        vals = -low.loc[order, col]
        ax.bar(x + (j - 1) * width, vals, width=width, label=col.replace("_failure_delta", ""))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["smoothed", "floor", "optimized", "reduced\nMPC"], rotation=20)
    ax.set_ylabel("Failure-rate reduction")
    ax.set_title("C. Mechanistic gains")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    y = low.loc[order, "energy_delta"]
    ax.bar(x, y, color=[colors[o] for o in order])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["smoothed", "floor", "optimized", "reduced\nMPC"], rotation=20)
    ax.set_ylabel("Energy delta (J cm^-2)")
    ax.set_title("D. Energy burden")

    ax = axes[1, 1]
    delay_rows = summary[summary["stratum"].str.startswith("low_delay_")].copy()
    delay_rows["delay"] = delay_rows["stratum"].str.replace("low_delay_", "", regex=False)
    delay_order = ["0 s", "1-30 s", "31-60 s", ">60 s"]
    for controller in order:
        g = delay_rows[delay_rows["controller_family"] == controller].set_index("delay").reindex(delay_order)
        ax.plot(delay_order, g["success_uplift"], marker="o", color=colors[controller], label=controller)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Low-class success uplift")
    ax.set_title("E. Delay-specific behavior")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1, 2]
    plot = low.loc[order].copy()
    ax.scatter(plot["mechanism_gain"], plot["safety_penalty"], s=140, c=[colors[o] for o in order])
    for controller in order:
        row = plot.loc[controller]
        ax.text(row["mechanism_gain"], row["safety_penalty"], " " + controller.split()[0], va="center", fontsize=8)
    ax.set_xlabel("Thermal + ROS failure reduction")
    ax.set_ylabel("Safety/energy penalty")
    ax.set_title("F. Pareto map")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_note(summary: pd.DataFrame) -> None:
    low = summary[summary["stratum"] == "from_low"].sort_values("scalar_utility", ascending=False)
    best = low.iloc[0]
    reduced = low[low["controller_family"] == "Reduced-order MPC"].iloc[0]
    lines = [
        "# Step 17 Pareto Synthesis",
        "",
        "Question: what is the clean submission story after locked validation of all controller families?",
        "",
        "## Interpretation",
        "",
        "The results do not support a claim that a new controller solves the biomedical control problem. They do support a methods contribution: a reproducible locked-validation framework exposes failure modes and Pareto trade-offs that nominal controller testing hides.",
        "",
        "## Low-Class Controller Summary",
        "",
        "```",
        low[
            [
                "controller_family",
                "success_uplift",
                "success_uplift_ci_low",
                "success_uplift_ci_high",
                "thermal_failure_delta",
                "ros_failure_delta",
                "overheat_failure_delta",
                "energy_delta",
                "scalar_utility",
                "dominated",
            ]
        ].round(3).to_string(index=False),
        "```",
        "",
        "## Main Findings",
        "",
        f"Best balanced controller family by the synthesis utility is `{best['controller_family']}`, but its low-class success uplift is {best['success_uplift']:.3f} [{best['success_uplift_ci_low']:.3f}, {best['success_uplift_ci_high']:.3f}], so it should not be presented as a definitive superior controller.",
        f"Reduced-order MPC is an important negative-control result: low-class overheat failure delta is {reduced['overheat_failure_delta']:.3f} and energy delta is {reduced['energy_delta']:.1f} J cm^-2 despite reductions in thermal/ROS failure modes.",
        "",
        "## Recommended Claim",
        "",
        "A locked, reproducible controller-validation framework for delayed biomedical feedback simulations reveals that apparent gains in thermal coverage and ROS completion can trade off against overheating, energy burden, and high-class performance. The contribution is the computational validation method, released software, and failure-mode/Pareto analysis, not a proposed clinical protocol.",
        "",
        "## Recommended Manuscript Title",
        "",
        "Locked validation and Pareto failure-mode analysis of delayed feedback controllers in photothermal-photodynamic biomedical simulations",
        "",
        "## Stop Rule",
        "",
        "Do not add more heuristic controllers before drafting. The project is now strong enough for a methods/software paper if the claims remain bounded. Additional controller search should be framed as future work or supplementary benchmarking.",
    ]
    (OUT_NOTES / "step17_pareto_synthesis_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    summary = pareto_flags(summarize(pairs))
    pairs.to_csv(OUT_DATA / "controller_family_locked_pairs.csv", index=False)
    summary.to_csv(OUT_DATA / "controller_family_pareto_summary.csv", index=False)
    metadata = {
        "controllers": {k: str(v.relative_to(ROOT)) for k, v in CONTROLLERS.items()},
        "purpose": "Final synthesis across locked controller-family validations.",
        "claim_boundary": "Method/software validation and Pareto failure-mode analysis, not clinical protocol superiority.",
    }
    (OUT_DATA / "controller_pareto_synthesis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figure(summary, pairs, OUT_FIG / "figure_17_pareto_synthesis.png")
    write_note(summary)
    low = summary[summary["stratum"] == "from_low"].sort_values("scalar_utility", ascending=False)
    print("Wrote Pareto synthesis outputs")
    print(low[["controller_family", "success_uplift", "thermal_failure_delta", "ros_failure_delta", "overheat_failure_delta", "energy_delta", "scalar_utility", "dominated"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
