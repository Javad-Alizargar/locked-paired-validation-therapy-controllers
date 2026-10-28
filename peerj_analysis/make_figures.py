"""Generate PeerJ-ready figures (900-3000 px per dimension, 300 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

DPI = 300
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

CTRL_COLOR = {
    "Constrained optimized": "#1f6feb",
    "Smoothed predictive": "#2f9e44",
    "Thermal-floor predictive": "#e8590c",
    "Reduced-order MPC": "#9c36b5",
}
CTRL_SHORT = {
    "Constrained optimized": "Constrained optimized",
    "Smoothed predictive": "Smoothed predictive",
    "Thermal-floor predictive": "Thermal-floor predictive",
    "Reduced-order MPC": "Reduced-order MPC",
}
STRATUM_ORDER = ["all", "low", "transitional", "high"]
STRATUM_NAME = {
    "all": "All",
    "low": "Predicted-low",
    "transitional": "Predicted-intermediate",
    "high": "Predicted-high",
}


def _save(fig, name: str) -> None:
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    assert 900 <= w <= 3000 and 900 <= h <= 3000, f"{name}: {w}x{h} px out of range"
    print(f"  {name}  {w}x{h} px")


# --------------------------------------------------------------------- Figure 1

def figure1() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc="#f1f3f5", ec="#343a40"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.7, linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     lw=0.9, color="#495057"))

    W, H = 29, 21
    ys = [71, 41, 11]
    xs = [2, 35.5, 69]

    box(xs[0], ys[0], W, H, "Design freeze\n3 seeds x 4 model families\nx 32 draws x 12 scenarios\n= 4,608 pairs per controller", "#e7f5ff")
    box(xs[1], ys[0], W, H, "Frozen scoring rule\ntrained on independent seeds,\nthen applied unchanged", "#e7f5ff")
    box(xs[2], ys[0], W, H, "Four controller families\nfull feedback (baseline)\nsmoothed / thermal-floor\nconstrained / reduced-order MPC", "#e7f5ff")

    box(xs[0], ys[1], W, H, "Paired comparison\nidentical seed, family,\ndraw, delay and noise", "#f8f9fa")
    box(xs[1], ys[1], W, H, "Cluster-robust inference\nbootstrap + permutation\nHolm correction, MDE", "#f8f9fa")
    box(xs[2], ys[1], W, H, "Multi-endpoint reporting\nsuccess, four failure modes,\nenergy, Pareto, weights", "#f8f9fa")

    box(xs[0], ys[2], W, H, "Specification curve\nseed x scenario x estimator\nx stratum", "#fff4e6")
    box(xs[1], ys[2], W, H, "Falsification checks\nno winner from a single\nrate or a single weighting", "#fff4e6")
    box(xs[2], ys[2], W, H, "Reproducible artefact\none entry point regenerates\nall tables and figures", "#fff4e6")

    for y in ys:
        arrow(xs[0] + W, y + H / 2, xs[1], y + H / 2)
        arrow(xs[1] + W, y + H / 2, xs[2], y + H / 2)
    for x in xs:
        arrow(x + W / 2, ys[0], x + W / 2, ys[1] + H)
        arrow(x + W / 2, ys[1], x + W / 2, ys[2] + H)

    ax.text(50, 96.5, "Locked paired-validation protocol", ha="center", fontsize=9.2, weight="bold")
    _save(fig, "Figure 1.png")


# --------------------------------------------------------------------- Figure 2

def figure2() -> None:
    p = pd.read_csv(RES / "primary_endpoints.csv")
    p = p[p.stratum.isin(STRATUM_ORDER)].copy()
    p["sorder"] = p.stratum.map({s: i for i, s in enumerate(STRATUM_ORDER)})
    p = p.sort_values(["sorder", "controller"])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y = 0
    ticks, labels = [], []
    for stratum in STRATUM_ORDER:
        g = p[p.stratum == stratum]
        ax.axhspan(y - 0.5, y + len(g) - 0.5, color="#f8f9fa" if stratum in ("all", "transitional") else "white", zorder=0)
        for _, r in g.iterrows():
            ax.plot([r.delta_success_ci_low, r.delta_success_ci_high], [y, y],
                    color=CTRL_COLOR[r.controller], lw=1.8, solid_capstyle="butt", zorder=2)
            ax.plot(r.delta_success, y, "o", ms=4.6, color=CTRL_COLOR[r.controller], zorder=3)
            ticks.append(y)
            labels.append(CTRL_SHORT[r.controller])
            y += 1
        ax.text(1.02, (y - 1 + y - len(g)) / 2, STRATUM_NAME[stratum], transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, weight="bold")
        y += 0.8

    ax.axvline(0, color="#343a40", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_ylim(-0.7, y - 0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Paired change in success probability (controller minus full-feedback baseline)")
    ax.set_xlim(-0.78, 0.20)
    ax.set_title("No controller improves overall success; predicted-low gains are not significant after correction",
                 fontsize=8.6, loc="left")
    ax.grid(axis="x", color="#dee2e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, "Figure 2.png")


# --------------------------------------------------------------------- Figure 3

def figure3() -> None:
    p = pd.read_csv(RES / "primary_endpoints.csv")
    alllev = p[p.stratum == "all"].set_index("controller")
    ctrls = list(CTRL_COLOR)
    metrics = [
        ("delta_thermal_failure", "Thermal failure", "Change in rate"),
        ("delta_ros_failure", "ROS failure", "Change in rate"),
        ("delta_overheat_failure", "Overheating", "Change in rate"),
        ("delta_energy_j_cm2", "Delivered energy", "Change (J cm$^{-2}$)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.1), sharey=True)
    for ax, (col, title, xlabel) in zip(axes, metrics):
        for i, c in enumerate(ctrls):
            r = alllev.loc[c]
            lo, hi = r[f"{col}_ci_low"], r[f"{col}_ci_high"]
            ax.plot([lo, hi], [i, i], color=CTRL_COLOR[c], lw=1.8)
            ax.plot(r[col], i, "o", ms=4.2, color=CTRL_COLOR[c])
        ax.axvline(0, color="#343a40", lw=0.9, ls="--")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7.4)
        ax.tick_params(axis="x", labelsize=6.6)
        ax.grid(axis="x", color="#dee2e6", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(ctrls)))
    axes[0].set_yticklabels([CTRL_SHORT[c] for c in ctrls], fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("Failure modes move in different directions, so success alone hides the trade-off",
                 fontsize=8.8, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "Figure 3.png")


# --------------------------------------------------------------------- Figure 4

def figure4() -> None:
    s = pd.read_csv(RES / "specification_curve.csv")
    s = s[(s.estimator == "paired cluster bootstrap") & (s.stratum == "all")].copy()
    ctrls = list(CTRL_COLOR)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.4), sharey=True)
    for ax, c in zip(axes, ctrls):
        g = s[s.controller == c].sort_values("delta").reset_index(drop=True)
        colors = ["#c92a2a" if v < 0 else "#2f9e44" for v in g.delta]
        ax.scatter(range(len(g)), g.delta, c=colors, s=14, zorder=3)
        ax.axhline(0, color="#343a40", lw=0.9)
        ax.set_title(CTRL_SHORT[c], fontsize=7.6)
        ax.set_xlabel("Specification (sorted)")
        ax.grid(axis="y", color="#dee2e6", lw=0.5)
        ax.set_axisbelow(True)
        frac = (g.delta > 0).mean()
        ax.text(0.04, 0.94, f"{frac:.0%} positive", transform=ax.transAxes,
                fontsize=7, va="top", color="#495057")
    axes[0].set_ylabel("Paired change in success probability")
    fig.suptitle("Sign of the headline effect flips across defensible analysis choices",
                 fontsize=8.8, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "Figure 4.png")


# --------------------------------------------------------------------- Figure 5

def figure5() -> None:
    par = pd.read_csv(RES / "pareto_summary.csv")
    w = pd.read_csv(RES / "weight_sensitivity.csv").set_index("controller")
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.15]}
    )

    def panel(ax, xcol, xlabel, title):
        for _, r in par.iterrows():
            ax.scatter(r[xcol], r.delta_success, s=46, color=CTRL_COLOR[r.controller],
                       edgecolor="#212529", linewidth=0.6, zorder=3)
        ax.axhline(0, color="#868e96", lw=0.8, ls="--")
        ax.axvline(0, color="#868e96", lw=0.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=7.0)
        ax.set_ylabel("Change in success probability", fontsize=7.0)
        ax.set_title(title, fontsize=7.8)
        ax.tick_params(labelsize=6.4)
        ax.grid(color="#dee2e6", lw=0.5); ax.set_axisbelow(True)

    panel(ax1, "delta_overheat_failure", "Change in overheating rate", "Safety trade-off")
    panel(ax2, "delta_energy_j_cm2", "Change in delivered energy (J cm$^{-2}$)", "Energy trade-off")

    order = w.sort_values("p_rank1", ascending=True).index
    ax3.barh(range(len(order)), w.loc[order, "p_rank1"],
             color=[CTRL_COLOR[c] for c in order], edgecolor="#212529", linewidth=0.5)
    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels([CTRL_SHORT[c] for c in order], fontsize=6.6)
    ax3.set_xlabel("Probability of ranking first\nunder random objective weights", fontsize=7.0)
    ax3.set_xlim(0, 0.55)
    ax3.set_title("No robust winner", fontsize=7.8)
    ax3.grid(axis="x", color="#dee2e6", lw=0.5); ax3.set_axisbelow(True)
    for i, c in enumerate(order):
        ax3.text(w.loc[c, "p_rank1"] + 0.01, i, f"{w.loc[c, 'p_rank1']:.0%}", va="center", fontsize=6.4)
    ax3.tick_params(labelsize=6.4)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=5,
                   markerfacecolor=CTRL_COLOR[c], markeredgecolor="#212529", label=CTRL_SHORT[c])
        for c in CTRL_COLOR
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.6, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("All four controllers are Pareto non-dominated", fontsize=8.8, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    _save(fig, "Figure 5.png")


if __name__ == "__main__":
    print("writing figures to", FIG)
    figure1(); figure2(); figure3(); figure4(); figure5()
