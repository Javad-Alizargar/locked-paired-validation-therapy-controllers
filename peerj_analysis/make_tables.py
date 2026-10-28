"""Build PeerJ-formatted tables (markdown + .docx)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)

CTRL_ORDER = ["Smoothed predictive", "Thermal-floor predictive", "Constrained optimized", "Reduced-order MPC"]
STRAT_ORDER = ["all", "low", "transitional", "high"]
STRAT_NAME = {"all": "All", "low": "Low score", "transitional": "Intermediate score", "high": "High score"}


def f3(x: float, dp: int = 3) -> str:
    return f"{x:+.{dp}f}"


def f0(x: float, dp: int = 3) -> str:
    return f"{x:.{dp}f}"


def ci(r: pd.Series, col: str, dp: int = 3) -> str:
    lo = f"{r[col + '_ci_low']:.{dp}f}"
    hi = f"{r[col + '_ci_high']:.{dp}f}"
    return f"{f3(r[col], dp)} ({lo} to {hi})"


def pfmt(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def docx_table(title: str, note: str, headers: list[str], rows: list[list[str]], path: Path, widths=None):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    for s in doc.sections:
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
        s.page_width = Inches(8.5); s.page_height = Inches(11)

    h = doc.add_paragraph()
    h.paragraph_format.keep_with_next = True
    run = h.add_run(title)
    run.bold = True
    run.font.size = Pt(11)

    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = OxmlElement("w:" + edge)
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "D9D9D9")
        borders.append(border)
    t._tbl.tblPr.append(borders)
    hdr = t.rows[0].cells
    for cell in hdr:
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "EAF1F8")
        cell._tc.get_or_add_tcPr().append(shade)
    for i, htxt in enumerate(headers):
        hdr[i].text = ""
        par = hdr[i].paragraphs[0]
        r = par.add_run(htxt)
        r.bold = True
        r.font.size = Pt(8)
        r.font.name = "Times New Roman"
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            par = cells[i].paragraphs[0]
            r = par.add_run(val)
            r.font.size = Pt(8)
            r.font.name = "Times New Roman"
    if widths:
        widths = [w * 6.5 / sum(widths) for w in widths]
        t.autofit = False
        for i, w in enumerate(widths):
            t.columns[i].width = Inches(w)
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)

    n = doc.add_paragraph()
    r = n.add_run(note)
    r.font.size = Pt(9)
    r.italic = True
    for par in doc.paragraphs:
        par.paragraph_format.line_spacing = 1.0
    for row in t.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.space_after = Pt(2)
                par.paragraph_format.space_before = Pt(2)
                par.paragraph_format.line_spacing = 1.0
    doc.save(path)
    print(f"  {path.name}")


def md_table(title: str, note: str, headers: list[str], rows: list[list[str]], path: Path):
    lines = [f"**{title}**", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines += ["", f"*{note}*", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {path.name}")


def main() -> None:
    p = pd.read_csv(RES / "primary_endpoints.csv")
    par = pd.read_csv(RES / "pareto_summary.csv").set_index("controller")
    p["corder"] = p.controller.map({c: i for i, c in enumerate(CTRL_ORDER)})
    p["sorder"] = p.stratum.map({s: i for i, s in enumerate(STRAT_ORDER)})

    # ---------------- Table 1: success endpoints
    h1 = ["Controller", "Pairs", "Baseline success", "Controller success",
          "Change in success (95% CI)", "p", "Approx. MDE (80%)", "Rescued / lost"]
    r1 = []
    for _, r in p[p.stratum == "all"].sort_values("corder").iterrows():
        r1.append([
            r.controller, f"{int(r.n_pairs)}", f0(r.baseline_success_rate),
            f0(r.controller_success_rate), ci(r, "delta_success"),
            pfmt(r.delta_success_p), f0(r.delta_success_mde80),
            f"{int(r.rescued)} / {int(r.lost)}",
        ])
    note1 = ("All 4,608 paired trials per controller (96 parent blocks; four families × 12 scenarios per block). "
             "Change is the paired risk difference against the same full-feedback baseline, with a 95% "
             "approximate seed-stratified parent-block bootstrap interval (10,000 replicates) and an exploratory sign-flip p-value "
             "(20,000 permutations). MDE is an approximate unadjusted normal-theory precision summary at 80% power, not a guarantee for Holm or sign-flip testing. It uses the observed "
             "bootstrap standard error. Overall contrasts are descriptive; the declared "
             "exploratory multiplicity family is the 12 controller × band contrasts in Table 3.")
    docx_table("Table 1. Success endpoints across the full simulation atlas.", note1, h1, r1,
               OUT / "Table 1.docx", widths=[1.4, 0.5, 0.65, 0.7, 1.5, 0.5, 0.7, 0.65])
    md_table("Table 1. Success endpoints across the full simulation atlas.", note1, h1, r1, OUT / "Table 1.md")

    # ---------------- Table 2: mechanism + energy
    h2 = ["Controller", "Thermal failure", "ROS failure", "Overheating",
          "Delivered energy (J cm⁻²)", "Pareto status"]
    r2 = []
    for _, r in p[p.stratum == "all"].sort_values("corder").iterrows():
        r2.append([
            r.controller,
            ci(r, "delta_thermal_failure"), ci(r, "delta_ros_failure"),
            ci(r, "delta_overheat_failure"), ci(r, "delta_energy_j_cm2", 1),
            "Non-dominated" if par.loc[r.controller, "pareto_optimal"] else "Dominated",
        ])
    note2 = ("Values are paired changes in the endpoint rate (or in delivered energy) against the full-feedback "
             "baseline, with approximate 95% seed-stratified parent-block bootstrap intervals (10,000 replicates). Negative failure changes indicate "
             "fewer failures. Hypoxic failure did not change for any controller (all deltas 0.000, 95% CI 0.000 to "
             "0.000) and is omitted. Pareto status is computed over the five objectives shown in Table 1 and this "
             "table; all four controllers are mutually non-dominated.")
    docx_table("Table 2. Failure-mode and energy endpoints.", note2, h2, r2,
               OUT / "Table 2.docx", widths=[1.5, 1.35, 1.3, 1.3, 1.55, 0.9])
    md_table("Table 2. Failure-mode and energy endpoints.", note2, h2, r2, OUT / "Table 2.md")

    # ---------------- Table 3: stratified success
    h3 = ["Controller", "Stratum", "Pairs", "Baseline success", "Change in success (95% CI)",
          "p", "Holm-adjusted p", "Approx. MDE (80%)"]
    r3 = []
    for _, r in p.sort_values(["sorder", "corder"]).iterrows():
        r3.append([
            r.controller, STRAT_NAME[r.stratum], f"{int(r.n_pairs)}",
            f0(r.baseline_success_rate), ci(r, "delta_success"),
            pfmt(r.delta_success_p), pfmt(r.delta_success_p_holm) if r.stratum != "all" else "n/a",
            f0(r.delta_success_mde80),
        ])
    note3 = ("Bands are retrospective baseline-score groups using realised baseline energy; they are unavailable before treatment. "
             "Parent-block counts are 96 overall, 54 low, 81 intermediate and 72 high. Blocks can span bands. "
             "The 12 controller × band contrasts receive a joint Holm adjustment of exploratory sign-flip p-values. "
             "Intervals are approximate parent-block bootstrap summaries conditional on seed allocation, not exact LHS-design inference. "
             "MDE uses an unadjusted normal approximation. The Monte Carlo minimum p is 1/20,001.")
    docx_table("Table 3. Success change by retrospective baseline-score band.", note3, h3, r3,
               OUT / "Table 3.docx", widths=[1.35, 1.05, 0.45, 0.65, 1.65, 0.65, 0.8, 0.6])
    md_table("Table 3. Success change by retrospective baseline-score band.", note3, h3, r3, OUT / "Table 3.md")

    # ---------------- Supplementary tables
    ends = pd.read_csv(RES / "endpoint_definitions.csv")
    hs = ["Criterion", "Rule", "Meaning"]
    rs = [[r.criterion, r.rule, r.meaning] for _, r in ends.iterrows()]
    notes = ("All fractions are unweighted radial-node fractions at the final time of a 3,000 s run. "
             "They are neither treatment-time nor spherical-volume fractions. max_T is the final spatial maximum. "
             "The stored safe_heat flag does not exclude earlier excursions or establish absence of injury. "
             "Thermal failure excludes cases failing safe_heat; marginal decreases do not imply better heating reliability.")
    docx_table("Table S2. Implemented final-time endpoint definitions.", notes, hs, rs, OUT / "Supplemental Table S2.docx", widths=[1.3, 2.4, 3.0])
    md_table("Table S2. Implemented final-time endpoint definitions.", notes, hs, rs, OUT / "Table S2.md")

    st = pd.read_csv(RES / "strata_audit.csv")
    hs2 = ["Scope", "Seed", "Model family", "n", "AUC", "Observed success", "Mean predicted"]
    rs2 = [[r.scope, r.seed, r.model_family, f"{int(r.n)}", f0(r.auc), f0(r.observed_success), f0(r.mean_predicted)]
           for _, r in st.iterrows()]
    notes2 = ("Retrospective discrimination: the score uses realised baseline average intensity and is not available before treatment. "
              "The external seed was not used to fit the rule. Held-out overall Brier score is 0.265. "
              "Recomputed band calibration is supplied in band_calibration.csv. No prospective predictive validity is claimed.")
    docx_table("Table S1. Discrimination of the frozen scoring rule.", notes2, hs2, rs2,
               OUT / "Supplemental Table S1.docx", widths=[1.5, 0.9, 1.1, 0.4, 0.6, 0.9, 0.9])
    md_table("Table S1. Discrimination of the frozen scoring rule.", notes2, hs2, rs2, OUT / "Table S1.md")

    ws = pd.read_csv(RES / "weight_sensitivity.csv")
    hs3 = ["Controller", "P(rank 1)", "P(top 2)", "P(rank last)", "Mean rank"]
    rs3 = [[r.controller, f0(r.p_rank1), f0(r.p_top2), f0(r.p_rank_last), f0(r.mean_rank, 2)]
           for _, r in ws.iterrows()]
    notes3 = ("200,000 Dirichlet(1,1,1,1,1) draws over five objectives (success, thermal failure, ROS failure, overheating, energy), "
              "min-max normalised across the four alternatives. Ranks are conditional on these objectives, candidates, "
              "normalisation and weight distribution. Pareto membership uses point estimates without uncertainty propagation; "
              "the baseline is a reference rather than a fifth candidate.")
    docx_table("Table S3. Rank stability under random objective weights.", notes3, hs3, rs3,
               OUT / "Supplemental Table S3.docx", widths=[1.7, 0.8, 0.8, 0.9, 0.8])
    md_table("Table S3. Rank stability under random objective weights.", notes3, hs3, rs3, OUT / "Table S3.md")


if __name__ == "__main__":
    print("writing tables to", OUT)
    main()
    import shutil
    submission_tables = HERE.parent / "peerj_submission" / "tables"
    submission_tables.mkdir(exist_ok=True)
    for item in OUT.glob("*.docx"):
        shutil.copy2(item, submission_tables / item.name)
