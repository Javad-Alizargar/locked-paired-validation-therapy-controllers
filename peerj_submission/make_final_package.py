"""Assemble the PeerJ Computer Science submission package into one folder.

Everything the journal needs is copied/converted into a single flat
directory, named so that each file's upload slot is obvious:

  Main manuscript.docx                          -> Manuscript
  Figure 1..5.png                               -> Figures (one per file)
  Table 1..3.docx                               -> Tables (one per file)
  Supplemental Table S1..S3.docx                -> Supplemental files
  Review-only - manuscript with figures inline.docx -> Review-only supplemental
  Cover letter.docx                             -> Cover letter / paste into form
  Figure captions.docx, Declarations.docx       -> optional reference copies

Usage:  python make_final_package.py [destination]
Default destination: D:\\PeerJ Computer Science\\PeerJ C S final
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Pt

from build_docx import add_runs, configure

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "peerj_analysis"
DEFAULT_DEST = Path(r"D:\PeerJ Computer Science\PeerJ C S final")


# --------------------------------------------------------------- conversions
def md_to_docx(src: Path, dest: Path) -> None:
    """Render a simple markdown document (headings, bullets, paragraphs)."""
    doc = Document()
    configure(doc)
    for raw in src.read_text(encoding="utf-8").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# "):
            continue  # document heading; the file name carries the meaning
        if line.startswith("## "):
            par = doc.add_paragraph()
            par.paragraph_format.space_before = Pt(12)
            par.paragraph_format.space_after = Pt(4)
            par.paragraph_format.keep_with_next = True
            run = par.add_run(line[3:].strip())
            run.bold = True
        elif line.startswith("### "):
            par = doc.add_paragraph()
            par.paragraph_format.space_before = Pt(10)
            par.paragraph_format.space_after = Pt(3)
            par.paragraph_format.keep_with_next = True
            run = par.add_run(line[4:].strip())
            run.bold = True
            run.italic = True
        elif line.startswith("- "):
            par = doc.add_paragraph(style="List Bullet")
            add_runs(par, line[2:].strip())
        else:
            par = doc.add_paragraph()
            add_runs(par, line.strip())
    doc.save(dest)


def build(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[Path, Path]] = []

    def put(src: Path, name: str) -> None:
        target = dest / name
        shutil.copy2(src, target)
        copied.append((src, target))

    # Manuscript of record
    put(HERE / "main_manuscript.docx", "Main manuscript.docx")

    # Reviewer convenience copy (upload as Review-only Supplemental File)
    put(HERE / "main_manuscript_review_copy.docx",
        "Review-only - manuscript with figures and tables inline.docx")

    # Figures, one file per figure, number in the filename (section 4.6)
    for n in range(1, 6):
        put(ANALYSIS / "figures" / f"Figure {n}.png", f"Figure {n}.png")

    # Tables, one file per table (section 4.8)
    for n in (1, 2, 3):
        put(HERE / "tables" / f"Table {n}.docx", f"Table {n}.docx")
    for n in ("S1", "S2", "S3"):
        put(HERE / "tables" / f"Supplemental Table {n}.docx",
            f"Supplemental Table {n}.docx")

    # Editorial documents, converted to Word
    md_to_docx(HERE / "cover_letter.md", dest / "Cover letter.docx")
    md_to_docx(HERE / "figure_captions.md", dest / "Figure captions.docx")
    md_to_docx(HERE / "declarations.md", dest / "Declarations.docx")

    # Verify every copied file is byte-identical to its source
    import hashlib

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    print(f"package: {dest}\n")
    for src, target in copied:
        ok = "OK " if sha(src) == sha(target) else "MISMATCH"
        print(f"  [{ok}] {target.name}")
    print(f"\n{len(copied)} files")


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DEST)
