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
import os
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Pt

from build_docx import add_runs, configure

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "peerj_analysis"
WINDOWS_DEST = Path(r"D:\PeerJ Computer Science\PeerJ C S final")
DEFAULT_DEST = Path(os.environ.get("PEERJ_PACKAGE_DIR", str(WINDOWS_DEST if os.name == "nt" and WINDOWS_DEST.parent.exists() else HERE / "upload_package")))


# --------------------------------------------------------------- conversions
def md_to_docx(src: Path, dest: Path) -> None:
    """Render a simple markdown document (headings, bullets, paragraphs)."""
    doc = Document()
    configure(doc)
    # Ancillary editorial files use readable single spacing; manuscript builders
    # retain the journal's 1.5 spacing.
    doc.styles["Normal"].paragraph_format.line_spacing = 1.0
    doc.styles["Normal"].font.size = Pt(11)
    doc.styles["Normal"].paragraph_format.space_after = Pt(4)
    for raw in src.read_text(encoding="utf-8").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# "):
            continue  # document heading; the file name carries the meaning
        if line.startswith("## "):
            par = doc.add_paragraph()
            par.paragraph_format.space_before = Pt(8)
            par.paragraph_format.space_after = Pt(3)
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
        put(ANALYSIS / "tables" / f"Table {n}.docx", f"Table {n}.docx")
    for n in ("S1", "S2", "S3"):
        put(ANALYSIS / "tables" / f"Supplemental Table {n}.docx",
            f"Supplemental Table {n}.docx")

    # Editorial documents, converted to Word
    md_to_docx(HERE / "cover_letter.md", dest / "Cover letter.docx")
    md_to_docx(HERE / "figure_captions.md", dest / "Figure captions.docx")
    md_to_docx(HERE / "declarations.md", dest / "Declarations.docx")
    root = HERE.parent
    put(root / "REFERENCE_AUDIT.md", "Review-only - reference and claim audit.md")
    put(root / "reference_audit" / "reference_verification.csv", "Review-only - reference verification.csv")
    manifest = root / "release_files.txt"
    with zipfile.ZipFile(dest / "Supplemental Code S1.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest.read_text(encoding="utf-8").splitlines():
            if not name or name.startswith("#"):
                continue
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or ".git" in path.parts:
                raise ValueError(f"Invalid release path: {name}")
            archive.write(path, name)
    (dest / "00_UPLOAD_GUIDE.md").write_text(
        "# Revised PeerJ Computer Science upload package\n\n"
        "Revision v1.1.1. Upload Main manuscript.docx as the manuscript; Figure 1–5.png separately; "
        "Table 1–3.docx separately; Supplemental Table S1–S3.docx and Supplemental Code S1.zip as supplements. "
        "The Review-only manuscript is a reviewer convenience file. Paste figure legends from Figure captions.docx. "
        "Use Declarations.docx for submission fields and Cover letter.docx for the cover letter.\n\n"
        "The original DOI 10.5281/zenodo.23726433 identifies v1.0.0, not this revision. "
        "The full revised package is supplied as a supplement; do not substitute an unverified DOI. "
        "See REVISION_NOTES.md in Supplemental Code S1 for the sixteen review fixes.\n",
        encoding="utf-8")

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
