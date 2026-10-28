"""Rebuild every analysis, table, figure and manuscript from committed data."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
for script in ["peerj_analysis/run_all.py", "peerj_analysis/make_tables.py",
               "peerj_analysis/make_figures.py", "peerj_submission/build_docx.py",
               "peerj_submission/build_docx_review.py", "peerj_submission/make_final_package.py"]:
    subprocess.run([sys.executable, "-B", str(ROOT / script)], cwd=ROOT, check=True)
