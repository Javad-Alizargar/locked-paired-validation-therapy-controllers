# Reproducible paired evaluation of therapy controllers

Reproducible paired evaluation of closed-loop therapy software: controller comparisons depend on sampling and scenario choices

Four alternatives are evaluated on the existing 4,608 paired trials each. The revised analysis retains 96 shared-parent blocks (four family variants and 12 scenarios per block), resampled within the three realised seeds. Bands use realised baseline energy and are retrospective. Endpoint fractions are final-time unweighted radial-node fractions over a 3,000 s horizon. No alternative demonstrates improved overall composite success. Bootstrap intervals and sign-flip tests are approximate exploratory summaries with explicit Latin-hypercube and sign-symmetry limitations.

## Reproduce

Use Python 3.12, install requirements-revision.txt and run:

```
python reproduce_submission.py
python peerj_analysis/test_revision.py
```

The reproduction script builds statistics, six tables, five figures, the submission manuscript, reviewer copy and upload package. run_all.py alone writes CSV/JSON statistics. PEERJ_PACKAGE_DIR selects the upload directory; otherwise the established Windows path is used if present, or peerj_submission/upload_package. See REPRODUCIBILITY.md and REVISION_NOTES.md for assumptions and audit resolution.

## Availability and version history

Original data and earlier v1.0.0 are archived at https://doi.org/10.5281/zenodo.23726433. That DOI does not identify the revised manuscript. Revision release v1.1.1 supplies the corrected sources, results and complete Supplemental Code S1 ZIP for journal review. Do not overwrite the original archive or claim a new DOI before it resolves. Repository visibility is unchanged.

## Package

peerj_analysis contains the inference, parent-draw mapping, diagnostic results, regression checks, tables and figures. peerj_submission contains manuscript sources and builders. The original simulation code and data remain unchanged under scripts and nature_computational_science. release_files.txt is the explicit public package allowlist. No local credentials, Git internals or QA renders are included.
