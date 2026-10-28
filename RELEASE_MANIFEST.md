# Release manifest

Release: `v1.0.0`
Date: 2026-10-28
Article: *Locked paired validation of closed-loop therapy software: controller
rankings are not robust to analysis choices* (PeerJ Computer Science, Research
Article)

This manifest defines the contents of the public reproducibility package. The
package is a curated subset of the private working repository: it contains
everything needed to regenerate every reported number, table and figure, and
nothing that is not needed.

## Included

### Manuscript

- `peerj_submission/main_manuscript.md` — manuscript source
- `peerj_submission/main_manuscript.docx` — submission file
- `peerj_submission/build_docx.py` — deterministic builder
- `peerj_submission/title_page.md`
- `peerj_submission/cover_letter.md`
- `peerj_submission/figure_captions.md`
- `peerj_submission/declarations.md`

### Analysis

- `peerj_analysis/analysis.py` — all statistics
- `peerj_analysis/run_all.py` — entry point
- `peerj_analysis/make_figures.py`
- `peerj_analysis/make_tables.py`
- `peerj_analysis/results/` — machine-readable results
- `peerj_analysis/figures/` — `Figure 1.png` … `Figure 5.png`
- `peerj_analysis/tables/` — `Table 1–3`, `Table S1–S3`

### Data

- `nature_computational_science/data/controller_pareto_synthesis/controller_family_locked_pairs.csv`
  — primary dataset, 18,432 × 92, SHA-256 `D8D59D86691039F30336ED4FF00A72643E3BD9475BB16856AF5DE67986B1174C`
- `nature_computational_science/data/controller_pareto_synthesis/controller_family_pareto_summary.csv`
- `nature_computational_science/data/controller_pareto_synthesis/controller_pareto_synthesis_metadata.json`
- `nature_computational_science/data/external_boundary_validation/` — held-out discrimination
- `nature_computational_science/data/multiseed_controllability/` — seed-level strata audit
- `nature_computational_science/data/predictive_control_reclassification/` — out-of-seed scored trials
- `nature_computational_science/data/constrained_policy_optimization/optimized_policy_metadata.json`
  — provenance of the tuned controller

### Code that produces the data

- `nature_computational_science/scripts/run_step12_low_class_failure_mode_decomposition.py`
- `nature_computational_science/scripts/run_step13_locked_large_validation_atlas.py`
- `nature_computational_science/scripts/run_step14_thermal_floor_predictive_validation.py`
- `nature_computational_science/scripts/run_step15_constrained_policy_optimization.py`
- `nature_computational_science/scripts/run_step16_reduced_mpc_pareto_validation.py`
- `nature_computational_science/scripts/run_step17_pareto_synthesis.py`

### Simulation model and controllers

- `scripts/run_first_model.py` — coupled Pennes bioheat + ROS model
- `scripts/run_predictive_delay_control.py` — smoothed and thermal-floor predictive controllers
- `scripts/run_feedback_ablation_noise_delay.py` — delay/noise ablation
- `scripts/run_robust_chance_optimization.py` — chance-constrained controller search

### Repository metadata

- `README.md`
- `REPRODUCIBILITY.md`
- `requirements.txt`
- `LICENSE` (MIT)
- `CITATION.cff`
- `.zenodo.json`
- `.gitignore`

## Excluded

- `.git/`, `__pycache__/`, `.DS_Store`, virtual environments, local logs
- Literature-search scaffolding (`data/raw/`, `data/processed/`,
  `nature_computational_science/data/literature_api/`) — API snapshots used
  during reference discovery, not required to reproduce any reported number
- Internal audit notes (`notes/`, `nature_computational_science/notes/`)
- Superseded figure drafts (`figures/`)
- Exploratory pipeline stages not used by the article
  (`run_step1`–`run_step11`)

## Release steps

1. `python peerj_analysis/run_all.py`
2. `python peerj_analysis/make_figures.py`
3. `python peerj_analysis/make_tables.py`
4. `python peerj_submission/build_docx.py`
5. Verify `controller_family_locked_pairs.csv` SHA-256 matches the value above.
6. Tag `v1.0.0` and publish the GitHub release.
7. Archive the release on Zenodo and record the DOI in
   `peerj_submission/main_manuscript.md` (Data availability),
   `peerj_submission/declarations.md`, `CITATION.cff` and `README.md`.

## Persistent identifiers

- GitHub repository: `https://github.com/Javad-Alizargar/locked-paired-validation-therapy-controllers`
- GitHub release tag: `v1.0.0`
- Zenodo DOI: `10.5281/zenodo.23726433`
- Zenodo concept DOI (all versions): `10.5281/zenodo.23726432`
