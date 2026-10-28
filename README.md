# Locked paired validation of closed-loop therapy software

Reproducibility package for the PeerJ Computer Science Research Article:

> **Locked paired validation of closed-loop therapy software: controller rankings are not robust to analysis choices**
> Javad Alizargar, Department of Nursing, National Taipei University of Nursing and Health Sciences

## What this repository contains

The study asks whether four delayed-feedback controllers for coupled
photothermal–photodynamic simulation remain superior to delayed full feedback
once the comparison is paired, cluster-robust, multiplicity-controlled, and
tested across a curve of defensible analysis specifications.

The answer is negative, and the negative answer is the contribution. No
controller improved overall success. The apparent benefit in the
predicted-low stratum (+0.031 to +0.056) did not survive Holm correction
(adjusted *p* = 0.629 in every case). All four controllers were Pareto
non-dominated, and the sign of the headline effect changed across 16
specifications for three of the four controllers.

The transferable contribution is the protocol: a locked validation atlas, a
frozen scoring rule, cluster-bootstrap intervals, exact sign-flip permutation
tests, declared multiplicity families, minimum detectable effects, a
specification curve, and an explicit Pareto statement.

## Repository layout

```
peerj_analysis/                     Analysis code and regenerated outputs
  analysis.py                       All statistics (bootstrap, permutation, Holm, MDE,
                                    specification curve, Pareto, weight sensitivity)
  make_figures.py                   Figure 1-5
  make_tables.py                    Tables 1-3 and S1-S3
  run_all.py                        Entry point: regenerates every result file
  results/                          Machine-readable results (CSV/JSON)
  figures/                          Figure 1.png - Figure 5.png (300 dpi)
  tables/                           Table 1-3, Table S1-S3 (.md and .docx)

nature_computational_science/
  data/controller_pareto_synthesis/
      controller_family_locked_pairs.csv     Primary dataset (18,432 paired rows)
  data/external_boundary_validation/         Held-out discrimination of the scoring rule
  data/multiseed_controllability/            Seed-level stratification audit
  data/predictive_control_reclassification/  Out-of-seed scored controller trials
  data/constrained_policy_optimization/      Training metadata for the tuned controller
  scripts/run_step*.py                       Pipeline that produces each dataset
  scripts/run_step17_pareto_synthesis.py     Regenerates the primary dataset

scripts/                            Simulation model and controller implementations
  run_first_model.py                Coupled Pennes bioheat + ROS model
  run_predictive_delay_control.py   Smoothed and thermal-floor predictive controllers
  run_feedback_ablation_noise_delay.py
  run_robust_chance_optimization.py

peerj_submission/                   Manuscript source and build
  main_manuscript.md                Markdown source
  main_manuscript.docx              Submission file (line numbers, 12 pt Times)
  build_docx.py                     Deterministic .docx builder
  title_page.md, cover_letter.md, figure_captions.md, declarations.md
```

## Reproducing the analysis

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python peerj_analysis/run_all.py      # results/*.csv, results/*.json
python peerj_analysis/make_figures.py # figures/Figure 1-5.png
python peerj_analysis/make_tables.py  # tables/Table 1-3, S1-S3
python peerj_submission/build_docx.py # peerj_submission/main_manuscript.docx
```

`run_all.py` reads only committed trial-level data and writes every number
reported in the manuscript. The resampling seed is fixed
(`SEED = 20261028`), so the bootstrap and permutation outputs are deterministic.

Regenerating the primary dataset itself (rather than the analysis of it) is
done with:

```bash
python nature_computational_science/scripts/run_step17_pareto_synthesis.py
```

This reproduces `controller_family_locked_pairs.csv` byte-for-byte
(SHA-256 `D8D59D86…B1174C`). The upstream controller validations
(`run_step13`–`run_step16`) are not re-run by default because the full atlas
is 9,216 simulations and takes several hours; their outputs are committed.

## Design summary

| Element | Value |
|---|---|
| Validation seeds | 20260720, 20260721, 20260722 |
| Model families | baseline, large geometry, low photosensitiser, slow recovery |
| Uncertainty draws | 32 per seed × family (Latin hypercube) |
| Sensing scenarios | 6 delays (0–180 s) × 2 noise levels (0%, 10%) |
| Trials per controller | 4,608 |
| Clusters | 384 (seed × family × draw), 12 paired trials each |
| Bootstrap replicates | 10,000 |
| Permutation vectors | 20,000 |
| Multiplicity families | 12 tests (4 controllers × 3 bands); 20 tests (5 endpoints × 4 controllers) |
| Specification curve | 16 specifications per controller |

## Key results

Paired change in success probability against delayed full feedback
(all trials, *n* = 4,608 per controller):

| Controller | Δ success | 95% CI | *p* | MDE (80%) |
|---|---|---|---|---|
| Smoothed predictive | −0.018 | −0.041 to 0.005 | 0.121 | 0.033 |
| Thermal-floor predictive | −0.049 | −0.081 to −0.016 | 0.004 | 0.046 |
| Constrained optimized | −0.007 | −0.037 to 0.024 | 0.659 | 0.044 |
| Reduced-order MPC | −0.432 | −0.472 to −0.391 | <0.001 | 0.058 |

Predicted-low stratum (*n* = 1,122; baseline success 0.383):

| Controller | Δ success | 95% CI | *p* | Holm |
|---|---|---|---|---|
| Smoothed predictive | +0.031 | −0.012 to 0.075 | 0.181 | 0.629 |
| Thermal-floor predictive | +0.056 | −0.011 to 0.122 | 0.126 | 0.629 |
| Constrained optimized | +0.050 | −0.012 to 0.112 | 0.132 | 0.629 |
| Reduced-order MPC | −0.210 | −0.283 to −0.140 | <0.001 | <0.001 |

## Data availability

The trial-level paired dataset, the controller definitions, the frozen scoring
rule, the endpoint definitions and the analysis code are archived with a
persistent DOI (see `CITATION.cff`).

## License

Code and data are released under the MIT License. See `LICENSE`.

## Contact

Javad Alizargar
Department of Nursing
National Taipei University of Nursing and Health Sciences
No. 365, Mingde Rd, Beitou District, Taipei City 112, Taiwan
Email: 8javad@ntunhs.edu.tw
ORCID: 0000-0003-2966-1315
