# Reproducibility

This document specifies exactly how every number, table and figure in the
manuscript is regenerated from committed data.

## Environment

- Python 3.12
- NumPy, pandas, SciPy, Matplotlib, python-docx (see `requirements.txt`)
- No GPU, no network access and no external data are required for the analysis

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## One-command reproduction

```bash
python peerj_analysis/run_all.py
python peerj_analysis/make_figures.py
python peerj_analysis/make_tables.py
python peerj_submission/build_docx.py
```

Outputs:

| Command | Writes |
|---|---|
| `run_all.py` | `peerj_analysis/results/*.csv`, `results/analysis_summary.json` |
| `make_figures.py` | `peerj_analysis/figures/Figure 1.png` … `Figure 5.png` |
| `make_tables.py` | `peerj_analysis/tables/Table 1–3`, `Table S1–S3` (`.md`, `.docx`) |
| `build_docx.py` | `peerj_submission/main_manuscript.docx` |

## Determinism

All resampling is seeded:

```python
B_BOOT = 10_000        # cluster bootstrap replicates
B_SIGN = 20_000        # sign-flip permutation vectors
SEED   = 20261028      # resampling seed
```

The bootstrap and permutation results are therefore reproducible exactly on
the same platform. NumPy's PCG64 generator is used, so results are stable
across NumPy versions that preserve the bit generator stream.

## Data provenance

The analysis reads one primary file:

```
nature_computational_science/data/controller_pareto_synthesis/controller_family_locked_pairs.csv
```

18,432 rows = 4 controllers × 4,608 paired trials; 92 columns.

`peerj_analysis/analysis.py::load_paired` enforces a set of contract assertions
before any statistic is computed, so a corrupted or truncated file fails loudly
rather than producing a plausible wrong number:

1. `success_uplift` is a signed indicator difference in {−1, 0, +1}.
2. `success_uplift == to_chance_success_int − from_chance_success_int`.
3. `success_uplift == rescued_success − lost_success`.
4. Every mechanism delta equals `to_flag − from_flag`.
5. Every cluster is balanced at exactly 12 trials.

## Regenerating the primary dataset

```bash
python nature_computational_science/scripts/run_step17_pareto_synthesis.py
```

This reproduces `controller_family_locked_pairs.csv` byte-for-byte:

```
SHA-256  D8D59D86691039F30336ED4FF00A72643E3BD9475BB16856AF5DE67986B1174C
```

The upstream controller validations are:

| Script | Produces |
|---|---|
| `run_step13_locked_large_validation_atlas.py` | Smoothed predictive pairs |
| `run_step14_thermal_floor_predictive_validation.py` | Thermal-floor predictive pairs |
| `run_step15_constrained_policy_optimization.py` | Constrained optimized pairs + training metadata |
| `run_step16_reduced_mpc_pareto_validation.py` | Reduced-order MPC pairs |

These are not re-run by default: the atlas is 9,216 simulations and takes
several hours serially. Their outputs are committed under
`nature_computational_science/data/`.

## Frozen scoring rule

The stratification rule is reproduced in `analysis.py` from the constants
recorded in the validation outputs:

```
logit(P) = -0.016 + 1.406*z1 + 0.930*z2 - 1.482*z3
z1: mean -0.625, sd 0.549   (log10 dose-capacity number)
z2: mean  0.319, sd 0.307   (log10 reaction number × observation quality)
z3: mean -3.530, sd 0.576   (log10 reaction-drive balance)
band boundaries: 0.364 and 0.595
```

The rule reproduces the stored per-trial predicted probabilities to a maximum
absolute error of 1.3 × 10⁻¹², which is floating-point round-off. Band
boundaries are the 33rd and 67th percentiles of the reference population, not
of the validation population; the resulting band sizes are 1,122 / 1,674 /
1,812 rather than equal thirds.

## Statistical pipeline

1. **Estimand.** Ratio of summed within-cluster paired differences to the
   total trial count. This equals the difference in success proportions and
   remains valid when clusters are unbalanced after stratification.
2. **Interval.** 95% cluster bootstrap, 10,000 resamples of the 384 clusters.
3. **Test.** Cluster-level sign-flip permutation, 20,000 sign vectors, two-sided,
   observed value included in the reference set. Exact under exchangeability;
   no degrees of freedom.
4. **Multiplicity.** Holm over 12 controller × band tests (success) and over
   20 controller × endpoint tests (mechanism and burden).
5. **Precision.** MDE at 80% power = 2.802 × cluster-robust SE, where
   2.802 = z₀.₉₇₅ + z₀.₈₀.
6. **Clustering cost.** Design effect = cluster-robust variance of the paired
   difference ÷ simple-random-sampling variance (range 2.56–5.54).
7. **Specification curve.** 16 specifications per controller: four seed subsets
   × four scenario subsets, each evaluated with the paired estimator.
8. **Pareto.** Five objectives (success, thermal failure, ROS failure,
   overheating, energy); dominance computed pairwise.
9. **Weights.** 200,000 Dirichlet draws over min–max-normalised objectives;
   rank distribution recorded.

## Figures

| File | Content | Size (px) |
|---|---|---|
| `Figure 1.png` | Locked protocol schematic | 1780 × 1076 |
| `Figure 2.png` | Paired success change by band | 2605 × 1379 |
| `Figure 3.png` | Failure-mode and energy endpoints | 2207 × 935 |
| `Figure 4.png` | Specification curve | 2208 × 1023 |
| `Figure 5.png` | Pareto structure and rank stability | 2268 × 1056 |

All are 300 dpi and within the journal's 900–3000 px requirement.

## Manuscript build

`peerj_submission/build_docx.py` produces the submission file with:

- US Letter, 1 inch margins
- 12 pt Times New Roman, left justified, 1.5 line spacing
- continuous line numbers (`w:lnNumType`)
- unnumbered sections
- author–year citations

The builder is deterministic: re-running it on the same markdown produces the
same document body (word-token count 7,064).
