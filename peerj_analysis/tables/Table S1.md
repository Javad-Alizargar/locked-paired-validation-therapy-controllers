**Table S1. Discrimination of the frozen scoring rule.**

| Scope | Seed | Model family | n | AUC | Observed success | Mean predicted |
| --- | --- | --- | --- | --- | --- | --- |
| locked validation (seeds 20260720-22) | seed_20260720 | baseline | 384 | 0.727 | 0.576 | 0.558 |
| locked validation (seeds 20260720-22) | seed_20260720 | large_geometry | 384 | 0.717 | 0.604 | 0.602 |
| locked validation (seeds 20260720-22) | seed_20260720 | low_photosensitizer | 384 | 0.717 | 0.534 | 0.476 |
| locked validation (seeds 20260720-22) | seed_20260720 | slow_recovery | 384 | 0.681 | 0.620 | 0.534 |
| locked validation (seeds 20260720-22) | seed_20260721 | baseline | 384 | 0.664 | 0.542 | 0.555 |
| locked validation (seeds 20260720-22) | seed_20260721 | large_geometry | 384 | 0.695 | 0.542 | 0.596 |
| locked validation (seeds 20260720-22) | seed_20260721 | low_photosensitizer | 384 | 0.705 | 0.409 | 0.478 |
| locked validation (seeds 20260720-22) | seed_20260721 | slow_recovery | 384 | 0.691 | 0.568 | 0.533 |
| locked validation (seeds 20260720-22) | seed_20260722 | baseline | 384 | 0.590 | 0.625 | 0.556 |
| locked validation (seeds 20260720-22) | seed_20260722 | large_geometry | 384 | 0.607 | 0.633 | 0.599 |
| locked validation (seeds 20260720-22) | seed_20260722 | low_photosensitizer | 384 | 0.702 | 0.474 | 0.469 |
| locked validation (seeds 20260720-22) | seed_20260722 | slow_recovery | 384 | 0.709 | 0.591 | 0.528 |
| external held-out (different seed) | seed_20260713 | baseline | 960 | 0.586 | 0.525 | 0.497 |
| external held-out (different seed) | seed_20260713 | large_geometry | 960 | 0.598 | 0.521 | 0.570 |
| external held-out (different seed) | seed_20260713 | low_photosensitizer | 960 | 0.630 | 0.495 | 0.357 |
| external held-out (different seed) | seed_20260713 | slow_recovery | 960 | 0.668 | 0.463 | 0.543 |
| external held-out (different seed) | seed_20260713 | all | 3840 | 0.610 | 0.501 | 0.492 |

*Discrimination of the frozen scoring rule. The locked validation rows are the analysis population; the external row is a fully held-out seed that was never used to fit or to tune the rule. Discrimination is modest, which is why strata are reported as predicted-baseline-success bands rather than as a validated controllability construct.*
