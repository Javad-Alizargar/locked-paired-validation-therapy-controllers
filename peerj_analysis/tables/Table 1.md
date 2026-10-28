**Table 1. Success endpoints across the full locked validation atlas.**

| Controller | Pairs | Baseline success | Controller success | Change in success (95% CI) | p | MDE (80% power) | Rescued / lost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Smoothed predictive | 4608 | 0.560 | 0.541 | -0.018 (-0.041 to 0.005) | 0.121 | 0.033 | 535 / 620 |
| Thermal-floor predictive | 4608 | 0.560 | 0.511 | -0.049 (-0.081 to -0.016) | 0.004 | 0.046 | 688 / 912 |
| Constrained optimized | 4608 | 0.560 | 0.553 | -0.007 (-0.037 to 0.024) | 0.659 | 0.044 | 703 / 735 |
| Reduced-order MPC | 4608 | 0.560 | 0.127 | -0.432 (-0.472 to -0.391) | <0.001 | 0.058 | 273 / 2265 |

*All 4,608 paired trials per controller (384 clusters; 12 scenarios per cluster). Change is the paired risk difference against the same full-feedback baseline, with a 95% cluster-bootstrap interval (10,000 replicates) and a cluster-level sign-flip p-value (20,000 permutations). MDE is the minimum detectable effect at 80% power given the observed cluster-robust standard error. These all-stratum contrasts are descriptive; the pre-declared multiplicity-controlled family is the 12 controller x stratum tests reported in Table 3.*
