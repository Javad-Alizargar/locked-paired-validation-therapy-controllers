**Table 1. Success endpoints across the full simulation atlas.**

| Controller | Pairs | Baseline success | Controller success | Change in success (95% CI) | p | Approx. MDE (80%) | Rescued / lost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Smoothed predictive | 4608 | 0.560 | 0.541 | -0.018 (-0.051 to 0.013) | 0.272 | 0.046 | 535 / 620 |
| Thermal-floor predictive | 4608 | 0.560 | 0.511 | -0.049 (-0.097 to 0.001) | 0.061 | 0.070 | 688 / 912 |
| Constrained optimized | 4608 | 0.560 | 0.553 | -0.007 (-0.046 to 0.033) | 0.737 | 0.056 | 703 / 735 |
| Reduced-order MPC | 4608 | 0.560 | 0.127 | -0.432 (-0.497 to -0.365) | <0.001 | 0.093 | 273 / 2265 |

*All 4,608 paired trials per controller (96 parent blocks; four families × 12 scenarios per block). Change is the paired risk difference against the same full-feedback baseline, with a 95% approximate seed-stratified parent-block bootstrap interval (10,000 replicates) and an exploratory sign-flip p-value (20,000 sign vectors). MDE is an approximate unadjusted normal-theory precision summary at 80% power, not a guarantee for Holm or sign-flip testing. It uses the observed bootstrap standard error. Overall contrasts are descriptive; the declared exploratory multiplicity family is the 12 controller × band contrasts in Table 3.*
