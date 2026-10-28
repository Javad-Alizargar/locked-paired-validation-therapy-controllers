**Table 3. Success change by retrospective baseline-score band.**

| Controller | Stratum | Pairs | Baseline success | Change in success (95% CI) | p | Holm-adjusted p | Approx. MDE (80%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Smoothed predictive | All | 4608 | 0.560 | -0.018 (-0.051 to 0.013) | 0.272 | n/a | 0.046 |
| Thermal-floor predictive | All | 4608 | 0.560 | -0.049 (-0.097 to 0.001) | 0.061 | n/a | 0.070 |
| Constrained optimized | All | 4608 | 0.560 | -0.007 (-0.046 to 0.033) | 0.737 | n/a | 0.056 |
| Reduced-order MPC | All | 4608 | 0.560 | -0.432 (-0.497 to -0.365) | <0.001 | n/a | 0.093 |
| Smoothed predictive | Low score | 1122 | 0.383 | +0.031 (-0.025 to 0.087) | 0.302 | 1.000 | 0.080 |
| Thermal-floor predictive | Low score | 1122 | 0.383 | +0.056 (-0.040 to 0.149) | 0.297 | 1.000 | 0.135 |
| Constrained optimized | Low score | 1122 | 0.383 | +0.050 (-0.024 to 0.122) | 0.215 | 1.000 | 0.103 |
| Reduced-order MPC | Low score | 1122 | 0.383 | -0.210 (-0.319 to -0.111) | <0.001 | 0.002 | 0.148 |
| Smoothed predictive | Intermediate score | 1674 | 0.477 | -0.004 (-0.054 to 0.047) | 0.913 | 1.000 | 0.072 |
| Thermal-floor predictive | Intermediate score | 1674 | 0.477 | -0.042 (-0.098 to 0.015) | 0.166 | 0.997 | 0.081 |
| Constrained optimized | Intermediate score | 1674 | 0.477 | -0.001 (-0.051 to 0.048) | 0.983 | 1.000 | 0.071 |
| Reduced-order MPC | Intermediate score | 1674 | 0.477 | -0.351 (-0.433 to -0.270) | <0.001 | <0.001 | 0.116 |
| Smoothed predictive | High score | 1812 | 0.746 | -0.063 (-0.107 to -0.016) | 0.015 | 0.121 | 0.065 |
| Thermal-floor predictive | High score | 1812 | 0.746 | -0.119 (-0.183 to -0.051) | 0.002 | 0.021 | 0.094 |
| Constrained optimized | High score | 1812 | 0.746 | -0.047 (-0.103 to 0.012) | 0.130 | 0.910 | 0.082 |
| Reduced-order MPC | High score | 1812 | 0.746 | -0.645 (-0.711 to -0.573) | <0.001 | <0.001 | 0.098 |

*Bands are retrospective baseline-score groups using realised baseline energy; they are unavailable before treatment. Parent-block counts are 96 overall, 54 low, 81 intermediate and 72 high. Blocks can span bands. The 12 controller × band contrasts receive a joint Holm adjustment of exploratory sign-flip p-values. Intervals are approximate parent-block bootstrap summaries conditional on seed allocation, not exact LHS-design inference. MDE uses an unadjusted normal approximation. The Monte Carlo minimum p is 1/20,001.*
