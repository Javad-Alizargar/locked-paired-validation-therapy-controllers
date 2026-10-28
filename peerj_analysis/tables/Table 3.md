**Table 3. Success change by predicted-baseline-success stratum.**

| Controller | Stratum | Pairs | Baseline success | Change in success (95% CI) | p | Holm-adjusted p | MDE (80% power) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Smoothed predictive | All | 4608 | 0.560 | -0.018 (-0.041 to 0.005) | 0.121 | n/a | 0.033 |
| Thermal-floor predictive | All | 4608 | 0.560 | -0.049 (-0.081 to -0.016) | 0.004 | n/a | 0.046 |
| Constrained optimized | All | 4608 | 0.560 | -0.007 (-0.037 to 0.024) | 0.659 | n/a | 0.044 |
| Reduced-order MPC | All | 4608 | 0.560 | -0.432 (-0.472 to -0.391) | <0.001 | n/a | 0.058 |
| Smoothed predictive | Predicted-low | 1122 | 0.383 | +0.031 (-0.012 to 0.075) | 0.181 | 0.629 | 0.062 |
| Thermal-floor predictive | Predicted-low | 1122 | 0.383 | +0.056 (-0.011 to 0.122) | 0.126 | 0.629 | 0.095 |
| Constrained optimized | Predicted-low | 1122 | 0.383 | +0.050 (-0.012 to 0.112) | 0.132 | 0.629 | 0.089 |
| Reduced-order MPC | Predicted-low | 1122 | 0.383 | -0.210 (-0.283 to -0.140) | <0.001 | <0.001 | 0.102 |
| Smoothed predictive | Predicted-intermediate | 1674 | 0.477 | -0.004 (-0.040 to 0.033) | 0.872 | 1.000 | 0.052 |
| Thermal-floor predictive | Predicted-intermediate | 1674 | 0.477 | -0.042 (-0.086 to 0.002) | 0.060 | 0.360 | 0.062 |
| Constrained optimized | Predicted-intermediate | 1674 | 0.477 | -0.001 (-0.044 to 0.042) | 0.980 | 1.000 | 0.062 |
| Reduced-order MPC | Predicted-intermediate | 1674 | 0.477 | -0.351 (-0.409 to -0.293) | <0.001 | <0.001 | 0.084 |
| Smoothed predictive | Predicted-high | 1812 | 0.746 | -0.063 (-0.096 to -0.028) | <0.001 | 0.002 | 0.049 |
| Thermal-floor predictive | Predicted-high | 1812 | 0.746 | -0.119 (-0.164 to -0.073) | <0.001 | <0.001 | 0.065 |
| Constrained optimized | Predicted-high | 1812 | 0.746 | -0.047 (-0.090 to -0.003) | 0.035 | 0.245 | 0.062 |
| Reduced-order MPC | Predicted-high | 1812 | 0.746 | -0.645 (-0.691 to -0.596) | <0.001 | <0.001 | 0.069 |

*Strata are terciles of a frozen logistic scoring rule for predicted baseline success. The rule was fitted on independent seeds and applied unchanged here, so stratum membership depends only on pre-treatment design quantities and never on the controller outcome. The 12 controller x stratum tests form the primary family for Holm adjustment. Exact p-values are reported to three decimal places; the minimum attainable p-value with 20,000 permutations is 0.00005.*
