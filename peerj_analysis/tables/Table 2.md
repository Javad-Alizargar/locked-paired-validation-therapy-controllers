**Table 2. Failure-mode and energy endpoints.**

| Controller | Thermal failure | ROS failure | Overheating | Delivered energy (J cm⁻²) | Pareto status |
| --- | --- | --- | --- | --- | --- |
| Smoothed predictive | +0.038 (0.009 to 0.069) | -0.023 (-0.041 to -0.009) | -0.012 (-0.025 to 0.003) | -23.5 (-33.6 to -13.2) | Non-dominated |
| Thermal-floor predictive | -0.037 (-0.081 to 0.006) | -0.017 (-0.031 to -0.005) | +0.095 (0.074 to 0.116) | +44.2 (31.1 to 57.8) | Non-dominated |
| Constrained optimized | -0.010 (-0.052 to 0.030) | -0.002 (-0.015 to 0.010) | +0.020 (0.008 to 0.031) | -59.1 (-65.3 to -53.0) | Non-dominated |
| Reduced-order MPC | -0.048 (-0.093 to -0.003) | -0.054 (-0.079 to -0.031) | +0.497 (0.456 to 0.538) | +549.7 (498.2 to 603.5) | Non-dominated |

*Values are paired changes in the endpoint rate (or in delivered energy) against the full-feedback baseline, with approximate 95% seed-stratified parent-block bootstrap intervals (10,000 replicates). Negative failure changes indicate fewer failures. Hypoxic failure did not change for any controller (all deltas 0.000, 95% CI 0.000 to 0.000) and is omitted. Pareto status is computed over the five objectives shown in Table 1 and this table; all four controllers are mutually non-dominated.*
