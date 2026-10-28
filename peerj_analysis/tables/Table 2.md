**Table 2. Failure-mode and energy endpoints.**

| Controller | Thermal failure | ROS failure | Overheating | Delivered energy (J cm⁻²) | Pareto status |
| --- | --- | --- | --- | --- | --- |
| Smoothed predictive | +0.038 (0.016 to 0.060) | -0.023 (-0.036 to -0.013) | -0.012 (-0.024 to -0.000) | -23.5 (-29.5 to -17.5) | Non-dominated |
| Thermal-floor predictive | -0.037 (-0.067 to -0.006) | -0.017 (-0.028 to -0.007) | +0.095 (0.081 to 0.110) | +44.2 (36.3 to 52.5) | Non-dominated |
| Constrained optimized | -0.010 (-0.042 to 0.021) | -0.002 (-0.012 to 0.009) | +0.020 (0.008 to 0.031) | -59.1 (-63.4 to -54.8) | Non-dominated |
| Reduced-order MPC | -0.048 (-0.081 to -0.018) | -0.054 (-0.072 to -0.037) | +0.497 (0.471 to 0.523) | +549.7 (521.6 to 578.3) | Non-dominated |

*Values are paired changes in the endpoint rate (or in delivered energy) against the full-feedback baseline, with 95% cluster-bootstrap intervals (10,000 replicates). Negative failure changes indicate fewer failures. Hypoxic failure did not change for any controller (all deltas 0.000, 95% CI 0.000 to 0.000) and is omitted. Pareto status is computed over the five objectives shown in Table 1 and this table; all four controllers are mutually non-dominated.*
