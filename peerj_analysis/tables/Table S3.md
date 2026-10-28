**Table S3. Rank stability under random objective weights.**

| Controller | P(rank 1) | P(top 2) | P(rank last) | Mean rank |
| --- | --- | --- | --- | --- |
| Thermal-floor predictive | 0.385 | 0.695 | 0.000 | 1.92 |
| Smoothed predictive | 0.279 | 0.470 | 0.160 | 2.41 |
| Constrained optimized | 0.228 | 0.652 | 0.118 | 2.24 |
| Reduced-order MPC | 0.107 | 0.183 | 0.723 | 3.43 |

*200,000 Dirichlet(1,1,1,1,1) draws over five objectives (success, thermal failure, ROS failure, overheating, energy), min-max normalised across the four alternatives. Ranks are conditional on these objectives, candidates, normalisation and weight distribution. Pareto membership uses point estimates without uncertainty propagation; the baseline is a reference rather than a fifth candidate.*
