**Table S3. Rank stability under random objective weights.**

| Controller | P(rank 1) | P(top 2) | P(rank last) | Mean rank |
| --- | --- | --- | --- | --- |
| Thermal-floor predictive | 0.385 | 0.695 | 0.000 | 1.92 |
| Smoothed predictive | 0.279 | 0.470 | 0.160 | 2.41 |
| Constrained optimized | 0.228 | 0.652 | 0.118 | 2.24 |
| Reduced-order MPC | 0.107 | 0.183 | 0.723 | 3.43 |

*200,000 draws from a uniform Dirichlet prior over the five objectives (success, thermal failure, ROS failure, overheating, energy), each min-max normalised across controllers before weighting. This replaces the single fixed-weight utility used in the earlier analysis, which is not reproducible as a preference statement.*
