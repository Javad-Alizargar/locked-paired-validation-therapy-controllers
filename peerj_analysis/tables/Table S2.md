**Table S2. Implemented final-time endpoint definitions.**

| Criterion | Rule | Meaning |
| --- | --- | --- |
| warm | warm_fraction ≥ 0.50 | final-time unweighted radial-node fraction with 43 ≤ T ≤ 50 °C |
| ros | ros_threshold_fraction ≥ 0.50 | final-time radial-node fraction with accumulated ROS dose ≥ 0.67 mM |
| hypoxia_ok | hypoxic_failure_fraction ≤ 0.35 | final-time radial-node fraction with O2 < 5 µM AND ROS below threshold |
| safe_heat | overheat_fraction ≤ 0.05 and max_T ≤ 55 °C | final-time temperature criterion; not treatment-wide safety |
| success | warm AND ros AND hypoxia_ok AND safe_heat | all four criteria simultaneously |
| thermal_failure | NOT warm AND safe_heat | not warm while satisfying the final-time temperature criterion |
| ros_failure | NOT ros | ROS dose below threshold |
| overheat_failure | NOT safe_heat | failure of the final-time temperature criterion |
| hypoxic_failure | NOT hypoxia_ok | failure of the final-time hypoxia criterion |

*All fractions are unweighted radial-node fractions at the final time of a 3,000 s run. They are neither treatment-time nor spherical-volume fractions. max_T is the final spatial maximum. The stored safe_heat flag does not exclude earlier excursions or establish absence of injury. Thermal failure excludes cases failing safe_heat; marginal decreases do not imply better heating reliability.*
