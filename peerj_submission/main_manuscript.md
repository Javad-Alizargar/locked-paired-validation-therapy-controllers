# Locked paired validation of closed-loop therapy software: controller rankings are not robust to analysis choices

## Abstract

**Background.** Closed-loop therapy software selects treatment inputs from delayed and noisy measurements. Reported gains are usually taken from a single nominal scenario, summarised as one aggregate success rate, and left uncorrected for the number of comparisons examined. We asked whether four delayed-feedback controllers for coupled photothermal-photodynamic simulation remain superior under a locked, paired, multiplicity-controlled protocol.

**Methods.** We built a locked validation atlas of 4,608 paired trials per controller: three validation seeds × four model families × 32 uncertainty draws × 12 delay/noise scenarios, forming 384 clusters of 12 trials. A logistic scoring rule, fitted on independent seeds and applied unchanged, assigned every trial to a predicted-baseline-success band from pre-treatment quantities only. Each controller was compared with delayed full feedback using 95% cluster-bootstrap intervals (10,000 replicates), cluster-level sign-flip permutation tests (20,000 permutations), Holm correction over the 12 controller × band tests, and minimum detectable effects at 80% power. We added a 16-specification curve per controller, Pareto dominance analysis, and a Dirichlet weight-sensitivity analysis.

**Results.** No controller improved overall success. The paired change was -0.018 (95% CI -0.041 to 0.005; p = 0.121) for smoothed predictive, -0.049 (-0.081 to -0.016; p = 0.004) for thermal-floor predictive, -0.007 (-0.037 to 0.024; p = 0.659) for constrained optimized, and -0.432 (-0.472 to -0.391; p < 0.001) for reduced-order MPC. In the predicted-low band the three plausible controllers had positive point estimates (+0.031 to +0.056), but none survived correction (Holm-adjusted p = 0.629 for all three). In the predicted-high band three controllers significantly reduced success. All four controllers were Pareto non-dominated, and under 200,000 random weightings the most frequent rank-one controller reached only 38.5%. The sign of the headline effect changed across defensible analysis choices: 37.5% of specifications were positive for constrained optimized, 25% for smoothed predictive, 25% for thermal-floor predictive, and 0% for reduced-order MPC.

**Conclusions.** Locked paired validation with multiplicity control removed every apparent advantage in this atlas. Evaluation of therapy software should report cluster-robust intervals, corrected p-values and multi-endpoint trade-offs, because a single success rate can reverse the ranking.

## Introduction

Closed-loop control is now a routine design pattern in thermal therapy software. Model predictive control has been applied to magnetic nanoparticle hyperthermia (Sebeke et al., 2019; Lad et al., 2025; Abu-Ayyad et al., 2024), temperature-feedback controllers have been validated on clinical hyperthermia hardware (Sharma et al., 2023), and temperature-based feedback has been studied for electromagnetic phased-array heating since the early 2000s (Kowalski & Jin, 2003). In photothermal and photodynamic therapy the control problem is harder still, because heating and reactive oxygen species (ROS) generation share the same optical input and compete for the same oxygen supply (Bucharskaya et al., 2022; Khan et al., 2025; Maleki et al., 2021). A controller that raises temperature also raises oxidative dose and consumes oxygen, so a decision that improves one endpoint can degrade another.

The evaluation practice around these controllers has not kept pace with the algorithms. A new controller is typically compared with a baseline in a nominal simulation, or across a modest stress grid, and judged on a single success rate. That design answers a narrow question: does the controller help in the specific scenarios that were run? It does not answer the question a reader actually cares about, which is whether the controller is better on average, for a stated population of uncertainty, at a stated level of statistical confidence, without having selected the favourable comparison. Three failure modes are well documented in adjacent fields. First, seed and run-to-run variance can be large enough to reorder methods that are reported as clearly separated (Henderson et al., 2018). Second, point estimates without interval estimates hide that ordering uncertainty, which is why stratified bootstrap intervals have been recommended for reinforcement-learning benchmarks (Agarwal et al., 2021). Third, the space of defensible analysis choices is wide, and unrecorded choices inflate apparent effects (Simmons et al., 2011; Steegen et al., 2016; Simonsohn et al., 2020). Reproducibility initiatives in machine learning have made the same point from the infrastructure side (Pineau et al., 2021), and the wider computational-science literature has documented both the scale of the problem (Baker, 2016) and practical remedies for it (Sandve et al., 2013; Wilkinson et al., 2016).

Biomedical simulation has a mature vocabulary for exactly this problem. Verification, validation and uncertainty quantification (VVUQ) frameworks require an explicit context of use, a documented evidence base, and a statement of the conditions under which a model is credible (Mulugeta et al., 2018; Erdemir et al., 2020; Pathmanathan et al., 2017; Santiago et al., 2022; Steinman & Migliavacca, 2018). The same discipline has been extended to digital twins for precision medicine (Sel et al., 2025), to multidomain photoacoustic modelling (Akhlaghi et al., 2019), and to mechanism-based oncology models intended for clinical use (Wu et al., 2022; Collin et al., 2022; Rasheed et al., 2020). Scientific machine learning supplies the surrogate and discovery tools that make such models tractable (Karniadakis et al., 2021; Rudy et al., 2017; Cai et al., 2021). What VVUQ practice has not generally supplied is a statistical protocol for *ranking controllers* against each other. VVUQ asks whether a model is credible; controller evaluation asks whether one decision rule beats another across a population of patients, geometries and sensing conditions, and that is an inferential question.

The inferential toolkit is standard but rarely applied in this setting. Paired designs remove between-trial heterogeneity, cluster bootstrap intervals respect the dependence induced by shared random draws, sign-flip permutation tests give exact randomisation p-values without distributional assumptions, and Holm adjustment controls the family-wise error rate when several controllers and several strata are examined (Efron & Tibshirani, 1993; DiCiccio & Efron, 1996; Winkler et al., 2014; Holm, 1979). Where a false-discovery interpretation is preferred, Benjamini-Hochberg control is the alternative (Benjamini & Hochberg, 1995). Minimum detectable effects, computed from the observed cluster-robust standard error, make the precision of a null result explicit rather than leaving it implicit (Cohen, 1988). Multi-objective optimisation supplies the vocabulary for the trade-off structure, and Pareto dominance is the natural summary when several endpoints matter and no defensible scalarisation exists (Verma et al., 2021).

We therefore asked a deliberately adversarial question. Given a locked validation atlas, a frozen scoring rule, paired trials, cluster-robust inference, multiplicity control, a specification curve, and a Pareto analysis, does any of four delayed-feedback controllers for coupled photothermal-photodynamic simulation remain superior to delayed full feedback? The protocol was designed so that a negative answer would be informative: the question was whether any controller would show a positive overall effect with a confidence interval excluding zero. None did. The three-stage protocol is summarised in Fig. 1.

The contribution is fourfold. First, we specify a locked paired-validation protocol for closed-loop therapy software, in which endpoint thresholds, the stratification rule and the estimator are fixed in version-controlled code and applied identically to every controller and every validation seed. Second, we provide a cluster-robust inference pipeline that reports effect sizes, 95% intervals, exact permutation p-values, Holm-adjusted p-values and minimum detectable effects for every controller × endpoint × stratum cell. Third, we add a specification curve that quantifies how far the headline conclusion moves across defensible analysis choices, and a Pareto and weight-sensitivity analysis that replaces an arbitrary scalar utility with an explicit statement about preference dependence. Fourth, we apply the protocol to a 4,608-trial-per-controller atlas and report the result, which is that no controller dominates and that the apparent advantage in the predicted-low band does not survive correction.

## Materials & Methods

### Simulation model

The case study is a one-dimensional radial model of a vascularised tumour spheroid, implemented in Python and used throughout as a fixed computational testbed rather than as a clinical predictor. The spheroid has a radius of 500 µm discretised on 96 radial nodes. Temperature is advanced with the Pennes bioheat equation using backward Euler time integration and a tridiagonal solve, with tissue density 1060 kg m⁻³, specific heat 3600 J kg⁻¹ K⁻¹, thermal conductivity 0.50 W m⁻¹ K⁻¹, and perfusion represented as a volumetric heat sink proportional to the local blood-flow parameter. The photothermal source is the product of a gold-nanorod absorption cross-section computed with Gans theory, a radial nanoparticle distribution with a 0.30 surface bias, and a radial optical fluence profile; nanorod length spans 40-100 nm and width 10-30 nm, and the medium refractive index is 1.33. A magnetic loss term is included so that the same testbed can represent magneto-plasmonic heating.

ROS generation is integrated on a 0.5 s substep inside each 10 s control interval. Singlet-oxygen dose accumulates only while the light is on, with a generation rate proportional to fluence, local photosensitiser concentration and an oxygen-saturation factor; photosensitiser is bleached at a rate that weakens at low concentration, and oxygen is replenished towards an initial 40 µM with a radial gradient. The singlet-oxygen threshold is 0.67 mM. The protocol window is 300 s of light-on time; duty cycle extends the wall-clock duration without changing the light-on dose.

Four derived quantities are recorded for every trial and are used later for stratification: the dose-capacity number, the reaction number multiplied by observation quality, the reaction-drive balance, and the delay-to-diffusion ratio. All four are computed from design and sensing parameters only and are therefore available before any treatment outcome is observed. Full equations, parameter values and evidence sources are given in the repository and in the model configuration files.

### Controller families

Five decision rules were compared. All share the same maximum intensity of 1.30 W cm⁻² and the same 10 s control interval.

*Delayed full feedback* is the baseline. It applies the most recent measured state directly, so its decisions are delayed by the sensing latency and corrupted by measurement noise but are otherwise ungated.

*Smoothed predictive* estimates the current state with a six-sample sliding window, applies a prediction gain of 0.85, and enforces a 0.35 °C thermal margin. It adds delay-dependent safety rules: at 60 s or more of delay it stops above 51.5 °C unless ROS coverage is already at 0.80, and at 120 s or more it stops above 51.0 °C unless ROS coverage is already at 0.75.

*Thermal-floor predictive* uses the same estimator and adds a thermal floor. If the warm fraction is below 0.48 and the maximum temperature is below 48.6 °C, it applies a floor intensity of 0.58 times the maximum, scaled down when oxygen is limiting. Once ROS coverage reaches 0.82 and the warm fraction target is met, it drops to a 0.22 W cm⁻² maintenance intensity or stops.

*Constrained optimized* is a re-parameterisation of the thermal-floor architecture, selected by a chance-constrained random search over 20 candidates on two independent training seeds (20260730 and 20260731) with eight draws each, then validated without further tuning on the three validation seeds. Its selected parameters are a maximum intensity of 1.128 W cm⁻², prediction gain 0.669, thermal margin 0.550 °C, thermal-floor warm fraction 0.478, thermal-floor temperature 48.52 °C, floor fraction 0.437, maintenance intensity 0.212 W cm⁻², long-delay safety temperature 50.64 °C, and mid-delay safety temperature 51.09 °C.

*Reduced-order MPC* is a receding-horizon benchmark. At each control step it propagates a four-state reduced model (temperature, warm fraction, ROS coverage, mean oxygen) four steps ahead over an eight-point intensity grid, discounts by 0.88 per step, and minimises a stage cost with weights 0.045 on energy, 4.8 on overheating, 0.9 on oxygen depletion, 1.8 on thermal deviation and 1.2 on ROS shortfall. It hard-stops above 52.2 °C unless ROS coverage is already at 0.78, and stops when oxygen falls below 8.0 µM with ROS coverage below 0.82.

### Locked validation atlas

The atlas crosses three validation seeds (20260720, 20260721, 20260722), four model families (baseline, large geometry, low photosensitiser, slow recovery), 32 uncertainty draws per seed and family sampled by Latin hypercube, and 12 sensing scenarios formed by the product of six delays (0, 30, 60, 90, 120 and 180 s) and two noise levels (0% and 10%). Each controller therefore contributes 3 × 4 × 32 × 12 = 4,608 trials, and each baseline-controller comparison is paired on seed, family, draw, delay and noise. The 384 seed × family × draw combinations define the clusters used for inference, with exactly 12 paired trials per cluster.

The protocol is locked in the following operational sense. Endpoint thresholds, the stratification rule and the inference estimator are specified in version-controlled code and applied identically to all four controllers and all three validation seeds. No controller parameter was re-tuned after the validation outputs were produced, and the validation seeds were not used to fit any controller or any part of the scoring rule. The chance-constrained search for the constrained-optimized policy used two different seeds. We describe the protocol as locked rather than pre-registered because the analysis code and the validation seeds coexisted in the same repository; what can be verified from the artefact is that the rule is frozen, applied without modification, and reproducible bit-for-bit.

### Frozen scoring rule and predicted-baseline-success bands

Trials were stratified by a logistic scoring rule for the probability of success under delayed full feedback. The rule uses three pre-treatment quantities: log10 of the dose-capacity number, log10 of the reaction number multiplied by observation quality, and log10 of the reaction-drive balance. It is

logit(P) = -0.016 + 1.406 z1 + 0.930 z2 - 1.482 z3,

where z1, z2 and z3 are standardised with means -0.625, 0.319 and -3.530 and standard deviations 0.549, 0.307 and 0.576, all estimated on independent seeds. The coefficients and standardisation constants were fixed before the validation atlas was analysed. Band boundaries of 0.364 and 0.595, the 33rd and 67th percentiles of the reference population, were also fixed in advance. Applying the rule to the validation atlas yields 1,122 predicted-low, 1,674 predicted-intermediate and 1,812 predicted-high trials per controller. The bands are therefore not equal thirds of the validation population; they are the reference-population thirds projected onto a population with a slightly different distribution.

We use the term *predicted-baseline-success band* deliberately. The score predicts how likely the baseline controller is to succeed, not how controllable a case is. Discrimination is modest: the area under the receiver-operating characteristic curve is 0.688 overall in the validation atlas, ranges from 0.590 to 0.727 across seed × family cells, and is 0.610 with a Brier score of 0.265 on a fully held-out seed that was used neither for fitting nor for tuning (Table S1). A score with this discrimination supports a stratified description but not a claim that the bands identify a latent controllability construct, and we make only the weaker claim.

### Endpoint definitions

Six endpoints were defined before the validation outputs were analysed (Table S2). A trial is *warm* when the fraction of the protocol window with temperature between 43 °C and 50 °C is at least 0.50. *ROS coverage* is met when the fraction of the window at or above the singlet-oxygen threshold is at least 0.50. *Hypoxic control* is met when the fraction of the window with oxygen below 5 µM and ROS below threshold is at most 0.35. *Safe heating* is met when the fraction of the window above 55 °C is at most 0.05 and the maximum temperature is at most 55 °C. *Success* requires all four. Three failure modes are defined as complements of the components: thermal failure is failure to warm while heating safely, ROS failure is failure to reach ROS coverage, and overheating is failure of safe heating. A fourth, hypoxic failure, is the complement of hypoxic control. Delivered energy in J cm⁻² and maximum temperature in °C are recorded as continuous outcomes.

### Statistical analysis

All inference is paired at the trial level and clustered at the seed × family × draw level. For a binary paired outcome the estimand is the ratio of summed within-cluster differences to the total number of trials, which equals the difference in success proportions. Cluster bootstrap intervals use 10,000 resamples of the 384 clusters, drawn once per controller × stratum cell and shared across endpoints so that intervals for correlated endpoints remain coherent. The primary test is a cluster-level sign-flip randomisation test with 20,000 sign vectors applied to the within-cluster sums; the test statistic is the ratio of summed signed cluster totals to the total trial count, and the p-value is the proportion of permutations whose absolute statistic is at least the observed absolute statistic, with the observed value included in the reference set. This test is exact under the null of exchangeable signs within a pair and requires no distributional assumption. Because the estimand is a ratio of sums rather than a mean of cluster means, the test remains valid when clusters are unbalanced, which they are once the data are stratified.

Multiplicity is controlled with Holm's sequentially rejective procedure over a locked family of 12 tests: four controllers × three bands, for the success endpoint (Holm, 1979). A second family of 20 tests covers five mechanism and burden endpoints (thermal failure, ROS failure, hypoxic failure, overheating, delivered energy) across four controllers at the all-trial level. p-values are reported to three decimal places, except that values below 0.001 are reported as such; the minimum attainable permutation p-value with 20,000 permutations is 0.00005.

Minimum detectable effects at 80% power are computed as 2.802 times the observed cluster-robust standard error, where 2.802 is the sum of the 0.975 standard-normal quantile and the 0.80 standard-normal quantile (Cohen, 1988). Design effects are reported as the ratio of the cluster-robust variance of the paired difference to the simple random-sampling variance, to make the cost of clustering explicit.

### Specification curve

To quantify dependence on analysis choices we recomputed the headline contrast under 16 specifications per controller: four seed subsets (all three seeds, or each seed alone) crossed with four scenario subsets (all scenarios, nominal only at 0 s delay and 0% noise, delayed only at 120 s or more, and noisy only at 10%). Each specification was evaluated with the paired cluster-bootstrap estimator. A naive unpaired difference of proportions with a Wald interval was computed in parallel as a contrast, to show the effect of ignoring pairing. The specification curve reports the fraction of specifications with a positive point estimate, together with the minimum and maximum estimates.

### Pareto dominance and weight sensitivity

Controller performance was summarised over five objectives: success change (maximised), thermal-failure change, ROS-failure change, overheating change and delivered-energy change (all minimised). Controller A dominates controller B if A is at least as good on every objective and strictly better on at least one. A controller is Pareto optimal if no other controller dominates it.

Because Pareto analysis deliberately declines to rank non-dominated points, we also quantified how much a ranking depends on weights. Each objective was min-max normalised across the four controllers and oriented so that larger is better. We then drew 200,000 weight vectors from a uniform Dirichlet distribution over the five objectives, computed the weighted score for each controller, and recorded the rank distribution. This replaces a fixed-weight scalar utility with an explicit statement about how often each controller would be preferred across the space of defensible preferences.

### Software, data and reproducibility

The analysis is implemented in Python 3.12 with NumPy, pandas, SciPy and Matplotlib. A single entry point regenerates every table and figure from the committed trial-level data, and the random seed for resampling is fixed. The trial-level paired dataset, the controller definitions, the scoring rule, the endpoint definitions and the analysis code are archived with a persistent DOI. Figure images are supplied as separate files at 300 dpi and between 900 and 3,000 pixels in each dimension.

### Ethics

This study uses no human participants, human tissue or animal subjects. It is a computational analysis of simulated data, so ethical approval and informed consent were not required.

### Use of generative AI

OpenAI ChatGPT (GPT-5) was used in October 2026 to draft and edit the manuscript text and to help structure the analysis code. It was used to accelerate language editing and code organisation, not to generate, analyse or interpret data. No numerical result in this article originates from the tool: every statistic was produced by executing the analysis code on the committed trial-level data and was checked by the author against the program output. The author confirms the originality and accuracy of the content, has reviewed the tool's terms of use and confirms that this use is consistent with them, and takes full responsibility for the integrity of the article, including the accuracy of all references. Generative AI was not used to create, alter or manipulate any figure, image or research datum; all figures were generated programmatically from the analysis outputs by the committed Python code.

## Results

### No controller improves overall success

Across all 4,608 paired trials per controller, the baseline success rate was 0.560. Three of the four controllers had negative point estimates and one was indistinguishable from zero (Table 1, Fig. 2). Smoothed predictive changed success by -0.018 (95% CI -0.041 to 0.005; permutation p = 0.121; minimum detectable effect 0.033). Thermal-floor predictive changed success by -0.049 (-0.081 to -0.016; p = 0.004; MDE 0.046). Constrained optimized changed success by -0.007 (-0.037 to 0.024; p = 0.659; MDE 0.044). Reduced-order MPC changed success by -0.432 (-0.472 to -0.391; p < 0.001; MDE 0.058). The confidence intervals for thermal-floor predictive and reduced-order MPC exclude zero, and both are negative. No controller produced a positive overall effect whose interval excluded zero.

The minimum detectable effects are informative about precision rather than about effect size. With 384 clusters and 4,608 paired trials, the design could detect a success difference of about 3.3 to 5.8 percentage points at 80% power. The observed effects for three controllers are at or below that threshold, so the correct reading of their intervals is that a small benefit cannot be excluded, not that a benefit was demonstrated. The design effect for the paired success difference ranged from 2.56 for smoothed predictive to 5.54 for reduced-order MPC, confirming that clustering is substantial and that trial-level inference would have been badly anticonservative.

### Failure modes move in opposite directions

The four controllers do not fail in the same way, and the pattern explains why a single success rate is an inadequate summary (Table 2, Fig. 3). Smoothed predictive was the only controller that reduced overheating, by 0.012 (-0.024 to -0.000; p = 0.050; Holm-adjusted p = 0.050), but it increased thermal failure by 0.038 (0.016 to 0.060; p < 0.001; Holm-adjusted p = 0.003) while reducing ROS failure by 0.023 (-0.036 to -0.013; Holm-adjusted p < 0.001) and saving 23.5 J cm⁻² (-29.5 to -17.5; Holm-adjusted p < 0.001). Constrained optimized increased overheating by 0.020 (0.008 to 0.031; Holm-adjusted p = 0.002) and saved 59.1 J cm⁻² (-63.4 to -54.8; Holm-adjusted p < 0.001), with no detectable change in thermal failure (-0.010, -0.042 to 0.021) or ROS failure (-0.002, -0.012 to 0.009). Thermal-floor predictive reduced thermal failure by 0.037 (-0.067 to -0.006; Holm-adjusted p = 0.042) and ROS failure by 0.017 (-0.028 to -0.007; Holm-adjusted p = 0.004), but increased overheating by 0.095 (0.081 to 0.110; Holm-adjusted p < 0.001) and spent 44.2 J cm⁻² more (36.3 to 52.5; Holm-adjusted p < 0.001). Reduced-order MPC reduced thermal failure by 0.048 (-0.081 to -0.018; Holm-adjusted p = 0.008) and ROS failure by 0.054 (-0.072 to -0.037; Holm-adjusted p < 0.001), but increased overheating by 0.497 (0.471 to 0.523; Holm-adjusted p < 0.001), spent 549.7 J cm⁻² more (521.6 to 578.3; Holm-adjusted p < 0.001), and raised maximum temperature by 15.7 °C (14.8 to 16.6). Hypoxic failure did not change for any controller; all four deltas were exactly 0.000 with degenerate intervals, so hypoxia is not the axis along which these controllers differ.

The pattern is a genuine trade-off rather than a set of independent wins. Smoothed predictive buys safety and energy savings with underdosing. Constrained optimized buys energy savings with overheating. Thermal-floor predictive buys thermal and ROS reliability with overheating and energy cost. Reduced-order MPC buys thermal and ROS reliability with a large overheating and energy penalty. No controller improves the efficacy endpoint and the safety endpoint simultaneously.

### Apparent gains in the predicted-low band do not survive correction

In the predicted-low band (n = 1,122 pairs, 181 clusters), where the baseline success rate was 0.383, the three plausible controllers all had positive point estimates (Table 3). Smoothed predictive changed success by +0.031 (95% CI -0.012 to 0.075; p = 0.181). Thermal-floor predictive changed success by +0.056 (-0.011 to 0.122; p = 0.126). Constrained optimized changed success by +0.050 (-0.012 to 0.112; p = 0.132). Reduced-order MPC changed success by -0.210 (-0.283 to -0.140; p < 0.001). After Holm adjustment across the 12 controller × band tests, the adjusted p-value was 0.629 for each of the three positive estimates, and below 0.001 for reduced-order MPC. Every confidence interval for the three positive estimates includes zero.

This is the result that most changes the interpretation of the atlas. Taken one controller at a time, and without correction, a +0.056 change in a stratum with a 0.383 baseline is a relative improvement of about 15% and would read as a promising signal. Taken as one of twelve comparisons, with an interval that spans zero and a minimum detectable effect of 0.095 for that cell, it is not evidence of benefit. The point estimates are consistent with a real but small effect that this design cannot resolve, and they are equally consistent with no effect. The honest statement is that the predicted-low band shows a non-significant positive signal that requires a larger atlas or a lower-variance endpoint to settle.

### Predicted-high cases are systematically harmed

In the predicted-high band (n = 1,812 pairs, 238 clusters), where the baseline success rate was 0.746, every controller had a negative point estimate and three remained significant after correction. Smoothed predictive changed success by -0.063 (-0.096 to -0.028; p < 0.001; Holm-adjusted p = 0.002). Thermal-floor predictive changed success by -0.119 (-0.164 to -0.073; p < 0.001; Holm-adjusted p < 0.001). Constrained optimized changed success by -0.047 (-0.090 to -0.003; p = 0.035), but this did not survive correction (Holm-adjusted p = 0.245). Reduced-order MPC changed success by -0.645 (-0.691 to -0.596; p < 0.001; Holm-adjusted p < 0.001). In the predicted-intermediate band only reduced-order MPC changed success significantly (-0.351, -0.409 to -0.293; Holm-adjusted p < 0.001); the other three estimates were within 0.043 of zero and none survived correction.

The asymmetry is the substantive finding. The controllers were designed to rescue cases that the baseline handles poorly, and in the band where the baseline already succeeds most of the time they interfere with a working protocol. This is consistent with a mechanism in which delay-dependent safety gates and floors, calibrated to protect marginal cases, trigger unnecessarily when the baseline is already on target. It also means that any claim of benefit must be conditional on a band, and that a marginal-band benefit would have to be weighed against a high-band cost in any deployment decision.

### No controller is Pareto-dominant and no weighting gives a robust winner

Over the five objectives, no controller dominated any other and none was dominated, so all four are Pareto optimal (Fig. 4). The dominance matrix is empty. This is not a technicality: it follows directly from the mechanism pattern, because each controller is best on at least one objective and worst on at least one other. Smoothed predictive is best on overheating and competitive on energy. Constrained optimized is best on energy. Thermal-floor predictive is best on thermal failure. Reduced-order MPC is best on ROS failure.

Rank stability under random weights shows how little the Pareto structure constrains a recommendation (Table S3). Across 200,000 Dirichlet weight draws, thermal-floor predictive ranked first in 38.5% of draws, smoothed predictive in 27.9%, constrained optimized in 22.8%, and reduced-order MPC in 10.7%. Thermal-floor predictive was in the top two in 69.5% of draws. Reduced-order MPC was ranked last in 72.3% of draws. No controller is preferred under a majority of defensible weightings, and the most frequent winner changes with the weighting. A paper that reported only the argmax of one fixed utility would therefore have reported a preference, not a finding.

### The sign of the effect depends on analysis choices

The specification curve makes the dependence explicit (Fig. 5). Across the 16 specifications per controller, the point estimate was positive in 37.5% of specifications for constrained optimized (range -0.104 to +0.125), 25% for smoothed predictive (-0.100 to +0.020), 25% for thermal-floor predictive (-0.174 to +0.117), and 0% for reduced-order MPC (-0.617 to -0.281). The primary specification, all seeds and all scenarios with the paired estimator, is a single point in each panel.

The pattern shows which choices matter. Restricting to the nominal scenario moves estimates upward for the three plausible controllers, because the nominal scenario removes the delay and noise conditions in which the safety gates misfire. Restricting to a single seed moves estimates in either direction by amounts comparable to the effects themselves, which is the seed-variance problem described for reinforcement-learning benchmarks. The naive unpaired estimator produces wider intervals and, for some specifications, a different significance verdict, because it discards the pairing that removes between-draw heterogeneity. None of these choices is indefensible in isolation. That is precisely the problem: a result that changes sign across defensible choices should not be reported as a sign.

### Discrimination of the scoring rule is modest

The bands are only as good as the score that defines them. Discrimination was 0.688 overall in the validation atlas, ranged from 0.590 to 0.727 across the 12 seed × family cells, and fell to 0.610 on a fully held-out seed with a Brier score of 0.265 (Table S1). Calibration was reasonable in the aggregate but coarse at the band level: observed baseline success was 0.383 in the predicted-low band, 0.477 in the predicted-intermediate band and 0.746 in the predicted-high band, against mean predicted probabilities of 0.398, 0.475 and 0.763 respectively. The ordering is correct and the spread is large, which is why stratification is informative. The overlap between adjacent bands is substantial, which is why a band label should not be read as a case-level prediction.

## Discussion

### Principal findings

Four delayed-feedback controllers for coupled photothermal-photodynamic simulation were compared with delayed full feedback across 4,608 paired trials each, with cluster-robust intervals, exact permutation tests, Holm correction, minimum detectable effects, a specification curve and a Pareto analysis. No controller improved overall success. Three controllers significantly reduced success overall or in the predicted-high band. In the predicted-low band, where a benefit was most plausible, all three plausible controllers had positive point estimates of 0.031 to 0.056, and none survived multiplicity correction; the Holm-adjusted p-value was 0.629 in every case. All four controllers were Pareto non-dominated, no weighting produced a majority winner, and the sign of the headline effect changed across defensible analysis choices for three of the four controllers.

### Why locked evaluation changes the conclusion

The difference between this result and a conventional report is not the data. It is the inferential frame. A conventional analysis would have compared controllers in the nominal scenario, or pooled scenarios without clustering, and would have reported the predicted-low uplift as a positive finding. Each of those steps is individually defensible and collectively misleading. Removing delay and noise removes the conditions the safety gates were built for. Pooling without clustering treats 12 trials from the same draw as independent, which the design effect of 2.6 to 5.5 shows they are not. Reporting one controller × one stratum without correction ignores that twelve such cells were examined. The specification curve quantifies the cumulative effect of these choices: for constrained optimized the estimate ranges from -0.104 to +0.125 across 16 defensible specifications, which is wider than any effect the study could reliably detect.

This is the same phenomenon that the multiverse and specification-curve literature documents in the social sciences (Simmons et al., 2011; Steegen et al., 2016; Simonsohn et al., 2020), and that benchmark studies document in machine learning (Henderson et al., 2018; Agarwal et al., 2021; Pineau et al., 2021). What is new here is the domain. Controller evaluation for therapy software has a VVUQ tradition that is strong on model credibility and weak on comparative inference, and the gap is consequential because controller rankings are used to justify design choices.

### Relation to previous work

The controllers studied here belong to a well-developed line of thermal-therapy control work. Model predictive control for magnetic nanoparticle hyperthermia has been demonstrated in simulation and in phantom (Sebeke et al., 2019; Lad et al., 2025; Abu-Ayyad et al., 2024), temperature-feedback control has been validated on clinical hardware (Sharma et al., 2023), and earlier work established the feasibility of temperature-based feedback for phased-array heating (Kowalski & Jin, 2003). Our reduced-order MPC is a deliberately simplified receding-horizon benchmark rather than a full MPC implementation, and its poor performance here should be read as a statement about a four-step reduced model under 0 to 180 s of delay, not about MPC in general.

The VVUQ literature supplies the framing for the credibility question (Mulugeta et al., 2018; Erdemir et al., 2020; Pathmanathan et al., 2017; Santiago et al., 2022; Steinman & Migliavacca, 2018; Sel et al., 2025) and for the specific case of imaging and oncology models (Akhlaghi et al., 2019; Wu et al., 2022; Collin et al., 2022). Our contribution is complementary: we assume the model is credible enough to serve as a testbed and ask a comparative question about decision rules within it. The two questions are different, and a credible model can still yield an unsupported controller ranking if the comparison is not designed.

The statistical components are standard. Cluster bootstrap intervals follow Efron and Tibshirani (Efron & Tibshirani, 1993; DiCiccio & Efron, 1996), sign-flip permutation inference for clustered data follows Winkler et al. (2014), and Holm control follows Holm (1979). The multi-objective framing follows the Pareto tradition in multi-objective optimisation (Verma et al., 2021). None of these methods is novel in isolation; the contribution is their combination into a protocol that is cheap to run, hard to game, and informative when the answer is negative.

### Implications for practice

Three practices follow directly. First, report the paired difference with a cluster-robust interval, not two marginal success rates. The marginal rates in this atlas differ by less than the clustering-induced uncertainty in several cells. Second, correct for the number of controller × endpoint × stratum cells examined, and state the family in advance. The predicted-low signal here is real as a point estimate and absent as evidence after correction. Third, report the multi-endpoint trade-off and decline to scalarise it. The Pareto analysis shows that every controller is best somewhere; a single utility would have hidden that.

A fourth implication concerns the reporting of negative results. The most useful output of this study is not a winner but a bound: the design could detect a 3.3 to 5.8 percentage point change at 80% power, and the observed predicted-low effects sit below that bound. Reporting the bound converts an ambiguous null into an actionable statement about what a follow-up atlas would need to resolve.

### Limitations

The model is a screening-level one-dimensional radial representation, not a patient-specific or three-dimensional model. It omits angular heterogeneity, discrete vasculature, and the full optical transport that a production simulator would include. The conclusions are therefore about controller behaviour in this testbed, and transferring them to a clinical setting would require re-running the protocol on a higher-fidelity model. The scoring rule has only modest discrimination (AUC 0.610 on a fully held-out seed), so the bands are coarse; a better score would sharpen the stratified estimates but would not by itself change the overall null. The atlas uses three validation seeds, chosen to keep the compute budget tractable; seed-level variance is visible in the specification curve and is one reason the intervals are wide. The constrained-optimized controller was selected from 20 candidates on two training seeds, which is a small search; a larger search could plausibly find a better parameterisation, though the locked validation would still be the test. Finally, the protocol is locked in code rather than prospectively registered, and while the artefact supports reproducibility, it cannot prove that no analyst saw a validation output before a threshold was set.

### Future directions

Three extensions would strengthen the design. A ten-seed atlas would roughly halve the minimum detectable effect and would settle whether the predicted-low signal is real; the compute cost is four to six hours of serial simulation per controller, which is feasible. A pre-registered version, with the analysis plan deposited before the validation seeds are run, would remove the residual ambiguity about locking. A hierarchical model that borrows strength across bands, rather than testing each band separately, would improve precision without inflating the family of tests. The protocol itself is model-agnostic and should transfer to other closed-loop therapy software, including radiation dose adaptation and drug-infusion control, wherever paired simulation and multi-endpoint outcomes are available.

## Conclusions

A locked, paired, multiplicity-controlled evaluation of four delayed-feedback controllers for coupled photothermal-photodynamic simulation found no controller that improves overall success. The apparent benefit in the predicted-low band, +0.031 to +0.056, did not survive correction, and three controllers significantly reduced success in the predicted-high band. All four controllers were Pareto non-dominated, no weighting produced a majority winner, and the sign of the headline effect changed across defensible analysis choices. The protocol, not the ranking, is the transferable contribution: cluster-robust intervals, exact permutation tests, declared multiplicity families, minimum detectable effects, a specification curve and an explicit Pareto statement together turn an ambiguous comparison into a bounded and reproducible one. Controller evaluations for therapy software should adopt these reporting requirements, because a single success rate can reverse the ranking.

## Acknowledgments

The author thanks the open-source maintainers of NumPy, pandas, SciPy and Matplotlib, on which the analysis depends.

## References

Abu-Ayyad M, Lad YS, Aguilar D, Karami K, Attaluri A. 2024. Model predictive control (MPC) applied to a simplified model, magnetic nanoparticle hyperthermia (MNPH) treatment process. *Biomedical Physics & Engineering Express* 10(4):045012. DOI: 10.1088/2057-1976/ad460a.

Agarwal R, Schwarzer M, Castro PS, Courville A, Bellemare MG. 2021. Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems* 34. Available at https://arxiv.org/abs/2108.13264

Akhlaghi N, Pfefer TJ, Wear KA, Garra BS, Vogt WC. 2019. Multidomain computational modeling of photoacoustic imaging: verification, validation, and image quality prediction. *Journal of Biomedical Optics* 24(12):1-12. DOI: 10.1117/1.JBO.24.12.121910.

Baker M. 2016. 1,500 scientists lift the lid on reproducibility. *Nature* 533(7604):452-454. DOI: 10.1038/533452a.

Benjamini Y, Hochberg Y. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society Series B: Statistical Methodology* 57(1):289-300. DOI: 10.1111/j.2517-6161.1995.tb02031.x.

Bucharskaya AB, Khlebtsov NG, Khlebtsov BN, Maslyakova GN, Navolokin NA, Genin VD, Genina EA, Tuchin VV. 2022. Photothermal and photodynamic therapy of tumors with plasmonic nanoparticles: challenges and prospects. *Materials* 15(4):1606. DOI: 10.3390/ma15041606.

Cai S, Mao Z, Wang Z, Yin M, Karniadakis GE. 2021. Physics-informed neural networks (PINNs) for fluid mechanics: a review. *Acta Mechanica Sinica* 37(12):1727-1738. DOI: 10.1007/s10409-021-01148-1.

Cohen J. 1988. *Statistical Power Analysis for the Behavioral Sciences*. 2nd edn. Hillsdale, NJ: Lawrence Erlbaum Associates.

Collin CB, Gebhardt T, Golebiewski M, Karaderi T, Hillemanns M, Khan FM, Salehzadeh-Yazdi A, Kirschner M, Krobitsch S, Kuepfer L. 2022. Computational models for clinical applications in personalized medicine: guidelines and recommendations for data integration and model validation. *Journal of Personalized Medicine* 12(2):166. DOI: 10.3390/jpm12020166.

DiCiccio TJ, Efron B. 1996. Bootstrap confidence intervals. *Statistical Science* 11(3):189-228. DOI: 10.1214/ss/1032280214.

Efron B, Tibshirani RJ. 1993. *An Introduction to the Bootstrap*. New York: Chapman & Hall/CRC. DOI: 10.1201/9780429246593.

Erdemir A, Mulugeta L, Ku JP, Drach A, Horner M, Morrison TM, Peng GCY, Vadigepalli R, Lytton WW, Myers JG. 2020. Credible practice of modeling and simulation in healthcare: ten rules from a multidisciplinary perspective. *Journal of Translational Medicine* 18(1):369. DOI: 10.1186/s12967-020-02540-4.

Henderson P, Islam R, Bachman P, Pineau J, Precup D, Meger D. 2018. Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence* 32(1). DOI: 10.1609/aaai.v32i1.11694.

Holm S. 1979. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6:65-70.

Karniadakis GE, Kevrekidis IG, Lu L, Perdikaris P, Wang S, Yang L. 2021. Physics-informed machine learning. *Nature Reviews Physics* 3(6):422-440. DOI: 10.1038/s42254-021-00314-5.

Khan S, Batool H, Tariq H, Noor A. 2025. Graphene oxide-based photothermal and photodynamic therapy: a systematic review. *Journal of Biomedical Materials Research Part B: Applied Biomaterials* 113(9):e35656. DOI: 10.1002/jbm.b.35656.

Kowalski ME, Jin JM. 2003. A temperature-based feedback control system for electromagnetic phased-array hyperthermia: theory and simulation. *Physics in Medicine and Biology* 48(5):633-651. DOI: 10.1088/0031-9155/48/5/306.

Lad YS, Pawar S, Arepally N, Carlton H, Hadjipanayis C, Ivkov R, Abu-Ayyad M, Attaluri A. 2025. Thermal dose feedback control systems applied to magnetic nanoparticle hyperthermia. *International Journal of Hyperthermia* 42(1):2491519. DOI: 10.1080/02656736.2025.2491519.

Maleki S, Farhadi M, Kamrava SK, Asghari A, Daneshi A. 2021. Simulation and in vitro experimental studies on targeted photothermal therapy of cancer using folate-PEG-gold nanorods. *Journal of Biomedical Physics and Engineering*. DOI: 10.31661/jbpe.v0i0.1108.

Mulugeta L, Drach A, Erdemir A, Hunt CA, Horner M, Ku JP, Myers JG, Vadigepalli R, Lytton WW. 2018. Credibility, replicability, and reproducibility in simulation for biomedicine and clinical applications in neuroscience. *Frontiers in Neuroinformatics* 12:18. DOI: 10.3389/fninf.2018.00018.

Pathmanathan P, Gray RA, Romero VJ, Morrison TM. 2017. Applicability analysis of validation evidence for biomedical computational models. *Journal of Verification, Validation and Uncertainty Quantification* 2(2):021005. DOI: 10.1115/1.4037671.

Pineau J, Vincent-Lamarre P, Sinha K, Lariviere V, Beygelzimer A, d'Alche-Buc F, Fox E, Larochelle H. 2021. Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program). *Journal of Machine Learning Research* 22(164):1-20.

Rasheed A, San O, Kvamsdal T. 2020. Digital twin: values, challenges and enablers from a modeling perspective. *IEEE Access* 8:21980-22012. DOI: 10.1109/access.2020.2970143.

Rudy SH, Brunton SL, Proctor JL, Kutz JN. 2017. Data-driven discovery of partial differential equations. *Science Advances* 3(4):e1602614. DOI: 10.1126/sciadv.1602614.

Sandve GK, Nekrutenko A, Taylor J, Hovig E. 2013. Ten simple rules for reproducible computational research. *PLoS Computational Biology* 9(10):e1003285. DOI: 10.1371/journal.pcbi.1003285.

Santiago A, Butakoff C, Eguzkitza B, Gray RA, May-Newman K, Pathmanathan P, Vu V, Vazquez M. 2022. Design and execution of a verification, validation, and uncertainty quantification plan for a numerical model of left ventricular flow after LVAD implantation. *PLoS Computational Biology* 18(6):e1010141. DOI: 10.1371/journal.pcbi.1010141.

Sebeke L, Deenen DA, Maljaars E, Heijman E, de Jager B, Heemels WPMH, Grull H. 2019. Model predictive control for MR-HIFU-mediated, uniform hyperthermia. *International Journal of Hyperthermia* 36(1):1039-1049. DOI: 10.1080/02656736.2019.1668065.

Sel K, Hawkins-Daarud A, Chaudhuri A, Osman D, Bahai A, Paydarfar D, Willcox K, Chung C, Jafari R. 2025. Survey and perspective on verification, validation, and uncertainty quantification of digital twins for precision medicine. *npj Digital Medicine* 8(1):40. DOI: 10.1038/s41746-025-01447-y.

Sharma A, Jangam A, Shen JLY, Ahmad A, Arepally N, Rodriguez B, Borrello J, Bouras A, Kleinberg L, Ding K, et al. 2023. Validation of a temperature-feedback controlled automated magnetic hyperthermia therapy device. *Cancers* 15(2):327. DOI: 10.3390/cancers15020327.

Simmons JP, Nelson LD, Simonsohn U. 2011. False-positive psychology: undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science* 22(11):1359-1366. DOI: 10.1177/0956797611417632.

Simonsohn U, Simmons JP, Nelson LD. 2020. Specification curve analysis. *Nature Human Behaviour* 4(11):1208-1214. DOI: 10.1038/s41562-020-0912-z.

Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W. 2016. Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science* 11(5):702-712. DOI: 10.1177/1745691616658637.

Steinman DA, Migliavacca F. 2018. Editorial: special issue on verification, validation, and uncertainty quantification of cardiovascular models: towards effective VVUQ for translating cardiovascular modelling to clinical utility. *Cardiovascular Engineering and Technology* 9(4):539-543. DOI: 10.1007/s13239-018-00393-z.

Verma S, Pant M, Snasel V. 2021. A comprehensive review on NSGA-II for multi-objective combinatorial optimization problems. *IEEE Access* 9:57757-57791. DOI: 10.1109/access.2021.3070634.

Wilkinson MD, Dumontier M, Aalbersberg IJ, Appleton G, Axton M, Baak A, Blomberg N, Boiten JW, da Silva Santos LB, Bourne PE, et al. 2016. The FAIR guiding principles for scientific data management and stewardship. *Scientific Data* 3(1):160018. DOI: 10.1038/sdata.2016.18.

Winkler AM, Ridgway GR, Webster MA, Smith SM, Nichols TE. 2014. Permutation inference for the general linear model. *NeuroImage* 92:381-397. DOI: 10.1016/j.neuroimage.2014.01.060.

Wu C, Lorenzo G, Hormuth DA, Lima EABF, Slavkova KP, DiCarlo JC, Virostko J, Phillips CM, Patt D, Chung C, et al. 2022. Integrating mechanism-based modeling with biomedical imaging to build practical digital twins for clinical oncology. *Biophysics Reviews* 3(2):021304. DOI: 10.1063/5.0086789.

## Additional information

### Competing interests

The author declares that he has no competing interests.

### Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

### Data availability

The trial-level paired dataset, the controller definitions, the frozen scoring rule, the endpoint definitions and the analysis code are archived at Zenodo (DOI: 10.5281/zenodo.23726433) and at https://github.com/Javad-Alizargar/locked-paired-validation-therapy-controllers (release v1.0.0).

### Code availability

The analysis is reproducible from a single entry point. `peerj_analysis/run_all.py` regenerates every table and figure in this article from the committed trial-level data with a fixed random seed. Figure files are provided separately as `Figure 1.png` to `Figure 5.png`.
