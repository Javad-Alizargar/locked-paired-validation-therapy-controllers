# Reproducible paired evaluation of closed-loop therapy software: controller comparisons depend on sampling and scenario choices

## Abstract

**Background.** Closed-loop therapy software is often compared using an aggregate simulation success rate. We evaluated whether apparent advantages persist when shared uncertainty draws, multiple comparisons and scenario dependence are made explicit.

**Methods.** Four controllers were compared with delayed full feedback across 4,608 paired trials each. Three seeds supplied 32 Latin-hypercube parent draws each; four transformed model families and 12 sensing scenarios reused each draw. We used approximate percentile intervals from 10,000 seed-stratified bootstrap resamples of the 96 parent blocks and exploratory Monte Carlo sign-flip tests with 20,000 vectors. Holm adjustment was applied jointly within 12 controller-by-band success tests and 20 mechanism tests. Bands used a frozen retrospective baseline score that includes realised baseline energy; they are unavailable before treatment. Success combines final-time radial-node criteria, rather than temporal coverage. Scenario sensitivity and point-estimate Pareto analyses were descriptive.

**Results.** All overall success differences were negative: smoothed predictive -0.018 (95% CI -0.051 to 0.013; p = 0.272), thermal-floor predictive -0.049 (95% CI -0.097 to 0.001; p = 0.061), Constrained optimized -0.007 (95% CI -0.046 to 0.033; p = 0.737), and reduced-order model predictive control -0.432 (95% CI -0.497 to -0.365; p < 0.001). Three positive low-band estimates ranged from +0.031 to +0.056; their unadjusted intervals included zero and Holm-adjusted p-values were 1.000. In the high band, thermal-floor predictive and model predictive control retained adjusted evidence of lower success. All four alternatives were non-dominated at their observed point estimates. Under 200,000 specified Dirichlet weight draws, the most frequent first-ranked controller attained 38.5%. Overall effects changed sign across seed and scenario subsets for three controllers.

**Conclusions.** No alternative demonstrated improved overall success in this atlas. Reproducible comparison requires tracing shared sampling units, auditing score provenance and endpoint meaning, and separating descriptive sensitivity analyses from inferential guarantees. The approximate uncertainty summaries do not establish population-wide or clinical superiority.

## Introduction

Closed-loop control is a common design pattern in thermal therapy software. Model predictive control has been studied for MR-guided high-intensity focused ultrasound (Sebeke et al., 2019) and magnetic nanoparticle hyperthermia (Abu-Ayyad et al., 2024; Lad et al., 2025). Temperature-feedback controllers have been evaluated on magnetic hyperthermia hardware (Sharma et al., 2023), and earlier work addressed phased-array heating (Kowalski & Jin, 2003). Gold-nanorod photothermal models describe optical heating (Maleki et al., 2021). Combined photothermal and photodynamic approaches additionally involve reactive oxygen species (ROS) generation and oxygen consumption (Bucharskaya et al., 2022); graphene-based combinations have also been reviewed (Khan et al., 2025). A decision that improves one endpoint can degrade another.

A comparison in one nominal simulation, or across a modest stress grid summarised by a single success rate, can leave the target population and uncertainty assumptions unclear. That design answers a narrow question: does the controller help in the specific scenarios that were run? It does not answer the question a reader actually cares about, which is whether the controller is better on average, for a stated population of uncertainty, at a stated level of statistical confidence, without having selected the favourable comparison. Three failure modes are well documented in adjacent fields. First, seed and run-to-run variance can be large enough to reorder methods that are reported as clearly separated (Henderson et al., 2018). Second, point estimates without interval estimates hide that ordering uncertainty, which is why stratified bootstrap intervals have been recommended for reinforcement-learning benchmarks (Agarwal et al., 2021). Third, the space of defensible analysis choices is wide, and unrecorded choices inflate apparent effects (Simmons, Nelson & Simonsohn, 2011; Steegen et al., 2016; Simonsohn, Simmons & Nelson, 2020). Reproducibility initiatives in machine learning have made the same point from the infrastructure side (Pineau et al., 2021), and the wider computational-science literature has documented both the scale of the problem (Baker, 2016) and practical remedies for it (Sandve et al., 2013; Wilkinson et al., 2016).

Biomedical simulation has a mature vocabulary for exactly this problem. Verification, validation and uncertainty quantification (VVUQ) frameworks require an explicit context of use, a documented evidence base, and a statement of the conditions under which a model is credible (Pathmanathan et al., 2017; Mulugeta et al., 2018; Steinman & Migliavacca, 2018; Erdemir et al., 2020; Santiago et al., 2022). The same discipline has been extended to digital twins for precision medicine (Sel et al., 2025), to multidomain photoacoustic modelling (Akhlaghi et al., 2019), and to personalised computational medicine and mechanism-based oncology models (Collin et al., 2022; Wu et al., 2022). Broader digital-twin modelling challenges are reviewed by Rasheed, San & Kvamsdal (2020). Related scientific machine-learning work describes equation discovery and physics-informed modelling (Rudy et al., 2017; Cai et al., 2021; Karniadakis et al., 2021); the present analysis does not use those methods. A comparative controller study must additionally specify a statistical target and sampling design for *ranking controllers* against each other. VVUQ asks whether a model is credible; controller evaluation asks whether one decision rule beats another across a population of patients, geometries and sensing conditions, and that is an inferential question.

The statistical toolkit is established, but its assumptions require attention. Pairing compares policies on matched cases; resampling must retain dependencies introduced by shared draws. Bootstrap intervals describe uncertainty only under the sampling approximation used, and sign-flip tests require an appropriate sign-symmetry null (Efron & Tibshirani, 1993; DiCiccio & Efron, 1996; Winkler et al., 2014). Holm adjustment controls family-wise error when its input p-values are valid (Holm, 1979); Benjamini-Hochberg control instead targets false discoveries (Benjamini & Hochberg, 1995). Approximate minimum detectable effects express precision under specified testing assumptions (Cohen, 1988). Pareto analysis describes multi-objective trade-offs without imposing a single utility (Verma, Pant & Snasel, 2021).

We therefore asked whether any of four delayed-feedback controllers improves composite success relative to delayed full feedback in an existing simulation atlas, and how that comparison depends on the sampling unit and scenario population. The computational testbed is a case study of evaluation practice, not a clinical efficacy study. The evaluation workflow is summarised in Fig. 1.

The contribution is a reproducible paired-evaluation workflow with explicit data contracts, parent-draw resampling, declared multiplicity families, scenario sensitivity and a descriptive Pareto statement. An audit of the original workflow identified shared draws across model families, post-treatment information in a score, and mismatches between endpoint descriptions and simulator outputs. The revised workflow makes those dependencies visible and reports the negative controller comparison with its practical limits. Standard statistical components are combined into an inspectable software artefact; no new inferential theorem or universally superior controller is claimed.

## Materials & Methods

### Simulation model

The case study is a one-dimensional radial model of a vascularised tumour spheroid, implemented in Python as a computational testbed. Each simulation uses 96 uniformly spaced radial nodes. Parent draws sample radius between 300 and 800 µm; the large-geometry family multiplies radius by 1.22 and clips it to 300–900 µm. Temperature is advanced with the Pennes bioheat equation (Pennes, 1948) using backward Euler integration and a tridiagonal solve. The testbed assumes tissue density of 1060 kg m⁻³, specific heat of 3600 J kg⁻¹ K⁻¹ and thermal conductivity of 0.50 W m⁻¹ K⁻¹; these fixed values are modelling assumptions, not measurements from a patient cohort. The actual controller atlas uses 808 nm illumination, 70 × 18 nm nanorods and a medium refractive index of 1.33. Its optical source uses a quasi-static Gans-style spheroidal absorption approximation (Gans, 1912) with a Rakić-style gold dielectric function (Rakić et al., 1998), combined with a radial nanoparticle profile with surface-bias parameter 0.30, and attenuated fluence. The broader simulator supports magnetic heating, but the closed-loop atlas uses the optical source alone.

ROS generation is integrated on a 0.5 s substep inside each 10 s control interval. Accumulated singlet-oxygen dose increases while illumination is commanded, at a rate proportional to fluence, local photosensitiser concentration and oxygen saturation. The model includes photobleaching and oxygen replenishment towards 40 µM, starting from a radial oxygen gradient. The photochemical constants, 40 µM oxygen reference and 0.67 mM singlet-oxygen threshold were adopted from the BPD-mediated macroscopic dosimetry model of Kim et al. (2016). That source fitted the threshold in mouse fibrosarcomas under 690 nm illumination. Its transfer to this 808 nm screening model is an assumption, not a validated wavelength-independent dose threshold. Every controller simulation runs for a fixed 3,000 s wall-clock horizon. Light-on time and delivered energy vary with the commands; this is not the earlier fixed-300 s-light-on duty-cycle experiment implemented elsewhere in the repository.

Dose-capacity and delay-to-diffusion quantities use sampled design and sensing parameters. In contrast, the reaction number includes realised average baseline intensity, computed as delivered baseline energy divided by protocol duration. The score therefore requires the completed baseline trajectory. Full equations and parameter configurations are supplied with the simulator, and the score provenance is stated explicitly below.

### Controller families

Five decision rules were compared with a common 10 s control interval. The baseline, smoothed, thermal-floor and reduced-order controllers have a 1.30 W cm⁻² command ceiling. Constrained optimized uses the lower ceiling selected during training.

*Delayed full feedback* is the baseline. It uses delayed, noisy measurements without the predictive estimator. It nevertheless applies temperature safety stops, oxygen-dependent restrictions, warm-fraction and ROS-completion rules, and maintenance commands. It is a gated feedback policy, not an ungated or uncontrolled treatment.

*Smoothed predictive* estimates the current state with a six-sample sliding window, applies a prediction gain of 0.85, and enforces a 0.35 °C thermal margin. It adds delay-dependent safety rules: at 60 s or more of delay it stops above 51.5 °C unless ROS coverage is already at 0.80, and at 120 s or more it stops above 51.0 °C unless ROS coverage is already at 0.75.

*Thermal-floor predictive* uses the same estimator and adds a thermal floor. If the warm fraction is below 0.48 and the maximum temperature is below 48.6 °C, it applies a floor intensity of 0.58 times the maximum, scaled down when oxygen is limiting. Once ROS coverage reaches 0.82 and the warm fraction target is met, it drops to a 0.22 W cm⁻² maintenance intensity or stops.

*Constrained optimized* is a re-parameterisation of the thermal-floor architecture, selected by a chance-constrained random search over 20 candidates on two independent training seeds (20260730 and 20260731) with eight draws each, then validated without further tuning on the three validation seeds. Its selected parameters are a maximum intensity of 1.128 W cm⁻², prediction gain 0.669, thermal margin 0.550 °C, thermal-floor warm fraction 0.478, thermal-floor temperature 48.52 °C, floor fraction 0.437, maintenance intensity 0.212 W cm⁻², long-delay safety temperature 50.64 °C, and mid-delay safety temperature 51.09 °C.

*Reduced-order MPC* is a receding-horizon benchmark. At each control step it propagates a four-state reduced model (temperature, warm fraction, ROS coverage, mean oxygen) four steps ahead over an eight-point intensity grid, discounts by 0.88 per step, and minimises a stage cost with weights 0.045 on energy, 4.8 on overheating, 0.9 on oxygen depletion, 1.8 on thermal deviation and 1.2 on ROS shortfall. It hard-stops above 52.2 °C unless ROS coverage is already at 0.78, and stops when oxygen falls below 8.0 µM with ROS coverage below 0.82.

### Validation atlas and analysis provenance

The atlas uses three validation seeds (20260720, 20260721, 20260722). Within each seed, one 32-row Latin-hypercube sample (McKay, Beckman & Conover, 1979) supplies perfusion, radius, photosensitiser concentration and nanoparticle volume fraction. Each row is copied into four model families: unchanged baseline, large geometry, low photosensitiser (concentration multiplied by 0.45 and clipped to 0.02–2.0 µM), and slow recovery (perfusion multiplied by 0.55 and clipped to 0.0015–0.030 s⁻¹). Each family variant is simulated under six delays (0, 30, 60, 90, 120, 180 s) and two noise levels (0%, 10%). A controller contributes 3 × 32 × 4 × 12 = 4,608 paired comparisons. Pairing matches seed, family, draw and sensing scenario. The resampling unit retains all four family variants and all 12 scenarios of a parent draw: 96 seed × parent-draw blocks, with 48 paired trials each. The 384 family-specific draw identifiers are not independent parent samples.

Controller parameters and the scoring coefficients were reused without tuning on these validation outcomes. However, the present statistical revision was undertaken after examination of the atlas, and is a retrospective reanalysis. Its bootstrap grouping and multiplicity implementation correct the earlier workflow. The analysis release fixes the revised code for reproducibility; that fact does not establish prospective locking or independence from analyst judgement. The earlier release remains identifiable in the version history. No prospective registration is claimed.

### Frozen retrospective baseline-score bands

Trials were grouped using a previously fitted logistic scoring equation, applied to the baseline trajectory only. Let C denote dose capacity, R the reaction number, Q observation quality and B the reaction-drive balance. The input transforms are log10(C), log10(R) × Q, and log10(B). In particular, the second transform is the logarithm of R multiplied by Q, not the logarithm of their product. The equation is

logit(P) = -0.016 + 1.406 z1 + 0.930 z2 - 1.482 z3,

where z1, z2 and z3 standardise those transforms with means -0.625, 0.319 and -3.530 and standard deviations 0.549, 0.307 and 0.576. Coefficients and constants came from separate development seeds. The reaction number is realised average intensity × photosensitiser concentration / perfusion, so this is a retrospective score. Band boundaries of 0.364 and 0.595 are reference-population quantiles, not validation-population terciles. The existing assignment yields 1,122 low, 1,674 intermediate and 1,812 high cases per controller. We retain it to describe the existing atlas, not as a prospective routing rule. Every controller comparison uses the same baseline-derived assignment; alternative-controller outcomes are not used to assign bands.

We use the terms *retrospective baseline-score band*, abbreviated to *low*, *intermediate* and *high*, throughout. These labels do not identify validated controllability or pre-treatment prognosis. The score has validation area under the receiver-operating characteristic curve (AUC) 0.688, with seed × family values 0.590–0.727, and AUC 0.610 on a separate held-out seed (Table S1). Its held-out Brier score is 0.265. These are retrospective discrimination summaries, not estimates of prospective clinical prediction performance.

### Endpoint definitions

The composite endpoint combines four conditions evaluated at the end of the 3,000 s run (Table S2). A case is warm if at least 0.50 of the equally weighted radial nodes have 43 ≤ T ≤ 50 °C. ROS coverage requires at least 0.50 of nodes to have accumulated dose ≥ 0.67 mM. Hypoxic control requires the fraction of nodes with oxygen < 5 µM and subthreshold ROS to be ≤ 0.35. The stored safe-heat flag requires an overheat-node fraction ≤ 0.05 and a final spatial maximum ≤ 55 °C; the latter condition also implies zero final nodes above 55 °C. Success requires all four flags. Thermal failure is NOT warm AND safe_heat, ROS failure is NOT ROS coverage, hypoxic failure is NOT hypoxic control, and overheating is NOT safe_heat. Delivered energy is recorded in J cm⁻². The stored maximum temperature is the final spatial maximum. None of these fractions is a fraction of treatment time or a spherical-volume fraction, and the temperature flags do not exclude earlier thermal excursions or demonstrate absence of tissue injury.

### Statistical analysis

All comparisons use within-trial controller-minus-baseline differences. The reported estimand averages over the equally represented families and sensing scenarios of this atlas. Percentile intervals use 10,000 bootstrap resamples of the parent blocks, sampling with replacement within each validation seed and retaining its observed allocation. The same block indices are shared across endpoints in each controller × band cell. Band filtering makes block sizes unequal; ratios of summed differences to summed counts preserve the trial-weighted estimand. The bootstrap preserves reuse across families and scenarios, but does not reproduce the dependent Latin-hypercube sampling law or estimate uncertainty across a population of seeds from only three seeds. Its 95% percentile intervals are therefore approximate empirical sensitivity summaries, not guaranteed design-based confidence intervals.

We additionally report exploratory Monte Carlo sign-flip tests using 20,000 random sign vectors applied to parent-block sums. The two-sided statistic is the absolute trial-weighted mean difference, with p = (1 + number of simulated statistics at least as extreme)/(20,000 + 1). Validity requires invariance of the joint block-difference distribution under the sign flips; zero mean alone is insufficient. There was no random assignment of controller labels, and the Latin-hypercube rows are dependent. We therefore do not claim exact randomisation inference or established population error control. Holm adjustments are applied jointly within two declared descriptive families: the 12 controller × band success contrasts and the 20 all-trial controller × mechanism contrasts (thermal failure, ROS failure, hypoxic failure, overheating, energy). Adjustment cannot repair invalid input p-values. Overall success contrasts are descriptive and excluded from those families. p-values are reported to three decimals, except values below 0.001; the Monte Carlo minimum is 1/20,001.

Approximate minimum detectable effects (MDEs) use (z0.975 + z0.80) × bootstrap standard error = 2.802 × standard error for an unadjusted, two-sided 5% normal-approximation test. They are precision summaries, not demonstrated power for the sign-flip procedure or the Holm-controlled family. The results also include conservative Bonferroni-12 normal-approximation planning references in precision_audit.csv. Neither calculation guarantees power under the Latin-hypercube design. Seed-specific effects and the specification curve show variation that the conditional within-seed bootstrap cannot settle.

### Specification curve

We evaluated 16 seed/scenario specifications per controller: four seed subsets (all seeds, or each seed alone) crossed with all scenarios, nominal only (0 s delay, 0% noise), delayed only (≥ 120 s), and noisy only (10%). Scenario restriction changes the target population and thus the estimand. These analyses measure population and seed sensitivity, not alternative estimates of one invariant effect. Each uses 2,000 parent-block bootstrap replicates; naive unpaired Wald intervals are retained as an illustrative comparator. The primary all-seed/all-scenario estimate is marked in the specification plot, with intervals and a machine-readable specification key. The fraction positive is descriptive, and no significance count is treated as a new family of discoveries.

### Pareto dominance and weight sensitivity

Controller performance was summarised over five objectives: success change (maximised), thermal-failure change, ROS-failure change, overheating change and delivered-energy change (all minimised). Controller A dominates controller B if A is at least as good on every objective and strictly better on at least one. A controller is Pareto optimal if no other controller dominates it.

Each objective was min-max normalised across the four alternatives and oriented so that larger is better. We drew 200,000 Dirichlet(1,1,1,1,1) weight vectors and recorded weighted-score ranks. These frequencies are conditional on this particular normalisation, objective set, controller set and weight distribution. They are not probabilities of clinical superiority or a distribution-free statement about preferences. Pareto membership is computed from observed means without uncertainty propagation. The baseline is the reference for changes, not a fifth candidate in the Pareto set.

### Software, data and reproducibility

The analysis uses Python 3.12 and NumPy, pandas, SciPy and Matplotlib (Hunter, 2007; McKinney, 2010; Harris et al., 2020; Virtanen et al., 2020). Exact package versions and platform information are recorded in environment_versions.json and requirements-revision.txt. No GPU is used. The baseline comparison and resampling code run on the committed data; a reproduction script executes the result, table, figure and document builders in sequence. Images are separate programmatic plots, at 300 dpi and within the journal raster limits. Third-party library references are included below.

### Ethics

This study uses no human participants, human tissue or animal subjects. It is a computational analysis of simulated data, so ethical approval and informed consent were not required.

### Use of generative AI

OpenAI ChatGPT (GPT-5) was previously used for drafting and code organisation. OpenAI Codex (GPT-6) was used in October 2026 to audit the submission, revise the statistical implementation and manuscript, and execute reproducibility and layout checks. AI assistance included identifying analysis defects, drafting revised code and text, and interpreting program output; it was not limited to language editing. All reported statistics are produced by the supplied executable code from the committed simulation data. No generative image model was used. Figures are deterministic plots generated by Python from the result files. The author remains responsible for verifying the revised analyses, interpretations and references and for retaining relevant prompt and revision records. The author confirms originality and accuracy, confirms having reviewed the tools’ terms of use for suitability, and takes responsibility for the integrity of the whole article, including references.

## Results

### All overall point estimates are negative

Across 4,608 paired comparisons per controller, the baseline success rate was 0.560 (Table 1, Fig. 2). Smoothed predictive changed success by -0.018 (95% CI -0.051 to 0.013; p = 0.272). Thermal-floor predictive changed success by -0.049 (95% CI -0.097 to 0.001; p = 0.061). Constrained optimized changed success by -0.007 (95% CI -0.046 to 0.033; p = 0.737). Reduced-order MPC changed success by -0.432 (95% CI -0.497 to -0.365; p < 0.001). All four point estimates were negative. Only the reduced-order MPC interval excluded zero under the revised parent-block bootstrap; the other three included zero. No alternative demonstrated improved overall success. These intervals and p-values retain the conditional, approximate interpretation specified in Methods.

Approximate all-trial per-comparison MDEs ranged from 0.046 to 0.093. The matching low-band MDEs were 0.080, 0.135 and 0.103 for smoothed predictive, thermal-floor predictive and Constrained optimized respectively. A band contrast has its own precision and cannot be judged against the all-trial MDE. Inclusion of zero in an interval does not establish equivalence or a benefit.

### Failure modes move in opposite directions

Changes in the stored final-time endpoints differed across alternatives (Table 2, Fig. 3). Smoothed predictive: thermal failure +0.038 (0.009 to 0.069; Holm-adjusted p = 0.143), ROS failure -0.023 (-0.041 to -0.009; Holm-adjusted p = 0.031), overheating -0.012 (-0.025 to 0.003; Holm-adjusted p = 0.828), energy (J cm⁻²) -23.5 (-33.6 to -13.2; Holm-adjusted p < 0.001). Thermal-floor predictive: thermal failure -0.037 (-0.081 to 0.006; Holm-adjusted p = 0.828), ROS failure -0.017 (-0.031 to -0.005; Holm-adjusted p = 0.124), overheating +0.095 (0.074 to 0.116; Holm-adjusted p < 0.001), energy (J cm⁻²) +44.2 (31.1 to 57.8; Holm-adjusted p < 0.001). Constrained optimized: thermal failure -0.010 (-0.052 to 0.030; Holm-adjusted p = 1.000), ROS failure -0.002 (-0.015 to 0.010; Holm-adjusted p = 1.000), overheating +0.020 (0.008 to 0.031; Holm-adjusted p = 0.027), energy (J cm⁻²) -59.1 (-65.3 to -53.0; Holm-adjusted p < 0.001). Reduced-order MPC: thermal failure -0.048 (-0.093 to -0.003; Holm-adjusted p = 0.336), ROS failure -0.054 (-0.079 to -0.031; Holm-adjusted p < 0.001), overheating +0.497 (0.456 to 0.538; Holm-adjusted p < 0.001), energy (J cm⁻²) +549.7 (498.2 to 603.5; Holm-adjusted p < 0.001). Hypoxic-failure changes were zero for all four alternatives, with degenerate empirical intervals. These are adjusted exploratory summaries under the stated sign-symmetry assumption; they are not confirmatory clinical findings.

Thermal failure excludes cases failing the final-time temperature criterion. Its reduction can therefore arise from movement into overheating, rather than improved heating reliability. The executable analysis supplies the joint warm/safe endpoint counts in warm_safe_joint.csv. We do not infer a causal mechanism from marginal failure differences. The observed increase in MPC overheating and energy, together with its lower success rate, is specific to this simplified benchmark and these final-time definitions.

### Positive low-band estimates remain imprecise

The retrospective low band contains 1,122 pairs in 54 parent blocks, with baseline success 0.383 (Table 3). Smoothed predictive: +0.031 (95% CI -0.025 to 0.087; p = 0.302); Holm-adjusted p = 1.000. Thermal-floor predictive: +0.056 (95% CI -0.040 to 0.149; p = 0.297); Holm-adjusted p = 1.000. Constrained optimized: +0.050 (95% CI -0.024 to 0.122; p = 0.215); Holm-adjusted p = 1.000. Reduced-order MPC: -0.210 (95% CI -0.319 to -0.111; p < 0.001); Holm-adjusted p = 0.002. The three positive estimates already had unadjusted intervals spanning zero. Holm correction did not change their non-significant status.

The low-band point estimates are compatible with benefit, no effect or harm within their approximate intervals. They are insufficient to identify a useful controller or to justify a prospective selection rule. In addition, band membership is defined partly by realised baseline energy, so these comparisons describe cases after baseline simulation; they are not evidence of treatment-effect modification identifiable before treatment.

### High-band success changes are consistently negative

The retrospective high band contains 1,812 pairs in 72 parent blocks, with baseline success 0.746. Smoothed predictive: -0.063 (95% CI -0.107 to -0.016; p = 0.015); Holm-adjusted p = 0.121. Thermal-floor predictive: -0.119 (95% CI -0.183 to -0.051; p = 0.002); Holm-adjusted p = 0.021. Constrained optimized: -0.047 (95% CI -0.103 to 0.012; p = 0.130); Holm-adjusted p = 0.910. Reduced-order MPC: -0.645 (95% CI -0.711 to -0.573; p < 0.001); Holm-adjusted p < 0.001. Under the revised exploratory testing procedure, thermal-floor predictive and reduced-order MPC retained adjusted p-values below 0.05. Smoothed predictive and Constrained optimized did not. The intermediate band contains 1,674 pairs in 81 parent blocks; its results appear in Table 3. Blocks can contribute trials to more than one band.

Negative high-band estimates indicate that changing the baseline policy can reduce the composite endpoint in cases with a high retrospective baseline score. The data do not establish why a gate or floor fired, whether it was unnecessary, or whether the effect could be predicted prospectively. Trace-level interventions or controller ablations would be needed to support such mechanistic explanations.

### Non-dominance and ranks depend on the declared objectives

Across the five observed objectives, none of the four alternatives dominated another (Fig. 4). This descriptive result does not require every alternative to be best on an individual objective. Constrained optimized had the least negative success change and largest energy saving; smoothed predictive had the lowest overheating change; reduced-order MPC had the largest reductions in the stored thermal- and ROS-failure rates. Thermal-floor predictive was not best on any single objective, but its combination was not dominated. Sampling uncertainty and alternative endpoint definitions could change the frontier.

Under the specified 200,000 Dirichlet weight draws, thermal-floor predictive ranked first in 38.5%, smoothed predictive in 27.9%, Constrained optimized in 22.8%, and reduced-order MPC in 10.7% (Table S3). Thermal-floor predictive ranked in the top two in 69.5%, while MPC ranked last in 72.3%. No alternative ranked first in more than half of these sampled vectors. This statement concerns the chosen weight distribution; it does not say that every weighting lacks a winner or that the weight distribution represents clinical preferences.

### The sign of the effect depends on analysis choices

The descriptive specification curve shows population sensitivity (Fig. 5). Across 16 seed/scenario subsets, estimates were positive in 37.5% for Constrained optimized (range -0.104 to +0.125), 25% for smoothed predictive (-0.100 to +0.020), 25% for thermal-floor predictive (-0.174 to +0.117), and 0% for reduced-order MPC (-0.617 to -0.281). The primary all-seed/all-scenario point is marked by a diamond. Intervals visualise approximate within-subset precision; the key supplies each subset identity.

Nominal-only comparisons omit the delays and noise included in the full atlas, and therefore answer a different population question. Single-seed subsets show variation across the three realised Latin-hypercube designs. The naive unpaired comparator changes the interval calculation, not the paired mean difference. These contrasts show that scenario selection, seed selection and variance estimation should be reported separately. They do not prove that arbitrary analyst choices reverse an otherwise fixed population effect.

### Discrimination of the scoring rule is modest

The retrospective score has validation AUC 0.688 and held-out-seed AUC 0.610 with Brier score 0.265 (Table S1). Mean predicted probabilities in the low, intermediate and high bands were 0.268, 0.480, 0.764, compared with observed baseline success 0.383, 0.477 and 0.746. The low band underpredicts observed success. An external Brier score of 0.265 is worse than the approximately 0.250 score of a constant forecast at the external prevalence. AUC alone is therefore insufficient evidence of useful calibration, and none of these retrospective diagnostics validates a pre-treatment score.

## Discussion

### Principal findings

Four delayed-feedback alternatives had negative overall point estimates against delayed full feedback. Parent-block resampling widened uncertainty relative to treating family variants as separate draws, and thermal-floor predictive no longer had an all-trial interval excluding zero. Low-band positive estimates remained imprecise before and after correction. All four alternatives were non-dominated at observed means under the declared objectives. The principal methodological finding is that reproducibility alone does not establish correct sampling units, valid score provenance or appropriate endpoint interpretation.

### Why auditing the evaluation changes the conclusion

The same committed outcomes can produce different uncertainty summaries when dependent family variants are treated as independent samples. The revised grouping preserves their common parent draw, while seed-stratified resampling retains the observed seed allocation. Similarly, grouping on a baseline score is not prospective stratification when realised baseline intensity enters the features. Neither a frozen equation nor a fixed random seed removes these design issues. The revisions retain all original trial outcomes and point estimates rather than selecting a more favourable controller result.

These concerns are consistent with those described in the multiverse and specification-curve literature in the social sciences (Simmons, Nelson & Simonsohn, 2011; Steegen et al., 2016; Simonsohn, Simmons & Nelson, 2020), and that benchmark studies document in machine learning (Henderson et al., 2018; Agarwal et al., 2021; Pineau et al., 2021). In this case study, the same concerns arise in therapy-software evaluation. Model credibility and comparative performance are distinct questions, and reproducible outputs can still support an invalid interpretation if sampling and scoring assumptions are not audited.

### Relation to previous work

The controllers studied here are related to thermal-therapy control work, including MR-HIFU control (Sebeke et al., 2019), magnetic nanoparticle hyperthermia (Abu-Ayyad et al., 2024; Lad et al., 2025), temperature-feedback hardware (Sharma et al., 2023), and phased-array heating (Kowalski & Jin, 2003). Our reduced-order MPC is a simplified receding-horizon benchmark. Its poor performance in this atlas concerns a four-step reduced model under 0–180 s sensing delay, not MPC in general.

The VVUQ literature supplies the framing for the credibility question (Pathmanathan et al., 2017; Mulugeta et al., 2018; Steinman & Migliavacca, 2018; Erdemir et al., 2020; Santiago et al., 2022; Sel et al., 2025) and for the specific case of imaging and oncology models (Akhlaghi et al., 2019; Collin et al., 2022; Wu et al., 2022). Our contribution is complementary: we use the existing model as a defined software testbed and ask a comparative question about decision rules within it, without asserting independent biophysical validation. The two questions are different, and a credible model can still yield an unsupported controller ranking if the comparison is not designed.

The statistical components are established: bootstrap methods (Efron & Tibshirani, 1993; DiCiccio & Efron, 1996), sign-flip inference under an appropriate sign-symmetry assumption (Winkler et al., 2014), Holm adjustment (Holm, 1979), and Pareto analysis (Verma, Pant & Snasel, 2021). Their implementation must be checked against the actual data-generating process. This study does not validate a new inferential method. It supplies an auditable case study and software workflow that expose sampling, scoring and endpoint assumptions.

### Implications for practice

Practical reporting should identify the parent sampling unit and any reuse of it, show paired effects with their uncertainty assumptions, and state every multiplicity family in executable form. Score features should be traced to the time at which they become available. Endpoint names should distinguish spatial coverage at the final time from coverage over treatment time. Multi-objective summaries should state the candidate set, normalisation and preference distribution, and distinguish observed non-dominance from uncertainty about dominance.

A precision summary is useful only for its corresponding population and procedure. The per-comparison normal-approximation MDE is not a familywise power guarantee, and adding seeds need not improve precision at the ideal inverse-square-root rate. Follow-up planning should simulate the full sampling and testing workflow under stated effect sizes, including the Latin-hypercube design, repeated model families and band assignment. A larger atlas cannot by itself make a retrospective score prospective.

### Limitations

The model is a one-dimensional radial screening representation, without angular heterogeneity, discrete vasculature or full optical transport. Its endpoints weight radial nodes equally, so central and peripheral tissue volumes do not receive spherical-volume weights. Final-time heating does not capture cumulative thermal dose or maximum temperature over treatment. The composite threshold has not been clinically validated. The optical implementation and transfer of BPD photochemical parameters to 808 nm have not been independently validated against the cited source experiments; those references establish model provenance rather than validation of this atlas. The retrospective score includes baseline treatment information and has modest discrimination and imperfect calibration. Consequently, none of the band results supports patient selection or clinical benefit.

The three realised Latin-hypercube designs supply dependent rows; the empirical parent-block bootstrap does not reproduce that sampling law. Treating seeds as fixed strata avoids pretending that three seeds provide a well-characterised seed distribution, but leaves between-seed uncertainty unresolved. Sign-flip p-values require a symmetry assumption not established by randomised controller assignment; Holm adjustment does not remove this limitation. The 16 specifications and weight analysis are descriptive and partly overlapping. The revised analysis was undertaken after outcomes were available. It improves transparency but is not an independently confirmed protocol.

### Future directions

Future work should first define prospective score features and fit them on independent development simulations. New validation runs should independently sample family-specific cases or explicitly preserve shared draws in a design-aware analysis. Dense traces should record treatment-wide peak temperature, time coverage and thermal dose, with volume-weighted spatial endpoints reported separately. More independently generated Latin-hypercube seeds would support empirical evaluation of uncertainty coverage and power. Controller ablations could test the proposed gate and floor mechanisms. These are new experiments; the present revision does not claim to have performed them.

## Conclusions

Reanalysis of the existing atlas found no alternative demonstrating improved overall composite success. The three positive low-band estimates were non-significant before and after Holm adjustment, and their baseline-derived membership is retrospective. The four alternatives were mutually non-dominated at observed means under the specified objectives, while ranks varied with sampled preferences. Controller comparisons should document the sampling unit, feature provenance, endpoint computation, multiplicity implementation and population targeted by each sensitivity analysis. Exact numerical reproduction is necessary for auditability but does not establish inferential or clinical validity.

## Acknowledgments

The author thanks the open-source maintainers of NumPy, pandas, SciPy and Matplotlib, on which the analysis depends.

## References

Abu-Ayyad M, Lad YS, Aguilar D, Karami K, Attaluri A. 2024. Model predictive control (MPC) applied to a simplified model, magnetic nanoparticle hyperthermia (MNPH) treatment process. *Biomedical Physics & Engineering Express* 10(4):045012. DOI: 10.1088/2057-1976/ad460a.

Agarwal R, Schwarzer M, Castro PS, Courville A, Bellemare MG. 2021. Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems* 34:29304-29320. Available at https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html

Akhlaghi N, Pfefer TJ, Wear KA, Garra BS, Vogt WC. 2019. Multidomain computational modeling of photoacoustic imaging: verification, validation, and image quality prediction. *Journal of Biomedical Optics* 24(12):1-12. DOI: 10.1117/1.JBO.24.12.121910.

Baker M. 2016. 1,500 scientists lift the lid on reproducibility. *Nature* 533(7604):452-454. DOI: 10.1038/533452a.

Benjamini Y, Hochberg Y. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society Series B: Statistical Methodology* 57(1):289-300. DOI: 10.1111/j.2517-6161.1995.tb02031.x.

Bucharskaya AB, Khlebtsov NG, Khlebtsov BN, Maslyakova GN, Navolokin NA, Genin VD, Genina EA, Tuchin VV. 2022. Photothermal and photodynamic therapy of tumors with plasmonic nanoparticles: challenges and prospects. *Materials* 15(4):1606. DOI: 10.3390/ma15041606.

Cai S, Mao Z, Wang Z, Yin M, Karniadakis GE. 2021. Physics-informed neural networks (PINNs) for fluid mechanics: a review. *Acta Mechanica Sinica* 37(12):1727-1738. DOI: 10.1007/s10409-021-01148-1.

Cohen J. 1988. *Statistical Power Analysis for the Behavioral Sciences*. 2nd edn. Hillsdale, NJ: Lawrence Erlbaum Associates.

Collin CB, Gebhardt T, Golebiewski M, Karaderi T, Hillemanns M, Khan FM, Salehzadeh-Yazdi A, Kirschner M, Krobitsch S, EU-STANDS4PM consortium, Kuepfer L. 2022. Computational models for clinical applications in personalized medicine: guidelines and recommendations for data integration and model validation. *Journal of Personalized Medicine* 12(2):166. DOI: 10.3390/jpm12020166.

DiCiccio TJ, Efron B. 1996. Bootstrap confidence intervals. *Statistical Science* 11(3):189-228. DOI: 10.1214/ss/1032280214.

Efron B, Tibshirani RJ. 1993. *An Introduction to the Bootstrap*. New York: Chapman & Hall/CRC. DOI: 10.1201/9780429246593.

Erdemir A, Mulugeta L, Ku JP, Drach A, Horner M, Morrison TM, Peng GCY, Vadigepalli R, Lytton WW, Myers JG. 2020. Credible practice of modeling and simulation in healthcare: ten rules from a multidisciplinary perspective. *Journal of Translational Medicine* 18(1):369. DOI: 10.1186/s12967-020-02540-4.

Gans R. 1912. Über die Form ultramikroskopischer Goldteilchen. *Annalen der Physik* 342(5):881-900. DOI: 10.1002/andp.19123420503.

Harris CR, Millman KJ, van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, Wieser E, Taylor J, Berg S, Smith NJ, Kern R, Picus M, Hoyer S, van Kerkwijk MH, Brett M, Haldane A, del Río JF, Wiebe M, Peterson P, Gérard-Marchant P, Sheppard K, Reddy T, Weckesser W, Abbasi H, Gohlke C, Oliphant TE. 2020. Array programming with NumPy. *Nature* 585(7825):357-362. DOI: 10.1038/s41586-020-2649-2.

Henderson P, Islam R, Bachman P, Pineau J, Precup D, Meger D. 2018. Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence* 32(1):3207-3214. DOI: 10.1609/aaai.v32i1.11694.

Holm S. 1979. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6:65-70.

Hunter JD. 2007. Matplotlib: a 2D graphics environment. *Computing in Science & Engineering* 9(3):90-95. DOI: 10.1109/MCSE.2007.55.

Karniadakis GE, Kevrekidis IG, Lu L, Perdikaris P, Wang S, Yang L. 2021. Physics-informed machine learning. *Nature Reviews Physics* 3(6):422-440. DOI: 10.1038/s42254-021-00314-5.

Khan S, Batool H, Tariq H, Noor A. 2025. Graphene oxide-based photothermal and photodynamic therapy: a systematic review. *Journal of Biomedical Materials Research Part B: Applied Biomaterials* 113(9):e35656. DOI: 10.1002/jbm.b.35656.

Kim MM, Penjweini R, Liang X, Zhu TC. 2016. Explicit macroscopic singlet oxygen modeling for benzoporphyrin derivative monoacid ring A (BPD)-mediated photodynamic therapy. *Journal of Photochemistry and Photobiology B: Biology* 164:314-322. DOI: 10.1016/j.jphotobiol.2016.09.031.

Kowalski ME, Jin JM. 2003. A temperature-based feedback control system for electromagnetic phased-array hyperthermia: theory and simulation. *Physics in Medicine and Biology* 48(5):633-651. DOI: 10.1088/0031-9155/48/5/306.

Lad YS, Pawar S, Arepally N, Carlton H, Hadjipanayis C, Ivkov R, Abu-Ayyad M, Attaluri A. 2025. Thermal dose feedback control systems applied to magnetic nanoparticle hyperthermia. *International Journal of Hyperthermia* 42(1):2491519. DOI: 10.1080/02656736.2025.2491519.

Maleki S, Farhadi M, Kamrava SK, Asghari A, Daneshi A. 2021. Simulation and in vitro experimental studies on targeted photothermal therapy of cancer using folate-PEG-gold nanorods. *Journal of Biomedical Physics and Engineering* 11(4):435-446. DOI: 10.31661/jbpe.v0i0.1108.

McKay MD, Beckman RJ, Conover WJ. 1979. A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics* 21(2):239-245. DOI: 10.1080/00401706.1979.10489755.

McKinney W. 2010. Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*:56-61. DOI: 10.25080/Majora-92bf1922-00a.

Mulugeta L, Drach A, Erdemir A, Hunt CA, Horner M, Ku JP, Myers JG Jr, Vadigepalli R, Lytton WW. 2018. Credibility, replicability, and reproducibility in simulation for biomedicine and clinical applications in neuroscience. *Frontiers in Neuroinformatics* 12:18. DOI: 10.3389/fninf.2018.00018.

Pathmanathan P, Gray RA, Romero VJ, Morrison TM. 2017. Applicability analysis of validation evidence for biomedical computational models. *Journal of Verification, Validation and Uncertainty Quantification* 2(2):021005. DOI: 10.1115/1.4037671.

Pennes HH. 1948. Analysis of tissue and arterial blood temperatures in the resting human forearm. *Journal of Applied Physiology* 1(2):93-122. DOI: 10.1152/jappl.1948.1.2.93.

Pineau J, Vincent-Lamarre P, Sinha K, Lariviere V, Beygelzimer A, d'Alche-Buc F, Fox E, Larochelle H. 2021. Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program). *Journal of Machine Learning Research* 22(164):1-20. Available at https://jmlr.org/papers/v22/20-303.html

Rakić AD, Djurišić AB, Elazar JM, Majewski ML. 1998. Optical properties of metallic films for vertical-cavity optoelectronic devices. *Applied Optics* 37(22):5271-5283. DOI: 10.1364/AO.37.005271.

Rasheed A, San O, Kvamsdal T. 2020. Digital twin: values, challenges and enablers from a modeling perspective. *IEEE Access* 8:21980-22012. DOI: 10.1109/access.2020.2970143.

Rudy SH, Brunton SL, Proctor JL, Kutz JN. 2017. Data-driven discovery of partial differential equations. *Science Advances* 3(4):e1602614. DOI: 10.1126/sciadv.1602614.

Sandve GK, Nekrutenko A, Taylor J, Hovig E. 2013. Ten simple rules for reproducible computational research. *PLoS Computational Biology* 9(10):e1003285. DOI: 10.1371/journal.pcbi.1003285.

Santiago A, Butakoff C, Eguzkitza B, Gray RA, May-Newman K, Pathmanathan P, Vu V, Vázquez M. 2022. Design and execution of a verification, validation, and uncertainty quantification plan for a numerical model of left ventricular flow after LVAD implantation. *PLoS Computational Biology* 18(6):e1010141. DOI: 10.1371/journal.pcbi.1010141.

Sebeke L, Deenen DA, Maljaars E, Heijman E, de Jager B, Heemels WPMH, Grüll H. 2019. Model predictive control for MR-HIFU-mediated, uniform hyperthermia. *International Journal of Hyperthermia* 36(1):1040-1050. DOI: 10.1080/02656736.2019.1668065.

Sel K, Hawkins-Daarud A, Chaudhuri A, Osman D, Bahai A, Paydarfar D, Willcox K, Chung C, Jafari R. 2025. Survey and perspective on verification, validation, and uncertainty quantification of digital twins for precision medicine. *npj Digital Medicine* 8(1):40. DOI: 10.1038/s41746-025-01447-y.

Sharma A, Jangam A, Shen JLY, Ahmad A, Arepally N, Rodriguez B, Borrello J, Bouras A, Kleinberg L, Ding K, Hadjipanayis C, Kraitchman DL, Ivkov R, Attaluri A. 2023. Validation of a temperature-feedback controlled automated magnetic hyperthermia therapy device. *Cancers* 15(2):327. DOI: 10.3390/cancers15020327.

Simmons JP, Nelson LD, Simonsohn U. 2011. False-positive psychology: undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science* 22(11):1359-1366. DOI: 10.1177/0956797611417632.

Simonsohn U, Simmons JP, Nelson LD. 2020. Specification curve analysis. *Nature Human Behaviour* 4(11):1208-1214. DOI: 10.1038/s41562-020-0912-z.

Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W. 2016. Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science* 11(5):702-712. DOI: 10.1177/1745691616658637.

Steinman DA, Migliavacca F. 2018. Editorial: special issue on verification, validation, and uncertainty quantification of cardiovascular models: towards effective VVUQ for translating cardiovascular modelling to clinical utility. *Cardiovascular Engineering and Technology* 9(4):539-543. DOI: 10.1007/s13239-018-00393-z.

Verma S, Pant M, Snasel V. 2021. A comprehensive review on NSGA-II for multi-objective combinatorial optimization problems. *IEEE Access* 9:57757-57791. DOI: 10.1109/access.2021.3070634.

Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, Burovski E, Peterson P, Weckesser W, Bright J, van der Walt SJ, Brett M, Wilson J, Millman KJ, Mayorov N, Nelson ARJ, Jones E, Kern R, Larson E, Carey CJ, Polat İ, Feng Y, Moore EW, VanderPlas J, Laxalde D, Perktold J, Cimrman R, Henriksen I, Quintero EA, Harris CR, Archibald AM, Ribeiro AH, Pedregosa F, van Mulbregt P, SciPy 1.0 Contributors. 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods* 17(3):261-272. DOI: 10.1038/s41592-019-0686-2.

Wilkinson MD, Dumontier M, Aalbersberg IJ, Appleton G, Axton M, Baak A, Blomberg N, Boiten JW, da Silva Santos LB, Bourne PE, Bouwman J, Brookes AJ, Clark T, Crosas M, Dillo I, Dumon O, Edmunds S, Evelo CT, Finkers R, Gonzalez-Beltran A, Gray AJG, Groth P, Goble C, Grethe JS, Heringa J, ’t Hoen PAC, Hooft R, Kuhn T, Kok R, Kok J, Lusher SJ, Martone ME, Mons A, Packer AL, Persson B, Rocca-Serra P, Roos M, van Schaik R, Sansone SA, Schultes E, Sengstag T, Slater T, Strawn G, Swertz MA, Thompson M, van der Lei J, van Mulligen E, Velterop J, Waagmeester A, Wittenburg P, Wolstencroft K, Zhao J, Mons B. 2016. The FAIR guiding principles for scientific data management and stewardship. *Scientific Data* 3(1):160018. DOI: 10.1038/sdata.2016.18.

Winkler AM, Ridgway GR, Webster MA, Smith SM, Nichols TE. 2014. Permutation inference for the general linear model. *NeuroImage* 92:381-397. DOI: 10.1016/j.neuroimage.2014.01.060.

Wu C, Lorenzo G, Hormuth DA, Lima EABF, Slavkova KP, DiCarlo JC, Virostko J, Phillips CM, Patt D, Chung C, Yankeelov TE. 2022. Integrating mechanism-based modeling with biomedical imaging to build practical digital twins for clinical oncology. *Biophysics Reviews* 3(2):021304. DOI: 10.1063/5.0086789.

## Additional information

### Competing interests

The author declares that he has no competing interests.

### Author contributions

Javad Alizargar: Conceptualization, Methodology, Software, Validation, Formal analysis, Data curation, Visualization, Writing - original draft, Writing - review and editing.

### Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

### Data availability

The original trial data and earlier analysis are archived at Zenodo (DOI: 10.5281/zenodo.23726433, version 1.0.0). The revised manuscript, analysis and generated results are supplied in the public repository https://github.com/Javad-Alizargar/locked-paired-validation-therapy-controllers (revision release v1.1.1). The original DOI identifies the earlier version and must not be interpreted as an archive of this revision. The complete revised code and data package is also supplied as Supplemental Code S1 for review.

### Code availability

The reproduction command is `python reproduce_submission.py`. It executes run_all.py, make_tables.py, make_figures.py, build_docx.py and build_docx_review.py in that order. The analysis entry point alone writes result CSV/JSON files; it does not build documents or images. The numerical audit includes score provenance, band calibration, seed effects, joint warm/safe counts and precision summaries. Figure files remain separate as Figure 1.png to Figure 5.png.
