# Reproducibility

## Commands

Use Python 3.12 with requirements-revision.txt. Run python reproduce_submission.py, then python peerj_analysis/test_revision.py. The reproduction script executes run_all.py, make_tables.py, make_figures.py, build_docx.py, build_docx_review.py and make_final_package.py. Exact environment metadata is written to peerj_analysis/results/environment_versions.json.

## Original data and parent mapping

The paired dataset is unchanged: SHA-256 D8D59D86691039F30336ED4FF00A72643E3BD9475BB16856AF5DE67986B1174C. It contains 18,432 paired rows. parent_draw_map.csv is the unique seed/family/atlas-ID/original-uncertainty-ID mapping extracted from upstream locked_large_validation_outcomes.csv. The generator concatenates four transformed copies of each seed's 32 original draws. load_paired validates the mapping and uses 96 parent blocks, with 48 paired trials per controller per block. Historical family-specific identifiers remain in legacy_cluster_id in memory. The source CSV is never rewritten.

## Uncertainty and multiplicity

The bootstrap samples parent blocks with replacement within each seed, retaining all family variants and scenarios. After band filtering, ratios of summed differences to counts preserve trial weighting. There are 10,000 replicates. This is an empirical sensitivity summary conditional on seed allocation; it does not reproduce the Latin-hypercube randomisation law or establish coverage across a population of seeds.

Exploratory sign-flip tests use 20,000 sign vectors, seed 20261028, and (1 + extreme count)/(20,001). They require joint sign invariance of parent-block differences; randomised controller assignment does not establish that assumption here. Holm is joint across 12 success-band cells and separately across 20 all-trial mechanism cells. It cannot repair invalid p-values. Overall success tests are descriptive. MDEs use an unadjusted normal approximation; precision_audit.csv supplies a separate Bonferroni-12 planning reference, not proven procedure-level power.

## Retrospective score and implemented endpoints

The score uses log10(dose capacity), log10(reaction number) multiplied by observation quality, and log10(reaction-drive balance). Reaction number uses realised average baseline intensity. The bands are therefore retrospective, and baseline assignments are reused for each alternative. band_calibration.csv records observed and predicted band means.

Endpoints are equally weighted radial-node fractions at the final time of a 3,000 s run. max_temperature_c is the final spatial maximum, not the treatment-wide peak. safe_heat does not prove absence of earlier excursions or injury. warm_safe_joint.csv exposes the exclusion of overheating from thermal failure.

## Sensitivity and ranking

The 16 specifications cross four seed subsets with four scenario populations. Scenario filtering changes the estimand. Figure 5 marks the primary point, shows approximate intervals and provides specification_plot_key.csv. Figure 4 shows point-estimate Pareto comparisons and ranks from 200,000 Dirichlet(1,1,1,1,1) vectors after min-max normalisation over four alternatives. The baseline is a reference, not a fifth candidate. Ranks depend on the chosen distribution, objectives and candidates.

## Determinism and availability

Run the complete workflow twice and compare CSV/JSON hashes in the pinned environment. DOCX ZIP timestamps do not imply numerical nondeterminism. The original v1.0.0 DOI 10.5281/zenodo.23726433 remains the earlier version. Revised release v1.1.1 and the complete Supplemental Code S1 ZIP supply the corrected artefacts; no new archive DOI is invented.
