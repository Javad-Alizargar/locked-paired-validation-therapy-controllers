# AGENTS.md — revised submission workflow

The user authorised correction of all pre-submission findings on 28 October 2026. This revision intentionally supersedes the earlier 384-family-cluster analysis and its expected p-values. Do not restore those obsolete inferential values. Original simulated outcomes are unchanged. The current manuscript is a retrospective evaluation case study with approximate uncertainty, not a prospectively validated clinical protocol.

## Work here

Manuscript source: peerj_submission/main_manuscript.md. Analysis: peerj_analysis/analysis.py. Figures/tables/results and DOCX files are generated: change sources and builders, never hand-edit the outputs. REVISION_NOTES.md records the sixteen audit resolutions. REPRODUCIBILITY.md describes the current workflow. Public supplementary contents are explicitly listed in release_files.txt.

## Commands and expected output

Use D:\PeerJ Computer Science\workspace\venv\Scripts\python.exe (Python 3.12).

```
python reproduce_submission.py
python peerj_analysis/test_revision.py
```

Expected: 4,608 pairs per controller, 96 parent blocks (48 trials each), 10,000 bootstrap replicates; validation AUC 0.688 and external retrospective AUC 0.610.

```
Constrained optimized      -0.007 [-0.046, +0.033] p=0.7373 MDE=0.056
Reduced-order MPC          -0.432 [-0.497, -0.365] p=0.0001 MDE=0.093
Smoothed predictive        -0.018 [-0.051, +0.013] p=0.2717 MDE=0.046
Thermal-floor predictive   -0.049 [-0.097, +0.001] p=0.0615 MDE=0.070
```

All overall point estimates remain negative. Three positive low-band estimates have Holm p=1.000. Only thermal-floor predictive and MPC have adjusted high-band p<0.05 under the explicitly exploratory sign-flip procedure. All four alternatives remain non-dominated at observed point estimates under the five declared objectives. Rank frequencies and specification point estimates remain descriptive. No superiority claim is justified.

## Scientific rules

- Never fabricate a number. Derive every statistic from executed code and check the CSVs. Original data SHA-256 is recorded in REPRODUCIBILITY.md.
- Keep all four transformed families of a parent LHS draw together. Bootstrap within seed. The empirical bootstrap is not exact LHS-design inference; sign flips require joint symmetry that is not established by randomised assignment.
- Apply Holm jointly to 12 success-band cells, and separately jointly to all 20 mechanism cells. Overall tests are descriptive. MDEs are approximate per-comparison normal-theory summaries.
- Score bands use realised baseline energy and are retrospective, unavailable before treatment. Never call them validated controllability or pre-treatment prognosis. log10(reaction number) is multiplied by observation quality, not log10 of the product.
- Endpoints use unweighted radial-node fractions at the final time of a 3,000 s run. The stored maximum is the final spatial maximum, not a treatment-wide peak. Do not claim clinical injury absence or temporal coverage.
- No prospective registration or prospective locking claim. This revision was made after outcome inspection. Keep the original release identifiable.
- Use UK spelling except the exact controller label Constrained optimized. Do not renumber Figures 4/5 or Tables S1/S2.

## Documents and privacy

Use US Letter, one-inch margins, 12 pt Times New Roman body text, 1.5 spacing, genuine Word heading styles, continuous manuscript line numbers. Figures and tables stay separate in the submission DOCX; the reviewer copy embeds them. Abstract must remain below 500 words and 3,000 characters. AI disclosure names actual tools and scope. Render and inspect every revised Word page before delivery; page count may change after authorised scientific revision.

Preserve the author block: Javad Alizargar, Department of Nursing, National Taipei University of Nursing and Health Sciences, No. 365 Mingde Rd, Beitou District, Taipei City 112, Taiwan; 8javad@ntunhs.edu.tw; ORCID 0000-0003-2966-1315.

Never read, print, copy or commit anything under D:\PeerJ Computer Science\APIS\. Keep public and private repository visibility unchanged. Sync only release_files.txt contents to the public repository. Exclude credentials, local QA renders, environments and Git internals. Use UTF-8 for writes.

The original DOI 10.5281/zenodo.23726433 identifies v1.0.0. Revision v1.1.0 is supplied as a public release and complete Supplemental Code S1 ZIP. Do not invent a new DOI or imply the original archive contains the revised manuscript.
