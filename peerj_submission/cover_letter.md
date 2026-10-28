# Cover letter

28 October 2026

The Editor
PeerJ Computer Science

Dear Editor,

We submit for consideration as a Research Article the manuscript entitled **"Locked paired validation of closed-loop therapy software: controller rankings are not robust to analysis choices"**.

The manuscript addresses a practical evaluation problem in scientific software. Closed-loop controllers for therapy are usually compared in a nominal simulation and judged on a single aggregate success rate. We show that this practice can reverse the ranking. We specify a locked paired-validation protocol, implement it as a reproducible analysis pipeline, and apply it to a 4,608-trial-per-controller atlas of four delayed-feedback controllers for coupled photothermal-photodynamic simulation.

The result is deliberately reported as a negative one. No controller improved overall success. The apparent benefit in a predicted-low stratum, +0.031 to +0.056, did not survive Holm correction across the 12 controller-by-stratum tests (adjusted p = 0.629 in every case). All four controllers were Pareto non-dominated, no weighting produced a majority winner, and the sign of the headline effect changed across 16 defensible analysis specifications. We regard this as the useful contribution: the protocol produces a bounded, reproducible answer even when the answer is that nothing wins.

We believe the work fits the journal's scope because the contribution is methodological and software-oriented rather than clinical. It concerns how computational experiments are designed, how uncertainty is propagated into comparative claims, and how an analysis artefact should be structured so that a reader can regenerate every number. The case study is a simulation model, but the protocol is model-agnostic and transfers to other closed-loop software.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. The author has no competing interests. All code and trial-level data are archived with a persistent DOI, and a single entry point regenerates every table and figure.

Thank you for considering the submission.

Yours sincerely,

Javad Alizargar
Department of Nursing
National Taipei University of Nursing and Health Sciences
No. 365, Mingde Rd, Beitou District, Taipei City 112, Taiwan
8javad@ntunhs.edu.tw
