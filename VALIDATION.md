# Revision validation

28 October 2026. Numerical reproduction, code regression checks and structural/visual document checks were completed before publishing revision v1.1.0.

- The revised analysis reproduced all 15 CSV/JSON outputs byte-for-byte on a second final run.
- A separate public checkout reproduced the numerical and environment outputs from the allowlisted package.
- Regression checks cover 96 shared-parent blocks, all family/scenario variants, seed allocation, a known Holm example, the joint 20-test family, low-band corrections and numeric figure citation order.
- All 32 explicitly formatted manuscript estimate/interval triples matched result CSVs.
- All 146 manuscript-source paragraphs matched the generated Word body.
- All six standalone Word tables matched their Markdown cells; upload copies matched generated source files.
- The manuscript and reviewer copy were exported using Microsoft Word 16 and inspected as rasterised PDF pages. The bundled LibreOffice renderer was attempted but LibreOffice was unavailable on this Windows host.
- Manuscript: 24 pages. Reviewer copy: 30 pages. Each of the six tables: one page. Cover letter, declarations and figure captions: one page each after ancillary spacing adjustments.
- Five figures meet the 900–3,000 pixel limits and approximately 300 dpi; multipart labels, primary-specification identity and uncertainty displays were checked.
- Abstract: 302 words, 2,284 plain-text characters. Author identity, US Letter dimensions, one-inch margins, 12-point Times body text, 1.5 manuscript spacing and continuous manuscript line numbering are retained. The main DOCX embeds no figures or tables.
- Original paired-data SHA-256 remains D8D59D86691039F30336ED4FF00A72643E3BD9475BB16856AF5DE67986B1174C.
- The complete revised supplementary package is built from release_files.txt and checked against its source contents. Local QA renders and credentials are excluded.

These checks establish reproducibility and implementation/document consistency. They do not establish clinical validity, exact bootstrap coverage under Latin-hypercube sampling, sign-symmetry assumptions, prospective score validity, or the author's final approval. Those limits are stated in the revised manuscript. The original v1.0.0 DOI remains identifiable; this revision does not invent a new DOI.
