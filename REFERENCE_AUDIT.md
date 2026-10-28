# Reference and display-item audit — 28 October 2026

The revised manuscript has 46 references (41 existing plus five model/sampling sources). All 42 DOI-bearing entries were looked up through public Crossref and OpenAlex APIs, with DOI-specific Europe PMC/PubMed queries and a direct NCBI E-utilities fetch. Four DOI-free entries were searched separately; books and older statistics references were not forced into spurious DOI matches. The CSV is the per-reference evidence and claim-support ledger; raw responses retain retrieval dates and request URLs.

## Corrected findings

1. **Unsupported mechanistic citation:** Maleki et al. describes photothermal heating, not the claimed ROS/oxygen kinetics. The introduction now gives it a separate heating claim and uses the combined-therapy review for the ROS context.
2. **Missing primary model citations:** Added Pennes (bioheat), Gans (spheroidal optics), Rakić et al. (dielectric model) and Kim et al. (macroscopic singlet-oxygen dosimetry). Fixed tissue properties are explicitly testbed assumptions rather than attributed patient measurements.
3. **Parameter transfer was not justified:** Kim et al.’s BPD threshold/parameters came from a mouse experiment at 690 nm. Methods and Limitations now identify transfer to 808 nm as unvalidated. This requires new physical validation for a therapeutic claim; adding a citation alone cannot provide it.
4. **Missing sampling citation:** Added McKay, Beckman & Conover for Latin-hypercube sampling. The existing warning that the empirical bootstrap does not reproduce its dependent sampling law remains.
5. **Excessively expanded SciPy entry:** Crossref expands consortium membership into 112 entries. Replaced that roster with SciPy’s official citation: 34 named authors and SciPy 1.0 Contributors. NumPy’s 26 and FAIR’s 53 named authors remain because these are genuine bylines, consistent with the supplied all-author guidance.
6. **Incorrect page range:** Sebeke is 1040–1050 according to PubMed and the publisher-formatted PDF, despite Crossref’s 1039–1049. Corrected using the primary PDF. Retained MR-HIFU as the modality.
7. **Incomplete conference/link metadata:** Added Agarwal’s 29304–29320 pages and official proceedings URL; Henderson’s 3207–3214 pages from the publisher PDF; JMLR’s direct article link. Preserved verified DOI strings and corrected available name diacritics/suffixes.
8. **Overbroad source scope:** Separated Rasheed’s general digital-twin review from oncology-specific sources. Related physics-informed ML references no longer imply that those methods are used or establish tractability here.
9. **Inference wording:** Discussion now names sign symmetry; title-page reporting now calls the procedures exploratory sign flips and approximate bootstrap rather than randomisation inference.

## Citation and display-item checks

Every listed reference is cited. Reference entries are alphabetical. Literature citations use author–year; grouped citations are chronological. Five figures are first cited in order 1, 2, 3, 4, 5; main tables in order 1, 2, 3; supplementary tables in order S1, S2, S3. All have separate matching output files and legends/notes. Figure 3 has A–D, Figure 4 A–C and Figure 5 A–D; captions name the correct panels. Results and caption values were compared with the generated tables and CSVs. No display items were renumbered.

## Journal requirements and limits

Checked the supplied PeerJ author instructions and policies (dated 28 October 2026), including full/clear/consistent references, all-author bibliography guidance, alphabetical listing, chronological grouped citations, numerical figure order, separate uploads, US Letter/one-inch margins/12 pt Times/line numbers and abstract limits. Live journal retrieval returned an access-block page; the local transcription is the requirements basis, not a claim of independently successful live-policy retrieval. Official SciPy guidance and original publisher records were independently accessed. Main submission DOCX keeps figures/tables separate; reviewer copy embeds them. AI disclosure and figure legends identify the actual tool/version. Code/data remain available as a full supplement and public revision release; the v1.0.0 DOI is not relabelled.

Metadata identity checks are not equivalent to full scientific validation. Claim review used available primary abstracts/full texts and official publisher records; the ledger states the limited role of contextual references. No cited DOI was identified as retracted by OpenAlex’s retrieved flags; that is not an exhaustive retraction-screening guarantee. The original 1993 Efron–Tibshirani book citation is retained: Crossref dates its linked edition 1994, an edition/date discrepancy rather than evidence that the original book is fictitious. No unsupported DOI was added to Cohen, Holm, Pineau or Agarwal.

The important remaining scientific limitation is model/parameter validation, including 690-to-808 nm transfer. This revision makes it explicit and confines the conclusions to a software-evaluation case study; it does not establish clinical validity.

## Evidence locations

- `reference_verification.csv`: each reference, DOI, PubMed/PMC/OpenAlex IDs and claim-support assessment.
- `api_records/`: dated public API JSON responses, including searches for DOI-free entries.
- `ncbi_pubmed_records.xml` / `ncbi_request.json`: direct NCBI API records and query.
- Available Europe PMC XML and NCBI BioC full text: primary claim checks; access errors are retained, not represented as full-text success.
- Official SciPy citation: https://scipy.org/citing-scipy/
- Sebeke publisher PDF: https://pure.tue.nl/ws/files/138750741/Model_predictive_control_for_MR_HIFU_mediated_uniform_hyperthermia.pdf
- Kim full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC5079817/

API summary: 42 DOI entries; 29 PubMed matches through Europe PMC; 29 direct NCBI records fetched. Abstract: 302 words, 2284 characters.
