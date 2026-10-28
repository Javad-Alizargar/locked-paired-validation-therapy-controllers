"""Generate a transparent audit ledger from saved public API evidence."""
import csv,json,re,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reference_audit'
s=(ROOT/'peerj_submission/main_manuscript.md').read_text(encoding='utf8')
body,tail=s.split('## References\n',1)
refs=[x.strip() for x in tail.split('## Additional information',1)[0].split('\n\n') if x.strip()]
old=json.loads((OUT/'bibliography.json').read_text(encoding='utf8'))
doi_to_idx={r['doi'].lower():r['index'] for r in old if r['doi']}
doi_to_idx.update({'10.1152/jappl.1948.1.2.93':42,'10.1002/andp.19123420503':43,'10.1364/ao.37.005271':44,'10.1016/j.jphotobiol.2016.09.031':45,'10.1080/00401706.1979.10489755':46})
roles={
'Abu-Ayyad':'MPC for magnetic nanoparticle heating; abstract supports the modality and controller claim.',
'Agarwal':'Uncertainty and interval reporting in RL benchmarks; official proceedings abstract and pagination checked.',
'Akhlaghi':'VVUQ in photoacoustic imaging; abstract describes verification and phantom validation.',
'Baker':'Survey of reproducibility concerns; cited as background, not evidence about this atlas.',
'Benjamini':'False-discovery-rate method; distinct from the Holm procedure actually used.',
'Bucharskaya':'Combined plasmonic photothermal/PDT context; review supports the general mechanism description.',
'Cai':'Review of physics-informed fluid modelling; scope restricted to related methods, not methods used here.',
'Cohen':'Normal-theory effect-size/power background; not validation of sign-flip or Holm family power.',
'Collin':'Guidelines for personalised computational models; abstract supports data-integration/validation context.',
'DiCiccio':'Bootstrap interval methods; does not establish validity under this dependent LHS design.',
'Efron':'Bootstrap background; original 1993 book retained, although Crossref dates the linked edition 1994.',
'Erdemir':'Context, data, limitations and review practices for credible biomedical modelling.',
'Gans':'Original spheroidal optical model provenance; does not validate this implementation or nanoparticle morphology.',
'Harris':'NumPy software citation; byline and article metadata checked.',
'Henderson':'RL experimental variance/reproducibility; abstract and publisher PDF support the analogy.',
'Holm':'Sequential family-wise adjustment; valid input p-values remain required.',
'Hunter':'Matplotlib software citation.',
'Karniadakis':'Physics-informed ML context; no claim that the present analysis uses this method.',
'Khan':'Graphene-based combined phototherapy review; isolated from claims about gold-nanorod kinetics.',
'Kim':'BPD macroscopic oxygen/photochemical parameters and 0.67 mM threshold; NCBI full text/Table 2 checked. Source uses mice and 690 nm; transfer to 808 nm explicitly unvalidated.',
'Kowalski':'Temperature feedback for electromagnetic phased-array hyperthermia; abstract supports claim.',
'Lad':'Thermal-dose feedback and PID/MPC in magnetic nanoparticle heating; abstract supports claim.',
'Maleki':'Gold-nanorod photothermal modelling and in-vitro heating; removed as support for ROS/oxygen-consumption mechanisms.',
'McKay':'Latin-hypercube sampling provenance; does not justify an ordinary iid bootstrap for its dependent rows.',
'McKinney':'pandas software citation and conference pagination.',
'Mulugeta':'Credibility and reproducibility for biomedical/neuroscience simulation.',
'Pathmanathan':'Applicability of validation evidence to a stated context of use; abstract supports claim.',
'Pennes':'Bioheat-equation provenance; not a source for the study-specific fixed tissue properties.',
'Pineau':'ML reproducibility programme; official JMLR abstract and bibliography checked.',
'Rakić':'Gold dielectric-function model provenance; implementation is described as Rakić-style rather than independently validated.',
'Rasheed':'General digital-twin modelling review; removed from the oncology-specific supporting group.',
'Rudy':'Data-driven PDE discovery; limited to related-method background.',
'Sandve':'Computational reproducibility practices; available full text checked.',
'Santiago':'VVUQ plan for ventricular/LVAD simulation; abstract supports this example.',
'Sebeke':'MR-HIFU MPC; modality correct, publisher PDF/PubMed pagination takes precedence over conflicting Crossref pages.',
'Sel':'Precision-medicine digital-twin VVUQ; abstract supports scope.',
'Sharma':'Temperature-feedback magnetic-hyperthermia device validation; abstract supports hardware example.',
'Simmons':'Analytic flexibility and false positives; adjacent-field context only.',
'Simonsohn':'Specification-curve analysis; present curve remains descriptive and does not implement the paper’s full joint-inference procedure.',
'Steegen':'Multiverse transparency; adjacent-field context only.',
'Steinman':'VVUQ cardiovascular editorial; framing reference, not an empirical validation study.',
'Verma':'NSGA-II/multi-objective review; supports general Pareto vocabulary, not this study’s chosen preference distribution.',
'Virtanen':'SciPy software citation; replaced 112 expanded Crossref entries with the official 34 named authors plus SciPy 1.0 Contributors.',
'Wilkinson':'FAIR data stewardship; genuine 53-author byline retained under the local all-author instruction.',
'Winkler':'Permutation/sign-flip assumptions; revised discussion specifies sign symmetry rather than exchangeability alone.',
'Wu':'Mechanism-based imaging digital twins for oncology; abstract supports scope.'}
rows=[]
for ref in refs:
 surname=ref.split()[0]
 year=re.search(r'\. ((?:19|20)\d{2})\.',ref).group(1)
 assert re.search(re.escape(surname)+r'[^\n]{0,75}'+year,body), ('Uncited reference',surname,year)
 m=re.search(r'DOI: (\S+)',ref)
 doi=m.group(1).rstrip('.') if m else ''
 idx=doi_to_idx.get(doi.lower()) if doi else {'Agarwal':2,'Cohen':8,'Holm':15,'Pineau':25}[surname]
 apis=[];pmid='';pmcid='';oa='';retracted='not available'
 for name in ['crossref','openalex','europepmc']:
  f=OUT/'api_records'/f'{idx:02d}_{name}.json'
  if not f.exists(): continue
  d=json.loads(f.read_text(encoding='utf8'))['response']
  if name=='crossref' and d.get('status')=='ok': apis.append('Crossref')
  if name=='openalex' and d.get('id'):
   apis.append('OpenAlex');oa=d['id'];retracted=str(d.get('is_retracted'))
   assert d.get('doi','').lower()=='https://doi.org/'+doi.lower(),('DOI mismatch',ref)
  if name=='europepmc':
   rr=d.get('resultList',{}).get('result',[])
   for r in rr:
    if r.get('source')=='MED':
     assert r.get('doi','').lower()==doi.lower(),('PubMed DOI mismatch',ref)
     pmid=r.get('id','');pmcid=r.get('pmcid','');apis.append('Europe PMC/PubMed')
 if not doi: apis=['OpenAlex/Crossref title search; primary publisher where available']
 rows.append({'reference':surname+' '+year,'doi':doi,'pmid':pmid,'pmcid':pmcid,'openalex_id':oa,'metadata_evidence':'; '.join(apis),'openalex_retracted':retracted,'claim_review':roles[surname]})
with (OUT/'reference_verification.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
figs=list(dict.fromkeys(re.findall(r'Fig\. (\d+)',body)))
tabs=list(dict.fromkeys(re.findall(r'Table (\d+)',body)))
supp=list(dict.fromkeys(re.findall(r'Table (S\d+)',body)))
assert figs==['1','2','3','4','5'],figs
assert tabs==['1','2','3'],tabs
assert supp==['S1','S2','S3'],supp
abstract=body.split('## Abstract\n',1)[1].split('## Introduction',1)[0]
plain=re.sub(r'\*\*','',abstract).strip()
assert len(plain)<3000 and len(plain.split())<500
ncbi=ET.parse(OUT/'ncbi_pubmed_records.xml').getroot()
ncbi_count=len(ncbi.findall('PubmedArticle'))
report='''# Reference and display-item audit — 28 October 2026

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
'''
report+=f'\nAPI summary: {sum(bool(r["doi"]) for r in rows)} DOI entries; {sum(bool(r["pmid"]) for r in rows)} PubMed matches through Europe PMC; {ncbi_count} direct NCBI records fetched. Abstract: {len(plain.split())} words, {len(plain)} characters.\n'
(ROOT/'REFERENCE_AUDIT.md').write_text(report,encoding='utf8')
(OUT/'checks.json').write_text(json.dumps({'references':len(refs),'doi_references':sum(bool(r['doi']) for r in rows),'figures_first_citation':figs,'main_tables_first_citation':tabs,'supplementary_tables_first_citation':supp,'abstract_words':len(plain.split()),'abstract_characters':len(plain),'ncbi_records':ncbi_count},indent=2),encoding='utf8')
print(report[-450:])
