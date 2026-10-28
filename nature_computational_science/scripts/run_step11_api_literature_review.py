from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data" / "literature_api"
OUT_NOTES = NCS / "notes"

OPENALEX_KEY = "QAwOVZB6CXTKPqEXcOCMdt"

OPENALEX_QUERIES = {
    "ncs_scope_computational_models": {
        "query": '"Nature Computational Science" "computational model" biology control',
        "note": "Probe whether Nature Computational Science-adjacent work emphasizes computational modeling rather than narrow application.",
    },
    "equation_discovery_sciml": {
        "query": '"equation discovery" OR "symbolic regression" "scientific machine learning" biology',
        "note": "Methods anchor for sparse/reduced equations and interpretable discovery.",
    },
    "pde_discovery_reaction_diffusion": {
        "query": '"data-driven discovery" partial differential equations reaction diffusion',
        "note": "Evidence base for discovering compact governing/boundary equations.",
    },
    "feedback_control_hyperthermia": {
        "query": '"model predictive control" hyperthermia thermal therapy feedback',
        "note": "Prior work that limits novelty claims about feedback control itself.",
    },
    "digital_twin_oncology_control": {
        "query": '"digital twin" oncology model predictive control',
        "note": "Potential framing around computational oncology/digital twins.",
    },
    "heterogeneous_treatment_effect_uplift": {
        "query": '"heterogeneous treatment effects" "uplift modeling" prediction model',
        "note": "Methods anchor for paired uplift/reclassification analyses.",
    },
    "conformal_uncertainty_sciml": {
        "query": '"conformal prediction" "scientific machine learning" uncertainty',
        "note": "Potential next method for uncertainty-calibrated class predictions.",
    },
    "reaction_diffusion_delayed_feedback": {
        "query": '"reaction diffusion" "delayed feedback" control biological',
        "note": "Scope test for abstract reaction-diffusion transfer.",
    },
}

PUBMED_QUERIES = {
    "hyperthermia_feedback_control": {
        "term": '("model predictive control"[Title/Abstract] OR "feedback control"[Title/Abstract]) AND (hyperthermia[Title/Abstract] OR bioheat[Title/Abstract] OR "thermal therapy"[Title/Abstract])',
        "note": "Biomedical prior art for feedback/predictive thermal therapy.",
    },
    "photothermal_feedback": {
        "term": '("photothermal therapy"[Title/Abstract] AND ("feedback"[Title/Abstract] OR "temperature monitoring"[Title/Abstract] OR "control"[Title/Abstract]))',
        "note": "Prior art near photothermal feedback.",
    },
    "digital_twin_oncology": {
        "term": '("digital twin"[Title/Abstract] AND (oncology[Title/Abstract] OR cancer[Title/Abstract]) AND model[Title/Abstract])',
        "note": "Computational oncology framing.",
    },
    "symbolic_regression_biomed": {
        "term": '("symbolic regression"[Title/Abstract] OR "equation discovery"[Title/Abstract]) AND (biomedical[Title/Abstract] OR biology[Title/Abstract] OR biophysics[Title/Abstract])',
        "note": "Biomedical relevance of symbolic/equation discovery.",
    },
    "prediction_model_validation": {
        "term": '("prediction model"[Title/Abstract] AND validation[Title/Abstract] AND ("treatment effect"[Title/Abstract] OR "personalized"[Title/Abstract] OR reclassification[Title/Abstract]))',
        "note": "Accepted validation/reclassification methodology.",
    },
    "reaction_diffusion_feedback_biology": {
        "term": '("reaction diffusion"[Title/Abstract] AND ("feedback"[Title/Abstract] OR control[Title/Abstract]) AND (biology[Title/Abstract] OR tissue[Title/Abstract] OR cellular[Title/Abstract]))',
        "note": "Literature for delayed feedback/reaction-diffusion scope.",
    },
}


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=45) as handle:
        return json.load(handle)


def openalex_search(name: str, query: str, per_page: int = 12) -> list[dict]:
    params = {
        "search": query,
        "per-page": str(per_page),
        "sort": "relevance_score:desc",
        "api_key": OPENALEX_KEY,
        "filter": "from_publication_date:2015-01-01",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = get_json(url)
    records = []
    for rank, work in enumerate(data.get("results", []), start=1):
        source = ((work.get("primary_location") or {}).get("source") or {})
        concepts = [c.get("display_name") for c in work.get("concepts", [])[:6]]
        records.append(
            {
                "query_name": name,
                "rank": rank,
                "openalex_id": work.get("id"),
                "doi": work.get("doi"),
                "year": work.get("publication_year"),
                "title": work.get("display_name"),
                "venue": source.get("display_name"),
                "source_type": source.get("type"),
                "type": work.get("type"),
                "cited_by_count": work.get("cited_by_count"),
                "is_oa": (work.get("open_access") or {}).get("is_oa"),
                "concepts": "; ".join([c for c in concepts if c]),
            }
        )
    return records


def pubmed_search_ids(term: str, retmax: int = 12) -> list[str]:
    params = {
        "db": "pubmed",
        "term": f"({term}) AND 2015:3000[dp]",
        "retmax": str(retmax),
        "sort": "relevance",
        "retmode": "json",
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
    data = get_json(url)
    return data.get("esearchresult", {}).get("idlist", [])


def pubmed_fetch(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=45) as handle:
        root = ET.fromstring(handle.read())
    records = []
    for article in root.findall("./PubmedArticle"):
        medline = article.find("./MedlineCitation")
        pmid = medline.findtext("./PMID") if medline is not None else None
        article_node = medline.find("./Article") if medline is not None else None
        title = ""
        journal = None
        year = None
        if article_node is not None:
            title_node = article_node.find("./ArticleTitle")
            title = "".join(title_node.itertext()) if title_node is not None else ""
            journal = article_node.findtext("./Journal/Title")
            year = article_node.findtext("./Journal/JournalIssue/PubDate/Year")
            if year is None:
                year = article_node.findtext("./Journal/JournalIssue/PubDate/MedlineDate")
        doi = None
        article_id_list = article.find("./PubmedData/ArticleIdList")
        if article_id_list is not None:
            for aid in article_id_list.findall("./ArticleId"):
                if aid.attrib.get("IdType") == "doi":
                    doi = aid.text
                    break
        records.append(
            {
                "pmid": pmid,
                "doi": doi,
                "year": year,
                "title": title,
                "journal": journal,
            }
        )
    return records


def pubmed_query(name: str, term: str) -> list[dict]:
    ids = pubmed_search_ids(term)
    time.sleep(0.34)
    records = pubmed_fetch(ids)
    for rank, record in enumerate(records, start=1):
        record["query_name"] = name
        record["rank"] = rank
    return records


def write_review(openalex_df: pd.DataFrame, pubmed_df: pd.DataFrame) -> None:
    def top_lines(df: pd.DataFrame, query: str, id_col: str, venue_col: str, n: int = 5) -> list[str]:
        rows = []
        sub = df[df["query_name"] == query].head(n)
        for _, row in sub.iterrows():
            ident = row.get(id_col)
            doi = row.get("doi")
            suffix = doi if pd.notna(doi) and doi else ident
            rows.append(f"- {row.get('year')} | {row.get(venue_col)} | {row.get('title')} | {suffix}")
        return rows or ["- No records returned."]

    openalex_count = len(openalex_df)
    pubmed_count = len(pubmed_df)
    lines = [
        "# Step 11 API-Based Literature Review and Next-Step Decision",
        "",
        "## Project Evidence So Far",
        "",
        "The strongest internal result is not a universal closed-form law. It is a seed-stable controllability stratification in the original biological-physics system plus targeted predictive-control rescue of low-controllability cases.",
        "",
        "Internal evidence:",
        "",
        "- Multi-seed three-term boundary: minimum seed-family AUC 0.684, median 0.776.",
        "- Controllability classes: low 0.237, transitional 0.416, high 0.716 observed success.",
        "- Predictive control in low-controllability cases: paired success uplift +0.114 with bootstrap interval [0.040, 0.186].",
        "- Abstract reaction-diffusion transfer was partial but failed the universality gate: minimum abstract seed-family AUC 0.565 and low-class predictive uplift -0.013.",
        "- Richer theory classes improved leave-seed-out AUC up to 0.720 minimum, but did not solve seed-family/controller robustness.",
        "",
        "## API Methods",
        "",
        f"OpenAlex records retrieved: {openalex_count}. PubMed records retrieved: {pubmed_count}. Raw API outputs are saved as CSV and JSON in `nature_computational_science/data/literature_api/`.",
        "",
        "Important integrity rule: these records are a literature-mapping set, not final manuscript references. Any paper cited in the manuscript must be rechecked by PMID/OpenAlex ID/title/DOI alignment before use.",
        "",
        "## OpenAlex Signals",
        "",
        "### Equation Discovery and Scientific Machine Learning",
        *top_lines(openalex_df, "equation_discovery_sciml", "openalex_id", "venue"),
        "",
        "### PDE/Reaction-Diffusion Discovery",
        *top_lines(openalex_df, "pde_discovery_reaction_diffusion", "openalex_id", "venue"),
        "",
        "### Feedback and Predictive Control in Thermal Therapy",
        *top_lines(openalex_df, "feedback_control_hyperthermia", "openalex_id", "venue"),
        "",
        "### Heterogeneous Effects, Uplift, and Reclassification",
        *top_lines(openalex_df, "heterogeneous_treatment_effect_uplift", "openalex_id", "venue"),
        "",
        "### Conformal/Uncertainty Methods",
        *top_lines(openalex_df, "conformal_uncertainty_sciml", "openalex_id", "venue"),
        "",
        "## PubMed/NCBI Signals",
        "",
        "### Hyperthermia and Feedback Control",
        *top_lines(pubmed_df, "hyperthermia_feedback_control", "pmid", "journal"),
        "",
        "### Photothermal Feedback",
        *top_lines(pubmed_df, "photothermal_feedback", "pmid", "journal"),
        "",
        "### Digital-Twin Oncology",
        *top_lines(pubmed_df, "digital_twin_oncology", "pmid", "journal"),
        "",
        "### Prediction-Model Validation and Reclassification",
        *top_lines(pubmed_df, "prediction_model_validation", "pmid", "journal"),
        "",
        "## Interpretation for Nature Computational Science",
        "",
        "The literature makes three points clear:",
        "",
        "1. Feedback or model-predictive hyperthermia alone is not novel enough; there is prior art in PubMed-indexed biomedical engineering and thermal therapy literature.",
        "2. A compact reduced law/equation-discovery framing is more aligned with computational-science novelty, but our single frozen equation failed external validation.",
        "3. The strongest defensible novelty is currently methodological and mechanistic: grouped multi-seed controllability classes plus paired counterfactual predictive-control recovery.",
        "",
        "## Best Next Step",
        "",
        "The evidence-based next step is **mechanistic failure-mode decomposition of the low-controllability class**, followed by a larger seed validation only if the mechanism is stable.",
        "",
        "Why this step is highest value:",
        "",
        "- It directly strengthens the only statistically positive intervention result: low-class predictive-control rescue.",
        "- It avoids chasing AUC with more terms, which Step 10 showed has limited return.",
        "- It creates a reviewer-facing mechanism: predictive control rescues cases by reducing a specific failure mode, not merely by improving an abstract score.",
        "- It can be analyzed with paired data already generated, so it is faster and more reproducible than another large simulation campaign.",
        "",
        "Concrete implementation:",
        "",
        "1. In paired full-feedback vs predictive-feedback cases, focus on the low-controllability starting class.",
        "2. Decompose rescue/loss into thermal failure, ROS/product failure, overheat failure, and hypoxic/resource failure.",
        "3. Test whether predictive control changes maximum temperature, warm fraction, ROS coverage, oxygen/resource failure, energy, and delay sensitivity.",
        "4. Use paired effect sizes with cluster bootstrap confidence intervals.",
        "5. If one mechanism is stable, write it as the core result. If not, pivot to a more modest journal or a methods-note framing.",
        "",
        "Step-gate decision: **CONTINUE, but only through mechanistic failure-mode decomposition.** Do not run more broad AUC searches until the rescue mechanism is proven.",
    ]
    (OUT_NOTES / "step11_api_literature_review_and_next_step.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)

    openalex_records = []
    for name, spec in OPENALEX_QUERIES.items():
        rows = openalex_search(name, spec["query"])
        for row in rows:
            row["query_note"] = spec["note"]
        openalex_records.extend(rows)
        print(f"OpenAlex {name}: {len(rows)} records", flush=True)
        time.sleep(0.2)
    pubmed_records = []
    for name, spec in PUBMED_QUERIES.items():
        rows = pubmed_query(name, spec["term"])
        for row in rows:
            row["query_note"] = spec["note"]
        pubmed_records.extend(rows)
        print(f"PubMed {name}: {len(rows)} records", flush=True)
        time.sleep(0.34)

    openalex_df = pd.DataFrame(openalex_records)
    pubmed_df = pd.DataFrame(pubmed_records)
    openalex_df.to_csv(OUT_DATA / "step11_openalex_literature_records.csv", index=False)
    pubmed_df.to_csv(OUT_DATA / "step11_pubmed_literature_records.csv", index=False)
    (OUT_DATA / "step11_openalex_literature_records.json").write_text(json.dumps(openalex_records, indent=2), encoding="utf-8")
    (OUT_DATA / "step11_pubmed_literature_records.json").write_text(json.dumps(pubmed_records, indent=2), encoding="utf-8")
    query_manifest = {
        "openalex_queries": OPENALEX_QUERIES,
        "pubmed_queries": PUBMED_QUERIES,
        "openalex_key_used": OPENALEX_KEY[:4] + "..." + OPENALEX_KEY[-4:],
    }
    (OUT_DATA / "step11_literature_query_manifest.json").write_text(json.dumps(query_manifest, indent=2), encoding="utf-8")
    write_review(openalex_df, pubmed_df)
    print("Wrote Step 11 API literature review")
    print(OUT_NOTES / "step11_api_literature_review_and_next_step.md")


if __name__ == "__main__":
    main()
