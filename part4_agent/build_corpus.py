"""
build_corpus.py -- Part 4

Combines Part 3's real 7,523-posting corpus with a small batch of live
postings fetched directly from company career pages in September 2026.

Why bother, when Part 3's corpus is already large and real: it isn't
current. Part 3's own article notes the corpus "doesn't carry per-posting
dates" -- and checking it directly shows why that matters for Part 4
specifically. A word-boundary search for "llm" across all 7,523
descriptions returns zero real hits (the non-boundary count in Part 3's
skill taxonomy is inflated by words like "fulfillment"), and terms like
"langchain", "vector database", and "prompt engineering" appear zero
times anywhere in the corpus. The AI Engineer title -- and the retrieval
/ agent tooling it implies -- is simply newer than every data source this
series has used so far. That's a real finding, not a gap to paper over,
which is why this script appends 9 live postings instead of silently
treating the Part 3 corpus as current.

Output: combined_corpus.csv, same schema as Part 3's cleaned_jobs.csv
plus a `source` column.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PART3_CSV = REPO_ROOT / "part3_role_classifier" / "cleaned_jobs.csv"
FRESH_JSON = HERE / "fresh_2026_postings.json"
OUT_CSV = HERE / "combined_corpus.csv"

CATEGORY_TO_TAXONOMY = {
    "AI Engineer": "ai_ml_engineer",
    "Machine Learning Engineer": "ai_ml_engineer",
    "Data Engineer": "data_engineer",
    "Data Analyst": "data_analyst",
    "Data Scientist": "data_scientist",
}


def main():
    part3 = pd.read_csv(PART3_CSV)
    part3["source"] = "part3_corpus"

    fresh_raw = json.loads(FRESH_JSON.read_text())
    fresh_rows = []
    for r in fresh_raw:
        fresh_rows.append({
            "job_title": r["job_title"],
            "role_category": CATEGORY_TO_TAXONOMY[r["category"]],
            "seniority": None,
            "company": r["company"],
            "location": r["location"],
            "city": None,
            "state": None,
            "company_rating": None,
            "salary_min": None,
            "salary_max": None,
            "description_length": len(r["description_text"]),
            "description": r["description_text"],
            "source": "live_careers_page_2026",
        })
    fresh = pd.DataFrame(fresh_rows)

    combined = pd.concat([part3, fresh], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(combined)} rows to {OUT_CSV}")
    print(combined["role_category"].value_counts())
    print(combined["source"].value_counts())


if __name__ == "__main__":
    main()
