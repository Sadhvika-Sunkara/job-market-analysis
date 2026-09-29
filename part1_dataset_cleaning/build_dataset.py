"""
Part 1 — reconstruction of the cleaning + role/skill tagging pipeline.

Source: stevendoll/kaggle-jobs (public GitHub repo), file
indeed-job-listings.csv — 100 raw Indeed postings scraped in early 2016
(post dates run Apr 8-27 2016 in this file; Part 1 describes a broader
~3-month scrape window). This is the exact source Part 1 used.

Steps (matching what Part 1 describes):
  1. Deduplicate on (job_title, company, city).
  2. Parse the raw post_date string into a real datetime.
  3. Assign role_category from the job title via keyword rules:
     "scientist"/"data science" -> Data Scientist
     "analyst"/"analytics"      -> Data Analyst
     "engineer" + "data"        -> Data Engineer (kept, but tiny -- 2 rows)
     everything else            -> Other
  4. Tag each posting against a ~30-term skill dictionary, scanning
     job_title + summary + body (case-insensitive, word-boundary regex).
"""
import re
import pandas as pd

RAW_PATH = "raw_indeed_job_listings.csv"
OUT_PATH = "cleaned_jobs.csv"

SKILLS = {
    "python": r"\bpython\b",
    "r_lang": r"\bR\b",
    "sql": r"\bsql\b",
    "java": r"\bjava\b",
    "scala": r"\bscala\b",
    "excel": r"\bexcel\b",
    "tableau": r"\btableau\b",
    "power_bi": r"power\s?bi",
    "sas": r"\bsas\b",
    "spss": r"\bspss\b",
    "hadoop": r"\bhadoop\b",
    "spark": r"\bspark\b",
    "aws": r"\baws\b",
    "azure": r"\bazure\b",
    "machine_learning": r"machine learning",
    "deep_learning": r"deep learning",
    "statistics": r"statistic",
    "data_mining": r"data mining",
    "predictive_modeling": r"predictive model",
    "data_visualization": r"data visuali[sz]ation",
    "big_data": r"big data",
    "nlp": r"natural language processing|\bnlp\b",
    "communication": r"communication",
    "presentation": r"presentation",
    "business_intelligence": r"business intelligence|\bbi\b",
    "ab_testing": r"a/b test",
    "regression": r"regression",
    "nosql": r"nosql",
    "git": r"\bgit\b",
    "agile": r"\bagile\b",
}


def label_role(title: str) -> str:
    tl = title.lower()
    if "engineer" in tl and "data" in tl:
        return "Data Engineer"
    if "scientist" in tl or "data science" in tl:
        return "Data Scientist"
    if "analyst" in tl or "analytics" in tl:
        return "Data Analyst"
    return "Other"


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"raw rows: {len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=["job_title", "company", "city"], keep="first")
    print(f"dropped {before - len(df)} duplicates (same title+company+city) -> {len(df)}")

    df["post_date_parsed"] = pd.to_datetime(df["post_date"], errors="coerce", utc=True)
    df["role_category"] = df["job_title"].apply(label_role)
    print(df["role_category"].value_counts())
    print((df["role_category"].value_counts(normalize=True) * 100).round(1))

    combined_text = (
        df["job_title"].fillna("") + " " + df["summary"].fillna("") + " " + df["body"].fillna("")
    ).str.lower()

    for skill, pattern in SKILLS.items():
        df[f"skill_{skill}"] = combined_text.str.contains(pattern, regex=True, case=False).astype(int)

    keep_cols = [
        "job_title", "role_category", "company", "city", "state", "location",
        "post_date_parsed", "views", "summary", "body",
    ] + [f"skill_{s}" for s in SKILLS]
    out = df[keep_cols].reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}: {out.shape}")

    # sanity check: skill frequency + DS vs DA split, to compare against Part 1's published numbers
    print("\nOverall skill frequency (top 10):")
    freqs = out[[f"skill_{s}" for s in SKILLS]].mean().sort_values(ascending=False) * 100
    print(freqs.head(10).round(1))

    ds = out[out["role_category"] == "Data Scientist"]
    da = out[out["role_category"] == "Data Analyst"]
    print(f"\nData Scientist n={len(ds)}, Data Analyst n={len(da)}")
    print("\nSkill  |  DS%  |  DA%")
    for s in ["python", "machine_learning", "excel", "business_intelligence", "statistics", "sql"]:
        col = f"skill_{s}"
        print(f"{s:25s} {ds[col].mean()*100:5.1f}  {da[col].mean()*100:5.1f}")


if __name__ == "__main__":
    main()
