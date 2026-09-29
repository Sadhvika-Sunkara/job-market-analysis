"""
Part 3 data prep — rebuilds cleaned_jobs.csv from the raw scraped source.

Source: natvalenz/DS_Jobs_PROJECT (public GitHub repo), a merged/deduped
scrape of ~16k Data Analyst / Data Scientist / Data Engineer / ML job
postings pulled from Indeed, LinkedIn, and Glassdoor. This mirrors the
raw input Part 1 of this series worked from.

This script:
  1. Loads the raw scrape.
  2. Drops rows with missing/near-empty descriptions and exact duplicate
     descriptions (repost noise).
  3. Derives a `role_category` label (data_analyst / data_scientist /
     data_engineer / ai_ml_engineer) from the job title using an explicit
     keyword taxonomy — titles that don't clearly map to one of the four
     roles (e.g. generic "Business Analyst", "Software Engineer") are
     dropped rather than force-labeled.
  4. Normalizes whitespace/encoding artifacts in the description text.
  5. Writes cleaned_jobs.csv.
"""
import re
import pandas as pd

RAW_PATH = "raw_fulldataset.csv"
OUT_PATH = "cleaned_jobs.csv"

MIN_DESC_CHARS = 150

ROLE_PATTERNS = [
    ("ai_ml_engineer", re.compile(
        r"machine learning|ml engineer|\bai engineer\b|artificial intelligence engineer|"
        r"deep learning|nlp engineer|computer vision engineer", re.I)),
    ("data_engineer", re.compile(
        r"data engineer|big data engineer|etl developer|etl engineer|analytics engineer",
        re.I)),
    ("data_scientist", re.compile(
        r"data scientist|data science", re.I)),
    ("data_analyst", re.compile(
        r"data analyst|business intelligence analyst|\bbi analyst\b|analytics analyst",
        re.I)),
]
# titles containing these should NOT count as data_scientist even if they
# match "data science" (management/leadership tracks, not IC roles)
DS_EXCLUDE = re.compile(r"manager|director|vp\b|vice president", re.I)


def label_role(title: str):
    if not isinstance(title, str):
        return None
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(title):
            if role == "data_scientist" and DS_EXCLUDE.search(title):
                continue
            return role
    return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\xa0", " ")
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"raw rows: {len(df)}")

    df = df.rename(columns={
        "Job Title": "job_title",
        "Job Description": "description_raw",
        "Company Name": "company",
        "Location": "location",
        "City": "city",
        "State": "state",
        "SalaryMin": "salary_min",
        "SalaryMax": "salary_max",
        "Rating": "company_rating",
    })

    df = df.dropna(subset=["description_raw", "job_title"])
    df["description"] = df["description_raw"].apply(clean_text)
    df = df[df["description"].str.len() >= MIN_DESC_CHARS]
    print(f"after empty/short-description filter: {len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=["description"], keep="first")
    print(f"dropped {before - len(df)} exact-duplicate descriptions -> {len(df)}")

    df["role_category"] = df["job_title"].apply(label_role)
    df = df.dropna(subset=["role_category"])
    print(f"after role-category labeling: {len(df)}")
    print(df["role_category"].value_counts())

    # seniority signal straight from title text (used later as a feature,
    # and as a sanity check against the source's own Junior/Senior flags)
    def seniority(title):
        tl = title.lower()
        if re.search(r"senior|sr\.|lead|principal|staff|head of", tl):
            return "senior"
        if re.search(r"junior|jr\.|intern|entry", tl):
            return "junior"
        return "mid"

    df["seniority"] = df["job_title"].apply(seniority)
    df["description_length"] = df["description"].str.len()

    keep_cols = [
        "job_title", "role_category", "seniority", "company", "location",
        "city", "state", "company_rating", "salary_min", "salary_max",
        "description_length", "description",
    ]
    out = df[keep_cols].reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: {out.shape}")


if __name__ == "__main__":
    main()
