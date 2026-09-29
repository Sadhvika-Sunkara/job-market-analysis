# Part 1 — Dataset Cleaning

Companion code for ["I Built a Data Pipeline to Analyze the Job Market I'm Trying to Enter — Part 1: What 95 Real Job..."](https://medium.com/@sadhvikasunkara4/i-built-a-data-pipeline-to-analyze-the-job-market-im-trying-to-enter-part-1-what-95-real-job-f6d8754fc7d6)

## Source

`raw_indeed_job_listings.csv` — 100 real Indeed job postings scraped in early 2016 (post dates in this file run early-to-late April 2016), concentrated in the VA/DC/MD corridor. Originally scraped by [stevendoll/kaggle-jobs](https://github.com/stevendoll/kaggle-jobs) (`indeed-job-listings.csv`).

## What `build_dataset.py` does

1. Deduplicates on `(job_title, company, city)` — 100 raw rows → 95 clean postings.
2. Parses the raw `post_date` string into a real datetime.
3. Assigns a `role_category` from the job title via keyword rules: `"scientist"`/`"data science"` → Data Scientist, `"analyst"`/`"analytics"` → Data Analyst, `"engineer"` + `"data"` → Data Engineer (kept, but only a handful of rows — too small to analyze on its own), everything else → Other.
4. Tags each posting against a ~30-term skill dictionary (Python, SQL, Machine Learning, Statistics, Tableau, Hadoop, Spark, Communication, and so on), scanning the job title, summary, and full posting body.

Output: `cleaned_jobs.csv` (95 rows).

## Run it

```
python build_dataset.py
```

## What it found

Role split: **Data Scientist 66%, Data Analyst 17%, Other 13%, Data Engineer ~4%** (too few Data Engineer postings to draw conclusions from).

Even at N=95, Data Scientist and Data Analyst postings asked for meaningfully different skills:

| Skill | Data Scientist | Data Analyst |
|---|---|---|
| Python | 57% | 0% |
| Machine Learning | 57% | 6% |
| Statistics | 79% | 75% |
| Excel | 6% | 44% |
| Business Intelligence | 14% | 38% |

That gap — and whether it holds up with more data — is what Part 2 and Part 3 build on.
