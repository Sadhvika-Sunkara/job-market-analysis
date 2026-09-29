# I Built a Data Pipeline to Analyze the Job Market I'm Trying to Enter

A 3-part (so far) series applying data cleaning, statistical analysis, and machine learning to real job postings — as a Data Science student job-hunting for Data Analyst / Data Scientist / Data Engineer / AI Engineer roles. Each part is a real, working pipeline with real evaluation numbers, not toy-dataset filler.

| Part | Article | Question | Data |
|---|---|---|---|
| 1 | [What 95 Real Job...](https://medium.com/@sadhvikasunkara4/i-built-a-data-pipeline-to-analyze-the-job-market-im-trying-to-enter-part-1-what-95-real-job-f6d8754fc7d6) | What does a real, small sample of the market actually look like? | 95 Indeed postings, early 2016, VA/DC/MD |
| 2 | [Can a Job Title...](https://medium.com/@sadhvikasunkara4/i-built-a-data-pipeline-to-analyze-the-job-market-im-trying-to-enter-part-2-can-a-job-title-841a9585ec62) | Can a job title alone predict required skills? | Same 95 postings, title-only, LOOCV |
| 3 | [Building a Model...(The ML)](https://medium.com/@sadhvikasunkara4/building-a-model-to-predict-what-skills-you-actually-need-part-3-the-ml-9baf05bca00f) | At real scale, with full posting text, can a model tell what *role* a posting is for? | 7,523 postings, held-out test set + CV |

## Repo structure

```
part1_dataset_cleaning/    raw scrape -> cleaned, role/skill-tagged dataset (95 rows)
part2_skill_prediction/    per-skill title-only classifiers, LOOCV (16 skills)
part3_role_classifier/     4-class role classifier on full description text (7,523 rows)
```

Each folder has its own README with the exact question it answers, how to run the code, and the real results.

## Why the dataset size jumps from 95 to 7,523

Part 2's honest conclusion was that title-only signal from 95 postings — one metro area, one quarter of 2016 — had hit its ceiling: only 4 of 16 modelable skills cleared a real F1 bar, and leave-one-out cross-validation was the only option at that sample size. Part 3 deliberately scales up to a much larger, more current, multi-source aggregation of postings to test two things at once: whether the 2016 findings generalize, and whether a harder task (predicting role from the full body text, not just a skill flag from the title) becomes tractable with enough data. Part 3's README and article walk through both results.

## What's next (Part 4)

Wrapping the Part 3 classifier in an agent: read a posting, classify its role, and tell you what to go learn.
