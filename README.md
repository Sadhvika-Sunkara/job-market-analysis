# I Built a Data Pipeline to Analyze the Job Market I'm Trying to Enter

A 4-part series applying data cleaning, statistical analysis, machine learning, and an agent built on top of all three, to real job postings — as a Data Science student job-hunting for Data Analyst / Data Scientist / Data Engineer / AI Engineer roles. Each part is a real, working pipeline with real evaluation numbers, not toy-dataset filler.

| Part | Article | Question | Data |
|---|---|---|---|
| 1 | [What 95 Real Job...](https://medium.com/@sadhvikasunkara4/i-built-a-data-pipeline-to-analyze-the-job-market-im-trying-to-enter-part-1-what-95-real-job-f6d8754fc7d6) | What does a real, small sample of the market actually look like? | 95 Indeed postings, early 2016, VA/DC/MD |
| 2 | [Can a Job Title...](https://medium.com/@sadhvikasunkara4/i-built-a-data-pipeline-to-analyze-the-job-market-im-trying-to-enter-part-2-can-a-job-title-841a9585ec62) | Can a job title alone predict required skills? | Same 95 postings, title-only, LOOCV |
| 3 | [Building a Model...(The ML)](https://medium.com/@sadhvikasunkara4/building-a-model-to-predict-what-skills-you-actually-need-part-3-the-ml-9baf05bca00f) | At real scale, with full posting text, can a model tell what *role* a posting is for? | 7,523 postings, held-out test set + CV |
| 4 | [I Built an AI Agent...(The Agent)](part4_agent/article.md) | Can that classifier + retrieval actually tell someone what to learn — and where does it break? | Part 3's 7,523 postings + 9 live Sept-2026 postings |

## Repo structure

```
part1_dataset_cleaning/    raw scrape -> cleaned, role/skill-tagged dataset (95 rows)
part2_skill_prediction/    per-skill title-only classifiers, LOOCV (16 skills)
part3_role_classifier/     4-class role classifier on full description text (7,523 rows)
part4_agent/               retrieval + the Part 3 classifier, wrapped into a queryable agent
```

Each folder has its own README with the exact question it answers, how to run the code, and the real results.

## Why the dataset size jumps from 95 to 7,523

Part 2's honest conclusion was that title-only signal from 95 postings — one metro area, one quarter of 2016 — had hit its ceiling: only 4 of 16 modelable skills cleared a real F1 bar, and leave-one-out cross-validation was the only option at that sample size. Part 3 deliberately scales up to a much larger, more current, multi-source aggregation of postings to test two things at once: whether the 2016 findings generalize, and whether a harder task (predicting role from the full body text, not just a skill flag from the title) becomes tractable with enough data. Part 3's README and article walk through both results.

## Part 4: wrapping the classifier in an agent

`part4_agent/` retrains Part 3's exact pipeline on the combined corpus and exposes it two ways: ask a free-text skill question, or paste a posting/resume and get back the predicted role plus a gap analysis of what's missing. It also surfaces a genuine, mechanistically-explained limitation: the classifier misclassifies a real, live "Senior AI Engineer" posting (LangChain, RAG, LLM fine-tuning) as Data Engineer, because `min_df=5` in the TF-IDF vectorizer drops any term appearing in fewer than 5 of the corpus's 7,532 documents — and "langchain" appears in exactly 4. Full writeup and the confident-vs-not-confident comparison in `part4_agent/article.md`.
