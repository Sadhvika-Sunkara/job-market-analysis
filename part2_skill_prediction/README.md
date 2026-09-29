# Part 2 — Can a Job Title Predict Required Skills?

Companion code for ["I Built a Data Pipeline to Analyze the Job Market I'm Trying to Enter — Part 2: Can a Job Title..."](https://medium.com/@sadhvikasunkara4/i-built-a-data-pipeline-to-analyze-the-job-market-im-trying-to-enter-part-2-can-a-job-title-841a9585ec62)

## The question

Given only a job title (not the posting body), can a model predict whether a specific skill will be required? Uses the 95-posting dataset from Part 1.

## What `skill_prediction.py` does

1. Filters the ~30 tagged skills down to the ones with enough examples to model: at least 15 postings that mention the skill and at least 15 that don't (out of 95). That leaves **16 modelable skills**.
2. For each of those 16 skills, trains a logistic regression classifier on TF-IDF features of the job title (unigrams + bigrams) plus the posting's `role_category`, and predicts whether the skill is mentioned anywhere in the full posting.
3. Evaluates with **leave-one-out cross-validation (LOOCV)** — with only 95 rows, a normal train/test split would leave too little data on either side to mean anything.
4. Reports F1, precision, recall, and accuracy per skill, plus the "always predict no" baseline accuracy for comparison (a reminder that accuracy alone is misleading when a skill is rare).

## Run it

```
python skill_prediction.py
```

## What it found

Four skills cleared a real bar (F1 ≥ 0.60) — title alone was a decent predictor:

| Skill | F1 |
|---|---|
| Statistics | 0.77 |
| Python | 0.71 |
| Machine Learning | 0.67 |
| Data Mining | 0.61 |

The other eleven modelable skills (SQL, Hadoop, Big Data, Communication, Tableau, Spark, SAS, Data Visualization, R, Java, Business Intelligence) landed in a mediocre F1 0.2–0.5 band — better than a coin flip, not good enough to build a tool on.

**The honest conclusion:** title-only signal from 95 postings in one metro area, one quarter of 2016, had basically hit its ceiling. Part 3 tests whether that signal holds up with ~80x the data and the full posting body instead of just the title.
