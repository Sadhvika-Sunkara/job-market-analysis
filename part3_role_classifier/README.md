# Part 3 — Predicting Role Category from Full Posting Text

Companion code for ["Building a Model to Predict What Skills You Actually Need (Part 3: The ML)"](https://medium.com/@sadhvikasunkara4/building-a-model-to-predict-what-skills-you-actually-need-part-3-the-ml-9baf05bca00f)

## The question

Part 2 showed title-only signal on 95 postings had hit its ceiling. Part 3 scales up — more data, full description text instead of just the title — and asks a harder question: **if you strip the title off a posting, can a model tell which of four roles it actually is, just from the body text?**

## Dataset

`cleaned_jobs.csv` (7,523 labeled postings) is built from a public aggregation of ~16,000 Data Analyst / Data Scientist / Data Engineer / ML job postings ([natvalenz/DS_Jobs_PROJECT](https://github.com/natvalenz/DS_Jobs_PROJECT)), cleaned and role-labeled by `build_dataset.py`. The raw scrape (`raw_fulldataset.csv`, ~70MB) isn't checked into this repo — regenerate it with:

```
curl -L -o raw_fulldataset.csv https://raw.githubusercontent.com/natvalenz/DS_Jobs_PROJECT/master/FullDataSet.csv
python build_dataset.py          # -> cleaned_jobs.csv
```

## Pipeline

| Script | What it does |
|---|---|
| `build_dataset.py` | Cleans the raw scrape, drops empty/duplicate descriptions, labels `role_category` from the title via keyword rules, drops ambiguous titles |
| `train_role_classifier.py` | TF-IDF + a 68-feature hand-built skill/structural taxonomy → compares Naive Bayes, Logistic Regression, Linear SVM, Random Forest via 5-fold CV, evaluates the winner on a held-out test set |
| `ablation_and_skills.py` | Tests whether the engineered feature block actually adds value on top of TF-IDF, and ranks skill-only signals per role |
| `robustness_check.py` | Strips literal title-echo phrases out of the description text and reruns CV, to check the model isn't just detecting the restated title |
| `make_charts.py` | Generates the three charts in `charts/` from the real results |

Run in order: `build_dataset.py` → `train_role_classifier.py` → `ablation_and_skills.py` → `robustness_check.py` → `make_charts.py`.

## Results (real, not illustrative)

- **Model selected:** Linear SVM, chosen by 5-fold CV macro-F1 (0.816) over Naive Bayes (0.617), Logistic Regression (0.784), and Random Forest (0.795).
- **Held-out test set:** 87.4% accuracy, 0.832 macro-F1, 0.961 macro ROC-AUC (one-vs-rest). Full per-class breakdown and confusion matrix in `model_results.json` and `article.md`.
- **Ablation:** the 68 engineered skill/structural features did *not* improve on TF-IDF alone (0.821 vs. 0.816 CV macro-F1) — TF-IDF already implicitly captures the same signal. Full writeup in `article.md`.
- **Robustness check:** stripping title-echo phrases dropped CV macro-F1 from 0.816 to 0.788 — a real drop, but still far above the skill-only baseline (0.625), confirming the model isn't just pattern-matching the restated title.
- **2016 vs. now:** four of five skill-gap relationships found in Part 1/2's tiny 2016 sample still hold in this much larger, more current dataset; one (SQL) reversed direction. See `charts/chart_2016_vs_now.png` and `article.md`.

All numbers are pulled directly from `model_results.json` and `ablation_results.json` — nothing in `article.md` is hand-typed or estimated.

## Full writeup

`article.md` is the published Medium post — problem framing, feature engineering decisions, model selection reasoning, and what I'd improve with more time or data.
