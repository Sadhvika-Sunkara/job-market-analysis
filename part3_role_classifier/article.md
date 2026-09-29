# Building a Model to Predict What Skills You Actually Need (Part 3: The ML)

*Part 3 of "I Built a Data Pipeline to Analyze the Job Market I'm Trying to Enter"*

In Part 1, I scraped and cleaned 95 real Indeed postings from a three-month window in early 2016, concentrated in the DC-metro federal-contractor market. I built role categories off the job titles (mostly Data Scientist and Data Analyst — Data Engineer only had 2 postings, too few to say anything about) and tagged each posting against a ~30-term skill dictionary. Even at that tiny scale, something real showed up: Python appeared in 58% of Data Scientist postings and 0% of Data Analyst postings. Machine Learning showed a similar gap.

In Part 2, I turned that observation into an actual model instead of a table of percentages. For each of 16 "modelable" skills — skills common enough in the 95 postings to have at least 15 positive and 15 negative examples — I trained a classifier that read *only the job title* and predicted whether that skill would be required, evaluated with leave-one-out cross-validation because 95 rows is nowhere near enough for a normal train/test split. Four skills cleared a real bar (F1 ≥ 0.60): Statistics (0.76), Python (0.71), Machine Learning (0.69), Data Mining (0.63). The other twelve — SQL, Tableau, Spark, Hadoop, and the rest — landed in a mediocre F1 0.30–0.49 band: better than guessing, not good enough to build anything on.

The honest conclusion of Part 2 was that title-only signal from 95 postings in one metro area during one quarter of 2016 had basically hit its ceiling. There were two ways to find out if that signal was real or a small-sample artifact: get more data, and stop throwing away the body of the posting. So for Part 3 I did both — moved from 95 postings to 7,523, and from title-only text to the full job description.

That gives me two things to actually check. First: **does the 2016 signal even hold up years later and 80x the data later?** (The new dataset doesn't carry per-posting dates, but it's a broad, current multi-source aggregation, not a 2016 snapshot.) Second, the harder question I set the model: **if you strip the title off a job posting entirely, can it figure out which of four roles the posting actually is, just from the body text?** This post walks through both — the comparison against Part 1's numbers, then the real ML pipeline: labeling, feature engineering (and the piece of it that didn't pay off), model selection, and full evaluation, including a robustness check that tries to catch the model cheating.

## Does the 2016 signal hold up at scale?

Before touching the model, I checked Part 1's own numbers against the new dataset — same skills, same Data Scientist vs. Data Analyst comparison, just 7,523 postings instead of 95, pulled from a much broader and more current aggregation of listings rather than one metro area in one quarter of 2016.

| Skill | 2016 gap (95 postings) | Now (7,523 postings) | Holds up? |
|---|---|---|---|
| Machine Learning | DS 58% − DA 12% = 46 pts | DS 57.0% − DA 7.4% = 49.6 pts | Yes — nearly identical |
| Excel | DA 41% − DS 6% = 35 pts | DA 59.3% − DS 31.6% = 27.7 pts | Yes — same direction, narrower |
| Statistics | DS 80% − DA 71% = 9 pts | DS 51.2% − DA 29.3% = 21.9 pts | Yes — direction holds, gap widened |
| Python | DS 58% − DA 0% = 58 pts | DS 65.6% − DA 28.5% = 37.1 pts | Direction holds, gap nearly halved |
| SQL | DS 41% − DA 35% = 6 pts (scientist-favored) | DA 63.8% − DS 46.1% = 17.7 pts (analyst-favored) | **No — it reversed** |

![Grouped bar chart comparing the Data Scientist vs. Data Analyst mention-rate gap for five skills in 2016 (95 postings) versus today (7,523 postings). Machine Learning, Excel, Statistics, and Python all keep the same direction; SQL flips from scientist-favored to analyst-favored.](chart_2016_vs_now.png)

Four of five relationships held up in direction, which is a genuinely reassuring sign that Part 1's tiny sample wasn't a fluke. The Python gap nearly halving is a real finding, not noise — it fits the broader story that Python stopped being an ML-specialist skill and became baseline tooling for analysts too, somewhere in the years between that 2016 scrape and whatever more recent snapshot this new dataset represents. And SQL flipping outright is worth taking at face value rather than explaining away: it's exactly the kind of thing you'd only catch by re-running the check, not by assuming an old finding still holds.

That comparison answers the first question. The second — and the harder one — needed the model.

## The problem, framed as ML

The task: **multi-class text classification.** Given only the free-text description of a job posting, predict which of four role categories it belongs to — `data_analyst`, `data_scientist`, `data_engineer`, or `ai_ml_engineer`. The job title itself is withheld from the model at prediction time; it's only used to generate the label. Unlike Part 2's per-skill title classifiers, this uses the entire posting body and asks about role rather than individual skills — a necessary building block for Part 4's tool, since you can't recommend a skill checklist for "this kind of job" until something can reliably tell what kind of job it is from the text alone.

That labeling step matters and deserves to be stated plainly: I don't have an independent, human-verified ground truth for "what kind of job this really is." What I have is the title the employer chose. So I built an explicit keyword taxonomy against the title field — anything matching `machine learning|ml engineer|deep learning|computer vision engineer` becomes `ai_ml_engineer`, anything matching `data engineer|etl engineer|analytics engineer` becomes `data_engineer`, and so on, with `data_scientist` titles containing "manager" or "director" excluded to avoid mixing IC and leadership tracks. Titles that didn't clearly match one of the four buckets — generic "Business Analyst," "Software Engineer," "Systems Analyst" — were dropped rather than force-fit. That cost me rows, but a wrong label is worse than a missing one.

Starting from the raw scrape (16,015 postings pulled from Indeed, LinkedIn, and Glassdoor listings), I dropped rows with missing or near-empty descriptions (under 150 characters — mostly scraping artifacts, not real postings) and removed exact-duplicate description text, which is common when the same req gets reposted. That left 14,276 usable postings. After applying the title taxonomy and keeping only the four target roles, I had **7,523 labeled examples**:

| Role | Count |
|---|---|
| Data Scientist | 2,902 |
| Data Analyst | 2,246 |
| Data Engineer | 1,841 |
| AI/ML Engineer | 534 |

That imbalance — AI/ML Engineer is roughly a fifth the size of Data Scientist — turns out to matter later, and I didn't paper over it.

## Feature engineering: more than default TF-IDF

The obvious baseline is TF-IDF over the description text and calling it done. I used TF-IDF, but not blindly, and I built a second, independent feature family on top of it so I could actually test whether the extra engineering was worth the code.

**Text features.** `TfidfVectorizer` with unigrams and bigrams, `min_df=5` (drop terms appearing in fewer than 5 postings — mostly typos and company-specific jargon), `max_df=0.6` (drop terms in more than 60% of postings — generic boilerplate like "equal opportunity employer"), `sublinear_tf=True` (dampens the effect of a word appearing 50 times vs. 5 times in one posting), and English stopword removal, capped at 6,000 features.

**Structured skill/tool features.** A hand-written taxonomy of ~56 skill and tool mentions, grouped into five families — core languages (Python, SQL, R, Scala, Java, C++), ML/DS tooling (TensorFlow, PyTorch, scikit-learn, XGBoost, NLP, computer vision, statistics), data-engineering tooling (Spark, Kafka, Airflow, ETL, Kubernetes, Snowflake, dbt), BI/analyst tooling (Tableau, Power BI, Excel, dashboards, KPIs), and general engineering practice (Git, Linux, Agile, CI/CD). Each becomes a binary "does this posting mention X" feature, regex-matched with word boundaries so "r" doesn't fire on every word containing the letter.

**Structural features.** On top of the skill flags: a per-group mention count, a years-of-experience figure extracted via regex, three binary degree-level flags (bachelor's/master's/PhD), description length, and word count. That's 68 engineered features total, standardized and concatenated onto the TF-IDF matrix with `scipy.sparse.hstack`.

Here's the part worth being honest about, because it's more interesting than a clean success story: **the engineered feature block did not improve the model once TF-IDF was already in play.** A 5-fold cross-validated ablation showed TF-IDF alone at 0.8206 macro-F1 versus TF-IDF + engineered features at 0.8158 — a small negative delta, within fold-to-fold noise (std ≈ 0.004). But the engineered features aren't useless: trained completely alone, with zero text vectorization, they hit 0.625 macro-F1 by themselves — far above the ~0.25 you'd get guessing among four classes, just not competitive with the full text model. The read: TF-IDF already implicitly encodes the same signal — if "Kubernetes" is discriminative, TF-IDF finds it as a token on its own, so a hand-built `skill_kubernetes` flag mostly duplicates information the vectorizer already has. Lesson for Part 4: feature engineering that looks reasonable on paper needs to be ablation-tested, not assumed.

## Model selection: why linear, and what I compared

With ~6,000 TF-IDF dimensions plus 68 structured features on ~6,000 training rows, this is a classic high-dimensional, sparse, bag-of-words problem — the regime where linear models tend to beat tree ensembles, because a linear separator can assign a small weight to thousands of sparse features cheaply, while trees have to greedily pick single-feature splits and struggle to exploit that same breadth efficiently.

I compared four candidates with 5-fold stratified cross-validation (macro-F1) on the training split, all with class-balanced weighting to account for the AI/ML Engineer class being underrepresented:

| Model | CV macro-F1 (mean ± std) |
|---|---|
| Multinomial Naive Bayes (TF-IDF only baseline) | 0.617 ± 0.005 |
| Logistic Regression (balanced) | 0.784 ± 0.011 |
| **Linear SVM (balanced, C=0.5)** | **0.816 ± 0.004** |
| Random Forest (200 trees, balanced) | 0.795 ± 0.010 |

![Horizontal bar chart of 5-fold cross-validated macro-F1 for four models: Naive Bayes 0.617, Logistic Regression 0.784, Random Forest 0.795, and Linear SVM 0.816 (selected, highlighted).](chart_model_comparison.png)

Naive Bayes is the floor — a reasonable sanity check, but its independence assumption between features costs it real accuracy. Random Forest did respectably (0.795) but confirmed the expectation: it doesn't beat a well-regularized linear model on sparse, high-dimensional text, and it's considerably slower to train. Linear SVM won, with the lowest variance across folds too, and I went with it as the final model.

## Held-out test results

After selecting Linear SVM by cross-validation, I evaluated it once on the untouched 20% test split (1,505 postings) — the numbers below are that single, honest hold-out run, not the cross-validation numbers:

- **Accuracy:** 0.874
- **Macro F1:** 0.832
- **Weighted F1:** 0.874
- **Macro ROC-AUC (one-vs-rest):** 0.961
- **Weighted ROC-AUC (one-vs-rest):** 0.967

Accuracy alone would have hidden the real story, which is why macro F1 and the per-class breakdown matter:

| Role | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| AI/ML Engineer | 0.663 | 0.645 | 0.654 | 107 |
| Data Analyst | 0.906 | 0.922 | 0.914 | 449 |
| Data Engineer | 0.869 | 0.883 | 0.876 | 368 |
| Data Scientist | 0.891 | 0.874 | 0.883 | 581 |

Data Analyst, Data Engineer, and Data Scientist all land in the high 0.8s to low 0.9s on F1. AI/ML Engineer is the clear weak point at 0.654 — no surprise given it's the smallest class (534 total examples, only 107 in the test set) and, per the confusion matrix, the one that most overlaps with Data Scientist:

```
                  pred_ai_ml  pred_analyst  pred_engineer  pred_scientist
true_ai_ml               69             4             11              23
true_analyst               0           414             13              22
true_engineer             10            16            325              17
true_scientist            25            23             25             508
```

![Heatmap confusion matrix for the held-out test set. Data Analyst (92%), Data Engineer (88%), and Data Scientist (87%) are all correctly classified most of the time; AI/ML Engineer is weaker at 64%, with most of its errors going to Data Scientist.](chart_confusion_matrix.png)

23 of 107 true AI/ML Engineer postings got called Data Scientist, and 25 of 581 true Data Scientist postings got called AI/ML Engineer — which tracks, since both talk about models, Python, and statistics, and the line between the two titles is genuinely blurry in how companies write job ads.

## A robustness check: is it learning skills, or just the title?

Job postings often restate their own title in the first line of the body ("Data Engineer in Philadelphia, PA…"), so some of what looks like the model learning "skills" could just be it detecting the word "engineer." To check this, I stripped literal title-echo phrases ("data scientist," "data engineer," "machine learning engineer") out of the description for every row where they appeared — 5,573 of 7,523 rows, not a rare edge case — and reran the same 5-fold CV.

Macro-F1 dropped from 0.816 to **0.788 ± 0.013**. That's a real drop, and it confirms the title-echo phrases were doing meaningful work. But 0.788 is still far above the 0.625 the skill-only features get on their own, and far above chance — so the model isn't purely pattern-matching the title back to itself. There's genuine content signal left once the obvious giveaway is removed.

## What the model actually learned

Looking at logistic regression coefficients trained on the skill features alone (isolated from TF-IDF, so the ranking isn't dominated by title-echo n-grams), the top distinguishing signals per role line up with intuition, but with real numbers behind them instead of a vibe:

- **AI/ML Engineer:** `machine_learning`, PhD mentioned, XGBoost, deep learning, computer vision, PyTorch
- **Data Analyst:** description length (analyst postings tend to be shorter), SQL, "reporting," Excel, Google Analytics
- **Data Engineer:** word count, `data_pipeline`, `data_warehouse`, ETL, Kafka, Spark
- **Data Scientist:** description length, statistics, R, XGBoost, Keras, Python

Raw mention rates tell the same story more concretely — TensorFlow appears in 24.0% of AI/ML Engineer postings versus 1.2% of Data Engineer postings; Airflow in 12.9% of Data Engineer postings versus 0.9% of Data Scientist postings; Excel in 59.3% of Data Analyst postings versus 0.3% of Data Engineer postings. One surprise: Python appears in 65.6% of Data Scientist and 61.0% of Data Engineer postings, but only 46.3% of AI/ML Engineer postings — likely because ML-engineer postings name specific frameworks (TensorFlow, PyTorch) rather than the underlying language.

## What I'd improve with more time or data

I'm not going to present 0.832 macro-F1 as a finished product. Given more runway: more AI/ML Engineer examples first — 534 rows is thin for a class already semantically close to Data Scientist, and I'd rather source more real postings than synthetically balance the classes. I'd also try transformer-based sentence embeddings alongside TF-IDF, since TF-IDF can't tell that "builds recommendation systems" and "ranks candidate items with a learned model" mean roughly the same thing without a repeated literal token. `LinearSVC` didn't fully converge within a reasonable iteration cap on this feature scale (a known liblinear quirk on wide combined sparse/dense matrices) — results were stable across two independent runs, but I'd move to `SGDClassifier` with hinge loss rather than just raising `max_iter`. And I'd run a real hyperparameter search over `C`, `max_features`, and `ngram_range` instead of hand-picked values, and consider a multi-label setup for genuinely hybrid postings ("Analytics Engineer," "ML/Data Scientist") that don't cleanly fit one of four buckets.

Part 4 takes this model and wraps it in something you can actually talk to: an agent that reads a posting, classifies it, and tells you what to go learn.
