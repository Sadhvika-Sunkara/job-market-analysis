# I Built an AI Agent That Tells You Exactly What to Learn to Get Hired (Part 4: The Agent)

*Part 4 of "I Built a Data Pipeline to Analyze the Job Market I'm Trying to Enter"*

Part 3 ended on a promise: "an agent that reads a posting, classifies it, and tells you what to go learn." This post builds that agent — and, honestly, it also breaks it a little, on purpose, because the way it breaks turned out to be the most useful finding in the whole series.

## Quick recap: Parts 1–3

**Part 1** scraped and cleaned 95 real Indeed postings from a three-month window in early 2016, concentrated in the DC-metro federal-contractor market, and tagged each one against a roughly 30-term skill dictionary. Even at that tiny scale, something real showed up: Python appeared in 58% of Data Scientist postings and 0% of Data Analyst postings.

**Part 2** turned that observation into a model instead of a percentage table. For each of 16 skills common enough in the 95 postings to model, a title-only classifier — evaluated with leave-one-out cross-validation, since 95 rows rules out a normal train/test split — tried to predict whether the posting body would mention that skill, using nothing but the job title. Four skills cleared a real bar (F1 ≥ 0.60): Statistics (0.77), Python (0.71), Machine Learning (0.67), Data Mining (0.61). The other twelve landed in a mediocre 0.2–0.5 band. The honest conclusion: title-only signal from 95 postings had hit its ceiling.

**Part 3** scaled up to answer whether that ceiling was a sample-size artifact or a real limit — 7,523 labeled postings (from a 16,015-posting raw aggregation across Indeed, LinkedIn, and Glassdoor), full description text instead of just titles, and a genuinely harder task: predict which of four roles (Data Analyst, Data Scientist, Data Engineer, AI/ML Engineer) a posting is for, from the body text alone, with the title withheld at prediction time. A Linear SVM over TF-IDF plus 68 hand-engineered skill/structural features won a four-way model comparison (5-fold CV macro-F1 0.816, beating Naive Bayes at 0.617, Logistic Regression at 0.784, Random Forest at 0.795) and scored 87.4% accuracy / 0.832 macro-F1 on a held-out test set. The weak point was AI/ML Engineer at F1 0.654 — the smallest class (534 of 7,523 rows) and the one most confusable with Data Scientist.

Part 4 takes that trained classifier and wraps it in something you can actually use.

## What Part 4 actually does

Two entry points, both grounded entirely in retrieved real postings — nothing here is LLM-generated text:

**Ask a question.** "What skills do I need to become a Data Engineer?" routes to a role category, retrieves the most relevant real postings for that role via TF-IDF + cosine similarity, and reports skill-mention frequencies plus verbatim quotes — using Part 3's own skill taxonomy so the numbers are apples-to-apples with the rest of the series.

**Paste a posting.** The literal "wraps the Part 3 classifier" feature: retrain Part 3's exact pipeline (TF-IDF + 68 engineered features + Linear SVM, this time on the full corpus rather than an 80/20 split, since this is the shipped tool rather than the benchmark run) to classify the pasted text into one of the four roles, retrieve comparable real postings for that predicted role, and report which commonly-required skills are already present in the pasted text versus which are missing.

```python
def analyze_posting(self, text, top_k=25):
    predicted, ranked = self.classify(text)
    retrieved = self.retrieve(text, predicted, top_k=top_k)
    corpus_freqs = self.skill_frequencies(retrieved["description"])
    own_flags = build_skill_features(pd.Series([text])).iloc[0]
    # skills common in retrieved postings but absent from `text`
    # become the "worth learning" list; see agent.py for the full version
```

`build_skill_features` isn't reimplemented — it's imported directly from Part 3's `train_role_classifier.py`. That matters for more than tidiness: it means Part 4's skill counts are computed by the exact same regex taxonomy Part 3 used to get its numbers, not a second, subtly different one that would make the two posts quietly incomparable.

## Real example: asking a question

Unedited output from `python3 agent.py "What skills do I need to become a Data Engineer?"`, against the combined 7,532-posting corpus:

```
Routed to: Data Engineer. Retrieved 20 real postings (out of 1843 in this role).

Top skills, ranked by % of retrieved postings that mention them:
  - python (40%, 8/20)
  - sql (40%, 8/20)
  - aws (40%, 8/20)
  - data warehouse (40%, 8/20)
  - spark (25%, 5/20)
  - scala (20%, 4/20)
  - redshift (15%, 3/20)
  - etl (15%, 3/20)

Grounding quotes (pulled verbatim from retrieved postings):
  - [Azure Data Engineer @ Tek Leaders] "Need experience with ADF V2,
    Azure Devops and CI/CD. Need Azure Data Catalog, Azure event hub..."
  - [Data Engineer/Big Data Engineer @ Eitacies] "Big Data, Cloud,
    Azure, AWS. We are looking for senior data engineer..."
```

That's a plain, useful answer, grounded in 1,843 real postings for that role. The more interesting run is the one that broke something.

## Real example: pasting a posting — and finding a genuine blind spot

I ran `analyze_posting` on the real, live "Senior AI Engineer" posting at Egen (fetched directly from their careers page in September 2026) — a posting full of LangChain, retrieval-augmented generation, LLM fine-tuning, and Vertex AI. Here's the unedited result:

```
Predicted role: Data Engineer
Decision function scores (higher = more confident), all 4 classes:
  - Data Engineer: +0.030
  - AI/ML Engineer: -0.401
  - Data Scientist: -0.647
  - Data Analyst: -1.428
```

That's wrong. It should be AI/ML Engineer. And the margin — +0.030 versus -0.401 — tells you it's not a confident wrong answer, it's a coin flip that landed on the wrong side. Compare that to running the same function on a genuinely Data-Engineer-shaped posting (Dun & Bradstreet's real "Data Engineer I" listing): +1.494 for Data Engineer against -0.925 for the next class. That's what a confident, correct call looks like. The Egen result isn't the model being generally unreliable — it's the model hitting something specific it has never seen.

I tracked down exactly what: `TfidfVectorizer(min_df=5)` — a completely reasonable setting for a 7,532-document corpus, meant to drop typos and one-off jargon — drops any term that appears in fewer than 5 training documents. "langchain" appears in exactly 4. It never becomes a feature the classifier can weight, positively or negatively. The same is true of "vector database," "prompt engineering," and effectively "llm" (a raw substring search found 110 hits, but every one of them turned out to be inside words like "fulfillment" — a word-boundary search finds zero real mentions across all 7,523 Part 3 postings). The AI Engineer title, and the retrieval-agent skill set it implies, is genuinely newer than every data source this series has drawn on, including the "current, multi-source aggregation" Part 3 used.

The independent half of the agent — the regex-based skill-frequency layer, which isn't subject to `min_df` at all — does register `skill_llm`, just honestly: 10%, 2 out of the 20 AI/ML Engineer postings retrieved for the "what skills" question above, and both of those 2 are the live 2026 postings I added specifically because the rest of the corpus doesn't have this vocabulary yet. The two halves of this agent disagree with each other on genuinely novel terms, for a real, traceable reason, and that disagreement is more informative than either half being silently confident.

## Why this is the right kind of finding for a portfolio piece

It would have been easy to quietly retrain with `min_df=1` or hand-pick a nicer example and call Part 4 a clean success. I didn't, for the same reason Part 3 didn't paper over the engineered-features ablation that showed no improvement, or the robustness check that showed the model partly relies on title-echo phrases. A system that only ever gets shown succeeding is not more trustworthy than one that gets shown failing in an explained way — it's just less examined. For a project whose entire premise is "let the real data talk," a classifier failing specifically and explainably on the newest, least-represented vocabulary in its training corpus is a more honest demonstration of how these systems behave in production than a cherry-picked win would be.

It's also a directly actionable finding, which is the whole point of this series: if you're evaluating any classifier trained on a real-world text corpus, check what `min_df` (or its equivalent) actually excludes before trusting it on emerging vocabulary — and if you have an independent, non-TF-IDF signal available (here, the regex skill layer), it's worth keeping around specifically because it fails differently than your main model.

## Grounding, and why there's no LLM call in this build

Both entry points are extractive by design — every line of output is either a counted frequency or a verbatim quote with its source attached, never generated text. That's a deliberate trade-off against a more natural-sounding generative summary, and it's the right one for this specific project: a tool that tells people what to learn needs to be checkable, and a template that only ever aggregates and quotes retrieved rows literally cannot hallucinate a skill that isn't in the data, whereas an LLM summarizing the same context genuinely can blend in something from its own pretraining, especially for a well-known role like Data Scientist where its prior is strong. The cost is fluency, not correctness — and for this project, that's the trade I want.

## What the whole series adds up to

Part 1 demonstrated real-world data cleaning on messy, duplicated, inconsistently formatted postings. Part 2 demonstrated honest statistical reasoning at a sample size too small for most people's comfort, including admitting where the signal ran out. Part 3 demonstrated a full supervised-learning pipeline at real scale — feature engineering, model selection, ablation, and a robustness check that tried to catch the model cheating. Part 4 demonstrates something different from all three: taking a trained model and a retrieval layer, wiring them into something a person can actually query, and then being honest in public about exactly where and why it breaks.

That range — data engineering, applied statistics, machine learning, and now the applied-AI-engineering work of shipping a model as a tool — on one continuous, real project, is the actual point of four posts instead of one. The repo is linked below; if you paste your own resume or a posting you're eyeing into `analyze_posting` and the gap list looks off, that's worth checking against the `min_df` explanation above before assuming it's just wrong.

---

### Sources

- [Part 1: What 95 Real Job Postings Actually Say](https://medium.com/@sadhvikasunkara4/i-built-a-data-pipeline-to-analyze-the-job-market-im-trying-to-enter-part-1-what-95-real-job-f6d8754fc7d6)
- [Part 2: Can a Job Title Predict Required Skills?](https://medium.com/@sadhvikasunkara4/i-built-a-data-pipeline-to-analyze-the-job-market-im-trying-to-enter-part-2-can-a-job-title-841a9585ec62)
- [Part 3: Building a Model to Predict What Skills You Actually Need](https://medium.com/@sadhvikasunkara4/building-a-model-to-predict-what-skills-you-actually-need-part-3-the-ml-9baf05bca00f)
- Part 1's raw source: [stevendoll/kaggle-jobs](https://github.com/stevendoll/kaggle-jobs) (Indeed postings, early 2016)
- Part 3's raw source: [natvalenz/DS_Jobs_PROJECT](https://github.com/natvalenz/DS_Jobs_PROJECT) (Indeed/LinkedIn/Glassdoor aggregation)
- Live September 2026 postings fetched directly from company career pages: [Egen – Senior AI Engineer](https://jobs.lever.co/egen/1b870652-5768-45e9-b55b-4420e6402314), [Jeeves – Senior AI Engineer](https://jobs.lever.co/tryjeeves/2f00206f-6091-4eed-8b5f-1325afdbfe30), [Dun & Bradstreet – Data Engineer I](https://jobs.lever.co/dnb/f6a4fe79-0170-4a61-a571-372103189d70), [Egen – Lead Data Engineer](https://jobs.lever.co/egen/38a572c3-2981-4970-82a4-03a683c58b25), [Bluesight – Data Analyst](https://jobs.lever.co/bluesight/371dd108-f884-48cb-8791-43643bdb5cf4), [Hevo Data – Senior Data Analyst](https://jobs.lever.co/hevodata/308fe760-b657-45d3-ab4c-a13efdcf230f), [TTEC Digital – Senior Business Data Analyst](https://jobs.lever.co/ttecdigital/f0ec4455-3e00-42e6-ab15-0bacfdaa2f2a), [Sonatype – Senior Data Scientist](https://jobs.lever.co/sonatype/e7059360-2031-4af1-9a77-92afaeb5e5be), [CI&T – Data Scientist](https://jobs.lever.co/ciandt/662600ba-acbf-462b-a916-b68bfc34d28b)
- scikit-learn `TfidfVectorizer`, `LinearSVC`, and `cosine_similarity` — [scikit-learn documentation](https://scikit-learn.org/stable/)
