# Part 4 — Wrapping the Classifier in an Agent

Companion code for "I Built an AI Agent That Tells You Exactly What to Learn to Get Hired (Part 4: The Agent)" — the closing post of the series.

## The question

Part 3 ended with: "an agent that reads a posting, classifies it, and tells you what to go learn." This is that agent.

## Corpus

`build_corpus.py` combines Part 3's real 7,523-posting corpus with 9 live postings fetched directly from company career pages (Lever/Greenhouse) in September 2026, covering Data Analyst, Data Scientist, Data Engineer, and AI Engineer roles. This isn't padding — it's necessary. Part 3's corpus, despite being large and "current" at the time it was aggregated, has essentially zero LLM-era vocabulary: a word-boundary search finds 0 real mentions of "llm" (the raw substring count is inflated by words like "fulfillment"), and "langchain", "vector database", and "prompt engineering" appear nowhere in 7,523 postings. The AI Engineer role — and the retrieval-agent tooling it implies — is newer than every data source this series has used. Run:

```
python build_corpus.py
```

to regenerate `combined_corpus.csv` (7,532 rows) from Part 3's dataset plus `fresh_2026_postings.json`.

## What `agent.py` does

Two entry points, both grounded entirely in retrieved real text — nothing here is LLM-generated:

**`answer_skill_question(query)`** — routes a free-text question ("what skills do I need to become a Data Engineer?") to one of the four role categories, retrieves the most similar real postings in that category via TF-IDF + cosine similarity, and reports skill-mention frequencies (using Part 3's own skill taxonomy, imported directly from `train_role_classifier.py`) plus verbatim supporting quotes and a source list.

**`analyze_posting(text)`** — the literal "wraps the Part 3 classifier" feature. Retrains Part 3's exact pipeline (TF-IDF + 68 engineered features + Linear SVM) on the full combined corpus, classifies the pasted text into one of the four roles, retrieves comparable real postings for that role, and reports which commonly-required skills are already present in the pasted text and which are missing.

```
python agent.py "What skills do I need to become a Data Engineer?"
python agent.py --analyze "<paste a job posting or resume summary here>"
```

## What it found — including a real limitation, not a hypothetical one

Running `analyze_posting` on the actual live Egen "Senior AI Engineer" posting (LangChain, RAG, LLM fine-tuning, Vertex AI) gets misclassified as **Data Engineer**, not AI/ML Engineer. This isn't a vague caveat — it's mechanistically explainable: `TfidfVectorizer(min_df=5)` drops any term appearing in fewer than 5 of the 7,532 training documents, and "langchain" appears in exactly 4. It never becomes a feature the classifier can see. The independent, regex-based skill-frequency layer (not subject to `min_df`) does register `skill_llm` correctly, just at a low, honest frequency (10% of retrieved AI/ML Engineer postings — literally only the 2 live postings that mention it). The two halves of this agent fail differently, and disagreeing with each other on genuinely novel vocabulary is a real, useful signal, not a bug to hide. Full discussion in the article.

## Dependencies

Needs everything Part 3 needs (`pandas`, `scikit-learn`, `scipy`) — `agent.py` imports `build_skill_features` directly from `../part3_role_classifier/train_role_classifier.py`, so that file must stay in place relative to this one.
