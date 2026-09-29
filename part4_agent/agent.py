"""
agent.py -- Part 4: "wraps the Part 3 classifier in an agent: read a
posting, classify its role, and tell you what to go learn" (from this
repo's own README and the closing line of Part 3's article).

Two entry points:

  answer_skill_question(query)
      Free-text question like "what skills do I need to become a Data
      Engineer?" -> routes to a role category -> retrieves the most
      relevant real postings from the combined corpus (TF-IDF + cosine
      similarity) -> synthesizes a grounded, cited answer from actual
      skill-mention frequencies, using Part 3's own skill taxonomy so
      the numbers are apples-to-apples with Parts 1-3.

  analyze_posting(text)
      Paste a job posting (or a resume / skills summary) -> the actual
      Part 3 model (TF-IDF + 68 engineered features + Linear SVM,
      retrained here on the full combined corpus rather than an 80/20
      split, since this is the shipped tool, not the benchmark run)
      classifies which of the four roles it matches -> retrieves real
      postings for that predicted role -> reports which commonly-required
      skills are present in the pasted text and which are missing. This
      is the actual "tell you what to go learn" gap check.

Both entry points refuse to generate free text: every claim is a
frequency count or a verbatim quote from a retrieved real posting, with
its source attached. Nothing here is an LLM call (see the module
docstring in the README for the discussion of why, and the honest
trade-offs of that choice vs. a generative synthesis layer).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORPUS_CSV = HERE / "combined_corpus.csv"

# Reuse Part 3's exact skill taxonomy and feature-engineering function
# rather than redefining a second, possibly-inconsistent one. This is
# the literal "wraps the Part 3 classifier" part.
sys.path.insert(0, str(REPO_ROOT / "part3_role_classifier"))
from train_role_classifier import build_skill_features, SKILL_GROUPS  # noqa: E402

RANDOM_STATE = 42

ROLE_LABELS = {
    "data_analyst": "Data Analyst",
    "data_scientist": "Data Scientist",
    "data_engineer": "Data Engineer",
    "ai_ml_engineer": "AI/ML Engineer",
}


def route_category(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["ai engineer", "ai/ml engineer", "ml engineer",
                             "machine learning engineer", "genai engineer"]):
        return "ai_ml_engineer"
    if "data engineer" in q:
        return "data_engineer"
    if any(k in q for k in ["data analyst", "business analyst", "bi analyst"]):
        return "data_analyst"
    if "data scientist" in q:
        return "data_scientist"
    return "data_scientist"


class JobMarketAgent:
    def __init__(self, corpus_path: Path = CORPUS_CSV):
        self.df = pd.read_csv(corpus_path)
        self.df["description"] = self.df["description"].fillna("")
        self._train_classifier()
        self._retrieval_vectorizers = {}
        self._retrieval_matrices = {}

    # ------------------------------------------------------------------
    # Classifier: same feature pipeline as Part 3's train_role_classifier.py,
    # trained on the full combined corpus (no held-out split) since this
    # is the deployed model, not the benchmark. Part 3's article already
    # reports the honest held-out numbers (87.4% accuracy, 0.832 macro-F1)
    # this same pipeline gets on unseen data -- see model_results.json.
    # ------------------------------------------------------------------
    def _train_classifier(self):
        X_text = self.df["description"].astype(str)
        y_raw = self.df["role_category"].astype(str)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw)

        struct_feats = build_skill_features(X_text)
        self._struct_cols = struct_feats.columns.tolist()

        self.tfidf = TfidfVectorizer(
            max_features=6000, ngram_range=(1, 2), min_df=5, max_df=0.6,
            sublinear_tf=True, stop_words="english",
        )
        X_tfidf = self.tfidf.fit_transform(X_text)

        self.scaler = StandardScaler()
        struct_scaled = self.scaler.fit_transform(struct_feats.values)

        X = sparse.hstack([X_tfidf, sparse.csr_matrix(struct_scaled)]).tocsr()

        self.clf = LinearSVC(class_weight="balanced", C=0.5,
                              random_state=RANDOM_STATE, max_iter=5000)
        self.clf.fit(X, y)

    def classify(self, text: str):
        struct = build_skill_features(pd.Series([text]))
        struct = struct.reindex(columns=self._struct_cols, fill_value=0)
        X_tfidf = self.tfidf.transform([text])
        struct_scaled = self.scaler.transform(struct.values)
        X = sparse.hstack([X_tfidf, sparse.csr_matrix(struct_scaled)]).tocsr()

        scores = self.clf.decision_function(X)[0]
        classes = self.label_encoder.classes_
        ranked = sorted(zip(classes, scores), key=lambda t: -t[1])
        predicted = ranked[0][0]
        return predicted, ranked

    # ------------------------------------------------------------------
    # Retrieval: TF-IDF + cosine similarity within a role_category subset.
    # ------------------------------------------------------------------
    def _get_retrieval_index(self, category: str):
        if category in self._retrieval_vectorizers:
            return self._retrieval_vectorizers[category], self._retrieval_matrices[category]
        subset = self.df[self.df["role_category"] == category]
        vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
        mat = vec.fit_transform(subset["description"])
        self._retrieval_vectorizers[category] = vec
        self._retrieval_matrices[category] = mat
        return vec, mat

    def retrieve(self, query_text: str, category: str, top_k: int = 20) -> pd.DataFrame:
        subset = self.df[self.df["role_category"] == category].copy()
        vec, mat = self._get_retrieval_index(category)
        qvec = vec.transform([query_text])
        scores = cosine_similarity(qvec, mat).flatten()
        subset = subset.assign(similarity=scores).sort_values("similarity", ascending=False)
        return subset.head(top_k)

    # ------------------------------------------------------------------
    # Skill frequency aggregation, using Part 3's own taxonomy.
    # ------------------------------------------------------------------
    def skill_frequencies(self, descriptions: pd.Series):
        feat = build_skill_features(descriptions)
        skill_cols = [c for c in feat.columns if c.startswith("skill_")]
        n = len(descriptions)
        if n == 0:
            return []
        means = feat[skill_cols].mean(axis=0).sort_values(ascending=False)
        return [
            (col.replace("skill_", ""), round(means[col] * 100), int(feat[col].sum()))
            for col in means.index if means[col] > 0
        ]

    # ------------------------------------------------------------------
    # Entry point 1: free-text skill question.
    # ------------------------------------------------------------------
    def answer_skill_question(self, query: str, top_k: int = 20) -> str:
        category = route_category(query)
        retrieved = self.retrieve(query, category, top_k=top_k)
        freqs = self.skill_frequencies(retrieved["description"])

        lines = [f"Q: {query}", ""]
        lines.append(
            f"Routed to: {ROLE_LABELS[category]}. Retrieved {len(retrieved)} real postings "
            f"(out of {(self.df['role_category'] == category).sum()} in this role in the corpus)."
        )
        lines.append("\nTop skills, ranked by % of retrieved postings that mention them:")
        for skill, pct, count in freqs[:12]:
            lines.append(f"  - {skill.replace('_', ' ')} ({pct}%, {count}/{len(retrieved)})")

        lines.append("\nGrounding quotes (pulled verbatim from retrieved postings):")
        quoted = 0
        for _, row in retrieved.iterrows():
            if quoted >= 3:
                break
            text = str(row["description"]).strip()
            if len(text) < 60:
                continue
            snippet = text[:220].replace("\n", " ") + "..."
            company = row["company"] if isinstance(row["company"], str) else "unlisted company"
            lines.append(f'  - [{row["job_title"]} @ {company}] "{snippet}"')
            quoted += 1

        lines.append("\nTop sources (by similarity):")
        for _, row in retrieved.head(5).iterrows():
            company = row["company"] if isinstance(row["company"], str) else "unlisted"
            lines.append(f'  - {row["job_title"]} @ {company} (sim={row["similarity"]:.3f}, source={row["source"]})')

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Entry point 2: classify a pasted posting, then gap-check it.
    # ------------------------------------------------------------------
    def analyze_posting(self, text: str, top_k: int = 25) -> str:
        predicted, ranked = self.classify(text)
        retrieved = self.retrieve(text, predicted, top_k=top_k)
        corpus_freqs = self.skill_frequencies(retrieved["description"])
        own_flags = build_skill_features(pd.Series([text])).iloc[0]

        lines = [f"Predicted role: {ROLE_LABELS[predicted]}"]
        lines.append("Decision function scores (higher = more confident), all 4 classes:")
        for cls, score in ranked:
            lines.append(f"  - {ROLE_LABELS[cls]}: {score:+.3f}")

        present = []
        missing = []
        for skill, pct, count in corpus_freqs:
            if pct < 15:
                continue  # not common enough in real postings to be a meaningful gap
            col = f"skill_{skill}"
            has_it = bool(own_flags.get(col, 0))
            (present if has_it else missing).append((skill, pct))

        lines.append(
            f"\nAmong {len(retrieved)} real {ROLE_LABELS[predicted]} postings retrieved as the "
            f"comparison set, skills mentioned in >=15% of them:"
        )
        lines.append("\nAlready present in the pasted text:")
        for skill, pct in present[:10]:
            lines.append(f"  - {skill.replace('_', ' ')} ({pct}% of comparable postings)")
        lines.append("\nCommonly required but NOT found in the pasted text -- worth learning:")
        for skill, pct in missing[:10]:
            lines.append(f"  - {skill.replace('_', ' ')} ({pct}% of comparable postings)")

        return "\n".join(lines)


if __name__ == "__main__":
    agent = JobMarketAgent()
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        posting_text = " ".join(sys.argv[2:])
        print(agent.analyze_posting(posting_text))
    else:
        query = " ".join(sys.argv[1:]) or "What skills do I need to become a Data Engineer?"
        print(agent.answer_skill_question(query))
