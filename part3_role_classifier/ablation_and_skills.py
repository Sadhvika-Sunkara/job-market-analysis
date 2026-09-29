"""
Two follow-up analyses for the Part 3 article:

1. Ablation: does the hand-built skill/structural feature block actually
   add predictive value on top of TF-IDF, or is it dead weight? Compares
   5-fold CV macro-F1 for LinearSVM on (a) TF-IDF only vs (b) TF-IDF +
   engineered features, using the exact same split/vectorizer settings
   as train_role_classifier.py.

2. Skill-only importance: fit logistic regression on ONLY the engineered
   skill/structural features (no TF-IDF at all) so the top coefficients
   aren't drowned out by title-echo n-grams like "data scientist". This
   answers "which mentioned skills/tools actually separate the four
   roles" in a way that's readable in the article.
"""
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from train_role_classifier import build_skill_features, RANDOM_STATE

df = pd.read_csv("cleaned_jobs.csv").dropna(subset=["description", "role_category"])
X_text = df["description"].astype(str)
le = LabelEncoder()
y = le.fit_transform(df["role_category"].astype(str))
class_names = le.classes_.tolist()

struct_feats = build_skill_features(X_text)
struct_cols = struct_feats.columns.tolist()

X_train_text, X_test_text, y_train, y_test, struct_train, struct_test = train_test_split(
    X_text, y, struct_feats, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

tfidf = TfidfVectorizer(
    max_features=6000, ngram_range=(1, 2), min_df=5, max_df=0.6,
    sublinear_tf=True, stop_words="english",
)
Xtr_tfidf = tfidf.fit_transform(X_train_text)

scaler = StandardScaler()
struct_train_scaled = scaler.fit_transform(struct_train.values)
Xtr_combined = sparse.hstack([Xtr_tfidf, sparse.csr_matrix(struct_train_scaled)]).tocsr()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
clf = LinearSVC(class_weight="balanced", C=0.5, random_state=RANDOM_STATE, max_iter=5000)

scores_tfidf_only = cross_val_score(clf, Xtr_tfidf, y_train, cv=skf, scoring="f1_macro", n_jobs=1)
scores_combined = cross_val_score(clf, Xtr_combined, y_train, cv=skf, scoring="f1_macro", n_jobs=1)

print("=== ABLATION: LinearSVM 5-fold CV macro-F1 ===")
print(f"TF-IDF only:            {scores_tfidf_only.mean():.4f} +/- {scores_tfidf_only.std():.4f}")
print(f"TF-IDF + engineered:    {scores_combined.mean():.4f} +/- {scores_combined.std():.4f}")
print(f"Delta:                  {scores_combined.mean() - scores_tfidf_only.mean():+.4f}")

# skill-only importance (no TF-IDF at all)
skill_scaler = StandardScaler()
struct_all_scaled = skill_scaler.fit_transform(struct_feats.values)
skill_logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
skill_logreg.fit(struct_all_scaled, y)

skill_cv_scores = cross_val_score(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    struct_all_scaled, y, cv=skf, scoring="f1_macro", n_jobs=1
)
print(f"\nSkill-features-only LogReg 5-fold CV macro-F1: {skill_cv_scores.mean():.4f} +/- {skill_cv_scores.std():.4f}")

top_skill_terms = {}
for i, cname in enumerate(class_names):
    coefs = skill_logreg.coef_[i]
    top_idx = np.argsort(coefs)[-10:][::-1]
    top_skill_terms[cname] = [(struct_cols[j], float(coefs[j])) for j in top_idx]
    print(f"\nTop skill/structural signals for {cname}:")
    for name, w in top_skill_terms[cname]:
        print(f"   {name}: {w:.3f}")

out = {
    "ablation": {
        "tfidf_only_mean_macro_f1": float(scores_tfidf_only.mean()),
        "tfidf_only_std_macro_f1": float(scores_tfidf_only.std()),
        "combined_mean_macro_f1": float(scores_combined.mean()),
        "combined_std_macro_f1": float(scores_combined.std()),
        "delta": float(scores_combined.mean() - scores_tfidf_only.mean()),
    },
    "skill_only_cv_macro_f1_mean": float(skill_cv_scores.mean()),
    "skill_only_cv_macro_f1_std": float(skill_cv_scores.std()),
    "top_skill_terms_per_class": top_skill_terms,
}
with open("ablation_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nSaved ablation_results.json")
