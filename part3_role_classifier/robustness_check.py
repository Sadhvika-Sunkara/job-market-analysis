"""
Robustness check: how much of the model's performance is just the posting
restating its own title in the body text (e.g. "Data Engineer in
Philadelphia...") versus genuine skill/content signal?

Strips literal role-title phrases out of each description, rebuilds the
combined TF-IDF + engineered feature matrix, and reruns the same 5-fold
CV macro-F1 for LinearSVM used in train_role_classifier.py, on the exact
same train split.
"""
import re

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from train_role_classifier import build_skill_features, RANDOM_STATE

TITLE_ECHO_PHRASES = {
    "data_scientist": [r"data scientists?", r"data sciences?"],
    "data_analyst": [r"data analysts?"],
    "data_engineer": [r"data engineers?", r"data engineerings?"],
    "ai_ml_engineer": [r"machine learning engineers?", r"ml engineers?", r"ai engineers?"],
}

df = pd.read_csv("cleaned_jobs.csv").dropna(subset=["description", "role_category"])


def strip_title_echo(row):
    text = row["description"]
    for phrase in TITLE_ECHO_PHRASES[row["role_category"]]:
        text = re.sub(phrase, " ", text, flags=re.I)
    return text


df["description_stripped"] = df.apply(strip_title_echo, axis=1)

X_text = df["description_stripped"].astype(str)
le = LabelEncoder()
y = le.fit_transform(df["role_category"].astype(str))

struct_feats = build_skill_features(X_text)

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
Xtr = sparse.hstack([Xtr_tfidf, sparse.csr_matrix(struct_train_scaled)]).tocsr()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
clf = LinearSVC(class_weight="balanced", C=0.5, random_state=RANDOM_STATE, max_iter=5000)
scores = cross_val_score(clf, Xtr, y_train, cv=skf, scoring="f1_macro", n_jobs=1)

print("=== ROBUSTNESS: title-echo phrases stripped from description ===")
print(f"LinearSVM 5-fold CV macro-F1 (title-echo stripped): {scores.mean():.4f} +/- {scores.std():.4f}")
print("(compare to 0.8158 +/- 0.0043 with title-echo left in)")

n_changed = (df["description"] != df["description_stripped"]).sum()
print(f"\nRows where at least one title-echo phrase was found & stripped: {n_changed} / {len(df)}")
