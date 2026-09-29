"""
Part 2 — reconstruction of the per-skill, title-only classifiers.

For each "modelable" skill (>=15 positive AND >=15 negative examples out
of the 95 cleaned postings from Part 1), train a classifier that reads
ONLY the job title and predicts whether that skill is mentioned anywhere
in the posting. Evaluated with leave-one-out cross-validation (LOOCV)
because N=95 is far too small for a normal train/test split.
"""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

df = pd.read_csv("../part1_dataset_cleaning/cleaned_jobs.csv")
skill_cols = [c for c in df.columns if c.startswith("skill_")]

MIN_CLASS_COUNT = 15
modelable = [c for c in skill_cols if df[c].sum() >= MIN_CLASS_COUNT and (len(df) - df[c].sum()) >= MIN_CLASS_COUNT]
print(f"{len(modelable)} modelable skills: {[m.replace('skill_', '') for m in modelable]}")

titles = df["job_title"].fillna("").astype(str)
role_dummies = pd.get_dummies(df["role_category"]).values  # role_category as an extra feature block
loo = LeaveOneOut()

results = {}
for skill_col in modelable:
    y = df[skill_col].values
    preds = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in loo.split(titles):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        Xtr_text = vec.fit_transform(titles.iloc[train_idx])
        Xte_text = vec.transform(titles.iloc[test_idx])
        Xtr = sparse.hstack([Xtr_text, role_dummies[train_idx]]).tocsr()
        Xte = sparse.hstack([Xte_text, role_dummies[test_idx]]).tocsr()

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, y[train_idx])
        preds[test_idx] = clf.predict(Xte)

    f1 = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    acc = accuracy_score(y, preds)
    baseline_acc = max(y.mean(), 1 - y.mean())

    name = skill_col.replace("skill_", "")
    results[name] = {
        "n_positive": int(y.sum()), "n_negative": int(len(y) - y.sum()),
        "f1": float(f1), "precision": float(prec), "recall": float(rec),
        "accuracy": float(acc), "always_no_baseline_accuracy": float(baseline_acc),
    }
    print(f"{name:24s} F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  acc={acc:.3f}  "
          f"(always-majority-class baseline acc={baseline_acc:.3f})")

strong = {k: v for k, v in results.items() if v["f1"] >= 0.60}
mid = {k: v for k, v in results.items() if 0.30 <= v["f1"] < 0.60}
weak = {k: v for k, v in results.items() if v["f1"] < 0.30}

print(f"\nStrong (F1>=0.60): {sorted(strong, key=lambda k: -strong[k]['f1'])}")
print(f"Mid (0.30-0.59): {sorted(mid, key=lambda k: -mid[k]['f1'])}")
print(f"Weak (<0.30): {sorted(weak, key=lambda k: -weak[k]['f1'])}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results.json")
