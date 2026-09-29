"""
Part 3 — role-category classifier.

Task: given only the free-text body of a job posting (title excluded),
predict which of four roles it is: data_analyst, data_scientist,
data_engineer, ai_ml_engineer.

Feature engineering combines three families:
  1. TF-IDF over the cleaned description text (word uni+bigrams).
  2. A hand-built skill/tool taxonomy -> ~48 binary "does this posting
     mention X" features (python, spark, tableau, kubernetes, ...).
  3. A handful of structural numeric features: description length,
     years-of-experience required (regex-extracted), degree level
     mentioned (bachelor/master/phd flags), count of distinct skills
     hit, and skill-category mix (ratio of ML-tool mentions vs
     BI-tool mentions vs data-eng-tool mentions).

Models compared: Multinomial Naive Bayes (bag-of-words baseline),
Logistic Regression (multinomial, balanced class weights), Linear SVM
(balanced class weights), and Random Forest on the same combined
feature matrix. Selection is by 5-fold stratified CV macro-F1 on the
training split; the winner is then scored once on a held-out test set.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Skill taxonomy — grouped so we can also build category-mix features
# ---------------------------------------------------------------------
SKILL_GROUPS = {
    "lang_core": [
        ("python", "python"), ("sql", "sql"), (r"\br\b", "r_lang"),
        ("scala", "scala"), (r"java\b", "java"), (r"c\+\+", "cpp"),
    ],
    "ml_ds": [
        ("machine learning", "machine_learning"), ("deep learning", "deep_learning"),
        ("tensorflow", "tensorflow"), ("pytorch", "pytorch"),
        ("scikit-learn", "scikit_learn"), ("keras", "keras"), ("nlp", "nlp"),
        ("computer vision", "computer_vision"), ("neural network", "neural_network"),
        ("xgboost", "xgboost"), ("statistics", "statistics"),
        ("a/b testing", "ab_testing"), ("regression", "regression"),
        ("classification model", "classification_model"),
        ("llm|large language model", "llm"),
    ],
    "data_eng": [
        ("spark", "spark"), ("hadoop", "hadoop"), ("kafka", "kafka"),
        ("airflow", "airflow"), ("etl", "etl"), ("data warehouse", "data_warehouse"),
        ("snowflake", "snowflake"), (r"dbt\b", "dbt"), ("kubernetes", "kubernetes"),
        ("docker", "docker"), ("aws", "aws"), ("azure", "azure"),
        ("gcp|google cloud", "gcp"), ("redshift", "redshift"),
        ("databricks", "databricks"), ("nosql", "nosql"), ("mongodb", "mongodb"),
        ("data pipeline", "data_pipeline"), ("data lake", "data_lake"),
    ],
    "bi_analyst": [
        ("tableau", "tableau"), ("power bi", "power_bi"), ("excel", "excel"),
        ("looker", "looker"), ("sql server", "sql_server"),
        ("data visualization", "data_visualization"), ("dashboard", "dashboard"),
        ("google analytics", "google_analytics"), ("kpi", "kpi"),
        ("stakeholder", "stakeholder"), ("reporting", "reporting"),
    ],
    "eng_general": [
        (r"git\b", "git"), ("linux", "linux"), ("agile", "agile"),
        (r"api\b", "api"), ("ci/cd", "ci_cd"),
    ],
}
ALL_SKILLS = [(group, pattern, name) for group, items in SKILL_GROUPS.items() for pattern, name in items]

DEGREE_PATTERNS = {
    "mentions_bachelor": re.compile(r"bachelor|b\.?s\.?\b|undergraduate degree", re.I),
    "mentions_master": re.compile(r"master|m\.?s\.?\b|msc\b", re.I),
    "mentions_phd": re.compile(r"ph\.?d|doctorate", re.I),
}
YEARS_EXP_RE = re.compile(r"(\d{1,2})\+?\s*(?:-\s*\d{1,2}\s*)?years?\s+(?:of\s+)?experience", re.I)


def build_skill_features(descriptions: pd.Series) -> pd.DataFrame:
    lower = descriptions.str.lower()
    cols = {}
    group_hits = {g: np.zeros(len(descriptions)) for g in SKILL_GROUPS}
    for group, regex_str, name in ALL_SKILLS:
        pattern = re.compile(regex_str, re.I)
        hit = lower.apply(lambda t: 1 if pattern.search(t) else 0).to_numpy()
        col_name = "skill_" + name
        cols[col_name] = hit
        group_hits[group] = group_hits[group] + hit

    feat = pd.DataFrame(cols)
    feat["n_skills_total"] = feat.sum(axis=1)
    for g in SKILL_GROUPS:
        feat[f"group_{g}_count"] = group_hits[g]

    # structural / numeric features
    for name, pat in DEGREE_PATTERNS.items():
        feat[name] = lower.apply(lambda t: 1 if pat.search(t) else 0).to_numpy()

    def extract_years(t):
        m = YEARS_EXP_RE.search(t)
        return int(m.group(1)) if m else 0

    feat["years_experience_required"] = lower.apply(extract_years).to_numpy()
    feat["description_length"] = descriptions.str.len().to_numpy()
    feat["word_count"] = descriptions.str.split().apply(len).to_numpy()
    return feat


def main():
    df = pd.read_csv("cleaned_jobs.csv")
    df = df.dropna(subset=["description", "role_category"])
    print(f"rows for modeling: {len(df)}")
    print(df["role_category"].value_counts())

    X_text = df["description"].astype(str)
    y_raw = df["role_category"].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()
    print("classes:", class_names)

    # engineered (non-TFIDF) features
    struct_feats = build_skill_features(X_text)
    struct_cols = struct_feats.columns.tolist()

    X_train_text, X_test_text, y_train, y_test, struct_train, struct_test = train_test_split(
        X_text, y, struct_feats, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.6,
        sublinear_tf=True,
        stop_words="english",
    )
    Xtr_tfidf = tfidf.fit_transform(X_train_text)
    Xte_tfidf = tfidf.transform(X_test_text)

    scaler = StandardScaler()
    struct_train_scaled = scaler.fit_transform(struct_train.values)
    struct_test_scaled = scaler.transform(struct_test.values)

    Xtr = sparse.hstack([Xtr_tfidf, sparse.csr_matrix(struct_train_scaled)]).tocsr()
    Xte = sparse.hstack([Xte_tfidf, sparse.csr_matrix(struct_test_scaled)]).tocsr()

    print(f"combined feature matrix: train={Xtr.shape}, test={Xte.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE
        ),
        "LinearSVM": LinearSVC(class_weight="balanced", C=0.5, random_state=RANDOM_STATE, max_iter=5000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=30, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=1
        ),
    }

    cv_results = {}
    for name, clf in candidates.items():
        # MultinomialNB needs non-negative features -> skip struct features that can be negative (scaled)
        if name == "MultinomialNB":
            scores = cross_val_score(clf, Xtr_tfidf, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        else:
            scores = cross_val_score(clf, Xtr, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        cv_results[name] = {"mean_macro_f1": float(scores.mean()), "std_macro_f1": float(scores.std()), "folds": scores.tolist()}
        print(f"{name}: CV macro-F1 = {scores.mean():.4f} +/- {scores.std():.4f}  folds={np.round(scores,4).tolist()}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_macro_f1"])
    print(f"\nBest by CV macro-F1: {best_name}")

    best_clf = candidates[best_name]
    if best_name == "MultinomialNB":
        best_clf.fit(Xtr_tfidf, y_train)
        y_pred = best_clf.predict(Xte_tfidf)
        y_score = best_clf.predict_proba(Xte_tfidf)
    else:
        best_clf.fit(Xtr, y_train)
        y_pred = best_clf.predict(Xte)
        if hasattr(best_clf, "predict_proba"):
            y_score = best_clf.predict_proba(Xte)
        else:
            y_score = best_clf.decision_function(Xte)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    try:
        auc_macro = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(y_test_bin, y_score, average="weighted", multi_class="ovr")
    except Exception as e:
        auc_macro = auc_weighted = None
        print("AUC computation failed:", e)

    print(f"\n=== HELD-OUT TEST RESULTS ({best_name}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    if auc_macro is not None:
        print(f"Macro ROC-AUC (OvR): {auc_macro:.4f}")
        print(f"Weighted ROC-AUC (OvR): {auc_weighted:.4f}")
    print("\nPer-class report:\n", report)
    print("Confusion matrix (rows=true, cols=pred), order:", class_names)
    print(cm)

    # also fit logistic regression specifically for interpretability (top terms per class)
    # even if it wasn't the CV winner, useful for the article's "what predicts what" section
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE)
    logreg.fit(Xtr, y_train)
    feat_names = np.array(tfidf.get_feature_names_out().tolist() + struct_cols)
    top_terms = {}
    for i, cname in enumerate(class_names):
        coefs = logreg.coef_[i]
        top_idx = np.argsort(coefs)[-15:][::-1]
        top_terms[cname] = [(feat_names[j], float(coefs[j])) for j in top_idx]

    # logreg standalone test performance too, for comparison table
    logreg_pred = logreg.predict(Xte)
    logreg_proba = logreg.predict_proba(Xte)
    logreg_acc = accuracy_score(y_test, logreg_pred)
    logreg_macro_f1 = f1_score(y_test, logreg_pred, average="macro")
    logreg_auc = roc_auc_score(y_test_bin, logreg_proba, average="macro", multi_class="ovr")

    results = {
        "n_rows_modeled": int(len(df)),
        "class_names": class_names,
        "class_counts": y_raw.value_counts().to_dict(),
        "feature_matrix_shape_train": list(Xtr.shape),
        "feature_matrix_shape_test": list(Xte.shape),
        "n_tfidf_features": int(Xtr_tfidf.shape[1]),
        "n_engineered_features": len(struct_cols),
        "engineered_feature_names": struct_cols,
        "cv_results": cv_results,
        "best_model_by_cv": best_name,
        "test_results": {
            "model": best_name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_roc_auc_ovr": auc_macro,
            "weighted_roc_auc_ovr": auc_weighted,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "logreg_test_results": {
            "accuracy": logreg_acc,
            "macro_f1": logreg_macro_f1,
            "macro_roc_auc_ovr": logreg_auc,
        },
        "top_terms_per_class_logreg": top_terms,
    }

    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved model_results.json")


if __name__ == "__main__":
    main()
