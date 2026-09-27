"""
train_model.py
---------------
Trains the complaint classification model.

Pipeline:
    Raw complaint text
        -> preprocessing.clean_text
        -> TfidfVectorizer
        -> LogisticRegression

This script:
1. Loads data/complaints.csv
2. Cleans the text
3. Splits into train/test sets (stratified, so every category is
   represented proportionally in both sets)
4. Trains a TF-IDF + Logistic Regression pipeline (the main model)
5. Optionally trains Multinomial Naive Bayes and Linear SVM for
   comparison (reported in model/metrics.json, but NOT used by the app)
6. Evaluates the main model on the held-out test set and prints/saves
   real metrics -- nothing here is hardcoded or fabricated
7. Saves the trained pipeline to model/complaint_classifier.pkl
8. Saves a confusion matrix plot to evaluation/confusion_matrix.png
9. Saves metrics.json for the Streamlit "Model Performance" page

Run with:
    python src/train_model.py
"""

import os
import json
import sys

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save the figure to disk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Make sure we can import sibling modules (preprocessing.py) regardless
# of the current working directory the script is run from.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text  # noqa: E402

# ----------------------------------------------------------------------
# Paths (all relative to this file, so the project works after moving)
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "complaints.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
MODEL_PATH = os.path.join(MODEL_DIR, "complaint_classifier.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_DIR, "confusion_matrix.png")

RANDOM_STATE = 42


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Run 'python src/generate_dataset.py' first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["complaint_text", "category"])
    df["complaint_text_clean"] = df["complaint_text"].apply(clean_text)
    return df


def build_main_pipeline():
    """TF-IDF + Logistic Regression -- the primary, production model."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams capture short phrases
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(f"\n--- {model_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
    }, y_pred


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_data()
    print(f"Loaded {len(df)} complaints across {df['category'].nunique()} categories.")

    X = df["complaint_text_clean"]
    y = df["category"]

    # Stratified split (as required): keeps class proportions similar
    # in train & test. NOTE: because this is a synthetic, template-based
    # dataset, a random/stratified split can place near-duplicate
    # sentences (same base template, different filler words) on both
    # sides of the split. This tends to push accuracy very high. We
    # also compute a stricter "group split" (train/test never share a
    # base template) purely for reporting, so the Model Performance
    # page can show both numbers honestly instead of hiding this
    # known limitation of synthetic data. The model that is actually
    # SAVED and used by the app is trained via the required stratified
    # split below.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    group_eval_metrics = None
    if "template_id" in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        g_train_idx, g_test_idx = next(splitter.split(X, y, groups=df["template_id"]))
        gX_train, gX_test = X.iloc[g_train_idx], X.iloc[g_test_idx]
        gy_train, gy_test = y.iloc[g_train_idx], y.iloc[g_test_idx]
        group_pipeline = build_main_pipeline()
        group_pipeline.fit(gX_train, gy_train)
        group_eval_metrics, _ = evaluate(
            group_pipeline, gX_test, gy_test,
            "Logistic Regression (STRICTER template-holdout split, reporting only)"
        )

    # ------------------------------------------------------------
    # Main model: TF-IDF + Logistic Regression
    # ------------------------------------------------------------
    print("\nTraining primary model (TF-IDF + Logistic Regression)...")
    main_pipeline = build_main_pipeline()
    main_pipeline.fit(X_train, y_train)
    main_metrics, y_pred = evaluate(main_pipeline, X_test, y_test, "Logistic Regression (primary)")

    # ------------------------------------------------------------
    # Optional comparison models (reported only, not used by the app)
    # ------------------------------------------------------------
    comparison_results = {}

    print("\nTraining comparison model (Multinomial Naive Bayes)...")
    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)),
        ("clf", MultinomialNB()),
    ])
    nb_pipeline.fit(X_train, y_train)
    nb_metrics, _ = evaluate(nb_pipeline, X_test, y_test, "Multinomial Naive Bayes (comparison)")
    comparison_results["Multinomial Naive Bayes"] = {
        "accuracy": nb_metrics["accuracy"],
        "precision": nb_metrics["precision"],
        "recall": nb_metrics["recall"],
        "f1_score": nb_metrics["f1_score"],
    }

    print("\nTraining comparison model (Linear SVM)...")
    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)),
        ("clf", LinearSVC(random_state=RANDOM_STATE)),
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_metrics, _ = evaluate(svm_pipeline, X_test, y_test, "Linear SVM (comparison)")
    comparison_results["Linear SVM"] = {
        "accuracy": svm_metrics["accuracy"],
        "precision": svm_metrics["precision"],
        "recall": svm_metrics["recall"],
        "f1_score": svm_metrics["f1_score"],
    }

    comparison_results["Logistic Regression (used by app)"] = {
        "accuracy": main_metrics["accuracy"],
        "precision": main_metrics["precision"],
        "recall": main_metrics["recall"],
        "f1_score": main_metrics["f1_score"],
    }

    # ------------------------------------------------------------
    # Confusion matrix (for the primary model)
    # ------------------------------------------------------------
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Logistic Regression (Complaint Category Classifier)")
    plt.ylabel("Actual Category")
    plt.xlabel("Predicted Category")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # ------------------------------------------------------------
    # Save the trained pipeline (this is what app.py loads)
    # ------------------------------------------------------------
    joblib.dump(main_pipeline, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

    # ------------------------------------------------------------
    # Save metrics.json for the Streamlit "Model Performance" page
    # ------------------------------------------------------------
    metrics_out = {
        "accuracy": main_metrics["accuracy"],
        "precision": main_metrics["precision"],
        "recall": main_metrics["recall"],
        "f1_score": main_metrics["f1_score"],
        "classification_report": main_metrics["classification_report"],
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_total_samples": len(df),
        "categories": labels,
        "model_comparison": comparison_results,
        "stricter_template_holdout_eval": (
            {
                "accuracy": group_eval_metrics["accuracy"],
                "precision": group_eval_metrics["precision"],
                "recall": group_eval_metrics["recall"],
                "f1_score": group_eval_metrics["f1_score"],
                "note": (
                    "This score comes from a stricter evaluation where the "
                    "test set uses complaint templates never seen in training "
                    "(not just different filler words). It is reported for "
                    "transparency and is lower than the main stratified-split "
                    "score because the synthetic dataset is template-based."
                ),
            }
            if group_eval_metrics else None
        ),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
