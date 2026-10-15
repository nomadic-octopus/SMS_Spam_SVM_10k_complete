"""Train an SMS-spam SVM on the 10k dataset with honest real-message testing.

Usage (from any working directory): python train.py
Synthetic messages are included in training but NEVER in the held-out test set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "sms_spam_pipeline.pkl"
METADATA_PATH = BASE_DIR / "model_info.json"
SEED = 42


def clean_text(text: str) -> str:
    """Small reusable text cleanup; TF-IDF handles word tokenization itself.

    Preserves digits and URLs because these may help recognize scams.
    Inference can pass raw text directly to the saved sklearn pipeline.
    """
    return " ".join(str(text).split())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and test; test has ONLY untouched real SMS messages."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run 'python download_data.py' to recreate it."
        )
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required = {"v1", "v2", "source", "template_family"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"spam.csv must include columns: {sorted(required)}. "
            "Run 'python build_dataset.py' to rebuild."
        )
    if len(df) != 10000 or df["v1"].value_counts().to_dict() != {"ham": 5000, "spam": 5000}:
        raise ValueError("Expected exactly 5,000 ham and 5,000 spam rows in spam.csv")
    if df[list(required)].isnull().any().any():
        raise ValueError("Dataset contains missing values")
    if not df["source"].isin(["original", "synthetic"]).all():
        raise ValueError("Unexpected source in spam.csv")
    if df["v2"].astype(str).str.strip().eq("").any():
        raise ValueError("Dataset contains empty SMS messages")
    # Do not allow the same SMS message in both train and held-out evaluation.
    normalized = df["v2"].astype(str).str.normalize("NFKC").str.casefold().str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    if normalized.duplicated().any():
        raise ValueError("Dataset contains duplicate SMS messages; regenerate dataset")

    genuine = df.loc[df["source"] == "original"].copy()
    synthetic = df.loc[df["source"] == "synthetic"].copy()
    if genuine.empty or synthetic.empty:
        raise ValueError("Both original and synthetic samples must be present")
    real_train, real_test = train_test_split(
        genuine, test_size=0.20, random_state=SEED, stratify=genuine["v1"]
    )
    train_df = pd.concat([real_train, synthetic], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = real_test.reset_index(drop=True)
    assert (test_df["source"] == "original").all()
    assert set(train_df["v2"]).isdisjoint(set(test_df["v2"]))
    return train_df, test_df


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            strip_accents="unicode", lowercase=True,
            ngram_range=(1, 2), max_features=12000, sublinear_tf=True,
            min_df=2,
        )),
        ("svm", LinearSVC(class_weight="balanced", random_state=SEED,
                          max_iter=5000)),
    ])


def train_model() -> Pipeline:
    train_df, test_df = load_and_split()
    print("=== SMS Spam Detection - 10,000 Samples (SVM) ===")
    print("Entire CSV class counts:", pd.read_csv(DATA_PATH)["v1"].value_counts().to_dict())
    print("Training rows:", len(train_df), "| by label:", train_df["v1"].value_counts().to_dict())
    print("Held-out REAL SMS rows:", len(test_df), "| by label:", test_df["v1"].value_counts().to_dict())
    print("Testing uses original messages only; no synthetic examples in test set.")

    # Keep template siblings in the same CV fold to reduce inflated scores.
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"svm__C": [0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=folds,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    grid.fit(train_df["v2"], train_df["v1"], groups=train_df["template_family"])
    model = grid.best_estimator_
    predictions = model.predict(test_df["v2"])
    truth = test_df["v1"]
    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision_spam": float(precision_score(truth, predictions, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(truth, predictions, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(truth, predictions, pos_label="spam", zero_division=0)),
        "f1_macro": float(f1_score(truth, predictions, average="macro", zero_division=0)),
    }
    cm = confusion_matrix(truth, predictions, labels=["ham", "spam"])
    print("\nBest SVM parameter:", grid.best_params_)
    print("Five-fold GROUPED CV macro-F1:", round(float(grid.best_score_), 4))
    print("\n== Held-out ORIGINAL SMS test results ==")
    for name, value in metrics.items():
        print(f"{name:17s}: {value:.4f}")
    print("Confusion matrix (rows=true, columns=pred; ham then spam):")
    print(cm)
    print(classification_report(truth, predictions, labels=["ham", "spam"], zero_division=0))

    info = {
        "csv_sha256": file_sha256(DATA_PATH),
        "scikit_learn_version": sklearn.__version__,
        "dataset_count": 10000,
        "dataset_class_counts": {"ham": 5000, "spam": 5000},
        "train_count": len(train_df),
        "real_test_count": len(test_df),
        "real_test_class_counts": {k: int(v) for k, v in truth.value_counts().items()},
        "train_source_counts": {k: int(v) for k, v in train_df["source"].value_counts().items()},
        "best_params": grid.best_params_,
        "cv_grouped_macro_f1": float(grid.best_score_),
        "test_real_metrics": metrics,
        "test_confusion_matrix_ham_spam": cm.tolist(),
        "evaluation_note": "Test set contains held-out original SMS only; synthetic messages are used for training only.",
    }
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print("Saved model:", MODEL_PATH)
    print("Saved evaluation and dataset fingerprint:", METADATA_PATH)
    return model


if __name__ == "__main__":
    train_model()
