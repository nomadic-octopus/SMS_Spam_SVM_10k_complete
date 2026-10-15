"""Predict SMS spam/ham from VS Code terminal using the trained SVM.

Examples:
  python predict.py "Congratulations, claim your prize now!"
  python predict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import sklearn

from train import DATA_PATH, METADATA_PATH, MODEL_PATH, file_sha256, train_model


def load_pipeline():
    """Retrain automatically when dataset, metadata, or sklearn version changes."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}. Run 'python download_data.py'."
        )
    retrain_reason = None
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        retrain_reason = "Model or metadata missing"
    else:
        try:
            info = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if info.get("csv_sha256") != file_sha256(DATA_PATH):
                retrain_reason = "spam.csv has changed"
            elif info.get("scikit_learn_version") != sklearn.__version__:
                retrain_reason = "scikit-learn version has changed"
        except (OSError, ValueError, KeyError) as error:
            retrain_reason = f"Could not read metadata ({error})"
    if retrain_reason:
        print(f"[INFO] {retrain_reason}; training a compatible SVM model...")
        return train_model()
    try:
        return joblib.load(MODEL_PATH)  # Only load your OWN locally generated pickle.
    except (OSError, ValueError, ImportError, AttributeError) as error:
        print(f"[INFO] Model could not be loaded ({error}); retraining...")
        return train_model()


def predict_sms(text: str, pipeline=None) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a nonempty SMS message")
    model = pipeline if pipeline is not None else load_pipeline()
    return "Spam" if model.predict([text])[0] == "spam" else "Ham"


def main() -> None:
    pipeline = load_pipeline()
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:]).strip()
        print(f"Message: {text}")
        print(f"Prediction: {predict_sms(text, pipeline)}")
        return
    print("\n--- SMS Spam Predictor (type 'exit' to quit) ---")
    examples = [
        "Congratulations! You have won a free iPhone. Click here now.",
        "Hi, are we meeting tomorrow at 3 PM?",
    ]
    for item in examples:
        print(f"Example: {item}\nPrediction: {predict_sms(item, pipeline)}\n")
    while True:
        try:
            text = input("Enter SMS message (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if text.casefold() in {"exit", "quit", "q"}:
            print("Goodbye!")
            return
        if text:
            print(f"--> Prediction: {predict_sms(text, pipeline)}\n")


if __name__ == "__main__":
    main()
