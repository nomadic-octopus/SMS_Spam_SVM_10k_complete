"""Fast offline smoke test of CSV, saved model, and CLI prediction logic.

Run: python test_system.py
"""
from __future__ import annotations

import json

import pandas as pd

from predict import load_pipeline, predict_sms
from train import DATA_PATH, METADATA_PATH, MODEL_PATH, file_sha256, load_and_split


def main() -> None:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert len(df) == 10000, "Dataset must have 10,000 rows"
    assert df.v1.value_counts().to_dict() == {"spam": 5000, "ham": 5000}
    assert set(df.source) == {"synthetic", "original"}
    assert not df[["v1", "v2", "source", "template_family"]].isna().any().any()
    train_df, test_df = load_and_split()
    assert len(train_df) + len(test_df) == len(df)
    assert set(test_df.source) == {"original"}
    assert set(train_df.v2).isdisjoint(set(test_df.v2))

    model = load_pipeline()
    assert MODEL_PATH.is_file() and METADATA_PATH.is_file()
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["csv_sha256"] == file_sha256(DATA_PATH)

    spam = "URGENT! You have won £500. Claim your reward now."
    ham = "Hi, can you email me the meeting notes tomorrow?"
    print("Spam example:", predict_sms(spam, model))
    print("Ham example: ", predict_sms(ham, model))
    assert predict_sms(spam, model) == "Spam"
    assert predict_sms(ham, model) == "Ham"
    print("\nPASS: 10,000-row data validation, real-only holdout, model metadata and predictions.")


if __name__ == "__main__":
    main()
