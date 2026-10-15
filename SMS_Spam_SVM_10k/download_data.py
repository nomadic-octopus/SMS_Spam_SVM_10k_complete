"""Create/restore spam.csv, keeping a local copy of original SMS messages.

Bundled SMS.csv works offline. Only missing SMS.csv requires internet access.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from build_dataset import ORIGINAL_CSV, OUTPUT_CSV, build_dataset

ORIGINAL_TSV_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)


def download_dataset(force: bool = False) -> Path:
    if OUTPUT_CSV.exists() and not force:
        print(f"Keeping existing 10,000-message dataset: {OUTPUT_CSV}")
        return OUTPUT_CSV
    if not ORIGINAL_CSV.exists():
        print("Original SMS.csv missing; downloading public SMS Spam Collection...")
        try:
            with urllib.request.urlopen(ORIGINAL_TSV_URL, timeout=30) as result:
                original = pd.read_csv(
                    result, sep="\t", header=None, names=["v1", "v2"],
                    encoding="utf-8"
                )
        except Exception as error:
            raise RuntimeError(
                "Could not download original SMS messages. Restore the bundled "
                "SMS.csv or connect to the internet."
            ) from error
        original.to_csv(ORIGINAL_CSV, index=False, encoding="utf-8")
        print("Downloaded original to:", ORIGINAL_CSV)
    build_dataset(OUTPUT_CSV)
    return OUTPUT_CSV


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore 10k SMS spam dataset")
    parser.add_argument("--force", action="store_true", help="Rebuild spam.csv even if present")
    args = parser.parse_args()
    download_dataset(force=args.force)
