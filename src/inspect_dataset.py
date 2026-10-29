"""Inspect the real NSL-KDD official train/test split from the repository root.

Run: python -m src.inspect_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

from src.preprocessing.nsl_kdd import dataset_summary, load_official_split


def main() -> None:
    """Load both official files and print compact validation summaries."""
    project_root = Path(__file__).resolve().parents[1]
    train, test = load_official_split(project_root / "data" / "raw")

    print("NSL-KDD OFFICIAL SPLIT VALIDATION")
    print("=================================")
    for split_name, frame in (("TRAIN", train), ("TEST", test)):
        print(f"\n{split_name}")
        print("-" * len(split_name))
        print(json.dumps(dataset_summary(frame), indent=2))
        print("\nRaw attack label counts:")
        print(frame["attack"].value_counts().to_string())

    print("\nValidation passed: both files loaded and all attack labels were mapped.")
    print("Note: this command inspects raw data only; it does not train a model.")


if __name__ == "__main__":
    main()
