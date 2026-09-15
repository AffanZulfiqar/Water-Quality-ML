"""
scripts/prepare_data.py
────────────────────────
Data preparation and validation script.

Downloads or verifies the water potability dataset,
performs basic integrity checks, and prints a dataset summary.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --download   # Requires kaggle CLI
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_config, load_raw_dataset, validate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("prepare_data")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and validate the water quality dataset.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--download", action="store_true",
                        help="Download from Kaggle using kaggle CLI (requires kaggle.json)")
    return parser.parse_args()


def download_from_kaggle(data_path: str) -> None:
    """Download the Kaggle Water Potability dataset."""
    import subprocess
    import os
    data_dir = Path(data_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", "adityakadiwal/water-potability",
         "-p", str(data_dir), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"Kaggle download failed:\n{result.stderr}")
        logger.info(
            "\nManual download instructions:\n"
            "1. Go to: https://www.kaggle.com/datasets/adityakadiwal/water-potability\n"
            "2. Click 'Download'\n"
            f"3. Extract and place 'water_potability.csv' in: {data_dir}"
        )
        sys.exit(1)
    logger.info(f"Dataset downloaded to: {data_dir}")


def print_dataset_summary(df: pd.DataFrame, target_col: str) -> None:
    """Print comprehensive dataset summary."""
    print("\n" + "═" * 60)
    print("  DATASET SUMMARY — Water Potability")
    print("═" * 60)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target column: '{target_col}'")

    print("\n── Class Distribution ──────────────────────────────────")
    counts = df[target_col].value_counts().sort_index()
    for val, cnt in counts.items():
        label = "Potable" if val == 1 else "Not Potable"
        pct = 100 * cnt / len(df)
        print(f"  Class {val} ({label:12s}): {cnt:5d} samples  ({pct:.1f}%)")

    print("\n── Missing Values ──────────────────────────────────────")
    missing = df.isnull().sum()
    for col, n in missing.items():
        if n > 0:
            pct = 100 * n / len(df)
            print(f"  {col:25s}: {n:4d} missing  ({pct:.1f}%)")
    if missing.sum() == 0:
        print("  No missing values found.")

    print("\n── Feature Statistics ──────────────────────────────────")
    features = [c for c in df.columns if c != target_col]
    desc = df[features].describe().round(3)
    print(desc.to_string())

    print("\n── Duplicate Rows ──────────────────────────────────────")
    n_dupes = df.duplicated().sum()
    print(f"  Exact duplicates: {n_dupes}")

    print("\n" + "═" * 60 + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_path = cfg["data"]["primary_dataset"]
    target_col = cfg["data"]["target_column"]

    # Download if requested
    if args.download:
        download_from_kaggle(data_path)

    # Load and validate
    df = load_raw_dataset(data_path)
    validate_dataset(df, target_col)
    print_dataset_summary(df, target_col)

    logger.info("Data preparation complete. Dataset is ready for training.")


if __name__ == "__main__":
    main()
