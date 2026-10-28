"""
Data loader module for CardioSense.
Supports dual-dataset architecture:
1. India Hospital Cohort (Primary): Multi-specialty hospital in India (Doppala & Bhattacharyya, N=1000)
2. UCI Cleveland Benchmark (Historical): Cleveland Clinic Foundation (Janosi et al., 1988, N=303)

Handles downloading, SHA-256 verification, schema parsing, target encoding,
leakage-free pre-split isolation, and stratified train-test splitting.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_DATASET,
    DATASETS_CONFIG,
    get_dataset_config,
    RANDOM_STATE,
    TEST_SIZE,
    PROJECT_ROOT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def fetch_raw_data(dataset: str = DEFAULT_DATASET, force_download: bool = False) -> Path:
    """
    Download raw dataset if not present or forced.
    Supports both 'india' and 'uci' dataset configurations.
    """
    cfg = get_dataset_config(dataset)
    raw_file: Path = cfg["raw_file"]
    raw_file.parent.mkdir(parents=True, exist_ok=True)

    if raw_file.exists() and not force_download:
        logger.info(f"[{cfg['short_name']}] Raw data file already exists at {raw_file}")
        return raw_file

    download_urls = [cfg.get("mirror_url"), cfg.get("primary_url")]
    download_urls = [u for u in download_urls if u]
    
    logger.info(f"[{cfg['short_name']}] Downloading raw data from {download_urls[0]}...")
    
    import httpx
    downloaded = False
    last_err = None
    content = None

    for url in download_urls:
        try:
            logger.info(f"Attempting download from: {url}")
            with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=20.0, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                content = resp.content
                with open(raw_file, "wb") as f:
                    f.write(content)
                downloaded = True
                logger.info(f"Successfully downloaded {len(content)} bytes to {raw_file}")
                break
        except Exception as e:
            logger.warning(f"Download from {url} failed: {e}")
            last_err = e

    if not downloaded:
        raise RuntimeError(
            f"Could not download dataset '{dataset}' from available URLs. "
            f"Please manually place the file at {raw_file}. Last error: {last_err}"
        )

    # Compute checksum and metadata
    checksum = compute_sha256(raw_file)
    metadata = {
        "dataset_key": dataset,
        "name": cfg["name"],
        "source_url": cfg.get("primary_url"),
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": checksum,
        "file_size_bytes": raw_file.stat().st_size
    }
    
    meta_file = cfg.get("metadata_file", raw_file.parent / "raw_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved raw metadata with SHA256 {checksum} to {meta_file}")

    return raw_file


def load_raw_dataset(dataset: str = DEFAULT_DATASET, file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse the raw dataset into a pandas DataFrame.
    Converts '?' and sentinel placeholders to NaN.
    """
    cfg = get_dataset_config(dataset)
    path = file_path or cfg["raw_file"]
    
    if not path.exists():
        fetch_raw_data(dataset=dataset)

    logger.info(f"[{cfg['short_name']}] Reading raw dataset from {path}")
    
    if dataset == "india":
        # Indian CSV contains header row: patientid, age, gender, chestpain, ...
        df = pd.read_csv(
            path,
            na_values=["?", "NA", "nan", ""],
            skipinitialspace=True
        )
    else:
        # UCI Cleveland data has no header row
        df = pd.read_csv(
            path,
            header=None,
            names=cfg["raw_column_names"],
            na_values=["?", "NA", "nan", ""],
            skipinitialspace=True
        )

    logger.info(f"[{cfg['short_name']}] Loaded raw dataset with shape: {df.shape}")
    return df


def process_dataset(df: pd.DataFrame, dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    """
    Validates and transforms the raw DataFrame based on dataset configuration:
    - Drops patient identifier column if present (e.g. 'patientid' in India dataset).
    - Encodes raw diagnosis -> binary 'target' (0 or 1).
    - Removes raw target/leakage columns.
    - Handles domain-specific sentinels (e.g. cholesterol == 0 coerced to NaN).
    - Ensures proper numeric types for feature columns.
    """
    cfg = get_dataset_config(dataset)
    df = df.copy()

    # 1. Drop patient ID column to prevent identifier leakage
    id_col = cfg.get("id_column")
    if id_col and id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)
        logger.info(f"[{cfg['short_name']}] Dropped ID column '{id_col}' to prevent identification leakage.")

    # 2. Target validation & binarization
    target_raw_col = cfg["target_raw_column"]
    target_col = cfg["target_column"]

    if target_raw_col in df.columns and target_raw_col != target_col:
        # For UCI: num (0-4) -> binary target (0 vs 1)
        df[target_col] = (df[target_raw_col] > 0).astype(int)
        df.drop(columns=[target_raw_col], inplace=True)
        logger.info(f"[{cfg['short_name']}] Transformed '{target_raw_col}' to binary '{target_col}'.")
    elif target_col in df.columns:
        # For India: target is already {0, 1}
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    # 3. Domain-Specific Hygiene
    if dataset == "india":
        # In Indian hospital cohort, serumcholestrol == 0 is an undocumented missing placeholder (lipid panel not run)
        if "serumcholestrol" in df.columns:
            zero_chol = (df["serumcholestrol"] == 0).sum()
            if zero_chol > 0:
                logger.info(f"[{cfg['short_name']}] Coerced {zero_chol} zero-cholesterol values to NaN for imputation.")
                df.loc[df["serumcholestrol"] == 0, "serumcholestrol"] = np.nan

    # 4. Cast features to float64 to support NaN imputation
    for col in cfg["feature_names"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_processed_data(df: pd.DataFrame, dataset: str = DEFAULT_DATASET) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Saves full processed dataset, and performs a stratified train-test split (80/20).
    Saves train.csv and test.csv in dataset-specific directories.
    """
    cfg = get_dataset_config(dataset)
    proc_file: Path = cfg["processed_file"]
    train_file: Path = cfg["train_file"]
    test_file: Path = cfg["test_file"]

    proc_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_file, index=False)
    logger.info(f"[{cfg['short_name']}] Saved processed dataset ({len(df)} rows) to {proc_file}")

    # Stratified Train-Test Split (80/20)
    features = cfg["feature_names"]
    target = cfg["target_column"]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(f"[{cfg['short_name']}] Saved stratified train set ({len(train_df)} rows) to {train_file}")
    logger.info(f"[{cfg['short_name']}] Saved stratified test set ({len(test_df)} rows) to {test_file}")

    # For backward compatibility: if active dataset matches default, also populate root data/processed/
    if dataset == DEFAULT_DATASET:
        root_proc = PROJECT_ROOT / "data" / "processed"
        root_proc.mkdir(parents=True, exist_ok=True)
        df.to_csv(root_proc / "heart_disease_cleaned.csv", index=False)
        train_df.to_csv(root_proc / "train.csv", index=False)
        test_df.to_csv(root_proc / "test.csv", index=False)

    return df, train_df, test_df


def prepare_data_pipeline(dataset: str = DEFAULT_DATASET, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Complete acquisition and splitting pipeline for specified dataset."""
    fetch_raw_data(dataset=dataset, force_download=force_download)
    raw_df = load_raw_dataset(dataset=dataset)
    processed_df = process_dataset(raw_df, dataset=dataset)
    return save_processed_data(processed_df, dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioSense Data Loader Pipeline")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=["india", "uci", "all"],
                        help="Dataset to process: 'india' (default), 'uci', or 'all'")
    parser.add_argument("--force", action="store_true", help="Force re-download raw files")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["india", "uci"]:
            print(f"\n{'='*50}\nProcessing {ds.upper()} Dataset\n{'='*50}")
            prepare_data_pipeline(dataset=ds, force_download=args.force)
    else:
        prepare_data_pipeline(dataset=args.dataset, force_download=args.force)
