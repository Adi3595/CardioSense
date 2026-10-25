"""
Data loader module for CardioSense.
Handles downloading, checksum calculation, schema parsing, target encoding,
and stratified train-test splitting.
"""

import hashlib
import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_FILE,
    RAW_METADATA_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    UCI_URL,
    RAW_COLUMN_NAMES,
    FEATURE_NAMES,
    TARGET_RAW_NAME,
    TARGET_NAME,
    RANDOM_STATE,
    TEST_SIZE,
    DATASET_SOURCE_INFO
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


def fetch_raw_data(force_download: bool = False) -> Path:
    """
    Download raw Cleveland dataset from UCI Machine Learning Repository
    if not already present or forced.
    """
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if RAW_DATA_FILE.exists() and not force_download:
        logger.info(f"Raw data file already exists at {RAW_DATA_FILE}")
        return RAW_DATA_FILE

    logger.info(f"Downloading raw data from UCI: {UCI_URL}")
    try:
        import httpx
        with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=15.0, follow_redirects=True) as client:
            resp = client.get(UCI_URL)
            resp.raise_for_status()
            content = resp.content
            with open(RAW_DATA_FILE, "wb") as f:
                f.write(content)
        logger.info(f"Successfully downloaded {len(content)} bytes to {RAW_DATA_FILE}")
    except Exception as e:
        logger.error(f"Failed downloading from UCI URL: {e}")
        raise RuntimeError(f"Could not download dataset from {UCI_URL}: {e}")

    # Compute checksum and metadata
    checksum = compute_sha256(RAW_DATA_FILE)
    metadata = {
        "source": DATASET_SOURCE_INFO["name"],
        "url": UCI_URL,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": checksum,
        "file_size_bytes": RAW_DATA_FILE.stat().st_size
    }
    with open(RAW_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved raw metadata with SHA256 {checksum} to {RAW_METADATA_FILE}")

    return RAW_DATA_FILE


def load_raw_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse the raw Cleveland dataset into a pandas DataFrame.
    Converts '?' and empty strings to NaN.
    """
    path = file_path or RAW_DATA_FILE
    if not path.exists():
        fetch_raw_data()

    logger.info(f"Reading raw dataset from {path}")
    df = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMN_NAMES,
        na_values=["?", "NA", "nan", ""],
        skipinitialspace=True
    )
    logger.info(f"Loaded raw dataset with shape: {df.shape}")
    return df


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and transforms the raw DataFrame:
    - Encodes raw diagnosis 'num' (0 = absence, 1-4 = presence) -> 'target' (0 or 1).
    - Removes raw 'num' column to prevent any target leakage.
    - Ensures proper numeric dtypes for numeric features.
    """
    df = df.copy()
    
    if TARGET_RAW_NAME not in df.columns and TARGET_NAME not in df.columns:
        raise ValueError(f"Neither '{TARGET_RAW_NAME}' nor '{TARGET_NAME}' found in dataframe columns: {df.columns.tolist()}")

    if TARGET_RAW_NAME in df.columns:
        # Documented conversion: 0 -> 0 (absence), >= 1 -> 1 (presence)
        df[TARGET_NAME] = (df[TARGET_RAW_NAME] > 0).astype(int)
        df.drop(columns=[TARGET_RAW_NAME], inplace=True)
        logger.info("Transformed target 'num' (0-4) to binary 'target' (0: absence, 1: presence).")
    elif TARGET_NAME in df.columns:
        # Check if already binary
        unique_targets = set(df[TARGET_NAME].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            df[TARGET_NAME] = (df[TARGET_NAME] > 0).astype(int)

    # Convert all feature columns to float64 so missing values (NaN) can be retained for imputation
    for col in FEATURE_NAMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_processed_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Saves full processed dataset, and performs a stratified train-test split (80/20).
    Saves train.csv and test.csv.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save full cleaned dataset
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    logger.info(f"Saved processed dataset to {PROCESSED_DATA_FILE}")

    # Stratified Train-Test Split
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)

    logger.info(f"Saved stratified train set ({len(train_df)} rows) to {TRAIN_DATA_FILE}")
    logger.info(f"Saved stratified test set ({len(test_df)} rows) to {TEST_DATA_FILE}")

    return df, train_df, test_df


def prepare_data_pipeline(force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Complete acquisition and splitting pipeline."""
    fetch_raw_data(force_download=force_download)
    raw_df = load_raw_dataset()
    processed_df = process_dataset(raw_df)
    return save_processed_data(processed_df)


if __name__ == "__main__":
    prepare_data_pipeline()
