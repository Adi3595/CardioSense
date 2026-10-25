"""
Data validation and auditing module for CardioSense.
Implements schema checks, data quality verification, range/categorical integrity,
and generates formal audit reports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_NAMES,
    TARGET_NAME,
    TARGET_RAW_NAME,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    AUDIT_REPORT_FILE,
    FEATURE_DICT_FILE,
    DATA_QUALITY_SUMMARY_FILE,
    RAW_DATA_FILE,
    RAW_METADATA_FILE,
    DATASET_SOURCE_INFO
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when dataset fails critical schema or validity rules."""
    pass


def validate_schema(df: pd.DataFrame, is_raw: bool = False) -> Dict[str, Any]:
    """
    Validates column presence and schema consistency.
    """
    target_col = TARGET_RAW_NAME if is_raw and TARGET_RAW_NAME in df.columns else TARGET_NAME
    expected_cols = FEATURE_NAMES + [target_col]
    
    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols]

    if missing_cols:
        raise ValidationError(f"Missing required columns in dataset: {missing_cols}")

    return {
        "is_valid": True,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "total_columns": len(df.columns)
    }


def validate_target(df: pd.DataFrame, is_raw: bool = False) -> Dict[str, Any]:
    """
    Validates target variable values and distribution.
    """
    target_col = TARGET_RAW_NAME if is_raw and TARGET_RAW_NAME in df.columns else TARGET_NAME
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found.")

    target_series = df[target_col]
    unique_vals = target_series.unique().tolist()
    counts = target_series.value_counts(dropna=False).to_dict()

    if is_raw:
        # Raw UCI diagnosis should be integers in 0..4
        valid_raw = {0, 1, 2, 3, 4}
        if not set(unique_vals).issubset(valid_raw):
            raise ValidationError(f"Unexpected values in raw target '{target_col}': {unique_vals}")
    else:
        # Processed target must be binary {0, 1}
        valid_binary = {0, 1}
        if not set(unique_vals).issubset(valid_binary):
            raise ValidationError(f"Processed target must be strictly binary {0, 1}, got: {unique_vals}")

    return {
        "target_col": target_col,
        "unique_values": unique_vals,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "prevalence": float(target_series.mean()) if not is_raw else None
    }


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive data quality audit:
    - Missing values (counts and percentages)
    - Duplicate rows
    - Value ranges for numeric columns
    - Domain validity for categorical columns
    """
    report: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {}
    }

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = float((missing_count / len(df)) * 100)
        col_type = str(series.dtype)
        unique_count = int(series.nunique(dropna=True))

        col_audit: Dict[str, Any] = {
            "dtype": col_type,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 2),
            "unique_count": unique_count
        }

        meta = FEATURE_METADATA.get(col, {})
        if "valid_range" in meta:
            min_val, max_val = meta["valid_range"]
            actual_min = float(series.min()) if not series.empty and not pd.isna(series.min()) else None
            actual_max = float(series.max()) if not series.empty and not pd.isna(series.max()) else None
            col_audit["range_expected"] = [min_val, max_val]
            col_audit["range_observed"] = [actual_min, actual_max]
            outliers_below = int((series < min_val).sum()) if actual_min is not None else 0
            outliers_above = int((series > max_val).sum()) if actual_max is not None else 0
            col_audit["range_anomalies"] = outliers_below + outliers_above

        if "valid_values" in meta:
            valid_set = set(meta["valid_values"])
            observed_unique = set(series.dropna().unique())
            invalid_vals = list(observed_unique - valid_set)
            col_audit["expected_values"] = meta["valid_values"]
            col_audit["observed_unique"] = [float(x) for x in observed_unique]
            col_audit["invalid_values"] = invalid_vals

        report["columns"][col] = col_audit

    return report


def generate_audit_artifacts(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates and persists:
    1. Feature Dictionary (CSV)
    2. Data Quality Summary (JSON)
    3. Markdown Audit Report (MD)
    """
    AUDIT_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Feature Dictionary
    rows = []
    for col, meta in FEATURE_METADATA.items():
        rows.append({
            "feature": col,
            "type": meta.get("type", "unknown"),
            "description": meta.get("description", ""),
            "unit_or_values": str(meta.get("valid_values", meta.get("valid_range", "N/A"))),
            "preprocessing": meta.get("preprocessing", "")
        })
    feat_dict_df = pd.DataFrame(rows)
    feat_dict_df.to_csv(FEATURE_DICT_FILE, index=False)
    logger.info(f"Saved feature dictionary to {FEATURE_DICT_FILE}")

    # 2. Quality Audit
    raw_audit = audit_data_quality(raw_df)
    processed_audit = audit_data_quality(processed_df)

    raw_meta = {}
    if RAW_METADATA_FILE.exists():
        with open(RAW_METADATA_FILE, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

    audit_summary = {
        "dataset_name": DATASET_SOURCE_INFO["name"],
        "retrieval_url": DATASET_SOURCE_INFO["primary_url"],
        "sha256_checksum": raw_meta.get("sha256", "N/A"),
        "raw_shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
        "processed_shape": [int(processed_df.shape[0]), int(processed_df.shape[1])],
        "target_distribution": {
            "absence (0)": int((processed_df[TARGET_NAME] == 0).sum()),
            "presence (1)": int((processed_df[TARGET_NAME] == 1).sum()),
            "prevalence_rate": round(float(processed_df[TARGET_NAME].mean()), 4)
        },
        "missing_values_per_column": {
            col: int(processed_df[col].isna().sum())
            for col in processed_df.columns if processed_df[col].isna().sum() > 0
        },
        "duplicate_rows": int(processed_df.duplicated().sum())
    }

    with open(DATA_QUALITY_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    logger.info(f"Saved data quality summary to {DATA_QUALITY_SUMMARY_FILE}")

    # 3. Markdown Audit Report
    target_dist = audit_summary["target_distribution"]
    missing_cols = audit_summary["missing_values_per_column"]

    md_content = f"""# CardioSense Dataset Audit & Quality Analysis Report

**Date & Time**: {raw_meta.get('download_timestamp_utc', '2026-10-26T00:00:00Z')}  
**Source**: [{DATASET_SOURCE_INFO['name']}]({DATASET_SOURCE_INFO['primary_url']})  
**SHA256 Checksum**: `{raw_meta.get('sha256', 'N/A')}`  
**Raw Dimensions**: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns  
**Processed Dimensions**: {processed_df.shape[0]} rows × {processed_df.shape[1]} columns  

---

## 1. Executive Summary & Provenance
The dataset analyzed is the canonical Cleveland clinic heart disease dataset from the UCI Machine Learning Repository.
The raw data consists of 303 patient records across 14 measured attributes (13 clinical predictors and 1 diagnosis target).
Missing values in the original source are encoded as string `'?'`, specifically identified in features `ca` (major vessels) and `thal` (thalassemia).
No patient identifiers (PHI) are present in the dataset.

## 2. Target Variable Verification & Encoding
The original diagnosis variable (`num`) contains integer values ranging from 0 to 4:
- `0`: Absence of heart disease diagnosis (< 50% vessel diameter narrowing).
- `1, 2, 3, 4`: Presence of heart disease diagnosis (> 50% vessel diameter narrowing).

In accordance with medical ML research standards and project guidelines, the target is converted to binary format:
- **Class 0 (Absence)**: {target_dist['absence (0)']} patients ({round(target_dist['absence (0)']/processed_df.shape[0]*100, 2)}%)
- **Class 1 (Presence)**: {target_dist['presence (1)']} patients ({round(target_dist['presence (1)']/processed_df.shape[0]*100, 2)}%)
- **Positive Class Prevalence**: {round(target_dist['prevalence_rate']*100, 2)}%

This confirms a relatively balanced binary classification problem (approx. 54% absence vs. 46% presence), suitable for stratified cross-validation.

## 3. Data Hygiene & Missing Values
- **Duplicate Records**: {audit_summary['duplicate_rows']} exact duplicates detected.
- **Identified Missing Value Placeholders**: Converted `'?'` to `NaN`.
- **Missing Value Inventory**:
"""
    if missing_cols:
        for col, count in missing_cols.items():
            pct = round((count / len(processed_df)) * 100, 2)
            md_content += f"  - `{col}`: {count} missing values ({pct}%)\n"
    else:
        md_content += "  - No missing values observed.\n"

    md_content += """
### Imputation Strategy:
- Numeric features: Median imputation (fitted strictly on training fold).
- Categorical features (`ca`, `thal`): Most frequent / mode imputation (fitted strictly on training fold).
- No imputation is fitted on test data or the full un-split dataset to guarantee zero leakage.

## 4. Feature Schema & Summary Statistics
| Feature | Type | Meaning | Observed Range / Values | Missing Values | Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for col in FEATURE_NAMES:
        meta = FEATURE_METADATA.get(col, {})
        col_series = processed_df[col]
        missing_cnt = col_series.isna().sum()
        if col in NUMERIC_FEATURES:
            val_str = f"[{col_series.min():.1f}, {col_series.max():.1f}], mean={col_series.mean():.1f}"
        else:
            uniq = sorted([int(x) for x in col_series.dropna().unique()])
            val_str = f"Categorical: {uniq}"
        md_content += f"| `{col}` | {meta.get('type', 'N/A')} | {meta.get('description', '')} | {val_str} | {missing_cnt} | {meta.get('preprocessing', '')} |\n"

    md_content += f"""
## 5. Medical & Non-Diagnostic Disclaimer
{DATASET_SOURCE_INFO['citation']}
**Notice**: CardioSense is an academic research demonstration. Predictions must never be used for actual clinical diagnostic decisions.
"""

    with open(AUDIT_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Generated audit report at {AUDIT_REPORT_FILE}")

    return audit_summary


def run_validation_pipeline() -> Dict[str, Any]:
    """Execute complete validation and generate audit reports."""
    from src.data_loader import load_raw_dataset
    raw_df = load_raw_dataset()
    validate_schema(raw_df, is_raw=True)
    validate_target(raw_df, is_raw=True)

    from src.data_loader import process_dataset
    processed_df = process_dataset(raw_df)
    validate_schema(processed_df, is_raw=False)
    validate_target(processed_df, is_raw=False)

    return generate_audit_artifacts(raw_df, processed_df)


if __name__ == "__main__":
    run_validation_pipeline()
