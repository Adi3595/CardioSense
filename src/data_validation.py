"""
Data validation and auditing module for CardioSense.
Supports dual-dataset validation:
1. India Hospital Cohort (Primary): 1000 rows × 14 columns
2. UCI Cleveland Benchmark (Historical): 303 rows × 14 columns

Implements schema checks, data quality verification, range/categorical integrity,
and generates formal audit reports for both cohorts.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_DATASET,
    get_dataset_config,
    DATASETS_CONFIG,
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
    PROJECT_ROOT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when dataset fails critical schema or validity rules."""
    pass


def validate_schema(df: pd.DataFrame, is_raw: bool = False, dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    """
    Validates column presence and schema consistency for the selected dataset.
    """
    cfg = get_dataset_config(dataset)
    
    if is_raw:
        expected_cols = cfg["raw_column_names"]
    else:
        target_col = cfg["target_column"]
        expected_cols = cfg["feature_names"] + [target_col]
    
    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols]

    if missing_cols:
        raise ValidationError(f"[{cfg['short_name']}] Missing required columns in dataset: {missing_cols}")

    return {
        "is_valid": True,
        "dataset": dataset,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "total_columns": len(df.columns)
    }


def validate_target(df: pd.DataFrame, is_raw: bool = False, dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    """
    Validates target variable values and distribution for the selected dataset.
    """
    cfg = get_dataset_config(dataset)
    target_col = cfg["target_raw_column"] if (is_raw and cfg["target_raw_column"] in df.columns) else cfg["target_column"]
    
    if target_col not in df.columns:
        raise ValidationError(f"[{cfg['short_name']}] Target column '{target_col}' not found.")

    target_series = df[target_col]
    unique_vals = target_series.unique().tolist()
    counts = target_series.value_counts(dropna=False).to_dict()

    if is_raw and dataset == "uci":
        valid_raw = {0, 1, 2, 3, 4}
        if not set(unique_vals).issubset(valid_raw):
            raise ValidationError(f"[{cfg['short_name']}] Unexpected values in raw target '{target_col}': {unique_vals}")
    else:
        # Processed target must be binary {0, 1}
        valid_binary = {0, 1}
        if not set(unique_vals).issubset(valid_binary):
            raise ValidationError(f"[{cfg['short_name']}] Target must be strictly binary {0, 1}, got: {unique_vals}")

    return {
        "target_col": target_col,
        "unique_values": unique_vals,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "prevalence": float(target_series.mean()) if not (is_raw and dataset == "uci") else None
    }


def audit_data_quality(df: pd.DataFrame, dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    """
    Performs comprehensive data quality audit:
    - Missing values (counts and percentages)
    - Duplicate rows
    - Value ranges for numeric columns
    - Domain validity for categorical columns
    """
    cfg = get_dataset_config(dataset)
    metadata_map = cfg["feature_metadata"]

    report: Dict[str, Any] = {
        "dataset": dataset,
        "dataset_name": cfg["name"],
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {}
    }

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = float((missing_count / len(df)) * 100) if len(df) > 0 else 0.0
        col_type = str(series.dtype)
        unique_count = int(series.nunique(dropna=True))

        col_audit: Dict[str, Any] = {
            "dtype": col_type,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 2),
            "unique_count": unique_count
        }

        meta = metadata_map.get(col, {})
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


def generate_audit_artifacts(raw_df: pd.DataFrame, processed_df: pd.DataFrame, dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    """
    Generates and persists:
    1. Feature Dictionary (CSV)
    2. Data Quality Summary (JSON)
    3. Markdown Audit Report (MD)
    """
    cfg = get_dataset_config(dataset)
    audit_dir: Path = cfg["audit_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)

    feat_dict_file = audit_dir / "feature_dictionary.csv"
    data_quality_file = audit_dir / "data_quality_summary.json"
    audit_report_file = audit_dir / "dataset_audit_report.md"

    # 1. Feature Dictionary
    rows = []
    for col, meta in cfg["feature_metadata"].items():
        rows.append({
            "feature": col,
            "type": meta.get("type", "unknown"),
            "description": meta.get("description", ""),
            "unit_or_values": str(meta.get("valid_values", meta.get("valid_range", "N/A"))),
            "preprocessing": meta.get("preprocessing", "")
        })
    feat_dict_df = pd.DataFrame(rows)
    feat_dict_df.to_csv(feat_dict_file, index=False)
    logger.info(f"[{cfg['short_name']}] Saved feature dictionary to {feat_dict_file}")

    # 2. Quality Summary
    target_col = cfg["target_column"]
    raw_meta = {}
    meta_path = cfg.get("metadata_file")
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

    audit_summary = {
        "dataset_key": dataset,
        "dataset_name": cfg["name"],
        "retrieval_url": cfg.get("primary_url"),
        "sha256_checksum": raw_meta.get("sha256", "N/A"),
        "raw_shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
        "processed_shape": [int(processed_df.shape[0]), int(processed_df.shape[1])],
        "target_distribution": {
            "absence (0)": int((processed_df[target_col] == 0).sum()),
            "presence (1)": int((processed_df[target_col] == 1).sum()),
            "prevalence_rate": round(float(processed_df[target_col].mean()), 4)
        },
        "missing_values_per_column": {
            col: int(processed_df[col].isna().sum())
            for col in processed_df.columns if processed_df[col].isna().sum() > 0
        },
        "duplicate_rows": int(processed_df.duplicated().sum())
    }

    with open(data_quality_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    logger.info(f"[{cfg['short_name']}] Saved data quality summary to {data_quality_file}")

    # 3. Markdown Audit Report
    target_dist = audit_summary["target_distribution"]
    missing_cols = audit_summary["missing_values_per_column"]

    md_content = f"""# CardioSense Dataset Audit & Quality Analysis Report: {cfg['name']}

**Date & Time**: {raw_meta.get('download_timestamp_utc', '2026-10-28T00:00:00Z')}  
**Source**: [{cfg['name']}]({cfg.get('primary_url', '')})  
**SHA256 Checksum**: `{raw_meta.get('sha256', 'N/A')}`  
**Raw Dimensions**: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns  
**Processed Dimensions**: {processed_df.shape[0]} rows × {processed_df.shape[1]} columns  

---

## 1. Executive Summary & Provenance
- **Cohort**: {cfg['name']} ({cfg['origin']})
- **Authors**: {cfg['authors']}
- **DOI / Citation**: {cfg.get('citation', cfg.get('doi', 'N/A'))}
- **Raw Columns**: {raw_df.shape[1]} attributes.
- **Patient Identifiers**: Removed from feature matrix to guarantee patient privacy and zero leakage.

## 2. Target Variable Verification & Distribution
- **Class 0 (Absence / Healthy)**: {target_dist['absence (0)']} patients ({round(target_dist['absence (0)']/processed_df.shape[0]*100, 2)}%)
- **Class 1 (Presence / CVD)**: {target_dist['presence (1)']} patients ({round(target_dist['presence (1)']/processed_df.shape[0]*100, 2)}%)
- **Positive Prevalence Rate**: {round(target_dist['prevalence_rate']*100, 2)}%

## 3. Data Hygiene & Missing Values
- **Duplicate Records**: {audit_summary['duplicate_rows']} exact duplicates detected.
- **Identified Missing Value Columns**:
"""
    if missing_cols:
        for col, count in missing_cols.items():
            pct = round((count / len(processed_df)) * 100, 2)
            md_content += f"  - `{col}`: {count} missing values ({pct}%)\n"
    else:
        md_content += "  - No missing values in processed feature matrix.\n"

    md_content += """
### Imputation Strategy:
- Numeric features: Median imputation (fitted strictly on training fold).
- Categorical features: Most frequent / mode imputation (fitted strictly on training fold).
- Split-First Protocol guarantees zero data leakage between development folds and holdout test data.

## 4. Feature Schema & Summary Statistics
| Feature | Type | Meaning | Observed Range / Values | Missing Values | Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for col in cfg["feature_names"]:
        meta = cfg["feature_metadata"].get(col, {})
        col_series = processed_df[col]
        missing_cnt = int(col_series.isna().sum())
        if col in cfg["numeric_features"]:
            val_str = f"[{col_series.min():.1f}, {col_series.max():.1f}], mean={col_series.mean():.1f}"
        else:
            uniq = sorted([int(x) for x in col_series.dropna().unique()])
            val_str = f"Categorical: {uniq}"
        md_content += f"| `{col}` | {meta.get('type', 'N/A')} | {meta.get('description', '')} | {val_str} | {missing_cnt} | {meta.get('preprocessing', '')} |\n"

    md_content += f"""
## 5. Non-Diagnostic Disclaimer
CardioSense is an educational and academic research tool. Predictions must never be used for actual clinical medical diagnosis or treatment decisions.
"""

    with open(audit_report_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"[{cfg['short_name']}] Generated audit report at {audit_report_file}")

    # For backward compatibility, if active matches default dataset, copy to root audit files
    if dataset == DEFAULT_DATASET:
        root_audit = PROJECT_ROOT / "reports" / "dataset_audit"
        root_audit.mkdir(parents=True, exist_ok=True)
        feat_dict_df.to_csv(root_audit / "feature_dictionary.csv", index=False)
        with open(root_audit / "data_quality_summary.json", "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)
        with open(root_audit / "dataset_audit_report.md", "w", encoding="utf-8") as f:
            f.write(md_content)

    return audit_summary


def run_validation_pipeline(dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    """Execute complete validation and generate audit reports for specified dataset."""
    from src.data_loader import load_raw_dataset, process_dataset
    raw_df = load_raw_dataset(dataset=dataset)
    validate_schema(raw_df, is_raw=True, dataset=dataset)
    validate_target(raw_df, is_raw=True, dataset=dataset)

    processed_df = process_dataset(raw_df, dataset=dataset)
    validate_schema(processed_df, is_raw=False, dataset=dataset)
    validate_target(processed_df, is_raw=False, dataset=dataset)

    return generate_audit_artifacts(raw_df, processed_df, dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioSense Data Validation Pipeline")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=["india", "uci", "all"],
                        help="Dataset to validate: 'india' (default), 'uci', or 'all'")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["india", "uci"]:
            print(f"\n{'='*50}\nValidating {ds.upper()} Dataset\n{'='*50}")
            run_validation_pipeline(dataset=ds)
    else:
        run_validation_pipeline(dataset=args.dataset)
