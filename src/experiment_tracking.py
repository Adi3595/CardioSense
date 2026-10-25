"""
Experiment tracking and artifact logging module for CardioSense.
Logs training metadata, cross-validation metrics, parameter grids, and environment details.
"""

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import sklearn
import xgboost
import shap

from src.config import (
    MODEL_METADATA_FILE,
    MODEL_COMPARISON_JSON,
    MODEL_COMPARISON_CSV,
    TEST_METRICS_JSON,
    DATASET_SOURCE_INFO,
    RAW_METADATA_FILE,
    RANDOM_STATE,
    CV_FOLDS,
    PRIMARY_SELECTION_METRIC
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_environment_info() -> Dict[str, str]:
    """Capture environment and package versions for reproducibility."""
    return {
        "python_version": sys.version.split()[0],
        "os_platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "shap_version": shap.__version__
    }


def save_experiment_metadata(
    models_cv_results: Dict[str, Any],
    test_results: Dict[str, Any],
    best_model_name: str
) -> Path:
    """
    Saves comprehensive experiment tracking metadata to models/ and reports/metrics/.
    """
    MODEL_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)

    raw_meta = {}
    if RAW_METADATA_FILE.exists():
        with open(RAW_METADATA_FILE, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

    metadata: Dict[str, Any] = {
        "project": "CardioSense",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": DATASET_SOURCE_INFO["name"],
            "url": DATASET_SOURCE_INFO["primary_url"],
            "sha256": raw_meta.get("sha256", "N/A")
        },
        "experimental_controls": {
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "primary_selection_metric": PRIMARY_SELECTION_METRIC,
            "train_test_split": "80/20 Stratified"
        },
        "environment": get_environment_info(),
        "best_model_selected": best_model_name,
        "models": {}
    }

    comparison_records: List[Dict[str, Any]] = []

    for name in models_cv_results:
        cv_data = models_cv_results[name]
        t_data = test_results.get(name, {})

        model_entry = {
            "best_params": cv_data.get("best_params", {}),
            "cv_metrics": cv_data.get("cv_metrics", {}),
            "test_metrics": t_data.get("metrics", {})
        }
        metadata["models"][name] = model_entry

        record = {
            "model": name,
            "cv_roc_auc_mean": cv_data.get("cv_metrics", {}).get("mean_roc_auc"),
            "cv_roc_auc_std": cv_data.get("cv_metrics", {}).get("std_roc_auc"),
            "cv_f1_mean": cv_data.get("cv_metrics", {}).get("mean_f1"),
            "test_accuracy": t_data.get("metrics", {}).get("accuracy"),
            "test_precision": t_data.get("metrics", {}).get("precision"),
            "test_recall": t_data.get("metrics", {}).get("recall"),
            "test_specificity": t_data.get("metrics", {}).get("specificity"),
            "test_f1": t_data.get("metrics", {}).get("f1_score"),
            "test_roc_auc": t_data.get("metrics", {}).get("roc_auc"),
            "false_negatives": t_data.get("metrics", {}).get("false_negatives"),
            "false_positives": t_data.get("metrics", {}).get("false_positives")
        }
        comparison_records.append(record)

    # Save JSON metadata
    with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved experiment metadata to {MODEL_METADATA_FILE}")

    with open(MODEL_COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    # Save CSV comparison table
    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    logger.info(f"Saved model comparison table to {MODEL_COMPARISON_CSV}")

    return MODEL_METADATA_FILE
