"""
Experiment tracking and artifact logging module for CardioSense.
Supports both India Hospital Cohort and UCI Cleveland Benchmark.
Logs training metadata, cross-validation metrics, parameter grids, and environment details.
"""

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import sklearn
import xgboost
import shap

from src.config import (
    DEFAULT_DATASET,
    get_dataset_config,
    DATASETS_CONFIG,
    RANDOM_STATE,
    CV_FOLDS,
    PRIMARY_SELECTION_METRIC,
    PROJECT_ROOT
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
    best_model_name: str,
    dataset: str = DEFAULT_DATASET
) -> Path:
    """
    Saves comprehensive experiment tracking metadata to dataset-specific directories.
    """
    cfg = get_dataset_config(dataset)
    models_dir: Path = cfg["models_dir"]
    metrics_dir: Path = cfg["metrics_dir"]

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = models_dir / "model_metadata.json"
    comparison_json = metrics_dir / "model_comparison.json"
    comparison_csv = metrics_dir / "model_comparison.csv"
    test_metrics_json = metrics_dir / "test_metrics.json"

    raw_meta = {}
    meta_path = cfg.get("metadata_file")
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

    metadata: Dict[str, Any] = {
        "project": "CardioSense",
        "dataset_key": dataset,
        "dataset_name": cfg["name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_info": {
            "name": cfg["name"],
            "url": cfg.get("primary_url"),
            "sha256": raw_meta.get("sha256", cfg.get("sha256", "N/A")),
            "authors": cfg.get("authors")
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

    # Save metadata JSON
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[{cfg['short_name']}] Saved model metadata to {metadata_file}")

    # Save Model Comparison JSON & CSV
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(comparison_csv, index=False)
    logger.info(f"[{cfg['short_name']}] Saved model comparison table to {comparison_csv}")

    # Save isolated test metrics
    with open(test_metrics_json, "w", encoding="utf-8") as f:
        json.dump({k: v.get("metrics", {}) for k, v in test_results.items()}, f, indent=2)

    # Backward compatibility for default primary dataset
    if dataset == DEFAULT_DATASET:
        root_models = PROJECT_ROOT / "models"
        root_metrics = PROJECT_ROOT / "reports" / "metrics"
        import shutil
        shutil.copy(metadata_file, root_models / "model_metadata.json")
        shutil.copy(comparison_json, root_metrics / "model_comparison.json")
        shutil.copy(comparison_csv, root_metrics / "model_comparison.csv")
        shutil.copy(test_metrics_json, root_metrics / "test_metrics.json")

    return metadata_file
