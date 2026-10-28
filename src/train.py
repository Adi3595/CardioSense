"""
Model training and comparison pipeline for CardioSense.
Supports dual-dataset training:
1. India Hospital Cohort (Primary): N=1000, 800 train / 200 test
2. UCI Cleveland Benchmark (Historical): N=303, 242 train / 61 test

Trains Logistic Regression, Random Forest, and XGBoost with Stratified 5-Fold Cross-Validation,
evaluates on held-out test set, runs error analysis and SHAP explainability,
and saves the production-ready inference pipeline.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import xgboost as xgb

from src.config import (
    DEFAULT_DATASET,
    get_dataset_config,
    DATASETS_CONFIG,
    RANDOM_STATE,
    CV_FOLDS,
    PRIMARY_SELECTION_METRIC,
    PARAM_GRIDS,
    PROJECT_ROOT
)
from src.preprocessing import build_pipeline
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_metrics_comparison_bar,
    perform_error_analysis
)
from src.explain import generate_shap_explanations
from src.experiment_tracking import save_experiment_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_base_models() -> Dict[str, Tuple[Any, bool]]:
    """
    Returns dictionary of (estimator, include_scaling) for the 3 model families.
    """
    return {
        "logistic_regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            True  # Scale continuous features for linear model
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            False  # Tree-based models are invariant to monotonic feature scaling
        ),
        "xgboost": (
            xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            ),
            False
        )
    }


def train_and_tune_model(
    model_name: str,
    base_estimator: Any,
    include_scaling: bool,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dataset: str = DEFAULT_DATASET
) -> Tuple[Any, Dict[str, Any]]:
    """
    Constructs pipeline, executes GridSearchCV with StratifiedKFold,
    and returns the best estimator and CV metrics.
    """
    cfg = get_dataset_config(dataset)
    logger.info(f"[{cfg['short_name']}] Training & Tuning {model_name} (5-Fold Stratified CV)...")
    pipeline = build_pipeline(base_estimator, dataset=dataset, include_scaling=include_scaling)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1"
        },
        refit=PRIMARY_SELECTION_METRIC,
        cv=cv,
        n_jobs=-1,
        return_train_score=False
    )
    
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    best_idx = grid_search.best_index_
    cv_res = grid_search.cv_results_

    cv_metrics = {
        "mean_roc_auc": round(float(cv_res["mean_test_roc_auc"][best_idx]), 4),
        "std_roc_auc": round(float(cv_res["std_test_roc_auc"][best_idx]), 4),
        "mean_accuracy": round(float(cv_res["mean_test_accuracy"][best_idx]), 4),
        "mean_precision": round(float(cv_res["mean_test_precision"][best_idx]), 4),
        "mean_recall": round(float(cv_res["mean_test_recall"][best_idx]), 4),
        "mean_f1": round(float(cv_res["mean_test_f1"][best_idx]), 4),
    }

    logger.info(f"[{cfg['short_name']}] {model_name} Best Params: {grid_search.best_params_}")
    logger.info(f"[{cfg['short_name']}] {model_name} 5-Fold CV ROC-AUC: {cv_metrics['mean_roc_auc']} ± {cv_metrics['std_roc_auc']}")

    return best_estimator, {
        "best_params": grid_search.best_params_,
        "cv_metrics": cv_metrics
    }


def train_pipeline(dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    """
    Full training, evaluation, explainability and serialization workflow for specified dataset.
    """
    cfg = get_dataset_config(dataset)
    train_file: Path = cfg["train_file"]
    test_file: Path = cfg["test_file"]
    models_dir: Path = cfg["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists() or not test_file.exists():
        logger.info(f"[{cfg['short_name']}] Partition files missing. Running data pipeline first...")
        from src.data_loader import prepare_data_pipeline
        prepare_data_pipeline(dataset=dataset)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    feature_cols = cfg["feature_names"]
    target_col = cfg["target_column"]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    logger.info(f"[{cfg['short_name']}] Dataset loaded: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows")

    base_models = get_base_models()
    trained_models = {}
    cv_results = {}
    test_results = {}

    # 1. Train and tune each model family
    for name, (base_est, include_scaling) in base_models.items():
        grid = PARAM_GRIDS.get(name, {})
        best_est, cv_info = train_and_tune_model(
            name, base_est, include_scaling, grid, X_train, y_train, dataset=dataset
        )
        trained_models[name] = best_est
        cv_results[name] = cv_info

        # Blind evaluation on untouched test set
        y_pred = best_est.predict(X_test)
        y_proba = best_est.predict_proba(X_test)[:, 1] if hasattr(best_est, "predict_proba") else None
        metrics = calculate_metrics(y_test.values, y_pred, y_proba)
        
        test_results[name] = {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": metrics
        }
        logger.info(
            f"[{cfg['short_name']}] {name.upper()} Test Metrics -> "
            f"Accuracy: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | ROC-AUC: {metrics['roc_auc']} | "
            f"FN: {metrics['false_negatives']} | FP: {metrics['false_positives']}"
        )

        # Serialize individual fitted models
        model_save_path = models_dir / f"{name}.joblib"
        joblib.dump(best_est, model_save_path, compress=3)
        logger.info(f"[{cfg['short_name']}] Serialized {name} to {model_save_path}")

    # 2. Select Champion Model based on CV ROC-AUC
    champion_name = max(cv_results, key=lambda k: cv_results[k]["cv_metrics"]["mean_roc_auc"])
    champion_pipeline = trained_models[champion_name]
    logger.info(f"[{cfg['short_name']}] >>> Selected Champion Model: '{champion_name}' <<<")

    # Serialize champion pipeline artifact
    champion_file = models_dir / "best_pipeline.joblib"
    joblib.dump(champion_pipeline, champion_file, compress=3)
    logger.info(f"[{cfg['short_name']}] Serialized champion pipeline artifact to {champion_file}")

    # 3. Generate Comparative Evaluation Visualizations
    plot_confusion_matrices(test_results, dataset=dataset)
    plot_roc_curves(test_results, dataset=dataset)
    plot_precision_recall_curves(test_results, dataset=dataset)
    plot_metrics_comparison_bar(test_results, dataset=dataset)

    # 4. Error Analysis on Champion Model
    champ_res = test_results[champion_name]
    perform_error_analysis(
        X_test, y_test,
        champ_res["y_pred"], champ_res["y_proba"],
        dataset=dataset
    )

    # 5. SHAP Explainability on Champion Model
    generate_shap_explanations(
        champion_pipeline, X_train, X_test, y_test,
        champion_name, dataset=dataset
    )

    # 6. Save Comprehensive Experiment Tracking Metadata
    save_experiment_metadata(cv_results, test_results, champion_name, dataset=dataset)

    # Backward compatibility: if active matches default primary dataset, also update root best_pipeline
    if dataset == DEFAULT_DATASET:
        root_champion_file = PROJECT_ROOT / "models" / "best_pipeline.joblib"
        joblib.dump(champion_pipeline, root_champion_file, compress=3)
        logger.info(f"Synchronized primary champion pipeline to root {root_champion_file}")

    logger.info(f"[{cfg['short_name']}] End-to-end training pipeline completed successfully.")
    return {
        "dataset": dataset,
        "champion_model": champion_name,
        "cv_results": cv_results,
        "test_results": test_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioSense Model Training Suite")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=["india", "uci", "all"],
                        help="Dataset to train: 'india' (default), 'uci', or 'all'")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["india", "uci"]:
            print(f"\n{'='*50}\nTraining Models on {ds.upper()} Dataset\n{'='*50}")
            train_pipeline(dataset=ds)
    else:
        train_pipeline(dataset=args.dataset)
