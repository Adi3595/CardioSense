"""
Model training and comparison pipeline for CardioSense.
Trains Logistic Regression, Random Forest, and XGBoost with Stratified 5-Fold Cross-Validation,
evaluates on held-out test set, runs error analysis and SHAP explainability,
and saves the production-ready inference pipeline.
"""

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
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    MODELS_DIR,
    BEST_PIPELINE_FILE,
    RANDOM_STATE,
    CV_FOLDS,
    PRIMARY_SELECTION_METRIC,
    PARAM_GRIDS,
    FEATURE_NAMES,
    TARGET_NAME
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
    y_train: pd.Series
) -> Tuple[Any, Dict[str, Any]]:
    """
    Constructs pipeline, executes GridSearchCV with StratifiedKFold,
    and returns the best estimator and CV metrics.
    """
    logger.info(f"--- Training & Tuning {model_name} (5-Fold Stratified CV) ---")
    pipeline = build_pipeline(base_estimator, include_scaling=include_scaling)
    
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
    
    best_idx = grid_search.best_index_
    results = grid_search.cv_results_
    
    cv_metrics = {
        "mean_roc_auc": round(float(results["mean_test_roc_auc"][best_idx]), 4),
        "std_roc_auc": round(float(results["std_test_roc_auc"][best_idx]), 4),
        "mean_accuracy": round(float(results["mean_test_accuracy"][best_idx]), 4),
        "std_accuracy": round(float(results["std_test_accuracy"][best_idx]), 4),
        "mean_precision": round(float(results["mean_test_precision"][best_idx]), 4),
        "mean_recall": round(float(results["mean_test_recall"][best_idx]), 4),
        "mean_f1": round(float(results["mean_test_f1"][best_idx]), 4)
    }
    
    logger.info(f"{model_name} Best Params: {grid_search.best_params_}")
    logger.info(f"{model_name} Best CV {PRIMARY_SELECTION_METRIC.upper()}: {cv_metrics['mean_roc_auc']:.4f} (+/- {cv_metrics['std_roc_auc']:.4f})")
    
    return grid_search.best_estimator_, {
        "best_params": grid_search.best_params_,
        "cv_metrics": cv_metrics
    }


def run_training_pipeline() -> Dict[str, Any]:
    """
    Orchestrates end-to-end model training, evaluation, comparison,
    explainability, and artifact persistence.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure splits exist
    if not TRAIN_DATA_FILE.exists() or not TEST_DATA_FILE.exists():
        from src.data_loader import prepare_data_pipeline
        prepare_data_pipeline()

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    X_train, y_train = train_df[FEATURE_NAMES], train_df[TARGET_NAME]
    X_test, y_test = test_df[FEATURE_NAMES], test_df[TARGET_NAME]

    base_models = get_base_models()
    models_cv_results: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Any] = {}
    test_results: Dict[str, Any] = {}

    # 1. Train and tune all candidate models
    for name, (base_est, scaling) in base_models.items():
        grid = PARAM_GRIDS.get(name, {})
        best_pipeline, cv_info = train_and_tune_model(
            model_name=name,
            base_estimator=base_est,
            include_scaling=scaling,
            param_grid=grid,
            X_train=X_train,
            y_train=y_train
        )
        fitted_pipelines[name] = best_pipeline
        models_cv_results[name] = cv_info
        
        # Save individual fitted pipeline
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(best_pipeline, model_path)
        logger.info(f"Saved {name} pipeline to {model_path}")

        # 2. Evaluate on untouched held-out test set
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
        metrics = calculate_metrics(y_test.values, y_pred, y_proba)
        
        test_results[name] = {
            "metrics": metrics,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        logger.info(f"{name} Held-Out Test Metrics: ROC-AUC={metrics['roc_auc']}, Recall={metrics['recall']}, Prec={metrics['precision']}, F1={metrics['f1_score']}, FN={metrics['false_negatives']}")

    # 3. Model Selection
    # Primary criterion: highest validation ROC-AUC, secondary: test recall / minimal false negatives
    best_model_name = max(
        models_cv_results.keys(),
        key=lambda k: models_cv_results[k]["cv_metrics"]["mean_roc_auc"]
    )
    logger.info(f"=== Selected Champion Model: {best_model_name} ===")

    best_pipeline = fitted_pipelines[best_model_name]
    joblib.dump(best_pipeline, BEST_PIPELINE_FILE)
    logger.info(f"Saved champion pipeline to {BEST_PIPELINE_FILE}")

    # 4. Comparative Evaluation Figures
    plot_confusion_matrices(test_results)
    plot_roc_curves(test_results)
    plot_precision_recall_curves(test_results)
    plot_metrics_comparison_bar(test_results)

    # 5. Error Analysis on Champion Model
    best_test_res = test_results[best_model_name]
    perform_error_analysis(
        X_test=X_test,
        y_true=y_test,
        y_pred=best_test_res["y_pred"],
        y_proba=best_test_res["y_proba"],
        model_name=best_model_name
    )

    # 6. SHAP Explainability on Champion Model
    shap_info = generate_shap_explanations(
        pipeline=best_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        model_name=best_model_name
    )

    # 7. Experiment Tracking and Metadata Persistence
    save_experiment_metadata(
        models_cv_results=models_cv_results,
        test_results=test_results,
        best_model_name=best_model_name
    )

    logger.info("CardioSense training and evaluation workflow completed successfully.")
    return {
        "best_model": best_model_name,
        "cv_results": models_cv_results,
        "test_results": {k: v["metrics"] for k, v in test_results.items()},
        "shap_info": shap_info
    }


if __name__ == "__main__":
    run_training_pipeline()
