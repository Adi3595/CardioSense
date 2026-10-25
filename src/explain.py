"""
SHAP Explainability module for CardioSense.
Computes TreeExplainer and LinearExplainer feature attributions.
Generates global summary plots, feature importance tables, and local waterfall explanations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    FEATURE_NAMES
)
from src.preprocessing import get_feature_names_out

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_shap_explanations(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """
    Computes SHAP explanations on transformed test features.
    Saves global summary plots, bar importance plots, and local prediction waterfalls.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["classifier"]

    # Transform features using the fitted preprocessor
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names_out(preprocessor)

    logger.info(f"Generating SHAP values for model '{model_name}' across {len(feature_names)} transformed features...")

    # Select appropriate explainer based on estimator architecture
    explainer_type = "TreeExplainer"
    if "xgb" in model_name.lower():
        explainer = shap.TreeExplainer(estimator)
        shap_values_raw = explainer(X_test_trans)
        shap_matrix = shap_values_raw.values
    elif "forest" in model_name.lower():
        explainer = shap.TreeExplainer(estimator)
        shap_values_raw = explainer(X_test_trans)
        # Random forest in shap can produce shape (N, D, 2) for binary classification
        if len(shap_values_raw.values.shape) == 3:
            shap_matrix = shap_values_raw.values[:, :, 1]  # Positive class (presence)
        else:
            shap_matrix = shap_values_raw.values
    else:
        # Linear model (Logistic Regression)
        explainer_type = "LinearExplainer"
        explainer = shap.LinearExplainer(estimator, X_train_trans)
        shap_values_raw = explainer(X_test_trans)
        shap_matrix = shap_values_raw.values

    # Clean feature names for clean plotting
    clean_feature_names = [f.replace("cat__", "").replace("num__", "") for f in feature_names]

    # 1. Global Feature Importance Table
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    importance_df = pd.DataFrame({
        "feature": clean_feature_names,
        "mean_absolute_shap": mean_abs_shap
    }).sort_values(by="mean_absolute_shap", ascending=False).reset_index(drop=True)

    importance_csv = METRICS_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    logger.info(f"Saved SHAP feature importance table to {importance_csv}")

    # 2. SHAP Summary Beeswarm Plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    shap.summary_plot(
        shap_matrix,
        X_test_trans,
        feature_names=clean_feature_names,
        plot_type="dot",
        show=False,
        max_display=12
    )
    plt.title(f"SHAP Global Feature Attribution ({model_name.replace('_', ' ').title()})", fontsize=13, pad=12)
    plt.tight_layout()
    summary_plot_path = FIGURES_DIR / "shap_summary_plot.png"
    fig.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved SHAP summary plot to {summary_plot_path}")

    # 3. SHAP Mean Absolute Importance Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top_10 = importance_df.head(10).iloc[::-1]
    ax.barh(top_10["feature"], top_10["mean_absolute_shap"], color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value| (Average Impact on Prediction)")
    ax.set_title(f"Top 10 Most Influential Features (SHAP - {model_name.replace('_', ' ').title()})")
    plt.tight_layout()
    bar_plot_path = FIGURES_DIR / "shap_feature_importance_bar.png"
    fig.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved SHAP bar plot to {bar_plot_path}")

    # 4. Local Explanations: 1 Correct Prediction & 1 Misclassified Prediction (if available)
    y_preds = pipeline.predict(X_test)
    correct_indices = np.where(y_preds == y_test.values)[0]
    error_indices = np.where(y_preds != y_test.values)[0]

    # Explain correct prediction
    correct_idx = int(correct_indices[0]) if len(correct_indices) > 0 else 0
    fig, ax = plt.subplots(figsize=(9, 5))
    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
        base_val = expected_val[1]
    else:
        base_val = float(expected_val) if np.isscalar(expected_val) else float(expected_val[0])

    exp_correct = shap.Explanation(
        values=shap_matrix[correct_idx],
        base_values=base_val,
        data=X_test_trans[correct_idx],
        feature_names=clean_feature_names
    )
    shap.plots.waterfall(exp_correct, max_display=10, show=False)
    true_label = int(y_test.iloc[correct_idx])
    pred_label = int(y_preds[correct_idx])
    plt.title(f"Local Explanation: Correct Prediction (True={true_label}, Pred={pred_label})", fontsize=12, pad=10)
    plt.tight_layout()
    local_correct_path = FIGURES_DIR / "shap_local_correct.png"
    plt.savefig(local_correct_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved local correct SHAP explanation to {local_correct_path}")

    # Explain incorrect prediction
    local_incorrect_path = None
    if len(error_indices) > 0:
        error_idx = int(error_indices[0])
        fig, ax = plt.subplots(figsize=(9, 5))
        exp_error = shap.Explanation(
            values=shap_matrix[error_idx],
            base_values=base_val,
            data=X_test_trans[error_idx],
            feature_names=clean_feature_names
        )
        shap.plots.waterfall(exp_error, max_display=10, show=False)
        err_true = int(y_test.iloc[error_idx])
        err_pred = int(y_preds[error_idx])
        plt.title(f"Local Explanation: Misclassification (True={err_true}, Pred={err_pred})", fontsize=12, pad=10)
        plt.tight_layout()
        local_incorrect_path = FIGURES_DIR / "shap_local_incorrect.png"
        plt.savefig(local_incorrect_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved local error SHAP explanation to {local_incorrect_path}")

    return {
        "explainer_type": explainer_type,
        "shap_version": shap.__version__,
        "top_features": importance_df.head(5).to_dict(orient="records"),
        "summary_plot": str(summary_plot_path),
        "bar_plot": str(bar_plot_path),
        "local_correct_plot": str(local_correct_path),
        "local_incorrect_plot": str(local_incorrect_path) if local_incorrect_path else None
    }
