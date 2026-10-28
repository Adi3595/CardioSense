"""
SHAP Explainability module for CardioSense.
Computes TreeExplainer and LinearExplainer feature attributions.
Supports both India Hospital Cohort and UCI Cleveland Benchmark.
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
    DEFAULT_DATASET,
    get_dataset_config,
    DATASETS_CONFIG,
    FIGURES_DIR,
    METRICS_DIR,
    PROJECT_ROOT
)
from src.preprocessing import get_feature_names_out

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_shap_explanations(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    dataset: str = DEFAULT_DATASET
) -> Dict[str, Any]:
    """
    Computes SHAP explanations on transformed test features for specified dataset.
    Saves global summary plots, bar importance plots, and local prediction waterfalls.
    """
    cfg = get_dataset_config(dataset)
    fig_dir: Path = cfg["figures_dir"]
    metrics_dir: Path = cfg["metrics_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["classifier"]

    # Transform features using the fitted preprocessor
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names_out(preprocessor)

    logger.info(f"[{cfg['short_name']}] Generating SHAP values for model '{model_name}' across {len(feature_names)} transformed features...")

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

    importance_csv = metrics_dir / "shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    logger.info(f"[{cfg['short_name']}] Saved SHAP feature importance table to {importance_csv}")

    # 2. SHAP Summary Beeswarm Plot
    summary_plot_path = fig_dir / "shap_summary_plot.png"
    fig, ax = plt.subplots(figsize=(9, 6.5))
    shap.summary_plot(
        shap_matrix,
        X_test_trans,
        feature_names=clean_feature_names,
        plot_type="dot",
        show=False,
        max_display=12
    )
    plt.title(f"SHAP Global Feature Attribution ({cfg['short_name']}: {model_name.replace('_', ' ').title()})", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[{cfg['short_name']}] Saved SHAP summary plot to {summary_plot_path}")

    # 3. Feature Importance Bar Plot
    bar_plot_path = fig_dir / "shap_feature_importance_bar.png"
    top_10 = importance_df.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top_10["feature"], top_10["mean_absolute_shap"], color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Prediction Magnitude)")
    ax.set_title(f"Top 10 Most Influential Predictors ({cfg['short_name']})", pad=10)
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f"{w:.4f}", (w, p.get_y() + p.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved SHAP importance bar chart to {bar_plot_path}")

    # 4. Local Explanations (Waterfall Plots)
    local_correct_path = fig_dir / "shap_local_correct.png"
    local_incorrect_path = fig_dir / "shap_local_incorrect.png"

    # Find a correct high-risk prediction and an incorrect prediction
    y_preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    correct_positives = np.where((y_test.values == 1) & (y_preds == 1))[0]
    misclassifications = np.where(y_test.values != y_preds)[0]

    correct_idx = int(correct_positives[np.argmax(probs[correct_positives])]) if len(correct_positives) > 0 else 0
    incorrect_idx = int(misclassifications[0]) if len(misclassifications) > 0 else 1

    # Base value for waterfall plot
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) == 2:
            base_val = ev[1]
        else:
            base_val = float(ev)
    else:
        base_val = float(np.mean(probs))

    def make_waterfall(idx: int, out_path: Path, title: str):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        expl = shap.Explanation(
            values=shap_matrix[idx],
            base_values=base_val,
            data=X_test_trans[idx],
            feature_names=clean_feature_names
        )
        shap.plots.waterfall(expl, max_display=8, show=False)
        plt.title(title, fontsize=11, pad=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    try:
        make_waterfall(correct_idx, local_correct_path, f"SHAP Local Deconstruction: Correct CAD Patient (True=1, Pred={probs[correct_idx]:.2f})")
        logger.info(f"[{cfg['short_name']}] Saved local correct waterfall plot to {local_correct_path}")
    except Exception as e:
        logger.warning(f"Could not generate correct waterfall: {e}")

    try:
        make_waterfall(incorrect_idx, local_incorrect_path, f"SHAP Local Deconstruction: Misclassified Patient (True={y_test.values[incorrect_idx]}, Pred={probs[incorrect_idx]:.2f})")
        logger.info(f"[{cfg['short_name']}] Saved local incorrect waterfall plot to {local_incorrect_path}")
    except Exception as e:
        logger.warning(f"Could not generate incorrect waterfall: {e}")

    # Backward compatibility for primary dataset
    if dataset == DEFAULT_DATASET:
        root_fig = PROJECT_ROOT / "reports" / "figures"
        root_metrics = PROJECT_ROOT / "reports" / "metrics"
        import shutil
        for p in [summary_plot_path, bar_plot_path, local_correct_path, local_incorrect_path]:
            if p.exists():
                shutil.copy(p, root_fig / p.name)
        if importance_csv.exists():
            shutil.copy(importance_csv, root_metrics / importance_csv.name)

    return {
        "explainer_type": explainer_type,
        "feature_importance_df": importance_df,
        "top_features": importance_df["feature"].head(8).tolist(),
        "summary_plot": summary_plot_path,
        "bar_plot": bar_plot_path
    }
