"""
Evaluation and error analysis module for CardioSense.
Computes comprehensive metrics, confusion matrices, ROC curves, PR curves,
and detailed test-set error diagnostics.
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
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    ERROR_ANALYSIS_CSV,
    FEATURE_NAMES,
    TARGET_NAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes all standard binary classification metrics.
    Positive class is strictly defined as 1 (Presence of Heart Disease).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Clinical diagnostic metrics
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))  # Recall / TPR
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0       # TNR
    precision = float(precision_score(y_true, y_pred, zero_division=0))  # PPV
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    
    auc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None
    avg_precision = float(average_precision_score(y_true, y_proba)) if y_proba is not None else None

    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "average_precision": round(avg_precision, 4) if avg_precision is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "total_test_samples": int(len(y_true))
    }


def plot_confusion_matrices(models_results: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Plots side-by-side annotated confusion matrices for all evaluated models."""
    output_path = save_path or FIGURES_DIR / "confusion_matrices_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(models_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(models_results.items()):
        ax = axes[idx]
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Absence (0)", "Presence (1)"],
            yticklabels=["Absence (0)", "Presence (1)"],
            ax=ax, annot_kws={"size": 14, "weight": "bold"}
        )
        fn = res["metrics"]["false_negatives"]
        fp = res["metrics"]["false_positives"]
        rec = res["metrics"]["recall"]
        prec = res["metrics"]["precision"]
        ax.set_title(f"{name.replace('_', ' ').title()}\nRecall: {rec:.2f} | Prec: {prec:.2f}\n(FN={fn}, FP={fp})", fontsize=11)
        ax.set_xlabel("Predicted Diagnosis")
        ax.set_ylabel("True Dataset Diagnosis")

    fig.suptitle("Test Set Confusion Matrix Comparison (Held-out N=61)", y=1.04, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix comparison to {output_path}")
    return output_path


def plot_roc_curves(models_results: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Plots multi-model ROC curves with AUC annotations."""
    output_path = save_path or FIGURES_DIR / "roc_curves_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    for idx, (name, res) in enumerate(models_results.items()):
        if "y_proba" in res and res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
            auc = res["metrics"]["roc_auc"]
            ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC = {auc:.3f})", linewidth=2.2)

    # Random chance line
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", alpha=0.6)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves - Test Set")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved ROC curves comparison to {output_path}")
    return output_path


def plot_precision_recall_curves(models_results: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Plots multi-model Precision-Recall curves."""
    output_path = save_path or FIGURES_DIR / "precision_recall_curves_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    for idx, (name, res) in enumerate(models_results.items()):
        if "y_proba" in res and res["y_proba"] is not None:
            precision, recall, _ = precision_recall_curve(res["y_true"], res["y_proba"])
            ap = res["metrics"].get("average_precision", 0.0)
            ax.plot(recall, precision, label=f"{name.replace('_', ' ').title()} (AP = {ap:.3f})", linewidth=2.2)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.4, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall Curves - Test Set")
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Precision-Recall curves to {output_path}")
    return output_path


def plot_metrics_comparison_bar(models_results: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Generates comparative bar chart across models for primary performance metrics."""
    output_path = save_path or FIGURES_DIR / "models_metrics_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_names = ["roc_auc", "recall", "precision", "f1_score", "accuracy"]
    display_names = ["ROC-AUC", "Recall (Sensitivity)", "Precision", "F1-Score", "Accuracy"]
    
    data = []
    for model_name, res in models_results.items():
        m = res["metrics"]
        for key, disp in zip(metrics_names, display_names):
            data.append({
                "Model": model_name.replace("_", " ").title(),
                "Metric": disp,
                "Score": m.get(key, 0.0)
            })
    
    df_plot = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=df_plot, x="Metric", y="Score", hue="Model", palette="deep", ax=ax, edgecolor="black", alpha=0.88)
    
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Metric Value")
    ax.set_title("Held-out Test Performance Across Model Families")
    ax.legend(title="Model Family", bbox_to_anchor=(1.02, 1), loc="upper left")
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.2f}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom", fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords="offset points")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved metrics comparison bar plot to {output_path}")
    return output_path


def perform_error_analysis(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str
) -> pd.DataFrame:
    """
    Identifies and analyzes misclassified observations on held-out test data.
    Documents true labels, predicted probabilities, error category (FP vs FN),
    and clinical predictor profiles.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    df_err = X_test.copy()
    df_err["true_target"] = y_true.values
    df_err["predicted_target"] = y_pred
    df_err["predicted_proba"] = np.round(y_proba, 4)
    df_err["prediction_correct"] = (df_err["true_target"] == df_err["predicted_target"])
    
    def categorize_error(row):
        if row["true_target"] == 1 and row["predicted_target"] == 0:
            return "False Negative (Missed Disease)"
        elif row["true_target"] == 0 and row["predicted_target"] == 1:
            return "False Positive (False Alarm)"
        return "Correct"

    df_err["error_type"] = df_err.apply(categorize_error, axis=1)
    
    # Filter only errors
    misclassified = df_err[~df_err["prediction_correct"]].copy()
    misclassified.sort_values(by="error_type", inplace=True)
    misclassified.to_csv(ERROR_ANALYSIS_CSV, index=True, index_label="test_set_index")
    logger.info(f"Saved {len(misclassified)} misclassified records to {ERROR_ANALYSIS_CSV}")

    return misclassified
