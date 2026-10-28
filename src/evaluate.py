"""
Evaluation and error analysis module for CardioSense.
Supports dual-dataset evaluation:
1. India Hospital Cohort (Primary): N=1000, 200 holdout test samples
2. UCI Cleveland Benchmark (Historical): N=303, 61 holdout test samples

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
    DEFAULT_DATASET,
    get_dataset_config,
    DATASETS_CONFIG,
    FIGURES_DIR,
    METRICS_DIR,
    PROJECT_ROOT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes all standard binary classification metrics.
    Positive class is strictly defined as 1 (Presence of Heart Disease / CVD).
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


def plot_confusion_matrices(models_results: Dict[str, Any], dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plots side-by-side annotated confusion matrices for all evaluated models."""
    cfg = get_dataset_config(dataset)
    output_path = save_path or cfg["figures_dir"] / "confusion_matrices_comparison.png"
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
        acc = res["metrics"]["accuracy"]
        rec = res["metrics"]["recall"]
        ax.set_title(f"{name.replace('_', ' ').title()}\nAcc: {acc:.2%} | Recall: {rec:.2%}", fontsize=12)
        ax.set_xlabel("Predicted Diagnosis")
        ax.set_ylabel("True Diagnosis")

    fig.suptitle(f"Test Set Confusion Matrices ({cfg['short_name']})", fontsize=14, y=1.03)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved confusion matrices figure to {output_path}")
    return output_path


def plot_roc_curves(models_results: Dict[str, Any], dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plots comparative Receiver Operating Characteristic (ROC) curves."""
    cfg = get_dataset_config(dataset)
    output_path = save_path or cfg["figures_dir"] / "roc_curves_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    for idx, (name, res) in enumerate(models_results.items()):
        if res.get("y_proba") is not None:
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
            auc = res["metrics"]["roc_auc"]
            ax.plot(fpr, tpr, lw=2.2, label=f"{name.replace('_', ' ').title()} (AUC = {auc:.4f})", color=PALETTE[idx % len(PALETTE)])

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title(f"ROC Curves - Model Comparison ({cfg['short_name']})")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved ROC curves figure to {output_path}")
    return output_path


def plot_precision_recall_curves(models_results: Dict[str, Any], dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plots comparative Precision-Recall curves."""
    cfg = get_dataset_config(dataset)
    output_path = save_path or cfg["figures_dir"] / "precision_recall_curves_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    for idx, (name, res) in enumerate(models_results.items()):
        if res.get("y_proba") is not None:
            precision, recall, _ = precision_recall_curve(res["y_true"], res["y_proba"])
            ap = res["metrics"].get("average_precision", 0.0)
            ax.plot(recall, precision, lw=2.2, label=f"{name.replace('_', ' ').title()} (AP = {ap:.4f})", color=PALETTE[idx % len(PALETTE)])

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title(f"Precision-Recall Curves ({cfg['short_name']})")
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved Precision-Recall curves figure to {output_path}")
    return output_path


def plot_metrics_comparison_bar(models_results: Dict[str, Any], dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plots multi-metric grouped bar chart comparing all models."""
    cfg = get_dataset_config(dataset)
    output_path = save_path or cfg["figures_dir"] / "models_metrics_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_names = ["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"]
    display_names = ["Accuracy", "Precision", "Recall (Sens.)", "Specificity", "F1-Score", "ROC-AUC"]

    data = []
    for model_name, res in models_results.items():
        for m_key, m_disp in zip(metrics_names, display_names):
            val = res["metrics"].get(m_key)
            if val is not None:
                data.append({
                    "Model": model_name.replace("_", " ").title(),
                    "Metric": m_disp,
                    "Score": val
                })

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="muted", ax=ax, edgecolor="black", alpha=0.9)
    ax.set_ylim(0.5, 1.02)
    ax.set_title(f"Diagnostic Performance Metrics Comparison ({cfg['short_name']})", pad=12)
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.legend(title="Model Family", loc="lower right")

    for p in ax.patches:
        h = p.get_height()
        if h > 0.1:
            ax.annotate(f"{h:.2f}",
                        xy=(p.get_x() + p.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, rotation=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved metrics comparison bar figure to {output_path}")
    return output_path


def perform_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    dataset: str = DEFAULT_DATASET,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Identifies and classifies all test-set misclassifications into False Negatives and False Positives.
    """
    cfg = get_dataset_config(dataset)
    target_col = cfg["target_column"]
    output_path = save_path or cfg["metrics_dir"] / "error_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_errors = X_test.copy()
    df_errors["true_target"] = y_test.values
    df_errors["predicted_target"] = y_pred
    df_errors["predicted_proba"] = np.round(y_proba, 4)
    df_errors["prediction_correct"] = df_errors["true_target"] == df_errors["predicted_target"]

    error_type = []
    for _, row in df_errors.iterrows():
        if row["true_target"] == 1 and row["predicted_target"] == 0:
            error_type.append("False Negative (Missed Disease)")
        elif row["true_target"] == 0 and row["predicted_target"] == 1:
            error_type.append("False Positive (False Alarm)")
        else:
            error_type.append("Correct")

    df_errors["error_type"] = error_type
    misclassified = df_errors[~df_errors["prediction_correct"]].copy()

    # Index column
    misclassified.insert(0, "test_set_index", misclassified.index)
    misclassified.to_csv(output_path, index=False)
    logger.info(f"[{cfg['short_name']}] Saved error analysis ({len(misclassified)} errors) to {output_path}")
    return misclassified
