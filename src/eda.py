"""
Exploratory Data Analysis (EDA) module for CardioSense.
Generates research-quality visualizations and statistical summaries.
All figures are saved to reports/figures/.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    PROCESSED_DATA_FILE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TARGET_NAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 15,
    "figure.dpi": 300
})

PALETTE = ["#1f77b4", "#ff7f0e"]  # Blue = Absence (0), Orange/Red = Presence (1)


def plot_target_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot target class distribution with counts and percentages."""
    output_path = save_path or FIGURES_DIR / "target_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[TARGET_NAME].value_counts().sort_index()
    percentages = (counts / len(df)) * 100

    bars = ax.bar(["Absence (0)", "Presence (1)"], counts.values, color=PALETTE, width=0.5, edgecolor="black", alpha=0.85)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f"{height} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_title("Distribution of Heart Disease Target (Cleveland Dataset)")
    ax.set_ylabel("Patient Count")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved target distribution figure to {output_path}")
    return output_path


def plot_numeric_distributions(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot histograms and KDEs for all continuous numeric features."""
    output_path = save_path or FIGURES_DIR / "numeric_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, feature in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        sns.histplot(
            data=df, x=feature, hue=TARGET_NAME,
            kde=True, element="step", stat="density", common_norm=False,
            palette=PALETTE, ax=ax, alpha=0.4
        )
        ax.set_title(f"Distribution of {feature.upper()}")
        ax.set_xlabel(feature)
        ax.legend(title="Target", labels=["Presence (1)", "Absence (0)"])

    # Remove extra subplot
    if len(NUMERIC_FEATURES) < len(axes):
        fig.delaxes(axes[-1])

    fig.suptitle("Continuous Clinical Attribute Distributions by Target Class", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved numeric distributions figure to {output_path}")
    return output_path


def plot_numeric_boxplots(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot boxplots for numeric features across target classes to inspect spreads and outliers."""
    output_path = save_path or FIGURES_DIR / "numeric_boxplots.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(18, 5))

    for idx, feature in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        sns.boxplot(data=df, x=TARGET_NAME, y=feature, hue=TARGET_NAME, palette=PALETTE, ax=ax, width=0.4, fliersize=4, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Absence (0)", "Presence (1)"])
        ax.set_title(feature.upper())
        ax.set_xlabel("Target Class")

    fig.suptitle("Clinical Attribute Boxplots by Heart Disease Status", y=1.03)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved numeric boxplots to {output_path}")
    return output_path


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot Pearson correlation matrix heatmap for numeric attributes and target."""
    output_path = save_path or FIGURES_DIR / "correlation_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = NUMERIC_FEATURES + [TARGET_NAME]
    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-0.6, vmax=0.6, center=0, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Pearson Correlation Heatmap (Continuous Features & Target)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved correlation heatmap to {output_path}")
    return output_path


def plot_categorical_distributions(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot categorical attribute counts stratified by target."""
    output_path = save_path or FIGURES_DIR / "categorical_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(cat_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(cat_cols):
        ax = axes[idx]
        sns.countplot(
            data=df, x=feature, hue=TARGET_NAME,
            palette=PALETTE, ax=ax, edgecolor="black", alpha=0.85
        )
        ax.set_title(f"Feature: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.legend(title="Target", labels=["Absence (0)", "Presence (1)"])

    # Remove unused subplots
    for idx in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Categorical Predictors Stratified by Heart Disease Status", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved categorical distributions to {output_path}")
    return output_path


def plot_feature_target_relationships(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Key diagnostic bivariate relationships showing clinical risk trade-offs."""
    output_path = save_path or FIGURES_DIR / "feature_target_relationships.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Age vs Thalach (Max Heart Rate)
    sns.scatterplot(
        data=df, x="age", y="thalach", hue=TARGET_NAME,
        palette=PALETTE, alpha=0.8, s=60, edgecolor="k", ax=axes[0, 0]
    )
    axes[0, 0].set_title("Age vs. Maximum Heart Rate (thalach)")
    axes[0, 0].legend(title="Target", labels=["Absence", "Presence"])

    # 2. Chest pain (cp) vs Target
    cp_pct = pd.crosstab(df["cp"], df[TARGET_NAME], normalize="index") * 100
    cp_pct.plot(kind="bar", stacked=True, color=PALETTE, ax=axes[0, 1], edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("Heart Disease Prevalence by Chest Pain Type (cp)")
    axes[0, 1].set_ylabel("Percentage (%)")
    axes[0, 1].set_xticklabels(["1: Typ. Angina", "2: Atyp. Angina", "3: Non-anginal", "4: Asymptomatic"], rotation=20)
    axes[0, 1].legend(["Absence (0)", "Presence (1)"], title="Diagnosis")

    # 3. Oldpeak (ST depression) vs Slope
    sns.boxplot(
        data=df, x="slope", y="oldpeak", hue=TARGET_NAME,
        palette=PALETTE, ax=axes[1, 0]
    )
    axes[1, 0].set_title("ST Depression (oldpeak) by ST Segment Slope")
    axes[1, 0].set_xticks([0, 1, 2])
    axes[1, 0].set_xticklabels(["1: Upsloping", "2: Flat", "3: Downsloping"])
    axes[1, 0].legend(title="Target", labels=["Absence", "Presence"])

    # 4. Thalassemia (thal) vs Target
    thal_pct = pd.crosstab(df["thal"], df[TARGET_NAME], normalize="index") * 100
    thal_pct.plot(kind="bar", stacked=True, color=PALETTE, ax=axes[1, 1], edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("Diagnosis Proportion by Thalassemia Status (thal)")
    axes[1, 1].set_ylabel("Percentage (%)")
    axes[1, 1].set_xticklabels(["3: Normal", "6: Fixed Defect", "7: Reversible"], rotation=0)
    axes[1, 1].legend(["Absence (0)", "Presence (1)"], title="Diagnosis")

    fig.suptitle("Key Clinical Risk Associations in Cleveland Heart Disease Dataset", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature-target relationships to {output_path}")
    return output_path


def run_eda_pipeline() -> Dict[str, str]:
    """Execute all EDA visualizations."""
    if not PROCESSED_DATA_FILE.exists():
        from src.data_loader import prepare_data_pipeline
        prepare_data_pipeline()

    df = pd.read_csv(PROCESSED_DATA_FILE)
    logger.info(f"Running EDA on dataset with shape {df.shape}")

    saved_plots = {
        "target_distribution": str(plot_target_distribution(df)),
        "numeric_distributions": str(plot_numeric_distributions(df)),
        "numeric_boxplots": str(plot_numeric_boxplots(df)),
        "correlation_heatmap": str(plot_correlation_heatmap(df)),
        "categorical_distributions": str(plot_categorical_distributions(df)),
        "feature_target_relationships": str(plot_feature_target_relationships(df))
    }
    return saved_plots


if __name__ == "__main__":
    run_eda_pipeline()
