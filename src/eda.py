"""
Exploratory Data Analysis (EDA) module for CardioSense.
Supports dual-dataset analysis:
1. India Hospital Cohort (Primary): 1000 subjects
2. UCI Cleveland Benchmark (Historical): 303 subjects

Generates research-quality visualizations and statistical summaries.
All figures are saved to dataset-specific directories (reports/figures/india and reports/figures/uci).
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_DATASET,
    get_dataset_config,
    DATASETS_CONFIG,
    FIGURES_DIR,
    PROJECT_ROOT
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

PALETTE = ["#1f77b4", "#ff7f0e"]  # Blue = Absence (0), Orange = Presence (1)


def plot_target_distribution(df: pd.DataFrame, dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plot target class distribution with counts and percentages."""
    cfg = get_dataset_config(dataset)
    target_col = cfg["target_column"]
    output_path = save_path or cfg["figures_dir"] / "target_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[target_col].value_counts().sort_index()
    percentages = (counts / len(df)) * 100

    bars = ax.bar(["Absence (0)", "Presence (1)"], counts.values, color=PALETTE, width=0.5, edgecolor="black", alpha=0.85)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f"{height} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_title(f"Target Distribution: {cfg['short_name']}")
    ax.set_ylabel("Patient Count")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved target distribution figure to {output_path}")
    return output_path


def plot_numeric_distributions(df: pd.DataFrame, dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plot histograms and KDEs for continuous numeric features."""
    cfg = get_dataset_config(dataset)
    target_col = cfg["target_column"]
    num_features = cfg["numeric_features"]
    output_path = save_path or cfg["figures_dir"] / "numeric_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_feats = len(num_features)
    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, feature in enumerate(num_features):
        ax = axes[idx]
        sns.histplot(
            data=df, x=feature, hue=target_col,
            kde=True, element="step", stat="density", common_norm=False,
            palette=PALETTE, ax=ax, alpha=0.4
        )
        ax.set_title(f"Distribution of {feature.upper()}")
        ax.set_xlabel(feature)

    # Clean up empty subplots
    for idx in range(n_feats, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"Continuous Attribute Distributions by Target ({cfg['short_name']})", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved numeric distributions figure to {output_path}")
    return output_path


def plot_numeric_boxplots(df: pd.DataFrame, dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plot boxplots to visualize clinical outliers across continuous features."""
    cfg = get_dataset_config(dataset)
    target_col = cfg["target_column"]
    num_features = cfg["numeric_features"]
    output_path = save_path or cfg["figures_dir"] / "numeric_boxplots.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_feats = len(num_features)
    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, feature in enumerate(num_features):
        ax = axes[idx]
        sns.boxplot(data=df, x=target_col, y=feature, palette=PALETTE, ax=ax, width=0.4)
        ax.set_title(f"Boxplot of {feature.upper()}")
        ax.set_xticklabels(["Absence (0)", "Presence (1)"])

    for idx in range(n_feats, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"Outlier & Dispersion Boxplots ({cfg['short_name']})", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved numeric boxplots figure to {output_path}")
    return output_path


def plot_correlation_heatmap(df: pd.DataFrame, dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plot correlation matrix across all clinical features and target."""
    cfg = get_dataset_config(dataset)
    output_path = save_path or cfg["figures_dir"] / "correlation_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = cfg["feature_names"] + [cfg["target_column"]]
    valid_cols = [c for c in cols if c in df.columns]
    corr = df[valid_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8.5})
    ax.set_title(f"Pearson Correlation Matrix ({cfg['short_name']})", pad=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved correlation heatmap to {output_path}")
    return output_path


def plot_categorical_distributions(df: pd.DataFrame, dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plot categorical feature proportions stratified by heart disease target."""
    cfg = get_dataset_config(dataset)
    target_col = cfg["target_column"]
    cat_features = cfg["categorical_features"]
    output_path = save_path or cfg["figures_dir"] / "categorical_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_feats = len(cat_features)
    n_cols = 4 if n_feats >= 8 else 3
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, feature in enumerate(cat_features):
        ax = axes[idx]
        sns.countplot(data=df, x=feature, hue=target_col, palette=PALETTE, ax=ax, alpha=0.85)
        ax.set_title(f"{feature.upper()} vs. Target")
        ax.set_ylabel("Count")

    for idx in range(n_feats, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"Categorical Predictor Proportions by Target ({cfg['short_name']})", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved categorical distributions figure to {output_path}")
    return output_path


def plot_feature_target_relationships(df: pd.DataFrame, dataset: str = DEFAULT_DATASET, save_path: Optional[Path] = None) -> Path:
    """Plot key bivariate relationships highlighting risk differentiation."""
    cfg = get_dataset_config(dataset)
    target_col = cfg["target_column"]
    output_path = save_path or cfg["figures_dir"] / "feature_target_relationships.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if dataset == "india":
        # India features: maxheartrate, age, restingBP, serumcholestrol
        sns.scatterplot(data=df, x="age", y="maxheartrate", hue=target_col, palette=PALETTE, ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title("Age vs. Maximum Heart Rate")

        sns.scatterplot(data=df, x="restingBP", y="serumcholestrol", hue=target_col, palette=PALETTE, ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_title("Resting BP vs. Serum Cholesterol")

        sns.boxplot(data=df, x="chestpain", y="oldpeak", hue=target_col, palette=PALETTE, ax=axes[1, 0])
        axes[1, 0].set_title("Chest Pain Type vs. ST Depression")

        sns.countplot(data=df, x="noofmajorvessels", hue=target_col, palette=PALETTE, ax=axes[1, 1])
        axes[1, 1].set_title("Major Vessels vs. Target Prevalence")
    else:
        # UCI features
        sns.scatterplot(data=df, x="age", y="thalach", hue=target_col, palette=PALETTE, ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title("Age vs. Maximum Heart Rate (thalach)")

        sns.scatterplot(data=df, x="trestbps", y="chol", hue=target_col, palette=PALETTE, ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_title("Resting BP vs. Serum Cholesterol")

        sns.boxplot(data=df, x="cp", y="oldpeak", hue=target_col, palette=PALETTE, ax=axes[1, 0])
        axes[1, 0].set_title("Chest Pain Type vs. ST Depression")

        sns.countplot(data=df, x="ca", hue=target_col, palette=PALETTE, ax=axes[1, 1])
        axes[1, 1].set_title("Major Vessels (ca) vs. Target Prevalence")

    fig.suptitle(f"Bivariate Clinical Risk Relationships ({cfg['short_name']})", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{cfg['short_name']}] Saved bivariate relationships figure to {output_path}")
    return output_path


def run_eda_pipeline(dataset: str = DEFAULT_DATASET) -> Dict[str, Path]:
    """Runs full EDA suite for specified dataset."""
    cfg = get_dataset_config(dataset)
    proc_file: Path = cfg["processed_file"]

    if not proc_file.exists():
        from src.data_loader import prepare_data_pipeline
        prepare_data_pipeline(dataset=dataset)

    df = pd.read_csv(proc_file)
    logger.info(f"[{cfg['short_name']}] Running EDA on {proc_file} ({df.shape[0]} rows)")

    figs = {
        "target_distribution": plot_target_distribution(df, dataset=dataset),
        "numeric_distributions": plot_numeric_distributions(df, dataset=dataset),
        "numeric_boxplots": plot_numeric_boxplots(df, dataset=dataset),
        "correlation_heatmap": plot_correlation_heatmap(df, dataset=dataset),
        "categorical_distributions": plot_categorical_distributions(df, dataset=dataset),
        "feature_target_relationships": plot_feature_target_relationships(df, dataset=dataset)
    }

    # If dataset matches default, also ensure root reports/figures has current primary figures
    if dataset == DEFAULT_DATASET:
        root_fig_dir = PROJECT_ROOT / "reports" / "figures"
        root_fig_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for name, path in figs.items():
            shutil.copy(path, root_fig_dir / path.name)

    logger.info(f"[{cfg['short_name']}] Complete EDA suite generated successfully.")
    return figs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioSense EDA Suite")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=["india", "uci", "all"],
                        help="Dataset for EDA: 'india' (default), 'uci', or 'all'")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["india", "uci"]:
            print(f"\n{'='*50}\nRunning EDA for {ds.upper()} Dataset\n{'='*50}")
            run_eda_pipeline(dataset=ds)
    else:
        run_eda_pipeline(dataset=args.dataset)
