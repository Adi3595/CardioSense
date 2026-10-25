"""Unit tests for SHAP explainability and feature attribution."""
from pathlib import Path
import pandas as pd
import pytest

from src.config import FIGURES_DIR, METRICS_DIR, BEST_PIPELINE_FILE


def test_shap_artifacts_exist():
    """Verify that SHAP global and local artifacts have been generated."""
    summary_plot = FIGURES_DIR / "shap_summary_plot.png"
    bar_plot = FIGURES_DIR / "shap_feature_importance_bar.png"
    local_correct = FIGURES_DIR / "shap_local_correct.png"
    importance_csv = METRICS_DIR / "shap_feature_importance.csv"

    assert summary_plot.exists(), f"Missing: {summary_plot}"
    assert bar_plot.exists(), f"Missing: {bar_plot}"
    assert local_correct.exists(), f"Missing: {local_correct}"
    assert importance_csv.exists(), f"Missing: {importance_csv}"

    # Verify CSV content
    df_imp = pd.read_csv(importance_csv)
    assert "feature" in df_imp.columns
    assert "mean_absolute_shap" in df_imp.columns
    assert len(df_imp) > 0
    # Values should be strictly non-negative
    assert (df_imp["mean_absolute_shap"] >= 0).all()
