"""Unit tests for model training and pipeline persistence."""
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import build_pipeline
from src.config import BEST_PIPELINE_FILE, FEATURE_NAMES


@pytest.fixture
def small_dataset():
    """Small synthetic dataset for fast training verification."""
    np.random.seed(42)
    n = 30
    data = {
        "age": np.random.uniform(30, 75, n),
        "sex": np.random.choice([0, 1], n),
        "cp": np.random.choice([1, 2, 3, 4], n),
        "trestbps": np.random.uniform(100, 180, n),
        "chol": np.random.uniform(150, 350, n),
        "fbs": np.random.choice([0, 1], n),
        "restecg": np.random.choice([0, 1, 2], n),
        "thalach": np.random.uniform(90, 190, n),
        "exang": np.random.choice([0, 1], n),
        "oldpeak": np.random.uniform(0, 5, n),
        "slope": np.random.choice([1, 2, 3], n),
        "ca": np.random.choice([0.0, 1.0, 2.0, np.nan], n),
        "thal": np.random.choice([3.0, 6.0, 7.0, np.nan], n),
        "target": np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)


def test_pipeline_fit_and_predict(small_dataset):
    """Test full pipeline fits and produces valid binary predictions."""
    X = small_dataset[FEATURE_NAMES]
    y = small_dataset["target"]

    pipeline = build_pipeline(LogisticRegression(max_iter=200), include_scaling=True)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_saved_champion_pipeline_reproducibility():
    """Test that the saved champion pipeline artifact can be loaded and executed."""
    assert BEST_PIPELINE_FILE.exists(), f"Missing artifact: {BEST_PIPELINE_FILE}"

    pipeline = joblib.load(BEST_PIPELINE_FILE)
    sample_patient = pd.DataFrame([{
        "age": 60.0,
        "sex": 1,
        "cp": 4,
        "trestbps": 140.0,
        "chol": 260.0,
        "fbs": 0,
        "restecg": 1,
        "thalach": 140.0,
        "exang": 1,
        "oldpeak": 2.0,
        "slope": 2,
        "ca": 1.0,
        "thal": 7.0
    }])[FEATURE_NAMES]

    pred = pipeline.predict(sample_patient)
    probs = pipeline.predict_proba(sample_patient)

    assert pred[0] in [0, 1]
    assert 0.0 <= probs[0][1] <= 1.0
