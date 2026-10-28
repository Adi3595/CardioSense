"""Unit tests for model training and pipeline persistence across datasets."""
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import build_pipeline
from src.config import BEST_PIPELINE_FILE, MODELS_DIR, get_dataset_config


@pytest.fixture
def small_uci_dataset():
    """Small synthetic dataset for fast UCI training verification."""
    np.random.seed(42)
    n = 30
    cfg = get_dataset_config("uci")
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


@pytest.fixture
def small_india_dataset():
    """Small synthetic dataset for fast India training verification."""
    np.random.seed(42)
    n = 30
    cfg = get_dataset_config("india")
    data = {
        "age": np.random.uniform(25, 75, n),
        "gender": np.random.choice([0, 1], n),
        "chestpain": np.random.choice([0, 1, 2, 3], n),
        "restingBP": np.random.uniform(95, 195, n),
        "serumcholestrol": np.random.uniform(140, 380, n),
        "fastingbloodsugar": np.random.choice([0, 1], n),
        "restingrelectro": np.random.choice([0, 1, 2], n),
        "maxheartrate": np.random.uniform(90, 210, n),
        "exerciseangia": np.random.choice([0, 1], n),
        "oldpeak": np.random.uniform(0, 5, n),
        "slope": np.random.choice([1, 2, 3], n),
        "noofmajorvessels": np.random.choice([0, 1, 2, 3], n),
        "target": np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)


def test_pipeline_fit_and_predict_uci(small_uci_dataset):
    """Test full pipeline fits and produces valid binary predictions on UCI schema."""
    cfg = get_dataset_config("uci")
    X = small_uci_dataset[cfg["feature_names"]]
    y = small_uci_dataset["target"]

    pipeline = build_pipeline(LogisticRegression(max_iter=200), dataset="uci", include_scaling=True)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_pipeline_fit_and_predict_india(small_india_dataset):
    """Test full pipeline fits and produces valid binary predictions on India schema."""
    cfg = get_dataset_config("india")
    X = small_india_dataset[cfg["feature_names"]]
    y = small_india_dataset["target"]

    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42), dataset="india", include_scaling=False)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_saved_india_champion_pipeline_reproducibility():
    """Test that the primary saved India champion pipeline artifact can be loaded and executed."""
    india_model_path = MODELS_DIR / "india" / "best_pipeline.joblib"
    if not india_model_path.exists():
        india_model_path = BEST_PIPELINE_FILE
    assert india_model_path.exists(), f"Missing artifact: {india_model_path}"

    pipeline = joblib.load(india_model_path)
    cfg = get_dataset_config("india")
    sample_patient = pd.DataFrame([{
        "age": 52.0,
        "gender": 1,
        "chestpain": 0,
        "restingBP": 150.0,
        "serumcholestrol": 240.0,
        "fastingbloodsugar": 0,
        "restingrelectro": 1,
        "maxheartrate": 155.0,
        "exerciseangia": 1,
        "oldpeak": 1.8,
        "slope": 2,
        "noofmajorvessels": 1
    }])[cfg["feature_names"]]

    pred = pipeline.predict(sample_patient)
    probs = pipeline.predict_proba(sample_patient)

    assert pred[0] in [0, 1]
    assert 0.0 <= probs[0][1] <= 1.0


def test_saved_uci_benchmark_pipeline_reproducibility():
    """Test that the saved UCI benchmark pipeline artifact can be loaded and executed."""
    uci_model_path = MODELS_DIR / "uci" / "best_pipeline.joblib"
    assert uci_model_path.exists(), f"Missing artifact: {uci_model_path}"

    pipeline = joblib.load(uci_model_path)
    cfg = get_dataset_config("uci")
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
    }])[cfg["feature_names"]]

    pred = pipeline.predict(sample_patient)
    probs = pipeline.predict_proba(sample_patient)

    assert pred[0] in [0, 1]
    assert 0.0 <= probs[0][1] <= 1.0
