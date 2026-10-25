"""Unit tests for data validation and schema integrity."""
import pytest
import numpy as np
import pandas as pd

from src.data_validation import (
    validate_schema,
    validate_target,
    audit_data_quality,
    ValidationError
)
from src.data_loader import process_dataset
from src.config import FEATURE_NAMES, TARGET_NAME, TARGET_RAW_NAME


@pytest.fixture
def sample_raw_df():
    """Synthetic raw Cleveland dataframe fixture."""
    data = {
        "age": [63.0, 67.0, 67.0, 37.0],
        "sex": [1.0, 1.0, 0.0, 1.0],
        "cp": [1.0, 4.0, 4.0, 3.0],
        "trestbps": [145.0, 160.0, 120.0, 130.0],
        "chol": [233.0, 286.0, 229.0, 250.0],
        "fbs": [1.0, 0.0, 0.0, 0.0],
        "restecg": [2.0, 2.0, 2.0, 0.0],
        "thalach": [150.0, 108.0, 129.0, 187.0],
        "exang": [0.0, 1.0, 1.0, 0.0],
        "oldpeak": [2.3, 1.5, 2.6, 3.5],
        "slope": [3.0, 2.0, 2.0, 3.0],
        "ca": [0.0, 3.0, 2.0, np.nan],
        "thal": [6.0, 3.0, 7.0, 3.0],
        "num": [0, 2, 1, 0]  # Raw target
    }
    return pd.DataFrame(data)


def test_target_conversion_canonical(sample_raw_df):
    """Test raw diagnosis values 0 -> 0, and 1..4 -> 1."""
    processed = process_dataset(sample_raw_df)
    assert TARGET_NAME in processed.columns
    assert TARGET_RAW_NAME not in processed.columns
    assert processed[TARGET_NAME].tolist() == [0, 1, 1, 0]


def test_validate_target_unexpected_values():
    """Test validation fails when invalid binary targets exist in processed data."""
    df_invalid = pd.DataFrame({TARGET_NAME: [0, 1, 2, -1]})
    with pytest.raises(ValidationError):
        validate_target(df_invalid, is_raw=False)


def test_validate_schema_missing_columns(sample_raw_df):
    """Test validation raises error when required feature columns are missing."""
    df_missing = sample_raw_df.drop(columns=["age", "chol"])
    with pytest.raises(ValidationError):
        validate_schema(df_missing, is_raw=True)


def test_missing_value_detection():
    """Test detection of NaN and placeholder missing values."""
    df = pd.DataFrame({
        "age": [50.0, np.nan],
        "sex": [1.0, 0.0],
        "cp": [1.0, 2.0],
        "trestbps": [120.0, 130.0],
        "chol": [200.0, 210.0],
        "fbs": [0.0, 0.0],
        "restecg": [0.0, 0.0],
        "thalach": [150.0, 160.0],
        "exang": [0.0, 0.0],
        "oldpeak": [1.0, 2.0],
        "slope": [1.0, 1.0],
        "ca": [np.nan, 0.0],
        "thal": [3.0, np.nan],
        "target": [0, 1]
    })
    audit = audit_data_quality(df)
    assert audit["columns"]["age"]["missing_count"] == 1
    assert audit["columns"]["ca"]["missing_count"] == 1
    assert audit["columns"]["thal"]["missing_count"] == 1
    assert audit["columns"]["sex"]["missing_count"] == 0


def test_duplicate_row_reporting():
    """Test duplicate row detection in audit."""
    df = pd.DataFrame({
        "age": [50.0, 50.0],
        "sex": [1.0, 1.0],
        "cp": [1.0, 1.0],
        "trestbps": [120.0, 120.0],
        "chol": [200.0, 200.0],
        "fbs": [0.0, 0.0],
        "restecg": [0.0, 0.0],
        "thalach": [150.0, 150.0],
        "exang": [0.0, 0.0],
        "oldpeak": [1.0, 1.0],
        "slope": [1.0, 1.0],
        "ca": [0.0, 0.0],
        "thal": [3.0, 3.0],
        "target": [1, 1]
    })
    audit = audit_data_quality(df)
    assert audit["duplicate_rows"] == 1
