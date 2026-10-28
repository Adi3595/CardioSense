"""Unit tests for data validation, schema integrity, and dual-dataset audits."""
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
from src.config import (
    get_dataset_config,
    FEATURE_NAMES,
    TARGET_NAME,
    TARGET_RAW_NAME
)


@pytest.fixture
def sample_raw_uci_df():
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


@pytest.fixture
def sample_raw_india_df():
    """Synthetic raw India Hospital cohort dataframe fixture."""
    data = {
        "patientid": [101, 102, 103, 104],
        "age": [52.0, 45.0, 61.0, 38.0],
        "gender": [1, 0, 1, 0],
        "chestpain": [0, 2, 3, 1],
        "restingBP": [150.0, 140.0, 165.0, 120.0],
        "serumcholestrol": [240.0, 0.0, 290.0, 210.0],  # 0.0 should be coerced to NaN
        "fastingbloodsugar": [0, 1, 0, 0],
        "restingrelectro": [1, 0, 2, 0],
        "maxheartrate": [155.0, 170.0, 120.0, 180.0],
        "exerciseangia": [1, 0, 1, 0],
        "oldpeak": [1.8, 0.0, 2.5, 0.4],
        "slope": [2, 1, 3, 1],
        "noofmajorvessels": [1, 0, 2, 0],
        "target": [1, 0, 1, 0]
    }
    return pd.DataFrame(data)


def test_uci_target_conversion_canonical(sample_raw_uci_df):
    """Test raw diagnosis values 0 -> 0, and 1..4 -> 1 for UCI benchmark."""
    processed = process_dataset(sample_raw_uci_df, dataset="uci")
    assert "target" in processed.columns
    assert "num" not in processed.columns
    assert processed["target"].tolist() == [0, 1, 1, 0]


def test_india_dataset_processing_and_zero_cholesterol(sample_raw_india_df):
    """Test India dataset processing: patientid dropped, zero cholesterol coerced to NaN."""
    processed = process_dataset(sample_raw_india_df, dataset="india")
    assert "patientid" not in processed.columns
    assert "target" in processed.columns
    assert np.isnan(processed.loc[1, "serumcholestrol"])
    assert processed.loc[0, "serumcholestrol"] == 240.0
    assert processed["target"].tolist() == [1, 0, 1, 0]


def test_validate_target_unexpected_values():
    """Test validation fails when invalid binary targets exist in processed data."""
    df_invalid = pd.DataFrame({"target": [0, 1, 2, -1]})
    with pytest.raises(ValidationError):
        validate_target(df_invalid, is_raw=False, dataset="india")
    with pytest.raises(ValidationError):
        validate_target(df_invalid, is_raw=False, dataset="uci")


def test_validate_schema_missing_columns(sample_raw_uci_df, sample_raw_india_df):
    """Test validation raises error when required feature columns are missing."""
    df_missing_uci = sample_raw_uci_df.drop(columns=["age", "chol"])
    with pytest.raises(ValidationError):
        validate_schema(df_missing_uci, is_raw=True, dataset="uci")

    df_missing_india = sample_raw_india_df.drop(columns=["gender", "restingBP"])
    with pytest.raises(ValidationError):
        validate_schema(df_missing_india, is_raw=True, dataset="india")


def test_missing_value_detection():
    """Test detection of NaN and placeholder missing values in UCI schema."""
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
    audit = audit_data_quality(df, dataset="uci")
    assert audit["columns"]["age"]["missing_count"] == 1
    assert audit["columns"]["ca"]["missing_count"] == 1
    assert audit["columns"]["thal"]["missing_count"] == 1
    assert audit["columns"]["sex"]["missing_count"] == 0


def test_duplicate_row_reporting():
    """Test duplicate row detection in audit."""
    df = pd.DataFrame({
        "age": [50.0, 50.0],
        "gender": [1, 1],
        "chestpain": [1, 1],
        "restingBP": [120.0, 120.0],
        "serumcholestrol": [200.0, 200.0],
        "fastingbloodsugar": [0, 0],
        "restingrelectro": [0, 0],
        "maxheartrate": [150.0, 150.0],
        "exerciseangia": [0, 0],
        "oldpeak": [1.0, 1.0],
        "slope": [1, 1],
        "noofmajorvessels": [0, 0],
        "target": [1, 1]
    })
    audit = audit_data_quality(df, dataset="india")
    assert audit["duplicate_rows"] == 1
