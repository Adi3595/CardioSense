"""Unit tests for preprocessing, imputation, and leakage prevention across datasets."""
import numpy as np
import pandas as pd
import pytest

from src.config import get_dataset_config
from src.preprocessing import create_preprocessor, get_feature_names_out


@pytest.fixture
def synthetic_uci_train_df():
    """Synthetic UCI Cleveland training partition fixture."""
    data = {
        "age": [50.0, 60.0, 70.0, 45.0, 55.0],
        "sex": [1, 0, 1, 0, 1],
        "cp": [1, 2, 3, 4, 2],
        "trestbps": [120.0, 140.0, 160.0, 110.0, 130.0],
        "chol": [200.0, 250.0, 300.0, 180.0, 220.0],
        "fbs": [0, 1, 0, 0, 1],
        "restecg": [0, 1, 2, 0, 1],
        "thalach": [150.0, 140.0, 130.0, 170.0, 160.0],
        "exang": [0, 1, 1, 0, 0],
        "oldpeak": [1.0, 2.5, 0.0, 1.2, 0.5],
        "slope": [1, 2, 3, 1, 2],
        "ca": [0.0, 1.0, 2.0, np.nan, 0.0],  # NaN to test imputation
        "thal": [3.0, 6.0, 7.0, 3.0, np.nan]  # NaN to test imputation
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_india_train_df():
    """Synthetic India Hospital training partition fixture."""
    data = {
        "age": [50.0, 60.0, 70.0, 45.0, 55.0],
        "gender": [1, 0, 1, 0, 1],
        "chestpain": [0, 1, 2, 3, 1],
        "restingBP": [120.0, 140.0, 160.0, 110.0, 130.0],
        "serumcholestrol": [200.0, 250.0, np.nan, 180.0, 220.0],  # NaN to test imputation
        "fastingbloodsugar": [0, 1, 0, 0, 1],
        "restingrelectro": [0, 1, 2, 0, 1],
        "maxheartrate": [150.0, 140.0, 130.0, 170.0, 160.0],
        "exerciseangia": [0, 1, 1, 0, 0],
        "oldpeak": [1.0, 2.5, 0.0, 1.2, 0.5],
        "slope": [1, 2, 3, 1, 2],
        "noofmajorvessels": [0, 1, 2, 0, 1]
    }
    return pd.DataFrame(data)


def test_uci_preprocessor_fit_transform_shape(synthetic_uci_train_df):
    """Test that UCI preprocessor fits, imputes NaNs, and transforms categorical features."""
    preprocessor = create_preprocessor(dataset="uci", include_scaling=True)
    transformed = preprocessor.fit_transform(synthetic_uci_train_df)
    
    assert transformed.shape[0] == len(synthetic_uci_train_df)
    assert not np.isnan(transformed).any()
    
    feature_names = get_feature_names_out(preprocessor)
    assert len(feature_names) == transformed.shape[1]


def test_india_preprocessor_fit_transform_shape(synthetic_india_train_df):
    """Test that India preprocessor fits, imputes NaNs, and handles 12 clinical predictors."""
    preprocessor = create_preprocessor(dataset="india", include_scaling=True)
    transformed = preprocessor.fit_transform(synthetic_india_train_df)
    
    assert transformed.shape[0] == len(synthetic_india_train_df)
    assert not np.isnan(transformed).any()
    
    feature_names = get_feature_names_out(preprocessor)
    assert len(feature_names) == transformed.shape[1]


def test_unseen_categories_handling(synthetic_uci_train_df, synthetic_india_train_df):
    """Test that OneHotEncoder handles unknown categories gracefully without error."""
    # Test on UCI
    preprocessor_uci = create_preprocessor(dataset="uci", include_scaling=False)
    preprocessor_uci.fit(synthetic_uci_train_df)
    test_df_uci = synthetic_uci_train_df.copy()
    test_df_uci.loc[0, "cp"] = 99
    transformed_uci = preprocessor_uci.transform(test_df_uci)
    assert transformed_uci.shape[0] == len(test_df_uci)
    assert not np.isnan(transformed_uci).any()

    # Test on India
    preprocessor_india = create_preprocessor(dataset="india", include_scaling=False)
    preprocessor_india.fit(synthetic_india_train_df)
    test_df_india = synthetic_india_train_df.copy()
    test_df_india.loc[0, "chestpain"] = 99
    transformed_india = preprocessor_india.transform(test_df_india)
    assert transformed_india.shape[0] == len(test_df_india)
    assert not np.isnan(transformed_india).any()


def test_scaling_disabled_preserves_values(synthetic_india_train_df):
    """Test that setting include_scaling=False does not standardize continuous features."""
    preprocessor_no_scale = create_preprocessor(dataset="india", include_scaling=False)
    transformed = preprocessor_no_scale.fit_transform(synthetic_india_train_df)
    
    # First numeric feature is age
    assert np.allclose(transformed[:, 0], synthetic_india_train_df["age"].values)
