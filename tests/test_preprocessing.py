"""Unit tests for preprocessing, imputation, and leakage prevention."""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_NAMES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import create_preprocessor, get_feature_names_out


@pytest.fixture
def synthetic_train_df():
    """Synthetic training partition fixture."""
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


def test_preprocessor_fit_transform_shape(synthetic_train_df):
    """Test that preprocessor fits, imputes NaNs, and transforms categorical features."""
    preprocessor = create_preprocessor(include_scaling=True)
    transformed = preprocessor.fit_transform(synthetic_train_df)
    
    assert transformed.shape[0] == len(synthetic_train_df)
    # Check that no NaNs remain after transformation
    assert not np.isnan(transformed).any()
    
    feature_names = get_feature_names_out(preprocessor)
    assert len(feature_names) == transformed.shape[1]


def test_unseen_categories_handling(synthetic_train_df):
    """Test that OneHotEncoder handles unknown categories gracefully without error."""
    preprocessor = create_preprocessor(include_scaling=False)
    preprocessor.fit(synthetic_train_df)

    # Test set containing unseen category in cp (e.g., 99)
    test_df = synthetic_train_df.copy()
    test_df.loc[0, "cp"] = 99  # Unseen category

    transformed_test = preprocessor.transform(test_df)
    assert transformed_test.shape[0] == len(test_df)
    assert not np.isnan(transformed_test).any()


def test_scaling_disabled_preserves_values(synthetic_train_df):
    """Test that setting include_scaling=False does not standardize continuous features."""
    preprocessor_no_scale = create_preprocessor(include_scaling=False)
    transformed = preprocessor_no_scale.fit_transform(synthetic_train_df)
    
    # First 5 features are numeric
    # age values should match raw imputed age values (50, 60, 70, 45, 55)
    assert np.allclose(transformed[:, 0], synthetic_train_df["age"].values)
