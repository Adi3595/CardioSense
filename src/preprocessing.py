"""
Preprocessing module for CardioSense.
Builds leakage-safe Scikit-Learn ColumnTransformer and Pipeline objects.
Guarantees that all imputers, scalers, and encoders are fit solely on training data.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_preprocessor(include_scaling: bool = True) -> ColumnTransformer:
    """
    Creates a ColumnTransformer for preprocessing clinical attributes.
    
    Parameters:
    -----------
    include_scaling : bool
        If True, applies StandardScaler to numeric features (vital for Logistic Regression).
        If False, passes numeric features through after median imputation (suitable for tree-based models).
    
    Returns:
    --------
    ColumnTransformer
    """
    if include_scaling:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"  # Guarantee no stray or label-leaking columns pass through
    )

    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts transformed feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps"):
            last_step = list(trans.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
                feature_names.extend(names)
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)

    return [str(f) for f in feature_names]


def build_pipeline(estimator, include_scaling: bool = True) -> Pipeline:
    """
    Combines preprocessor and estimator into a unified inference pipeline.
    
    Parameters:
    -----------
    estimator : scikit-learn compatible classifier
    include_scaling : bool
        Whether to scale numeric features.
    
    Returns:
    --------
    Pipeline
    """
    preprocessor = create_preprocessor(include_scaling=include_scaling)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    return pipeline
