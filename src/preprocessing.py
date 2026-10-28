"""
Preprocessing module for CardioSense.
Builds leakage-safe Scikit-Learn ColumnTransformer and Pipeline objects.
Supports both:
1. India Hospital Cohort (Primary): 12 features (5 numeric, 7 categorical)
2. UCI Cleveland Benchmark (Historical): 13 features (5 numeric, 8 categorical)

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
    DEFAULT_DATASET,
    get_dataset_config,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_preprocessor(dataset: str = DEFAULT_DATASET, include_scaling: bool = True) -> ColumnTransformer:
    """
    Creates a ColumnTransformer for preprocessing clinical attributes.
    
    Parameters:
    -----------
    dataset : str
        Dataset key ('india' or 'uci').
    include_scaling : bool
        If True, applies StandardScaler to numeric features (vital for Logistic Regression).
        If False, passes numeric features through after median imputation (suitable for tree-based models).
    
    Returns:
    --------
    ColumnTransformer
    """
    try:
        cfg = get_dataset_config(dataset)
        num_cols = cfg["numeric_features"]
        cat_cols = cfg["categorical_features"]
    except Exception:
        # Fallback to active config constants if legacy call
        num_cols = NUMERIC_FEATURES
        cat_cols = CATEGORICAL_FEATURES

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
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
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


def build_pipeline(estimator, dataset: str = DEFAULT_DATASET, include_scaling: bool = True) -> Pipeline:
    """
    Combines preprocessor and estimator into a unified inference pipeline.
    
    Parameters:
    -----------
    estimator : scikit-learn compatible classifier
    dataset : str
        Dataset key ('india' or 'uci').
    include_scaling : bool
        Whether to scale numeric features.
    
    Returns:
    --------
    Pipeline
    """
    preprocessor = create_preprocessor(dataset=dataset, include_scaling=include_scaling)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    return pipeline
