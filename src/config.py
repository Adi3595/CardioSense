"""
Configuration module for CardioSense ML pipeline.
Defines paths, feature definitions, target mappings, hyperparameter spaces,
and random seeds for full reproducibility.
"""

from pathlib import Path
from typing import Dict, List, Any

# Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "processed.cleveland.data"
RAW_METADATA_FILE = RAW_DATA_DIR / "raw_metadata.json"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_cleaned.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Models and Artifacts Paths
MODELS_DIR = PROJECT_ROOT / "models"
BEST_PIPELINE_FILE = MODELS_DIR / "best_pipeline.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Reports Paths
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
AUDIT_DIR = REPORTS_DIR / "dataset_audit"

AUDIT_REPORT_FILE = AUDIT_DIR / "dataset_audit_report.md"
FEATURE_DICT_FILE = AUDIT_DIR / "feature_dictionary.csv"
DATA_QUALITY_SUMMARY_FILE = AUDIT_DIR / "data_quality_summary.json"
MODEL_COMPARISON_JSON = METRICS_DIR / "model_comparison.json"
MODEL_COMPARISON_CSV = METRICS_DIR / "model_comparison.csv"
TEST_METRICS_JSON = METRICS_DIR / "test_metrics.json"
ERROR_ANALYSIS_CSV = METRICS_DIR / "error_analysis.csv"

# Dataset Source Information
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATASET_SOURCE_INFO = {
    "name": "UCI Heart Disease (Cleveland database)",
    "primary_url": UCI_URL,
    "mirror_url": "https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data",
    "citation": "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository.",
    "expected_rows": 303,
    "expected_raw_columns": 14,
    "missing_value_placeholders": ["?", "NA", "nan", ""]
}

# Schema & Features
RAW_COLUMN_NAMES: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num"
]

FEATURE_NAMES: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
]

TARGET_RAW_NAME = "num"
TARGET_NAME = "target"

# Feature Types
NUMERIC_FEATURES: List[str] = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

# Discrete and categorical features
CATEGORICAL_FEATURES: List[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

# Feature Documentation Dictionary
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "age": {
        "description": "Age in years",
        "type": "numeric",
        "unit": "years",
        "valid_range": (18, 100),
        "preprocessing": "Median imputation + StandardScaler"
    },
    "sex": {
        "description": "Sex (1 = male; 0 = female)",
        "type": "categorical_binary",
        "valid_values": [0, 1],
        "preprocessing": "Passthrough / OneHot"
    },
    "cp": {
        "description": "Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)",
        "type": "categorical_nominal",
        "valid_values": [1, 2, 3, 4],
        "preprocessing": "OneHotEncoder(handle_unknown='ignore')"
    },
    "trestbps": {
        "description": "Resting blood pressure on admission to hospital",
        "type": "numeric",
        "unit": "mm Hg",
        "valid_range": (80, 250),
        "preprocessing": "Median imputation + StandardScaler"
    },
    "chol": {
        "description": "Serum cholesterol",
        "type": "numeric",
        "unit": "mg/dl",
        "valid_range": (100, 600),
        "preprocessing": "Median imputation + StandardScaler"
    },
    "fbs": {
        "description": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
        "type": "categorical_binary",
        "valid_values": [0, 1],
        "preprocessing": "Passthrough / OneHot"
    },
    "restecg": {
        "description": "Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)",
        "type": "categorical_nominal",
        "valid_values": [0, 1, 2],
        "preprocessing": "OneHotEncoder(handle_unknown='ignore')"
    },
    "thalach": {
        "description": "Maximum heart rate achieved during exercise test",
        "type": "numeric",
        "unit": "bpm",
        "valid_range": (60, 230),
        "preprocessing": "Median imputation + StandardScaler"
    },
    "exang": {
        "description": "Exercise induced angina (1 = yes; 0 = no)",
        "type": "categorical_binary",
        "valid_values": [0, 1],
        "preprocessing": "Passthrough / OneHot"
    },
    "oldpeak": {
        "description": "ST depression induced by exercise relative to rest",
        "type": "numeric",
        "unit": "mm",
        "valid_range": (0.0, 10.0),
        "preprocessing": "Median imputation + StandardScaler"
    },
    "slope": {
        "description": "Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)",
        "type": "categorical_nominal",
        "valid_values": [1, 2, 3],
        "preprocessing": "OneHotEncoder(handle_unknown='ignore')"
    },
    "ca": {
        "description": "Number of major vessels (0-3) colored by fluoroscopy",
        "type": "categorical_ordinal_discrete",
        "valid_values": [0, 1, 2, 3],
        "preprocessing": "Most frequent imputation + OneHot"
    },
    "thal": {
        "description": "Thalassemia status (3 = normal; 6 = fixed defect; 7 = reversible defect)",
        "type": "categorical_nominal",
        "valid_values": [3, 6, 7],
        "preprocessing": "Most frequent imputation + OneHot"
    },
    "target": {
        "description": "Diagnosis of heart disease (0 = absence [< 50% diameter narrowing], 1 = presence [> 50% diameter narrowing])",
        "type": "binary_target",
        "valid_values": [0, 1],
        "preprocessing": "Binary encoding (0 -> 0; 1,2,3,4 -> 1)"
    }
}

# Experimental Controls
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PRIMARY_SELECTION_METRIC = "roc_auc"  # Justified: measures threshold-independent discrimination on imbalanced/small data

# Hyperparameter Grids (Restrained to prevent overfitting on N=303)
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
        "classifier__max_iter": [1000]
    },
    "random_forest": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"]
    },
    "xgboost": {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 4, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
        "classifier__reg_lambda": [1.0, 5.0]
    }
}

# Legal & Medical Disclaimer
DISCLAIMER_TEXT = (
    "DISCLAIMER: CardioSense is an academic research and educational prototype developed "
    "for studying explainable machine learning methodologies. It is NOT clinically validated, "
    "NOT certified as a medical device (CE/FDA), and MUST NOT be used for real-world medical diagnosis, "
    "risk assessment, or treatment decisions."
)
