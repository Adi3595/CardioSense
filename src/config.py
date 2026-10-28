"""
Configuration module for CardioSense ML pipeline.
Supports dual-dataset architecture:
1. India Hospital Cohort (Primary): Multi-specialty hospital in India (Doppala & Bhattacharyya, Mendeley Data, N=1000)
2. UCI Cleveland Benchmark (Historical baseline): Cleveland Clinic Foundation (Janosi et al., 1988, N=303)

Defines dataset schemas, file paths, validation constraints, hyperparameter search spaces,
and random seeds for full scientific reproducibility.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Top-level Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
AUDIT_DIR = REPORTS_DIR / "dataset_audit"

# Default Dataset Selection ("india" is primary, "uci" is historical benchmark)
DEFAULT_DATASET = os.getenv("CARDIO_DATASET", "india").lower()

# ==============================================================================
# DATASET CONFIGURATIONS: INDIA vs. UCI
# ==============================================================================

DATASETS_CONFIG: Dict[str, Dict[str, Any]] = {
    "india": {
        "name": "Indian Hospital Cardiovascular Disease Dataset",
        "short_name": "India Hospital Cohort",
        "origin": "Multi-specialty hospital, India",
        "authors": "Bhanu Prakash Doppala and Debnath Bhattacharyya",
        "doi": "10.17632/dzz48mvjht.1",
        "primary_url": "https://data.mendeley.com/datasets/dzz48mvjht/1",
        "mirror_url": "https://raw.githubusercontent.com/RishBez/CVD-Disease-Detection/main/Cardiovascular_Disease_Dataset.csv",
        "citation": "Doppala, B. P., & Bhattacharyya, D. (2021). Cardiovascular_Disease_Dataset. Mendeley Data, V1, doi: 10.17632/dzz48mvjht.1",
        "sha256": "9376367dde41e79e432d6402ea5d9e44223fc318ff54ed09966d721923947d1e",
        "expected_rows": 1000,
        "expected_raw_columns": 14,
        "id_column": "patientid",
        "target_raw_column": "target",
        "target_column": "target",
        "raw_file": RAW_DATA_DIR / "india" / "Cardiovascular_Disease_Dataset.csv",
        "metadata_file": RAW_DATA_DIR / "india" / "raw_metadata.json",
        "processed_file": PROCESSED_DATA_DIR / "india" / "heart_disease_cleaned.csv",
        "train_file": PROCESSED_DATA_DIR / "india" / "train.csv",
        "test_file": PROCESSED_DATA_DIR / "india" / "test.csv",
        "models_dir": MODELS_DIR / "india",
        "figures_dir": FIGURES_DIR / "india",
        "metrics_dir": METRICS_DIR / "india",
        "audit_dir": AUDIT_DIR / "india",
        "raw_column_names": [
            "patientid", "age", "gender", "chestpain", "restingBP", "serumcholestrol",
            "fastingbloodsugar", "restingrelectro", "maxheartrate", "exerciseangia",
            "oldpeak", "slope", "noofmajorvessels", "target"
        ],
        "feature_names": [
            "age", "gender", "chestpain", "restingBP", "serumcholestrol",
            "fastingbloodsugar", "restingrelectro", "maxheartrate", "exerciseangia",
            "oldpeak", "slope", "noofmajorvessels"
        ],
        "numeric_features": [
            "age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"
        ],
        "categorical_features": [
            "gender", "chestpain", "fastingbloodsugar", "restingrelectro",
            "exerciseangia", "slope", "noofmajorvessels"
        ],
        "feature_metadata": {
            "age": {
                "description": "Patient age in years",
                "type": "numeric",
                "unit": "years",
                "valid_range": (18, 100),
                "preprocessing": "Median imputation + StandardScaler"
            },
            "gender": {
                "description": "Biological sex (0 = female, 1 = male)",
                "type": "categorical_binary",
                "valid_values": [0, 1],
                "preprocessing": "Passthrough / OneHot"
            },
            "chestpain": {
                "description": "Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)",
                "type": "categorical_nominal",
                "valid_values": [0, 1, 2, 3],
                "preprocessing": "OneHotEncoder(handle_unknown='ignore')"
            },
            "restingBP": {
                "description": "Resting systolic blood pressure upon hospital intake",
                "type": "numeric",
                "unit": "mm Hg",
                "valid_range": (70, 260),
                "preprocessing": "Median imputation + StandardScaler"
            },
            "serumcholestrol": {
                "description": "Serum cholesterol level (0 values are coerced to NaN and imputed)",
                "type": "numeric",
                "unit": "mg/dL",
                "valid_range": (100, 700),
                "preprocessing": "Sentinel 0 coercion + Median imputation + StandardScaler"
            },
            "fastingbloodsugar": {
                "description": "Fasting blood sugar > 120 mg/dL (1 = true; 0 = false)",
                "type": "categorical_binary",
                "valid_values": [0, 1],
                "preprocessing": "Passthrough / OneHot"
            },
            "restingrelectro": {
                "description": "Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)",
                "type": "categorical_nominal",
                "valid_values": [0, 1, 2],
                "preprocessing": "OneHotEncoder(handle_unknown='ignore')"
            },
            "maxheartrate": {
                "description": "Maximum heart rate achieved during exercise test",
                "type": "numeric",
                "unit": "bpm",
                "valid_range": (60, 230),
                "preprocessing": "Median imputation + StandardScaler"
            },
            "exerciseangia": {
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
                "description": "Slope of peak exercise ST segment (0, 1: upsloping, 2: flat, 3: downsloping)",
                "type": "categorical_nominal",
                "valid_values": [0, 1, 2, 3],
                "preprocessing": "OneHotEncoder(handle_unknown='ignore')"
            },
            "noofmajorvessels": {
                "description": "Number of major vessels (0-3) colored by fluoroscopy",
                "type": "categorical_nominal",
                "valid_values": [0, 1, 2, 3],
                "preprocessing": "Most frequent imputation + OneHot"
            },
            "target": {
                "description": "Cardiovascular disease presence (0 = absence / healthy, 1 = presence of CVD)",
                "type": "binary_target",
                "valid_values": [0, 1],
                "preprocessing": "Binary classification target"
            }
        }
    },
    "uci": {
        "name": "UCI Cleveland Heart Disease Benchmark",
        "short_name": "UCI Cleveland Benchmark",
        "origin": "Cleveland Clinic Foundation, USA",
        "authors": "Janosi, Steinbrunn, Pfisterer, Detrano (1988)",
        "doi": "10.24432/C52P4X",
        "primary_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "mirror_url": "https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data",
        "citation": "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository.",
        "sha256": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
        "expected_rows": 303,
        "expected_raw_columns": 14,
        "id_column": None,
        "target_raw_column": "num",
        "target_column": "target",
        "raw_file": RAW_DATA_DIR / "uci" / "processed.cleveland.data",
        "metadata_file": RAW_DATA_DIR / "uci" / "raw_metadata.json",
        "processed_file": PROCESSED_DATA_DIR / "uci" / "heart_disease_cleaned.csv",
        "train_file": PROCESSED_DATA_DIR / "uci" / "train.csv",
        "test_file": PROCESSED_DATA_DIR / "uci" / "test.csv",
        "models_dir": MODELS_DIR / "uci",
        "figures_dir": FIGURES_DIR / "uci",
        "metrics_dir": METRICS_DIR / "uci",
        "audit_dir": AUDIT_DIR / "uci",
        "raw_column_names": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal", "num"
        ],
        "feature_names": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal"
        ],
        "numeric_features": [
            "age", "trestbps", "chol", "thalach", "oldpeak"
        ],
        "categorical_features": [
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
        ],
        "feature_metadata": {
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
                "type": "categorical_nominal",
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
    }
}


def get_dataset_config(dataset_name: str = None) -> Dict[str, Any]:
    """Retrieves dataset configuration dictionary for 'india' or 'uci'."""
    key = (dataset_name or DEFAULT_DATASET).lower()
    if key not in DATASETS_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASETS_CONFIG.keys())}")
    return DATASETS_CONFIG[key]


# ==============================================================================
# BACKWARD COMPATIBILITY EXPORTS (Defaults to India primary, supports UCI fallback)
# ==============================================================================
# For existing legacy imports in tests or scripts expecting direct constants:
_active_cfg = get_dataset_config(DEFAULT_DATASET)

RAW_DATA_FILE = RAW_DATA_DIR / "processed.cleveland.data"  # Legacy UCI root path
RAW_METADATA_FILE = RAW_DATA_DIR / "raw_metadata.json"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_cleaned.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

BEST_PIPELINE_FILE = MODELS_DIR / "best_pipeline.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

AUDIT_REPORT_FILE = AUDIT_DIR / "dataset_audit_report.md"
FEATURE_DICT_FILE = AUDIT_DIR / "feature_dictionary.csv"
DATA_QUALITY_SUMMARY_FILE = AUDIT_DIR / "data_quality_summary.json"
MODEL_COMPARISON_JSON = METRICS_DIR / "model_comparison.json"
MODEL_COMPARISON_CSV = METRICS_DIR / "model_comparison.csv"
TEST_METRICS_JSON = METRICS_DIR / "test_metrics.json"
ERROR_ANALYSIS_CSV = METRICS_DIR / "error_analysis.csv"

# Active schema constants
RAW_COLUMN_NAMES: List[str] = _active_cfg["raw_column_names"]
FEATURE_NAMES: List[str] = _active_cfg["feature_names"]
TARGET_RAW_NAME = _active_cfg["target_raw_column"]
TARGET_NAME = _active_cfg["target_column"]
NUMERIC_FEATURES: List[str] = _active_cfg["numeric_features"]
CATEGORICAL_FEATURES: List[str] = _active_cfg["categorical_features"]
FEATURE_METADATA: Dict[str, Dict[str, Any]] = _active_cfg["feature_metadata"]

# Experimental Controls
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PRIMARY_SELECTION_METRIC = "roc_auc"

# Hyperparameter Grids
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
    "for studying explainable machine learning methodologies on cardiovascular cohorts. "
    "It is NOT clinically validated, NOT certified as a medical device (CE/FDA), and "
    "MUST NOT be used for real-world medical diagnosis, risk assessment, or treatment decisions."
)
