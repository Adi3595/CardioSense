# CardioSense: Explainable Machine Learning for Heart Disease Risk Classification

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000.svg)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Tests Passing](https://img.shields.io/badge/pytest-18%20passed-brightgreen.svg)]()

> **Notice & Medical Disclaimer**: CardioSense is an educational and academic research project investigating machine learning methodology and feature explainability on clinical benchmarks. It is **not clinically certified**, **not medical software**, and **must not be used for real-world medical diagnosis or treatment planning**.

---

## 1. Project Overview & Research Questions
CardioSense evaluates how modern machine learning architectures predict binary heart disease status (absence vs. presence of coronary artery narrowing >50%) using non-invasive and minimally invasive clinical features from the canonical **UCI Cleveland Heart Disease** cohort.

### Core Research Questions
- **RQ1 (Model Parity)**: How do linear baselines (Logistic Regression with $L_2$ regularization) compare against ensemble architectures (Random Forest, XGBoost) under identical, leakage-safe stratified evaluation?
- **RQ2 (Diagnostic Trade-offs)**: What trade-offs exist between precision, recall, F1-score, ROC-AUC, and clinical false-negative rates?
- **RQ3 (SHAP Attribution)**: Which clinical features contribute most strongly to prediction outputs according to game-theoretic Shapley values?
- **RQ4 (Pipeline Reproducibility)**: Can the complete data acquisition, cleaning, stratified cross-validation, and serving pipeline be reproduced deterministically?
- **RQ5 (Stability)**: How stable are model parameters and discriminative rankings given the small sample scale ($N=303$)?

---

## 2. Dataset Hygiene & Provenance
- **Primary Source**: [UCI Machine Learning Repository Heart Disease (Cleveland Database)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Direct Retrieval URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- **Dataset SHA256 Checksum**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
- **Cohort Dimensions**: 303 patient observations × 14 raw variables.
- **Target Encoding**:
  - Raw `num = 0` $\rightarrow$ `0` (Absence of Heart Disease, $N=164$, $54.1\%$)
  - Raw `num \in {1, 2, 3, 4}` $\rightarrow$ `1` (Presence of Heart Disease, $N=139$, $45.9\%$)
- **Missing Value Handling**:
  - Missing placeholders (`?`) identified in `ca` ($N=4$, $1.32\%$) and `thal` ($N=2$, $0.66\%$).
  - Imputation: Median for numeric features; Most-frequent mode for categorical features. Fitted strictly within training folds to avoid target leakage.

### Feature Dictionary
| Feature | Type | Description | Reference Range / Categories |
| :--- | :--- | :--- | :--- |
| `age` | Numeric | Patient age in years | 29 – 77 years |
| `sex` | Categorical | Sex | 1 = male; 0 = female |
| `cp` | Categorical | Chest pain type | 1: Typ. Angina, 2: Atyp. Angina, 3: Non-anginal, 4: Asymptomatic |
| `trestbps` | Numeric | Resting blood pressure | 94 – 200 mm Hg |
| `chol` | Numeric | Serum cholesterol | 126 – 564 mg/dl |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl | 1 = true; 0 = false |
| `restecg` | Categorical | Resting ECG findings | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy |
| `thalach` | Numeric | Max heart rate achieved during exercise | 71 – 202 bpm |
| `exang` | Categorical | Exercise-induced angina | 1 = yes; 0 = no |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest | 0.0 – 6.2 mm |
| `slope` | Categorical | Peak exercise ST segment slope | 1: Upsloping, 2: Flat, 3: Downsloping |
| `ca` | Discrete | Major vessels (0-3) colored by fluoroscopy | 0, 1, 2, 3 |
| `thal` | Categorical | Thallium scintigraphy status | 3: Normal, 6: Fixed defect, 7: Reversible defect |

---

## 3. Measured Experimental Results

All experiments utilized an **80/20 Stratified Partition** ($N_{\text{train}}=242, N_{\text{test}}=61$) and **Stratified 5-Fold Cross-Validation** on development data ($R=42$).

### Table: Cross-Validation & Held-Out Test Set Performance
| Model | CV ROC-AUC (Mean ± Std) | Test ROC-AUC | Test Recall | Test Precision | Test F1 | Test Accuracy | False Negatives (FN) | False Positives (FP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $0.9025 \pm 0.0144$ | **0.9665** | **0.9286** | 0.8387 | 0.8814 | 0.8852 | **2** | 5 |
| **Random Forest** | **0.9038 ± 0.0238** | 0.9502 | **0.9286** | 0.8387 | 0.8814 | 0.8852 | **2** | 5 |
| **XGBoost** | $0.8897 \pm 0.0246$ | 0.9513 | **0.9286** | **0.8667** | **0.8966** | **0.9016** | **2** | 4 |

### Visual Artifacts Generated:
- **Target Distribution**: `reports/figures/target_distribution.png`
- **Continuous Distributions**: `reports/figures/numeric_distributions.png`
- **Outlier Boxplots**: `reports/figures/numeric_boxplots.png`
- **Correlation Matrix**: `reports/figures/correlation_heatmap.png`
- **Categorical Proportions**: `reports/figures/categorical_distributions.png`
- **Clinical Risk Bivariates**: `reports/figures/feature_target_relationships.png`
- **Confusion Matrix Comparison**: `reports/figures/confusion_matrices_comparison.png`
- **ROC Curves**: `reports/figures/roc_curves_comparison.png`
- **Precision-Recall Curves**: `reports/figures/precision_recall_curves_comparison.png`
- **Multi-Metric Comparison Bar**: `reports/figures/models_metrics_comparison.png`
- **SHAP Beeswarm Summary**: `reports/figures/shap_summary_plot.png`
- **SHAP Feature Importance**: `reports/figures/shap_feature_importance_bar.png`
- **Local Correct Waterfall**: `reports/figures/shap_local_correct.png`
- **Local Error Waterfall**: `reports/figures/shap_local_incorrect.png`

---

## 4. Repository Structure
```
CardioSense/
├── data/
│   ├── raw/
│   │   ├── processed.cleveland.data    # Original downloaded UCI data
│   │   └── raw_metadata.json           # SHA256, URL, retrieval timestamp
│   └── processed/
│       ├── heart_disease_cleaned.csv   # Cleaned 303-row dataset
│       ├── train.csv                   # Stratified training split (N=242)
│       └── test.csv                    # Stratified holdout test split (N=61)
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory analysis notebook
│   ├── 02_model_training_and_comparison.ipynb # Training & benchmark notebook
│   └── 03_model_explainability.ipynb   # SHAP explainability notebook
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, grids, random seeds, schema
│   ├── data_loader.py                  # Dataset download, hash check & split
│   ├── data_validation.py              # Schema validation & audit generator
│   ├── eda.py                          # Publication-grade visualization logic
│   ├── preprocessing.py                # Leakage-safe ColumnTransformer & Pipeline
│   ├── train.py                        # 5-fold CV hyperparameter tuning
│   ├── evaluate.py                     # Diagnostic metrics & error analysis
│   ├── explain.py                      # SHAP TreeExplainer & waterfall plots
│   ├── experiment_tracking.py          # Metadata logger
│   └── reporting.py                    # Markdown report synthesizer
├── models/
│   ├── best_pipeline.joblib            # Champion deployable inference pipeline
│   ├── logistic_regression.joblib      # Fitted baseline model
│   ├── random_forest.joblib            # Fitted Random Forest model
│   ├── xgboost.joblib                  # Fitted XGBoost model
│   └── model_metadata.json             # Full experiment run metadata
├── reports/
│   ├── dataset_audit/
│   │   ├── dataset_audit_report.md     # Audit documentation
│   │   ├── feature_dictionary.csv      # Feature dictionary
│   │   └── data_quality_summary.json   # Quality metrics
│   ├── figures/                        # All 14 generated research plots
│   ├── metrics/
│   │   ├── model_comparison.csv        # Cross-model metric comparison table
│   │   ├── error_analysis.csv          # Held-out misclassification diagnostics
│   │   └── shap_feature_importance.csv # Mean absolute SHAP values
│   └── research_results_report.md      # Formal research paper summary
├── api/
│   ├── __init__.py
│   ├── main.py                         # FastAPI application
│   └── schemas.py                      # Pydantic request & response schemas
├── frontend/                           # Next.js 16 interactive research web portal
│   ├── app/                            # App router, pages, and soft-paint design system
│   ├── public/figures/                 # Publication research plots & SHAP visuals
│   ├── package.json                    # Frontend dependencies
│   └── next.config.mjs                 # Next.js configuration
├── tests/
│   ├── test_data_validation.py         # Schema and target conversion tests
│   ├── test_preprocessing.py           # Imputation & leakage tests
│   ├── test_training.py                # Model fitting and loading tests
│   ├── test_evaluation.py              # Metric calculation tests
│   ├── test_explainability.py          # SHAP artifact tests
│   └── test_api.py                     # Endpoint & input validation tests
├── requirements.txt                    # Locked dependencies
├── pytest.ini                          # Test configuration
├── Dockerfile                          # Production container image
├── docker-compose.yml                  # Multi-service composition
└── README.md
```

---

## 5. Quickstart & Execution Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Acquisition & Quality Audit
```bash
python -m src.data_loader
python -m src.data_validation
```

### 3. Exploratory Data Analysis (EDA)
```bash
python -m src.eda
```

### 4. Train Models, Benchmark & Run SHAP
```bash
python -m src.train
```

### 5. Generate Research Report
```bash
python -m src.reporting
```

### 6. Run Automated Test Suite
```bash
python -m pytest -v
```

---

## 6. Inference API (FastAPI)

Start the local inference service:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### Health Check Request
```bash
curl -X GET http://127.0.0.1:8000/health
```

### Prediction Inference Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 63.0,
       "sex": 1,
       "cp": 1,
       "trestbps": 145.0,
       "chol": 233.0,
       "fbs": 1,
       "restecg": 2,
       "thalach": 150.0,
       "exang": 0,
       "oldpeak": 2.3,
       "slope": 3,
       "ca": 0.0,
       "thal": 6.0
     }'
```

---

## 7. Next.js Research Web Portal

Launch the fullstack Next.js web portal:
```bash
cd frontend
npm run dev
```
Access the application at `http://localhost:3000`.

**Features**:
- Complete research showcase (Provenance, Data Hygiene, 6 EDA publication plots with zoom lightbox)
- Model Benchmarks & Curve switchers (Confusion Matrices, ROC, Precision-Recall, Bar comparison)
- Error Analysis on the 7 test misclassifications (False Negatives vs False Positives)
- SHAP Beeswarm & Waterfall local patient explanations
- Live Clinical Risk Simulator communicating in real-time with the FastAPI `/predict` backend

---

## 8. Deployment & Containers
```bash
# Start backend API (FastAPI)
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Start frontend application (Next.js)
cd frontend && npm run dev
```

---

## 9. Key Limitations & Disclaimer
1. **Small Sample Dataset**: The Cleveland cohort contains 303 cases recorded in 1988. Modern cardiology uses additional markers (troponin, coronary calcium scoring, CTA).
2. **False Negatives**: While all three models achieved $92.86\%$ sensitivity, the 2 false negative cases underline that machine learning cannot replace clinical angiography.
3. **SHAP Interpretation**: Attributions denote predictive correlations within this specific feature space, not biochemical causation.
