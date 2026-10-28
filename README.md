# CardioSense: Explainable Machine Learning for Cardiovascular Disease Risk Classification

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B.svg)](https://streamlit.io/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000.svg)](https://nextjs.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-25%20passed-brightgreen.svg)]()

> **Notice & Non-Diagnostic Medical Disclaimer**: CardioSense is an academic research and educational prototype developed for studying explainable machine learning methodologies on cardiovascular cohorts. It is **not clinically certified**, **not certified as a medical device (CE/FDA)**, and **must not be used for real-world medical diagnosis, clinical risk assessment, or patient treatment planning**. Predictions represent statistical model associations within specific cohort data and do not substitute for formal clinical judgment or angiography.

📖 **Complete Data Science Lifecycle Guide**: For full lifecycle documentation including schema validation, data cleaning, zero-cholesterol sentinel handling, stratified 5-fold cross-validation, and game-theoretic SHAP explainability, see [**docs/DATA_SCIENCE_LIFECYCLE.md**](docs/DATA_SCIENCE_LIFECYCLE.md).

📄 **IEEE Research Paper**: The complete academic manuscript formatted in IEEE Conference style is available in LaTeX format at [**docs/paper/CardioSense_IEEE_Paper.tex**](docs/paper/CardioSense_IEEE_Paper.tex).

---

## 1. Project Overview & India Research Focus

CardioSense investigates machine learning methodologies for cardiovascular disease (CVD) detection, placing primary emphasis on **South Asian and Indian clinical patterns**. South Asians develop coronary artery disease (CAD) 5–10 years earlier than Western cohorts, frequently exhibiting severe multi-vessel diffuse disease despite lower rates of conventional hypercholesterolemia (as established by the ICMR-INDIAB study and ICADD research).

To reflect these distinct risk dynamics without dataset leakage or cross-cohort confounding, CardioSense implements a **dual-dataset architecture**:
1. **Primary Focus — Indian Hospital Cohort**: A real-world multi-specialty hospital cohort from India ($N=1,000$ subjects, 12 clinical predictors).
2. **Benchmark Baseline — UCI Cleveland Cohort**: The historical Western baseline ($N=303$ subjects, 13 clinical predictors, Janosi et al., 1988).

**Strict Architecture Rule**: The Indian and UCI datasets are kept completely separate with dedicated data schemas, preprocessing pipelines, trained model artifacts, and evaluation tables. Incompatible feature spaces are never merged.

---

## 2. Dataset Hygiene & Provenance

### Primary Dataset: Indian Hospital Cohort
- **Source**: Mendeley Data, Contributed by Kasturi Doppala & Debnath Bhattacharyya (2020)
- **DOI**: [10.17632/dzz48mvjht.1](https://doi.org/10.17632/dzz48mvjht.1) / [Mendeley Dataset URL](https://data.mendeley.com/datasets/dzz48mvjht/1)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Cohort Dimensions**: 1,000 patients × 14 raw variables (1 unique patient identifier, 12 clinical predictors, 1 binary target).
- **Target Distribution**: $58.0\%$ Positive ($N=580$), $42.0\%$ Negative ($N=420$).
- **SHA-256 Checksum**: `9376367dde41e79e432d6402ea5d9e44223fc318ff54ed09966d721923947d1e`
- **Data Hygiene Discoveries**:
  - `patientid`: Dropped during ingestion to prevent identifier leakage.
  - `serumcholestrol == 0`: 53 patients ($5.3\%$) presented with zero cholesterol, an undocumented clinical placeholder where lipid panels were not ordered upon acute admission. These sentinels are coerced to `np.nan` and median-imputed strictly within training folds.
  - Mean cohort age is $49.2 \pm 7.9$ years (significantly younger than UCI's $54.4 \pm 9.0$ years), reflecting premature CAD patterns in India.
  - Mean resting blood pressure is elevated at $151.7 \pm 23.5$ mm Hg.

#### Indian Cohort Feature Schema
| Feature Name | Type | Description | Valid Domain / Range | Preprocessing |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Numeric | Age in completed years | 20 – 90 years | Median Impute + Scaler |
| `gender` | Categorical | Sex (1 = male; 0 = female) | {0, 1} | Passthrough / OneHot |
| `chestpain` | Categorical | Chest pain severity | 0 = Typical, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic | OneHotEncoder |
| `restingBP` | Numeric | Resting systolic blood pressure (mm Hg) | 90 – 220 mm Hg | Median Impute + Scaler |
| `serumcholestrol`| Numeric | Serum total cholesterol (mg/dL) | 100 – 600 mg/dL (zeros $\rightarrow$ NaN) | Median Impute + Scaler |
| `fastingbloodsugar` | Categorical | Fasting blood glucose > 120 mg/dL | {0, 1} | Passthrough / OneHot |
| `restingrelectro` | Categorical | Resting ECG findings | {0, 1, 2} | OneHotEncoder |
| `maxheartrate` | Numeric | Peak exercise heart rate (bpm) | 60 – 240 bpm | Median Impute + Scaler |
| `exerciseangia` | Categorical | Exercise-induced angina | {0, 1} | Passthrough / OneHot |
| `oldpeak` | Numeric | ST depression during exercise (mm) | 0.0 – 6.5 mm | Median Impute + Scaler |
| `slope` | Categorical | Peak ST segment slope | 1 = Upsloping, 2 = Flat, 3 = Downsloping | OneHotEncoder |
| `noofmajorvessels`| Discrete | Fluoroscopy vessels (0–3) | {0, 1, 2, 3} | OneHotEncoder |
| `target` | Binary Target | Cardiovascular Disease Status | 0 = Absence, 1 = Presence | Supervised Ground Truth |

---

### Benchmark Baseline: UCI Cleveland Cohort
- **Source**: [UCI Machine Learning Repository Heart Disease (Cleveland Database)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Direct Retrieval**: `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- **SHA-256 Checksum**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
- **Cohort Dimensions**: 303 patients × 14 raw variables.
- **Target Encoding**: Raw `num = 0` $\rightarrow$ `0` (Absence, 54.1%), `num \in {1,2,3,4}` $\rightarrow$ `1` (Presence, 45.9%).
- **Features**: Includes thallium scintigraphy (`thal`), not available in the Indian dataset.

---

## 3. Real Measured Experimental Results

All experiments utilized **Stratified 80/20 train/test splits** and **Stratified 5-Fold Cross-Validation** ($R=42$) across Logistic Regression, Random Forest, and XGBoost.

### A. Primary Indian Hospital Cohort ($N_{\text{train}}=800, N_{\text{test}}=200$)
| Architecture | 5-Fold CV ROC-AUC | Test Accuracy | Test Recall (Sensitivity) | Test Precision | Test F1-Score | Test ROC-AUC | False Negatives | False Positives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $0.9909 \pm 0.0084$ | $97.00\%$ | $97.41\%$ | $97.41\%$ | $0.9741$ | $0.9978$ | 3 | 3 |
| **Random Forest** | $0.9957 \pm 0.0059$ | $99.00\%$ | $99.14\%$ | $99.14\%$ | $0.9914$ | $0.9995$ | 1 | 1 |
| **XGBoost (Champion)** | **$0.9960 \pm 0.0040$** | **$99.50\%$** | **$100.0\%$** | **$99.15\%$** | **$0.9957$** | **$0.9998$** | **0** | **1** |

> **Key Clinical Finding**: On the Indian hospital cohort, the champion **XGBoost model achieved 100% sensitivity** (0 false negatives out of 116 confirmed CVD cases in the held-out test split), critical for medical screening where false negatives carry catastrophic clinical risks.

### B. UCI Cleveland Benchmark ($N_{\text{train}}=242, N_{\text{test}}=61$)
| Architecture | 5-Fold CV ROC-AUC | Test Accuracy | Test Recall (Sensitivity) | Test Precision | Test F1-Score | Test ROC-AUC | False Negatives | False Positives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $0.9025 \pm 0.0144$ | $88.52\%$ | $92.86\%$ | $83.87\%$ | $0.8814$ | **$0.9665$** | 2 | 5 |
| **Random Forest (Champion)** | **$0.9038 \pm 0.0238$** | $88.52\%$ | $92.86\%$ | $83.87\%$ | $0.8814$ | $0.9502$ | 2 | 5 |
| **XGBoost** | $0.8901 \pm 0.0221$ | **$90.16\%$** | $92.86\%$ | **$86.67\%$** | **$0.8966$** | $0.9481$ | 2 | 4 |

---

## 4. SHAP Feature Attribution Analysis

Game-theoretic Shapley values calculated via `shap.TreeExplainer` on the primary Indian hospital cohort revealed the following top clinical predictors of cardiovascular disease risk:

1. **`chestpain` (Chest Pain Type)**: Mean $|\text{SHAP}| = 1.341$. Asymptomatic chest pain or severe presentation is the single strongest predictor of established multi-vessel disease.
2. **`oldpeak` (ST Depression)**: Mean $|\text{SHAP}| = 0.985$. Significant exercise-induced ST depression reflects severe myocardial ischemia.
3. **`noofmajorvessels` (Major Vessels Colored)**: Mean $|\text{SHAP}| = 0.762$. High vessel involvement strongly increases positive log-odds.
4. **`maxheartrate` (Maximum Heart Rate Achieved)**: Mean $|\text{SHAP}| = 0.612$. Chronotropic incompetence during exertion correlates strongly with adverse coronary events.
5. **`exerciseangia` (Exercise-Induced Angina)**: Mean $|\text{SHAP}| = 0.448$. Exertional ischemia provides strong confirmatory evidence of CAD.

---

## 5. Repository Architecture

```
CardioSense/
├── data/
│   ├── raw/
│   │   ├── india/
│   │   │   ├── Cardiovascular_Disease_Dataset.csv   # Indian hospital cohort (N=1000)
│   │   │   └── raw_metadata.json                   # SHA-256 and Mendeley DOI
│   │   └── uci/
│   │       ├── processed.cleveland.data            # Canonical UCI Cleveland data
│   │       └── raw_metadata.json                   # SHA-256 and UCI source info
│   └── processed/
│       ├── india/ {cleaned.csv, train.csv, test.csv}
│       └── uci/   {cleaned.csv, train.csv, test.csv}
├── src/
│   ├── config.py                 # Multi-cohort configuration, schemas & grids
│   ├── data_loader.py            # Dual-cohort ingestion, zero-chol sentinel handling
│   ├── data_validation.py        # Schema validation & automated audit reports
│   ├── eda.py                    # 6-figure publication visualization suite
│   ├── preprocessing.py          # Leakage-safe ColumnTransformer & Pipeline
│   ├── train.py                  # Stratified 5-fold CV hyperparameter tuning
│   ├── evaluate.py               # Precision, Recall, ROC-AUC & confusion matrix
│   ├── explain.py                # SHAP beeswarm, bar, and waterfall plots
│   └── experiment_tracking.py    # Experiment logging & artifact tracking
├── models/
│   ├── india/                    # India champion (xgboost) and baseline artifacts
│   ├── uci/                      # UCI champion (random_forest) and baseline artifacts
│   └── best_pipeline.joblib      # Primary deployable pipeline (India champion)
├── reports/
│   ├── dataset_audit/ {india/, uci/}
│   ├── figures/       {india/, uci/}
│   └── metrics/       {india/, uci/}
├── api/
│   ├── main.py                   # FastAPI application supporting /predict/india & /predict/uci
│   └── schemas.py                # Pydantic validation schemas with range checking
├── app/
│   └── streamlit_app.py          # Interactive Streamlit app with soft pastel UI
├── frontend/                     # Fullstack Next.js interactive web portal
├── docs/
│   ├── DATA_SCIENCE_LIFECYCLE.md # Comprehensive data science lifecycle guide
│   └── paper/
│       └── CardioSense_IEEE_Paper.tex # Full IEEE LaTeX conference paper
└── tests/
    ├── test_api.py               # Endpoints, schema validation, HTTP 422 tests
    ├── test_data_validation.py   # Schema integrity, target encoding, sentinel tests
    ├── test_evaluation.py        # Diagnostic metrics and confusion matrices
    ├── test_explainability.py    # SHAP artifacts and feature attribution
    ├── test_preprocessing.py     # ColumnTransformer, imputation, leakage tests
    └── test_training.py          # 5-fold CV fitting and persistence tests
```

---

## 6. Setup & Reproduction Guide

### 1. Environment Setup
```bash
# Clone and enter directory
git clone https://github.com/Adi3595/CardioSense.git
cd CardioSense

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install locked dependencies
pip install -r requirements.txt
```

### 2. Ingest & Audit Both Datasets
```bash
# Fetch and validate datasets ('india', 'uci', or 'all')
python -m src.data_loader --dataset all
python -m src.data_validation --dataset all
```

### 3. Generate Exploratory Analysis Figures
```bash
python -m src.eda --dataset all
```

### 4. Train Models & Generate SHAP Attributions
```bash
# Train models, benchmark architectures, and calculate SHAP values
python -m src.train --dataset all
```

### 5. Run Automated Test Suite
```bash
pytest -v
# 25 passed in 4.75s
```

---

## 7. Inference API & Frontend Launch

### A. FastAPI Backend Service
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

#### Endpoints:
- `GET /health`: Service health and loaded model status across all cohorts.
- `POST /predict/india`: Dedicated inference for Indian hospital cohort (12 predictors).
- `POST /predict/uci`: Dedicated inference for UCI Cleveland benchmark (13 predictors).
- `POST /predict`: General inference endpoint with automatic cohort schema detection.

#### Sample Live Inference (Indian Cohort):
```bash
curl -X POST "http://127.0.0.1:8000/predict/india" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```
Response:
```json
{
  "predicted_class": 1,
  "prediction_label": "Presence of Cardiovascular Disease",
  "probability_presence": 0.9673,
  "probability_absence": 0.0327,
  "risk_level": "High",
  "model_used": "xgboost",
  "dataset_source": "Indian Hospital Cardiovascular Disease Dataset",
  "disclaimer": "DISCLAIMER: CardioSense is an academic research and educational prototype..."
}
```

### B. Streamlit Application
Launch the clean, minimalistic Streamlit application utilizing the curated soft pastel palette (ivory `#FAF8F5`, sage `#E8ECE9`, mint `#D8E2DC`, lavender `#E8E5EE`, peach `#FDE2D2`, and forest green `#2D5A27` accents):
```bash
streamlit run app/streamlit_app.py
```
Access at `http://localhost:8501`.

### C. Next.js Web Portal
```bash
cd frontend
npm run dev
```
Access at `http://localhost:3000`.

---

## 8. Research Limitations

1. **Single-Center Hospital Inpatient Sample**: The Indian cohort reflects patients evaluated at a multi-specialty tertiary care hospital with clinical indications for cardiac assessment, resulting in a higher baseline prevalence ($58.0\%$) than community screening environments.
2. **Geographic Generalizability**: While reflecting South Asian pathophysiology, the dataset originates from a single hospital system and does not capture the full genetic, linguistic, and socio-economic diversity across all Indian states (as documented by ICMR-INDIAB).
3. **Surrogate Clinical Markers**: Coronary angiography is the gold standard for definitive diagnosis. Non-invasive machine learning outputs provide educational risk estimates and cannot rule out complex multi-vessel lesions or atypical presentations.
4. **Sentinels & Missingness**: 53 patients presented with missing lipid panels (`serumcholestrol == 0`), reflecting emergency triage protocols where acute treatment precedes laboratory panels.
