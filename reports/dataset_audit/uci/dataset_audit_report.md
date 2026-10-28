# CardioSense Dataset Audit & Quality Analysis Report: UCI Cleveland Heart Disease Benchmark

**Date & Time**: 2026-10-25T21:31:11.369384+00:00  
**Source**: [UCI Cleveland Heart Disease Benchmark](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)  
**SHA256 Checksum**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`  
**Raw Dimensions**: 303 rows × 14 columns  
**Processed Dimensions**: 303 rows × 14 columns  

---

## 1. Executive Summary & Provenance
- **Cohort**: UCI Cleveland Heart Disease Benchmark (Cleveland Clinic Foundation, USA)
- **Authors**: Janosi, Steinbrunn, Pfisterer, Detrano (1988)
- **DOI / Citation**: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository.
- **Raw Columns**: 14 attributes.
- **Patient Identifiers**: Removed from feature matrix to guarantee patient privacy and zero leakage.

## 2. Target Variable Verification & Distribution
- **Class 0 (Absence / Healthy)**: 164 patients (54.13%)
- **Class 1 (Presence / CVD)**: 139 patients (45.87%)
- **Positive Prevalence Rate**: 45.87%

## 3. Data Hygiene & Missing Values
- **Duplicate Records**: 0 exact duplicates detected.
- **Identified Missing Value Columns**:
  - `ca`: 4 missing values (1.32%)
  - `thal`: 2 missing values (0.66%)

### Imputation Strategy:
- Numeric features: Median imputation (fitted strictly on training fold).
- Categorical features: Most frequent / mode imputation (fitted strictly on training fold).
- Split-First Protocol guarantees zero data leakage between development folds and holdout test data.

## 4. Feature Schema & Summary Statistics
| Feature | Type | Meaning | Observed Range / Values | Missing Values | Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | numeric | Age in years | [29.0, 77.0], mean=54.4 | 0 | Median imputation + StandardScaler |
| `sex` | categorical_binary | Sex (1 = male; 0 = female) | Categorical: [0, 1] | 0 | Passthrough / OneHot |
| `cp` | categorical_nominal | Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic) | Categorical: [1, 2, 3, 4] | 0 | OneHotEncoder(handle_unknown='ignore') |
| `trestbps` | numeric | Resting blood pressure on admission to hospital | [94.0, 200.0], mean=131.7 | 0 | Median imputation + StandardScaler |
| `chol` | numeric | Serum cholesterol | [126.0, 564.0], mean=246.7 | 0 | Median imputation + StandardScaler |
| `fbs` | categorical_binary | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) | Categorical: [0, 1] | 0 | Passthrough / OneHot |
| `restecg` | categorical_nominal | Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy) | Categorical: [0, 1, 2] | 0 | OneHotEncoder(handle_unknown='ignore') |
| `thalach` | numeric | Maximum heart rate achieved during exercise test | [71.0, 202.0], mean=149.6 | 0 | Median imputation + StandardScaler |
| `exang` | categorical_binary | Exercise induced angina (1 = yes; 0 = no) | Categorical: [0, 1] | 0 | Passthrough / OneHot |
| `oldpeak` | numeric | ST depression induced by exercise relative to rest | [0.0, 6.2], mean=1.0 | 0 | Median imputation + StandardScaler |
| `slope` | categorical_nominal | Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping) | Categorical: [1, 2, 3] | 0 | OneHotEncoder(handle_unknown='ignore') |
| `ca` | categorical_nominal | Number of major vessels (0-3) colored by fluoroscopy | Categorical: [0, 1, 2, 3] | 4 | Most frequent imputation + OneHot |
| `thal` | categorical_nominal | Thalassemia status (3 = normal; 6 = fixed defect; 7 = reversible defect) | Categorical: [3, 6, 7] | 2 | Most frequent imputation + OneHot |

## 5. Non-Diagnostic Disclaimer
CardioSense is an educational and academic research tool. Predictions must never be used for actual clinical medical diagnosis or treatment decisions.
