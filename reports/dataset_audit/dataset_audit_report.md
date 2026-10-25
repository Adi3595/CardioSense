# CardioSense Dataset Audit & Quality Analysis Report

**Date & Time**: 2026-10-25T21:31:11.369384+00:00  
**Source**: [UCI Heart Disease (Cleveland database)](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)  
**SHA256 Checksum**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`  
**Raw Dimensions**: 303 rows × 14 columns  
**Processed Dimensions**: 303 rows × 14 columns  

---

## 1. Executive Summary & Provenance
The dataset analyzed is the canonical Cleveland clinic heart disease dataset from the UCI Machine Learning Repository.
The raw data consists of 303 patient records across 14 measured attributes (13 clinical predictors and 1 diagnosis target).
Missing values in the original source are encoded as string `'?'`, specifically identified in features `ca` (major vessels) and `thal` (thalassemia).
No patient identifiers (PHI) are present in the dataset.

## 2. Target Variable Verification & Encoding
The original diagnosis variable (`num`) contains integer values ranging from 0 to 4:
- `0`: Absence of heart disease diagnosis (< 50% vessel diameter narrowing).
- `1, 2, 3, 4`: Presence of heart disease diagnosis (> 50% vessel diameter narrowing).

In accordance with medical ML research standards and project guidelines, the target is converted to binary format:
- **Class 0 (Absence)**: 164 patients (54.13%)
- **Class 1 (Presence)**: 139 patients (45.87%)
- **Positive Class Prevalence**: 45.87%

This confirms a relatively balanced binary classification problem (approx. 54% absence vs. 46% presence), suitable for stratified cross-validation.

## 3. Data Hygiene & Missing Values
- **Duplicate Records**: 0 exact duplicates detected.
- **Identified Missing Value Placeholders**: Converted `'?'` to `NaN`.
- **Missing Value Inventory**:
  - `ca`: 4 missing values (1.32%)
  - `thal`: 2 missing values (0.66%)

### Imputation Strategy:
- Numeric features: Median imputation (fitted strictly on training fold).
- Categorical features (`ca`, `thal`): Most frequent / mode imputation (fitted strictly on training fold).
- No imputation is fitted on test data or the full un-split dataset to guarantee zero leakage.

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
| `ca` | categorical_ordinal_discrete | Number of major vessels (0-3) colored by fluoroscopy | Categorical: [0, 1, 2, 3] | 4 | Most frequent imputation + OneHot |
| `thal` | categorical_nominal | Thalassemia status (3 = normal; 6 = fixed defect; 7 = reversible defect) | Categorical: [3, 6, 7] | 2 | Most frequent imputation + OneHot |

## 5. Medical & Non-Diagnostic Disclaimer
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository.
**Notice**: CardioSense is an academic research demonstration. Predictions must never be used for actual clinical diagnostic decisions.
