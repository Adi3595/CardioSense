# CardioSense Dataset Audit & Quality Analysis Report: Indian Hospital Cardiovascular Disease Dataset

**Date & Time**: 2026-10-28T00:00:00Z  
**Source**: [Indian Hospital Cardiovascular Disease Dataset](https://data.mendeley.com/datasets/dzz48mvjht/1)  
**SHA256 Checksum**: `9376367dde41e79e432d6402ea5d9e44223fc318ff54ed09966d721923947d1e`  
**Raw Dimensions**: 1000 rows × 14 columns  
**Processed Dimensions**: 1000 rows × 13 columns  

---

## 1. Executive Summary & Provenance
- **Cohort**: Indian Hospital Cardiovascular Disease Dataset (Multi-specialty hospital, India)
- **Authors**: Bhanu Prakash Doppala and Debnath Bhattacharyya
- **DOI / Citation**: Doppala, B. P., & Bhattacharyya, D. (2021). Cardiovascular_Disease_Dataset. Mendeley Data, V1, doi: 10.17632/dzz48mvjht.1
- **Raw Columns**: 14 attributes.
- **Patient Identifiers**: Removed from feature matrix to guarantee patient privacy and zero leakage.

## 2. Target Variable Verification & Distribution
- **Class 0 (Absence / Healthy)**: 420 patients (42.0%)
- **Class 1 (Presence / CVD)**: 580 patients (58.0%)
- **Positive Prevalence Rate**: 58.0%

## 3. Data Hygiene & Missing Values
- **Duplicate Records**: 0 exact duplicates detected.
- **Identified Missing Value Columns**:
  - `serumcholestrol`: 53 missing values (5.3%)

### Imputation Strategy:
- Numeric features: Median imputation (fitted strictly on training fold).
- Categorical features: Most frequent / mode imputation (fitted strictly on training fold).
- Split-First Protocol guarantees zero data leakage between development folds and holdout test data.

## 4. Feature Schema & Summary Statistics
| Feature | Type | Meaning | Observed Range / Values | Missing Values | Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | numeric | Patient age in years | [20.0, 80.0], mean=49.2 | 0 | Median imputation + StandardScaler |
| `gender` | categorical_binary | Biological sex (0 = female, 1 = male) | Categorical: [0, 1] | 0 | Passthrough / OneHot |
| `chestpain` | categorical_nominal | Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic) | Categorical: [0, 1, 2, 3] | 0 | OneHotEncoder(handle_unknown='ignore') |
| `restingBP` | numeric | Resting systolic blood pressure upon hospital intake | [94.0, 200.0], mean=151.7 | 0 | Median imputation + StandardScaler |
| `serumcholestrol` | numeric | Serum cholesterol level (0 values are coerced to NaN and imputed) | [85.0, 602.0], mean=328.9 | 53 | Sentinel 0 coercion + Median imputation + StandardScaler |
| `fastingbloodsugar` | categorical_binary | Fasting blood sugar > 120 mg/dL (1 = true; 0 = false) | Categorical: [0, 1] | 0 | Passthrough / OneHot |
| `restingrelectro` | categorical_nominal | Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy) | Categorical: [0, 1, 2] | 0 | OneHotEncoder(handle_unknown='ignore') |
| `maxheartrate` | numeric | Maximum heart rate achieved during exercise test | [71.0, 202.0], mean=145.5 | 0 | Median imputation + StandardScaler |
| `exerciseangia` | categorical_binary | Exercise induced angina (1 = yes; 0 = no) | Categorical: [0, 1] | 0 | Passthrough / OneHot |
| `oldpeak` | numeric | ST depression induced by exercise relative to rest | [0.0, 6.2], mean=2.7 | 0 | Median imputation + StandardScaler |
| `slope` | categorical_nominal | Slope of peak exercise ST segment (0, 1: upsloping, 2: flat, 3: downsloping) | Categorical: [0, 1, 2, 3] | 0 | OneHotEncoder(handle_unknown='ignore') |
| `noofmajorvessels` | categorical_nominal | Number of major vessels (0-3) colored by fluoroscopy | Categorical: [0, 1, 2, 3] | 0 | Most frequent imputation + OneHot |

## 5. Non-Diagnostic Disclaimer
CardioSense is an educational and academic research tool. Predictions must never be used for actual clinical medical diagnosis or treatment decisions.
