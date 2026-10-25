# CardioSense: Research Results & Model Evaluation Report

**Study Title**: Explainable Machine Learning for Heart Disease Risk Classification  
**Date**: October 2026  
**Dataset**: UCI Heart Disease (Cleveland Database)  
**Primary Checksum (SHA256)**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`  
**Environment**: Python 3.13.7, Scikit-Learn 1.8.0, XGBoost 3.2.0, SHAP 0.52.0  

---

## 1. Research Objectives & Problem Formulation
CardioSense investigates whether clinical patient attributes from non-invasive and minimally invasive tests can effectively classify heart disease presence (angiographic disease definition: >50% diameter narrowing in at least one major vessel).

### Specific Research Questions Addressed:
- **RQ1**: How do linear baselines (Logistic Regression) compare against ensemble architectures (Random Forest, XGBoost) under identical, leakage-safe stratified evaluation?
- **RQ2**: What trade-offs emerge between precision, recall, specificity, and false-negative rates in small-sample clinical data?
- **RQ3**: Which clinical features drive model risk predictions according to game-theoretic SHAP attributions?
- **RQ4**: Can this research pipeline be reproduced deterministically across data splits, feature transformations, and inference boundaries?
- **RQ5**: How stable are observed metrics given the limited dataset scale ($N=303$)?

---

## 2. Dataset Hygiene & Provenance
- **Raw Observations**: 303 instances, 14 columns.
- **Target Mapping**: Converted categorical diagnosis (0 = absence; 1, 2, 3, 4 = presence) into binary `target` ($0$ vs. $1$).
- **Class Proportions**:
  - Absence (0): 164 (54.1%)
  - Presence (1): 139 (45.9%)
- **Data Quality & Missing Values**:
  - `ca` (major vessels): 4 missing values ($1.32\%$) encoded as `?`.
  - `thal` (thalassemia): 2 missing values ($0.66\%$) encoded as `?`.
  - **Leakage Prevention**: All imputation (median for numeric, mode for categorical), scaling (StandardScaler), and OneHot encoding are fitted exclusively within training folds during cross-validation, with zero knowledge of held-out test data.

---

## 3. Experimental Controls & Protocol
- **Data Partition**: 80% development train set ($N=242$), 20% untouched holdout test set ($N=61$), stratified by target.
- **Validation**: Stratified 5-Fold Cross-Validation on the development partition.
- **Hyperparameter Optimization**: Systematic Grid Search optimizing for **ROC-AUC** as the primary selection criterion.
- **Random Seed**: Fixed to `42` across all splits, random forests, and gradient boosters.

---

## 4. Quantitative Results & Model Comparison

### Table 1: Cross-Validation & Held-Out Test Evaluation Summary
| Model Family | CV ROC-AUC (Mean ± Std) | Test ROC-AUC | Test Recall | Test Precision | Test F1-Score | Test Accuracy | False Negatives (FN) | False Positives (FP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9025 ± 0.0144 | 0.9665 | 0.9286 | 0.8387 | 0.8814 | 0.8852 | 2 | 5 |
| **Random Forest** | 0.9038 ± 0.0238 | 0.9502 | 0.9286 | 0.8387 | 0.8814 | 0.8852 | 2 | 5 |
| **Xgboost** | 0.8897 ± 0.0246 | 0.9513 | 0.9286 | 0.8667 | 0.8966 | 0.9016 | 2 | 4 |

### Key Findings (RQ1 & RQ2):
1. **Model Parity**: All three models demonstrated strong discrimination on the held-out test set ($ROC-AUC > 0.95$). Logistic Regression achieved the highest test ROC-AUC ($0.9665$), while Random Forest yielded the highest cross-validation stability ($0.9038 \pm 0.0238$).
2. **False Negatives in Clinical Context**: On the test set ($N=61$, with 28 true positive cases), all three models produced only **2 False Negatives** (Recall = $92.86\%$), correctly detecting 26 of 28 positive heart disease patients.
3. **Ensemble vs. Linear Baseline**: On small datasets ($N=303$), heavily regularized linear models with appropriate standard scaling perform on par with or slightly outperform complex ensembles, resisting the high variance typical of deep decision trees.

---

## 5. SHAP Explainability Findings (RQ3)
SHAP (TreeExplainer) was applied to the champion model (`random_forest`) across 28 transformed features.

### Table 2: Top 8 Most Influential Clinical Features (SHAP Global Attribution)
| Rank | Feature | Mean |SHAP| (Attribution Impact) | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | `thal_3.0` | 0.0734 | Key driver of risk prediction |
| 2 | `ca_0.0` | 0.0681 | Key driver of risk prediction |
| 3 | `thal_7.0` | 0.0545 | Key driver of risk prediction |
| 4 | `cp_4.0` | 0.0511 | Key driver of risk prediction |
| 5 | `oldpeak` | 0.0325 | Key driver of risk prediction |
| 6 | `thalach` | 0.0302 | Key driver of risk prediction |
| 7 | `exang_1.0` | 0.0277 | Key driver of risk prediction |
| 8 | `slope_2.0` | 0.0267 | Key driver of risk prediction |

### Biological & Clinical Interpretation:
- **Thalassemia (`thal_7.0` - Reversible Defect)**: Patients presenting with reversible perfusion defects during stress testing exhibit the strongest positive SHAP push toward a heart disease prediction.
- **Chest Pain Type (`cp_4.0` - Asymptomatic Angina)**: Asymptomatic presentation is paradoxically the strongest indicator of severe underlying coronary artery disease in this clinical cohort.
- **Fluoroscopy Vessels (`ca_0.0`)**: Absence of fluoroscopy-colored vessels strongly reduces predicted risk, functioning as a primary protective feature.
- **Maximum Heart Rate (`thalach`)**: Lower achieved heart rate during stress exercise correlates with significantly higher risk attributions.

---

## 6. Error Analysis
Out of 61 held-out test patients, 7 were misclassified by the champion model (5 False Positives and 2 False Negatives).

### Detailed Breakdown:
- **False Negatives (2 cases)**:
  - Patients with positive angiographic disease who exhibited atypical symptoms (e.g., normal ST slopes or lower fluoroscopy vessel counts) that masked their condition. In a real-world clinical triage setting, these represent critical risks that warrant conservative threshold calibration.
- **False Positives (5 cases)**:
  - Patients without significant angiographic narrowing who presented with severe risk factors (such as elderly age, high blood pressure, and abnormal ST depression). These "false alarms" lead to secondary testing rather than missed care.

---

## 7. Limitations & Stability (RQ5)
1. **Sample Size**: With $N=303$, metric point estimates have non-trivial confidence intervals. While cross-validation validates consistency, larger cohorts are required for statistical power.
2. **Historical Cohort**: The Cleveland dataset originates from 1988. Diagnostic criteria, imaging technologies, and demographic profiles have evolved substantially.
3. **Single-Site Selection**: Lack of external multi-center validation limits generalizability across diverse patient populations.

---

## 8. Non-Diagnostic Medical Disclaimer
> [!CAUTION]
> **{DISCLAIMER_TEXT}**
