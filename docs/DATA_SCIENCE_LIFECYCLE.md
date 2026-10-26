# CardioSense: End-to-End Data Science Lifecycle Specification
## Explainable Machine Learning for Heart Disease Risk Classification

**Project Name**: CardioSense  
**Author**: Aditya (Adi3595)  
**Dataset**: UCI Heart Disease Repository (Cleveland Clinic Foundation)  
**Verification SHA256**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`  
**Stack**: Python 3.13, Scikit-Learn 1.8, XGBoost 3.2, SHAP 0.52, FastAPI, Next.js 16, Pytest  

---

## 1. Executive Summary & Clinical Problem Formulation

Cardiovascular disease (CVD) remains the leading cause of global mortality. While coronary catheter angiography is the definitive clinical reference standard for diagnosing coronary artery disease (CAD), it is invasive, resource-intensive, and carries procedural risks.

**CardioSense** evaluates whether supervised machine learning models trained on 13 non-invasive and minimally invasive clinical features (demographics, resting vitals, exercise electrocardiography, and fluoroscopy) can reliably discriminate between the absence and presence of significant coronary artery disease (>50% vessel stenosis) while providing game-theoretic **SHAP (SHapley Additive exPlanations)** feature attributions for every prediction.

### Core Research Questions (RQ1–RQ5)
* **RQ1 (Model Parity)**: Can heavily regularized linear baselines (Logistic Regression with $L_2$ penalty) perform on par with non-linear tree ensembles (Random Forest, XGBoost) on small-scale clinical cohorts ($N=303$)?
* **RQ2 (Clinical Sensitivity Trade-offs)**: What are the precision, recall, specificity, and false-negative trade-offs, and can models maintain high sensitivity ($>90\%$) to minimize missed disease cases?
* **RQ3 (Feature Attribution)**: What physiological factors primarily drive model predictions according to SHAP values?
* **RQ4 (Reproducibility)**: Can the complete pipeline—from automated retrieval and leakage-safe preprocessing to model serialization and API serving—be deterministically reproduced?
* **RQ5 (Stability)**: Given the sample size ($N=303$), how stable are cross-validation metrics across folds?

---

## 2. Dataset Hygiene, Provenance & Ingestion

### 2.1 Provenance & Cryptographic Verification
* **Source**: [UCI Machine Learning Repository Heart Disease (Cleveland Database)](https://archive.ics.uci.edu/dataset/45/heart+disease)
* **Direct Data URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
* **Raw Dimensions**: 303 observations × 14 attributes.
* **Integrity Hash**: Evaluated using SHA256 on byte stream:
  `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`

### 2.2 Feature Dictionary
| Feature | Type | Unit / Format | Clinical Description | Reference Range / Classes |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Continuous | Years | Patient age | 29 – 77 years |
| `sex` | Categorical | Binary | Biological sex | 1 = male; 0 = female |
| `cp` | Categorical | Nominal | Chest pain presentation | 1: Typ. Angina, 2: Atyp. Angina, 3: Non-anginal, 4: Asymptomatic |
| `trestbps` | Continuous | mm Hg | Resting systolic blood pressure | 94 – 200 mm Hg |
| `chol` | Continuous | mg/dl | Serum cholesterol level | 126 – 564 mg/dl |
| `fbs` | Categorical | Binary | Fasting blood sugar > 120 mg/dl | 1 = true; 0 = false |
| `restecg` | Categorical | Nominal | Resting electrocardiographic results | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy |
| `thalach` | Continuous | bpm | Max exercise heart rate achieved | 71 – 202 bpm |
| `exang` | Categorical | Binary | Exercise-induced angina | 1 = yes; 0 = no |
| `oldpeak` | Continuous | mm | Exercise-induced ST depression | 0.0 – 6.2 mm |
| `slope` | Categorical | Ordinal | Peak exercise ST segment slope | 1: Upsloping, 2: Flat, 3: Downsloping |
| `ca` | Discrete | Count (0-3) | Major vessels colored by fluoroscopy | 0, 1, 2, 3 vessels |
| `thal` | Categorical | Nominal | Thallium scintigraphy defect status | 3: Normal, 6: Fixed defect, 7: Reversible defect |
| `target` | Categorical | Binary | Angiographic disease status | 0 = Absence (<50% stenosis); 1 = Presence (≥50%) |

---

## 3. How We Cleaned the Data: Step-by-Step Methodology

### Step 1: Handling Non-Standard Missing Placeholders
In the raw Cleveland data file, missing records were entered as ASCII question marks (`?`). Loading the dataset using standard CSV parsers would cause pandas to infer object/string datatypes, masking missingness and corrupting numerical operations.

* **Audit Results**:
  * Feature `ca`: 4 missing values (1.32%)
  * Feature `thal`: 2 missing values (0.66%)
  * Total rows affected: 6 out of 303 (1.98%)
  * Duplicate rows: 0
* **Cleaning Execution** (`src/data_loader.py`):
```python
df = pd.read_csv(
    raw_path,
    header=None,
    names=RAW_COLUMN_NAMES,
    na_values=["?", "NA", "nan", ""],  # Coerces all placeholder strings to np.nan
    skipinitialspace=True
)
```

### Step 2: Target Variable Binarization
The raw diagnosis column (`num`) contains integer values 0, 1, 2, 3, 4 indicating the count of diseased major vessels:
* `num = 0`: Absence of CAD ($N=164, 54.12\%$)
* `num \in {1, 2, 3, 4}`: Presence of CAD ($N=139, 45.88\%$)

To prevent **target leakage**, we converted `num` to a binary `target` column and **immediately dropped the raw `num` column**:
```python
df[TARGET_NAME] = (df[TARGET_RAW_NAME] > 0).astype(int)
df.drop(columns=[TARGET_RAW_NAME], inplace=True)
```

### Step 3: Strict Train-Test Partitioning (Leakage Prevention)
To prevent statistical contamination (information about test distributions leaking into imputers and scalers), the 80/20 train-test split was executed **prior to calculating any feature statistics**:
* **Training partition**: $N_{\text{train}} = 242$ observations
* **Testing partition**: $N_{\text{test}} = 61$ observations (untouched holdout)
* **Stratification**: Preserved the exact $45.88\%$ class balance across train and test partitions (`random_state=42`).

### Step 4: Preprocessing Pipeline Architecture
Transformations are isolated using Scikit-Learn `ColumnTransformer` (`src/preprocessing.py`):

1. **Numeric Feature Pipeline**:
   * Continuous variables: `["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]`
   * **Imputation**: `SimpleImputer(strategy="median")`. Median was chosen over mean because clinical features such as `chol` (max $564\text{ mg/dl}$) and `oldpeak` (max $6.2\text{ mm}$) exhibit right-skewed distributions.
   * **Scaling**: `StandardScaler()`. Calculates $z = \frac{x - \mu}{\sigma}$ strictly from training folds.
2. **Categorical Feature Pipeline**:
   * Nominal variables: `["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]`
   * **Imputation**: `SimpleImputer(strategy="most_frequent")` (mode imputation).
   * **Encoding**: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
3. **Remainder Policy**: `remainder="drop"` ensures no unintended columns pass through to inference.

---

## 4. Exploratory Data Analysis (EDA) & Clinical Hypotheses

The automated EDA module (`src/eda.py`) produced 6 high-resolution figures stored in `reports/figures/`:

1. **Target Distribution (`01_target_distribution.png`)**:
   * 164 Absence ($54.1\%$) vs. 139 Presence ($45.9\%$). The balance allows training without artificial oversampling (SMOTE).
2. **Age & Physiological Distributions (`02_numeric_distributions.png`)**:
   * Mean age for positive cases ($56.6\text{ years}$) is higher than negative cases ($52.6\text{ years}$).
   * Resting blood pressure (`trestbps`) and cholesterol (`chol`) show substantial overlap between cohorts, indicating that non-invasive risk assessment requires multivariate modeling.
3. **Correlation Matrix (`03_correlation_matrix.png`)**:
   * Strongest positive correlations with disease: `oldpeak` ($r = +0.42$), `cp = 4.0` ($r = +0.41$), `exang = 1` ($r = +0.43$).
   * Strongest negative correlation (protective): `thalach` ($r = -0.42$).
4. **The Chest Pain Paradox (`04_categorical_distributions.png`)**:
   * Category 4 (Asymptomatic angina) is associated with the highest disease rate ($72.7\%$), while Category 1 (Typical angina) has lower disease representation. In clinical cohorts, patients who are asymptomatic yet referred for catheterization often present with silent myocardial ischemia.
5. **Fluoroscopy Vessels (`ca`) & Thallium Scintigraphy (`thal`)**:
   * Zero colored vessels (`ca = 0`) correlates with a low disease rate ($26\%$). When $\ge 1$ vessels are colored, disease probability rises to $>75\%$.
   * Reversible defects (`thal = 7`) demonstrate strong predictive power for active coronary ischemia.

---

## 5. Machine Learning Modeling & Cross-Validation Protocol

### 5.1 Validation Setup
* **Cross-Validation**: Stratified 5-Fold Cross-Validation on the 242-sample training set.
* **Selection Metric**: Primary optimization on **Cross-Validation ROC-AUC**.
* **Model Families Evaluated**:
  1. **Logistic Regression (L2 Regularization)**: Linear benchmark with calibrated log-odds outputs.
  2. **Random Forest Classifier**: Non-linear bagging ensemble of de-correlated decision trees.
  3. **XGBoost (Extreme Gradient Boosting)**: Gradient boosted decision trees with shrinkage and regularization.

### 5.2 Optimal Hyperparameter Configuration
* **Logistic Regression**: $C = 0.1$, `penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`.
* **Random Forest**: `n_estimators=100`, `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=1`.
* **XGBoost**: `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`, `n_estimators=50`.

---

## 6. Comprehensive Evaluation & Benchmark Results

### 6.1 Performance Summary Table (Test Set: $N=61$, 28 Positive Cases)
| Metric | Logistic Regression | Random Forest (Champion) | XGBoost |
| :--- | :---: | :---: | :---: |
| **CV ROC-AUC (Mean ± Std)** | $0.9025 \pm 0.0144$ | **$0.9038 \pm 0.0238$** | $0.8897 \pm 0.0246$ |
| **Test ROC-AUC** | **$0.9665$** | $0.9502$ | $0.9513$ |
| **Test Recall / Sensitivity** | **$92.86\%$** (26/28) | **$92.86\%$** (26/28) | **$92.86\%$** (26/28) |
| **Test Precision** | $83.87\%$ | $83.87\%$ | **$86.67\%$** |
| **Test Specificity** | $84.85\%$ (28/33) | $84.85\%$ (28/33) | **$87.88\%$** (29/33) |
| **Test F1-Score** | $0.8814$ | $0.8814$ | **$0.8966$** |
| **Test Accuracy** | $88.52\%$ (54/61) | $88.52\%$ (54/61) | **$90.16\%$** (55/61) |
| **False Negatives (FN)** | **2** | **2** | **2** |
| **False Positives (FP)** | 5 | 5 | **4** |

### 6.2 Key Scientific Insights
1. **Model Parity (RQ1)**: On small clinical cohorts ($N=303$), heavily regularized Logistic Regression ($C=0.1$) achieves performance comparable to non-linear tree ensembles ($0.9665$ vs. $0.9502$ ROC-AUC).
2. **Clinical Sensitivity (RQ2)**: All three models achieved $92.86\%$ recall, successfully detecting 26 of 28 positive heart disease patients on the test set.

---

## 7. Error Diagnostics & Misclassification Deep Dive

Out of 61 test patients, the champion Random Forest model misclassified 7 patients (2 False Negatives, 5 False Positives):

### 7.1 False Negatives (Critical Triage Misses)
1. **Case ID #10**: Positive patient presenting with completely normal exercise ST segment (`oldpeak = 0.0`, `slope = 1`), normal thallium scintigraphy (`thal = 3.0`), and zero fluoroscopy vessels (`ca = 0.0`). The patient had atypical presentation that evaded non-invasive indicators.
2. **Case ID #48**: Female patient with non-anginal chest pain (`cp = 3.0`) and high exercise capacity (`thalach = 172\text{ bpm}`). The high exercise heart rate masked underlying vessel narrowing.

### 7.2 False Positives (Secondary Testing Triggers)
5 patients without significant angiographic disease presented with risk profiles typical of severe CAD (age >60, resting blood pressure >150 mm Hg, abnormal ST depression >2.0 mm). In clinical practice, these cases trigger secondary diagnostic imaging rather than missed care.

---

## 8. Explainable Machine Learning (SHAP)

Using TreeExplainer on the champion Random Forest model across 28 encoded features:

### 8.1 Top 8 Predictive Clinical Drivers
| Rank | Feature | Mean \|SHAP\| | Directionality & Clinical Meaning |
| :---: | :--- | :---: | :--- |
| 1 | `thal_3.0` | **0.0734** | Normal thallium scan strongly lowers predicted risk. |
| 2 | `ca_0.0` | **0.0681** | Zero fluoroscopy colored vessels provides primary protection. |
| 3 | `thal_7.0` | **0.0545** | Reversible perfusion defect during stress testing elevates risk. |
| 4 | `cp_4.0` | **0.0511** | Asymptomatic chest pain presentation strongly elevates risk. |
| 5 | `oldpeak` | **0.0325** | Greater exercise-induced ST depression elevates risk. |
| 6 | `thalach` | **0.0302** | High maximum achieved heart rate reflects healthy cardiac reserve. |
| 7 | `exang_1.0` | **0.0277** | Exercise-induced angina increases predicted probability. |
| 8 | `slope_2.0` | **0.0267** | Flat ST segment slope during exercise increases risk. |

### 8.2 Local Waterfall Deconstruction
Individual prediction explanations decompose the model output from the base expectation $E[f(X)]$ to the final probability, showing exact positive and negative force contributions for each clinical variable.

---

## 9. System Architecture & Deployment

```
                    ┌───────────────────────────┐
                    │  Next.js 16 Web Portal    │
                    │  (Port 3000, App Router)  │
                    └─────────────┬─────────────┘
                                  │ HTTP / JSON
                                  ▼
                    ┌───────────────────────────┐
                    │    FastAPI REST Backend   │
                    │   (Port 8000, Pydantic)   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  Scikit-Learn Pipeline    │
                    │  (models/best_pipeline)   │
                    └───────────────────────────┘
```

1. **FastAPI Backend (`api/main.py`)**:
   * `GET /health`: Health status and loaded model metadata.
   * `POST /predict`: Validates 13 features via Pydantic schemas, runs inference through `best_pipeline.joblib`, and returns predicted class, probabilities, risk level, and medical disclaimer.
2. **Next.js 16 Web Portal (`frontend/`)**:
   * Minimalist soft-paint gouache palette: Warm ivory canvas (`#FAF8F4`), soft white cards (`#FFFFFF`), sage (`#DCE8D5`), mint (`#DCEFE8`), lavender (`#E9E0F2`), and forest green (`#355B4B`).
   * Sections: Overview, Interactive Prediction Workspace, Model Lab (tabbed benchmark curves), Explainability Studio (SHAP plots), and Research Notes.
3. **Automated Testing Suite (`tests/`)**:
   * 18 unit and integration tests passing (`pytest.ini`).

---

## 10. Medical Disclaimer & Known Limitations

> [!CAUTION]
> **DISCLAIMER**: CardioSense is an academic research and educational prototype developed for studying explainable machine learning methodologies. It is **not clinically validated**, **not certified as a medical device (CE/FDA)**, and **must not be used for real-world medical diagnosis, risk assessment, or treatment decisions**. Always consult a licensed healthcare professional for medical concerns.

### Known Limitations
1. **Cohort Vintage & Size**: The Cleveland dataset comprises 303 cases recorded in 1988. Contemporary cardiology employs advanced biomarkers (high-sensitivity troponin, coronary artery calcium scoring, CT angiography).
2. **Single-Center Demographics**: The cohort lacks multi-ethnic representation, limiting generalization to broader patient populations without domain adaptation.
3. **SHAP Interpretation**: Feature attributions reflect predictive associations within this specific feature space, not biochemical causation.
