"""
CardioSense: Interactive Clinical ML Research & Demonstration Portal (Streamlit).
Supports dual-cohort risk evaluation:
1. Indian Hospital Cohort (Primary): Multi-specialty hospital in India (Doppala & Bhattacharyya, N=1000)
2. UCI Cleveland Benchmark (Historical baseline): Cleveland Clinic Foundation (Janosi et al., 1988, N=303)

Adheres to a soft pastel clinical palette (ivory, sage, mint, lavender, forest green).
Provides dynamic schema forms, real-time pipeline inference, benchmark metrics,
SHAP explainability plots, and ethical research disclaimers.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_DATASET,
    DATASETS_CONFIG,
    get_dataset_config,
    MODELS_DIR,
    BEST_PIPELINE_FILE,
    DISCLAIMER_TEXT,
    FIGURES_DIR
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & SOFT PASTEL STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioSense | Cardiovascular ML Portal",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Soft Pastel Aesthetic (Ivory, Sage, Mint, Lavender, Forest Green)
st.markdown("""
<style>
    /* Main Canvas: Warm Ivory */
    .stApp {
        background-color: #FAF8F5;
        color: #2D3748;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #E9E0F2 0%, #DCEFE8 50%, #DCE8D5 100%);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        border: 1px solid #D1D5DB;
    }
    .hero-title {
        color: #1F3D2B;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #374151;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Soft Pastel Cards */
    .pastel-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        margin-right: 8px;
    }
    .badge-mint { background-color: #DCEFE8; color: #164E63; }
    .badge-sage { background-color: #DCE8D5; color: #1F3D2B; }
    .badge-lavender { background-color: #E9E0F2; color: #4C1D95; }
    .badge-peach { background-color: #FDE2D2; color: #7C2D12; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR: DATASET COHORT SELECTOR & BENCHMARK PROVENANCE
# -----------------------------------------------------------------------------
st.sidebar.title("🫀 CardioSense")
st.sidebar.markdown("**Cohort & Model Selection**")

dataset_choice = st.sidebar.radio(
    "Select Clinical Cohort:",
    options=["india", "uci"],
    format_func=lambda x: "🇮🇳 Indian Hospital Cohort (N=1,000) [Primary]" if x == "india" else "🏛️ UCI Cleveland Benchmark (N=303) [Historical]",
    index=0
)

cfg = get_dataset_config(dataset_choice)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Cohort Details:**")
st.sidebar.markdown(f"• **Name:** {cfg['name']}")
st.sidebar.markdown(f"• **Origin:** {cfg['origin']}")
st.sidebar.markdown(f"• **Sample Size:** {cfg['expected_rows']} subjects")
st.sidebar.markdown(f"• **Authors:** {cfg['authors']}")
if cfg.get("doi"):
    st.sidebar.markdown(f"• **DOI:** `{cfg['doi']}`")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Notice**: In accordance with research standards, models and metrics for Indian and UCI cohorts are evaluated independently without data contamination."
)

# -----------------------------------------------------------------------------
# HERO BANNER & MEDICAL DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">CardioSense: Explainable Cardiovascular Risk Stratification</div>
    <div class="hero-subtitle">
        Active Research Cohort: <strong>{cfg['name']}</strong> ({cfg['origin']})<br>
        Benchmarked across L2-Regularized Logistic Regression, Random Forest Bagging, and XGBoost with SHAP game-theoretic explainability.
    </div>
</div>
""", unsafe_allow_html=True)

st.warning(f"⚠️ **Medical & Non-Diagnostic Disclaimer**: {DISCLAIMER_TEXT}")

# -----------------------------------------------------------------------------
# MODEL ARTIFACT LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_dataset_model(ds_key: str):
    """Loads champion pipeline artifact for specified dataset."""
    model_path = cfg["models_dir"] / "best_pipeline.joblib"
    if not model_path.exists():
        model_path = BEST_PIPELINE_FILE
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model for {ds_key}: {e}")
        return None

pipeline = load_dataset_model(dataset_choice)

if pipeline is None:
    st.error(
        f"Model artifact not found for `{cfg['short_name']}`. "
        f"Please run `python -m src.train --dataset {dataset_choice}` in your terminal."
    )
    st.stop()

# -----------------------------------------------------------------------------
# TABS: 1. Risk Estimator, 2. Benchmark Lab, 3. SHAP Studio, 4. Research Notes
# -----------------------------------------------------------------------------
tab_pred, tab_bench, tab_shap, tab_notes = st.tabs([
    "🩺 Patient Risk Simulator",
    "📊 Model Benchmark Lab",
    "🔍 SHAP Explainability Studio",
    "📖 Cohort & Epidemiology Notes"
])

# =============================================================================
# TAB 1: PATIENT RISK SIMULATOR
# =============================================================================
with tab_pred:
    st.subheader(f"Patient Diagnostic Form ({cfg['short_name']})")
    st.markdown("Enter non-invasive clinical vitals and diagnostic test findings to compute estimated risk:")

    with st.form("patient_input_form"):
        col1, col2, col3 = st.columns(3)

        if dataset_choice == "india":
            # ----------------- INDIA FORM (12 FEATURES) -----------------
            with col1:
                st.markdown("##### 1. Demographics & Intake Vitals")
                age = st.number_input("Patient Age (years)", min_value=18.0, max_value=100.0, value=52.0, step=1.0)
                gender = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
                chestpain = st.selectbox(
                    "Chest Pain Presentation",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: {
                        0: "0: Typical Angina",
                        1: "1: Atypical Angina",
                        2: "2: Non-Anginal Pain",
                        3: "3: Asymptomatic"
                    }[x]
                )
                restingBP = st.number_input("Resting Systolic BP (mm Hg)", min_value=70.0, max_value=250.0, value=145.0, step=1.0)

            with col2:
                st.markdown("##### 2. Metabolic & Resting Tests")
                serumcholestrol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0.0, max_value=650.0, value=240.0, step=5.0,
                                                  help="Note: Value of 0 will be median-imputed by pipeline")
                fastingbloodsugar = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dL",
                    options=[0, 1],
                    format_func=lambda x: "False (≤120 mg/dL) [0]" if x == 0 else "True (>120 mg/dL) [1]"
                )
                restingrelectro = st.selectbox(
                    "Resting ECG Findings",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "0: Normal",
                        1: "1: ST-T Wave Abnormality",
                        2: "2: Left Ventricular Hypertrophy"
                    }[x]
                )
                maxheartrate = st.number_input("Maximum Achieved Heart Rate (bpm)", min_value=60.0, max_value=230.0, value=150.0, step=1.0)

            with col3:
                st.markdown("##### 3. Exercise Stress & Angiography")
                exerciseangia = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
                oldpeak = st.number_input("ST Depression Relative to Rest (mm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
                slope = st.selectbox(
                    "ST Segment Slope at Peak Exercise",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: {0: "0: Upsloping (Type A)", 1: "1: Upsloping (Type B)", 2: "2: Flat (Ischemic)", 3: "3: Downsloping (Severe)"}[x]
                )
                noofmajorvessels = st.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3])

            submitted = st.form_submit_button("⚡ Compute Risk Stratification", use_container_width=True)

            if submitted:
                input_data = {
                    "age": age,
                    "gender": gender,
                    "chestpain": chestpain,
                    "restingBP": restingBP,
                    "serumcholestrol": np.nan if serumcholestrol == 0 else serumcholestrol,
                    "fastingbloodsugar": fastingbloodsugar,
                    "restingrelectro": restingrelectro,
                    "maxheartrate": maxheartrate,
                    "exerciseangia": exerciseangia,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "noofmajorvessels": noofmajorvessels
                }
                input_df = pd.DataFrame([input_data])[cfg["feature_names"]]

        else:
            # ----------------- UCI FORM (13 FEATURES) -----------------
            with col1:
                st.markdown("##### 1. Demographics & Vitals")
                age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=55.0, step=1.0)
                sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
                cp = st.selectbox(
                    "Chest Pain Type",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {1: "1: Typical Angina", 2: "2: Atypical Angina", 3: "3: Non-Anginal", 4: "4: Asymptomatic"}[x]
                )
                trestbps = st.number_input("Resting BP (mm Hg)", min_value=80.0, max_value=250.0, value=130.0, step=1.0)
                chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=240.0, step=5.0)

            with col2:
                st.markdown("##### 2. Metabolic & Resting Vitals")
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "False (0)" if x == 0 else "True (1)")
                restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: {0: "0: Normal", 1: "1: ST-T Abnormality", 2: "2: LV Hypertrophy"}[x])
                thalach = st.number_input("Max Heart Rate (bpm)", min_value=60.0, max_value=230.0, value=150.0, step=1.0)
                exang = st.selectbox("Exercise Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")

            with col3:
                st.markdown("##### 3. Stress & Scintigraphy")
                oldpeak = st.number_input("ST Depression (oldpeak in mm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope = st.selectbox("Peak Exercise ST Slope", options=[1, 2, 3], format_func=lambda x: {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}[x])
                ca = st.selectbox("Major Vessels Colored (ca, 0-3)", options=[0.0, 1.0, 2.0, 3.0])
                thal = st.selectbox("Thallium Scan (thal)", options=[3.0, 6.0, 7.0], format_func=lambda x: {3.0: "3: Normal", 6.0: "6: Fixed Defect", 7.0: "7: Reversible Defect"}[x])

            submitted = st.form_submit_button("⚡ Compute Risk Stratification", use_container_width=True)

            if submitted:
                input_data = {
                    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
                }
                input_df = pd.DataFrame([input_data])[cfg["feature_names"]]

    # Prediction Output Block
    if submitted:
        pred_class = int(pipeline.predict(input_df)[0])
        probs = pipeline.predict_proba(input_df)[0]
        p_presence = probs[1]
        p_absence = probs[0]

        st.markdown("---")
        st.markdown("### Diagnostic Prediction Output")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if pred_class == 1:
                st.error("🚨 **Classification: Disease Presence**")
            else:
                st.success("✅ **Classification: Disease Absence**")
            st.metric("Probability of Disease", f"{p_presence:.2%}")

        with res_col2:
            st.metric("Probability of Absence", f"{p_absence:.2%}")
            if p_presence < 0.35:
                risk_tag = "Low Risk"
                badge_class = "badge-mint"
            elif p_presence <= 0.65:
                risk_tag = "Moderate Risk"
                badge_class = "badge-peach"
            else:
                risk_tag = "High Risk"
                badge_class = "badge-lavender"
            st.markdown(f"Clinical Risk Stratum: <span class='metric-badge {badge_class}'>{risk_tag}</span>", unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"**Model Provenance**: `{type(pipeline.named_steps['classifier']).__name__}`")
            st.markdown(f"**Cohort Trained**: `{cfg['name']}`")
            st.markdown(f"**Preprocessing**: Leakage-Safe ColumnTransformer")

# =============================================================================
# TAB 2: MODEL BENCHMARK LAB
# =============================================================================
with tab_bench:
    st.subheader(f"Experimental Performance Benchmarks ({cfg['short_name']})")
    st.markdown("Evaluated under Stratified 5-Fold Cross-Validation on training data and tested blindly on untouched holdout test partition.")

    comp_csv = cfg["metrics_dir"] / "model_comparison.csv"
    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(
            df_comp.style.format({
                "cv_roc_auc_mean": "{:.4f}",
                "cv_roc_auc_std": "{:.4f}",
                "cv_f1_mean": "{:.4f}",
                "test_accuracy": "{:.2%}",
                "test_precision": "{:.2%}",
                "test_recall": "{:.2%}",
                "test_specificity": "{:.2%}",
                "test_f1": "{:.4f}",
                "test_roc_auc": "{:.4f}"
            }),
            use_container_width=True
        )
    else:
        st.info("Benchmark metrics table will appear once model training completes.")

    # Display Benchmark Figures
    st.markdown("#### Diagnostic Curves & Error Comparison")
    b_col1, b_col2 = st.columns(2)

    cm_fig = cfg["figures_dir"] / "confusion_matrices_comparison.png"
    roc_fig = cfg["figures_dir"] / "roc_curves_comparison.png"
    pr_fig = cfg["figures_dir"] / "precision_recall_curves_comparison.png"
    bar_fig = cfg["figures_dir"] / "models_metrics_comparison.png"

    with b_col1:
        if cm_fig.exists():
            st.image(str(cm_fig), caption="Held-Out Test Set Confusion Matrices", use_container_width=True)
        if pr_fig.exists():
            st.image(str(pr_fig), caption="Precision-Recall Curves Comparison", use_container_width=True)

    with b_col2:
        if roc_fig.exists():
            st.image(str(roc_fig), caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
        if bar_fig.exists():
            st.image(str(bar_fig), caption="Diagnostic Performance Metrics Comparison", use_container_width=True)

# =============================================================================
# TAB 3: SHAP EXPLAINABILITY STUDIO
# =============================================================================
with tab_shap:
    st.subheader(f"Game-Theoretic SHAP Feature Attribution ({cfg['short_name']})")
    st.markdown(
        "Shapley Additive exPlanations (SHAP) decomposes predictions into fair additive contributions for each clinical variable, "
        "satisfying cooperative game-theoretic axioms of Efficiency, Symmetry, and Additivity."
    )

    shap_beeswarm = cfg["figures_dir"] / "shap_summary_plot.png"
    shap_bar = cfg["figures_dir"] / "shap_feature_importance_bar.png"
    shap_correct = cfg["figures_dir"] / "shap_local_correct.png"
    shap_incorrect = cfg["figures_dir"] / "shap_local_incorrect.png"

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        if shap_beeswarm.exists():
            st.image(str(shap_beeswarm), caption="SHAP Global Beeswarm Summary Plot", use_container_width=True)
        if shap_correct.exists():
            st.image(str(shap_correct), caption="Local Waterfall: Correct Positive Patient", use_container_width=True)

    with s_col2:
        if shap_bar.exists():
            st.image(str(shap_bar), caption="Top 10 Most Influential Clinical Predictors", use_container_width=True)
        if shap_incorrect.exists():
            st.image(str(shap_incorrect), caption="Local Waterfall: Critical Misclassification Case", use_container_width=True)

# =============================================================================
# TAB 4: COHORT & EPIDEMIOLOGY NOTES
# =============================================================================
with tab_notes:
    st.subheader("Cardiovascular Disease in India: Clinical Context & Research Scope")
    st.markdown("""
    ### Why an India-Focused Cardiovascular Machine Learning Study?
    1. **Early Onset**: South Asian populations, particularly in India, experience premature coronary artery disease (CAD) 
       **1 to 2 decades earlier** than Western populations. Severe myocardial infarctions frequently occur before age 50.
    2. **High Prevalence of Dyslipidemia & Hypertension**: In the Indian hospital cohort, baseline resting blood pressure 
       averages **151.7 mm Hg** (substantially higher than Western cohorts), accompanied by high rates of central adiposity and diabetes.
    3. **Primary Research Dataset**:
       - **Source**: Mendeley Data, DOI: [10.17632/dzz48mvjht.1](https://doi.org/10.17632/dzz48mvjht.1)
       - **Authors**: Bhanu Prakash Doppala and Debnath Bhattacharyya (2021).
       - **Cohort**: 1,000 patient encounters from an Indian multi-specialty hospital.
    4. **Complementary National Registries**:
       - **ICMR-INDIAB**: [Indian Council of Medical Research](https://nhrdr.icmr.org.in/datasets/icmr-indiab) national diabetes and cardiovascular survey.
       - **ICADD**: Indian Coronary Artery Disease Dataset ([medRxiv](https://www.medrxiv.org/content/10.1101/2023.05.13.23289934v1)).
    """)
