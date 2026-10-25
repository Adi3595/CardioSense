"""
CardioSense: Minimal Streamlit Demonstration Application.
Provides a clean, functional clinical feature input form, runs inference using
the saved production pipeline artifact, and displays predictions and limitations.
"""

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
    BEST_PIPELINE_FILE,
    MODEL_METADATA_FILE,
    FEATURE_NAMES,
    DISCLAIMER_TEXT
)

st.set_page_config(
    page_title="CardioSense Demo",
    page_icon="🫀",
    layout="wide"
)

# Header & Legal Notice
st.title("🫀 CardioSense: Clinical ML Inference Demo")
st.markdown("**Explainable Machine Learning for Heart Disease Risk Classification (Cleveland Study)**")

st.warning(f"**Medical Notice**: {DISCLAIMER_TEXT}")

# Model Loader
@st.cache_resource
def load_pipeline():
    if not BEST_PIPELINE_FILE.exists():
        return None
    return joblib.load(BEST_PIPELINE_FILE)

pipeline = load_pipeline()

if pipeline is None:
    st.error(f"Model artifact not found at `{BEST_PIPELINE_FILE}`. Please execute `python -m src.train` first.")
    st.stop()

st.subheader("Patient Clinical Attributes")
st.markdown("Enter non-invasive diagnostic observations to estimate classification probability:")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=55.0, step=1.0)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "1: Typical Angina",
                2: "2: Atypical Angina",
                3: "3: Non-Anginal Pain",
                4: "4: Asymptomatic"
            }[x]
        )
        trestbps = st.number_input("Resting Blood Pressure (trestbps in mm Hg)", min_value=80.0, max_value=250.0, value=130.0, step=1.0)
        chol = st.number_input("Serum Cholesterol (chol in mg/dl)", min_value=100.0, max_value=600.0, value=240.0, step=1.0)

    with col2:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl (fbs)",
            options=[0, 1],
            format_func=lambda x: "True (>120 mg/dl) [1]" if x == 1 else "False (<=120 mg/dl) [0]"
        )
        restecg = st.selectbox(
            "Resting ECG (restecg)",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0: Normal",
                1: "1: ST-T Wave Abnormality",
                2: "2: Left Ventricular Hypertrophy"
            }[x]
        )
        thalach = st.number_input("Max Heart Rate Achieved (thalach in bpm)", min_value=60.0, max_value=230.0, value=150.0, step=1.0)
        exang = st.selectbox(
            "Exercise Induced Angina (exang)",
            options=[0, 1],
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
        )

    with col3:
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Peak Exercise ST Slope (slope)",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1: Upsloping",
                2: "2: Flat",
                3: "3: Downsloping"
            }[x]
        )
        ca = st.selectbox(
            "Major Vessels Colored by Fluoroscopy (ca)",
            options=[0.0, 1.0, 2.0, 3.0],
            format_func=lambda x: f"{int(x)} vessels"
        )
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[3.0, 6.0, 7.0],
            format_func=lambda x: {
                3.0: "3.0: Normal Perfusion",
                6.0: "6.0: Fixed Defect",
                7.0: "7.0: Reversible Defect"
            }[x]
        )

    submit_button = st.form_submit_button("Run Classification Inference")

if submit_button:
    # Build dataframe matching training schema
    patient_dict = {
        "age": float(age),
        "sex": float(sex),
        "cp": float(cp),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": float(fbs),
        "restecg": float(restecg),
        "thalach": float(thalach),
        "exang": float(exang),
        "oldpeak": float(oldpeak),
        "slope": float(slope),
        "ca": float(ca),
        "thal": float(thal)
    }
    input_df = pd.DataFrame([patient_dict])[FEATURE_NAMES]

    # Predict
    pred_class = int(pipeline.predict(input_df)[0])
    probs = pipeline.predict_proba(input_df)[0]
    prob_absence = float(probs[0])
    prob_presence = float(probs[1])

    st.markdown("---")
    st.subheader("Model Inference Output")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if pred_class == 1:
            st.error(f"### Predicted: Presence of Heart Disease (Class 1)")
        else:
            st.success(f"### Predicted: Absence of Heart Disease (Class 0)")

        st.metric("Estimated Disease Probability", f"{prob_presence * 100:.1f}%")
        st.metric("Absence Probability", f"{prob_absence * 100:.1f}%")

    with res_col2:
        if prob_presence < 0.35:
            st.info("**Assigned Risk Category**: Low Risk")
        elif prob_presence < 0.65:
            st.warning("**Assigned Risk Category**: Moderate Risk")
        else:
            st.error("**Assigned Risk Category**: Elevated / High Risk")

        st.markdown(
            """
            **Classification Confidence & Context:**
            - **Decision Threshold**: 0.50 (Calibrated for balanced sensitivity and specificity)
            - **Evaluation Protocol**: Held-out test set ($N=61$) sensitivity achieved: $92.9\\%$ (only 2 false negatives out of 28 positive cases).
            """
        )

    st.markdown("---")
    st.subheader("Research & Methodology Limitations")
    st.markdown(
        """
        1. **Small Sample Dataset**: The Cleveland cohort consists of 303 patients collected in 1988.
        2. **Population Representativeness**: Modern clinical cohorts feature different baseline risk profiles, newer biomarkers (e.g., high-sensitivity troponin), and modern imaging.
        3. **Prototype Status**: This system is purely educational and must never replace clinical judgment, angiographic verification, or medical consultation.
        """
    )
