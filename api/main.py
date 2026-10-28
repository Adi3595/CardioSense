"""
FastAPI application service for CardioSense.
Exposes endpoints for health checks and real-time model inference.
Supports dual-dataset serving:
1. India Hospital Cohort (Primary): Multi-specialty hospital in India (Doppala & Bhattacharyya, N=1000)
2. UCI Cleveland Benchmark (Historical baseline): Cleveland Clinic Foundation (Janosi et al., 1988, N=303)

Loads pre-trained, leakage-safe pipeline artifacts without retraining.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    PatientInputSchema,
    PatientInputSchemaIndia,
    PatientInputSchemaUCI,
    PredictionResponseSchema,
    HealthResponseSchema
)
from src.config import (
    DEFAULT_DATASET,
    DATASETS_CONFIG,
    get_dataset_config,
    MODELS_DIR,
    BEST_PIPELINE_FILE,
    MODEL_METADATA_FILE,
    DISCLAIMER_TEXT,
    FIGURES_DIR,
    PROJECT_ROOT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# State container for loaded models
app_state: Dict[str, Any] = {
    "pipelines": {},
    "metadata": {},
    "version": "2.0.0",
    "active_dataset": "india"
}


def load_model_artifacts():
    """Load fitted pipeline artifacts for all supported datasets."""
    # 1. Load India model (Primary)
    india_file = MODELS_DIR / "india" / "best_pipeline.joblib"
    if not india_file.exists():
        india_file = BEST_PIPELINE_FILE  # Fallback to root

    if india_file.exists():
        try:
            app_state["pipelines"]["india"] = joblib.load(india_file)
            logger.info(f"Successfully loaded India model artifact from {india_file}")
            meta_file = MODELS_DIR / "india" / "model_metadata.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    app_state["metadata"]["india"] = json.load(f)
        except Exception as e:
            logger.error(f"Error loading India model artifact: {e}")

    # 2. Load UCI model (Benchmark)
    uci_file = MODELS_DIR / "uci" / "best_pipeline.joblib"
    if not uci_file.exists() and "india" not in app_state["pipelines"]:
        uci_file = BEST_PIPELINE_FILE

    if uci_file.exists():
        try:
            app_state["pipelines"]["uci"] = joblib.load(uci_file)
            logger.info(f"Successfully loaded UCI model artifact from {uci_file}")
            meta_file = MODELS_DIR / "uci" / "model_metadata.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    app_state["metadata"]["uci"] = json.load(f)
        except Exception as e:
            logger.error(f"Error loading UCI model artifact: {e}")

    # Set default pipeline for legacy callers
    app_state["pipeline"] = app_state["pipelines"].get("india") or app_state["pipelines"].get("uci")
    app_state["model_name"] = (
        app_state["metadata"].get("india", {}).get("best_model_selected") or
        app_state["metadata"].get("uci", {}).get("best_model_selected", "Champion Pipeline")
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for FastAPI application."""
    load_model_artifacts()
    yield
    app_state["pipelines"].clear()
    app_state["pipeline"] = None


app = FastAPI(
    title="CardioSense ML Inference API",
    description="Explainable Machine Learning Service for Cardiovascular Disease Risk Stratification (India Cohort & UCI Benchmark).",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static figures and web frontend
app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")
app.mount("/web", StaticFiles(directory=PROJECT_ROOT / "web"), name="web")


def _run_inference(pipeline, input_df: pd.DataFrame, dataset_name: str, model_id: str) -> PredictionResponseSchema:
    """Helper function to execute prediction and format response."""
    try:
        pred_class = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]
        prob_absence = float(probabilities[0])
        prob_presence = float(probabilities[1])
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(e)}"
        )

    # Risk Level categorization
    if prob_presence < 0.35:
        risk_level = "Low"
    elif prob_presence <= 0.65:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    if "cleveland" in dataset_name.lower() or "uci" in dataset_name.lower():
        label = "Presence of Heart Disease" if pred_class == 1 else "Absence of Heart Disease"
    else:
        label = "Presence of Cardiovascular Disease" if pred_class == 1 else "Absence of Cardiovascular Disease"

    return PredictionResponseSchema(
        predicted_class=pred_class,
        prediction_label=label,
        probability_presence=round(prob_presence, 4),
        probability_absence=round(prob_absence, 4),
        risk_level=risk_level,
        model_used=model_id,
        dataset_source=dataset_name,
        disclaimer=DISCLAIMER_TEXT
    )


@app.get("/", tags=["General"])
async def root(request: Request):
    """Serves the interactive research dashboard or JSON metadata based on Accept header."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        dashboard_index = PROJECT_ROOT / "web" / "index.html"
        if dashboard_index.exists():
            return FileResponse(dashboard_index)
    return {
        "service": "CardioSense ML Service",
        "version": app_state["version"],
        "primary_focus": "Indian Hospital Cardiovascular Disease Risk Classification",
        "benchmark_support": "UCI Cleveland Heart Disease Cohort",
        "docs_url": "/docs",
        "health_check": "/health",
        "predict_india": "/predict/india",
        "predict_uci": "/predict/uci",
        "predict_general": "/predict",
        "disclaimer": DISCLAIMER_TEXT
    }


@app.get("/health", response_model=HealthResponseSchema, tags=["Health"])
async def health_check():
    """Service health and model availability check across all supported cohorts."""
    if not app_state["pipelines"]:
        load_model_artifacts()

    is_loaded = len(app_state["pipelines"]) > 0
    available = list(app_state["pipelines"].keys())

    return HealthResponseSchema(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=app_state.get("model_name", "Champion Pipeline"),
        active_dataset="india" if "india" in available else (available[0] if available else "none"),
        available_datasets=available,
        version=app_state["version"],
        disclaimer=DISCLAIMER_TEXT
    )


@app.post("/predict/india", response_model=PredictionResponseSchema, tags=["Inference"])
async def predict_india_patient(patient: PatientInputSchemaIndia):
    """
    Dedicated inference endpoint for the Indian Hospital Cohort (Doppala & Bhattacharyya).
    Validates against the 12 verified Indian clinical predictor schema.
    """
    pipeline = app_state["pipelines"].get("india")
    if pipeline is None:
        load_model_artifacts()
        pipeline = app_state["pipelines"].get("india")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="India model pipeline is not loaded.")

    cfg = get_dataset_config("india")
    data_dict = patient.model_dump()
    input_df = pd.DataFrame([data_dict])[cfg["feature_names"]]

    model_name = app_state["metadata"].get("india", {}).get("best_model_selected", "champion_xgboost")
    return _run_inference(pipeline, input_df, cfg["name"], model_name)


@app.post("/predict/uci", response_model=PredictionResponseSchema, tags=["Inference"])
async def predict_uci_patient(patient: PatientInputSchemaUCI):
    """
    Dedicated inference endpoint for the UCI Cleveland Benchmark (Janosi et al., 1988).
    Validates against the 13 canonical UCI clinical predictor schema.
    """
    pipeline = app_state["pipelines"].get("uci")
    if pipeline is None:
        load_model_artifacts()
        pipeline = app_state["pipelines"].get("uci")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="UCI benchmark pipeline is not loaded.")

    cfg = get_dataset_config("uci")
    data_dict = patient.model_dump()
    input_df = pd.DataFrame([data_dict])[cfg["feature_names"]]

    model_name = app_state["metadata"].get("uci", {}).get("best_model_selected", "champion_random_forest")
    return _run_inference(pipeline, input_df, cfg["name"], model_name)


@app.post("/predict", response_model=PredictionResponseSchema, tags=["Inference"])
async def predict_general(
    payload: Dict[str, Any] = Body(...),
    dataset: Optional[str] = Query(None, description="Target dataset: 'india' (default) or 'uci'")
):
    """
    General inference endpoint supporting dynamic schema validation and dataset selection.
    Automatically identifies whether payload matches the Indian or UCI schema if not specified.
    """
    from pydantic import ValidationError as PydanticValidationError
    from fastapi.exceptions import RequestValidationError

    if not app_state["pipelines"]:
        load_model_artifacts()

    # Determine dataset: from query parameter, or payload keys
    selected_ds = dataset.lower() if dataset else None
    if not selected_ds:
        if "gender" in payload or "restingBP" in payload or "maxheartrate" in payload:
            selected_ds = "india"
        elif "sex" in payload or "trestbps" in payload or "thalach" in payload or "thal" in payload:
            selected_ds = "uci"
        else:
            selected_ds = "india"  # Default primary

    try:
        if selected_ds == "uci":
            validated_input = PatientInputSchemaUCI(**payload)
            return await predict_uci_patient(validated_input)
        else:
            validated_input = PatientInputSchemaIndia(**payload)
            return await predict_india_patient(validated_input)
    except PydanticValidationError as e:
        raise RequestValidationError(e.errors())
