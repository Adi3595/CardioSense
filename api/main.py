"""
FastAPI application service for CardioSense.
Exposes endpoints for health checks and real-time model inference.
Loads the pre-trained, leakage-safe pipeline artifact without retraining.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    PatientInputSchema,
    PredictionResponseSchema,
    HealthResponseSchema
)
from src.config import (
    BEST_PIPELINE_FILE,
    MODEL_METADATA_FILE,
    FEATURE_NAMES,
    DISCLAIMER_TEXT,
    FIGURES_DIR,
    PROJECT_ROOT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# State container
app_state = {
    "pipeline": None,
    "model_name": "Unknown",
    "version": "1.0.0"
}


def load_model_artifact():
    """Load fitted pipeline artifact from disk."""
    if not BEST_PIPELINE_FILE.exists():
        logger.warning(f"Model artifact not found at {BEST_PIPELINE_FILE}")
        return None

    try:
        pipeline = joblib.load(BEST_PIPELINE_FILE)
        logger.info(f"Successfully loaded model artifact from {BEST_PIPELINE_FILE}")
        
        # Read model metadata if available
        if MODEL_METADATA_FILE.exists():
            with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                app_state["model_name"] = meta.get("best_model_selected", "Champion Pipeline")

        return pipeline
    except Exception as e:
        logger.error(f"Error loading model artifact: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for FastAPI application."""
    app_state["pipeline"] = load_model_artifact()
    yield
    app_state["pipeline"] = None


app = FastAPI(
    title="CardioSense ML Inference API",
    description="Explainable Machine Learning Service for Heart Disease Risk Classification.",
    version="1.0.0",
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
        "docs_url": "/docs",
        "dashboard_url": "/",
        "health_check": "/health",
        "predict_endpoint": "/predict",
        "disclaimer": DISCLAIMER_TEXT
    }


@app.get("/dashboard", tags=["General"])
async def dashboard():
    """Serves the interactive research dashboard."""
    dashboard_index = PROJECT_ROOT / "web" / "index.html"
    if dashboard_index.exists():
        return FileResponse(dashboard_index)
    raise HTTPException(status_code=404, detail="Dashboard index not found.")


@app.get("/health", response_model=HealthResponseSchema, tags=["Health"])
async def health_check():
    """Service health and model availability check."""
    # Attempt reload if not loaded
    if app_state["pipeline"] is None:
        app_state["pipeline"] = load_model_artifact()

    is_loaded = app_state["pipeline"] is not None
    return HealthResponseSchema(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=app_state["model_name"],
        version=app_state["version"],
        disclaimer=DISCLAIMER_TEXT
    )


@app.post("/predict", response_model=PredictionResponseSchema, tags=["Inference"])
async def predict_heart_disease(patient: PatientInputSchema):
    """
    Predicts heart disease presence/absence and estimated risk probability
    given patient clinical features.
    """
    # Verify model artifact is loaded
    if app_state["pipeline"] is None:
        app_state["pipeline"] = load_model_artifact()

    if app_state["pipeline"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline artifact is not loaded or missing from storage."
        )

    try:
        # Convert Pydantic model to DataFrame adhering to exact feature order
        input_data = patient.model_dump()
        df = pd.DataFrame([input_data])[FEATURE_NAMES]

        pipeline = app_state["pipeline"]
        predicted_class = int(pipeline.predict(df)[0])

        # Probabilities
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df)[0]
            prob_absence = float(probs[0])
            prob_presence = float(probs[1])
        else:
            prob_presence = 1.0 if predicted_class == 1 else 0.0
            prob_absence = 1.0 - prob_presence

        # Risk categorization
        if prob_presence < 0.35:
            risk_level = "Low"
        elif prob_presence < 0.65:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        label = "Presence of Heart Disease" if predicted_class == 1 else "Absence of Heart Disease"

        return PredictionResponseSchema(
            predicted_class=predicted_class,
            prediction_label=label,
            probability_presence=round(prob_presence, 4),
            probability_absence=round(prob_absence, 4),
            risk_level=risk_level,
            model_used=app_state["model_name"],
            disclaimer=DISCLAIMER_TEXT
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation failed: {str(e)}"
        )
