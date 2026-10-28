"""
Pydantic schemas for the CardioSense REST API.
Enforces strict type checks, range validation, and standard response formats.
Supports dual-dataset schemas:
1. PatientInputSchemaIndia (Primary: 12 clinical predictors)
2. PatientInputSchemaUCI (Benchmark: 13 clinical predictors)
"""

from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class PatientInputSchemaIndia(BaseModel):
    """
    Input schema representing clinical patient measurements for the Indian Hospital Cohort.
    Source: Multi-specialty hospital in India (Doppala & Bhattacharyya, Mendeley Data, 2021).
    """
    age: float = Field(..., ge=18.0, le=100.0, description="Age in years (18-100)")
    gender: int = Field(..., description="Sex (1 = male, 0 = female)")
    chestpain: int = Field(..., ge=0, le=3, description="Chest pain type (0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic)")
    restingBP: float = Field(..., ge=70.0, le=260.0, description="Resting blood pressure in mm Hg (70-260)")
    serumcholestrol: float = Field(..., ge=0.0, le=700.0, description="Serum cholesterol in mg/dL (0 is treated as missing)")
    fastingbloodsugar: int = Field(..., description="Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)")
    restingrelectro: int = Field(..., ge=0, le=2, description="Resting ECG (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)")
    maxheartrate: float = Field(..., ge=60.0, le=230.0, description="Maximum heart rate achieved in bpm (60-230)")
    exerciseangia: int = Field(..., description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise (0.0 - 10.0)")
    slope: int = Field(..., ge=0, le=3, description="Slope of the peak exercise ST segment (0, 1: upsloping, 2: flat, 3: downsloping)")
    noofmajorvessels: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3) colored by fluoroscopy")

    @field_validator("gender", "fastingbloodsugar", "exerciseangia")
    @classmethod
    def validate_binary(cls, v: int, info) -> int:
        if v not in (0, 1):
            raise ValueError(f"Field '{info.field_name}' must be binary (0 or 1), got {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 53.0,
                "gender": 1,
                "chestpain": 2,
                "restingBP": 171.0,
                "serumcholestrol": 250.0,
                "fastingbloodsugar": 0,
                "restingrelectro": 1,
                "maxheartrate": 147.0,
                "exerciseangia": 0,
                "oldpeak": 5.3,
                "slope": 3,
                "noofmajorvessels": 3
            }
        }
    }


class PatientInputSchemaUCI(BaseModel):
    """
    Input schema representing clinical patient measurements for the UCI Cleveland Benchmark.
    Source: Cleveland Clinic Foundation (Janosi et al., 1988).
    """
    age: float = Field(..., ge=18.0, le=100.0, description="Age in years (18-100)")
    sex: int = Field(..., description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1: typical, 2: atypical, 3: non-anginal, 4: asymptomatic)")
    trestbps: float = Field(..., ge=80.0, le=250.0, description="Resting blood pressure in mm Hg (80-250)")
    chol: float = Field(..., ge=100.0, le=600.0, description="Serum cholesterol in mg/dl (100-600)")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)")
    thalach: float = Field(..., ge=60.0, le=230.0, description="Maximum heart rate achieved in bpm (60-230)")
    exang: int = Field(..., description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise (0.0 - 10.0)")
    slope: int = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)")
    ca: Optional[float] = Field(None, ge=0.0, le=3.0, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: Optional[float] = Field(None, description="Thalassemia (3: normal, 6: fixed defect, 7: reversible defect)")

    @field_validator("sex", "fbs", "exang")
    @classmethod
    def validate_binary(cls, v: int, info) -> int:
        if v not in (0, 1):
            raise ValueError(f"Field '{info.field_name}' must be binary (0 or 1), got {v}")
        return v

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v not in (3.0, 6.0, 7.0, 3, 6, 7):
            raise ValueError(f"Field 'thal' must be one of [3, 6, 7], got {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63.0,
                "sex": 1,
                "cp": 1,
                "trestbps": 145.0,
                "chol": 233.0,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150.0,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0.0,
                "thal": 6.0
            }
        }
    }


# Backward compatibility alias (existing tests import PatientInputSchema)
PatientInputSchema = PatientInputSchemaUCI


class PredictionResponseSchema(BaseModel):
    """Output schema for model prediction inference."""
    predicted_class: int = Field(..., description="Binary prediction: 0 = Absence, 1 = Presence")
    prediction_label: str = Field(..., description="Human-readable diagnosis interpretation")
    probability_presence: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of heart disease presence")
    probability_absence: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of absence")
    risk_level: Literal["Low", "Moderate", "High"] = Field(..., description="Categorized clinical risk level")
    model_used: str = Field(..., description="Identifier of the model used for inference")
    dataset_source: str = Field("India Hospital Cohort (Doppala & Bhattacharyya, Mendeley Data)", description="Origin cohort of the model")
    disclaimer: str = Field(..., description="Educational and non-diagnostic legal disclaimer")


class HealthResponseSchema(BaseModel):
    """Output schema for service health status."""
    status: str = Field(..., description="API operational status")
    model_loaded: bool = Field(..., description="Whether the trained ML pipeline is loaded in memory")
    model_name: str = Field(..., description="Name of the deployed champion model")
    active_dataset: str = Field("india", description="Primary active dataset for real-time predictions")
    available_datasets: List[str] = Field(default=["india", "uci"], description="List of supported dataset models")
    version: str = Field("2.0.0", description="CardioSense service version")
    disclaimer: str = Field(..., description="Medical disclaimer")
