"""FastAPI service for SentriNet's saved NSL-KDD classifiers."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.inference.predict import ARTIFACTS, DEFAULT_ARTIFACT_DIR, predict_records
from src.preprocessing.pipeline import CATEGORICAL_FEATURES, MODEL_FEATURE_COLUMNS

app = FastAPI(
    title="SentriNet API",
    description=(
        "Classify structured NSL-KDD connection-feature records. "
        "This API does not capture live packets."
    ),
    version="0.1.0",
)

# Allow the separately served local frontend to call this API during development.
# Restrict this to common local development origins rather than allowing every site.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

NUMERIC_FEATURES = [
    column for column in MODEL_FEATURE_COLUMNS
    if column not in CATEGORICAL_FEATURES
]


class PredictionRequest(BaseModel):
    """One or more connection records using NSL-KDD's model feature names."""

    model_config = ConfigDict(extra="forbid")
    records: list[dict[str, Any]] = Field(min_length=1)


class PredictionItem(BaseModel):
    prediction: str
    confidence: float | None = None
    attack_probability: float | None = None


class PredictionResponse(BaseModel):
    task: str
    count: int
    predictions: list[PredictionItem]


def _validate_records(records: list[dict[str, Any]]) -> None:
    """Raise a clear client error before calling the model on malformed input."""
    for index, record in enumerate(records):
        missing = [name for name in MODEL_FEATURE_COLUMNS if name not in record]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Record {index} is missing required features.",
                    "missing_features": missing,
                },
            )

        for name in CATEGORICAL_FEATURES:
            value = record[name]
            if not isinstance(value, str):
                raise HTTPException(
                    status_code=422,
                    detail=f"Record {index}: feature '{name}' must be a string.",
                )
            if not value.strip():
                raise HTTPException(
                    status_code=422,
                    detail=f"Record {index}: feature '{name}' must not be empty.",
                )

        for name in NUMERIC_FEATURES:
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"Record {index}: feature '{name}' must be numeric.",
                )
            if not math.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Record {index}: feature '{name}' must be finite.",
                )
            if value < 0:
                raise HTTPException(
                    status_code=422,
                    detail=f"Record {index}: feature '{name}' must be non-negative.",
                )


def _predict(request: PredictionRequest, task: str) -> PredictionResponse:
    _validate_records(request.records)
    try:
        results = predict_records(request.records, task=task)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Saved model artifacts are unavailable. Run "
                "'python -m src.training.save_models' from the repository root."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(
        task=task,
        count=len(results),
        predictions=[PredictionItem(**result) for result in results],
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Report service availability without loading large model artifacts."""
    present = {
        task: (Path(DEFAULT_ARTIFACT_DIR) / filename).is_file()
        for task, filename in ARTIFACTS.items()
    }
    return {
        "status": "ok",
        "models_ready": all(present.values()),
        "models_present": present,
        "input_contract": "40 NSL-KDD connection features per record",
    }


@app.post("/predict/binary", response_model=PredictionResponse)
def predict_binary(request: PredictionRequest) -> PredictionResponse:
    """Predict Normal or Attack for each submitted connection record."""
    return _predict(request, "binary_label")


@app.post("/predict/multiclass", response_model=PredictionResponse)
def predict_multiclass(request: PredictionRequest) -> PredictionResponse:
    """Predict Normal, DoS, Probe, R2L, or U2R for each record."""
    return _predict(request, "class_label")
