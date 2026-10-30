"""FastAPI service for SentriNet's saved NSL-KDD classifiers."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_CSV_RECORDS = 5000

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
            try:
                finite = math.isfinite(value)
            except OverflowError:
                # Extremely large JSON integers cannot be represented as floats.
                finite = False
            if not finite:
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



def _parse_csv_records(content: bytes) -> list[dict[str, Any]]:
    """Parse and validate a feature CSV before sending records to inference."""
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded CSV is empty.")
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV files must be 5 MB or smaller.")
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="CSV must use UTF-8 encoding.") from exc

    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    headers = reader.fieldnames
    if not headers:
        raise HTTPException(status_code=422, detail="CSV is missing a header row.")
    if len(headers) != len(set(headers)):
        raise HTTPException(status_code=422, detail="CSV contains duplicate column names.")

    missing = [name for name in MODEL_FEATURE_COLUMNS if name not in headers]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "CSV is missing required feature columns.", "missing_features": missing},
        )

    records: list[dict[str, Any]] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise HTTPException(
                status_code=422,
                detail=f"CSV row {row_number} has more values than the header.",
            )
        record: dict[str, Any] = {}
        for name in MODEL_FEATURE_COLUMNS:
            raw_value = row.get(name)
            if raw_value is None or not raw_value.strip():
                raise HTTPException(
                    status_code=422,
                    detail=f"CSV row {row_number}: feature '{name}' is empty.",
                )
            if name in CATEGORICAL_FEATURES:
                record[name] = raw_value.strip()
            else:
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=f"CSV row {row_number}: feature '{name}' must be numeric.",
                    ) from exc
                if not math.isfinite(value):
                    raise HTTPException(
                        status_code=422,
                        detail=f"CSV row {row_number}: feature '{name}' must be finite.",
                    )
                if value < 0:
                    raise HTTPException(
                        status_code=422,
                        detail=f"CSV row {row_number}: feature '{name}' must be non-negative.",
                    )
                record[name] = value
        records.append(record)
        if len(records) > MAX_CSV_RECORDS:
            raise HTTPException(
                status_code=413,
                detail=f"CSV files may contain at most {MAX_CSV_RECORDS} records.",
            )

    if not records:
        raise HTTPException(status_code=422, detail="CSV contains a header but no data records.")
    return records


@app.post("/predict/csv", response_model=PredictionResponse)
async def predict_csv(
    file: UploadFile = File(...),
    task: str = Form("binary_label"),
) -> PredictionResponse:
    """Validate a UTF-8 feature CSV and classify its rows in one request."""
    if task not in {"binary_label", "class_label"}:
        raise HTTPException(
            status_code=422,
            detail="task must be 'binary_label' or 'class_label'.",
        )
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload a .csv file.")
    content = await file.read(MAX_CSV_BYTES + 1)
    records = _parse_csv_records(content)
    return _predict(PredictionRequest(records=records), task)



@app.post("/validate/csv")
async def validate_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """Validate a CSV and return a small preview without running inference."""
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload a .csv file.")
    content = await file.read(MAX_CSV_BYTES + 1)
    records = _parse_csv_records(content)
    return {
        "valid": True,
        "count": len(records),
        "columns": len(MODEL_FEATURE_COLUMNS),
        "preview": records[:5],
    }
