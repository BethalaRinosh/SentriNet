"""Load saved SentriNet pipelines and predict from connection feature records.

Input records must contain the 40 model features defined by the preprocessing
module. Additional fields are ignored, but missing required features are errors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from src.preprocessing.pipeline import MODEL_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models"
ARTIFACTS = {
    "binary_label": "sentrinet_binary.joblib",
    "class_label": "sentrinet_multiclass.joblib",
}


def load_model(
    task: str,
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> Any:
    """Load a persisted pipeline for binary or five-class prediction."""
    if task not in ARTIFACTS:
        raise ValueError(f"Unsupported task {task!r}; choose one of {list(ARTIFACTS)}.")
    artifact_path = Path(artifact_dir) / ARTIFACTS[task]
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found: {artifact_path}. "
            "Run 'python -m src.training.save_models' first."
        )
    return joblib.load(artifact_path)


def predict_records(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    task: str = "binary_label",
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> list[dict[str, Any]]:
    """Predict labels and model confidence for one or more connection records.

    The confidence is the maximum predict_proba score, not a calibrated
    probability. Binary output additionally includes the model's Attack score.
    """
    if task not in ARTIFACTS:
        raise ValueError(f"Unsupported task {task!r}; choose one of {list(ARTIFACTS)}.")

    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        if not records:
            raise ValueError("At least one connection record is required.")
        frame = pd.DataFrame(list(records))

    if frame.empty:
        raise ValueError("At least one connection record is required.")

    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))

    # Explicitly enforce training-time feature order and exclude all extra fields.
    X = frame.loc[:, MODEL_FEATURE_COLUMNS]
    pipeline = load_model(task, artifact_dir=artifact_dir)
    predictions = pipeline.predict(X)

    model = pipeline.named_steps["model"]
    probabilities = pipeline.predict_proba(X) if hasattr(pipeline, "predict_proba") else None
    classes = list(model.classes_)
    output: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions):
        item: dict[str, Any] = {"prediction": str(prediction)}
        if probabilities is not None:
            row_scores = probabilities[index]
            item["confidence"] = float(max(row_scores))
            if task == "binary_label" and "Attack" in classes:
                item["attack_probability"] = float(row_scores[classes.index("Attack")])
        output.append(item)
    return output
