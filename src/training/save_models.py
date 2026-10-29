"""Train and persist the cross-validation-selected SentriNet pipelines.

This step deliberately selects models from KDDTrain+-only cross-validation,
then refits those model configurations on the full official KDDTrain+ split.
The official KDDTest+ file is never opened by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from src.preprocessing.nsl_kdd import add_target_columns, load_nsl_kdd_file
from src.preprocessing.pipeline import MODEL_FEATURE_COLUMNS, build_model_pipeline, select_features_and_target
from src.training.official_evaluation import create_locked_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42

# These are the recorded Step 8 CV results. The official test metrics do not
# participate in selecting the deployable candidate.
SELECTED_MODELS: dict[str, dict[str, Any]] = {
    "binary_label": {
        "model_name": "Random Forest",
        "artifact_name": "sentrinet_binary.joblib",
        "labels": ["Normal", "Attack"],
        "cv_folds": 5,
        "cv_macro_f1": 0.998955,
    },
    "class_label": {
        "model_name": "Random Forest",
        "artifact_name": "sentrinet_multiclass.joblib",
        "labels": ["Normal", "DoS", "Probe", "R2L", "U2R"],
        "cv_folds": 5,
        "cv_macro_f1": 0.946326,
    },
}


def build_model_metadata(
    *,
    task: str,
    model_name: str,
    estimator_parameters: dict[str, Any],
    training_rows: int,
) -> dict[str, Any]:
    """Build JSON-safe provenance metadata without test-set-derived values."""
    if task not in SELECTED_MODELS:
        raise ValueError(f"Unsupported task {task!r}; choose one of {list(SELECTED_MODELS)}.")
    selection = SELECTED_MODELS[task]
    return {
        "project": "SentriNet",
        "task": task,
        "task_description": (
            "Normal vs Attack" if task == "binary_label"
            else "Normal, DoS, Probe, R2L, U2R"
        ),
        "model_name": model_name,
        "selection_method": (
            "Highest recorded mean stratified cross-validation macro F1 among "
            "the tuned Random Forest and Decision Tree candidates. Selection "
            "used KDDTrain+ only; KDDTest+ was not used to choose this model."
        ),
        "cross_validation": {
            "folds": selection["cv_folds"],
            "scoring": "f1_macro",
            "mean_macro_f1": selection["cv_macro_f1"],
            "source": "Step 8 KDDTrain+-only randomized search",
        },
        "training": {
            "dataset": "NSL-KDD KDDTrain+",
            "rows": int(training_rows),
            "random_state": RANDOM_STATE,
            "fit_split": "full official KDDTrain+",
        },
        "features": list(MODEL_FEATURE_COLUMNS),
        "labels": list(selection["labels"]),
        "estimator_parameters": estimator_parameters,
        "preprocessing": [
            "StandardScaler for numeric features",
            "OneHotEncoder(handle_unknown='ignore') for protocol_type, service, flag",
        ],
        "runtime": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
        "artifact": selection["artifact_name"],
        "warning": (
            "Benchmark model only. Confidence values from predict_proba are "
            "not guaranteed to be calibrated probabilities."
        ),
    }


def save_selected_models(
    *,
    train_path: str | Path = TRAIN_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Fit the two CV-selected Random Forest pipelines and persist them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only the official training split is loaded here.
    train_path = Path(train_path)
    train_frame = add_target_columns(load_nsl_kdd_file(train_path))
    digest = hashlib.sha256()
    with train_path.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata: dict[str, Any] = {
        "project": "SentriNet",
        "training_dataset": "NSL-KDD KDDTrain+",
        "training_file": train_path.name,
        "training_file_sha256": digest.hexdigest(),
        "training_rows": int(len(train_frame)),
        "official_test_loaded": False,
        "selection_source": "Step 8 training-only cross-validation",
        "models": {},
    }

    for task, selection in SELECTED_MODELS.items():
        X_train, y_train = select_features_and_target(train_frame, target=task)
        candidates = create_locked_candidates(task)
        model_name = selection["model_name"]
        if model_name not in candidates:
            raise RuntimeError(f"Selected model {model_name!r} is not a candidate for {task}.")

        pipeline = build_model_pipeline(candidates[model_name])
        pipeline.fit(X_train, y_train)
        artifact_path = output_dir / selection["artifact_name"]
        joblib.dump(pipeline, artifact_path)

        estimator_parameters = pipeline.named_steps["model"].get_params(deep=False)
        # Convert NumPy scalar-like values to standard JSON values when present.
        safe_parameters = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in estimator_parameters.items()
        }
        model_metadata = build_model_metadata(
            task=task,
            model_name=model_name,
            estimator_parameters=safe_parameters,
            training_rows=len(X_train),
        )
        metadata["models"][task] = model_metadata
        print(f"Saved {task} pipeline: {artifact_path}")
        print(
            f"Selected using training-only CV macro F1 "
            f"{selection['cv_macro_f1']:.6f}; refit on {len(X_train):,} training rows."
        )

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved model metadata: {metadata_path}")
    print("Only KDDTrain+ was loaded. KDDTest+ was not loaded or scored.")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=TRAIN_PATH,
        help="Path to the official KDDTrain+ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for local model artifacts (default: models/).",
    )
    args = parser.parse_args()
    save_selected_models(train_path=args.train_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
