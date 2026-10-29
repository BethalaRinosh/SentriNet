"""Binary baseline experiments on the official NSL-KDD training split only.

Run from the repository root:
    python -m src.training.binary_baselines
    python -m src.training.binary_baselines --sample-size 30000

The official KDDTest+ file is intentionally not loaded here. A stratified
validation subset is carved from KDDTrain+ for baseline comparison; later
steps will tune candidates with cross-validation and reserve KDDTest+ for final
held-out evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.nsl_kdd import (
    add_target_columns,
    load_nsl_kdd_file,
)
from src.preprocessing.pipeline import build_model_pipeline, select_features_and_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "baseline_binary"
DEFAULT_SAMPLE_SIZE = 30_000
RANDOM_STATE = 42


def create_binary_models() -> dict[str, ClassifierMixin]:
    """Return a compact, deterministic set of five classical classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Linear SVM": LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000
        ),
    }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: Any,
    *,
    y_score: Any | None = None,
) -> dict[str, float | None]:
    """Calculate class-aware binary metrics and optional ROC-AUC."""
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "f1_weighted": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "roc_auc": None,
    }
    if y_score is not None:
        result["roc_auc"] = float(roc_auc_score(
            (pd.Series(y_true).astype(str) == "Attack").astype(int), y_score
        ))
    return result


def _positive_attack_scores(pipeline: Any, X: pd.DataFrame) -> Any | None:
    """Return continuous Attack scores for ROC-AUC when the estimator supports it."""
    model = pipeline.named_steps["model"]
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        classes = list(model.classes_)
        if "Attack" in classes:
            return probabilities[:, classes.index("Attack")]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        classes = list(model.classes_)
        if len(classes) == 2 and classes[1] == "Attack":
            return scores
        if len(classes) == 2 and classes[0] == "Attack":
            return -scores
    return None


def run_binary_baselines(
    *,
    data_path: str | Path = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare five models using a stratified validation split from KDDTrain+."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    # Only the official training file is read. KDDTest+ is reserved for Step 9.
    frame = add_target_columns(load_nsl_kdd_file(data_path))
    if sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_state)

    X, y = select_features_and_target(frame, target="binary_label")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=validation_size, random_state=random_state, stratify=y
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | None]] = []
    details: dict[str, Any] = {}

    for name, estimator in create_binary_models().items():
        pipeline = build_model_pipeline(estimator)
        fit_started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        predictions = pipeline.predict(X_valid)
        predict_seconds = time.perf_counter() - predict_started
        scores = _positive_attack_scores(pipeline, X_valid)
        metrics = calculate_metrics(y_valid, predictions, y_score=scores)

        row: dict[str, float | str | None] = {
            "model": name,
            **metrics,
            "fit_seconds": round(fit_seconds, 4),
            "validation_predict_seconds": round(predict_seconds, 4),
            "training_rows": len(X_train),
            "validation_rows": len(X_valid),
            "sample_rows": len(frame),
            "random_state": random_state,
        }
        rows.append(row)
        details[name] = {
            "metrics": metrics,
            "fit_seconds": round(fit_seconds, 4),
            "validation_predict_seconds": round(predict_seconds, 4),
            "confusion_matrix_labels": ["Normal", "Attack"],
            "confusion_matrix": confusion_matrix(
                y_valid, predictions, labels=["Normal", "Attack"]
            ).tolist(),
            "classification_report": classification_report(
                y_valid, predictions, labels=["Normal", "Attack"],
                output_dict=True, zero_division=0,
            ),
        }
        print(f"\n{name}")
        print(pd.Series(row).to_string())

    results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    results.to_csv(output_dir / "binary_baseline_comparison.csv", index=False)
    with (output_dir / "binary_baseline_details.json").open("w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

    chart = results.sort_values("f1_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(chart["model"], chart["f1_macro"])
    ax.set(
        title="Binary baseline comparison on KDDTrain+ validation split",
        xlabel="Macro F1",
        ylabel="Model",
        xlim=(0, 1),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "binary_baseline_macro_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nBaseline summary (sorted by macro F1):")
    print(results[[
        "model", "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted", "roc_auc", "fit_seconds",
        "validation_predict_seconds",
    ]].to_string(index=False))
    print(f"\nSaved comparison: {output_dir / 'binary_baseline_comparison.csv'}")
    print(f"Saved per-model details: {output_dir / 'binary_baseline_details.json'}")
    print(f"Saved chart: {output_dir / 'binary_baseline_macro_f1.png'}")
    print("Validation is an internal split from KDDTrain+; KDDTest+ was not loaded.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="Maximum reproducible sample from KDDTrain+ (default: 30000).",
    )
    parser.add_argument(
        "--validation-size", type=float, default=0.2,
        help="Fraction of the sampled training data used for validation.",
    )
    args = parser.parse_args()
    run_binary_baselines(
        sample_size=args.sample_size,
        validation_size=args.validation_size,
    )


if __name__ == "__main__":
    main()
