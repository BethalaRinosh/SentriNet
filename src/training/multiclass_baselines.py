"""Five-class baseline experiments on the official NSL-KDD training split only.

Run from the repository root:
    python -m src.training.multiclass_baselines
    python -m src.training.multiclass_baselines --sample-size 30000

By default this uses the full official KDDTrain+ split and an internal
stratified validation split. KDDTest+ is never loaded by this module.
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
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.nsl_kdd import (
    MULTICLASS_ORDER,
    add_target_columns,
    load_nsl_kdd_file,
)
from src.preprocessing.pipeline import build_model_pipeline, select_features_and_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "baseline_multiclass"
RANDOM_STATE = 42


def create_multiclass_models() -> dict[str, ClassifierMixin]:
    """Return five deterministic baseline classifiers with imbalance-aware weights where supported."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, random_state=RANDOM_STATE, class_weight="balanced"
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
            class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000, dual=False, tol=1e-3
        ),
    }


def calculate_multiclass_metrics(
    y_true: pd.Series,
    y_pred: Any,
) -> dict[str, float]:
    """Return overall metrics that weight every class equally as well as by support."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(
            y_true, y_pred, labels=MULTICLASS_ORDER,
            average="macro", zero_division=0,
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, labels=MULTICLASS_ORDER,
            average="macro", zero_division=0,
        )),
        "f1_macro": float(f1_score(
            y_true, y_pred, labels=MULTICLASS_ORDER,
            average="macro", zero_division=0,
        )),
        "f1_weighted": float(f1_score(
            y_true, y_pred, labels=MULTICLASS_ORDER,
            average="weighted", zero_division=0,
        )),
    }


def run_multiclass_baselines(
    *,
    data_path: str | Path = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_size: int | None = None,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare five classifiers for Normal, DoS, Probe, R2L, and U2R."""
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be at least 1 when supplied.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    # Only the official training file is read; the official test file is untouched.
    frame = add_target_columns(load_nsl_kdd_file(data_path))
    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_state)

    X, y = select_features_and_target(frame, target="class_label")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=validation_size, random_state=random_state, stratify=y
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | int]] = []
    details: dict[str, Any] = {}

    for name, estimator in create_multiclass_models().items():
        pipeline = build_model_pipeline(estimator)
        fit_started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        predictions = pipeline.predict(X_valid)
        predict_seconds = time.perf_counter() - predict_started
        metrics = calculate_multiclass_metrics(y_valid, predictions)
        report = classification_report(
            y_valid, predictions, labels=MULTICLASS_ORDER,
            output_dict=True, zero_division=0,
        )
        matrix = confusion_matrix(
            y_valid, predictions, labels=MULTICLASS_ORDER
        )

        rows.append({
            "model": name,
            **metrics,
            "fit_seconds": round(fit_seconds, 4),
            "validation_predict_seconds": round(predict_seconds, 4),
            "training_rows": len(X_train),
            "validation_rows": len(X_valid),
            "sample_rows": len(frame),
            "random_state": random_state,
        })
        details[name] = {
            "metrics": metrics,
            "fit_seconds": round(fit_seconds, 4),
            "validation_predict_seconds": round(predict_seconds, 4),
            "class_order": MULTICLASS_ORDER,
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
        }
        print(f"\n{name}")
        print(pd.Series(rows[-1]).to_string())
        print("Per-class precision / recall / F1 / support:")
        print(classification_report(
            y_valid, predictions, labels=MULTICLASS_ORDER, zero_division=0
        ))

    results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    results.to_csv(output_dir / "multiclass_baseline_comparison.csv", index=False)
    with (output_dir / "multiclass_baseline_details.json").open("w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

    chart = results.sort_values("f1_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(chart["model"], chart["f1_macro"])
    ax.set(
        title="Five-class baseline comparison on KDDTrain+ validation split",
        xlabel="Macro F1",
        ylabel="Model",
        xlim=(0, 1),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(
        output_dir / "multiclass_baseline_macro_f1.png",
        dpi=160, bbox_inches="tight",
    )
    plt.close(fig)

    print("\nFive-class baseline summary (sorted by macro F1):")
    print(results[[
        "model", "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted", "fit_seconds",
        "validation_predict_seconds",
    ]].to_string(index=False))
    print(f"\nSaved comparison: {output_dir / 'multiclass_baseline_comparison.csv'}")
    print(f"Saved per-class details: {output_dir / 'multiclass_baseline_details.json'}")
    print(f"Saved chart: {output_dir / 'multiclass_baseline_macro_f1.png'}")
    print("Validation is an internal split from KDDTrain+; KDDTest+ was not loaded.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Optional reproducible sample size from KDDTrain+; default uses all rows.",
    )
    parser.add_argument(
        "--validation-size", type=float, default=0.2,
        help="Fraction of training split used for internal validation (default: 0.2).",
    )
    args = parser.parse_args()
    run_multiclass_baselines(
        sample_size=args.sample_size,
        validation_size=args.validation_size,
    )


if __name__ == "__main__":
    main()
