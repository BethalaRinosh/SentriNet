"""Training-only diagnostics for error analysis, imbalance and robustness.

Run from the repository root:
    python -m src.training.diagnostics

This experiment uses a stratified holdout carved only from KDDTrain+.
It never opens KDDTest+ and never changes the deployed model artifacts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.preprocessing.nsl_kdd import MULTICLASS_ORDER, add_target_columns, load_nsl_kdd_file
from src.preprocessing.pipeline import build_preprocessor, select_features_and_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "training_diagnostics"
RANDOM_STATE = 42
LABELS = MULTICLASS_ORDER


def per_class_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, dict[str, float | int]]:
    """Return stable per-class precision, recall, F1 and support."""
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in LABELS
    }


def summarize_predictions(y_true: pd.Series, y_pred: Any) -> dict[str, Any]:
    """Build class-aware metrics and a confusion matrix in fixed label order."""
    return {
        "accuracy": float((pd.Series(y_true).reset_index(drop=True) ==
                           pd.Series(y_pred).reset_index(drop=True)).mean()),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )),
        "per_class": per_class_metrics(y_true, y_pred),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def build_experiment_pipeline(strategy: str) -> ImbalancedPipeline:
    """Build baseline, class-weighted or train-fold oversampled Random Forest."""
    if strategy not in {"baseline", "class_weight", "random_oversampling"}:
        raise ValueError("Unknown strategy.")
    model = RandomForestClassifier(
        n_estimators=100,
        max_features=0.5,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample" if strategy == "class_weight" else None,
    )
    steps: list[tuple[str, Any]] = [("preprocessor", build_preprocessor())]
    if strategy == "random_oversampling":
        steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbalancedPipeline(steps)


def make_robustness_variants(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create explicit stress-test copies; original records remain unchanged."""
    variants = {"clean": X.copy()}
    unseen = X.copy()
    unseen["protocol_type"] = "__unseen_protocol__"
    unseen["service"] = "__unseen_service__"
    unseen["flag"] = "__unseen_flag__"
    variants["unseen_categories"] = unseen

    perturbed = X.copy()
    numeric = [
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "dst_host_count", "dst_host_srv_count",
    ]
    for column in numeric:
        values = pd.to_numeric(perturbed[column], errors="coerce").fillna(0)
        perturbed[column] = (values * 1.10).clip(lower=0)
    variants["selected_numeric_plus_10pct"] = perturbed
    return variants


def run_training_diagnostics(
    *,
    train_path: str | Path = TRAIN_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    validation_size: float = 0.25,
) -> dict[str, Any]:
    """Run all diagnostics on a stratified KDDTrain+-only holdout."""
    if not 0.1 <= validation_size <= 0.4:
        raise ValueError("validation_size must be between 0.1 and 0.4.")

    frame = add_target_columns(load_nsl_kdd_file(train_path))
    X, y = select_features_and_target(frame, target="class_label")
    X_train, X_valid, y_train, y_valid, raw_train, raw_valid = train_test_split(
        X, y, frame["attack"], test_size=validation_size,
        random_state=RANDOM_STATE, stratify=y,
    )

    results: dict[str, Any] = {
        "protocol": (
            "Stratified holdout split created exclusively from KDDTrain+. "
            "KDDTest+ was not loaded, scored, or used for model selection."
        ),
        "random_state": RANDOM_STATE,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "class_counts_train": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "class_counts_validation": {str(k): int(v) for k, v in y_valid.value_counts().items()},
        "imbalance_experiments": {},
        "robustness_experiments": {},
    }
    predictions_by_strategy: dict[str, pd.Series] = {}
    fitted_baseline = None

    for strategy in ("baseline", "class_weight", "random_oversampling"):
        pipeline = build_experiment_pipeline(strategy)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        predicted = pipeline.predict(X_valid)
        predictions_by_strategy[strategy] = pd.Series(predicted, index=y_valid.index)
        results["imbalance_experiments"][strategy] = {
            **summarize_predictions(y_valid, predicted),
            "fit_seconds": round(fit_seconds, 4),
        }
        if strategy == "baseline":
            fitted_baseline = pipeline

    # Error attribution by true NSL-KDD attack name, not just family.
    baseline_predictions = predictions_by_strategy["baseline"]
    error_frame = pd.DataFrame({
        "true_family": y_valid.astype(str),
        "predicted_family": baseline_predictions.astype(str),
        "raw_attack": raw_valid,
    }, index=y_valid.index)
    errors = error_frame[error_frame["true_family"] != error_frame["predicted_family"]]
    results["baseline_error_analysis"] = {
        "misclassified_rows": int(len(errors)),
        "misclassification_rate": float(len(errors) / len(error_frame)),
        "true_to_predicted_family": {
            f"{true} -> {pred}": int(count)
            for (true, pred), count in errors.groupby(
                ["true_family", "predicted_family"], observed=True
            ).size().sort_values(ascending=False).items()
        },
        "raw_attack_errors": {
            str(name): int(count)
            for name, count in errors["raw_attack"].value_counts().items()
        },
    }

    assert fitted_baseline is not None
    for variant_name, variant_X in make_robustness_variants(X_valid).items():
        variant_pred = fitted_baseline.predict(variant_X)
        results["robustness_experiments"][variant_name] = summarize_predictions(
            y_valid, variant_pred
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "training_diagnostics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    pd.DataFrame([
        {"strategy": name, "macro_f1": detail["macro_f1"],
         "accuracy": detail["accuracy"], "fit_seconds": detail["fit_seconds"]}
        for name, detail in results["imbalance_experiments"].items()
    ]).to_csv(destination / "imbalance_comparison.csv", index=False)
    pd.DataFrame([
        {"true_family": row.true_family, "predicted_family": row.predicted_family,
         "raw_attack": row.raw_attack}
        for row in errors.itertuples(index=False)
    ]).to_csv(destination / "baseline_misclassifications.csv", index=False)

    print("Training-only diagnostics complete.")
    print(f"Training rows: {len(X_train):,}; validation rows: {len(X_valid):,}")
    print(pd.DataFrame([
        {"strategy": name, "accuracy": detail["accuracy"],
         "macro_f1": detail["macro_f1"]}
        for name, detail in results["imbalance_experiments"].items()
    ]).to_string(index=False))
    print(f"Reports saved under: {destination}")
    print("KDDTest+ was not loaded. Experimental models do not replace deployed artifacts.")
    return results


def main() -> None:
    run_training_diagnostics()


if __name__ == "__main__":
    main()
