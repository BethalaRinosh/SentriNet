"""Leakage-safe stratified cross-validation and tuning for SentriNet.

Run from the repository root:
    python -m src.training.cv_tuning
    python -m src.training.cv_tuning --sample-size 30000 --n-iter 2

Only KDDTrain+ is loaded. KDDTest+ remains untouched for final evaluation.
Preprocessing is inside each estimator Pipeline, so every CV fold fits its own
encoder and scaler. The best candidate is refitted on all supplied training
rows by RandomizedSearchCV, but this module does not save a production model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.nsl_kdd import add_target_columns, load_nsl_kdd_file
from src.preprocessing.pipeline import build_model_pipeline, select_features_and_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "cv_tuning"
RANDOM_STATE = 42
TARGETS = {
    "binary_label": "Binary: Normal vs Attack",
    "class_label": "Five-class: Normal, DoS, Probe, R2L, U2R",
}


def create_tuning_candidates(
    task: str, random_state: int = RANDOM_STATE
) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return the two baseline leaders and compact randomized-search spaces."""
    if task not in TARGETS:
        raise ValueError(f"Unsupported task {task!r}; choose one of {list(TARGETS)}.")

    return {
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            {
                "model__max_depth": [None, 20, 30],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    }


def build_stratified_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Create a shuffled, reproducible stratified K-fold splitter."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert NumPy scalar-like parameter values to JSON-friendly Python values."""
    safe: dict[str, Any] = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            value = value.item()
        safe[key] = value
    return safe


def run_cv_tuning(
    *,
    data_path: str | Path = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_size: int | None = None,
    cv_folds: int = 5,
    n_iter: int = 4,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Tune Random Forest and Decision Tree for both targets using macro F1."""
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be at least 1 when supplied.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    # Deliberately load only the official training split.
    frame = add_target_columns(load_nsl_kdd_file(data_path))
    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    for target, target_description in TARGETS.items():
        X, y = select_features_and_target(frame, target=target)
        class_counts = y.value_counts(dropna=False)
        if class_counts.empty or int(class_counts.min()) < cv_folds:
            raise ValueError(
                f"Cannot run {cv_folds}-fold stratified CV for {target}: "
                f"smallest class has {int(class_counts.min()) if not class_counts.empty else 0} "
                "records. Use more training rows or fewer folds."
            )

        splitter = build_stratified_cv(cv_folds, random_state)
        details[target] = {
            "description": target_description,
            "training_rows": int(len(frame)),
            "class_counts": {str(label): int(count) for label, count in class_counts.items()},
            "scoring": "f1_macro",
            "cv_folds": cv_folds,
            "random_state": random_state,
            "models": {},
        }

        for model_name, (estimator, parameter_space) in create_tuning_candidates(
            target, random_state=random_state
        ).items():
            pipeline = build_model_pipeline(estimator)
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameter_space,
                n_iter=n_iter,
                scoring="f1_macro",
                cv=splitter,
                refit=True,
                n_jobs=1,
                random_state=random_state,
                return_train_score=False,
                error_score="raise",
                verbose=1,
            )
            print(f"\nTuning {model_name} | {target_description}")
            search.fit(X, y)

            best_params = _json_safe_params(search.best_params_)
            best_cv = float(search.best_score_)
            cv_results = []
            for index, params in enumerate(search.cv_results_["params"]):
                cv_results.append({
                    "rank": int(search.cv_results_["rank_test_score"][index]),
                    "mean_test_f1_macro": float(search.cv_results_["mean_test_score"][index]),
                    "std_test_f1_macro": float(search.cv_results_["std_test_score"][index]),
                    "mean_fit_seconds": float(search.cv_results_["mean_fit_time"][index]),
                    "params": _json_safe_params(params),
                })

            summary_rows.append({
                "target": target,
                "target_description": target_description,
                "model": model_name,
                "best_cv_f1_macro": best_cv,
                "cv_f1_macro_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                "cv_folds": cv_folds,
                "randomized_candidates": n_iter,
                "training_rows": len(frame),
                "random_state": random_state,
                "best_params_json": json.dumps(best_params, sort_keys=True),
            })
            details[target]["models"][model_name] = {
                "best_cv_f1_macro": best_cv,
                "best_cv_f1_macro_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "best_params": best_params,
                "tested_candidates": cv_results,
                "refit_on_all_training_rows": True,
                "model_artifact_saved": False,
            }
            print(f"Best mean CV macro F1: {best_cv:.6f} ± "
                  f"{details[target]['models'][model_name]['best_cv_f1_macro_std']:.6f}")
            print(f"Best parameters: {json.dumps(best_params, sort_keys=True)}")

    results = pd.DataFrame(summary_rows).sort_values(
        ["target", "best_cv_f1_macro"], ascending=[True, False]
    )
    results.to_csv(output_dir / "cv_tuning_summary.csv", index=False)
    with (output_dir / "cv_tuning_details.json").open("w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

    chart_data = results.copy()
    chart_data["label"] = chart_data["target"].map({
        "binary_label": "Binary",
        "class_label": "Five-class",
    }) + " | " + chart_data["model"]
    chart_data = chart_data.sort_values("best_cv_f1_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(chart_data["label"], chart_data["best_cv_f1_macro"])
    ax.set(
        title=f"Best {cv_folds}-fold CV macro F1 after randomized tuning",
        xlabel="Mean cross-validation macro F1",
        ylabel="Task and model",
        xlim=(0, 1),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "cv_tuning_macro_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nCross-validation tuning summary:")
    print(results[[
        "target", "model", "best_cv_f1_macro", "cv_f1_macro_std",
        "cv_folds", "randomized_candidates",
    ]].to_string(index=False))
    print(f"\nSaved summary: {output_dir / 'cv_tuning_summary.csv'}")
    print(f"Saved details: {output_dir / 'cv_tuning_details.json'}")
    print(f"Saved chart: {output_dir / 'cv_tuning_macro_f1.png'}")
    print("Only KDDTrain+ was loaded. KDDTest+ was not loaded or scored.")
    print("Best pipelines were refitted in memory but no model artifact was saved.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Optional reproducible sample from KDDTrain+; default uses all rows.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds (default: 5).")
    parser.add_argument("--n-iter", type=int, default=4, help="Random parameter settings per model/task (default: 4).")
    args = parser.parse_args()
    run_cv_tuning(
        sample_size=args.sample_size,
        cv_folds=args.cv_folds,
        n_iter=args.n_iter,
    )


if __name__ == "__main__":
    main()
