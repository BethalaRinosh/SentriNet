"""Final evaluation on the untouched official NSL-KDD KDDTest+ split.

Run from the repository root:
    python -m src.training.official_evaluation

Hyperparameters are locked to Step 8's KDDTrain+-only CV results. All four
pipelines are fitted on the full KDDTrain+ before this module loads KDDTest+.
The test split is evaluated once and must not be used to retune the models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.nsl_kdd import (
    MULTICLASS_ORDER,
    add_target_columns,
    load_nsl_kdd_file,
)
from src.preprocessing.pipeline import build_model_pipeline, select_features_and_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTest+.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "official_evaluation"
RANDOM_STATE = 42
TASKS = {
    "binary_label": {
        "description": "Binary: Normal vs Attack",
        "labels": ["Normal", "Attack"],
    },
    "class_label": {
        "description": "Five-class: Normal, DoS, Probe, R2L, U2R",
        "labels": MULTICLASS_ORDER,
    },
}


def create_locked_candidates(task: str) -> dict[str, Any]:
    """Build candidates using the exact best parameters recorded in Step 8."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}; choose one of {list(TASKS)}.")

    if task == "binary_label":
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            "Decision Tree": DecisionTreeClassifier(
                criterion="gini",
                max_depth=20,
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        }

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            max_features=0.5,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: Any,
    *,
    labels: list[str],
    y_score: Any | None = None,
) -> dict[str, float | None]:
    """Calculate class-aware metrics, with ROC-AUC only when scores are supplied."""
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "precision_weighted": float(precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )),
        "recall_weighted": float(recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )),
        "f1_weighted": float(f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )),
        "roc_auc": None,
    }
    if y_score is not None:
        result["roc_auc"] = float(roc_auc_score(y_true == "Attack", y_score))
    return result


def _distribution(frame: pd.DataFrame, target: str, labels: list[str]) -> dict[str, Any]:
    counts = frame[target].value_counts().reindex(labels, fill_value=0)
    total = int(len(frame))
    return {
        "rows": total,
        "counts": {label: int(counts[label]) for label in labels},
        "proportions": {
            label: float(counts[label] / total) for label in labels
        },
    }


def run_official_evaluation(
    *,
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Fit locked candidates on KDDTrain+, then evaluate once on KDDTest+."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: load only KDDTrain+, then fit all locked candidates.
    train_frame = add_target_columns(load_nsl_kdd_file(train_path))
    fitted: dict[str, dict[str, Any]] = {}
    fit_times: dict[str, float] = {}
    train_distributions: dict[str, Any] = {}

    for task, task_info in TASKS.items():
        X_train, y_train = select_features_and_target(train_frame, target=task)
        train_distributions[task] = _distribution(train_frame, task, task_info["labels"])
        fitted[task] = {}
        for model_name, estimator in create_locked_candidates(task).items():
            pipeline = build_model_pipeline(estimator)
            started = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            fitted[task][model_name] = pipeline
            fit_times[f"{task}|{model_name}"] = fit_seconds
            print(
                f"Fitted {model_name} for {task} on {len(X_train):,} KDDTrain+ rows "
                f"in {fit_seconds:.2f}s"
            )

    # Phase 2: only after model configurations are fixed and fitted, load KDDTest+.
    test_frame = add_target_columns(load_nsl_kdd_file(test_path))
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "evaluation_protocol": (
            "Hyperparameters fixed from Step 8 KDDTrain+-only CV. Each pipeline "
            "fit on full KDDTrain+ before KDDTest+ was loaded. KDDTest+ is not "
            "used for model selection or tuning."
        ),
        "random_state": RANDOM_STATE,
        "training_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_test_distribution": {},
        "tasks": {},
    }

    for task, task_info in TASKS.items():
        labels = task_info["labels"]
        X_test, y_test = select_features_and_target(test_frame, target=task)
        test_distribution = _distribution(test_frame, task, labels)
        details["train_test_distribution"][task] = {
            "train": train_distributions[task],
            "test": test_distribution,
            "test_minus_train_percentage_points": {
                label: 100.0 * (
                    test_distribution["proportions"][label]
                    - train_distributions[task]["proportions"][label]
                )
                for label in labels
            },
        }
        details["tasks"][task] = {"description": task_info["description"], "models": {}}

        for model_name, pipeline in fitted[task].items():
            predict_started = time.perf_counter()
            predictions = pipeline.predict(X_test)
            predict_seconds = time.perf_counter() - predict_started
            scores = None
            if task == "binary_label" and hasattr(pipeline, "predict_proba"):
                classes = list(pipeline.named_steps["model"].classes_)
                scores = pipeline.predict_proba(X_test)[:, classes.index("Attack")]

            metrics = calculate_metrics(
                y_test, predictions, labels=labels, y_score=scores
            )
            matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()
            report = classification_report(
                y_test, predictions, labels=labels, output_dict=True, zero_division=0
            )
            fit_seconds = fit_times[f"{task}|{model_name}"]
            row = {
                "task": task,
                "model": model_name,
                **metrics,
                "fit_seconds": round(fit_seconds, 4),
                "test_predict_seconds": round(predict_seconds, 4),
                "test_rows_per_second": round(
                    len(X_test) / predict_seconds if predict_seconds > 0 else 0.0, 2
                ),
                "train_rows": len(train_frame),
                "test_rows": len(test_frame),
            }
            rows.append(row)
            details["tasks"][task]["models"][model_name] = {
                "metrics": metrics,
                "fit_seconds": fit_seconds,
                "test_predict_seconds": predict_seconds,
                "test_rows_per_second": row["test_rows_per_second"],
                "confusion_matrix_labels": labels,
                "confusion_matrix": matrix,
                "classification_report": report,
                "test_support": {label: int((y_test == label).sum()) for label in labels},
            }

            fig, ax = plt.subplots(figsize=(7.5, 6))
            image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
            ax.set(
                title=f"{task_info['description']} | {model_name}\nOfficial KDDTest+",
                xlabel="Predicted label",
                ylabel="True label",
                xticks=range(len(labels)),
                yticks=range(len(labels)),
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
            threshold = max(max(values) for values in matrix) / 2 if matrix else 0
            for i, row_values in enumerate(matrix):
                for j, value in enumerate(row_values):
                    ax.text(j, i, str(value), ha="center", va="center",
                            color="white" if value > threshold else "black")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            safe_task = task.replace("_", "-")
            safe_model = model_name.lower().replace(" ", "-")
            fig.savefig(
                output_dir / f"confusion-{safe_task}-{safe_model}.png",
                dpi=160, bbox_inches="tight",
            )
            plt.close(fig)

            print(f"\n{task} | {model_name}")
            print(pd.Series(row).to_string())
            print(classification_report(
                y_test, predictions, labels=labels, target_names=labels, zero_division=0
            ))

    results = pd.DataFrame(rows).sort_values(["task", "f1_macro"], ascending=[True, False])
    results.to_csv(output_dir / "official_evaluation_summary.csv", index=False)
    with (output_dir / "official_evaluation_details.json").open("w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    chart = results.copy()
    chart["label"] = chart["task"].map({
        "binary_label": "Binary",
        "class_label": "Five-class",
    }) + " | " + chart["model"]
    chart = chart.sort_values("f1_macro", ascending=True)
    ax.barh(chart["label"], chart["f1_macro"])
    ax.set(
        title="Official KDDTest+ evaluation: macro F1",
        xlabel="Macro F1",
        ylabel="Task and model",
        xlim=(0, 1),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "official_evaluation_macro_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nOfficial KDDTest+ evaluation summary:")
    print(results[[
        "task", "model", "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted", "roc_auc", "fit_seconds",
        "test_predict_seconds", "test_rows_per_second",
    ]].to_string(index=False))
    print(f"Saved summary: {output_dir / 'official_evaluation_summary.csv'}")
    print(f"Saved details: {output_dir / 'official_evaluation_details.json'}")
    print(f"Saved chart and four confusion matrices under: {output_dir}")
    print("Evaluation complete. Do not use these test results to retune hyperparameters.")
    return results


def main() -> None:
    run_official_evaluation()


if __name__ == "__main__":
    main()
