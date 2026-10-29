"""Unit tests for the final official-split evaluation utilities."""
import pandas as pd
import pytest

from src.training.official_evaluation import (
    TASKS,
    calculate_metrics,
    create_locked_candidates,
)


def test_locked_candidates_match_step_8_selected_parameters():
    binary = create_locked_candidates("binary_label")
    assert set(binary) == {"Random Forest", "Decision Tree"}
    assert binary["Random Forest"].max_features == "sqrt"
    assert binary["Random Forest"].min_samples_leaf == 1
    assert binary["Decision Tree"].max_depth == 20

    multiclass = create_locked_candidates("class_label")
    assert multiclass["Random Forest"].max_features == 0.5
    assert multiclass["Random Forest"].min_samples_leaf == 4
    assert multiclass["Decision Tree"].min_samples_leaf == 2
    assert multiclass["Decision Tree"].min_samples_split == 5


def test_locked_candidates_reject_unknown_task():
    with pytest.raises(ValueError, match="Unsupported task"):
        create_locked_candidates("unknown")


def test_task_label_orders_are_explicit_and_stable():
    assert TASKS["binary_label"]["labels"] == ["Normal", "Attack"]
    assert TASKS["class_label"]["labels"] == ["Normal", "DoS", "Probe", "R2L", "U2R"]


def test_binary_metrics_include_roc_auc_from_attack_scores():
    y_true = pd.Series(["Normal", "Normal", "Attack", "Attack"])
    y_pred = ["Normal", "Attack", "Attack", "Attack"]
    scores = [0.05, 0.70, 0.80, 0.95]

    metrics = calculate_metrics(
        y_true, y_pred, labels=["Normal", "Attack"], y_score=scores
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["f1_macro"] == pytest.approx(0.7333333333)
    assert metrics["f1_weighted"] == pytest.approx(0.75)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_multiclass_metrics_without_scores_have_no_roc_auc():
    labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    y_true = pd.Series(["Normal", "DoS", "Probe", "R2L", "U2R"])
    y_pred = ["Normal", "DoS", "Probe", "Normal", "U2R"]

    metrics = calculate_metrics(y_true, y_pred, labels=labels)

    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["precision_macro"] >= 0.0
    assert metrics["recall_macro"] >= 0.0
    assert metrics["f1_macro"] >= 0.0
    assert metrics["f1_weighted"] >= 0.0
    assert metrics["roc_auc"] is None
