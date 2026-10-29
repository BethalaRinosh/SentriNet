"""Unit tests for binary baseline model definitions and metrics."""
import pandas as pd
import pytest

from src.training.binary_baselines import calculate_metrics, create_binary_models


def test_five_baseline_classifiers_are_configured():
    models = create_binary_models()
    assert len(models) == 5
    assert {
        "Logistic Regression", "Decision Tree", "Random Forest",
        "K-Nearest Neighbors", "Linear SVM",
    } == set(models)


def test_binary_metrics_are_class_aware_and_reproducible():
    y_true = pd.Series(["Normal", "Attack", "Attack", "Normal"])
    y_pred = ["Normal", "Attack", "Normal", "Normal"]
    metrics = calculate_metrics(y_true, y_pred, y_score=[0.1, 0.9, 0.7, 0.2])

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["f1_macro"] == pytest.approx(0.7333333333)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_none_when_no_continuous_score_is_supplied():
    metrics = calculate_metrics(
        pd.Series(["Normal", "Attack"]), ["Normal", "Attack"]
    )
    assert metrics["roc_auc"] is None
