"""Unit tests for five-class baseline model setup and metrics."""
import pandas as pd
import pytest

from src.preprocessing.nsl_kdd import MULTICLASS_ORDER
from src.training.multiclass_baselines import (
    calculate_multiclass_metrics,
    create_multiclass_models,
)


def test_five_multiclass_baseline_classifiers_are_configured():
    models = create_multiclass_models()
    assert len(models) == 5
    assert {
        "Logistic Regression", "Decision Tree", "Random Forest",
        "K-Nearest Neighbors", "Linear SVM",
    } == set(models)


def test_multiclass_metrics_include_all_five_classes():
    # One example per class means macro and weighted F1 have equal weights.
    y_true = pd.Series(["Normal", "DoS", "Probe", "R2L", "U2R"])
    y_pred = ["Normal", "DoS", "Probe", "Normal", "U2R"]
    metrics = calculate_multiclass_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["f1_macro"] == pytest.approx(0.7333333333)
    assert metrics["f1_weighted"] == pytest.approx(0.7333333333)
    assert list(MULTICLASS_ORDER) == ["Normal", "DoS", "Probe", "R2L", "U2R"]


def test_multiclass_metric_function_returns_expected_keys():
    metrics = calculate_multiclass_metrics(
        pd.Series(["Normal", "DoS"]), ["Normal", "DoS"]
    )
    assert set(metrics) == {
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"
    }
