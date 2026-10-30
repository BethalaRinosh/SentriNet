"""Unit tests for training-only diagnostic helpers; no benchmark files required."""
import pandas as pd
import pytest

from src.preprocessing.pipeline import MODEL_FEATURE_COLUMNS
from src.training.diagnostics import (
    build_experiment_pipeline,
    make_robustness_variants,
    per_class_metrics,
    summarize_predictions,
)


def _sample_features():
    row = {name: 0 for name in MODEL_FEATURE_COLUMNS}
    row.update({"protocol_type": "tcp", "service": "http", "flag": "SF"})
    return pd.DataFrame([row, row.copy()])


def test_per_class_metrics_preserve_all_five_labels_and_support():
    truth = pd.Series(["Normal", "DoS", "Probe", "R2L", "U2R"])
    predicted = ["Normal", "DoS", "Normal", "R2L", "Normal"]
    result = per_class_metrics(truth, predicted)

    assert list(result) == ["Normal", "DoS", "Probe", "R2L", "U2R"]
    assert result["U2R"]["support"] == 1
    assert result["Probe"]["recall"] == 0.0


def test_summary_uses_stable_confusion_matrix_label_order():
    truth = pd.Series(["Normal", "DoS", "U2R"])
    result = summarize_predictions(truth, ["Normal", "Normal", "U2R"])

    assert result["confusion_matrix_labels"] == ["Normal", "DoS", "Probe", "R2L", "U2R"]
    assert len(result["confusion_matrix"]) == 5
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_robustness_variants_are_explicit_and_do_not_mutate_source():
    original = _sample_features()
    variants = make_robustness_variants(original)

    assert set(variants) == {
        "clean", "unseen_categories", "selected_numeric_plus_10pct"
    }
    assert variants["unseen_categories"].iloc[0]["service"] == "__unseen_service__"
    assert variants["selected_numeric_plus_10pct"].iloc[0]["src_bytes"] == 0
    assert original.iloc[0]["service"] == "http"


def test_experiment_pipeline_strategies_have_expected_sampling_and_weights():
    baseline = build_experiment_pipeline("baseline")
    weighted = build_experiment_pipeline("class_weight")
    oversampled = build_experiment_pipeline("random_oversampling")

    assert "sampler" not in baseline.named_steps
    assert weighted.named_steps["model"].class_weight == "balanced_subsample"
    assert "sampler" in oversampled.named_steps

    with pytest.raises(ValueError, match="Unknown strategy"):
        build_experiment_pipeline("smote_magic")
