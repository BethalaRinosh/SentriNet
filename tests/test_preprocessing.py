"""Unit tests for the preprocessing contract; no benchmark data is fabricated."""
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing.pipeline import (
    CATEGORICAL_FEATURES,
    CONSTANT_FEATURES,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_model_pipeline,
    build_preprocessor,
    select_features_and_target,
)


def test_model_features_exclude_targets_metadata_and_constant_column():
    assert len(MODEL_FEATURE_COLUMNS) == 40
    assert set(CATEGORICAL_FEATURES) == {"protocol_type", "service", "flag"}
    assert "num_outbound_cmds" in CONSTANT_FEATURES
    assert not ({"attack", "difficulty", "binary_label", "class_label"} & set(MODEL_FEATURE_COLUMNS))
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(MODEL_FEATURE_COLUMNS)


def test_select_features_and_target_returns_only_model_inputs():
    row = {column: 0 for column in MODEL_FEATURE_COLUMNS}
    row.update({"protocol_type": "tcp", "service": "http", "flag": "SF",
                "attack": "normal", "difficulty": 0,
                "binary_label": "Normal", "class_label": "Normal"})
    X, y = select_features_and_target(pd.DataFrame([row]), target="class_label")

    assert list(X.columns) == MODEL_FEATURE_COLUMNS
    assert "attack" not in X.columns
    assert "difficulty" not in X.columns
    assert y.iloc[0] == "Normal"


def test_only_supported_estimators_receive_balanced_class_weights():
    pipeline = build_model_pipeline(LogisticRegression(), balance_classes=True)
    assert pipeline.named_steps["model"].class_weight == "balanced"

    with pytest.raises(ValueError, match="does not support class_weight"):
        build_model_pipeline(KNeighborsClassifier(), balance_classes=True)


def test_preprocessor_is_unfitted_and_handles_unseen_categories():
    preprocessor = build_preprocessor()
    assert not hasattr(preprocessor, "transformers_")
    categorical_pipeline = {name: transformer for name, transformer, _ in preprocessor.transformers}["categorical"]
    encoder = categorical_pipeline.named_steps["onehot"]
    assert encoder.handle_unknown == "ignore"
