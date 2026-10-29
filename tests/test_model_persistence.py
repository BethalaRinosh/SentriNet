"""Tests for persisted model metadata and reproducible inference."""
import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.inference.predict import ARTIFACTS, predict_records
from src.preprocessing.pipeline import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    build_model_pipeline,
)
from src.training.save_models import SELECTED_MODELS, build_model_metadata


def _feature_frame(rows: int = 25) -> pd.DataFrame:
    data = []
    categories = {
        "protocol_type": ["tcp", "udp", "icmp"],
        "service": ["http", "private", "domain_u"],
        "flag": ["SF", "S0", "REJ"],
    }
    for i in range(rows):
        row = {}
        for j, column in enumerate(MODEL_FEATURE_COLUMNS):
            if column in CATEGORICAL_FEATURES:
                row[column] = categories[column][i % len(categories[column])]
            else:
                row[column] = float((i + j) % 7)
        data.append(row)
    return pd.DataFrame(data)


@pytest.fixture
def saved_test_models(tmp_path):
    X = _feature_frame()
    binary_y = pd.Series(["Normal" if i % 2 == 0 else "Attack" for i in range(len(X))])
    multiclass_labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    multiclass_y = pd.Series([multiclass_labels[i % 5] for i in range(len(X))])

    for task, y in [("binary_label", binary_y), ("class_label", multiclass_y)]:
        pipeline = build_model_pipeline(
            RandomForestClassifier(n_estimators=5, random_state=42)
        )
        pipeline.fit(X, y)
        joblib.dump(pipeline, tmp_path / ARTIFACTS[task])
    return tmp_path, X.iloc[0].to_dict()


def test_selected_models_are_the_training_cv_winners():
    assert SELECTED_MODELS["binary_label"]["model_name"] == "Random Forest"
    assert SELECTED_MODELS["class_label"]["model_name"] == "Random Forest"
    assert SELECTED_MODELS["binary_label"]["cv_macro_f1"] == pytest.approx(0.998955)
    assert SELECTED_MODELS["class_label"]["cv_macro_f1"] == pytest.approx(0.946326)


def test_metadata_records_train_only_selection_and_feature_contract():
    metadata = build_model_metadata(
        task="binary_label",
        model_name="Random Forest",
        estimator_parameters={"n_estimators": 100, "random_state": 42},
        training_rows=125973,
    )
    assert metadata["training"]["dataset"] == "NSL-KDD KDDTrain+"
    assert metadata["training"]["rows"] == 125973
    assert metadata["cross_validation"]["scoring"] == "f1_macro"
    assert "KDDTest+ was not used" in metadata["selection_method"]
    assert metadata["features"] == MODEL_FEATURE_COLUMNS
    assert metadata["artifact"] == "sentrinet_binary.joblib"


def test_binary_inference_returns_label_confidence_and_attack_score(saved_test_models):
    artifact_dir, row = saved_test_models
    row["protocol_type"] = "future_protocol"
    row["service"] = "unseen_service"
    row["flag"] = "UNSEEN"
    row["num_outbound_cmds"] = 0  # Extra non-model feature is safely ignored.

    result = predict_records([row], task="binary_label", artifact_dir=artifact_dir)

    assert len(result) == 1
    assert result[0]["prediction"] in {"Normal", "Attack"}
    assert 0.0 <= result[0]["confidence"] <= 1.0
    assert 0.0 <= result[0]["attack_probability"] <= 1.0


def test_multiclass_inference_returns_supported_label(saved_test_models):
    artifact_dir, row = saved_test_models

    result = predict_records([row], task="class_label", artifact_dir=artifact_dir)

    assert result[0]["prediction"] in {"Normal", "DoS", "Probe", "R2L", "U2R"}
    assert 0.0 <= result[0]["confidence"] <= 1.0
    assert "attack_probability" not in result[0]


def test_inference_rejects_missing_features(saved_test_models):
    artifact_dir, row = saved_test_models
    del row["src_bytes"]

    with pytest.raises(ValueError, match="Missing required model features.*src_bytes"):
        predict_records([row], task="binary_label", artifact_dir=artifact_dir)


def test_inference_rejects_empty_records(saved_test_models):
    artifact_dir, _ = saved_test_models

    with pytest.raises(ValueError, match="At least one connection record"):
        predict_records([], task="binary_label", artifact_dir=artifact_dir)


def test_inference_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unsupported task"):
        predict_records([{}], task="unknown")
