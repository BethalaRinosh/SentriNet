"""Tests for training-split feature-importance aggregation."""
import pytest

from src.feature_selection import aggregate_importances
from src.preprocessing.pipeline import MODEL_FEATURE_COLUMNS


def test_one_hot_importances_are_aggregated_to_original_features():
    result = aggregate_importances(
        ["duration", "protocol_type_tcp", "protocol_type_udp", "service_http"],
        [0.1, 0.2, 0.3, 0.4],
        raw_features=["duration", "protocol_type", "service"],
    )
    values = result.set_index("feature")["importance"].to_dict()
    assert values["duration"] == pytest.approx(0.1)
    assert values["protocol_type"] == pytest.approx(0.5)
    assert values["service"] == pytest.approx(0.4)


def test_aggregation_rejects_unmapped_or_mismatched_inputs():
    with pytest.raises(ValueError, match="equal length"):
        aggregate_importances(["duration"], [], raw_features=["duration"])
    with pytest.raises(ValueError, match="Cannot map"):
        aggregate_importances(["unknown_encoded_feature"], [1.0], raw_features=["duration"])


def test_default_raw_feature_list_matches_preprocessing_contract():
    assert len(MODEL_FEATURE_COLUMNS) == 40
