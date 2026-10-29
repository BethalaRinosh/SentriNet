"""Unit tests for Step 8 cross-validation and tuning configuration."""
import numpy as np
import pytest

from src.training.cv_tuning import (
    build_stratified_cv,
    create_tuning_candidates,
)


def test_tuning_candidates_are_the_two_baseline_leaders():
    candidates = create_tuning_candidates("binary_label")
    assert set(candidates) == {"Random Forest", "Decision Tree"}
    for estimator, parameter_space in candidates.values():
        assert parameter_space
        assert all(key.startswith("model__") for key in parameter_space)
        assert estimator.random_state == 42


def test_stratified_cv_is_reproducible_and_keeps_class_proportions():
    X = np.arange(60).reshape(30, 2)
    y = np.array(["Normal"] * 20 + ["Attack"] * 10)
    cv = build_stratified_cv(n_splits=5, random_state=42)
    splits_a = list(cv.split(X, y))
    splits_b = list(build_stratified_cv(n_splits=5, random_state=42).split(X, y))

    assert len(splits_a) == 5
    for (train_a, valid_a), (train_b, valid_b) in zip(splits_a, splits_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(valid_a, valid_b)
        assert set(y[train_a]) == {"Normal", "Attack"}
        assert set(y[valid_a]) == {"Normal", "Attack"}


def test_stratified_cv_rejects_fewer_than_two_folds():
    with pytest.raises(ValueError, match="at least 2"):
        build_stratified_cv(n_splits=1)


def test_tuning_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unsupported task"):
        create_tuning_candidates("unknown_target")
