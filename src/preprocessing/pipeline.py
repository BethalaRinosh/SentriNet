"""Leakage-safe preprocessing for SentriNet's NSL-KDD experiments.

Use the returned sklearn Pipeline and call fit only with training data. The
official test split must only be passed to predict/score after training.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

from src.preprocessing.nsl_kdd import FEATURE_COLUMNS

# NSL-KDD documents num_outbound_cmds as a constant-zero feature. Exclude it
# from model input because it cannot help distinguish classes.
CONSTANT_FEATURES = ["num_outbound_cmds"]
MODEL_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CONSTANT_FEATURES
]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERIC_FEATURES = [
    column for column in MODEL_FEATURE_COLUMNS
    if column not in CATEGORICAL_FEATURES
]
TARGET_COLUMNS = {"attack", "difficulty", "binary_label", "class_label"}


def select_features_and_target(
    frame: pd.DataFrame,
    target: str = "binary_label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Select only the 40 intended input features and one target column.

    The raw attack name, difficulty metadata, and any other target column are
    never included in X. Both binary and five-class targets are supported.
    """
    if target not in {"binary_label", "class_label"}:
        raise ValueError(
            "target must be 'binary_label' or 'class_label'; "
            "raw attack labels and difficulty metadata are not targets."
        )
    missing_features = sorted(set(MODEL_FEATURE_COLUMNS) - set(frame.columns))
    if missing_features:
        raise ValueError(
            "Input frame is missing required feature columns: "
            + ", ".join(missing_features)
        )
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is missing from the input frame.")

    X = frame.loc[:, MODEL_FEATURE_COLUMNS].copy()
    y = frame.loc[:, target].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create an unfitted transformer for numeric and categorical features.

    Numeric features are standardized. The three categorical features are
    one-hot encoded, with unseen test-time categories safely ignored. Calling
    this function does not fit or learn any statistics.
    """
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator: Any,
    *,
    balance_classes: bool = False,
) -> Pipeline:
    """Clone an estimator and combine it with the unfitted preprocessor.

    If requested, set class_weight='balanced' for estimators that support it.
    Scikit-learn derives weights from labels passed to each fit call, so fitting
    this pipeline on training folds avoids using test labels. Estimators without
    a class_weight parameter must use balance_classes=False and be handled
    explicitly during model selection.

    Since the scaler and encoder live inside the pipeline, cross-validation can
    refit them within each fold rather than leaking statistics across folds.
    """
    model = clone(estimator)
    if balance_classes:
        parameters = model.get_params(deep=False)
        if "class_weight" not in parameters:
            raise ValueError(
                f"{model.__class__.__name__} does not support class_weight. "
                "Use balance_classes=False for this estimator."
            )
        model.set_params(class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
