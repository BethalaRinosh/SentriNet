"""Training-split-only feature importance for SentriNet.

Run from the repository root:
    python -m src.feature_selection
    python -m src.feature_selection --target class_label --sample-size 30000

The official KDDTest+ split is deliberately not loaded or used here. This is
exploratory impurity-based Random Forest importance, not proof of causality or
a substitute for held-out evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.nsl_kdd import add_target_columns, load_nsl_kdd_file
from src.preprocessing.pipeline import (
    MODEL_FEATURE_COLUMNS,
    build_preprocessor,
    select_features_and_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "feature_importance"


def aggregate_importances(
    transformed_names: list[str],
    importances: list[float],
    raw_features: list[str] = MODEL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Sum one-hot category importances back into their original feature groups."""
    if len(transformed_names) != len(importances):
        raise ValueError("Feature names and importance values must have equal length.")

    # Longest-first prevents a shorter raw feature name from matching a prefix.
    ordered_features = sorted(raw_features, key=len, reverse=True)
    totals = {feature: 0.0 for feature in raw_features}
    for name, importance in zip(transformed_names, importances):
        matched = next(
            (feature for feature in ordered_features
             if name == feature or name.startswith(feature + "_")),
            None,
        )
        if matched is None:
            raise ValueError(
                f"Cannot map transformed feature {name!r} to a raw input feature."
            )
        totals[matched] += float(importance)

    result = pd.DataFrame(
        {"feature": list(totals), "importance": list(totals.values())}
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def run_feature_importance(
    *,
    data_path: str | Path = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    target: str = "binary_label",
    sample_size: int = 30_000,
    n_estimators: int = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit a Random Forest on a reproducible sample of KDDTrain+ only."""
    if target not in {"binary_label", "class_label"}:
        raise ValueError("target must be 'binary_label' or 'class_label'.")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    # Load only the official training file. KDDTest+ is not touched by this script.
    frame = add_target_columns(load_nsl_kdd_file(data_path))
    if sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_state)

    X, y = select_features_and_target(frame, target=target)
    preprocessor = build_preprocessor()
    transformed_X = preprocessor.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(transformed_X, y)

    transformed_names = list(preprocessor.get_feature_names_out())
    ranked = aggregate_importances(
        transformed_names,
        list(model.feature_importances_),
    )
    ranked["target"] = target
    ranked["training_rows_used"] = len(frame)
    ranked["method"] = "Random Forest impurity importance; one-hot groups summed"
    ranked["random_state"] = random_state

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{target}_feature_importance.csv"
    plot_path = destination / f"{target}_top20_feature_importance.png"
    ranked.to_csv(csv_path, index=False)

    top = ranked.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set(
        title=f"Top 20 NSL-KDD features ({target})",
        xlabel="Aggregated Random Forest impurity importance",
        ylabel="Original input feature",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Target: {target}")
    print(f"Training rows sampled: {len(frame):,} of official KDDTrain+")
    print(f"Random state: {random_state}; trees: {n_estimators}")
    print("\nTop 20 original input features:")
    print(ranked.head(20)[["feature", "importance"]].to_string(index=False))
    print(f"Saved ranking: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print(
        "\nCaution: impurity importance is exploratory and can favor features "
        "with many possible split points. It is not causal evidence or a "
        "held-out performance metric."
    )
    print("Only KDDTrain+ was loaded; the official test split was not used.")
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["binary_label", "class_label"],
        default="binary_label", help="Prediction target to analyze.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=30_000,
        help="Maximum reproducible sample size from KDDTrain+ (default: 30000).",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=150,
        help="Number of Random Forest trees (default: 150).",
    )
    args = parser.parse_args()
    run_feature_importance(
        target=args.target,
        sample_size=args.sample_size,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
