"""Exploratory data analysis for the official NSL-KDD splits.

Run from the repository root:
    python -m src.eda

The script reads only the real local KDDTrain+.txt and KDDTest+.txt files.
It does not synthesize data, fit transformations, or train a model. Figures are
written to reports/eda/ (ignored by Git because they are generated artifacts).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Generate files without requiring an interactive window.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing.nsl_kdd import FEATURE_COLUMNS, load_official_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"
CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]

def _save_plot(fig: plt.Figure, filename: str) -> None:
    """Save a consistently styled plot and release its memory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / filename}")


def _combined(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Tag rows with their official split for side-by-side descriptive plots."""
    train_view = train.copy()
    test_view = test.copy()
    train_view["split"] = "Train (KDDTrain+)"
    test_view["split"] = "Test (KDDTest+)"
    return pd.concat([train_view, test_view], ignore_index=True)


def plot_class_counts(data: pd.DataFrame) -> None:
    """Plot five-class counts by official split; raw counts show support."""
    counts = (
        data.groupby(["split", "class_label"], observed=False)
        .size().rename("records").reset_index()
    )
    counts["class_label"] = counts["class_label"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=counts, x="class_label", y="records", hue="split",
                order=CLASS_ORDER, ax=ax)
    ax.set(title="NSL-KDD class counts by official split",
           xlabel="Mapped class", ylabel="Number of records")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save_plot(fig, "class_counts_by_split.png")


def plot_class_proportions(data: pd.DataFrame) -> None:
    """Plot percentages so train/test composition can be compared fairly."""
    counts = (
        data.groupby(["split", "class_label"], observed=False)
        .size().rename("records").reset_index()
    )
    counts["class_label"] = counts["class_label"].astype(str)
    counts["percent"] = counts.groupby("split")["records"].transform(
        lambda values: values / values.sum() * 100
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=counts, x="class_label", y="percent", hue="split",
                order=CLASS_ORDER, ax=ax)
    ax.set(title="Class composition by official split",
           xlabel="Mapped class", ylabel="Records within split (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save_plot(fig, "class_proportions_by_split.png")


def plot_binary_counts(data: pd.DataFrame) -> None:
    """Show the binary Normal/Attack counts for both splits."""
    counts = data.groupby(["split", "binary_label"]).size().rename(
        "records").reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="binary_label", y="records", hue="split",
                order=["Normal", "Attack"], ax=ax)
    ax.set(title="Binary target counts by official split",
           xlabel="Binary target", ylabel="Number of records")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save_plot(fig, "binary_counts_by_split.png")


def plot_protocol_mix(data: pd.DataFrame) -> None:
    """Compare protocol counts across splits without fitting any encoder."""
    counts = data.groupby(["split", "protocol_type"]).size().rename(
        "records").reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="protocol_type", y="records", hue="split", ax=ax)
    ax.set(title="Protocol-type counts by official split",
           xlabel="Protocol type", ylabel="Number of records")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save_plot(fig, "protocol_counts_by_split.png")


def plot_numeric_distributions(data: pd.DataFrame) -> None:
    """Plot selected skewed numeric features on log1p scale for readability."""
    columns = ["duration", "src_bytes", "dst_bytes"]
    missing = [column for column in columns if column not in FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"Expected numeric EDA columns are absent: {missing}")

    fig, axes = plt.subplots(1, len(columns), figsize=(15, 4.8))
    for ax, column in zip(axes, columns):
        values = data[[column, "split", "class_label"]].copy()
        # Values are non-negative in the NSL-KDD schema; log1p retains zeros.
        values["log_value"] = np.log1p(
            pd.to_numeric(values[column], errors="raise").clip(lower=0)
        )
        sns.histplot(data=values, x="log_value", hue="split", bins=45,
                     element="step", stat="density", common_norm=False,
                     ax=ax)
        ax.set(title=column, xlabel=f"log(1 + {column})", ylabel="Density")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Selected numeric feature distributions (log1p scale)", y=1.04)
    _save_plot(fig, "numeric_feature_distributions.png")


def print_observations(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print reproducible count/proportion evidence for the EDA notes."""
    print("\nEDA DATA CHECK")
    print("==============")
    for name, frame in (("Train", train), ("Test", test)):
        print(f"\n{name}: {len(frame):,} rows")
        counts = frame["class_label"].value_counts().reindex(
            CLASS_ORDER, fill_value=0
        )
        proportions = (counts / len(frame) * 100).round(2)
        print(pd.DataFrame({"count": counts, "percent": proportions}).to_string())
        print("\nProtocol type counts:")
        print(frame["protocol_type"].value_counts().to_string())

    train_pct = train["class_label"].value_counts(normalize=True).reindex(
        CLASS_ORDER, fill_value=0
    ) * 100
    test_pct = test["class_label"].value_counts(normalize=True).reindex(
        CLASS_ORDER, fill_value=0
    ) * 100
    shift = (test_pct - train_pct).sort_values(key=lambda series: series.abs(),
                                               ascending=False)
    print("\nLargest absolute class-proportion differences (test minus train, pp):")
    print(shift.round(2).to_string())
    print("\nInterpretation reminders:")
    print("- R2L and U2R have much less training support than Normal and DoS.")
    print("- Compare proportions as well as counts because the official splits differ in size.")
    print("- A large train/test class shift can affect generalization; do not rebalance the test set.")
    print("- EDA is descriptive only. No preprocessing is fitted and no model is trained.")


def main() -> None:
    """Load official data, print observations, and write five EDA figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    train, test = load_official_split(PROJECT_ROOT / "data" / "raw")
    combined = _combined(train, test)

    print_observations(train, test)
    plot_class_counts(combined)
    plot_class_proportions(combined)
    plot_binary_counts(combined)
    plot_protocol_mix(combined)
    plot_numeric_distributions(combined)
    print(f"\nEDA complete. Figures saved under: {OUTPUT_DIR}")
    print("No models were trained; official train/test split was preserved.")


if __name__ == "__main__":
    main()
