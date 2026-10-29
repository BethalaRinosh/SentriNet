"""Load and label the official NSL-KDD train/test files.

The loader preserves the official split. It does not download, generate, or
randomly split data. It validates the raw schema and maps target labels;
feature transformations are implemented separately in pipeline.py.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# NSL-KDD contains 41 connection features, an attack label, and a difficulty
# score. The difficulty score is metadata, not a model input.
FEATURE_COLUMNS: list[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
LABEL_COLUMN = "attack"
DIFFICULTY_COLUMN = "difficulty"
ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN, DIFFICULTY_COLUMN]

# Standard NSL-KDD attack-to-family mapping. Unknown labels are rejected
# rather than silently assigned to a guessed class.
ATTACK_TO_CATEGORY: dict[str, str] = {
    # Denial of Service (DoS)
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe / surveillance
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote-to-local (R2L)
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L",
    "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L",
    # User-to-root (U2R)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}
MULTICLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


class DatasetValidationError(ValueError):
    """Raised when an NSL-KDD file is absent or does not match the schema."""


def _clean_attack_labels(labels: pd.Series) -> pd.Series:
    """Normalize labels, including the trailing period used in some test labels."""
    cleaned = labels.astype("string").str.strip().str.lower().str.removesuffix(".")
    if cleaned.isna().any() or cleaned.eq("").any():
        raise DatasetValidationError("Attack labels contain empty or missing values.")
    return cleaned


def load_nsl_kdd_file(path: str | Path) -> pd.DataFrame:
    """Read one NSL-KDD file with the canonical 43-column schema.

    Parameters
    ----------
    path:
        Path to KDDTrain+.txt or KDDTest+.txt.

    Returns
    -------
    pandas.DataFrame
        41 features, normalized raw attack label, and difficulty metadata.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"NSL-KDD file not found: {file_path}. "
            "Place KDDTrain+.txt and KDDTest+.txt in data/raw/."
        )
    if file_path.stat().st_size == 0:
        raise DatasetValidationError(f"NSL-KDD file is empty: {file_path}")

    frame = pd.read_csv(file_path, header=None, names=ALL_COLUMNS, sep=",")
    if frame.empty:
        raise DatasetValidationError(f"No records were read from {file_path}.")
    if frame.shape[1] != 43:
        raise DatasetValidationError(
            f"Expected 43 columns (41 features + attack + difficulty), "
            f"but found {frame.shape[1]} in {file_path}."
        )
    frame[LABEL_COLUMN] = _clean_attack_labels(frame[LABEL_COLUMN])
    unknown = sorted(set(frame[LABEL_COLUMN]) - ({"normal"} | set(ATTACK_TO_CATEGORY)))
    if unknown:
        raise DatasetValidationError(
            "Unmapped attack labels found: " + ", ".join(unknown)
        )
    return frame


def add_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with binary and five-class targets added."""
    if LABEL_COLUMN not in frame.columns:
        raise DatasetValidationError(f"Required target column {LABEL_COLUMN!r} is missing.")
    result = frame.copy()
    raw_labels = _clean_attack_labels(result[LABEL_COLUMN])
    unknown = sorted(set(raw_labels) - ({"normal"} | set(ATTACK_TO_CATEGORY)))
    if unknown:
        raise DatasetValidationError(
            "Cannot map unknown attack labels: " + ", ".join(unknown)
        )
    result["binary_label"] = raw_labels.map(
        lambda label: "Normal" if label == "normal" else "Attack"
    )
    result["class_label"] = raw_labels.map(
        lambda label: "Normal" if label == "normal" else ATTACK_TO_CATEGORY[label]
    )
    result["class_label"] = pd.Categorical(
        result["class_label"], categories=MULTICLASS_ORDER, ordered=True
    )
    return result


def load_official_split(
    data_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load KDDTrain+ and KDDTest+ separately without re-splitting."""
    root = Path(data_dir)
    train = add_target_columns(load_nsl_kdd_file(root / "KDDTrain+.txt"))
    test = add_target_columns(load_nsl_kdd_file(root / "KDDTest+.txt"))
    return train, test


def dataset_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Build a compact, serializable inspection summary for one loaded split."""
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "dtypes": {str(key): str(value) for key, value in frame.dtypes.items()},
        "binary_distribution": {
            str(key): int(value)
            for key, value in frame["binary_label"].value_counts().items()
        },
        "multiclass_distribution": {
            str(key): int(value)
            for key, value in frame["class_label"].value_counts().items()
        },
    }
