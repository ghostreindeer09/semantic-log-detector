"""
Chronological splitter for CIC-IDS2017.

Implements day-based temporal splitting that avoids the temporal leakage
inherent in random stratified splits.  The split is determined by the
source CSV filename (which encodes the capture day), not by parsing
potentially unreliable timestamp strings.

Split strategy:
    Train:      Monday (Jul 3) + Tuesday (Jul 4) + Wednesday (Jul 5)
    Validation: Thursday (Jul 6)
    Test:       Friday (Jul 7)
"""

import logging
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Day mapping from CIC-IDS2017 filenames                              #
# ------------------------------------------------------------------ #

# Each file pattern maps to a (day_name, day_index) tuple.
# day_index encodes chronological order (0=Monday .. 4=Friday).
_FILE_DAY_PATTERNS = [
    (re.compile(r"Monday",    re.IGNORECASE), "Monday",       0),
    (re.compile(r"Tuesday",   re.IGNORECASE), "Tuesday",      1),
    (re.compile(r"Wednesday", re.IGNORECASE), "Wednesday",    2),
    (re.compile(r"Thursday",  re.IGNORECASE), "Thursday",     3),
    (re.compile(r"Friday",    re.IGNORECASE), "Friday",       4),
]

# Default split assignment
SPLIT_ASSIGNMENT = {
    0: "train",       # Monday   — benign baseline
    1: "train",       # Tuesday  — Brute Force
    2: "train",       # Wednesday — DoS
    3: "validation",  # Thursday  — Web Attack + Infiltration
    4: "test",        # Friday    — Bot + PortScan + DDoS
}


def infer_day_from_filename(filename: str) -> Tuple[str, int]:
    """
    Infer the capture day from a CIC-IDS2017 CSV filename.

    Returns (day_name, day_index) or raises ValueError if unrecognised.
    """
    basename = os.path.basename(filename)
    for pattern, day_name, day_idx in _FILE_DAY_PATTERNS:
        if pattern.search(basename):
            return day_name, day_idx
    raise ValueError(
        f"Cannot determine capture day from filename: {basename}. "
        f"Expected a CIC-IDS2017 filename containing a day name."
    )


def load_with_day_labels(
    csv_files: List[str],
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """
    Load CIC-IDS2017 CSVs and add ``capture_day`` and ``day_index`` columns.

    This replaces the standard ``load_raw_data()`` when chronological
    splitting is needed.  The returned DataFrame retains all original
    columns (including ``Timestamp``) plus the two new columns.
    """
    frames = []
    for fpath in csv_files:
        day_name, day_idx = infer_day_from_filename(fpath)
        logger.info("Loading %s → %s (day %d)", os.path.basename(fpath), day_name, day_idx)

        df = pd.read_csv(fpath, encoding=encoding, low_memory=False)
        # Normalize column names (same logic as data_loader._normalize_columns)
        df.columns = [
            c.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            for c in df.columns
        ]
        df["capture_day"] = day_name
        df["day_index"] = day_idx
        df["source_file"] = os.path.basename(fpath)
        frames.append(df)
        logger.info("  → %d rows", len(df))

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined: %d rows across %d files, days: %s",
        len(combined),
        len(csv_files),
        sorted(combined["capture_day"].unique()),
    )
    return combined


def chronological_split(
    df: pd.DataFrame,
    split_assignment: Dict[int, str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame with ``day_index`` column into train/val/test
    based on chronological day assignment.

    Args:
        df: DataFrame with ``day_index`` column (0–4).
        split_assignment: Mapping from day_index → "train"/"validation"/"test".
            Defaults to the standard Mon–Wed/Thu/Fri split.

    Returns:
        (train_df, val_df, test_df) with original indices reset.
    """
    if split_assignment is None:
        split_assignment = SPLIT_ASSIGNMENT

    if "day_index" not in df.columns:
        raise ValueError(
            "DataFrame must have a 'day_index' column. "
            "Use load_with_day_labels() to load the data."
        )

    train_mask = df["day_index"].map(split_assignment) == "train"
    val_mask = df["day_index"].map(split_assignment) == "validation"
    test_mask = df["day_index"].map(split_assignment) == "test"

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    logger.info("Chronological split:")
    logger.info("  Train: %d rows (days: %s)", len(train_df),
                sorted(train_df["capture_day"].unique()))
    logger.info("  Val:   %d rows (days: %s)", len(val_df),
                sorted(val_df["capture_day"].unique()))
    logger.info("  Test:  %d rows (days: %s)", len(test_df),
                sorted(test_df["capture_day"].unique()))

    # Log class distributions
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if "label" in split_df.columns:
            dist = split_df["label"].value_counts().head(10).to_dict()
            logger.info("  %s label distribution: %s", name, dist)

    return train_df, val_df, test_df


def log_split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str = "label_multiclass",
) -> Dict:
    """
    Generate a summary of the chronological split for reporting.

    Returns a dict suitable for JSON serialization.
    """
    summary = {"strategy": "chronological_day_split"}

    for name, df in [("train", train), ("validation", val), ("test", test)]:
        split_info = {
            "rows": len(df),
            "days": sorted(df["capture_day"].unique().tolist())
                    if "capture_day" in df.columns else [],
        }
        if label_col in df.columns:
            split_info["class_distribution"] = (
                df[label_col].value_counts().to_dict()
            )
        if "label_binary" in df.columns:
            binary = df["label_binary"].value_counts().to_dict()
            split_info["binary_distribution"] = {
                "benign": binary.get(0, 0),
                "attack": binary.get(1, 0),
            }
        summary[name] = split_info

    return summary
