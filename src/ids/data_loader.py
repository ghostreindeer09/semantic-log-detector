"""
Data loading, validation, cleaning, and splitting for CIC-IDS2017.

Handles:
- Multi-file CSV loading
- Schema validation
- Feature name normalization
- Label mapping (binary + multi-class)
- Duplicate removal
- Stratified train/val/test split
- Class distribution logging
"""

import os
import re
import glob
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Label mapping helpers                                              #
# ------------------------------------------------------------------ #

MULTICLASS_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS",
    "DoS Hulk": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "Heartbleed": "DoS",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Infiltration": "Infiltration",
}

# Web‐attack labels in CIC-IDS2017 have encoding artifacts 
WEB_ATTACK_PATTERN = re.compile(r"web\s*attack", re.IGNORECASE)


def _map_multiclass(label: str) -> str:
    """Map raw label string to consolidated multi-class label."""
    label = label.strip()
    if label in MULTICLASS_MAP:
        return MULTICLASS_MAP[label]
    if WEB_ATTACK_PATTERN.search(label):
        return "Web Attack"
    return "Others"


def _map_binary(label: str) -> int:
    """Map raw label to binary (0=BENIGN, 1=ATTACK)."""
    return 0 if label.strip() == "BENIGN" else 1


# ------------------------------------------------------------------ #
#  Core data loading                                                  #
# ------------------------------------------------------------------ #

def discover_csv_files(dataset_dir: str) -> List[str]:
    """Find all CIC-IDS2017 CSV files in the given directory."""
    patterns = [
        os.path.join(dataset_dir, "*.csv"),
        os.path.join(dataset_dir, "**", "*.csv"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    logger.info("Discovered %d CSV file(s) in %s", len(files), dataset_dir)
    for f in files:
        logger.info("  -> %s", os.path.basename(f))
    return files


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalize column names."""
    df.columns = [c.strip() for c in df.columns]
    # Normalize to snake_case for consistency
    rename_map = {}
    for col in df.columns:
        normalized = col.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        rename_map[col] = normalized
    # Keep 'Label' explicitly as 'label'
    df = df.rename(columns=rename_map)
    return df


def _validate_schema(dfs: List[pd.DataFrame], file_names: List[str]) -> None:
    """Ensure all dataframes share the same schema."""
    ref_cols = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], 1):
        curr_cols = set(df.columns)
        if curr_cols != ref_cols:
            missing = ref_cols - curr_cols
            extra = curr_cols - ref_cols
            logger.warning(
                "Schema mismatch in %s: missing=%s, extra=%s",
                file_names[i], missing, extra
            )
            # Use intersection to proceed
            common = ref_cols & curr_cols
            logger.warning("Using %d common columns", len(common))


def load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and concatenate all CIC-IDS2017 CSV files.

    Returns a single DataFrame with normalized column names.
    """
    dataset_dir = config["paths"]["raw_dataset_dir"]
    encoding = config["data"].get("encoding", "latin-1")

    csv_files = discover_csv_files(dataset_dir)
    dfs = []
    file_names = []

    for fpath in csv_files:
        logger.info("Loading %s ...", os.path.basename(fpath))
        df = pd.read_csv(fpath, encoding=encoding, low_memory=False)
        df = _normalize_columns(df)
        # Track source file for chronological splitting
        df["source_file"] = os.path.basename(fpath)
        dfs.append(df)
        file_names.append(os.path.basename(fpath))
        logger.info("  -> %d rows, %d cols", len(df), len(df.columns))

    _validate_schema(dfs, file_names)

    # Concatenate using common columns (source_file is always present)
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)

    combined = pd.concat([df[common_cols] for df in dfs], ignore_index=True)
    logger.info("Combined dataset: %d rows, %d columns", len(combined), len(combined.columns))

    # Optional subsample for debugging
    max_samples = config["data"].get("max_samples")
    if max_samples and max_samples < len(combined):
        combined = combined.sample(n=max_samples, random_state=config["data"]["random_seed"])
        logger.info("Subsampled to %d rows (debug mode)", max_samples)

    return combined


# ------------------------------------------------------------------ #
#  Cleaning & Feature Engineering                                     #
# ------------------------------------------------------------------ #

def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Clean the raw dataset: handle missing/inf, remove duplicates."""
    initial_rows = len(df)

    # Replace inf with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Report missing values
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        logger.info("Columns with missing values:")
        for col, count in cols_with_missing.items():
            logger.info("  %s: %d (%.2f%%)", col, count, 100 * count / len(df))

    # Fill missing numeric with median
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Remove duplicates
    if config["data"].get("remove_duplicates", True):
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            logger.info("Removed %d duplicate rows (%.2f%%)", removed, 100 * removed / before)

    logger.info("Cleaning: %d -> %d rows", initial_rows, len(df))
    return df.reset_index(drop=True)


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary and multi-class label columns from the raw label."""
    label_col = "label"
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")

    df["label_binary"] = df[label_col].apply(_map_binary)
    df["label_multiclass"] = df[label_col].apply(_map_multiclass)

    # Encode multi-class as integers
    unique_mc = sorted(df["label_multiclass"].unique())
    mc_encoding = {name: idx for idx, name in enumerate(unique_mc)}
    df["label_multiclass_encoded"] = df["label_multiclass"].map(mc_encoding)

    logger.info("Binary labels: %s", dict(df["label_binary"].value_counts()))
    logger.info("Multi-class labels (%d classes):", len(unique_mc))
    for name in unique_mc:
        count = (df["label_multiclass"] == name).sum()
        pct = 100 * count / len(df)
        logger.info("  %s: %d (%.2f%%)", name, count, pct)
    logger.info("Multi-class encoding: %s", mc_encoding)

    return df, mc_encoding


def identify_features(df: pd.DataFrame) -> List[str]:
    """Identify structured flow feature columns (exclude labels, metadata, and non-numeric)."""
    exclude_cols = {
        "label", "label_binary", "label_multiclass", "label_multiclass_encoded",
        # Metadata columns preserved for chronological splitting / provenance
        "capture_day", "day_index", "source_file", "timestamp",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    logger.info("Identified %d numeric feature columns", len(feature_cols))
    return feature_cols


# ------------------------------------------------------------------ #
#  Stratified Split                                                    #
# ------------------------------------------------------------------ #

def stratified_split(
    df: pd.DataFrame,
    config: Dict[str, Any],
    stratify_col: str = "label_binary"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified 70/15/15 train/val/test split."""
    seed = config["data"]["random_seed"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed,
        stratify=df[stratify_col]
    )

    # Second split: train vs val
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=seed,
        stratify=train_val[stratify_col]
    )

    logger.info("Split sizes: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        dist = split_df[stratify_col].value_counts().to_dict()
        logger.info("  %s distribution: %s", split_name, dist)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ------------------------------------------------------------------ #
#  Save / Load processed data                                         #
# ------------------------------------------------------------------ #

def save_processed_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    mc_encoding: Dict[str, int],
    config: Dict[str, Any],
) -> None:
    """Save processed splits and metadata to disk."""
    out_dir = config["paths"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

    metadata = {
        "feature_columns": feature_cols,
        "multiclass_encoding": mc_encoding,
        "num_features": len(feature_cols),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "total_rows": len(train) + len(val) + len(test),
        "binary_distribution": {
            "train": train["label_binary"].value_counts().to_dict(),
            "val": val["label_binary"].value_counts().to_dict(),
            "test": test["label_binary"].value_counts().to_dict(),
        },
        "multiclass_distribution": {
            "train": train["label_multiclass"].value_counts().to_dict(),
            "val": val["label_multiclass"].value_counts().to_dict(),
            "test": test["label_multiclass"].value_counts().to_dict(),
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Processed data saved to %s", out_dir)


def load_processed_splits(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load previously processed splits and metadata."""
    proc_dir = config["paths"]["processed_dir"]

    train = pd.read_parquet(os.path.join(proc_dir, "train.parquet"))
    val = pd.read_parquet(os.path.join(proc_dir, "val.parquet"))
    test = pd.read_parquet(os.path.join(proc_dir, "test.parquet"))

    with open(os.path.join(proc_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    logger.info(
        "Loaded processed splits: train=%d, val=%d, test=%d",
        len(train), len(val), len(test)
    )
    return train, val, test, metadata


# ------------------------------------------------------------------ #
#  Master pipeline entry point                                        #
# ------------------------------------------------------------------ #

def run_data_pipeline(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict]:
    """
    Execute full data loading & processing pipeline.

    Returns (train, val, test, feature_cols, mc_encoding).
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA LOADING & VALIDATION")
    logger.info("=" * 60)

    # Check for cached processed data
    proc_dir = config["paths"]["processed_dir"]
    if os.path.exists(os.path.join(proc_dir, "train.parquet")):
        logger.info("Found pre-processed data in %s. Loading...", proc_dir)
        train, val, test, metadata = load_processed_splits(config)
        feature_cols = metadata["feature_columns"]
        mc_encoding = metadata["multiclass_encoding"]
        return train, val, test, feature_cols, mc_encoding

    # Load raw data
    df = load_raw_data(config)

    # Clean
    df = clean_data(df, config)

    # Create labels
    df, mc_encoding = create_labels(df)

    # Identify features
    feature_cols = identify_features(df)

    # Split
    train, val, test = stratified_split(df, config, stratify_col="label_binary")

    # Save
    save_processed_splits(train, val, test, feature_cols, mc_encoding, config)

    return train, val, test, feature_cols, mc_encoding
