"""
Full chronological validation execution script.

Runs the complete pipeline:
1. Load all 8 CIC-IDS2017 CSVs with day labels
2. Clean + create labels
3. Generate chronological split (Mon-Wed / Thu / Fri)
4. Train fresh LightGBM (identical hyperparams)
5. Evaluate binary + multiclass
6. Threshold calibration (Default / Youden-J / Max-F1)
7. Compare against random split baseline
8. Output all JSON + report files

Usage:
    python scripts/run_chronological_validation.py
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ids.config_loader import load_config
from src.ids.data_loader import clean_data, create_labels, identify_features, stratified_split
from src.ids.chronological_splitter import (
    load_with_day_labels,
    chronological_split,
    log_split_summary,
    infer_day_from_filename,
)
from src.ids.structured_trainer import StructuredTrainer
from src.ids.evaluator import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    run_full_evaluation,
)
from src.ids.threshold_calibrator import calibrate_threshold


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        force=True,
    )
    logger = logging.getLogger("chrono_validation")

    config = load_config("config/ids_config.yaml")

    DATASET_DIR = "Datasets"
    OUTPUT_DIR = "outputs/chronological_eval"
    METRICS_DIR = "outputs/metrics"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # =================================================================
    # STEP 1: Discover and load all CSVs with day labels
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 1: LOADING CIC-IDS2017 DATA WITH DAY LABELS")
    logger.info("=" * 70)

    import glob
    csv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
    logger.info("Found %d CSV files:", len(csv_files))
    for f in csv_files:
        day_name, day_idx = infer_day_from_filename(f)
        logger.info("  %s → %s (day %d)", os.path.basename(f), day_name, day_idx)

    raw_df = load_with_day_labels(csv_files, encoding="latin-1")
    logger.info("Combined shape: %s", raw_df.shape)
    logger.info("Columns: %s", raw_df.columns.tolist()[:10])

    # =================================================================
    # STEP 2: Verify timestamps and source-day metadata
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 2: VERIFYING METADATA")
    logger.info("=" * 70)

    logger.info("capture_day values: %s", raw_df["capture_day"].value_counts().to_dict())
    logger.info("day_index values: %s", raw_df["day_index"].value_counts().to_dict())
    logger.info("source_file values: %s", raw_df["source_file"].nunique())

    # Check for timestamp column
    ts_candidates = [c for c in raw_df.columns if "timestamp" in c.lower() or "time" in c.lower()]
    logger.info("Timestamp-like columns: %s", ts_candidates)
    if ts_candidates:
        ts_col = ts_candidates[0]
        logger.info("Sample timestamps: %s", raw_df[ts_col].head(3).tolist())

    # =================================================================
    # STEP 3: Clean + create labels
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 3: CLEANING AND LABEL CREATION")
    logger.info("=" * 70)

    raw_df = clean_data(raw_df, config)
    raw_df, mc_encoding = create_labels(raw_df)

    feature_cols = [
        c for c in identify_features(raw_df)
        if c not in {"capture_day", "day_index", "source_file", "timestamp"}
    ]
    logger.info("Feature columns: %d", len(feature_cols))

    # =================================================================
    # STEP 4: Generate chronological split
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 4: CHRONOLOGICAL SPLIT (Mon-Wed / Thu / Fri)")
    logger.info("=" * 70)

    train_c, val_c, test_c = chronological_split(raw_df)

    split_summary = log_split_summary(train_c, val_c, test_c)
    with open(os.path.join(OUTPUT_DIR, "split_summary.json"), "w") as f:
        json.dump(_make_serializable(split_summary), f, indent=2)

    # Log class distributions per split
    for name, df in [("TRAIN", train_c), ("VAL", val_c), ("TEST", test_c)]:
        logger.info("%s: %d rows", name, len(df))
        logger.info("  Days: %s", sorted(df["capture_day"].unique()))
        logger.info("  Binary dist: %s", df["label_binary"].value_counts().to_dict())
        logger.info("  Multiclass dist: %s", df["label_multiclass"].value_counts().to_dict())

    # =================================================================
    # STEP 5: Train fresh LightGBM on chronological training split
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 5: TRAINING LIGHTGBM (CHRONOLOGICAL SPLIT)")
    logger.info("=" * 70)

    # Force binary task first (matches original experiment)
    config["training"]["task"] = "binary"
    trainer_bin = StructuredTrainer(config)
    train_log_bin = trainer_bin.train(train_c, val_c, feature_cols)
    logger.info("Binary training complete: %s", train_log_bin)

    # Save checkpoint
    chrono_ckpt = os.path.join(OUTPUT_DIR, "checkpoints")
    trainer_bin.save(chrono_ckpt)

    # =================================================================
    # STEP 6: Binary evaluation + threshold calibration
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 6: BINARY EVALUATION + THRESHOLD CALIBRATION")
    logger.info("=" * 70)

    y_true_bin = test_c["label_binary"].values
    y_pred_bin = trainer_bin.predict(test_c)
    y_proba_bin = trainer_bin.predict_proba(test_c)

    binary_metrics = compute_binary_metrics(y_true_bin, y_pred_bin, y_proba_bin)
    logger.info("Binary metrics computed.")

    # Threshold calibration
    threshold_results = calibrate_threshold(
        y_true=y_true_bin,
        y_proba=y_proba_bin,
        config=config,
        output_dir=os.path.join(OUTPUT_DIR, "threshold"),
    )
    logger.info("Threshold calibration complete.")

    # Plots
    binary_eval = run_full_evaluation(
        y_true=y_true_bin,
        y_pred=y_pred_bin,
        y_proba=y_proba_bin,
        task="binary",
        output_dir=OUTPUT_DIR,
        prefix="chrono_binary",
        class_names=["BENIGN", "ATTACK"],
    )

    # =================================================================
    # STEP 7: Multiclass evaluation
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 7: MULTICLASS EVALUATION")
    logger.info("=" * 70)

    # ── Key challenge: chronological split produces disjoint classes ──
    # Train has: {BENIGN, Brute Force, DoS} (encoded 0, 2, 4)
    # Val has:   {BENIGN, Web Attack, Infiltration} (encoded 0, 5, 7)
    # Test has:  {BENIGN, Bot, DDoS, PortScan} (encoded 0, 1, 3, 6)
    #
    # LightGBM cannot validate on classes not in training.
    # Solution: filter validation set to training-only classes for early stopping.
    # Then evaluate on test set — the model will only predict training classes,
    # which is exactly the behavior we want to measure.

    train_classes = set(train_c["label_multiclass_encoded"].unique())
    logger.info("Training multiclass classes: %s", train_classes)

    # Filter val to only include classes present in training
    val_filtered = val_c[val_c["label_multiclass_encoded"].isin(train_classes)].reset_index(drop=True)
    logger.info("Val filtered: %d → %d rows (kept only training classes)", len(val_c), len(val_filtered))

    config["training"]["task"] = "multiclass"
    trainer_mc = StructuredTrainer(config)
    train_log_mc = trainer_mc.train(train_c, val_filtered, feature_cols)
    logger.info("Multiclass training complete.")

    # ── Evaluate on full test set ──
    # The model can only predict {BENIGN=0, Brute Force=2, DoS=4}.
    # Test set has {BENIGN=0, Bot=1, DDoS=3, PortScan=6}.
    # Bot, DDoS, PortScan will be misclassified into one of the 3 known classes.
    # This is the CORRECT chronological evaluation behavior.

    class_names = sorted(mc_encoding.keys(), key=lambda k: mc_encoding[k])
    train_class_names = sorted(
        [k for k, v in mc_encoding.items() if v in train_classes],
        key=lambda k: mc_encoding[k],
    )
    logger.info("Model knows classes: %s", train_class_names)
    logger.info("Test set has classes: %s",
                sorted(test_c["label_multiclass"].unique()))

    y_true_mc = test_c["label_multiclass_encoded"].values
    y_pred_mc = trainer_mc.predict(test_c)

    try:
        y_proba_mc = trainer_mc.predict_proba(test_c)
    except Exception as e:
        logger.warning("Could not get multiclass probabilities: %s", e)
        y_proba_mc = None

    # Build per-class detection rates manually (since classes are disjoint)
    per_class_detection = {}
    for cls_name, cls_idx in mc_encoding.items():
        mask = y_true_mc == cls_idx
        total = int(mask.sum())
        if total == 0:
            continue
        detected = int((y_pred_mc[mask] == cls_idx).sum())
        per_class_detection[cls_name] = {
            "total": total,
            "detected": detected,
            "detection_rate": round(detected / total, 6) if total > 0 else 0.0,
            "in_training": cls_idx in train_classes,
        }
        logger.info("  %s: %d/%d detected (%.2f%%) [trained=%s]",
                     cls_name, detected, total, 100 * detected / total,
                     cls_idx in train_classes)

    # Compute overall multiclass accuracy
    mc_accuracy = float((y_pred_mc == y_true_mc).mean())
    logger.info("Multiclass accuracy (all classes): %.4f", mc_accuracy)

    # Build mc_metrics manually since standard compute_multiclass_metrics
    # may not handle the class mismatch well
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score as sk_f1,
        precision_score as sk_prec, recall_score as sk_rec,
    )

    # Use all classes that appear in either true or predicted
    all_labels = sorted(set(y_true_mc) | set(y_pred_mc))
    all_label_names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in all_labels]

    mc_metrics = {
        "accuracy": mc_accuracy,
        "macro_f1": float(sk_f1(y_true_mc, y_pred_mc, labels=all_labels, average="macro", zero_division=0)),
        "weighted_f1": float(sk_f1(y_true_mc, y_pred_mc, labels=all_labels, average="weighted", zero_division=0)),
        "macro_precision": float(sk_prec(y_true_mc, y_pred_mc, labels=all_labels, average="macro", zero_division=0)),
        "macro_recall": float(sk_rec(y_true_mc, y_pred_mc, labels=all_labels, average="macro", zero_division=0)),
        "weighted_precision": float(sk_prec(y_true_mc, y_pred_mc, labels=all_labels, average="weighted", zero_division=0)),
        "weighted_recall": float(sk_rec(y_true_mc, y_pred_mc, labels=all_labels, average="weighted", zero_division=0)),
        "per_class_detection": per_class_detection,
        "confusion_matrix": confusion_matrix(y_true_mc, y_pred_mc, labels=all_labels).tolist(),
        "class_names": all_label_names,
        "note": "Model trained on {BENIGN, Brute Force, DoS}. Test contains {BENIGN, Bot, DDoS, PortScan}. "
                "Unseen attack classes are expected to be misclassified — this measures true generalization.",
    }

    # Generate classification report for logging
    report = classification_report(
        y_true_mc, y_pred_mc, labels=all_labels,
        target_names=all_label_names, zero_division=0,
    )
    logger.info("Classification Report:\n%s", report)

    # Save confusion matrix plot
    try:
        mc_eval = run_full_evaluation(
            y_true=y_true_mc,
            y_pred=y_pred_mc,
            y_proba=None,  # Skip probability-based metrics to avoid shape mismatch
            task="multiclass",
            output_dir=OUTPUT_DIR,
            prefix="chrono_multiclass",
            class_names=all_label_names,
        )
    except Exception as e:
        logger.warning("Multiclass plot generation failed (expected with class mismatch): %s", e)

    # =================================================================
    # STEP 8: Assemble chronological_metrics.json
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 8: ASSEMBLING FINAL METRICS")
    logger.info("=" * 70)

    chrono_metrics = {
        "split_strategy": "chronological_day_split",
        "split_assignment": {
            "train": ["Monday", "Tuesday", "Wednesday"],
            "validation": ["Thursday"],
            "test": ["Friday"],
        },
        "split_sizes": {
            "train": len(train_c),
            "validation": len(val_c),
            "test": len(test_c),
        },
        "binary": binary_metrics,
        "multiclass": mc_metrics,
        "threshold_calibration": threshold_results,
        "training_log": {
            "binary": train_log_bin,
            "multiclass": train_log_mc,
        },
    }

    chrono_path = os.path.join(METRICS_DIR, "chronological_metrics.json")
    with open(chrono_path, "w") as f:
        json.dump(_make_serializable(chrono_metrics), f, indent=2)
    logger.info("Chronological metrics saved to %s", chrono_path)

    # Also save to chronological_eval dir
    with open(os.path.join(OUTPUT_DIR, "chronological_metrics.json"), "w") as f:
        json.dump(_make_serializable(chrono_metrics), f, indent=2)

    # =================================================================
    # STEP 9: Build comparison with random split baseline
    # =================================================================
    logger.info("=" * 70)
    logger.info("  STEP 9: COMPARISON WITH RANDOM SPLIT")
    logger.info("=" * 70)

    random_path = os.path.join(METRICS_DIR, "random_split_metrics.json")
    if os.path.exists(random_path):
        with open(random_path, "r") as f:
            random_metrics = json.load(f)
    else:
        # Fall back to individual files
        random_metrics = {
            "binary": json.load(open(os.path.join(METRICS_DIR, "structured_binary_metrics.json"))),
            "multiclass": json.load(open(os.path.join(METRICS_DIR, "structured_multiclass_metrics.json"))),
            "threshold_calibration": json.load(open(os.path.join(METRICS_DIR, "threshold.json"))),
        }

    comparison = _build_comparison(random_metrics, chrono_metrics)
    comp_path = os.path.join(OUTPUT_DIR, "evaluation_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(_make_serializable(comparison), f, indent=2)
    logger.info("Comparison saved to %s", comp_path)

    # Print summary
    _print_summary(random_metrics, chrono_metrics, logger)

    logger.info("=" * 70)
    logger.info("  CHRONOLOGICAL VALIDATION COMPLETE")
    logger.info("  Metrics:    %s", chrono_path)
    logger.info("  Comparison: %s", comp_path)
    logger.info("  Plots:      %s", OUTPUT_DIR)
    logger.info("=" * 70)


def _build_comparison(random_metrics, chrono_metrics):
    """Build structured comparison dict."""
    comp = {"strategy_a": "random_split", "strategy_b": "chronological"}

    rb = random_metrics.get("binary", {})
    cb = chrono_metrics.get("binary", {})

    binary_comp = {}
    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc",
              "false_positive_rate", "false_negative_rate", "detection_rate"]:
        rv = rb.get(m)
        cv = cb.get(m)
        if rv is not None and cv is not None:
            binary_comp[m] = {
                "random": round(float(rv), 6),
                "chronological": round(float(cv), 6),
                "delta": round(float(cv) - float(rv), 6),
                "pct_change": round(100 * (float(cv) - float(rv)) / (float(rv) + 1e-10), 2),
            }
    comp["binary"] = binary_comp

    rm = random_metrics.get("multiclass", {})
    cm = chrono_metrics.get("multiclass", {})
    mc_comp = {}
    for m in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]:
        rv = rm.get(m)
        cv = cm.get(m)
        if rv is not None and cv is not None:
            mc_comp[m] = {
                "random": round(float(rv), 6),
                "chronological": round(float(cv), 6),
                "delta": round(float(cv) - float(rv), 6),
            }
    comp["multiclass"] = mc_comp

    # Per-class detection
    rpcd = rm.get("per_class_detection", {})
    cpcd = cm.get("per_class_detection", {})
    per_class = {}
    for cls in sorted(set(list(rpcd.keys()) + list(cpcd.keys()))):
        rd = rpcd.get(cls, {}).get("detection_rate")
        cd = cpcd.get(cls, {}).get("detection_rate")
        per_class[cls] = {
            "random": round(float(rd), 4) if rd is not None else None,
            "chronological": round(float(cd), 4) if cd is not None else None,
            "delta": round(float(cd) - float(rd), 4) if rd is not None and cd is not None else None,
        }
    comp["per_class_detection"] = per_class

    return comp


def _print_summary(random_metrics, chrono_metrics, logger):
    rb = random_metrics.get("binary", {})
    cb = chrono_metrics.get("binary", {})

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║          BINARY METRICS: RANDOM vs CHRONOLOGICAL            ║")
    logger.info("╠══════════════════════════╤══════════╤══════════╤═════════════╣")
    logger.info("║ Metric                   │ Random   │ Chrono   │ Delta       ║")
    logger.info("╠══════════════════════════╪══════════╪══════════╪═════════════╣")
    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc",
              "false_positive_rate", "false_negative_rate"]:
        rv = rb.get(m)
        cv = cb.get(m)
        if rv is not None and cv is not None:
            d = float(cv) - float(rv)
            logger.info("║ %-24s │ %8.4f │ %8.4f │ %+10.4f  ║", m, float(rv), float(cv), d)
    logger.info("╚══════════════════════════╧══════════╧══════════╧═════════════╝")

    rm = random_metrics.get("multiclass", {})
    cm = chrono_metrics.get("multiclass", {})
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║       MULTICLASS METRICS: RANDOM vs CHRONOLOGICAL           ║")
    logger.info("╠══════════════════════════╤══════════╤══════════╤═════════════╣")
    for m in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]:
        rv = rm.get(m)
        cv = cm.get(m)
        if rv is not None and cv is not None:
            d = float(cv) - float(rv)
            logger.info("║ %-24s │ %8.4f │ %8.4f │ %+10.4f  ║", m, float(rv), float(cv), d)
    logger.info("╚══════════════════════════╧══════════╧══════════╧═════════════╝")


if __name__ == "__main__":
    main()
