"""
Chronological Evaluator — runs both random and chronological split
evaluations with identical model/hyperparameters/threshold calibration,
then outputs side-by-side metrics for comparison.

Usage:
    python -m src.ids.chronological_evaluator --config config/ids_config.yaml

    Dry-run (no data needed):
    python -m src.ids.chronological_evaluator --dry-run

Outputs:
    outputs/metrics/random_split_metrics.json
    outputs/metrics/chronological_metrics.json
    outputs/metrics/evaluation_comparison.json
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.ids.config_loader import load_config
from src.ids.data_loader import (
    clean_data,
    create_labels,
    discover_csv_files,
    identify_features,
    stratified_split,
)
from src.ids.chronological_splitter import (
    chronological_split,
    load_with_day_labels,
    log_split_summary,
)
from src.ids.structured_trainer import StructuredTrainer
from src.ids.evaluator import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    run_full_evaluation,
)
from src.ids.threshold_calibrator import calibrate_threshold

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Evaluation functions                                                #
# ------------------------------------------------------------------ #

def _evaluate_split(
    trainer: StructuredTrainer,
    test_df: pd.DataFrame,
    task: str,
    mc_encoding: Dict[str, int],
    prefix: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on a test split.

    Returns a combined metrics dict with binary + multiclass results.
    """
    class_names = sorted(mc_encoding.keys(), key=lambda k: mc_encoding[k])
    results = {}

    # ── Binary evaluation ──
    y_true_bin = test_df["label_binary"].values
    y_pred_bin = (trainer.predict_proba(test_df) >= 0.5).astype(int) if task == "binary" else None

    if task == "binary":
        y_pred = trainer.predict(test_df)
        y_proba = trainer.predict_proba(test_df)
        binary_metrics = compute_binary_metrics(y_true_bin, y_pred, y_proba)
        results["binary"] = binary_metrics

        # Threshold calibration
        threshold_results = calibrate_threshold(
            y_true=y_true_bin,
            y_proba=y_proba,
            config={"threshold": {"methods": ["youden_j", "max_f1"], "default_method": "youden_j"}},
            output_dir=os.path.join(output_dir, f"{prefix}_threshold"),
        )
        results["threshold_calibration"] = threshold_results

    # ── Multiclass evaluation (always computed) ──
    # Switch to multiclass task temporarily
    original_task = trainer.task
    trainer.task = "multiclass"
    try:
        y_true_mc = test_df["label_multiclass_encoded"].values
        y_pred_mc = trainer.predict(test_df)
        y_proba_mc = trainer.predict_proba(test_df)

        mc_metrics = compute_multiclass_metrics(
            y_true_mc, y_pred_mc, class_names, y_proba_mc
        )
        results["multiclass"] = mc_metrics
    except Exception as e:
        logger.warning("Multiclass evaluation failed: %s", e)
        results["multiclass"] = {"error": str(e)}
    finally:
        trainer.task = original_task

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2, default=str)
    logger.info("Metrics saved to %s", out_path)

    return results


def _train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    mc_encoding: Dict[str, int],
    config: Dict[str, Any],
    prefix: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Train a fresh model on train_df, evaluate on test_df.
    """
    task = config["training"]["task"]

    # Train
    trainer = StructuredTrainer(config)
    train_log = trainer.train(train_df, val_df, feature_cols)

    # Evaluate
    results = _evaluate_split(
        trainer, test_df, task, mc_encoding, prefix, output_dir
    )
    results["training_log"] = train_log
    results["split_strategy"] = prefix

    return results


# ------------------------------------------------------------------ #
#  Main pipeline                                                       #
# ------------------------------------------------------------------ #

def run_comparison(config_path: str = "config/ids_config.yaml") -> Dict:
    """
    Run the full comparison: random split vs chronological split.

    Both use identical model architecture, hyperparameters, and
    threshold calibration methodology.
    """
    config = load_config(config_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        force=True,
    )

    output_dir = os.path.join(config["paths"]["output_dir"], "chronological_eval")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load raw data with day labels ──
    csv_files = discover_csv_files(config["paths"]["raw_dataset_dir"])
    raw_df = load_with_day_labels(csv_files, config["data"].get("encoding", "latin-1"))

    # Clean
    raw_df = clean_data(raw_df, config)

    # Create labels
    raw_df, mc_encoding = create_labels(raw_df)

    # Identify features (excluding day/source columns)
    exclude_extra = {"capture_day", "day_index", "source_file", "timestamp"}
    feature_cols = [
        c for c in identify_features(raw_df)
        if c not in exclude_extra
    ]

    logger.info("=" * 70)
    logger.info("  STRATEGY A: RANDOM SPLIT")
    logger.info("=" * 70)

    train_r, val_r, test_r = stratified_split(raw_df, config, stratify_col="label_binary")
    random_results = _train_and_evaluate(
        train_r, val_r, test_r, feature_cols, mc_encoding,
        config, "random_split", output_dir,
    )

    logger.info("=" * 70)
    logger.info("  STRATEGY B: CHRONOLOGICAL SPLIT")
    logger.info("=" * 70)

    train_c, val_c, test_c = chronological_split(raw_df)
    chrono_results = _train_and_evaluate(
        train_c, val_c, test_c, feature_cols, mc_encoding,
        config, "chronological", output_dir,
    )

    # ── Chronological split summary ──
    split_summary = log_split_summary(train_c, val_c, test_c)
    with open(os.path.join(output_dir, "chronological_split_summary.json"), "w") as f:
        json.dump(split_summary, f, indent=2, default=str)

    # ── Comparison ──
    comparison = _build_comparison(random_results, chrono_results)
    with open(os.path.join(output_dir, "evaluation_comparison.json"), "w") as f:
        json.dump(_make_serializable(comparison), f, indent=2, default=str)
    logger.info("Comparison saved to %s", os.path.join(output_dir, "evaluation_comparison.json"))

    # Also save to standard metrics dir
    metrics_dir = config["paths"]["metrics_dir"]
    for name, data in [("random_split_metrics", random_results),
                       ("chronological_metrics", chrono_results)]:
        path = os.path.join(metrics_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(_make_serializable(data), f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("  COMPARISON COMPLETE")
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 70)

    return comparison


def _build_comparison(
    random_results: Dict, chrono_results: Dict
) -> Dict:
    """Build a structured comparison between two evaluation runs."""
    comparison = {"strategy_a": "random_split", "strategy_b": "chronological"}

    # Binary metrics comparison
    if "binary" in random_results and "binary" in chrono_results:
        rb = random_results["binary"]
        cb = chrono_results["binary"]
        binary_comp = {}
        for metric in ["accuracy", "precision", "recall", "f1_score",
                        "roc_auc", "pr_auc", "false_positive_rate",
                        "false_negative_rate"]:
            rv = rb.get(metric)
            cv = cb.get(metric)
            if rv is not None and cv is not None:
                binary_comp[metric] = {
                    "random_split": round(rv, 6),
                    "chronological": round(cv, 6),
                    "delta": round(cv - rv, 6),
                    "pct_change": round(100 * (cv - rv) / (rv + 1e-10), 2),
                }
        comparison["binary_metrics"] = binary_comp

    # Multiclass metrics comparison
    if "multiclass" in random_results and "multiclass" in chrono_results:
        rm = random_results["multiclass"]
        cm = chrono_results["multiclass"]
        mc_comp = {}
        for metric in ["accuracy", "macro_f1", "weighted_f1",
                        "macro_precision", "macro_recall"]:
            rv = rm.get(metric)
            cv = cm.get(metric)
            if rv is not None and cv is not None:
                mc_comp[metric] = {
                    "random_split": round(rv, 6),
                    "chronological": round(cv, 6),
                    "delta": round(cv - rv, 6),
                    "pct_change": round(100 * (cv - rv) / (rv + 1e-10), 2),
                }
        comparison["multiclass_metrics"] = mc_comp

        # Per-class detection rate comparison
        if "per_class_detection" in rm and "per_class_detection" in cm:
            per_class = {}
            all_classes = set(rm["per_class_detection"].keys()) | set(cm["per_class_detection"].keys())
            for cls in sorted(all_classes):
                rdet = rm["per_class_detection"].get(cls, {}).get("detection_rate", None)
                cdet = cm["per_class_detection"].get(cls, {}).get("detection_rate", None)
                per_class[cls] = {
                    "random_split": round(rdet, 4) if rdet is not None else None,
                    "chronological": round(cdet, 4) if cdet is not None else None,
                    "delta": round(cdet - rdet, 4) if rdet is not None and cdet is not None else None,
                }
            comparison["per_class_detection"] = per_class

    return comparison


# ------------------------------------------------------------------ #
#  Utilities                                                           #
# ------------------------------------------------------------------ #

def _make_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="CIC-IDS2017 Chronological vs Random Split Evaluation"
    )
    parser.add_argument(
        "--config", type=str, default="config/ids_config.yaml",
        help="Path to IDS configuration file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate script without loading data",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run mode: script parsed successfully.")
        print("Modules imported: config_loader, data_loader, chronological_splitter,")
        print("  structured_trainer, evaluator, threshold_calibrator")
        print("Ready to run when CIC-IDS2017 data is available.")
        return

    run_comparison(args.config)


if __name__ == "__main__":
    main()
