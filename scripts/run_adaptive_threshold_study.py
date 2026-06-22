"""
Adaptive Threshold Evaluation Study.

Simulates streaming evaluation of the LightGBM IDS on the Friday
chronological test set using multiple adaptive thresholding strategies.

Phases:
  3. Streaming evaluation with adaptive + fixed baselines
  4. Baseline comparison with detection delay metrics
  5. Visualization (timeline, comparison charts, drift direction)
  6. Scientific interpretation report

Usage:
    python scripts/run_adaptive_threshold_study.py
"""

import csv
import glob
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ids.config_loader import load_config
from src.ids.data_loader import clean_data, create_labels, identify_features
from src.ids.chronological_splitter import load_with_day_labels, chronological_split
from src.ids.structured_trainer import StructuredTrainer
from src.monitoring.adaptive_threshold import AdaptiveThresholdController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    force=True,
)
logger = logging.getLogger("adaptive_threshold_study")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "adaptive_threshold")
CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "chronological_eval", "checkpoints"
)


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


# ===================================================================
# DATA + MODEL LOADING
# ===================================================================

def load_data_and_model(config):
    """Load CIC-IDS2017 data, chronological split, and saved model."""
    logger.info("=" * 70)
    logger.info("  LOADING DATA + MODEL")
    logger.info("=" * 70)

    csv_files = sorted(glob.glob(os.path.join("Datasets", "*.csv")))
    raw_df = load_with_day_labels(csv_files, encoding="latin-1")
    raw_df = clean_data(raw_df, config)
    raw_df, mc_encoding = create_labels(raw_df)

    feature_cols = [
        c for c in identify_features(raw_df)
        if c not in {"capture_day", "day_index", "source_file", "timestamp"}
    ]

    train_c, val_c, test_c = chronological_split(raw_df)
    config["training"]["task"] = "binary"
    trainer = StructuredTrainer.load(CHECKPOINT_DIR, config)

    return train_c, val_c, test_c, feature_cols, trainer


# ===================================================================
# CAMPAIGN SEGMENTATION
# ===================================================================

def segment_campaigns(test_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Segment the Friday test set into attack campaigns by source file.

    Returns a list of campaign dicts with start/end indices and attack type.
    """
    campaigns = []
    source_files = test_df["source_file"].values

    # Track file boundaries
    current_file = source_files[0]
    campaign_start = 0

    for i in range(1, len(source_files)):
        if source_files[i] != current_file:
            # End of segment
            segment = test_df.iloc[campaign_start:i]
            attack_types = segment.loc[
                segment["label_binary"] == 1, "label_multiclass"
            ].unique().tolist()
            n_attacks = int((segment["label_binary"] == 1).sum())

            campaigns.append({
                "source_file": current_file,
                "start_idx": campaign_start,
                "end_idx": i - 1,
                "n_samples": i - campaign_start,
                "n_attacks": n_attacks,
                "attack_types": attack_types,
            })

            current_file = source_files[i]
            campaign_start = i

    # Final segment
    segment = test_df.iloc[campaign_start:]
    attack_types = segment.loc[
        segment["label_binary"] == 1, "label_multiclass"
    ].unique().tolist()
    n_attacks = int((segment["label_binary"] == 1).sum())
    campaigns.append({
        "source_file": current_file,
        "start_idx": campaign_start,
        "end_idx": len(test_df) - 1,
        "n_samples": len(test_df) - campaign_start,
        "n_attacks": n_attacks,
        "attack_types": attack_types,
    })

    for c in campaigns:
        logger.info("  Campaign: %s (%d samples, %d attacks, types: %s)",
                     c["source_file"], c["n_samples"], c["n_attacks"],
                     c["attack_types"])

    return campaigns


# ===================================================================
# STREAMING EVALUATION
# ===================================================================

def run_streaming_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    controller: AdaptiveThresholdController,
    config_name: str,
) -> Dict[str, Any]:
    """
    Run streaming evaluation with an adaptive controller.

    The threshold at step i is computed using only scores from steps 0..i-1.
    """
    n = len(y_true)
    predictions = np.zeros(n, dtype=int)
    thresholds = np.zeros(n, dtype=float)
    frozen_flags = np.zeros(n, dtype=int)

    for i in range(n):
        score = float(y_proba[i])

        # Decision uses CURRENT threshold (before this score updates it)
        thresholds[i] = controller.threshold
        predictions[i] = 1 if controller.get_decision(score) else 0
        frozen_flags[i] = 1 if controller._frozen else 0

        # Update threshold for NEXT sample
        controller.update(score, step=i)

    return {
        "config_name": config_name,
        "predictions": predictions,
        "thresholds": thresholds,
        "frozen_flags": frozen_flags,
        "controller_history": controller.export_history(),
        "controller_stats": controller.get_stats(),
    }


def run_fixed_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    config_name: str,
) -> Dict[str, Any]:
    """Run evaluation with a fixed threshold (baseline)."""
    predictions = (y_proba >= threshold).astype(int)
    thresholds = np.full(len(y_true), threshold)
    frozen_flags = np.zeros(len(y_true), dtype=int)

    return {
        "config_name": config_name,
        "predictions": predictions,
        "thresholds": thresholds,
        "frozen_flags": frozen_flags,
        "controller_history": None,
        "controller_stats": {"threshold": threshold, "strategy": "fixed"},
    }


# ===================================================================
# METRICS COMPUTATION
# ===================================================================

def compute_metrics(
    y_true: np.ndarray,
    result: Dict,
    campaigns: List[Dict],
) -> Dict[str, Any]:
    """Compute all metrics including detection delay."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )

    preds = result["predictions"]
    thresholds = result["thresholds"]

    tp = int(((y_true == 1) & (preds == 1)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())

    n_total = len(y_true)
    n_alerts = int(preds.sum())

    metrics = {
        "config": result["config_name"],
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 6),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 6),
        "fnr": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total_alerts": n_alerts,
        "alert_rate": round(n_alerts / n_total, 6),
        "total_samples": n_total,
    }

    # Threshold stability
    if result["controller_stats"].get("strategy") != "fixed":
        t_array = thresholds
        metrics["threshold_mean"] = round(float(np.mean(t_array)), 6)
        metrics["threshold_std"] = round(float(np.std(t_array)), 6)
        metrics["threshold_min"] = round(float(np.min(t_array)), 6)
        metrics["threshold_max"] = round(float(np.max(t_array)), 6)
        # Count significant threshold changes (>1% shift)
        diffs = np.abs(np.diff(t_array))
        metrics["threshold_changes_gt_1pct"] = int((diffs > 0.01).sum())

        # Freeze stats
        frozen_flags = result["frozen_flags"]
        n_frozen = int(frozen_flags.sum())
        metrics["steps_frozen"] = n_frozen
        metrics["pct_frozen"] = round(100.0 * n_frozen / n_total, 2)

        # Recall during frozen vs unfrozen
        if n_frozen > 0:
            frozen_mask = frozen_flags == 1
            unfrozen_mask = frozen_flags == 0
            frozen_attacks = y_true[frozen_mask].sum()
            if frozen_attacks > 0:
                metrics["recall_during_freeze"] = round(
                    float(preds[frozen_mask & (y_true == 1)].sum() / frozen_attacks), 6
                )
            else:
                metrics["recall_during_freeze"] = None
            unfrozen_attacks = y_true[unfrozen_mask].sum()
            if unfrozen_attacks > 0:
                metrics["recall_during_unfrozen"] = round(
                    float(preds[unfrozen_mask & (y_true == 1)].sum() / unfrozen_attacks), 6
                )
            else:
                metrics["recall_during_unfrozen"] = None

        # Drift event count
        history = result.get("controller_history")
        if history:
            metrics["drift_events"] = len(history.get("drift_events", []))
            metrics["freeze_events"] = len(history.get("freeze_events", []))
    else:
        metrics["threshold_mean"] = thresholds[0]
        metrics["threshold_std"] = 0.0

    # --- Detection delay per campaign ---
    detection_delays = []
    for camp in campaigns:
        if camp["n_attacks"] == 0:
            continue

        start = camp["start_idx"]
        end = camp["end_idx"] + 1

        camp_y = y_true[start:end]
        camp_preds = preds[start:end]

        # Find first attack sample in this campaign
        attack_indices = np.where(camp_y == 1)[0]
        if len(attack_indices) == 0:
            continue

        first_attack_offset = int(attack_indices[0])

        # Find first TP in this campaign
        tp_indices = np.where((camp_y == 1) & (camp_preds == 1))[0]
        if len(tp_indices) == 0:
            ttfd = None  # Never detected
            delay_samples = camp["n_attacks"]  # All missed
        else:
            first_tp_offset = int(tp_indices[0])
            ttfd = first_tp_offset - first_attack_offset
            delay_samples = ttfd

        # Campaign-level recall
        camp_tp = int(((camp_y == 1) & (camp_preds == 1)).sum())
        camp_recall = round(camp_tp / camp["n_attacks"], 6) if camp["n_attacks"] > 0 else 0.0

        detection_delays.append({
            "source_file": camp["source_file"],
            "attack_types": camp["attack_types"],
            "n_attacks": camp["n_attacks"],
            "ttfd_samples": ttfd,
            "campaign_recall": camp_recall,
            "campaign_tp": camp_tp,
        })

    metrics["detection_delays"] = detection_delays

    return metrics


# ===================================================================
# VISUALIZATION
# ===================================================================

def plot_threshold_timeline(
    y_true: np.ndarray,
    results: Dict[str, Dict],
    best_adaptive_key: str,
    output_dir: str,
):
    """Plot threshold evolution over time for the best adaptive config."""
    result = results[best_adaptive_key]
    thresholds = result["thresholds"]
    frozen_flags = result["frozen_flags"]
    n = len(thresholds)
    steps = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    # --- Panel 1: Threshold over time ---
    ax = axes[0]
    ax.plot(steps, thresholds, color="#1E88E5", linewidth=0.8, label="Adaptive Threshold")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Fixed 0.5")
    ax.axhline(y=0.001, color="green", linestyle="--", alpha=0.5, linewidth=1, label="Fixed 0.001")

    # Shade frozen regions
    freeze_starts = []
    in_freeze = False
    for i in range(n):
        if frozen_flags[i] == 1 and not in_freeze:
            freeze_starts.append(i)
            in_freeze = True
        elif frozen_flags[i] == 0 and in_freeze:
            ax.axvspan(freeze_starts[-1], i, alpha=0.15, color="orange", zorder=0)
            in_freeze = False
    if in_freeze:
        ax.axvspan(freeze_starts[-1], n, alpha=0.15, color="orange", zorder=0)

    # Mark drift events
    history = result.get("controller_history")
    if history and history.get("drift_events"):
        for de in history["drift_events"]:
            color = "#E53935" if de["direction"] == "upward" else "#43A047"
            marker = "^" if de["direction"] == "upward" else "v"
            ax.axvline(x=de["step"], color=color, alpha=0.3, linewidth=0.5)

    ax.set_ylabel("Threshold", fontsize=11)
    ax.set_title(f"Adaptive Threshold Timeline — {best_adaptive_key}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2)

    # --- Panel 2: Attack presence ---
    ax = axes[1]
    # Downsample for visualization
    chunk = max(1, n // 2000)
    attack_rate = np.array([
        y_true[i:i+chunk].mean() for i in range(0, n, chunk)
    ])
    x_chunks = np.arange(len(attack_rate)) * chunk
    ax.fill_between(x_chunks, 0, attack_rate, alpha=0.6, color="#F44336", label="Attack Rate")
    ax.set_ylabel("Attack Rate", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2)

    # --- Panel 3: Frozen state ---
    ax = axes[2]
    frozen_chunks = np.array([
        frozen_flags[i:i+chunk].mean() for i in range(0, n, chunk)
    ])
    ax.fill_between(x_chunks, 0, frozen_chunks, alpha=0.6, color="#FF9800", label="Frozen")
    ax.set_ylabel("Frozen", fontsize=10)
    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_timeline.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Threshold timeline saved.")


def plot_comparison_charts(all_metrics: List[Dict], output_dir: str):
    """Generate comparison bar charts."""
    names = [m["config"] for m in all_metrics]
    n_configs = len(names)

    # --- Precision / Recall / F1 comparison ---
    fig, ax = plt.subplots(figsize=(max(12, n_configs * 1.5), 6))
    x = np.arange(n_configs)
    width = 0.25

    precision = [m["precision"] for m in all_metrics]
    recall = [m["recall"] for m in all_metrics]
    f1 = [m["f1"] for m in all_metrics]

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#43A047", alpha=0.8)
    bars2 = ax.bar(x, recall, width, label="Recall", color="#1E88E5", alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label="F1", color="#E53935", alpha=0.8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Method Comparison — Precision / Recall / F1", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- Alert volume comparison ---
    fig, ax = plt.subplots(figsize=(max(12, n_configs * 1.5), 6))
    alerts = [m["total_alerts"] for m in all_metrics]
    colors = ["#FF6F00" if "Fixed" in n else "#1E88E5" for n in names]
    bars = ax.bar(x, alerts, color=colors, alpha=0.8)
    ax.set_ylabel("Total Alerts", fontsize=12)
    ax.set_title("Alert Volume Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, alerts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                f"{val:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alert_volume_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Comparison charts saved.")


def plot_detection_delay(all_metrics: List[Dict], output_dir: str):
    """Plot detection delay (TTFD) per campaign per method."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Collect campaign names
    campaign_names = []
    for m in all_metrics:
        for dd in m.get("detection_delays", []):
            name = dd["source_file"][:30]
            if name not in campaign_names:
                campaign_names.append(name)

    if not campaign_names:
        logger.warning("No detection delay data to plot.")
        plt.close()
        return

    x = np.arange(len(campaign_names))
    width = 0.8 / len(all_metrics)

    for i, m in enumerate(all_metrics):
        ttfds = []
        for cn in campaign_names:
            dd_match = [dd for dd in m.get("detection_delays", [])
                        if dd["source_file"][:30] == cn]
            if dd_match and dd_match[0]["ttfd_samples"] is not None:
                ttfds.append(dd_match[0]["ttfd_samples"])
            else:
                ttfds.append(0)

        offset = (i - len(all_metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, ttfds, width, label=m["config"], alpha=0.8)

    ax.set_ylabel("Time to First Detection (samples)", fontsize=12)
    ax.set_title("Detection Delay per Campaign", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(campaign_names, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_delay.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Detection delay plot saved.")


def plot_drift_direction(result: Dict, output_dir: str):
    """Plot drift direction and magnitude over time."""
    history = result.get("controller_history")
    if not history or not history.get("drift_events"):
        logger.info("No drift events to plot.")
        return

    events = history["drift_events"]
    steps = [e["step"] for e in events]
    ks_stats = [e["ks_statistic"] for e in events]
    mean_shifts = [e["mean_shift"] for e in events]
    directions = [e["direction"] for e in events]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: KS statistic over time
    ax = axes[0]
    colors = ["#E53935" if d == "upward" else "#43A047" if d == "downward" else "#9E9E9E"
              for d in directions]
    ax.bar(steps, ks_stats, width=max(1, steps[-1]//200) if steps else 1,
           color=colors, alpha=0.8)
    ax.set_ylabel("KS Statistic", fontsize=11)
    ax.set_title("Drift Events — Magnitude & Direction", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    # Legend
    up_patch = mpatches.Patch(color="#E53935", alpha=0.8, label="Upward drift")
    down_patch = mpatches.Patch(color="#43A047", alpha=0.8, label="Downward drift")
    ax.legend(handles=[up_patch, down_patch], fontsize=9)

    # Panel 2: Mean shift
    ax = axes[1]
    ax.bar(steps, mean_shifts, width=max(1, steps[-1]//200) if steps else 1,
           color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean Score Shift", fontsize=11)
    ax.set_xlabel("Sample Index", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drift_direction_timeline.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Drift direction timeline saved.")


# ===================================================================
# REPORT GENERATION
# ===================================================================

def generate_report(
    all_metrics: List[Dict],
    best_adaptive_key: str,
    campaigns: List[Dict],
    output_dir: str,
):
    """Generate the adaptive threshold operational report."""
    report_path = os.path.join(output_dir, "adaptive_threshold_report.md")

    # Find fixed baselines and best adaptive for comparison
    fixed_05 = next((m for m in all_metrics if m["config"] == "Fixed_0.5"), None)
    fixed_001 = next((m for m in all_metrics if m["config"] == "Fixed_0.001"), None)
    best_adaptive = next((m for m in all_metrics if m["config"] == best_adaptive_key), None)
    adaptive_methods = [m for m in all_metrics if "Fixed" not in m["config"]]

    with open(report_path, "w") as f:
        f.write("# Drift-Aware Adaptive Thresholding — Operational Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("---\n\n")

        # --- Section 1: Why static thresholds failed ---
        f.write("## 1. Why Static Thresholds Failed\n\n")
        f.write("The LightGBM IDS was trained on Monday–Wednesday traffic (Brute Force, DoS attacks). "
                "When deployed on Friday traffic (DDoS, PortScan, Bot), the model's prediction score "
                "distribution shifted dramatically:\n\n")
        f.write("- **Benign traffic**: median score = 0.0008 (very low, correctly)\n")
        f.write("- **Malicious traffic**: median score = 0.2025 (should be near 1.0, but compressed)\n\n")
        f.write("The model *ranks* attacks above benign (ROC-AUC = 0.93) but assigns them low "
                "absolute probabilities. A fixed threshold of 0.5 misses 97% of attacks. "
                "A fixed threshold of 0.001 catches 99.7% of attacks but also generates a "
                "23% false positive rate because it's below the benign score distribution's tail.\n\n")
        f.write("Neither threshold can simultaneously achieve high recall and low FPR because "
                "the score distributions overlap in the [0.001, 0.04] range — and the optimal "
                "decision boundary is different for different traffic regimes.\n\n")

        # --- Section 2: Baseline comparison ---
        f.write("## 2. Method Comparison\n\n")
        f.write("| Method | Precision | Recall | F1 | FPR | Alerts | Alert Rate |\n")
        f.write("|--------|-----------|--------|------|------|--------|------------|\n")
        for m in all_metrics:
            f.write(f"| {m['config']} | {m['precision']:.4f} | {m['recall']:.4f} | "
                    f"{m['f1']:.4f} | {m['fpr']:.4f} | {m['total_alerts']:,} | "
                    f"{m['alert_rate']:.4f} |\n")

        f.write("\n### Fixed Baselines\n\n")
        if fixed_05:
            f.write(f"- **Fixed 0.5**: Recall = {fixed_05['recall']:.4f}, "
                    f"FPR = {fixed_05['fpr']:.4f}, Alerts = {fixed_05['total_alerts']:,}\n")
        if fixed_001:
            f.write(f"- **Fixed 0.001**: Recall = {fixed_001['recall']:.4f}, "
                    f"FPR = {fixed_001['fpr']:.4f}, Alerts = {fixed_001['total_alerts']:,}\n")

        # --- Section 3: Adaptive threshold analysis ---
        f.write("\n## 3. Adaptive Threshold Analysis\n\n")

        if best_adaptive:
            f.write(f"### Best Adaptive Configuration: {best_adaptive_key}\n\n")
            f.write(f"- **F1**: {best_adaptive['f1']:.4f}\n")
            f.write(f"- **Precision**: {best_adaptive['precision']:.4f}\n")
            f.write(f"- **Recall**: {best_adaptive['recall']:.4f}\n")
            f.write(f"- **FPR**: {best_adaptive['fpr']:.4f}\n")
            f.write(f"- **Total Alerts**: {best_adaptive['total_alerts']:,}\n")
            if "threshold_std" in best_adaptive:
                f.write(f"- **Threshold Mean ± Std**: {best_adaptive['threshold_mean']:.4f} ± "
                        f"{best_adaptive['threshold_std']:.4f}\n")
                f.write(f"- **Threshold Range**: [{best_adaptive.get('threshold_min', 0):.4f}, "
                        f"{best_adaptive.get('threshold_max', 0):.4f}]\n")

            # Compare against fixed 0.001
            if fixed_001:
                f1_change = best_adaptive["f1"] - fixed_001["f1"]
                fpr_change = best_adaptive["fpr"] - fixed_001["fpr"]
                alert_reduction = fixed_001["total_alerts"] - best_adaptive["total_alerts"]
                f.write(f"\n### vs Fixed 0.001 Baseline\n\n")
                f.write(f"- F1 change: {f1_change:+.4f}\n")
                f.write(f"- FPR change: {fpr_change:+.4f}\n")
                f.write(f"- Alert reduction: {alert_reduction:+,} "
                        f"({100*alert_reduction/max(fixed_001['total_alerts'],1):+.1f}%)\n")

        # --- Section 4: Threshold stability ---
        f.write("\n## 4. Threshold Stability\n\n")
        f.write("| Config | Mean | Std | Min | Max | Changes >1% |\n")
        f.write("|--------|------|-----|-----|-----|-------------|\n")
        for m in adaptive_methods:
            f.write(f"| {m['config']} | {m.get('threshold_mean',0):.4f} | "
                    f"{m.get('threshold_std',0):.4f} | "
                    f"{m.get('threshold_min',0):.4f} | "
                    f"{m.get('threshold_max',0):.4f} | "
                    f"{m.get('threshold_changes_gt_1pct',0)} |\n")

        # --- Section 5: Freezing behavior ---
        f.write("\n## 5. Incident-Aware Freezing\n\n")
        has_freezing = any(m.get("steps_frozen", 0) > 0 for m in adaptive_methods)
        if has_freezing:
            f.write("| Config | Steps Frozen | % Frozen | Recall (Frozen) | Recall (Unfrozen) | Freeze Events |\n")
            f.write("|--------|-------------|----------|-----------------|-------------------|---------------|\n")
            for m in adaptive_methods:
                rf = m.get("recall_during_freeze")
                ru = m.get("recall_during_unfrozen")
                f.write(f"| {m['config']} | {m.get('steps_frozen',0):,} | "
                        f"{m.get('pct_frozen',0):.1f}% | "
                        f"{rf if rf is not None else 'N/A'} | "
                        f"{ru if ru is not None else 'N/A'} | "
                        f"{m.get('freeze_events',0)} |\n")
            f.write("\nFreezing prevents the threshold from adapting to attack traffic as 'normal'. "
                    "The recall-during-freeze metric shows whether the threshold was at a useful level "
                    "when it was locked.\n")
        else:
            f.write("No freeze events were triggered during evaluation. This could mean:\n")
            f.write("- The anomaly rate never exceeded the freeze threshold (15%)\n")
            f.write("- The adaptive threshold tracked the score distribution closely enough "
                    "that the anomaly rate remained below the trigger\n")

        # --- Section 6: Detection delay ---
        f.write("\n## 6. Detection Delay\n\n")
        f.write("Time-to-first-detection (TTFD) measures samples from first attack in a campaign "
                "to the first true positive alert.\n\n")
        f.write("| Method | Campaign | Attack Types | TTFD (samples) | Campaign Recall |\n")
        f.write("|--------|----------|-------------|----------------|----------------|\n")
        for m in all_metrics:
            for dd in m.get("detection_delays", []):
                ttfd_str = str(dd["ttfd_samples"]) if dd["ttfd_samples"] is not None else "NEVER"
                f.write(f"| {m['config']} | {dd['source_file'][:35]} | "
                        f"{', '.join(dd['attack_types'])} | {ttfd_str} | "
                        f"{dd['campaign_recall']:.4f} |\n")

        # --- Section 7: Drift events ---
        f.write("\n## 7. Drift Events & Direction\n\n")
        for m in adaptive_methods:
            drift_count = m.get("drift_events", 0)
            if drift_count > 0:
                f.write(f"**{m['config']}**: {drift_count} drift events detected\n\n")

        # --- Section 8: SOC operational assessment ---
        f.write("\n## 8. SOC Operational Assessment\n\n")
        f.write("### Would this reduce alert fatigue?\n\n")
        if fixed_001 and best_adaptive:
            if best_adaptive["total_alerts"] < fixed_001["total_alerts"]:
                reduction = fixed_001["total_alerts"] - best_adaptive["total_alerts"]
                f.write(f"**Yes.** The best adaptive method generated {reduction:,} fewer alerts "
                        f"({100*reduction/max(fixed_001['total_alerts'],1):.1f}% reduction) "
                        f"compared to Fixed 0.001.\n\n")
            else:
                f.write(f"**No.** The adaptive method generated more alerts "
                        f"({best_adaptive['total_alerts']:,}) than Fixed 0.001 "
                        f"({fixed_001['total_alerts']:,}).\n\n")

        f.write("### Was recall preserved?\n\n")
        if best_adaptive:
            f.write(f"Adaptive recall = {best_adaptive['recall']:.4f}")
            if fixed_001:
                f.write(f" vs Fixed 0.001 recall = {fixed_001['recall']:.4f}")
            f.write("\n\n")

        f.write("### How frequently did the threshold change?\n\n")
        if best_adaptive:
            changes = best_adaptive.get("threshold_changes_gt_1pct", 0)
            f.write(f"The threshold changed significantly (>1%) on **{changes}** occasions "
                    f"across {best_adaptive['total_samples']:,} samples.\n\n")

        # --- Section 9: Limitations ---
        f.write("## 9. Limitations\n\n")
        f.write("1. **No true temporal ordering**: The Friday test data is ordered by source file, "
                "not by true network timestamp. Within each file, flow ordering may not be strictly "
                "chronological.\n\n")
        f.write("2. **Single-day evaluation**: The adaptive system was evaluated on one day of traffic. "
                "Multi-day evaluation across varying attack patterns would provide stronger evidence.\n\n")
        f.write("3. **No concept drift in features**: The adaptive threshold only adjusts the decision "
                "boundary — it does not address feature-level drift. If the model's feature representations "
                "degrade, threshold adaptation cannot compensate.\n\n")
        f.write("4. **Label-free operation tradeoff**: The quantile-based approach does not use labels, "
                "which makes it deployable but also means it cannot directly optimize recall or precision. "
                "It assumes that anomalous scores are in the tail of the distribution.\n\n")
        f.write("5. **Warmup period vulnerability**: During warmup, the controller uses a fallback "
                "threshold. If the initial traffic distribution differs significantly from what the "
                "fallback was calibrated on, early detection performance may suffer.\n\n")

        # --- Section 10: Conclusion ---
        f.write("## 10. Conclusion\n\n")
        f.write("### The Complete Engineering Story\n\n")
        f.write("1. **High offline metrics can be misleading.** Random split F1 = 0.997 collapsed to "
                "0.056 under temporal evaluation.\n\n")
        f.write("2. **Temporal validation exposes deployment failure.** The model retained discrimination "
                "(ROC-AUC = 0.93) but the default threshold was 500× too high.\n\n")
        f.write("3. **Calibration alone cannot solve severe distribution shift.** Platt and Isotonic "
                "calibration, trained on Thursday's distribution, did not transfer to Friday.\n\n")
        f.write("4. **Adaptive decision systems may provide a more robust operating model.** ")

        if best_adaptive and fixed_001:
            if best_adaptive["fpr"] < fixed_001["fpr"] and best_adaptive["recall"] > 0.80:
                f.write("The adaptive threshold achieved a better precision-recall tradeoff than any "
                        "fixed threshold, demonstrating that dynamic sensitivity adjustment is a viable "
                        "approach to the distribution shift problem.\n")
            elif best_adaptive["fpr"] < fixed_001["fpr"]:
                f.write("The adaptive threshold reduced false positives but at some cost to recall, "
                        "illustrating the fundamental tradeoff. Whether this is operationally acceptable "
                        "depends on the SOC's capacity and risk tolerance.\n")
            else:
                f.write("In this experiment, adaptive thresholding did not clearly dominate the best "
                        "fixed threshold. This is an honest negative result: the score distribution shift "
                        "in this dataset may be too abrupt for smooth adaptation. The controller's bounded "
                        "updates prevented it from tracking the shift fast enough, while unbounded updates "
                        "would risk oscillation. This tension is inherent to adaptive systems.\n")

    logger.info("Report written to %s", report_path)


# ===================================================================
# MAIN
# ===================================================================

def main():
    start_time = time.time()

    config = load_config("config/ids_config.yaml")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data + model ----
    train_c, val_c, test_c, feature_cols, trainer = load_data_and_model(config)

    y_true = test_c["label_binary"].values
    y_proba = trainer.predict_proba(test_c)

    # Sanity check
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_proba)
    logger.info("SANITY CHECK: ROC-AUC=%.4f (expected 0.9289)", auc)
    assert abs(auc - 0.9289) < 0.01, f"ROC-AUC mismatch: {auc}"

    # ---- Segment campaigns ----
    campaigns = segment_campaigns(test_c)

    # ---- Define configurations ----
    configs = {
        "Fixed_0.5": {"type": "fixed", "threshold": 0.5},
        "Fixed_0.001": {"type": "fixed", "threshold": 0.001},
        "Q99_W5K": {"type": "adaptive", "strategy": "quantile", "quantile": 99.0,
                     "window_size": 5000},
        "Q995_W5K": {"type": "adaptive", "strategy": "quantile", "quantile": 99.5,
                      "window_size": 5000},
        "Q999_W5K": {"type": "adaptive", "strategy": "quantile", "quantile": 99.9,
                      "window_size": 5000},
        "Q995_W1K": {"type": "adaptive", "strategy": "quantile", "quantile": 99.5,
                      "window_size": 1000},
        "Q995_W10K": {"type": "adaptive", "strategy": "quantile", "quantile": 99.5,
                       "window_size": 10000},
        "Drift_Q995": {"type": "adaptive", "strategy": "combined", "quantile": 99.5,
                        "window_size": 5000},
    }

    # ---- Run all evaluations ----
    logger.info("=" * 70)
    logger.info("  RUNNING STREAMING EVALUATIONS")
    logger.info("=" * 70)

    results = {}
    all_metrics = []

    for name, cfg in configs.items():
        logger.info("--- Evaluating: %s ---", name)

        if cfg["type"] == "fixed":
            result = run_fixed_evaluation(y_true, y_proba, cfg["threshold"], name)
        else:
            controller = AdaptiveThresholdController(
                strategy=cfg["strategy"],
                quantile=cfg["quantile"],
                window_size=cfg["window_size"],
                fallback_threshold=0.001,
                threshold_floor=0.0005,
                threshold_ceiling=0.5,
                max_threshold_step=0.05,
                smoothing_alpha=0.3,
                drift_ks_trigger=0.1,
                drift_check_interval=1000,
                freeze_anomaly_rate=0.15,
                freeze_window=500,
                freeze_duration=5000,
            )
            result = run_streaming_evaluation(y_true, y_proba, controller, name)

        results[name] = result
        metrics = compute_metrics(y_true, result, campaigns)
        all_metrics.append(metrics)

        logger.info("  %s → F1=%.4f  P=%.4f  R=%.4f  FPR=%.4f  Alerts=%d",
                     name, metrics["f1"], metrics["precision"],
                     metrics["recall"], metrics["fpr"], metrics["total_alerts"])

    # ---- Find best adaptive config (by F1) ----
    adaptive_metrics = [m for m in all_metrics if "Fixed" not in m["config"]]
    best_adaptive = max(adaptive_metrics, key=lambda m: m["f1"])
    best_adaptive_key = best_adaptive["config"]
    logger.info("Best adaptive config: %s (F1=%.4f)", best_adaptive_key, best_adaptive["f1"])

    # ---- Save threshold_over_time.csv for best adaptive ----
    best_result = results[best_adaptive_key]
    csv_path = os.path.join(OUTPUT_DIR, "threshold_over_time.csv")
    # Downsample to every 100th sample for manageable file size
    step_interval = 100
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["step", "threshold", "score", "prediction",
                          "true_label", "frozen"])
        for i in range(0, len(y_true), step_interval):
            writer.writerow([
                i,
                round(float(best_result["thresholds"][i]), 6),
                round(float(y_proba[i]), 6),
                int(best_result["predictions"][i]),
                int(y_true[i]),
                int(best_result["frozen_flags"][i]),
            ])
    logger.info("Threshold time series saved to %s", csv_path)

    # ---- Save metrics JSON ----
    metrics_path = os.path.join(OUTPUT_DIR, "adaptive_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(_make_serializable(all_metrics), f, indent=2)

    comparison_path = os.path.join(OUTPUT_DIR, "comparison.json")
    comparison = {m["config"]: _make_serializable(m) for m in all_metrics}
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Metrics saved.")

    # ---- Visualizations ----
    logger.info("=" * 70)
    logger.info("  GENERATING VISUALIZATIONS")
    logger.info("=" * 70)

    plot_threshold_timeline(y_true, results, best_adaptive_key, OUTPUT_DIR)
    plot_comparison_charts(all_metrics, OUTPUT_DIR)
    plot_detection_delay(all_metrics, OUTPUT_DIR)

    # Drift direction plot for the combined strategy
    drift_result = results.get("Drift_Q995")
    if drift_result:
        plot_drift_direction(drift_result, OUTPUT_DIR)

    # ---- Report ----
    logger.info("=" * 70)
    logger.info("  GENERATING REPORT")
    logger.info("=" * 70)

    generate_report(all_metrics, best_adaptive_key, campaigns, OUTPUT_DIR)

    # ---- Summary ----
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  ADAPTIVE THRESHOLD STUDY COMPLETE (%.1f seconds)", elapsed)
    logger.info("  Outputs: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    for root, dirs, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            logger.info("  %s (%d bytes)", os.path.relpath(fpath, OUTPUT_DIR), size)


if __name__ == "__main__":
    main()
