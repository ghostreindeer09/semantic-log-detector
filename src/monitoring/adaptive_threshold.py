"""
Adaptive Threshold Controller for drift-aware intrusion detection.

Provides three thresholding strategies:
  - quantile:  Set threshold at the N-th percentile of recent scores.
  - drift:     Use KS-test drift detection to trigger threshold resets.
  - combined:  Quantile-based with drift-triggered reference window resets.

Production features:
  - Bounded threshold updates (floor, ceiling, max step, EMA smoothing)
  - Incident-aware threshold freezing when anomaly rates are high
  - Drift direction analysis (upward vs downward score shifts)
  - Full audit trail (threshold history, drift events, freeze events)
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveThresholdController:
    """
    Drift-aware adaptive threshold controller for binary classification.

    Parameters
    ----------
    strategy : str
        One of "quantile", "drift", "combined".
    quantile : float
        Percentile for quantile-based thresholding (e.g. 99.5).
    window_size : int
        Number of recent scores to maintain in the sliding buffer.
    fallback_threshold : float
        Threshold used during warmup before enough samples accumulate.
    threshold_floor : float
        Minimum allowed threshold value.
    threshold_ceiling : float
        Maximum allowed threshold value.
    max_threshold_step : float
        Maximum allowed change per update cycle.
    smoothing_alpha : float
        EMA smoothing factor (0 = no change, 1 = instant).
    drift_ks_trigger : float
        KS statistic threshold to trigger a drift event.
    drift_check_interval : int
        Check for drift every N samples.
    freeze_anomaly_rate : float
        Anomaly rate above which the threshold is frozen.
    freeze_window : int
        Window size for anomaly rate computation.
    freeze_duration : int
        Minimum number of steps to keep the threshold frozen.
    """

    def __init__(
        self,
        strategy: str = "quantile",
        quantile: float = 99.5,
        window_size: int = 5000,
        fallback_threshold: float = 0.001,
        threshold_floor: float = 0.0005,
        threshold_ceiling: float = 0.5,
        max_threshold_step: float = 0.05,
        smoothing_alpha: float = 0.3,
        drift_ks_trigger: float = 0.1,
        drift_check_interval: int = 1000,
        freeze_anomaly_rate: float = 0.15,
        freeze_window: int = 500,
        freeze_duration: int = 5000,
    ):
        if strategy not in ("quantile", "drift", "combined"):
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy
        self.quantile = quantile
        self.window_size = window_size
        self.fallback_threshold = fallback_threshold
        self.threshold_floor = threshold_floor
        self.threshold_ceiling = threshold_ceiling
        self.max_threshold_step = max_threshold_step
        self.smoothing_alpha = smoothing_alpha
        self.drift_ks_trigger = drift_ks_trigger
        self.drift_check_interval = drift_check_interval
        self.freeze_anomaly_rate = freeze_anomaly_rate
        self.freeze_window = freeze_window
        self.freeze_duration = freeze_duration

        # State
        self.threshold = fallback_threshold
        self.score_buffer: deque = deque(maxlen=window_size)
        self.reference_scores: Optional[np.ndarray] = None
        self._reference_set = False
        self._step_count = 0
        self._warmup_complete = False

        # Freeze state
        self._frozen = False
        self._freeze_remaining = 0
        self._recent_decisions: deque = deque(maxlen=freeze_window)

        # History / audit trail
        self.threshold_history: List[Tuple[int, float, str]] = []
        self.drift_events: List[Dict[str, Any]] = []
        self.freeze_events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_decision(self, score: float) -> bool:
        """Return True if the score exceeds the current threshold (= alert)."""
        return score >= self.threshold

    def update(self, score: float, step: int) -> float:
        """
        Ingest a new score, update threshold for the NEXT sample.

        Parameters
        ----------
        score : float
            The prediction probability for the current sample.
        step : int
            The global step index.

        Returns
        -------
        float
            The (possibly updated) threshold.
        """
        self.score_buffer.append(score)
        self._step_count += 1

        # Track recent decisions for freeze logic
        decision = 1 if score >= self.threshold else 0
        self._recent_decisions.append(decision)

        # Check freeze logic
        self._update_freeze_state(step)

        # If frozen, do not update threshold
        if self._frozen:
            return self.threshold

        # Check warmup
        min_samples = min(self.window_size, 200)
        if len(self.score_buffer) < min_samples:
            return self.threshold

        if not self._warmup_complete:
            self._warmup_complete = True
            logger.debug("Warmup complete at step %d", step)

        # Strategy dispatch
        if self.strategy == "quantile":
            self._update_quantile(step)
        elif self.strategy == "drift":
            self._update_drift_triggered(step)
        elif self.strategy == "combined":
            self._update_combined(step)

        return self.threshold

    def get_stats(self) -> Dict[str, Any]:
        """Return current controller state."""
        return {
            "threshold": self.threshold,
            "strategy": self.strategy,
            "quantile": self.quantile,
            "window_size": self.window_size,
            "samples_seen": self._step_count,
            "buffer_fill": len(self.score_buffer),
            "warmup_complete": self._warmup_complete,
            "frozen": self._frozen,
            "freeze_remaining": self._freeze_remaining,
            "num_drift_events": len(self.drift_events),
            "num_freeze_events": len(self.freeze_events),
            "num_threshold_updates": len(self.threshold_history),
        }

    def export_history(self) -> Dict[str, Any]:
        """Export full audit trail."""
        return {
            "threshold_history": [
                {"step": s, "threshold": t, "reason": r}
                for s, t, r in self.threshold_history
            ],
            "drift_events": self.drift_events,
            "freeze_events": self.freeze_events,
            "final_stats": self.get_stats(),
        }

    # ------------------------------------------------------------------ #
    #  Strategy implementations                                            #
    # ------------------------------------------------------------------ #

    def _update_quantile(self, step: int):
        """Quantile-based threshold update."""
        scores = np.array(self.score_buffer)
        raw_quantile = float(np.percentile(scores, self.quantile))
        candidate = self._apply_smoothing(raw_quantile)
        new_threshold = self._apply_bounds(candidate)

        if new_threshold != self.threshold:
            self.threshold_history.append((step, new_threshold, "quantile"))
            self.threshold = new_threshold

    def _update_drift_triggered(self, step: int):
        """Drift-triggered threshold update with direction analysis."""
        # Set reference on first full buffer
        if not self._reference_set and len(self.score_buffer) >= self.window_size:
            self.reference_scores = np.array(self.score_buffer).copy()
            self._reference_set = True
            # Compute initial threshold from reference
            self._update_quantile(step)
            return

        if not self._reference_set:
            return

        # Check for drift periodically
        if self._step_count % self.drift_check_interval != 0:
            return

        drift_info = self._analyze_drift_direction(
            self.reference_scores, np.array(self.score_buffer)
        )

        if drift_info["ks_statistic"] > self.drift_ks_trigger:
            self.drift_events.append({
                "step": step,
                "ks_statistic": round(drift_info["ks_statistic"], 6),
                "p_value": round(drift_info["p_value"], 6),
                "direction": drift_info["direction"],
                "mean_shift": round(drift_info["mean_shift"], 6),
                "median_shift": round(drift_info["median_shift"], 6),
            })

            # Respond based on direction
            self._respond_to_drift(drift_info, step)

            # Update reference
            self.reference_scores = np.array(self.score_buffer).copy()

    def _update_combined(self, step: int):
        """Combined: quantile-based with drift-triggered resets."""
        # Always do quantile update
        self._update_quantile(step)

        # Also check drift periodically
        if not self._reference_set and len(self.score_buffer) >= self.window_size:
            self.reference_scores = np.array(self.score_buffer).copy()
            self._reference_set = True
            return

        if not self._reference_set:
            return

        if self._step_count % self.drift_check_interval != 0:
            return

        drift_info = self._analyze_drift_direction(
            self.reference_scores, np.array(self.score_buffer)
        )

        if drift_info["ks_statistic"] > self.drift_ks_trigger:
            self.drift_events.append({
                "step": step,
                "ks_statistic": round(drift_info["ks_statistic"], 6),
                "p_value": round(drift_info["p_value"], 6),
                "direction": drift_info["direction"],
                "mean_shift": round(drift_info["mean_shift"], 6),
                "median_shift": round(drift_info["median_shift"], 6),
            })

            self._respond_to_drift(drift_info, step)
            self.reference_scores = np.array(self.score_buffer).copy()

    # ------------------------------------------------------------------ #
    #  Drift direction analysis                                            #
    # ------------------------------------------------------------------ #

    def _analyze_drift_direction(
        self, reference: np.ndarray, current: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze drift magnitude and direction between two score sets."""
        from scipy.stats import ks_2samp

        ks_stat, p_value = ks_2samp(reference, current)

        ref_median = float(np.median(reference))
        cur_median = float(np.median(current))
        median_shift = cur_median - ref_median

        ref_mean = float(np.mean(reference))
        cur_mean = float(np.mean(current))
        mean_shift = cur_mean - ref_mean

        if mean_shift > 1e-6:
            direction = "upward"
        elif mean_shift < -1e-6:
            direction = "downward"
        else:
            direction = "stable"

        return {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "direction": direction,
            "median_shift": median_shift,
            "mean_shift": mean_shift,
            "ref_mean": ref_mean,
            "cur_mean": cur_mean,
        }

    def _respond_to_drift(self, drift_info: Dict, step: int):
        """Adjust threshold based on drift direction."""
        direction = drift_info["direction"]
        current_anomaly_rate = self._current_anomaly_rate()

        if direction == "upward":
            if current_anomaly_rate > self.freeze_anomaly_rate:
                # High anomaly rate + upward shift = likely attack, freeze
                # handled by freeze logic
                pass
            else:
                # Scores shifting up but anomaly rate normal → feature drift
                # Cautiously raise threshold
                scores = np.array(self.score_buffer)
                raw_q = float(np.percentile(scores, self.quantile))
                candidate = self._apply_smoothing(raw_q)
                new_threshold = self._apply_bounds(candidate)
                if new_threshold != self.threshold:
                    self.threshold_history.append(
                        (step, new_threshold, f"drift_upward_adjust")
                    )
                    self.threshold = new_threshold

        elif direction == "downward":
            # Scores decreasing → attacks may be scoring lower
            # Lower threshold to maintain sensitivity
            scores = np.array(self.score_buffer)
            raw_q = float(np.percentile(scores, self.quantile))
            # Use higher alpha for faster downward adaptation
            fast_alpha = min(self.smoothing_alpha * 2, 0.8)
            candidate = fast_alpha * raw_q + (1 - fast_alpha) * self.threshold
            new_threshold = self._apply_bounds(candidate)
            if new_threshold < self.threshold:
                self.threshold_history.append(
                    (step, new_threshold, "drift_downward_sensitivity")
                )
                self.threshold = new_threshold

    # ------------------------------------------------------------------ #
    #  Incident-aware freezing                                             #
    # ------------------------------------------------------------------ #

    def _current_anomaly_rate(self) -> float:
        """Compute the anomaly rate in the recent decision window."""
        if len(self._recent_decisions) < min(self.freeze_window, 100):
            return 0.0
        return float(np.mean(list(self._recent_decisions)))

    def _update_freeze_state(self, step: int):
        """Check whether to freeze or unfreeze the threshold."""
        anomaly_rate = self._current_anomaly_rate()

        if self._frozen:
            self._freeze_remaining -= 1
            # Unfreeze if duration expired AND anomaly rate has dropped
            if self._freeze_remaining <= 0 and anomaly_rate < self.freeze_anomaly_rate:
                self._frozen = False
                self._freeze_remaining = 0
                logger.debug("Threshold UNFROZEN at step %d (anomaly_rate=%.3f)",
                             step, anomaly_rate)
        else:
            # Check if we should freeze
            if (anomaly_rate > self.freeze_anomaly_rate and
                    len(self._recent_decisions) >= min(self.freeze_window, 100)):
                self._frozen = True
                self._freeze_remaining = self.freeze_duration
                self.freeze_events.append({
                    "step": step,
                    "anomaly_rate": round(anomaly_rate, 4),
                    "threshold_at_freeze": round(self.threshold, 6),
                    "freeze_duration": self.freeze_duration,
                    "action": "freeze_start",
                })
                logger.debug(
                    "Threshold FROZEN at step %d (anomaly_rate=%.3f, threshold=%.4f)",
                    step, anomaly_rate, self.threshold,
                )

    # ------------------------------------------------------------------ #
    #  Bounded update mechanics                                            #
    # ------------------------------------------------------------------ #

    def _apply_smoothing(self, raw_value: float) -> float:
        """Apply EMA smoothing."""
        return self.smoothing_alpha * raw_value + (1 - self.smoothing_alpha) * self.threshold

    def _apply_bounds(self, candidate: float) -> float:
        """Apply step-size limits and floor/ceiling."""
        delta = candidate - self.threshold
        if abs(delta) > self.max_threshold_step:
            candidate = self.threshold + np.sign(delta) * self.max_threshold_step

        return float(np.clip(candidate, self.threshold_floor, self.threshold_ceiling))
