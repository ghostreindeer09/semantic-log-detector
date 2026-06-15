"""
Drift Monitor — detects concept drift in embedding / feature space.

Supports **named channels** so that each analysis engine (structured IDS,
semantic anomaly) can maintain an independent drift baseline and window.
"""

import numpy as np
from typing import Dict, List, Optional, Union


class _DriftChannel:
    """Single-channel drift detector using a z-score on vector norms."""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.metric_window: List[float] = []
        self.baseline_stats: Optional[Dict[str, float]] = None
        self.drift_alert = False

    def add_sample(self, vector: np.ndarray) -> bool:
        """Add a sample vector, return True if drift is detected."""
        metric = float(np.linalg.norm(vector))
        self.metric_window.append(metric)

        if len(self.metric_window) >= self.window_size:
            recent = np.array(self.metric_window[-self.window_size:])

            if self.baseline_stats is None:
                self.baseline_stats = {
                    'mean': float(np.mean(recent)),
                    'std': float(np.std(recent)),
                }
                return False

            current_mean = float(np.mean(recent))
            z_score = abs(current_mean - self.baseline_stats['mean']) / (
                self.baseline_stats['std'] + 1e-6
            )

            if z_score > self.z_threshold:
                self.drift_alert = True
                return True
            else:
                self.drift_alert = False

        return False

    def get_status(self) -> Dict:
        return {
            'drift_alert': self.drift_alert,
            'samples': len(self.metric_window),
            'baseline_set': self.baseline_stats is not None,
            'recent_norms': [
                round(v, 4) for v in self.metric_window[-5:]
            ] if self.metric_window else [],
        }


class DriftMonitor:
    """
    Multi-channel drift monitor.

    Each analysis engine registers samples under its own channel name
    (e.g. ``"structured_ids"``, ``"semantic_anomaly"``).  Drift is
    tracked independently per channel.

    Backward-compatible: calling ``add_sample(vector)`` without a
    channel name routes to a ``"default"`` channel.
    """

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.channels: Dict[str, _DriftChannel] = {}

    def _get_channel(self, name: str) -> _DriftChannel:
        if name not in self.channels:
            self.channels[name] = _DriftChannel(
                window_size=self.window_size,
                z_threshold=self.z_threshold,
            )
        return self.channels[name]

    def add_sample(
        self,
        channel_or_vector: Union[str, np.ndarray],
        vector: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Add a sample and check for drift.

        Supports two calling conventions:
        - ``add_sample("structured_ids", vector)``  — named channel
        - ``add_sample(vector)``                     — backward-compat (default channel)
        """
        if isinstance(channel_or_vector, str):
            if vector is None:
                raise ValueError(
                    "When channel name is provided, vector must also be given."
                )
            channel_name = channel_or_vector
        else:
            vector = channel_or_vector
            channel_name = "default"

        return self._get_channel(channel_name).add_sample(vector)

    def get_status(self) -> Dict:
        """Return per-channel drift status."""
        if not self.channels:
            return {'drift_alert': False, 'channels': {}}

        channel_status = {
            name: ch.get_status() for name, ch in self.channels.items()
        }
        any_drift = any(ch.drift_alert for ch in self.channels.values())

        return {
            'drift_alert': any_drift,
            'channels': channel_status,
        }

