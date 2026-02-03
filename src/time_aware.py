"""
Time-Aware Module - Temporal context and sliding window analysis.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque


@dataclass
class TemporalContext:
    """Temporal context for a log entry."""
    log_id: str
    timestamp: datetime
    time_adjusted_score: float
    original_score: float
    window_stats: Dict
    temporal_weight: float


class TimeAwareModule:
    """Handles temporal context and sliding window analysis."""
    
    def __init__(
        self,
        window_size: int = 100,
        temporal_weight: float = 0.3,
        decay_factor: float = 0.95,
        time_bucket_minutes: int = 5
    ):
        self.window_size = window_size
        self.temporal_weight = temporal_weight
        self.decay_factor = decay_factor
        self.time_bucket_minutes = time_bucket_minutes
        
        self.score_window: deque = deque(maxlen=window_size)
        self.timestamp_window: deque = deque(maxlen=window_size)
        self.baseline_stats: Dict = {}
    
    def update_window(
        self,
        score: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add a score to the sliding window."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.score_window.append(score)
        self.timestamp_window.append(timestamp)
    
    def get_window_stats(self) -> Dict:
        """Get statistics from the current window."""
        if len(self.score_window) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        scores = np.array(list(self.score_window))
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'count': len(scores)
        }
    
    def compute_temporal_weight(self, timestamp: datetime) -> float:
        """Compute temporal weight based on recency."""
        if len(self.timestamp_window) == 0:
            return 1.0
        
        recent_ts = max(self.timestamp_window)
        time_diff = (recent_ts - timestamp).total_seconds()
        
        # Apply exponential decay
        weight = self.decay_factor ** (time_diff / 3600)  # Per hour
        return max(weight, 0.1)  # Minimum weight
    
    def adjust_score(
        self,
        log_id: str,
        original_score: float,
        timestamp: Optional[datetime] = None
    ) -> TemporalContext:
        """Adjust anomaly score based on temporal context."""
        if timestamp is None:
            timestamp = datetime.now()
        
        window_stats = self.get_window_stats()
        temp_weight = self.compute_temporal_weight(timestamp)
        
        # Adjust score based on window context
        if window_stats['count'] > 10 and window_stats['std'] > 0:
            z_score = (original_score - window_stats['mean']) / window_stats['std']
            context_factor = 1 + (z_score * self.temporal_weight)
            adjusted_score = original_score * max(0.5, min(context_factor, 2.0))
        else:
            adjusted_score = original_score
        
        # Apply temporal weight
        final_score = adjusted_score * temp_weight
        
        # Update window
        self.update_window(original_score, timestamp)
        
        return TemporalContext(
            log_id=log_id, timestamp=timestamp,
            time_adjusted_score=min(final_score, 1.0),
            original_score=original_score,
            window_stats=window_stats, temporal_weight=temp_weight
        )
    
    def detect_burst(self, threshold_multiplier: float = 2.0) -> bool:
        """Detect if there's a burst in anomaly scores."""
        if len(self.score_window) < 10:
            return False
        
        recent = list(self.score_window)[-10:]
        older = list(self.score_window)[:-10]
        
        if len(older) < 5:
            return False
        
        return np.mean(recent) > threshold_multiplier * np.mean(older)
    
    def get_time_series(self) -> Tuple[List[datetime], List[float]]:
        """Get time series data from window."""
        return list(self.timestamp_window), list(self.score_window)
    
    def reset(self) -> None:
        """Reset the sliding window."""
        self.score_window.clear()
        self.timestamp_window.clear()


class TemporalEmbedder:
    """Adds temporal features to embeddings."""
    
    def __init__(self, embedding_dim: int = 384, temporal_dim: int = 16):
        self.embedding_dim = embedding_dim
        self.temporal_dim = temporal_dim
    
    def encode_timestamp(self, timestamp: datetime) -> np.ndarray:
        """Encode timestamp as features."""
        features = [
            timestamp.hour / 24,
            timestamp.minute / 60,
            timestamp.weekday() / 7,
            timestamp.day / 31,
            timestamp.month / 12,
            np.sin(2 * np.pi * timestamp.hour / 24),
            np.cos(2 * np.pi * timestamp.hour / 24),
            np.sin(2 * np.pi * timestamp.weekday() / 7),
            np.cos(2 * np.pi * timestamp.weekday() / 7),
        ]
        
        # Pad to temporal_dim
        while len(features) < self.temporal_dim:
            features.append(0.0)
        
        return np.array(features[:self.temporal_dim], dtype=np.float32)
    
    def augment_embedding(
        self,
        embedding: np.ndarray,
        timestamp: datetime
    ) -> np.ndarray:
        """Add temporal features to embedding."""
        temporal_features = self.encode_timestamp(timestamp)
        return np.concatenate([embedding, temporal_features])
