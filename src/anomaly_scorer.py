"""
Anomaly Scorer Module - Distance-based anomaly scoring.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AnomalyScore:
    """Anomaly score result for a log entry."""
    log_id: str
    score: float
    is_anomaly: bool
    confidence: float
    distances: List[float]
    nearest_neighbors: List[str]
    timestamp: Optional[datetime] = None


class AnomalyScorer:
    """Distance-based anomaly scorer using k-NN distances."""
    
    def __init__(
        self,
        k_neighbors: int = 5,
        threshold_percentile: float = 95,
        min_distance_threshold: float = 0.5
    ):
        self.k_neighbors = k_neighbors
        self.threshold_percentile = threshold_percentile
        self.min_distance_threshold = min_distance_threshold
        self.baseline_distances: List[float] = []
        self.threshold: Optional[float] = None
    
    def fit_baseline(self, distances: List[float]) -> None:
        """Fit baseline from normal log distances."""
        self.baseline_distances = distances
        self.threshold = np.percentile(distances, self.threshold_percentile)
        print(f"Baseline fitted. Threshold: {self.threshold:.4f}")
    
    def compute_score(
        self,
        distances: np.ndarray,
        log_id: str,
        neighbor_ids: List[str],
        timestamp: Optional[datetime] = None
    ) -> AnomalyScore:
        """Compute anomaly score for a single log."""
        valid_distances = distances[distances >= 0]
        
        if len(valid_distances) == 0:
            return AnomalyScore(
                log_id=log_id, score=1.0, is_anomaly=True,
                confidence=1.0, distances=[], nearest_neighbors=[],
                timestamp=timestamp
            )
        
        mean_distance = float(np.mean(valid_distances))
        
        # Normalize score between 0 and 1
        if self.threshold is not None:
            score = min(mean_distance / (2 * self.threshold), 1.0)
        else:
            score = min(mean_distance / self.min_distance_threshold, 1.0)
        
        is_anomaly = (
            mean_distance > (self.threshold or self.min_distance_threshold)
        )
        
        # Calculate confidence
        if self.baseline_distances:
            z_score = abs(mean_distance - np.mean(self.baseline_distances)) / max(np.std(self.baseline_distances), 0.01)
            confidence = min(z_score / 3, 1.0)
        else:
            confidence = score
        
        return AnomalyScore(
            log_id=log_id, score=score, is_anomaly=is_anomaly,
            confidence=confidence, distances=valid_distances.tolist(),
            nearest_neighbors=neighbor_ids, timestamp=timestamp
        )
    
    def batch_score(
        self,
        all_distances: np.ndarray,
        log_ids: List[str],
        all_neighbor_ids: List[List[str]],
        timestamps: Optional[List[datetime]] = None
    ) -> List[AnomalyScore]:
        """Compute anomaly scores for multiple logs."""
        scores = []
        for i, log_id in enumerate(log_ids):
            ts = timestamps[i] if timestamps else None
            score = self.compute_score(
                all_distances[i], log_id, all_neighbor_ids[i], ts
            )
            scores.append(score)
        return scores
    
    def get_anomalies(
        self,
        scores: List[AnomalyScore],
        min_confidence: float = 0.5
    ) -> List[AnomalyScore]:
        """Filter and sort anomalies by score."""
        anomalies = [
            s for s in scores 
            if s.is_anomaly and s.confidence >= min_confidence
        ]
        return sorted(anomalies, key=lambda x: x.score, reverse=True)
