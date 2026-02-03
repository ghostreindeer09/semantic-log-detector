"""
Anomaly Detection Pipeline - Main orchestration module.
"""

import os
import yaml
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .preprocessor import LogPreprocessor, ParsedLog
from .encoder import LogEncoder
from .vector_db import VectorDatabase
from .similarity_search import SimilaritySearcher
from .anomaly_scorer import AnomalyScorer, AnomalyScore
from .explanation_engine import ExplanationEngine, Explanation
from .time_aware import TimeAwareModule


@dataclass
class DetectionResult:
    """Complete detection result for a log."""
    log_id: str
    raw_log: str
    cleaned_log: str
    anomaly_score: AnomalyScore
    explanation: Optional[Explanation]
    temporal_context: Optional[dict]


class AnomalyDetectionPipeline:
    """Main pipeline orchestrating all components."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._initialize_components()
        self.is_fitted = False
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        default_config = {
            'model': {'encoder': 'all-MiniLM-L6-v2', 'embedding_dim': 384, 'device': 'cpu'},
            'vector_db': {'index_type': 'Flat', 'nlist': 100, 'metric': 'L2'},
            'anomaly': {'k_neighbors': 5, 'threshold_percentile': 95, 'min_distance_threshold': 0.5},
            'time_aware': {'window_size': 100, 'temporal_weight': 0.3},
            'preprocessing': {'max_log_length': 512, 'remove_timestamps': True, 'remove_ips': True}
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return default_config
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        cfg = self.config
        
        self.preprocessor = LogPreprocessor(
            max_length=cfg['preprocessing']['max_log_length'],
            remove_timestamps=cfg['preprocessing']['remove_timestamps'],
            remove_ips=cfg['preprocessing']['remove_ips']
        )
        
        self.encoder = LogEncoder(
            model_name=cfg['model']['encoder'],
            device=cfg['model']['device']
        )
        
        self.vector_db = VectorDatabase(
            embedding_dim=cfg['model']['embedding_dim'],
            index_type=cfg['vector_db']['index_type'],
            metric=cfg['vector_db']['metric']
        )
        
        self.similarity_searcher = SimilaritySearcher(
            vector_db=self.vector_db,
            encoder=self.encoder,
            k_neighbors=cfg['anomaly']['k_neighbors']
        )
        
        self.anomaly_scorer = AnomalyScorer(
            k_neighbors=cfg['anomaly']['k_neighbors'],
            threshold_percentile=cfg['anomaly']['threshold_percentile'],
            min_distance_threshold=cfg['anomaly']['min_distance_threshold']
        )
        
        self.explanation_engine = ExplanationEngine(vector_db=self.vector_db)
        
        self.time_aware = TimeAwareModule(
            window_size=cfg['time_aware']['window_size'],
            temporal_weight=cfg['time_aware']['temporal_weight']
        )
        
        self.log_texts: Dict[str, str] = {}
    
    def fit(self, normal_logs: List[str], source: str = "training") -> None:
        """Train the model on normal (non-anomalous) logs."""
        print(f"Fitting pipeline with {len(normal_logs)} normal logs...")
        
        # Preprocess
        parsed_logs = self.preprocessor.parse_logs(normal_logs, source)
        cleaned_texts = [log.cleaned_text for log in parsed_logs]
        
        # Encode
        print("Encoding logs...")
        embeddings = self.encoder.encode(cleaned_texts, show_progress=True)
        
        # Store in vector database
        ids = [log.log_id for log in parsed_logs]
        metadata = [{'cleaned_text': log.cleaned_text, 'source': source} for log in parsed_logs]
        timestamps = [log.timestamp for log in parsed_logs]
        
        # Train the index if needed (for IVF indices)
        if not self.vector_db.is_trained:
            self.vector_db.train(embeddings)
        
        self.vector_db.add(embeddings, ids, metadata, timestamps)
        
        # Store texts
        for log in parsed_logs:
            self.log_texts[log.log_id] = log.cleaned_text
        
        # Compute baseline distances
        print("Computing baseline distances...")
        search_result = self.vector_db.search(embeddings, k=self.config['anomaly']['k_neighbors'] + 1)
        
        baseline_distances = []
        for i, distances in enumerate(search_result.distances):
            valid = distances[distances > 0]  # Exclude self
            if len(valid) > 0:
                baseline_distances.append(float(np.mean(valid)))
        
        self.anomaly_scorer.fit_baseline(baseline_distances)
        self.is_fitted = True
        print(f"Pipeline fitted. Vector DB size: {self.vector_db.size()}")
    
    def detect(self, log: str, source: str = "inference") -> DetectionResult:
        """Detect anomaly in a single log."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before detection")
        
        # Preprocess
        parsed = self.preprocessor.parse_log(log, source)
        
        # Encode
        embedding = self.encoder.encode(parsed.cleaned_text)
        
        # Search
        search_result = self.vector_db.search(embedding, k=self.config['anomaly']['k_neighbors'])
        
        # Score
        anomaly_score = self.anomaly_scorer.compute_score(
            search_result.distances[0],
            parsed.log_id,
            search_result.ids[0],
            parsed.timestamp
        )
        
        # Temporal adjustment
        temporal_ctx = self.time_aware.adjust_score(
            parsed.log_id, anomaly_score.score, parsed.timestamp
        )
        anomaly_score.score = temporal_ctx.time_adjusted_score
        
        # Generate explanation if anomaly
        explanation = None
        if anomaly_score.is_anomaly:
            explanation = self.explanation_engine.explain(
                parsed.log_id, parsed.cleaned_text, anomaly_score.score,
                anomaly_score.nearest_neighbors, anomaly_score.distances
            )
        
        return DetectionResult(
            log_id=parsed.log_id, raw_log=log, cleaned_log=parsed.cleaned_text,
            anomaly_score=anomaly_score, explanation=explanation,
            temporal_context={'score': temporal_ctx.time_adjusted_score, 'weight': temporal_ctx.temporal_weight}
        )
    
    def detect_batch(self, logs: List[str], source: str = "inference") -> List[DetectionResult]:
        """Detect anomalies in batch."""
        return [self.detect(log, source) for log in logs]
    
    def save(self, path: str) -> None:
        """Save the pipeline state."""
        os.makedirs(path, exist_ok=True)
        self.vector_db.save(os.path.join(path, 'vector_db'))
        print(f"Pipeline saved to {path}")
    
    def load(self, path: str) -> None:
        """Load pipeline state."""
        self.vector_db = VectorDatabase.load(os.path.join(path, 'vector_db'))
        self.similarity_searcher.vector_db = self.vector_db
        self.explanation_engine.vector_db = self.vector_db
        self.is_fitted = True
        print(f"Pipeline loaded from {path}")
