#!/usr/bin/env python3
"""
Enhanced Train Model - Combines semantic embeddings with metadata features.
"""

import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.siem_dataset import SIEMDataLoader
from src.preprocessor import LogPreprocessor
from src.encoder import LogEncoder
from src.anomaly_scorer import AnomalyScorer
from src.explanation_engine import ExplanationEngine
from src.time_aware import TimeAwareModule

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


class HybridAnomalyPipeline:
    """
    Hybrid pipeline combining:
    1. Semantic embeddings (Sentence-BERT)
    2. Statistical features (risk_score, severity)
    3. Isolation Forest for outlier detection
    """
    
    SEVERITY_SCORES = {
        'info': 0.0, 'low': 0.2, 'medium': 0.4,
        'high': 0.6, 'critical': 0.8, 'emergency': 1.0
    }
    
    def __init__(self, k_neighbors: int = 5, semantic_weight: float = 0.4):
        self.k_neighbors = k_neighbors
        self.semantic_weight = semantic_weight  # Weight for semantic vs metadata
        self.preprocessor = LogPreprocessor()
        self.encoder = None
        self.nn_model = None
        self.iso_forest = None
        self.scaler = StandardScaler()
        self.embeddings = None
        self.ids = []
        self.metadata_list = []
        self.texts = {}
        self.explanation_engine = ExplanationEngine()
        self.time_aware = TimeAwareModule()
        self.is_fitted = False
        self.baseline_mean = 0
        self.baseline_std = 1
    
    def _extract_features(self, metadata: dict) -> np.ndarray:
        """Extract numerical features from metadata."""
        risk_score = metadata.get('risk_score', 50) / 100.0
        severity = self.SEVERITY_SCORES.get(metadata.get('severity', 'info'), 0.0)
        return np.array([risk_score, severity])
    
    def fit(self, logs: list, ids: list = None, metadata: list = None):
        """Train on normal logs with metadata."""
        print(f"\n🎯 Training on {len(logs)} logs...")
        
        # Initialize encoder
        if self.encoder is None:
            print("Loading Sentence-BERT model...")
            self.encoder = LogEncoder(model_name='all-MiniLM-L6-v2', device='cpu')
        
        # Preprocess
        print("Preprocessing logs...")
        parsed = self.preprocessor.parse_logs(logs)
        cleaned = [p.cleaned_text for p in parsed]
        
        # Store IDs and metadata
        self.ids = ids if ids else [p.log_id for p in parsed]
        self.metadata_list = metadata if metadata else [{}] * len(logs)
        for i, p in enumerate(parsed):
            self.texts[self.ids[i]] = p.cleaned_text
        
        # Encode semantically
        print("Encoding logs...")
        self.embeddings = self.encoder.encode(cleaned, show_progress=True)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float64)
        print(f"Semantic embeddings shape: {self.embeddings.shape}")
        
        # Extract metadata features
        print("Extracting metadata features...")
        meta_features = np.array([
            self._extract_features(m) for m in self.metadata_list
        ])
        self.scaler.fit(meta_features)
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Combine features
        combined = np.hstack([
            self.embeddings * self.semantic_weight,
            meta_features_scaled * (1 - self.semantic_weight)
        ])
        
        # Fit k-NN on combined features
        print("Fitting k-NN model...")
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.k_neighbors + 1, len(combined)),
            metric='euclidean',
            algorithm='auto'
        )
        self.nn_model.fit(combined)
        
        # Fit Isolation Forest for outlier detection
        print("Fitting Isolation Forest...")
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.iso_forest.fit(combined)
        
        # Compute baseline
        print("Computing baseline distances...")
        distances, _ = self.nn_model.kneighbors(combined)
        baseline = distances[:, 1:].mean(axis=1)
        self.baseline_mean = baseline.mean()
        self.baseline_std = baseline.std()
        
        self.is_fitted = True
        print(f"✅ Training complete!")
        print(f"   Baseline mean distance: {self.baseline_mean:.4f}")
        print(f"   Baseline std: {self.baseline_std:.4f}")
    
    def detect(self, log: str, metadata: dict = None):
        """Detect anomaly in a single log."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        metadata = metadata or {}
        
        # Preprocess
        parsed = self.preprocessor.parse_log(log)
        
        # Encode
        embedding = self.encoder.encode(parsed.cleaned_text)
        embedding = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        
        # Extract metadata features
        meta_feat = self._extract_features(metadata).reshape(1, -1)
        meta_feat_scaled = self.scaler.transform(meta_feat)
        
        # Combine features
        combined = np.hstack([
            embedding * self.semantic_weight,
            meta_feat_scaled * (1 - self.semantic_weight)
        ])
        
        # k-NN distance
        distances, indices = self.nn_model.kneighbors(combined)
        mean_distance = distances[0, 1:].mean() if len(distances[0]) > 1 else distances[0, 0]
        
        # Isolation Forest score (-1 = outlier, 1 = inlier)
        iso_score = self.iso_forest.decision_function(combined)[0]
        iso_pred = self.iso_forest.predict(combined)[0]
        
        # Combine scores
        distance_score = (mean_distance - self.baseline_mean) / max(self.baseline_std, 0.01)
        normalized_iso = -iso_score  # Flip so higher = more anomalous
        
        # Weighted combination
        final_score = 0.5 * min(max(distance_score / 3, 0), 1) + 0.5 * min(max(normalized_iso + 0.5, 0), 1)
        final_score = min(final_score, 1.0)
        
        # Determine if anomaly
        is_anomaly = (
            iso_pred == -1 or  # Isolation Forest says outlier
            distance_score > 2 or  # >2 std from mean
            metadata.get('severity', '') in {'high', 'critical', 'emergency'}
        )
        
        neighbor_ids = [self.ids[i] for i in indices[0]]
        
        # Generate explanation
        explanation = None
        if is_anomaly:
            explanation = self.explanation_engine.explain(
                parsed.log_id, parsed.cleaned_text, final_score,
                neighbor_ids, distances[0].tolist()
            )
        
        return {
            'log_id': parsed.log_id,
            'cleaned_log': parsed.cleaned_text,
            'score': final_score,
            'is_anomaly': is_anomaly,
            'distance_score': distance_score,
            'iso_score': iso_score,
            'neighbors': neighbor_ids[:3],
            'explanation': explanation
        }
    
    def detect_batch(self, logs: list, metadata_list: list = None):
        """Detect anomalies in batch."""
        metadata_list = metadata_list or [{}] * len(logs)
        return [self.detect(log, meta) for log, meta in zip(logs, metadata_list)]

    def save(self, path: str):
        """Save the pipeline to disk."""
        import pickle
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # We don't save the encoder model itself, just the config to reload it
        # The nn_model, iso_forest, and scaler need to be pickled
        state = {
            'k_neighbors': self.k_neighbors,
            'semantic_weight': self.semantic_weight,
            'ids': self.ids,
            'metadata_list': self.metadata_list,
            'texts': self.texts,
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'is_fitted': self.is_fitted
        }
        
        # Save state
        with open(f"{path}.state", 'wb') as f:
            pickle.dump(state, f)
            
        # Save sklearn models
        with open(f"{path}.models", 'wb') as f:
            pickle.dump({
                'nn_model': self.nn_model,
                'iso_forest': self.iso_forest,
                'scaler': self.scaler
            }, f)
            
        print(f"Pipeline saved to {path} (.state and .models)")

    @classmethod
    def load(cls, path: str):
        """Load the pipeline from disk."""
        import pickle
        
        # Load state
        with open(f"{path}.state", 'rb') as f:
            state = pickle.load(f)
            
        # Create instance
        pipeline = cls(
            k_neighbors=state['k_neighbors'],
            semantic_weight=state['semantic_weight']
        )
        
        # Restore state
        pipeline.ids = state['ids']
        pipeline.metadata_list = state['metadata_list']
        pipeline.texts = state['texts']
        pipeline.baseline_mean = state['baseline_mean']
        pipeline.baseline_std = state['baseline_std']
        pipeline.is_fitted = state['is_fitted']
        
        # Load models
        with open(f"{path}.models", 'rb') as f:
            models = pickle.load(f)
            pipeline.nn_model = models['nn_model']
            pipeline.iso_forest = models['iso_forest']
            pipeline.scaler = models['scaler']
        
        # Initialize encoder (it wasn't pickled)
        print("Loading Sentence-BERT model...")
        pipeline.encoder = LogEncoder(model_name='all-MiniLM-L6-v2', device='cpu')
        
        print(f"Pipeline loaded from {path}")
        return pipeline


def main():
    print("=" * 60)
    print("🔍 Hybrid Semantic Log Anomaly Detection")
    print("   Using: Sentence-BERT + Isolation Forest + k-NN")
    print("=" * 60)
    
    # Load dataset
    print("\n📦 Loading Advanced SIEM Dataset...")
    loader = SIEMDataLoader(max_samples=3000)
    loader.load()
    
    stats = loader.get_stats()
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total records: {stats['total_records']}")
    print(f"   Normal logs: {stats['normal_count']}")
    print(f"   Anomaly logs: {stats['anomaly_count']}")
    
    # Get full metadata
    all_metadata = loader.get_metadata()
    
    # Split data
    df = loader.df
    normal_mask = ~df['is_anomaly']
    anomaly_mask = df['is_anomaly']
    
    normal_logs = df.loc[normal_mask, 'description'].tolist()
    normal_meta = [m for m, is_anom in zip(all_metadata, df['is_anomaly']) if not is_anom]
    normal_ids = df.loc[normal_mask, 'event_id'].tolist()
    
    anomaly_logs = df.loc[anomaly_mask, 'description'].tolist()
    anomaly_meta = [m for m, is_anom in zip(all_metadata, df['is_anomaly']) if is_anom]
    
    print(f"\n   Normal logs for training: {len(normal_logs)}")
    print(f"   Anomaly logs for testing: {len(anomaly_logs)}")
    
    # Initialize pipeline
    pipeline = HybridAnomalyPipeline(k_neighbors=5, semantic_weight=0.6)
    
    # Train on normal logs
    pipeline.fit(normal_logs, normal_ids, normal_meta)
    
    # Test on normal logs
    print("\n✅ Testing on normal logs...")
    tp_normal = 0
    test_normal = list(zip(normal_logs[:100], normal_meta[:100]))
    for log, meta in test_normal:
        result = pipeline.detect(log, meta)
        if not result['is_anomaly']:
            tp_normal += 1
    print(f"   True Negatives: {tp_normal}/100 ({tp_normal}%)")
    
    # Test on anomaly logs
    print("\n🚨 Testing on anomaly logs...")
    tp_anomaly = 0
    detected = []
    test_anomaly = list(zip(anomaly_logs[:100], anomaly_meta[:100]))
    for log, meta in test_anomaly:
        result = pipeline.detect(log, meta)
        if result['is_anomaly']:
            tp_anomaly += 1
            detected.append((result, meta))
    print(f"   True Positives: {tp_anomaly}/100 ({tp_anomaly}%)")
    
    # Show samples
    if detected:
        print("\n📋 Sample Detections:")
        for i, (result, meta) in enumerate(detected[:3]):
            print(f"\n   --- Anomaly {i+1} ---")
            print(f"   Severity: {meta.get('severity', 'N/A')}")
            print(f"   Risk Score: {meta.get('risk_score', 'N/A')}")
            print(f"   Event Type: {meta.get('event_type', 'N/A')}")
            print(f"   Detection Score: {result['score']:.2%}")
            print(f"   Log: {result['cleaned_log'][:80]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Final Results")
    print("=" * 60)
    accuracy = (tp_normal + tp_anomaly) / 200
    print(f"   True Negative Rate: {tp_normal}%")
    print(f"   True Positive Rate: {tp_anomaly}%")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print("\n✨ Training and evaluation completed!")
    
    # Save model
    pipeline.save("models/siem_model")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
