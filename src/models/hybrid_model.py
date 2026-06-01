"""
Hybrid Anomaly Detection Pipeline.
Combines Sentence-BERT semantic embeddings with Isolation Forest
anomaly scoring for log-based threat detection.
"""
import json
import os
import logging
import pickle
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HybridAnomalyPipeline:

    def __init__(self):
        self.is_fitted = False
        self._embedder = None
        self._anomaly_model = None
        self._scaler = None
        self._threshold = 0.5

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading Sentence-BERT model...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-BERT loaded.")
        return self._embedder

    def _encode(self, texts):
        embedder = self._get_embedder()
        return embedder.encode(texts, show_progress_bar=True, batch_size=64)

    def _clean_log(self, log_text):
        return log_text.strip().lower()

    def fit(self, normal_logs, log_ids=None, metadata=None):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        logger.info("Fitting pipeline on %d normal logs...", len(normal_logs))
        cleaned = [self._clean_log(t) for t in normal_logs]
        embeddings = self._encode(cleaned)

        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(embeddings)

        self._anomaly_model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self._anomaly_model.fit(scaled)

        scores = self._anomaly_model.decision_function(scaled)
        anomaly_scores = -scores
        self._threshold = float(np.percentile(anomaly_scores, 95))
        logger.info("Threshold calibrated at: %.4f", self._threshold)

        self.is_fitted = True
        logger.info("Pipeline fitting complete.")
        return self

    def detect(self, log_text, metadata=None):
        if not self.is_fitted:
            return {"score": 0.0, "is_anomaly": False, "embedding": [],
                    "cleaned_log": log_text, "explanation": None}

        cleaned = self._clean_log(log_text)
        embedding = self._encode([cleaned])[0]
        scaled = self._scaler.transform([embedding])

        raw_score = self._anomaly_model.decision_function(scaled)[0]
        anomaly_score = float(-raw_score)
        is_anomaly = anomaly_score >= self._threshold

        explanation = None
        if is_anomaly:
            explanation = type("Explanation", (), {
                "summary": (
                    f"Anomaly score {anomaly_score:.4f} exceeds threshold "
                    f"{self._threshold:.4f}. Log pattern deviates from normal baseline."
                )
            })()

        return {
            "score": anomaly_score,
            "is_anomaly": is_anomaly,
            "embedding": embedding.tolist(),
            "cleaned_log": cleaned,
            "explanation": explanation,
        }

    def save(self, model_path):
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline.")
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, "anomaly_model.pkl"), "wb") as f:
            pickle.dump(self._anomaly_model, f)
        with open(os.path.join(model_path, "scaler.pkl"), "wb") as f:
            pickle.dump(self._scaler, f)
        meta = {"threshold": self._threshold}
        with open(os.path.join(model_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Pipeline saved to %s", model_path)

    @classmethod
    def load(cls, model_path):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        meta_path = os.path.join(model_path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")
        instance = cls()
        with open(os.path.join(model_path, "anomaly_model.pkl"), "rb") as f:
            instance._anomaly_model = pickle.load(f)
        with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
            instance._scaler = pickle.load(f)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        instance._threshold = meta.get("threshold", 0.5)
        instance.is_fitted = True
        logger.info("HybridAnomalyPipeline loaded from %s", model_path)
        return instance
