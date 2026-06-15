"""
Structured Analyzer — wraps the trained LightGBM IDS model.

Accepts network-flow features from a ``DetectionRequest`` and produces
an ``AnalysisResult`` that can be fused with other engines.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ..core.schemas import AnalysisEngine, AnalysisResult, DetectionRequest

logger = logging.getLogger(__name__)

# CIC-IDS2017 multiclass encoding (index → label)
_DEFAULT_CLASS_MAP: Dict[int, str] = {
    0: "BENIGN",
    1: "Bot",
    2: "Brute Force",
    3: "DDoS",
    4: "DoS",
    5: "Infiltration",
    6: "PortScan",
    7: "Web Attack",
}


class StructuredAnalyzer:
    """
    Adapter that wraps the trained ``StructuredTrainer`` artefacts
    (LightGBM model + StandardScaler) for real-time API serving.

    Does **not** depend on ``StructuredTrainer`` at runtime — it loads
    the serialised model and scaler directly so that the heavy
    ``src.ids`` training code does not need to be imported.
    """

    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._feature_cols: List[str] = []
        self._task: str = "binary"
        self._model_type: str = "lightgbm"
        self._class_map: Dict[int, str] = _DEFAULT_CLASS_MAP
        self._is_ready = False

    # ---------------------------------------------------------------- #
    #  Loading                                                          #
    # ---------------------------------------------------------------- #

    @classmethod
    def load(cls, model_dir: str) -> "StructuredAnalyzer":
        """
        Load a trained structured model from disk.

        Expects the directory to contain:
        - ``structured_model.pkl``
        - ``scaler.pkl``  (optional — kept for compatibility)
        - ``structured_meta.json``

        Falls back to ``outputs/checkpoints`` if the primary dir
        does not contain the expected files.
        """
        instance = cls()

        # Resolve model directory
        meta_path = os.path.join(model_dir, "structured_meta.json")
        model_path = os.path.join(model_dir, "structured_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        if not os.path.exists(meta_path):
            logger.warning(
                "structured_meta.json not found in %s — "
                "model may not be trained yet.",
                model_dir,
            )
            return instance

        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        instance._feature_cols = meta.get("feature_cols", [])
        instance._task = meta.get("task", "binary")
        instance._model_type = meta.get("model_type", "lightgbm")

        # Load model
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                instance._model = pickle.load(f)
        else:
            logger.warning("structured_model.pkl not found in %s", model_dir)
            return instance

        # Load scaler (optional — the model may have been trained without)
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                instance._scaler = pickle.load(f)

        instance._is_ready = True
        logger.info(
            "StructuredAnalyzer loaded from %s (task=%s, features=%d)",
            model_dir,
            instance._task,
            len(instance._feature_cols),
        )
        return instance

    # ---------------------------------------------------------------- #
    #  Analyzer interface                                               #
    # ---------------------------------------------------------------- #

    def analyze(self, request: DetectionRequest) -> AnalysisResult:
        """
        Classify a single network-flow event.

        Extracts ``request.flow_features`` into the expected feature
        vector, runs the LightGBM model, and returns an ``AnalysisResult``.
        """
        if not self._is_ready or request.flow_features is None:
            return AnalysisResult(
                engine=AnalysisEngine.STRUCTURED,
                is_anomaly=False,
                confidence=0.0,
                raw_score=0.0,
                predicted_class=None,
                details={"error": "Model not ready or no flow features"},
            )

        # ── Build feature vector ──
        row: Dict[str, float] = {}
        for col in self._feature_cols:
            row[col] = request.flow_features.get(col, 0.0)

        df = pd.DataFrame([row], columns=self._feature_cols)
        X = df.values.astype(np.float32)

        # Scale if scaler is available
        if self._scaler is not None:
            X = self._scaler.transform(X)
        X[~np.isfinite(X)] = 0.0

        # ── Predict ──
        pred_label = int(self._model.predict(X)[0])
        proba = self._model.predict_proba(X)

        if self._task == "binary":
            # Binary: proba shape (1, 2) → take positive-class probability
            if proba.ndim == 2:
                confidence = float(proba[0, 1])
            else:
                confidence = float(proba[0])
            is_anomaly = pred_label == 1
            predicted_class = "BENIGN" if pred_label == 0 else "Attack"
        else:
            # Multiclass: proba shape (1, n_classes)
            confidence = float(proba[0, pred_label])
            predicted_class = self._class_map.get(pred_label, f"class_{pred_label}")
            is_anomaly = predicted_class != "BENIGN"

        return AnalysisResult(
            engine=AnalysisEngine.STRUCTURED,
            is_anomaly=is_anomaly,
            confidence=confidence,
            raw_score=confidence,  # LightGBM probabilities are calibrated
            predicted_class=predicted_class if is_anomaly else None,
            details={
                "all_probabilities": {
                    self._class_map.get(i, f"class_{i}"): round(float(p), 4)
                    for i, p in enumerate(proba[0])
                },
                "feature_vector": X[0].tolist(),
                "task": self._task,
            },
        )

    def is_ready(self) -> bool:
        return self._is_ready

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self._is_ready,
            "model_type": self._model_type,
            "task": self._task,
            "n_features": len(self._feature_cols),
        }

    @property
    def feature_columns(self) -> Set[str]:
        """Return the set of feature column names the model expects."""
        return set(self._feature_cols)
