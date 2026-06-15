"""
Semantic Analyzer — wraps the HybridAnomalyPipeline (Sentence-BERT + Isolation Forest).

Accepts free-text log lines from a ``DetectionRequest`` and produces
an ``AnalysisResult`` for downstream fusion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..core.schemas import AnalysisEngine, AnalysisResult, DetectionRequest
from ..models.hybrid_model import HybridAnomalyPipeline

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    Adapter that wraps ``HybridAnomalyPipeline`` behind the uniform
    ``Analyzer`` interface.

    The underlying pipeline uses:
    - Sentence-BERT (all-MiniLM-L6-v2) for embedding log text
    - IsolationForest for anomaly scoring
    - StandardScaler for embedding normalisation
    """

    def __init__(self, pipeline: HybridAnomalyPipeline | None = None) -> None:
        self._pipeline = pipeline or HybridAnomalyPipeline()

    # ---------------------------------------------------------------- #
    #  Loading                                                          #
    # ---------------------------------------------------------------- #

    @classmethod
    def load(cls, model_path: str) -> "SemanticAnalyzer":
        """
        Load a trained HybridAnomalyPipeline from *model_path*.

        If the model directory does not exist or loading fails, returns
        an instance with an unfitted pipeline (``is_ready() == False``).
        """
        try:
            pipeline = HybridAnomalyPipeline.load(model_path)
            logger.info("SemanticAnalyzer loaded from %s", model_path)
            return cls(pipeline)
        except Exception as exc:
            logger.warning(
                "Could not load semantic model from %s: %s. "
                "Starting with unfitted pipeline.",
                model_path,
                exc,
            )
            return cls()

    # ---------------------------------------------------------------- #
    #  Analyzer interface                                               #
    # ---------------------------------------------------------------- #

    def analyze(self, request: DetectionRequest) -> AnalysisResult:
        """
        Score a single log line for anomaly.

        Delegates to ``HybridAnomalyPipeline.detect()`` and normalises
        the Isolation Forest decision-function output into a 0–1
        confidence score.
        """
        if not self.is_ready() or not request.log_text:
            return AnalysisResult(
                engine=AnalysisEngine.SEMANTIC,
                is_anomaly=False,
                confidence=0.0,
                raw_score=0.0,
                predicted_class=None,
                details={"error": "Model not ready or no log text"},
            )

        raw = self._pipeline.detect(request.log_text, request.metadata)

        raw_score = float(raw.get("score", 0.0))
        is_anomaly = bool(raw.get("is_anomaly", False))

        # Normalise raw anomaly score to [0, 1] confidence.
        # The IsolationForest decision function is unbounded; we clamp
        # relative to the calibrated threshold.
        threshold = self._pipeline._threshold or 1.0
        if threshold > 0:
            confidence = min(max(raw_score / (2.0 * threshold), 0.0), 1.0)
        else:
            confidence = min(max(raw_score, 0.0), 1.0)

        # If the model flagged it as anomaly, ensure confidence ≥ 0.5
        if is_anomaly:
            confidence = max(confidence, 0.5)

        # Build explanation snippet
        explanation = None
        if raw.get("explanation"):
            explanation = raw["explanation"].summary

        return AnalysisResult(
            engine=AnalysisEngine.SEMANTIC,
            is_anomaly=is_anomaly,
            confidence=round(confidence, 4),
            raw_score=raw_score,
            predicted_class=None,  # Unsupervised — no class label
            details={
                "cleaned_log": raw.get("cleaned_log", ""),
                "embedding": raw.get("embedding", []),
                "threshold": threshold,
                "explanation": explanation,
            },
        )

    def is_ready(self) -> bool:
        return self._pipeline.is_fitted

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self._pipeline.is_fitted,
            "model": "all-MiniLM-L6-v2 + IsolationForest",
            "threshold": getattr(self._pipeline, "_threshold", None),
        }
