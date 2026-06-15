"""
Input Router — determines which analysis engine(s) should process a request.
"""

from __future__ import annotations

import logging
from typing import List, Set

from .schemas import AnalysisEngine, DetectionRequest

logger = logging.getLogger(__name__)

# Minimum number of recognised flow features required to route
# to the structured IDS analyzer.
_DEFAULT_MIN_FEATURES = 10


class InputRouter:
    """
    Inspects a ``DetectionRequest`` and returns the list of
    ``AnalysisEngine``s that should handle it.

    Routing rules:
    1. If ``flow_features`` are present and contain enough known columns
       → include STRUCTURED.
    2. If ``log_text`` is present and non-empty → include SEMANTIC.
    3. Both can fire simultaneously for hybrid inputs.
    """

    def __init__(
        self,
        known_feature_columns: Set[str],
        min_feature_threshold: int = _DEFAULT_MIN_FEATURES,
    ):
        """
        Args:
            known_feature_columns: The set of feature names the structured
                model was trained on (e.g. the 78 CIC-IDS2017 columns).
            min_feature_threshold: How many of those columns must appear
                in the request for the structured engine to accept it.
        """
        self.known_feature_columns = known_feature_columns
        self.min_feature_threshold = min_feature_threshold

    def route(self, request: DetectionRequest) -> List[AnalysisEngine]:
        """Return the engine(s) that should process *request*."""
        engines: List[AnalysisEngine] = []

        # ── Structured flow analysis ──
        if request.flow_features:
            provided = set(request.flow_features.keys())
            overlap = provided & self.known_feature_columns
            if len(overlap) >= self.min_feature_threshold:
                engines.append(AnalysisEngine.STRUCTURED)
            else:
                logger.debug(
                    "Flow features provided (%d) but only %d recognised "
                    "(need %d). Skipping structured engine.",
                    len(provided),
                    len(overlap),
                    self.min_feature_threshold,
                )

        # ── Semantic log analysis ──
        if request.log_text and request.log_text.strip():
            engines.append(AnalysisEngine.SEMANTIC)

        if not engines:
            raise ValueError(
                "Input does not match any analysis engine. "
                "Provide valid log_text or at least "
                f"{self.min_feature_threshold} recognised flow features."
            )

        logger.debug(
            "Routed event %s → %s",
            request.event_id,
            [e.value for e in engines],
        )
        return engines
