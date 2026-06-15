"""
Alert Builder — fuses analysis results into a unified SOCAlert.

Responsible for:
- Score fusion when multiple engines run on the same event
- Severity calculation from fused score + rule matches
- MITRE ATT&CK mapping via the shared MitreMapper
- Rule-engine evaluation
- Drift-alert aggregation
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from .schemas import (
    AnalysisEngine,
    AnalysisResult,
    DetectionRequest,
    MitreMapping,
    RuleMatch,
    SOCAlert,
    Severity,
)

if TYPE_CHECKING:
    from ..mitre.mapper import MitreMapper
    from ..monitoring.drift import DriftMonitor
    from ..rules.engine import RuleEngine

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Score fusion weights                                                #
# ------------------------------------------------------------------ #
# Structured model is supervised + calibrated → higher trust.
# Semantic model is unsupervised anomaly detection → complementary signal.
_STRUCTURED_WEIGHT = 0.70
_SEMANTIC_WEIGHT = 0.30

# Hard-override thresholds
_STRUCTURED_HIGH_CONF = 0.95
_SEMANTIC_HIGH_CONF = 0.90


def fuse_scores(results: List[AnalysisResult]) -> float:
    """
    Weighted fusion of confidence scores from one or more engines.

    When both structured and semantic engines fire:
    - Base score is a weighted average.
    - If either engine has very high confidence, that signal dominates.

    When only one engine fires, its confidence is used directly.
    """
    if not results:
        return 0.0

    if len(results) == 1:
        return results[0].confidence

    structured = next(
        (r for r in results if r.engine == AnalysisEngine.STRUCTURED), None
    )
    semantic = next(
        (r for r in results if r.engine == AnalysisEngine.SEMANTIC), None
    )

    if structured and semantic:
        base = (
            structured.confidence * _STRUCTURED_WEIGHT
            + semantic.confidence * _SEMANTIC_WEIGHT
        )
        # Hard-override: trust a very confident supervised classifier
        if structured.confidence >= _STRUCTURED_HIGH_CONF:
            return structured.confidence
        if semantic.confidence >= _SEMANTIC_HIGH_CONF:
            return max(base, semantic.confidence)
        return base

    # Fallback: average all available
    return sum(r.confidence for r in results) / len(results)


def compute_severity(
    threat_score: float,
    rule_matches: List[RuleMatch],
) -> Severity:
    """
    Compute SOC alert severity from the fused threat score and any
    deterministic rule matches.
    """
    # Any rule match escalates to at least MEDIUM
    if rule_matches:
        if threat_score >= 0.7:
            return Severity.CRITICAL
        return Severity.HIGH

    if threat_score >= 0.9:
        return Severity.CRITICAL
    if threat_score >= 0.7:
        return Severity.HIGH
    if threat_score >= 0.5:
        return Severity.MEDIUM
    if threat_score >= 0.3:
        return Severity.LOW
    return Severity.INFO


# ------------------------------------------------------------------ #
#  AlertBuilder                                                        #
# ------------------------------------------------------------------ #

class AlertBuilder:
    """
    Constructs a ``SOCAlert`` from raw analysis results plus enrichment.
    """

    def __init__(
        self,
        mitre_mapper: "MitreMapper",
        rule_engine: "RuleEngine",
        drift_monitor: "DriftMonitor",
    ):
        self.mitre_mapper = mitre_mapper
        self.rule_engine = rule_engine
        self.drift_monitor = drift_monitor

    # ---------------------------------------------------------------- #

    def build(
        self,
        request: DetectionRequest,
        analysis_results: List[AnalysisResult],
        processing_time_ms: float,
    ) -> SOCAlert:
        """
        Fuse analysis results + enrichment into a single ``SOCAlert``.
        """
        # ── 1. Rule engine ──
        rule_matches = self._run_rules(request)

        # ── 2. Score fusion ──
        threat_score = fuse_scores(analysis_results)

        # If a rule matched, bump the minimum score
        if rule_matches:
            threat_score = max(threat_score, 0.6)

        is_threat = (
            threat_score >= 0.5
            or any(r.is_anomaly for r in analysis_results)
            or len(rule_matches) > 0
        )

        # ── 3. Severity ──
        severity = compute_severity(threat_score, rule_matches)

        # ── 4. Extract best predicted class (from structured engine) ──
        predicted_class: Optional[str] = None
        threat_category: Optional[str] = None
        for r in analysis_results:
            if r.engine == AnalysisEngine.STRUCTURED and r.predicted_class:
                predicted_class = r.predicted_class
                threat_category = "intrusion"
                break
        if predicted_class is None:
            # Fall back to semantic
            for r in analysis_results:
                if r.engine == AnalysisEngine.SEMANTIC and r.is_anomaly:
                    threat_category = "anomaly"
                    break

        # ── 5. MITRE mapping ──
        mitre = self._map_mitre(
            predicted_class=predicted_class,
            rule_matches=rule_matches,
            log_text=request.log_text or "",
        )

        # ── 6. Drift ──
        drift_alert = self._check_drift(analysis_results)

        # ── 7. Build alert ──
        return SOCAlert(
            event_id=request.event_id,
            timestamp=request.timestamp or datetime.utcnow(),
            severity=severity,
            is_threat=is_threat,
            threat_score=round(threat_score, 4),
            threat_category=threat_category,
            predicted_class=predicted_class,
            analyses=analysis_results,
            mitre=mitre,
            rule_matches=rule_matches,
            drift_alert=drift_alert,
            processing_time_ms=round(processing_time_ms, 2),
            engines_used=[r.engine.value for r in analysis_results],
            source=request.source,
            raw_input_type=request.input_type,
        )

    # ---------------------------------------------------------------- #
    #  Private helpers                                                  #
    # ---------------------------------------------------------------- #

    def _run_rules(self, request: DetectionRequest) -> List[RuleMatch]:
        """Evaluate the rule engine and return any matches."""
        if not request.log_text:
            return []

        result = self.rule_engine.check_rules(
            request.log_text, request.metadata
        )
        if result.get("rule_based_alert"):
            return [
                RuleMatch(
                    rule_id=result["rule_id"],
                    rule_name=result.get("rule_reason", result["rule_id"]),
                    description=result.get("rule_reason", ""),
                )
            ]
        return []

    def _map_mitre(
        self,
        predicted_class: Optional[str],
        rule_matches: List[RuleMatch],
        log_text: str,
    ) -> MitreMapping:
        """Delegate to the MITRE mapper with the best available signal."""
        rule_id = rule_matches[0].rule_id if rule_matches else None
        return self.mitre_mapper.map_detection(
            predicted_class=predicted_class,
            rule_id=rule_id,
            log_text=log_text,
        )

    def _check_drift(self, results: List[AnalysisResult]) -> bool:
        """Check each engine's embedding/feature vector for drift."""
        import numpy as np

        for r in results:
            vector = r.details.get("embedding") or r.details.get("feature_vector")
            if vector is not None:
                try:
                    arr = np.asarray(vector, dtype=np.float32)
                    if self.drift_monitor.add_sample(r.engine.value, arr):
                        return True
                except Exception as exc:
                    logger.warning("Drift check failed for %s: %s", r.engine, exc)
        return False
