"""
Shared schemas for the SOC Detection Engine.

This module defines every data contract in the system:
- DetectionRequest  : unified input accepted by the API
- AnalysisResult    : intermediate output from a single analyzer
- RuleMatch         : output from the rule engine
- MitreMapping      : MITRE ATT&CK mapping for a detection
- SOCAlert          : final unified alert returned by the API

All other modules import these types; none of them define their own
ad-hoc dicts for detection results.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------ #
#  Enums                                                               #
# ------------------------------------------------------------------ #

class AnalysisEngine(str, Enum):
    """Identifies which analysis engine produced a result."""
    STRUCTURED = "structured_ids"
    SEMANTIC = "semantic_anomaly"
    RULE_ENGINE = "rule_engine"


class Severity(str, Enum):
    """SOC alert severity levels (descending)."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class InputType(str, Enum):
    """Describes what kind of data was submitted."""
    FLOW = "flow"
    LOG = "log"
    HYBRID = "hybrid"


# ------------------------------------------------------------------ #
#  Request                                                             #
# ------------------------------------------------------------------ #

class DetectionRequest(BaseModel):
    """
    Unified detection request.

    Accepts structured network-flow features, free-text log lines, or both.
    At least one of ``log_text`` or ``flow_features`` must be provided.
    """

    # Identity
    event_id: str = Field(
        ..., min_length=1, max_length=256,
        description="Unique identifier for the event being analyzed.",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="When the event occurred. Defaults to server receive time.",
    )
    source: Optional[str] = Field(
        default=None, max_length=256,
        description='Origin of the event (e.g. "firewall", "syslog", "auth-service").',
    )

    # Text log input  ➜  Semantic analyzer
    log_text: Optional[str] = Field(
        default=None, max_length=10_000,
        description="Free-text log line for semantic anomaly analysis.",
    )

    # Structured flow input  ➜  IDS analyzer
    flow_features: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Numeric network-flow features (e.g. CIC-IDS2017 schema). "
            "Keys are feature names, values are float measurements."
        ),
    )

    # Metadata for rule engine / enrichment
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata (ip_address, user, hostname, …).",
    )

    @model_validator(mode="after")
    def _at_least_one_input(self) -> "DetectionRequest":
        if not self.log_text and not self.flow_features:
            raise ValueError(
                "Must provide at least one of: log_text, flow_features"
            )
        return self

    @property
    def input_type(self) -> InputType:
        """Classify the input based on which fields are populated."""
        has_flow = self.flow_features is not None and len(self.flow_features) > 0
        has_log = self.log_text is not None and len(self.log_text.strip()) > 0
        if has_flow and has_log:
            return InputType.HYBRID
        if has_flow:
            return InputType.FLOW
        return InputType.LOG


# ------------------------------------------------------------------ #
#  Analysis Result  (intermediate — per engine)                        #
# ------------------------------------------------------------------ #

class AnalysisResult(BaseModel):
    """
    Output from a single analysis engine.

    Both ``StructuredAnalyzer`` and ``SemanticAnalyzer`` produce this type.
    The ``AlertBuilder`` fuses one or more of these into a ``SOCAlert``.
    """

    engine: AnalysisEngine
    is_anomaly: bool
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Calibrated / normalised confidence score.",
    )
    raw_score: float = Field(
        ...,
        description="Engine-native score (LightGBM probability, IsoForest decision, …).",
    )
    predicted_class: Optional[str] = Field(
        default=None,
        description='Attack category (e.g. "DDoS", "DoS") or None for anomaly detection.',
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific details (feature importances, nearest-neighbor info, …).",
    )


# ------------------------------------------------------------------ #
#  Rule Match                                                          #
# ------------------------------------------------------------------ #

class RuleMatch(BaseModel):
    """A single rule-engine match."""

    rule_id: str
    rule_name: str
    description: str
    matched: bool = True


# ------------------------------------------------------------------ #
#  MITRE Mapping                                                       #
# ------------------------------------------------------------------ #

class MitreMapping(BaseModel):
    """MITRE ATT&CK mapping for a detection."""

    technique_id: Optional[str] = None
    technique_name: Optional[str] = None
    tactic: Optional[str] = None
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How confident we are in this mapping.",
    )


# ------------------------------------------------------------------ #
#  SOCAlert  (final unified output)                                    #
# ------------------------------------------------------------------ #

class SOCAlert(BaseModel):
    """
    Unified alert schema — the single output type for all detections.

    Returned by every ``/detect*`` endpoint regardless of which engine(s)
    processed the event.
    """

    # Identity
    alert_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:16],
        description="Unique alert identifier.",
    )
    event_id: str = Field(..., description="Original event ID from request.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the detection was produced.",
    )

    # Verdict
    severity: Severity
    is_threat: bool = Field(
        ..., description="Final binary decision: threat or benign.",
    )
    threat_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fused confidence score across all engines.",
    )

    # Classification
    threat_category: Optional[str] = Field(
        default=None,
        description='High-level category: "intrusion", "anomaly", "policy_violation".',
    )
    predicted_class: Optional[str] = Field(
        default=None,
        description='Specific class from structured model (e.g. "DDoS", "Brute Force").',
    )

    # Per-engine detail
    analyses: List[AnalysisResult] = Field(
        default_factory=list,
        description="Results from each engine that processed this event.",
    )

    # MITRE ATT&CK
    mitre: MitreMapping = Field(
        default_factory=MitreMapping,
        description="Best MITRE mapping for this detection.",
    )

    # Rule matches
    rule_matches: List[RuleMatch] = Field(
        default_factory=list,
        description="Deterministic rule matches (if any).",
    )

    # Operational
    drift_alert: bool = Field(
        default=False,
        description="Whether drift was detected in the engine that processed this event.",
    )
    processing_time_ms: float = Field(
        ..., ge=0.0,
        description="Total server-side processing time in milliseconds.",
    )
    engines_used: List[str] = Field(
        default_factory=list,
        description='Which engines ran (e.g. ["structured_ids", "semantic_anomaly"]).',
    )

    # Source
    source: Optional[str] = None
    raw_input_type: InputType = Field(
        ..., description="Whether the original request contained flow, log, or both.",
    )
