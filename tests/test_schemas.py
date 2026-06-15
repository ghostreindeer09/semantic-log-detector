"""
Tests for src.core.schemas — shared data contracts.
"""

import pytest
from datetime import datetime

from src.core.schemas import (
    AnalysisEngine,
    AnalysisResult,
    DetectionRequest,
    InputType,
    MitreMapping,
    RuleMatch,
    Severity,
    SOCAlert,
)


# ------------------------------------------------------------------ #
#  DetectionRequest                                                    #
# ------------------------------------------------------------------ #


class TestDetectionRequest:
    """Tests for DetectionRequest validation and routing."""

    def test_log_only(self):
        req = DetectionRequest(
            event_id="e1", log_text="Failed login for root"
        )
        assert req.input_type == InputType.LOG
        assert req.flow_features is None

    def test_flow_only(self):
        req = DetectionRequest(
            event_id="e2", flow_features={"destination_port": 443.0}
        )
        assert req.input_type == InputType.FLOW
        assert req.log_text is None

    def test_hybrid(self):
        req = DetectionRequest(
            event_id="e3",
            log_text="Suspicious traffic",
            flow_features={"destination_port": 80.0},
        )
        assert req.input_type == InputType.HYBRID

    def test_empty_rejected(self):
        with pytest.raises(Exception):
            DetectionRequest(event_id="e4")

    def test_blank_log_text_is_log(self):
        """A whitespace-only log_text should be treated as LOG (not HYBRID)."""
        req = DetectionRequest(event_id="e5", log_text="   ")
        # The model_validator passes because log_text is not None,
        # but input_type should not be HYBRID since there are no features.
        assert req.input_type == InputType.LOG

    def test_empty_flow_features_is_log(self):
        req = DetectionRequest(
            event_id="e6", log_text="hello", flow_features={}
        )
        assert req.input_type == InputType.LOG

    def test_metadata_defaults_to_dict(self):
        req = DetectionRequest(event_id="e7", log_text="x")
        assert req.metadata == {}

    def test_event_id_too_long(self):
        with pytest.raises(Exception):
            DetectionRequest(event_id="x" * 300, log_text="y")

    def test_log_text_max_length(self):
        with pytest.raises(Exception):
            DetectionRequest(event_id="e8", log_text="x" * 10_001)


# ------------------------------------------------------------------ #
#  AnalysisResult                                                      #
# ------------------------------------------------------------------ #


class TestAnalysisResult:
    def test_basic(self):
        r = AnalysisResult(
            engine=AnalysisEngine.STRUCTURED,
            is_anomaly=True,
            confidence=0.95,
            raw_score=0.95,
            predicted_class="DDoS",
        )
        assert r.engine == AnalysisEngine.STRUCTURED
        assert r.is_anomaly is True
        assert r.predicted_class == "DDoS"

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            AnalysisResult(
                engine=AnalysisEngine.SEMANTIC,
                is_anomaly=False,
                confidence=1.5,
                raw_score=0.5,
            )


# ------------------------------------------------------------------ #
#  SOCAlert                                                            #
# ------------------------------------------------------------------ #


class TestSOCAlert:
    def test_basic_construction(self):
        alert = SOCAlert(
            event_id="e1",
            severity=Severity.HIGH,
            is_threat=True,
            threat_score=0.85,
            processing_time_ms=12.5,
            raw_input_type=InputType.LOG,
        )
        assert alert.is_threat is True
        assert alert.alert_id  # auto-generated
        assert len(alert.analyses) == 0

    def test_severity_values(self):
        for s in ["critical", "high", "medium", "low", "info"]:
            assert Severity(s).value == s


# ------------------------------------------------------------------ #
#  MitreMapping                                                        #
# ------------------------------------------------------------------ #


class TestMitreMapping:
    def test_empty(self):
        m = MitreMapping()
        assert m.technique_id is None
        assert m.confidence == 0.0

    def test_populated(self):
        m = MitreMapping(
            technique_id="T1110",
            technique_name="Brute Force",
            tactic="Credential Access",
            confidence=0.95,
        )
        assert m.technique_id == "T1110"
