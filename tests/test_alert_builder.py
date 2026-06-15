"""
Tests for src.core.alert_builder — score fusion, severity, and alert construction.
"""

import pytest

from src.core.alert_builder import AlertBuilder, compute_severity, fuse_scores
from src.core.schemas import (
    AnalysisEngine,
    AnalysisResult,
    DetectionRequest,
    RuleMatch,
    Severity,
)
from src.mitre.mapper import MitreMapper
from src.monitoring.drift import DriftMonitor
from src.rules.engine import RuleEngine


# ------------------------------------------------------------------ #
#  Score Fusion                                                        #
# ------------------------------------------------------------------ #


class TestFuseScores:
    def test_single_result(self):
        r = AnalysisResult(
            engine=AnalysisEngine.STRUCTURED,
            is_anomaly=True,
            confidence=0.8,
            raw_score=0.8,
        )
        assert fuse_scores([r]) == 0.8

    def test_empty(self):
        assert fuse_scores([]) == 0.0

    def test_dual_weighted(self):
        s = AnalysisResult(
            engine=AnalysisEngine.STRUCTURED,
            is_anomaly=True,
            confidence=0.6,
            raw_score=0.6,
        )
        m = AnalysisResult(
            engine=AnalysisEngine.SEMANTIC,
            is_anomaly=True,
            confidence=0.4,
            raw_score=0.4,
        )
        fused = fuse_scores([s, m])
        expected = 0.6 * 0.70 + 0.4 * 0.30
        assert abs(fused - expected) < 1e-6

    def test_structured_high_conf_override(self):
        s = AnalysisResult(
            engine=AnalysisEngine.STRUCTURED,
            is_anomaly=True,
            confidence=0.96,
            raw_score=0.96,
        )
        m = AnalysisResult(
            engine=AnalysisEngine.SEMANTIC,
            is_anomaly=False,
            confidence=0.1,
            raw_score=0.1,
        )
        fused = fuse_scores([s, m])
        assert fused == 0.96  # Hard override

    def test_semantic_high_conf_override(self):
        s = AnalysisResult(
            engine=AnalysisEngine.STRUCTURED,
            is_anomaly=False,
            confidence=0.3,
            raw_score=0.3,
        )
        m = AnalysisResult(
            engine=AnalysisEngine.SEMANTIC,
            is_anomaly=True,
            confidence=0.92,
            raw_score=0.92,
        )
        fused = fuse_scores([s, m])
        # max(weighted_avg, 0.92)
        assert fused == 0.92


# ------------------------------------------------------------------ #
#  Severity                                                            #
# ------------------------------------------------------------------ #


class TestComputeSeverity:
    def test_critical(self):
        assert compute_severity(0.95, []) == Severity.CRITICAL

    def test_high(self):
        assert compute_severity(0.75, []) == Severity.HIGH

    def test_medium(self):
        assert compute_severity(0.55, []) == Severity.MEDIUM

    def test_low(self):
        assert compute_severity(0.35, []) == Severity.LOW

    def test_info(self):
        assert compute_severity(0.1, []) == Severity.INFO

    def test_rule_escalates(self):
        rule = RuleMatch(rule_id="R001", rule_name="test", description="test")
        assert compute_severity(0.4, [rule]) == Severity.HIGH

    def test_rule_plus_high_score(self):
        rule = RuleMatch(rule_id="R001", rule_name="test", description="test")
        assert compute_severity(0.8, [rule]) == Severity.CRITICAL


# ------------------------------------------------------------------ #
#  AlertBuilder                                                        #
# ------------------------------------------------------------------ #


class TestAlertBuilder:
    @pytest.fixture
    def builder(self):
        return AlertBuilder(
            mitre_mapper=MitreMapper(),
            rule_engine=RuleEngine(),
            drift_monitor=DriftMonitor(),
        )

    def test_benign_log(self, builder):
        req = DetectionRequest(event_id="a1", log_text="normal activity")
        result = AnalysisResult(
            engine=AnalysisEngine.SEMANTIC,
            is_anomaly=False,
            confidence=0.1,
            raw_score=0.1,
        )
        alert = builder.build(req, [result], processing_time_ms=5.0)
        assert alert.is_threat is False
        assert alert.severity == Severity.INFO
        assert "semantic_anomaly" in alert.engines_used

    def test_attack_with_mitre(self, builder):
        req = DetectionRequest(event_id="a2", log_text="brute force login")
        result = AnalysisResult(
            engine=AnalysisEngine.SEMANTIC,
            is_anomaly=True,
            confidence=0.8,
            raw_score=0.8,
        )
        alert = builder.build(req, [result], processing_time_ms=10.0)
        assert alert.is_threat is True
        # "brute" keyword triggers MITRE mapping
        assert alert.mitre.technique_id == "T1110"

    def test_structured_attack_class_mitre(self, builder):
        req = DetectionRequest(
            event_id="a3",
            flow_features={"destination_port": 80.0},
        )
        result = AnalysisResult(
            engine=AnalysisEngine.STRUCTURED,
            is_anomaly=True,
            confidence=0.9,
            raw_score=0.9,
            predicted_class="DDoS",
        )
        alert = builder.build(req, [result], processing_time_ms=3.0)
        assert alert.predicted_class == "DDoS"
        assert alert.threat_category == "intrusion"
        assert alert.mitre.technique_id == "T1498.001"

    def test_empty_results(self, builder):
        """No engine results → benign alert."""
        req = DetectionRequest(event_id="a4", log_text="nothing here")
        alert = builder.build(req, [], processing_time_ms=1.0)
        assert alert.is_threat is False
        assert len(alert.engines_used) == 0
