"""
Tests for src.core.router — input routing logic.
"""

import pytest

from src.core.router import InputRouter
from src.core.schemas import AnalysisEngine, DetectionRequest


# Simulated set of known CIC-IDS2017 features (subset for testing)
KNOWN_FEATURES = {
    "destination_port",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "fwd_packet_length_max",
    "bwd_packet_length_max",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "fwd_iat_total",
}


@pytest.fixture
def router():
    return InputRouter(known_feature_columns=KNOWN_FEATURES, min_feature_threshold=10)


class TestInputRouter:
    def test_log_only(self, router):
        req = DetectionRequest(
            event_id="r1", log_text="Failed login from 10.0.0.1"
        )
        engines = router.route(req)
        assert engines == [AnalysisEngine.SEMANTIC]

    def test_flow_with_enough_features(self, router):
        features = {col: 1.0 for col in list(KNOWN_FEATURES)[:10]}
        req = DetectionRequest(event_id="r2", flow_features=features)
        engines = router.route(req)
        assert AnalysisEngine.STRUCTURED in engines

    def test_flow_with_too_few_features(self, router):
        features = {col: 1.0 for col in list(KNOWN_FEATURES)[:5]}
        req = DetectionRequest(event_id="r3", flow_features=features)
        with pytest.raises(ValueError, match="does not match"):
            router.route(req)

    def test_hybrid(self, router):
        features = {col: 1.0 for col in list(KNOWN_FEATURES)[:10]}
        req = DetectionRequest(
            event_id="r4",
            log_text="Suspicious traffic detected",
            flow_features=features,
        )
        engines = router.route(req)
        assert AnalysisEngine.STRUCTURED in engines
        assert AnalysisEngine.SEMANTIC in engines

    def test_unknown_features_ignored(self, router):
        """Features not in the known set don't count toward the threshold."""
        features = {f"unknown_{i}": 1.0 for i in range(20)}
        req = DetectionRequest(
            event_id="r5",
            log_text="fallback to semantic",
            flow_features=features,
        )
        engines = router.route(req)
        # Only semantic should fire because none of the features are recognized
        assert engines == [AnalysisEngine.SEMANTIC]

    def test_mixed_known_unknown(self, router):
        """Some known + some unknown. Only known ones count."""
        features = {col: 1.0 for col in list(KNOWN_FEATURES)[:10]}
        features["totally_fake_col"] = 999.0
        req = DetectionRequest(event_id="r6", flow_features=features)
        engines = router.route(req)
        assert AnalysisEngine.STRUCTURED in engines
