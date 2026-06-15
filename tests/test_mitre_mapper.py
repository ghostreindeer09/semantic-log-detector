"""
Tests for src.mitre.mapper — expanded MITRE ATT&CK mapping.
"""

import pytest

from src.core.schemas import MitreMapping
from src.mitre.mapper import MitreMapper


@pytest.fixture
def mapper():
    return MitreMapper()


class TestMitreMapper:
    # ── IDS class mapping (Priority 1) ──

    def test_ddos_class(self, mapper):
        m = mapper.map_detection(predicted_class="DDoS")
        assert m.technique_id == "T1498.001"
        assert m.tactic == "Impact"
        assert m.confidence == 0.9

    def test_dos_class(self, mapper):
        m = mapper.map_detection(predicted_class="DoS")
        assert m.technique_id == "T1499.003"

    def test_portscan_class(self, mapper):
        m = mapper.map_detection(predicted_class="PortScan")
        assert m.technique_id == "T1595.001"
        assert m.tactic == "Reconnaissance"

    def test_brute_force_class(self, mapper):
        m = mapper.map_detection(predicted_class="Brute Force")
        assert m.technique_id == "T1110.001"

    def test_web_attack_class(self, mapper):
        m = mapper.map_detection(predicted_class="Web Attack")
        assert m.technique_id == "T1190"

    def test_bot_class(self, mapper):
        m = mapper.map_detection(predicted_class="Bot")
        assert m.technique_id == "T1071"

    def test_infiltration_class(self, mapper):
        m = mapper.map_detection(predicted_class="Infiltration")
        assert m.technique_id == "T1078"

    # ── Rule ID mapping (Priority 2) ──

    def test_rule_r001(self, mapper):
        m = mapper.map_detection(rule_id="R001")
        assert m.technique_id == "T1110"
        assert m.confidence == 0.95

    def test_rule_r002(self, mapper):
        m = mapper.map_detection(rule_id="R002")
        assert m.technique_id == "T1078"

    # ── IDS class takes priority over rule ID ──

    def test_class_overrides_rule(self, mapper):
        m = mapper.map_detection(predicted_class="DDoS", rule_id="R001")
        assert m.technique_id == "T1498.001"  # Class wins

    # ── Keyword fallback (Priority 3) ──

    def test_brute_keyword(self, mapper):
        m = mapper.map_detection(log_text="Brute force attack detected")
        assert m.technique_id == "T1110"
        assert m.confidence == 0.5

    def test_sudo_keyword(self, mapper):
        m = mapper.map_detection(log_text="sudo: user gained root")
        assert m.technique_id == "T1078"

    def test_powershell_keyword(self, mapper):
        m = mapper.map_detection(log_text="powershell encoded command")
        assert m.technique_id == "T1059"

    def test_scan_keyword(self, mapper):
        m = mapper.map_detection(log_text="portscan detected on range")
        assert m.technique_id == "T1595"

    def test_no_match(self, mapper):
        m = mapper.map_detection(log_text="normal user activity")
        assert m.technique_id is None
        assert m.confidence == 0.0

    # ── Backward-compatible interface ──

    def test_map_alert_legacy(self, mapper):
        result = mapper.map_alert(rule_id="R001")
        assert result["mitre_technique_id"] == "T1110"
        assert result["mitre_technique_name"] == "Brute Force"
        assert result["mitre_tactic"] == "Credential Access"

    def test_map_alert_no_match(self, mapper):
        result = mapper.map_alert(log_text="nothing special")
        assert result["mitre_technique_id"] is None
