"""
Tests for the unified API — end-to-end integration.

Uses FastAPI's TestClient for synchronous testing without launching
a server.  The structured analyzer will only be 'ready' if trained
model artifacts exist in outputs/checkpoints/.
"""

import os
import pytest

from fastapi.testclient import TestClient

# Need to set up imports before the app is created
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers the lifespan (model loading)."""
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ #
#  Health                                                              #
# ------------------------------------------------------------------ #


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_structure(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "engines" in data
        assert "structured_ids" in data["engines"]
        assert "semantic_anomaly" in data["engines"]
        assert "version" in data


# ------------------------------------------------------------------ #
#  /detect — auto-routing                                              #
# ------------------------------------------------------------------ #


class TestDetectAutoRoute:
    def test_log_text_returns_alert(self, client):
        resp = client.post(
            "/detect",
            json={"event_id": "test-1", "log_text": "Failed login for root"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "alert_id" in data
        assert "severity" in data
        assert "is_threat" in data
        assert "threat_score" in data
        assert data["raw_input_type"] == "log"

    def test_missing_both_fields_422(self, client):
        resp = client.post("/detect", json={"event_id": "test-2"})
        assert resp.status_code == 422

    def test_empty_event_id_422(self, client):
        resp = client.post(
            "/detect", json={"event_id": "", "log_text": "hello"}
        )
        assert resp.status_code == 422


# ------------------------------------------------------------------ #
#  /detect/log — explicit semantic                                     #
# ------------------------------------------------------------------ #


class TestDetectLog:
    def test_returns_alert(self, client):
        resp = client.post(
            "/detect/log",
            json={
                "event_id": "log-1",
                "log_text": "sudo: user gained root access",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "semantic_anomaly" in data["engines_used"] or len(data["engines_used"]) == 0

    def test_no_log_text_422(self, client):
        resp = client.post(
            "/detect/log",
            json={
                "event_id": "log-2",
                "flow_features": {"destination_port": 80},
            },
        )
        assert resp.status_code == 422


# ------------------------------------------------------------------ #
#  /detect/flow — explicit structured                                  #
# ------------------------------------------------------------------ #


class TestDetectFlow:
    def test_no_flow_features_422(self, client):
        resp = client.post(
            "/detect/flow",
            json={"event_id": "flow-1", "log_text": "hello"},
        )
        assert resp.status_code == 422


# ------------------------------------------------------------------ #
#  Response schema conformance                                         #
# ------------------------------------------------------------------ #


class TestResponseSchema:
    def test_soc_alert_fields(self, client):
        resp = client.post(
            "/detect",
            json={"event_id": "schema-1", "log_text": "test log entry"},
        )
        data = resp.json()
        # Required SOCAlert fields
        required = [
            "alert_id",
            "event_id",
            "timestamp",
            "severity",
            "is_threat",
            "threat_score",
            "analyses",
            "mitre",
            "rule_matches",
            "drift_alert",
            "processing_time_ms",
            "engines_used",
            "raw_input_type",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_mitre_structure(self, client):
        resp = client.post(
            "/detect",
            json={
                "event_id": "mitre-1",
                "log_text": "brute force SSH login attempt",
            },
        )
        mitre = resp.json()["mitre"]
        assert "technique_id" in mitre
        assert "technique_name" in mitre
        assert "tactic" in mitre
