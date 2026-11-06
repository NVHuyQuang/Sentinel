"""Unit tests: Spec §5.2 contract shapes (no providers, no I/O)."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ServiceSignalWindow,
    TriageDecision,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
SERVICES = ("auth-service", "order-service", "payment-service", "api-gateway")


def _window(service: str, minutes: int) -> ServiceSignalWindow:
    return ServiceSignalWindow(
        service=service,
        window_start=NOW - timedelta(minutes=minutes),
        window_end=NOW,
        request_rate=120.0,
        error_rate=0.02,
        latency_p50_ms=80.0,
        latency_p95_ms=180.0,
        latency_p99_ms=320.0,
        cpu_utilization=0.4,
        memory_utilization=0.5,
    )


def _features(window_minutes: int = 1) -> IncidentFeatures:
    # 1-min and 5-min window per affected service (Spec §4.1, §5.2).
    return IncidentFeatures(
        candidate_id="cand-1",
        opened_at=NOW,
        primary_service="order-service",
        affected_services=list(SERVICES),
        windows=[_window(s, m) for s in SERVICES for m in (window_minutes, 5)],
        detector_scores={"statistical": 0.7, "isolation": 0.8, "change_point": 0.6},
        anomaly_score=0.82,
        critical_rule_triggered=True,
        dependency_edges=[("api-gateway", "order-service")],
        symptom_tags=["latency_up", "pool_saturated"],
    )


def test_incident_features_accepts_multi_service_windows():
    features = _features()
    assert len(features.windows) == 2 * len(SERVICES)
    per_service = {}
    for w in features.windows:
        per_service.setdefault(w.service, []).append(w)
    assert set(per_service) == set(SERVICES)
    assert all(len(ws) == 2 for ws in per_service.values())


def _decision(**overrides) -> dict:
    base = {
        "category": "LATENCY",
        "severity": "SEV3",
        "priority_score": 0.5,
        "investigate": True,
        "provider": "rules",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "category", ["LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN"]
)
def test_triage_decision_accepts_all_categories(category):
    decision = TriageDecision(**_decision(category=category))
    assert decision.category == category


@pytest.mark.parametrize("severity", ["SEV1", "SEV2", "SEV3", "SEV4"])
def test_triage_decision_accepts_all_severities(severity):
    decision = TriageDecision(**_decision(severity=severity))
    assert decision.severity == severity


@pytest.mark.parametrize("category", ["PERFORMANCE_DEGRADATION", "High", "latency", ""])
def test_triage_decision_rejects_bad_category(category):
    with pytest.raises(ValidationError):
        TriageDecision(**_decision(category=category))


@pytest.mark.parametrize("severity", ["SEV0", "SEV5", "Critical", "high", ""])
def test_triage_decision_rejects_bad_severity(severity):
    with pytest.raises(ValidationError):
        TriageDecision(**_decision(severity=severity))


@pytest.mark.parametrize("score", [-0.1, 1.1, 85.0])
def test_triage_decision_rejects_out_of_range_priority_score(score):
    with pytest.raises(ValidationError):
        TriageDecision(**_decision(priority_score=score))


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_triage_decision_accepts_in_range_priority_score(score):
    assert TriageDecision(**_decision(priority_score=score)).priority_score == score
