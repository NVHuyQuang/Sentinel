"""Unit tests: T0 rules provider over IncidentFeatures alone (no I/O)."""

from datetime import datetime, timedelta, timezone

import pytest

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ServiceSignalWindow,
)
from services.decision_engine.providers.rules import RulesProvider, rule_severity

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _window(service, minutes, **overrides):
    params = {
        "service": service,
        "window_start": NOW - timedelta(minutes=minutes),
        "window_end": NOW,
        "request_rate": 120.0,
        "error_rate": 0.01,
        "latency_p50_ms": 100.0,
        "latency_p95_ms": 400.0,
        "latency_p99_ms": 500.0,
        "cpu_utilization": 0.4,
        "memory_utilization": 0.5,
        "db_pool_utilization": None,
    }
    params.update(overrides)
    return ServiceSignalWindow(**params)


def _features(
    primary="order-service",
    windows=None,
    anomaly_score=0.5,
    critical_rule_triggered=False,
    deployment_services=(),
    symptom_tags=(),
):
    return IncidentFeatures(
        candidate_id="cand-1",
        opened_at=NOW,
        primary_service=primary,
        affected_services=[primary],
        windows=windows or [_window(primary, 1), _window(primary, 5)],
        detector_scores={"statistical": 0.7},
        anomaly_score=anomaly_score,
        critical_rule_triggered=critical_rule_triggered,
        deployment_services=list(deployment_services),
        symptom_tags=list(symptom_tags),
    )


def _paired(service, current_p95, baseline_p95, **current_overrides):
    current = _window(service, 1, latency_p95_ms=current_p95, **current_overrides)
    baseline = _window(service, 5, latency_p95_ms=baseline_p95)
    return [current, baseline]


async def test_latency_up():
    features = _features(
        windows=_paired("order-service", 2000.0, 400.0),
        symptom_tags=["latency_up"],
    )
    decision = await RulesProvider().triage(features)
    assert decision.provider == "rules"
    assert decision.model_version is None
    assert decision.category == "LATENCY"
    assert decision.severity == "SEV2"  # ratio exactly 5.0
    assert decision.investigate is True
    assert 0.0 <= decision.priority_score <= 1.0


async def test_error_up():
    windows = _paired("order-service", 600.0, 400.0, error_rate=0.4)
    decision = await RulesProvider().triage(_features(windows=windows, symptom_tags=["error_up"]))
    assert decision.category == "ERROR_SPIKE"
    assert decision.severity == "SEV2"


async def test_api_gateway_error_rate_is_sev1():
    windows = _paired("api-gateway", 600.0, 400.0, error_rate=0.55)
    features = _features(primary="api-gateway", windows=windows, symptom_tags=["error_up"])
    decision = await RulesProvider().triage(features)
    assert decision.category == "ERROR_SPIKE"
    assert decision.severity == "SEV1"


async def test_cpu_high_is_resource():
    windows = _paired("order-service", 600.0, 400.0, cpu_utilization=0.92)
    decision = await RulesProvider().triage(
        _features(windows=windows, symptom_tags=["cpu_high"])
    )
    assert decision.category == "RESOURCE"


async def test_db_pool_saturated_is_resource():
    windows = _paired("order-service", 900.0, 400.0, db_pool_utilization=0.95)
    decision = await RulesProvider().triage(
        _features(windows=windows, symptom_tags=["pool_saturated"])
    )
    assert decision.category == "RESOURCE"
    # Pool saturation has no Section 5.1 severity rule, so severity stays
    # metric driven (ratio 2.25 -> SEV3 here, ratio 1.0 -> SEV4 below).
    assert decision.severity == "SEV3"


async def test_dependency_timeout():
    windows = _paired(
        "order-service", 1200.0, 400.0, error_rate=0.02, dependency_timeout_rate=0.2
    )
    decision = await RulesProvider().triage(
        _features(windows=windows, symptom_tags=["dependency_timeout"])
    )
    assert decision.category == "DEPENDENCY"
    assert decision.severity == "SEV3"  # ratio 3.0


async def test_normal_is_unknown_sev4():
    features = _features(
        windows=_paired("order-service", 400.0, 400.0),
        anomaly_score=0.2,
        symptom_tags=["normal"],
    )
    decision = await RulesProvider().triage(features)
    assert decision.category == "UNKNOWN"
    assert decision.severity == "SEV4"
    assert decision.investigate is False
    assert decision.deployment_relevant is False
    assert decision.priority_score == 0.25


@pytest.mark.parametrize(
    "error_rate,expected",
    [
        (0.049, "SEV4"),
        (0.05, "SEV3"),
        (0.199, "SEV3"),
        (0.2, "SEV2"),
        (0.5, "SEV2"),  # SEV1 needs the gateway, not just a high rate
    ],
)
async def test_error_rate_thresholds(error_rate, expected):
    windows = _paired("order-service", 400.0, 400.0, error_rate=error_rate)
    decision = await RulesProvider().triage(_features(windows=windows))
    assert decision.severity == expected


@pytest.mark.parametrize("error_rate", [0.5, 0.9])
async def test_gateway_error_rate_sev1_boundary(error_rate):
    windows = _paired("api-gateway", 400.0, 400.0, error_rate=error_rate)
    decision = await RulesProvider().triage(
        _features(primary="api-gateway", windows=windows)
    )
    assert decision.severity == "SEV1"


@pytest.mark.parametrize(
    "ratio,expected",
    [(1.9, "SEV4"), (2.0, "SEV3"), (4.9, "SEV3"), (5.0, "SEV2")],
)
async def test_latency_ratio_thresholds(ratio, expected):
    windows = _paired("order-service", 400.0 * ratio, 400.0)
    decision = await RulesProvider().triage(_features(windows=windows))
    assert decision.severity == expected


async def test_any_affected_service_counts():
    windows = _paired("order-service", 400.0, 400.0) + _paired(
        "auth-service", 400.0, 400.0, error_rate=0.3
    )
    features = _features(windows=windows)
    decision = await RulesProvider().triage(features)
    assert decision.severity == "SEV2"
    assert rule_severity(features) == "SEV2"


def test_rule_severity_only_sev1_sev2():
    gateway = _features(primary="api-gateway", windows=_paired("api-gateway", 400.0, 400.0, error_rate=0.6))
    assert rule_severity(gateway) == "SEV1"
    assert rule_severity(_features(windows=_paired("order-service", 400.0, 400.0, error_rate=0.3))) == "SEV2"
    assert rule_severity(_features(windows=_paired("order-service", 400.0, 400.0, error_rate=0.1))) is None
    assert rule_severity(_features(windows=_paired("order-service", 400.0, 400.0))) is None


async def test_critical_and_high_anomaly_investigate():
    normal = _paired("order-service", 400.0, 400.0)
    critical = await RulesProvider().triage(
        _features(windows=list(normal), critical_rule_triggered=True)
    )
    assert critical.severity == "SEV4"
    assert critical.investigate is True
    high_anomaly = await RulesProvider().triage(
        _features(windows=list(normal), anomaly_score=0.9)
    )
    assert high_anomaly.investigate is True


async def test_recent_deployment_marks_deployment_relevant():
    windows = _paired("order-service", 400.0, 400.0, recent_deployment=True)
    decision = await RulesProvider().triage(_features(windows=windows))
    assert decision.deployment_relevant is True


async def test_empty_windows_still_valid():
    decision = await RulesProvider().triage(_features(windows=[]))
    assert decision.category == "UNKNOWN"
    assert decision.severity == "SEV4"
