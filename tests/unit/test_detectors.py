"""Unit tests: detectors and ensemble formula (Spec 4.3, D0-D4)."""

from datetime import datetime, timedelta, timezone

from services.decision_engine.providers.base import ServiceSignalWindow
from services.signal_processor.detectors import (
    CusumDetector,
    IsolationForestDetector,
    RobustBaselineDetector,
)
from services.signal_processor.ensemble import (
    ENSEMBLE_WEIGHTS,
    CandidateGate,
    compute_ensemble_score,
    critical_rule_fires,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _window(error_rate=0.02, latency_p95_ms=180.0, cpu_utilization=0.4):
    return ServiceSignalWindow(
        service="order-service",
        window_start=NOW - timedelta(seconds=60),
        window_end=NOW,
        request_rate=100.0,
        error_rate=error_rate,
        latency_p50_ms=80.0,
        latency_p95_ms=latency_p95_ms,
        latency_p99_ms=320.0,
        cpu_utilization=cpu_utilization,
        memory_utilization=0.5,
        db_pool_utilization=0.5,
        dependency_timeout_rate=0.01,
    )


def _normal_windows(count=20):
    return [_window(error_rate=0.02, latency_p95_ms=180.0) for _ in range(count)]


def test_robust_detector_scores_in_unit_interval():
    detector = RobustBaselineDetector(z_threshold=3.5)
    detector.fit(_normal_windows(20))
    steady = detector.score_window(_window())
    spike = detector.score_window(_window(error_rate=0.9, latency_p95_ms=2000.0))
    assert 0.0 <= steady <= 1.0
    assert 0.0 <= spike <= 1.0
    assert spike > steady


def test_isolation_detector_scores_in_unit_interval():
    detector = IsolationForestDetector(n_estimators=200, contamination=0.03)
    detector.fit(_normal_windows(24))
    steady = detector.score_window(_window())
    spike = detector.score_window(
        _window(error_rate=0.85, latency_p95_ms=2500.0, cpu_utilization=0.98)
    )
    assert 0.0 <= steady <= 1.0
    assert 0.0 <= spike <= 1.0


def test_isolation_detector_handles_few_rows():
    detector = IsolationForestDetector(n_estimators=200, contamination=0.03)
    detector.fit(_normal_windows(5))
    score = detector.score_window(_window())
    assert 0.0 <= score <= 1.0
    tiny = IsolationForestDetector(n_estimators=200, contamination=0.03)
    tiny.fit(_normal_windows(3))
    assert 0.0 <= tiny.score_window(_window(error_rate=0.9)) <= 1.0


def test_cusum_detector_scores_in_unit_interval():
    detector = CusumDetector(drift=0.5, threshold=5.0)
    detector.fit(_normal_windows(20))
    steady_score = detector.score_series([0.02] * 12)
    spike_score = detector.score_series([0.02] * 6 + [0.6] * 6)
    assert 0.0 <= steady_score <= 1.0
    assert 0.0 <= spike_score <= 1.0
    assert spike_score > steady_score


def test_ensemble_weights_sum_and_formula():
    assert abs(sum(ENSEMBLE_WEIGHTS) - 1.0) <= 1e-9
    assert abs(ENSEMBLE_WEIGHTS[0] - 0.35) <= 1e-12
    assert abs(ENSEMBLE_WEIGHTS[1] - 0.40) <= 1e-12
    assert abs(ENSEMBLE_WEIGHTS[2] - 0.25) <= 1e-12
    got = compute_ensemble_score(0.2, 0.5, 0.8)
    expected = 0.35 * 0.2 + 0.40 * 0.5 + 0.25 * 0.8
    assert abs(got - expected) <= 1e-9


def test_gate_hysteresis_and_cooldown():
    gate = CandidateGate(trigger_threshold=0.70, clear_threshold=0.45, cooldown_seconds=180)
    first = NOW
    assert gate.evaluate("order-service", 0.9, False, first) is True
    assert gate.is_active("order-service") is True
    assert gate.evaluate("order-service", 0.9, False, first + timedelta(seconds=30)) is False
    assert gate.evaluate("order-service", 0.30, False, first + timedelta(seconds=60)) is False
    assert gate.is_active("order-service") is False


def test_critical_rule_bypass_and_missing_never_fires():
    assert critical_rule_fires(_window(error_rate=0.55)) is True
    assert critical_rule_fires(_window(error_rate=0.02)) is False
    empty = ServiceSignalWindow(
        service="order-service",
        window_start=NOW - timedelta(seconds=60),
        window_end=NOW,
        request_rate=100.0,
        error_rate=0.02,
        latency_p50_ms=80.0,
        latency_p95_ms=180.0,
        latency_p99_ms=320.0,
        cpu_utilization=0.4,
        memory_utilization=0.5,
        db_pool_utilization=None,
        dependency_timeout_rate=None,
    )
    assert critical_rule_fires(empty) is False
