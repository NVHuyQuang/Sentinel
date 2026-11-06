"""Unit tests: sklearn Random Forest provider T1 (in-memory bundle, no disk)."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from services.decision_engine.policy import triage_with_fallback
from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    ServiceSignalWindow,
)
from services.decision_engine.providers.sklearn import (
    FEATURE_ORDER,
    RandomForestProvider,
    extract_features,
    fit_bundle,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)

CATEGORIES = ("LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN")
SEVERITIES = ("SEV1", "SEV2", "SEV3", "SEV4")


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


def _features(**overrides):
    params = {
        "primary": "order-service",
        "current": None,
        "baseline": None,
        "anomaly_score": 0.5,
        "detector_scores": {"statistical": 0.7},
    }
    params.update(overrides)
    current = params["current"] or _window(params["primary"], 1)
    baseline = params["baseline"] or _window(params["primary"], 5)
    return IncidentFeatures(
        candidate_id="cand-1",
        opened_at=NOW,
        primary_service=params["primary"],
        affected_services=[params["primary"]],
        windows=[current, baseline],
        detector_scores=dict(params["detector_scores"]),
        anomaly_score=params["anomaly_score"],
        symptom_tags=["latency_up"],
    )


def _train_tiny_bundle():
    rows = []
    y_category = []
    y_severity = []
    for i in range(20):
        current = _window(
            "order-service",
            1,
            error_rate=0.02 * i,
            latency_p95_ms=100.0 + 100.0 * i,
            db_pool_utilization=0.1 * i if i % 2 == 0 else None,
        )
        features = _features(
            current=current,
            anomaly_score=0.1 + 0.04 * i,
            detector_scores={"statistical": 0.1 * (i % 10)},
        )
        rows.append(extract_features(features))
        y_category.append(CATEGORIES[i % len(CATEGORIES)])
        y_severity.append(SEVERITIES[i % len(SEVERITIES)])
    return fit_bundle(rows, y_category, y_severity, model_version="test-smoke")


def test_feature_vector_shape_and_order_stable():
    first = _features(
        current=_window(
            "order-service", 1, error_rate=0.123, latency_p95_ms=777.0
        )
    )
    second = _features(
        current=_window("order-service", 1, error_rate=0.456),
        anomaly_score=0.99,
    )
    vec_first = extract_features(first)
    vec_second = extract_features(second)
    assert len(FEATURE_ORDER) == 23
    assert len(vec_first) == len(FEATURE_ORDER)
    assert len(vec_second) == len(FEATURE_ORDER)
    assert vec_first[FEATURE_ORDER.index("primary_error_rate")] == pytest.approx(0.123)
    assert vec_first[FEATURE_ORDER.index("primary_latency_p95_ms")] == pytest.approx(
        777.0
    )
    assert vec_first[FEATURE_ORDER.index("max_error_rate")] == pytest.approx(0.123)
    assert vec_first[FEATURE_ORDER.index("anomaly_score")] == pytest.approx(0.5)
    assert vec_second[FEATURE_ORDER.index("anomaly_score")] == pytest.approx(0.99)
    assert vec_first[FEATURE_ORDER.index("window_count")] == pytest.approx(2.0)
    assert vec_first[FEATURE_ORDER.index("affected_service_count")] == pytest.approx(
        1.0
    )


def test_missing_db_pool_stays_nan():
    missing = _features(
        current=_window("order-service", 1, db_pool_utilization=None)
    )
    present = _features(
        current=_window("order-service", 1, db_pool_utilization=0.92)
    )
    missing_vec = extract_features(missing)
    present_vec = extract_features(present)
    assert math.isnan(
        missing_vec[FEATURE_ORDER.index("primary_db_pool_utilization")]
    )
    assert math.isnan(missing_vec[FEATURE_ORDER.index("max_db_pool_utilization")])
    assert present_vec[FEATURE_ORDER.index("primary_db_pool_utilization")] == (
        pytest.approx(0.92)
    )
    assert not math.isnan(
        present_vec[FEATURE_ORDER.index("primary_db_pool_utilization")]
    )
    assert math.isnan(missing_vec[FEATURE_ORDER.index("detector_isolation")])


def test_bundle_stores_shared_feature_order():
    bundle = _train_tiny_bundle()
    assert bundle["feature_order"] == list(FEATURE_ORDER)
    assert bundle["model_version"] == "test-smoke"


def test_missing_model_file_is_provider_error(monkeypatch):
    with pytest.raises(ProviderError, match="model file not found"):
        RandomForestProvider(model_path="does-not-exist-12345.joblib")
    monkeypatch.delenv("SKLEARN_MODEL_PATH", raising=False)
    with pytest.raises(ProviderError, match="missing model file"):
        RandomForestProvider()


async def test_trained_once_serves_many():
    bundle = _train_tiny_bundle()
    provider = RandomForestProvider(bundle=bundle)
    assert provider.model_version == "test-smoke"
    first = await provider.triage(_features())
    second = await provider.triage(
        _features(anomaly_score=0.95, detector_scores={"isolation": 0.2})
    )
    for decision in (first, second):
        assert decision.provider == "sklearn"
        assert decision.category in CATEGORIES
        assert decision.severity in SEVERITIES
        assert 0.0 <= decision.priority_score <= 1.0
        assert decision.latency_ms is not None and decision.latency_ms >= 0.0
        assert any(k.startswith("category_") for k in decision.raw_scores)
        assert any(k.startswith("severity_") for k in decision.raw_scores)
        assert decision.raw_scores["category_confidence"] == pytest.approx(
            decision.priority_score
        )


async def test_priority_in_range():
    provider = RandomForestProvider(bundle=_train_tiny_bundle())
    decision = await provider.triage(_features())
    assert 0.0 <= decision.priority_score <= 1.0


async def test_plugs_into_triage_with_fallback():
    provider = RandomForestProvider(bundle=_train_tiny_bundle())
    decision = await triage_with_fallback(_features(), primary=provider)
    assert decision.provider == "sklearn"
    assert decision.category in CATEGORIES
