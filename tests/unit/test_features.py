"""Unit tests: window aggregation over 15s scrapes (Spec 4.1, 4.2)."""

from datetime import datetime, timedelta, timezone

from services.decision_engine.providers.base import ServiceSignalWindow
from services.signal_processor.features import (
    CONTEXT_WINDOW_SECONDS,
    SHORT_WINDOW_SECONDS,
    RawScrape,
    aggregate_window,
    build_evaluation_windows,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _series(service="order-service", count=20, **overrides):
    base = {
        "request_rate": 100.0,
        "error_rate": 0.02,
        "latency_p50_ms": 80.0,
        "latency_p95_ms": 180.0,
        "latency_p99_ms": 320.0,
        "cpu_utilization": 0.4,
        "memory_utilization": 0.5,
        "db_pool_utilization": 0.6,
        "dependency_timeout_rate": 0.01,
    }
    base.update(overrides)
    rows = []
    for index in range(count):
        rows.append(
            RawScrape(
                service=service,
                timestamp=NOW - timedelta(seconds=15 * (count - index)),
                **base,
            )
        )
    return rows


def test_short_and_context_windows_from_15s_series():
    scrapes = _series(count=20)
    short, context = build_evaluation_windows("order-service", scrapes, NOW)
    assert isinstance(short, ServiceSignalWindow)
    assert isinstance(context, ServiceSignalWindow)
    assert (short.window_end - short.window_start).total_seconds() == (SHORT_WINDOW_SECONDS)
    assert (context.window_end - context.window_start).total_seconds() == (CONTEXT_WINDOW_SECONDS)
    assert short.window_start == NOW - timedelta(seconds=SHORT_WINDOW_SECONDS)
    assert context.window_start == NOW - timedelta(seconds=CONTEXT_WINDOW_SECONDS)
    assert short.request_rate == 100.0
    assert short.error_rate == 0.02
    assert context.latency_p95_ms == 180.0
    assert short.cpu_utilization == 0.4


def test_missing_optional_metric_stays_none():
    scrapes = _series(count=8, db_pool_utilization=None, dependency_timeout_rate=None)
    short, context = build_evaluation_windows("order-service", scrapes, NOW)
    assert short.db_pool_utilization is None
    assert short.dependency_timeout_rate is None
    assert context.db_pool_utilization is None
    assert context.dependency_timeout_rate is None
    assert short.request_rate == 100.0


def test_partial_samples_average_without_zero_fill():
    scrapes = _series(count=4)
    scrapes[0].request_rate = 80.0
    scrapes[1].request_rate = None
    scrapes[2].request_rate = 120.0
    scrapes[3].request_rate = 100.0
    window = aggregate_window(
        "order-service",
        scrapes,
        NOW - timedelta(seconds=60),
        NOW,
    )
    assert window.request_rate == 100.0
    assert window.db_pool_utilization == 0.6


def test_counts_sum_and_deployment_flag_any():
    scrapes = _series(count=4)
    scrapes[0].log_error_count = 1
    scrapes[1].log_error_count = 2
    scrapes[2].trace_error_count = 3
    scrapes[3].recent_deployment = True
    window = aggregate_window(
        "order-service",
        scrapes,
        NOW - timedelta(seconds=60),
        NOW,
    )
    assert window.log_error_count == 3
    assert window.trace_error_count == 3
    assert window.recent_deployment is True
