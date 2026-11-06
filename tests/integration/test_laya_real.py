"""Real Laya spot-check (integration).

Skipped when the laya package, a CUDA device or the Laya checkpoint is
absent, so the suite never depends on a GPU. A checkpoint download/init
failure surfaces as ProviderError and also skips with the reason.
"""

import importlib.util
from datetime import datetime, timedelta, timezone

import pytest

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    ServiceSignalWindow,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - missing torch/CUDA means skip
        return False


_LAYA_INSTALLED = importlib.util.find_spec("laya") is not None

pytestmark = pytest.mark.skipif(
    not (_LAYA_INSTALLED and _has_cuda()),
    reason="requires the laya package and a CUDA device (checkpoint must also be present)",
)


async def test_laya_real_triage():
    from services.decision_engine.providers.laya import LayaAdapter

    current = ServiceSignalWindow(
        service="order-service",
        window_start=NOW - timedelta(minutes=1),
        window_end=NOW,
        request_rate=120.0,
        error_rate=0.082,
        latency_p50_ms=300.0,
        latency_p95_ms=3480.0,
        latency_p99_ms=4000.0,
        cpu_utilization=0.55,
        memory_utilization=0.5,
        db_pool_utilization=0.92,
    )
    baseline = ServiceSignalWindow(
        service="order-service",
        window_start=NOW - timedelta(minutes=5),
        window_end=NOW,
        request_rate=120.0,
        error_rate=0.01,
        latency_p50_ms=200.0,
        latency_p95_ms=400.0,
        latency_p99_ms=450.0,
        cpu_utilization=0.4,
        memory_utilization=0.4,
    )
    features = IncidentFeatures(
        candidate_id="cand-1",
        opened_at=NOW,
        primary_service="order-service",
        affected_services=["order-service"],
        windows=[current, baseline],
        detector_scores={"statistical": 0.9},
        anomaly_score=0.91,
        critical_rule_triggered=True,
        symptom_tags=["latency_up", "pool_saturated"],
    )
    try:
        adapter = LayaAdapter(device="cuda")
        decision = await adapter.triage(features)
    except ProviderError as exc:
        pytest.skip(f"Laya checkpoint unavailable on this host: {exc}")
    assert decision.provider == "laya"
    assert decision.category in {"LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN"}
    assert decision.severity in {"SEV1", "SEV2", "SEV3", "SEV4"}
    assert 0.0 <= decision.priority_score <= 1.0
    assert decision.latency_ms is not None and decision.latency_ms > 0.0
