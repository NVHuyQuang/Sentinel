"""Real OmniRoute spot-check (integration, exactly ONE call).

Skipped when OMNIROUTE_API_KEY is missing or localhost:20128 is unreachable
(probe with a 2 s timeout first). Exactly one triage call is made when it
runs: OmniRoute has a semantic cache (identical prompts return the same
response id in about 2.1 s), so repeats would measure cache hits, not fresh
model behavior.
"""

import os
import socket
from datetime import datetime, timedelta, timezone

import pytest

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ServiceSignalWindow,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _reachable(host="127.0.0.1", port=20128, timeout=2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not (os.getenv("OMNIROUTE_API_KEY") and _reachable()),
    reason="requires OMNIROUTE_API_KEY and a live OmniRoute on localhost:20128",
)


async def test_llm_real_one_call():
    from services.decision_engine.providers.llm import LLMProvider

    current = ServiceSignalWindow(
        service="order-service",
        window_start=NOW - timedelta(minutes=1),
        window_end=NOW,
        request_rate=120.0,
        error_rate=0.24,
        latency_p50_ms=800.0,
        latency_p95_ms=3420.0,
        latency_p99_ms=4000.0,
        cpu_utilization=0.55,
        memory_utilization=0.5,
        db_pool_utilization=0.92,
        recent_deployment=True,
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
        candidate_id="cand-real-1",
        opened_at=NOW,
        primary_service="order-service",
        affected_services=["order-service"],
        windows=[current, baseline],
        detector_scores={"statistical": 0.9},
        anomaly_score=0.91,
        critical_rule_triggered=True,
        symptom_tags=["latency_up", "pool_saturated"],
    )
    provider = LLMProvider()
    decision = await provider.triage(features)
    assert decision.provider == "llm"
    assert decision.category in {
        "LATENCY",
        "ERROR_SPIKE",
        "RESOURCE",
        "DEPENDENCY",
        "UNKNOWN",
    }
    assert decision.severity in {"SEV1", "SEV2", "SEV3", "SEV4"}
    assert 0.0 <= decision.priority_score <= 1.0
    assert decision.latency_ms is not None and decision.latency_ms > 0.0
    assert decision.model_version is not None and decision.model_version != ""
    assert decision.raw_scores["prompt_tokens"] > 0.0
    assert decision.raw_scores["completion_tokens"] > 0.0
