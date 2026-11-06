"""Kill-fallback test (integration, mocked provider, no GPU).

Laya is an in-process library, so there is no HTTP process to kill: the
kill is simulated at provider-object level, exactly like the spike's
_DeadRouter (docs/laya-contract.md Section 5). The router answers the
untimed warmup once, then every later predict raises ProviderError, i.e.
the provider dies after startup. The selector must return rules on the
next triage with zero manual steps.
"""

import copy
from datetime import datetime, timedelta, timezone

from services.decision_engine.policy import triage_with_fallback
from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    ServiceSignalWindow,
)
from services.decision_engine.providers.laya import LayaAdapter

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)

MINIMAL_RAW = {
    "model": "laya-rl-agent",
    "answers": {
        "category": {
            "type": "choice",
            "choice": "LATENCY",
            "probabilities": {"LATENCY": 0.6},
            "confidence": 0.2,
            "answer_confidence": 0.6,
            "action": {"act_probability": 1.0},
        },
        "severity": {
            "type": "choice",
            "choice": "SEV4",
            "probabilities": {"SEV4": 0.7},
            "confidence": 0.4,
            "answer_confidence": 0.7,
            "action": {"act_probability": 1.0},
        },
        "priority_score": {
            "type": "score",
            "score": 4.5,
            "probabilities": {"4": 0.5},
            "confidence": 0.0,
            "answer_confidence": 0.5,
            "action": {"act_probability": 1.0},
        },
        "investigate": {
            "type": "noul",
            "noul": 0.9,
            "confidence": 0.9,
            "answer_confidence": 0.9,
            "action": {"act_probability": 1.0},
        },
        "deployment_relevant": {
            "type": "noul",
            "noul": 0.1,
            "confidence": 0.1,
            "answer_confidence": 0.1,
            "action": {"act_probability": 1.0},
        },
    },
    "routing": {"repo": "convaiinnovations/laya/multilingual"},
}


class KillRouter:
    """Answers warmup, then raises ProviderError like a dead Laya instance."""

    def __init__(self):
        self.calls = 0

    def predict(self, text, questions):
        self.calls += 1
        if self.calls == 1:  # startup warmup
            return copy.deepcopy(MINIMAL_RAW)
        raise ProviderError("simulated Laya process death (in-process)")


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
    }
    params.update(overrides)
    return ServiceSignalWindow(**params)


def _features(service="order-service", current=None, baseline=None):
    current = current or _window(service, 1)
    baseline = baseline or _window(service, 5)
    return IncidentFeatures(
        candidate_id="cand-1",
        opened_at=NOW,
        primary_service=service,
        affected_services=[service],
        windows=[current, baseline],
        detector_scores={"statistical": 0.7},
        anomaly_score=0.7,
        symptom_tags=["latency_up"],
    )


async def test_dead_laya_falls_back_to_rules_next_triage():
    adapter = LayaAdapter(device="cpu", timeout_ms=5000, router=KillRouter())
    decision = await triage_with_fallback(_features(), primary=adapter)
    assert decision.provider == "rules"
    assert decision.severity in {"SEV1", "SEV2", "SEV3", "SEV4"}


async def test_severity_floor_raises_healthy_laya_decision():
    # Laya says SEV4, but the deterministic rule fires SEV1 (gateway 0.6):
    # the floor raises, it never lowers.
    adapter = LayaAdapter(device="cpu", timeout_ms=5000, router=_HealthyRouter())
    features = _features(
        service="api-gateway",
        current=_window("api-gateway", 1, error_rate=0.6),
        baseline=_window("api-gateway", 5, error_rate=0.6),
    )
    decision = await triage_with_fallback(features, primary=adapter)
    assert decision.provider == "laya"
    assert decision.severity == "SEV1"


async def test_severity_unchanged_when_no_rule_fires():
    adapter = LayaAdapter(device="cpu", timeout_ms=5000, router=_HealthyRouter())
    decision = await triage_with_fallback(_features(), primary=adapter)
    assert decision.provider == "laya"
    assert decision.severity == "SEV4"


class _HealthyRouter:
    def predict(self, text, questions):
        return copy.deepcopy(MINIMAL_RAW)
