"""Unit tests: Laya parser + adapter shell. Router is always mocked here.

Never loads the Laya checkpoint and needs no CUDA. VALID_RAW_JSON is one
real raw_response copied from docs/spike_laya_samples-cuda.jsonl (s00).
"""

import copy
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    ServiceSignalWindow,
)
from services.decision_engine.providers.laya import (
    QUESTIONS,
    LayaAdapter,
    parse_result,
    render_features,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)

# One real raw_response from docs/spike_laya_samples-cuda.jsonl, sample s00
# (DEP/SEV4, score 4.5339, noul 0.7206/0.8579, multilingual routing).
VALID_RAW_JSON = r"""{
  "model": "laya-rl-agent",
  "answers": {
    "category": {
      "type": "choice",
      "choice": "DEPENDENCY",
      "probabilities": {
        "LATENCY": 0.1636,
        "ERROR_SPIKE": 0.1057,
        "RESOURCE": 0.1537,
        "DEPENDENCY": 0.3154,
        "UNKNOWN": 0.2615
      },
      "confidence": 0.0455,
      "answer_confidence": 0.3154,
      "action": {"act_probability": 1.0}
    },
    "severity": {
      "type": "choice",
      "choice": "SEV4",
      "probabilities": {"SEV1": 0.115, "SEV2": 0.2183, "SEV3": 0.2342, "SEV4": 0.4325},
      "confidence": 0.0742,
      "answer_confidence": 0.4325,
      "action": {"act_probability": 1.0}
    },
    "priority_score": {
      "type": "score",
      "score": 4.5339,
      "legend": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"},
      "probabilities": {
        "0": 0.0963, "1": 0.0963, "2": 0.0993, "3": 0.1025, "4": 0.1025,
        "5": 0.1025, "6": 0.0993, "7": 0.0993, "8": 0.0993, "9": 0.1025
      },
      "confidence": 0.0001,
      "answer_confidence": 0.1025,
      "action": {"act_probability": 1.0}
    },
    "investigate": {
      "type": "noul",
      "noul": 0.7206,
      "confidence": 0.7206,
      "answer_confidence": 0.7206,
      "action": {"act_probability": 1.0}
    },
    "deployment_relevant": {
      "type": "noul",
      "noul": 0.8579,
      "confidence": 0.8579,
      "answer_confidence": 0.8579,
      "action": {"act_probability": 1.0}
    }
  },
  "usage": {
    "input_tokens": 697,
    "output_tokens": 0,
    "state_tokens": 90,
    "state_tokens_dropped": 0,
    "truncated": false,
    "truncated_questions": []
  },
  "routing": {
    "model": "multilingual",
    "repo": "convaiinnovations/laya/multilingual",
    "reason": "Latin script, language not identified and no non-English letters; using default (multilingual)",
    "detection": {
      "script": "latin",
      "script_profile": {"latin": 1.0},
      "language": null,
      "is_english": true,
      "language_undecided": true,
      "diacritic_rate": 0.0,
      "non_latin_fraction": 0.0,
      "mixed_segment": null
    },
    "workflow": null
  }
}"""

RAW = json.loads(VALID_RAW_JSON)


class FakeRouter:
    """Mock Router: records predict calls, replays a canned raw payload."""

    def __init__(self, raw=None, delay=0.0, error=None):
        self.raw = RAW if raw is None else raw
        self.delay = delay
        self.error = error
        self.calls = []

    def predict(self, text, questions):
        self.calls.append((text, questions))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.raw)


def _features():
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
    return IncidentFeatures(
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


def test_parse_valid_fixture():
    decision = parse_result(RAW)
    assert decision.provider == "laya"
    assert decision.category == "DEPENDENCY"
    assert decision.severity == "SEV4"
    assert decision.priority_score == pytest.approx(4.5339 / 9.0)
    assert decision.investigate is True  # noul 0.7206 >= 0.5
    assert decision.deployment_relevant is True  # noul 0.8579 >= 0.5
    assert decision.model_version == "convaiinnovations/laya/multilingual"
    assert decision.raw_scores["score_expected"] == pytest.approx(4.5339)


def test_parse_missing_key_is_malformed():
    with pytest.raises(ProviderError, match="malformed: missing decision key"):
        parse_result({"answers": {"category": {"choice": "LATENCY"}}})


def test_parse_none_is_malformed():
    with pytest.raises(ProviderError, match="malformed"):
        parse_result(None)


def test_parse_out_of_vocab_choice():
    raw = copy.deepcopy(RAW)
    raw["answers"]["category"]["choice"] = "NOT_A_CHOICE"
    raw["answers"]["severity"]["choice"] = "SEV9"
    with pytest.raises(ProviderError, match="out-of-vocab"):
        parse_result(raw)


@pytest.mark.parametrize("score", [9.5, -1.0])
def test_parse_out_of_range_priority(score):
    raw = copy.deepcopy(RAW)
    raw["answers"]["priority_score"]["score"] = score
    with pytest.raises(ProviderError, match="out of range"):
        parse_result(raw)


def test_parse_bad_priority_type_is_malformed():
    raw = copy.deepcopy(RAW)
    raw["answers"]["priority_score"]["score"] = "high"
    with pytest.raises(ProviderError, match="malformed: bad priority_score"):
        parse_result(raw)


def test_parse_missing_noul_is_malformed():
    raw = copy.deepcopy(RAW)
    del raw["answers"]["investigate"]["noul"]
    with pytest.raises(ProviderError, match="malformed: bad noul"):
        parse_result(raw)


def test_render_matches_spike_format():
    text = render_features(_features())
    assert text == (
        "primary_service order-service affected order-service "
        "error_rate 0.082 p95_latency_ms 3480.0 baseline_p95_ms 400.0 "
        "cpu_utilization 0.55 memory_utilization 0.5 db_pool 0.92 "
        "anomaly_score 0.91 critical_rule_triggered true "
        "recent_deployment true symptom_tags latency_up,pool_saturated"
    )


def test_render_pool_missing_is_na():
    features = _features()
    # Rebuild with pool None on the current (shortest) window.
    windows = [
        w.model_copy(update={"db_pool_utilization": None}) for w in features.windows
    ]
    features = features.model_copy(update={"windows": windows})
    assert " db_pool n/a " in render_features(features)


def test_adapter_warmup_once_then_triage():
    router = FakeRouter()
    adapter = LayaAdapter(device="cpu", timeout_ms=5000, router=router)
    assert router.calls == [("warmup", QUESTIONS)]
    assert adapter.timeout_ms == 5000


async def test_adapter_triage_records_latency_and_device():
    router = FakeRouter()
    adapter = LayaAdapter(device="cuda", router=router)
    assert adapter.timeout_ms == 2000.0  # GPU production default above P95
    decision = await adapter.triage(_features())
    assert decision.provider == "laya"
    assert decision.category == "DEPENDENCY"
    assert decision.latency_ms is not None and decision.latency_ms >= 0.0
    assert decision.raw_scores["device_cuda"] == 1.0
    assert router.calls[-1][0].startswith("primary_service order-service ")


def test_adapter_default_timeout_cpu_above_p95():
    adapter = LayaAdapter(device="cpu", router=FakeRouter())
    assert adapter.timeout_ms >= 1800.0


def test_adapter_rejects_bad_device_and_timeout():
    with pytest.raises(ValueError, match="device"):
        LayaAdapter(device="tpu", router=FakeRouter())
    with pytest.raises(ValueError, match="timeout_ms"):
        LayaAdapter(device="cpu", timeout_ms=0, router=FakeRouter())


def test_construction_failure_is_provider_error():
    router = FakeRouter(error=RuntimeError("no checkpoint"))
    with pytest.raises(ProviderError, match="laya warmup failed"):
        LayaAdapter(device="cpu", router=router)


async def test_predict_exception_is_provider_error():
    # Router that survives warmup then fails on real predicts.
    router = FakeRouter()
    adapter = LayaAdapter(device="cpu", router=router)
    router.error = RuntimeError("boom")
    with pytest.raises(ProviderError, match="laya predict failed"):
        await adapter.triage(_features())


async def test_predict_timeout_is_provider_error():
    router = FakeRouter(delay=0.3)
    adapter = LayaAdapter(device="cuda", timeout_ms=50, router=router)
    with pytest.raises(ProviderError, match="timeout"):
        await adapter.triage(_features())
