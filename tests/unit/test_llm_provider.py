"""Unit tests: OmniRoute LLM provider T3 (mocked transport, no network, no .env)."""

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from services.decision_engine.policy import triage_with_fallback
from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    ServiceSignalWindow,
)
from services.decision_engine.providers.laya import render_features
from services.decision_engine.providers.llm import (
    LLMProvider,
    parse_llm_response,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)

# One real raw_response trimmed to parser-relevant keys, from
# docs/spike_omniroute_samples.jsonl sample t00 (cold, LATENCY/SEV1/0.95).
VALID_RAW_JSON = """{
  "id": "chatcmpl-cfp-9f9b6afd-dfd",
  "object": "chat.completion",
  "created": 1793962436,
  "model": "deepseek-ai/deepseek-v4-pro-0813",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "{\\n  \\"category\\": \\"LATENCY\\",\\n  \\"severity\\": \\"SEV1\\",\\n  \\"priority_score\\": 0.95,\\n  \\"investigate\\": true,\\n  \\"deployment_relevant\\": true\\n}"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 453,
    "completion_tokens": 540,
    "total_tokens": 993,
    "cached_tokens": 0,
    "prompt_tokens_details": {"cached_tokens": 0},
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0
  }
}"""

VALID_RAW = json.loads(VALID_RAW_JSON)

WRONG_ENUM_RAW = {
    "id": "chatcmpl-test-wrong",
    "object": "chat.completion",
    "created": 1793962436,
    "model": "deepseek-ai/deepseek-v4-pro-0813",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": (
                    '{"category": "Performance Degradation", "severity": "High",'
                    ' "priority_score": 95, "investigate": true,'
                    ' "deployment_relevant": false}'
                ),
            },
        }
    ],
    "usage": {"prompt_tokens": 453, "completion_tokens": 100, "total_tokens": 553},
}

MALFORMED_RAW = {
    "id": "chatcmpl-test-malformed",
    "object": "chat.completion",
    "created": 1793962436,
    "model": "deepseek-ai/deepseek-v4-pro-0813",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "not json at all {{"},
        }
    ],
    "usage": {"prompt_tokens": 453, "completion_tokens": 10, "total_tokens": 463},
}

INVALID_MODEL_BODY = {
    "error": {
        "message": (
            "Model 'auto/does-not-exist-zzz' is not a valid combo or provider. "
            "Unknown built-in auto combo."
        ),
        "type": "invalid_request_error",
        "code": "bad_request",
    }
}

UNAUTHORIZED_BODY = {
    "error": {
        "message": "Invalid API key",
        "type": "invalid_api_key",
        "code": "invalid_api_key",
    }
}


class MockHandler:
    def __init__(self, status=200, payload=None, payload_list=None):
        self.status = status
        self.payload = payload
        self.payload_list = payload_list
        self.calls = 0
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.requests.append(request)
        if self.payload_list is not None:
            payload = self.payload_list[min(self.calls - 1, len(self.payload_list) - 1)]
            return httpx.Response(200, json=payload)
        return httpx.Response(self.status, json=self.payload)


def _features():
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


async def _triage_with_handler(handler, features=None, **kwargs):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    async with client:
        provider = LLMProvider(
            api_key=kwargs.pop("api_key", "test-key"),
            base_url=kwargs.pop("base_url", "http://test/v1"),
            model=kwargs.pop("model", "auto/deepseek"),
            client=client,
            **kwargs,
        )
        return await provider.triage(features or _features())


def test_parse_valid_fixture():
    decision = parse_llm_response(VALID_RAW, latency_ms=42.0)
    assert decision.provider == "llm"
    assert decision.category == "LATENCY"
    assert decision.severity == "SEV1"
    assert decision.priority_score == pytest.approx(0.95)
    assert decision.investigate is True
    assert decision.deployment_relevant is True
    assert decision.model_version == "deepseek-ai/deepseek-v4-pro-0813"
    assert decision.latency_ms == pytest.approx(42.0)
    assert decision.raw_scores["prompt_tokens"] == pytest.approx(453.0)
    assert decision.raw_scores["completion_tokens"] == pytest.approx(540.0)
    assert decision.raw_scores["total_tokens"] == pytest.approx(993.0)
    assert decision.raw_scores["cost_reported"] == pytest.approx(0.0)


def test_parse_wrong_enum_is_provider_error():
    with pytest.raises(ProviderError, match="out-of-vocab"):
        parse_llm_response(WRONG_ENUM_RAW)


def test_parse_malformed_json_is_provider_error():
    with pytest.raises(ProviderError, match="malformed: invalid JSON"):
        parse_llm_response(MALFORMED_RAW)


def test_parse_missing_key_is_provider_error():
    with pytest.raises(ProviderError, match="missing decision key"):
        parse_llm_response({"choices": [{"message": {"content": "{}"}}]})


def test_parse_bad_boolean_is_provider_error():
    raw = json.loads(VALID_RAW_JSON)
    raw["choices"][0]["message"]["content"] = (
        '{"category": "LATENCY", "severity": "SEV1", "priority_score": 0.5,'
        ' "investigate": "yes", "deployment_relevant": true}'
    )
    with pytest.raises(ProviderError, match="malformed"):
        parse_llm_response(raw)


async def test_valid_payload_triage_and_request_shape():
    handler = MockHandler(payload=VALID_RAW)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    async with client:
        provider = LLMProvider(
            api_key="test-key", base_url="http://test/v1", client=client
        )
        features = _features()
        decision = await provider.triage(features)
    assert decision.provider == "llm"
    assert decision.category == "LATENCY"
    assert decision.severity == "SEV1"
    assert decision.priority_score == pytest.approx(0.95)
    assert decision.model_version == "deepseek-ai/deepseek-v4-pro-0813"
    assert decision.raw_scores["prompt_tokens"] == pytest.approx(453.0)
    assert decision.raw_scores["completion_tokens"] == pytest.approx(540.0)
    assert decision.raw_scores["cost_reported"] == pytest.approx(0.0)
    assert decision.latency_ms is not None and decision.latency_ms >= 0.0
    assert handler.calls == 1
    body = json.loads(handler.requests[0].content.decode("utf-8"))
    assert body["model"] == "auto/deepseek"
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    system = body["messages"][0]["content"]
    assert "LATENCY, ERROR_SPIKE, RESOURCE, DEPENDENCY, UNKNOWN" in system
    assert "SEV1, SEV2, SEV3, SEV4" in system
    assert "priority_score MUST be a number between 0 and 1" in system
    user = body["messages"][1]["content"]
    assert render_features(features) in user
    assert "category in [LATENCY,ERROR_SPIKE,RESOURCE,DEPENDENCY,UNKNOWN]" in user


async def test_wrong_enum_retries_once_then_provider_error():
    handler = MockHandler(payload=WRONG_ENUM_RAW)
    with pytest.raises(ProviderError, match="out-of-vocab"):
        await _triage_with_handler(handler)
    assert handler.calls == 2


async def test_malformed_json_retries_once_then_provider_error():
    handler = MockHandler(payload=MALFORMED_RAW)
    with pytest.raises(ProviderError, match="malformed: invalid JSON"):
        await _triage_with_handler(handler)
    assert handler.calls == 2


async def test_invalid_model_400_no_retry():
    handler = MockHandler(status=400, payload=INVALID_MODEL_BODY)
    with pytest.raises(ProviderError, match="llm http 400"):
        await _triage_with_handler(handler)
    assert handler.calls == 1


async def test_unauthorized_401_no_retry():
    handler = MockHandler(status=401, payload=UNAUTHORIZED_BODY)
    with pytest.raises(ProviderError, match="llm http 401"):
        await _triage_with_handler(handler)
    assert handler.calls == 1


async def test_missing_key_is_provider_error_no_http(monkeypatch):
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    handler = MockHandler(payload=VALID_RAW)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    async with client:
        provider = LLMProvider(client=client)
        with pytest.raises(ProviderError, match="missing OMNIROUTE_API_KEY"):
            await provider.triage(_features())
    assert handler.calls == 0


async def test_defaults_match_spec(monkeypatch):
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(MockHandler(payload=VALID_RAW)),
        base_url="http://test/v1",
    )
    async with client:
        provider = LLMProvider(client=client)
    assert provider.base_url == "http://localhost:20128/v1"
    assert provider.model == "auto/deepseek"
    assert provider.timeout_s == 120.0


async def test_plugs_into_triage_with_fallback():
    handler = MockHandler(payload=VALID_RAW)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    async with client:
        provider = LLMProvider(
            api_key="test-key", base_url="http://test/v1", client=client
        )
        decision = await triage_with_fallback(_features(), primary=provider)
    assert decision.provider == "llm"
    assert decision.category == "LATENCY"
