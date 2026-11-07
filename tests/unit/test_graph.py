"""System Two graph with a mocked LLM: transitions, budgets, validators, injection, outages."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    ServiceSignalWindow,
)
from services.investigation.graph import SystemTwo, _free_text_queries
from services.investigation.llm import ChatClient
from services.investigation.toolbox import RoundBudgetExceeded, ToolBox, ToolBudgetExceeded
from services.telemetry.backend import METRIC_ALLOWLIST, SnapshotBackend, parse_ts
from simulator.recorder import forbidden_terms
from tests.llm_doubles import INJECTION, ScriptedLLM

NOW = parse_ts("2026-11-06T03:11:30Z")


def _w(svc: str) -> ServiceSignalWindow:
    return ServiceSignalWindow(
        service=svc, window_start=NOW - timedelta(minutes=1), window_end=NOW, request_rate=15,
        error_rate=0.4, latency_p50_ms=100, latency_p95_ms=2100, latency_p99_ms=2200,
        cpu_utilization=0.3, memory_utilization=0.5)


FEATURES = IncidentFeatures(
    candidate_id="inc-1", opened_at=NOW, primary_service="order-service",
    affected_services=["order-service", "api-gateway"],
    windows=[_w("order-service"), _w("order-service"), _w("api-gateway"), _w("api-gateway")],
    detector_scores={"statistical": 1.0}, anomaly_score=0.8, symptom_tags=["error_up"])


@pytest.fixture
def backend(work):
    return SnapshotBackend(work / "test").at(NOW)


async def _run(llm, backend, **kw):
    return await SystemTwo(llm, **kw).investigate(backend, "inc-1", FEATURES)


async def test_happy_path_state_transitions_and_report(backend):
    llm = ScriptedLLM()
    res = await _run(llm, backend)
    assert llm.roles == ["Context Collector", "Hypothesis Generator", "Evidence Investigator",
                         "RCA Synthesizer"]
    assert [t["agent"] for t in res.extras["agent_trace"]] == [
        "context", "hypotheses", "investigator", "gate", "rca"]
    assert res.outcome == "investigated" and res.report.investigation_complete
    assert "connection pool" in res.report.probable_root_cause
    ids = {e.id for e in res.evidence}
    assert all(set(c.evidence_ids) <= ids and c.evidence_ids
               for c in res.report.claims if c.kind == "fact")
    assert res.report.unresolved_questions
    assert res.extras["calls_per_round"] == {0: 7, 1: 3}
    assert res.extras["llm"]["calls"] == 4 and res.extras["llm"]["prompt_tokens"] > 0


async def test_prompts_carry_no_ground_truth_terms(backend):
    llm = ScriptedLLM()
    await _run(llm, backend)
    text = " ".join(s + u for s, u in llm.prompts)
    assert forbidden_terms(text) == []
    assert "db_connection_pool_exhaustion" not in text and "onset" not in text


async def test_hypothesis_agent_cannot_confirm_a_cause(backend):
    res = await _run(ScriptedLLM("confirm"), backend)
    hyp = next(t for t in res.extras["agent_trace"] if t["agent"] == "hypotheses")
    assert {"confirmed", "status", "root_cause"} <= set(hyp["ignored_confirm_fields"])


async def test_at_most_three_hypotheses(backend):
    res = await _run(ScriptedLLM("five"), backend)
    hyp = next(t for t in res.extras["agent_trace"] if t["agent"] == "hypotheses")
    assert len(hyp["hypotheses"]) == 3 and hyp["dropped"] == 2


async def test_budget_exhaustion_is_inconclusive_never_confident(backend):
    llm = ScriptedLLM("never")
    res = await _run(llm, backend)
    inv = [t for t in res.extras["agent_trace"] if t["agent"] == "investigator"]
    assert len(inv) == 3  # round cap
    assert res.outcome == "inconclusive"
    assert res.report.probable_root_cause is None and res.report.confidence_level == "low"
    assert not res.report.investigation_complete and "incomplete" in res.report.summary


async def test_total_tool_budget_stops_queries(backend):
    res = await _run(ScriptedLLM(), backend, max_calls=1)  # 1 item cannot support a cause
    assert len(res.trace) == 1 and res.outcome == "inconclusive"
    assert res.report is None or res.report.probable_root_cause is None


def test_toolbox_round_and_total_caps(backend):
    box = ToolBox(backend, uuid4(), max_calls=30, round_max=12)
    start, end = backend.start, backend.now
    metrics = sorted(METRIC_ALLOWLIST)
    for _ in range(2):
        box.new_round()
        for i in range(12):
            box.query_metrics("order-service", metrics[i % len(metrics)], start, end)
        with pytest.raises(RoundBudgetExceeded):
            box.query_metrics("order-service", metrics[0], start, end)
    box.new_round()
    for i in range(6):
        box.query_metrics("order-service", metrics[i], start, end)
    with pytest.raises(ToolBudgetExceeded, match="tool call budget of 30"):
        box.query_metrics("order-service", metrics[0], start, end)
    assert box.calls == 30


async def test_rca_validator_rejects_then_retry_succeeds(backend):
    llm = ScriptedLLM("bad_rca_once")
    res = await _run(llm, backend)
    assert llm.rca_calls == 2 and res.outcome == "investigated"


async def test_rca_validator_rejection_twice_fails_explicitly(backend):
    res = await _run(ScriptedLLM("bad_rca_always"), backend)
    assert res.outcome == "inconclusive" and res.report is None
    assert "rejected after 2 attempts" in res.error


async def test_tool_output_injection_is_data_not_instruction(backend):
    class Poisoned(SnapshotBackend):
        def query_logs(self, *a, **k):
            row = {"timestamp": "2026-11-06T03:11:00.000Z", "service": "order-service",
                   "level": "ERROR", "trace_id": "t", "span_id": "s", "event": INJECTION,
                   "duration_ms": 1.0, "message": INJECTION}
            return [row]

    poisoned = Poisoned.__new__(Poisoned)
    poisoned.__dict__.update(backend.__dict__)
    llm = ScriptedLLM("inject")
    res = await _run(llm, poisoned)
    user = next(u for _, u in llm.prompts if "IGNORE" in u)
    assert user.index("<<<DATA") < user.index("IGNORE") < user.index("DATA>>>")
    assert not any(t["tool"] == "restart_service" for t in res.trace)  # requested, never run
    assert len(res.extras["rejected_queries"]) == 2  # unknown tool + raw query text
    assert {t["tool"] for t in res.trace} <= {
        "query_metrics", "query_logs", "query_traces", "get_deployment_events",
        "get_service_dependencies", "search_historical_incidents"}


async def test_llm_outage_ends_inconclusive_without_canned_answer(backend):
    res = await _run(ScriptedLLM("dead"), backend)
    assert res.outcome == "inconclusive" and res.report is None and not res.evidence
    assert "llm unavailable" in res.error


async def test_unparseable_context_reply_fails_inconclusive(backend):
    res = await _run(ScriptedLLM("plain_text"), backend)
    assert res.outcome == "inconclusive" and "no valid query" in res.error


async def test_time_budget_exhaustion_is_inconclusive(backend):
    ticks = iter([0.0] + [999.0] * 100)
    res = await _run(ScriptedLLM(), backend, clock=lambda: next(ticks), deadline_s=180)
    assert res.outcome == "inconclusive" and res.report is None and "time budget" in res.error


async def test_at_most_three_concurrent_investigations(backend):
    llm = ScriptedLLM(delay=0.01)
    s2 = SystemTwo(llm)
    out = await asyncio.gather(*[s2.investigate(backend, f"inc-{i}", FEATURES)
                                 for i in range(6)])
    assert llm.max_active <= 3 and all(o.outcome == "investigated" for o in out)


def test_free_text_tool_call_parser_is_fragile_but_bounded():
    text = "tool_call:\n- name: query_metrics\n  arguments: service: order-service metric: error_rate"
    assert _free_text_queries(text) == [
        {"service": "order-service", "metric": "error_rate", "tool": "query_metrics"}]
    assert _free_text_queries("I think we should look at the database") == []


def _chat(handler, tmp_path=None, key="k"):
    return ChatClient(api_key=key, cache_dir=tmp_path,
                      client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_chat_client_caches_records_usage_and_maps_errors(tmp_path):
    calls = []

    def ok(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "upstream-x", "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            "choices": [{"message": {"content": "{\"a\": 1}"}}]})

    c = _chat(ok, tmp_path)
    first, second = await c.complete("s", "u"), await c.complete("s", "u")
    assert len(calls) == 1 and not first.cached and second.cached
    assert calls[0]["temperature"] == 0 and first.resolved_model == "upstream-x"
    assert second.usage == first.usage and second.text == first.text

    with pytest.raises(ProviderError, match="http 401"):
        await _chat(lambda r: httpx.Response(401)).complete("s", "u")
    with pytest.raises(ProviderError, match="OMNIROUTE_API_KEY"):
        await _chat(ok, key="").complete("s", "u2")
    with pytest.raises(ProviderError, match="unavailable"):
        await _chat(lambda r: httpx.Response(503)).complete("s", "u3")


def test_clock_type_is_utc():
    assert isinstance(NOW, datetime) and NOW.tzinfo == timezone.utc
