"""R0-R3 ablation: variant contracts, fairness and metrics (scripted LLM doubles only)."""

import asyncio
import json
import re

import pytest

import evaluation.f01_runner as runner
from evaluation.ablation import capture, metrics_for, render_markdown, run_ablation
from services.investigation.graph import VARIANTS, SystemTwo
from services.telemetry.backend import SnapshotBackend
from simulator.fault_controller import GroundTruthStore
from simulator.recorder import forbidden_terms
from tests.llm_doubles import FEATURES, NOW, ScriptedLLM


@pytest.fixture
def backend(work):
    return SnapshotBackend(work / "test").at(NOW)


@pytest.fixture(scope="module")
def captured(work):
    return asyncio.run(capture(work))


def _manifest(work):
    return GroundTruthStore(work / "ground_truth").read(runner.RUN_ID)


async def _run(variant, llm, backend, **kw):
    return await SystemTwo(llm, variant=variant, **kw).investigate(backend, "inc-1", FEATURES)


async def test_r0_prompt_is_fixed_and_makes_no_tool_calls(backend):
    a, b = ScriptedLLM(), ScriptedLLM()
    ra, _ = await _run("R0", a, backend), await _run("R0", b, backend)
    assert a.prompts == b.prompts and len(a.prompts) == 1
    assert a.roles == ["RCA Synthesizer"] and ra.trace == []
    system, user = a.prompts[0]
    assert "Tool catalog" not in system and len(re.findall(r"^E\d+ ", user, re.MULTILINE)) == 1
    assert [e.source_type for e in ra.evidence] == ["incident_summary"]


def test_all_variants_share_one_budget():
    budgets = {v: SystemTwo(ScriptedLLM(), variant=v).budget() for v in VARIANTS}
    assert len({json.dumps(b, sort_keys=True) for b in budgets.values()}) == 1
    assert budgets["R1"] == {"max_calls": 30, "round_max": 12, "deadline_s": 180.0,
                             "max_rounds": 3}


@pytest.mark.parametrize("variant", ["R1", "R2", "R3"])
async def test_caps_hold_in_every_tool_variant(backend, variant):
    res = await _run(variant, ScriptedLLM(), backend, max_calls=3, round_max=2)
    assert len(res.trace) <= 3 and max(res.extras["calls_per_round"].values()) <= 2


async def test_r3_with_empty_history_behaves_like_r2(backend):
    r2, r3 = await _run("R2", ScriptedLLM(), backend), await _run("R3", ScriptedLLM(), backend)
    assert [(t["tool"], t["params"]) for t in r2.trace] == [(t["tool"], t["params"])
                                                            for t in r3.trace]
    assert [e.id for e in r2.evidence] == [e.id for e in r3.evidence]
    assert r2.report.model_dump_json() == r3.report.model_dump_json()
    assert r3.extras["history"] == "empty store" and r2.extras["history"] == "n/a"


async def test_history_tool_exists_only_in_r3_and_returns_nothing(backend):
    r2 = await _run("R2", ScriptedLLM("history"), backend)
    r3 = await _run("R3", ScriptedLLM("history"), backend)
    assert not any(t["tool"] == "search_historical_incidents" for t in r2.trace)
    assert len(r2.extras["rejected_queries"]) == 1
    (hit,) = [e for e in r3.evidence if e.source_type == "search_historical_incidents"]
    assert hit.observation == "0 similar past incidents found"  # nothing fabricated


async def test_validator_runs_in_r1_r2_r3_but_not_r0(backend, work):
    r0 = await _run("R0", ScriptedLLM("bad_rca_always"), backend)
    assert r0.report is not None  # baseline keeps its raw, unvalidated answer
    m = metrics_for(r0, _manifest(work))
    assert m["unsupported_claim_rate"] > 0
    for v in ("R1", "R2", "R3"):
        res = await _run(v, ScriptedLLM("bad_rca_always"), backend)
        assert res.report is None and "rejected after 2 attempts" in res.error
        assert res.extras["validator_rejections"] == 1


async def test_metrics_for_multi_agent_run(backend, work):
    m = metrics_for(await _run("R2", ScriptedLLM(), backend), _manifest(work))
    assert m["exact_fault"] and m["category_correct"] and m["completed"]
    assert m["evidence_precision"] == 1.0 and m["evidence_recall"] >= 0.8
    assert m["unsupported_claim_rate"] == 0.0 and m["alt_coverage"] == 0.5
    assert m["cost"] == "not reported" and m["cache_state"] == "cold"
    assert m["tokens"] > 0 and m["llm_calls"] == 4 and m["resolved_upstream"] == ["mock-upstream-1"]


async def test_prompts_of_every_variant_are_free_of_ground_truth(backend):
    for v in VARIANTS:
        llm = ScriptedLLM()
        await _run(v, llm, backend)
        text = " ".join(s + u for s, u in llm.prompts)
        assert forbidden_terms(text) == [] and "db_connection_pool_exhaustion" not in text


async def test_run_ablation_fairness_and_report(captured, work):
    made: list[tuple[str, int]] = []

    def factory(variant, rep):
        made.append((variant, rep))
        return ScriptedLLM()

    rep = await run_ablation(captured, lambda: _manifest(work), factory, repeats=3, real=True)
    assert made == [(v, r) for v in VARIANTS for r in range(3)]  # one fresh LLM per run
    assert set(rep["summary"]) == set(VARIANTS)
    assert rep["summary"]["R2"]["exact_fault"] == "3/3"
    assert rep["summary"]["R0"]["tool_calls"]["max"] == 0
    assert rep["summary"]["R1"]["tool_calls"]["max"] <= 30
    assert "first real-model comparison" in rep["notice"]
    assert "indicative, not conclusive" in rep["verdict"]
    assert "prompt" not in json.dumps(rep, default=str).replace("prompt_tokens", "")
    md = render_markdown(rep)
    assert "| R3 |" in md and "Verdict (RQ3" in md


async def test_report_without_real_model_says_so(captured, work):
    rep = await run_ablation(captured, lambda: _manifest(work), lambda v, r: ScriptedLLM(),
                             repeats=1, real=False)
    assert "NO REAL-MODEL EVIDENCE" in rep["notice"] and "cannot be answered" in rep["verdict"]
