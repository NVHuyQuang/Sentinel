"""F01 end to end on recorded replay with System Two (mocked LLM; identical evidence every run)."""

import asyncio
import json

import pytest

import evaluation.f01_runner as runner
from evaluation.f01_runner import StubLayaRouter, record_all
from evaluation.score import evaluate
from services.decision_engine.providers.base import ProviderError
from services.decision_engine.providers.laya import LayaAdapter
from services.decision_engine.providers.rules import RulesProvider
from services.investigation.graph import SystemTwo
from services.investigation.llm import ChatClient
from services.pipeline import fit_detectors, run_pipeline
from services.telemetry.backend import SnapshotBackend
from simulator.fault_controller import GroundTruthStore
from tests.llm_doubles import ScriptedLLM


def _laya():
    return LayaAdapter(device="cpu", timeout_ms=5000, router=StubLayaRouter())


@pytest.fixture(scope="module")
def replay(work):
    train, test = SnapshotBackend(work / "train"), SnapshotBackend(work / "test")
    llm = ScriptedLLM()
    result = asyncio.run(run_pipeline(test, fit_detectors(train), _laya(), RulesProvider(),
                                      SystemTwo(llm).investigate))
    return test, result, llm


def _manifest(work):
    return GroundTruthStore(work / "ground_truth").read(runner.RUN_ID)


def test_detection_recall_and_delay(work, replay):
    test, result, _ = replay
    rep = evaluate(_manifest(work), test, result, "stub-laya")
    assert rep["verdict"]["detected"] and rep["verdict"]["no_false_positives"]
    assert rep["detection_delay_s"] <= 90
    assert rep["verdict"]["chain_order_matches"]


def test_triage_routes_through_laya_with_severity_floor(replay):
    _, result, _ = replay
    (dec,) = result.triage.values()
    assert dec.provider == "laya"  # primary answered; stub Laya says SEV4
    assert dec.severity in ("SEV1", "SEV2")  # rule floor raised it (core error rate >= 0.2)
    feats = next(iter(result.features.values()))
    assert {"order-service", "api-gateway"} <= set(feats.affected_services)
    assert "pool_saturated" in feats.symptom_tags


async def test_dead_primary_falls_back_to_rules(work):
    class Dead(StubLayaRouter):
        calls = 0

        def predict(self, text, questions):  # answers the startup warmup, then dies
            Dead.calls += 1
            if Dead.calls == 1:
                return super().predict(text, questions)
            raise ProviderError("down")

    train, test = SnapshotBackend(work / "train"), SnapshotBackend(work / "test")
    laya = LayaAdapter(device="cpu", timeout_ms=5000, router=Dead())
    result = await run_pipeline(test, fit_detectors(train), laya, RulesProvider(),
                                SystemTwo(ScriptedLLM()).investigate)
    (dec,) = result.triage.values()
    assert dec.provider == "rules" and dec.category == "RESOURCE"


def test_multi_agent_rca_matches_hidden_truth_and_records_cost(work, replay):
    test, result, llm = replay
    (inv,) = result.investigations.values()
    assert inv.outcome == "investigated"
    assert inv.report.probable_root_cause == "Database connection pool exhaustion on order-service"
    assert llm.roles == ["Context Collector", "Hypothesis Generator", "Evidence Investigator",
                         "RCA Synthesizer"]
    assert len(inv.trace) <= 30 and max(inv.extras["calls_per_round"].values()) <= 12
    ids = {e.id for e in inv.evidence}
    facts = [c for c in inv.report.claims if c.kind == "fact"]
    assert facts and all(c.evidence_ids and set(c.evidence_ids) <= ids for c in facts)
    rep = evaluate(_manifest(work), test, result, "stub-laya")
    assert rep["verdict"]["root_cause_match"] and rep["verdict"]["facts_all_linked"]
    assert rep["rca"]["evidence_precision"] == 1.0 and rep["rca"]["evidence_recall"] >= 0.8
    i = rep["investigation"]
    assert i["duration_s"] >= 0 and i["llm"]["prompt_tokens"] > 0 and i["llm"]["calls"] == 4
    assert i["llm"]["route"] == "mock-route" and i["llm"]["cost"] == "not reported"


def test_lifecycle_resolves_after_five_clean_minutes_and_closes(work, replay):
    test, result, _ = replay
    (inc,) = result.incidents
    assert inc.status == "closed" and inc.resolved_at is not None
    rep = evaluate(_manifest(work), test, result, "stub-laya")
    assert rep["verdict"]["resolved_after_recovery"]
    assert rep["resolve_delay_after_recovery_s"] >= 300  # 10 clean evaluations at 30 s
    events = [e["event"] for e in result.trace]
    assert events.index("triaged") < events.index("investigation_done") < events.index(
        "incident_resolved") < events.index("incident_closed")


def test_replay_is_identical_across_runs(work, tmp_path):
    again = tmp_path / "again"
    record_all(again)
    for name in ("train", "test"):
        assert (work / name / "checksums.json").read_text() == (
            again / name / "checksums.json").read_text()

    async def once(root):
        train, test = SnapshotBackend(root / "train"), SnapshotBackend(root / "test")
        res = await run_pipeline(test, fit_detectors(train), RulesProvider(), None,
                                 SystemTwo(ScriptedLLM()).investigate)
        (inv,) = res.investigations.values()
        return inv.report.model_dump_json(), [e.model_dump_json() for e in inv.evidence]

    assert asyncio.run(once(work)) == asyncio.run(once(again))


async def test_ground_truth_read_only_after_investigation(work, monkeypatch):
    events: list[str] = []
    real_read = GroundTruthStore.read

    def read(self, run_id):
        events.append("manifest_read")
        return real_read(self, run_id)

    class Spy(SystemTwo):
        async def investigate(self, *a, **k):
            out = await super().investigate(*a, **k)
            events.append("investigation_done")
            return out

    monkeypatch.setattr(GroundTruthStore, "read", read)
    out = work / "reports"
    report = await runner.run(work, out, investigator=Spy(ScriptedLLM()))
    assert events == ["investigation_done", "manifest_read"]
    assert report["verdict"]["all_pass"]
    assert json.loads((out / "f01_report.json").read_text())["investigation"]["outcome"] == (
        "investigated")
    assert "investigation: investigated" in (out / "f01_report.md").read_text()


async def test_unreachable_omniroute_ends_inconclusive_and_report_says_so(work, tmp_path):
    llm = ChatClient(api_key="k", base_url="http://127.0.0.1:9/v1", timeout_s=2.0)
    report = await runner.run(work, tmp_path / "r", investigator=SystemTwo(llm))
    assert report["investigation"]["outcome"] == "inconclusive"
    assert "llm unavailable" in report["investigation"]["error"]
    assert not report["verdict"]["rca_produced"] and not report["verdict"]["all_pass"]
    assert "rca" not in report  # no canned root cause
    assert "llm unavailable" in (tmp_path / "r" / "f01_report.md").read_text()
