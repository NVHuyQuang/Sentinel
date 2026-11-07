"""F01 end to end on recorded replay: identical evidence every run (SnapshotBackend)."""

import asyncio
import json

import pytest

import evaluation.f01_runner as runner
from evaluation.f01_runner import StubLayaRouter, record_all
from evaluation.score import evaluate
from services.decision_engine.providers.base import ProviderError
from services.decision_engine.providers.laya import LayaAdapter
from services.decision_engine.providers.rules import RulesProvider
from services.investigation.stub import investigate
from services.pipeline import fit_detectors, run_pipeline
from services.telemetry.backend import SnapshotBackend
from simulator.fault_controller import GroundTruthStore


@pytest.fixture(scope="module")
def replay(work):
    train, test = SnapshotBackend(work / "train"), SnapshotBackend(work / "test")
    laya = LayaAdapter(device="cpu", timeout_ms=5000, router=StubLayaRouter())
    result = asyncio.run(
        run_pipeline(test, fit_detectors(train), laya, RulesProvider(), investigate))
    return test, result


def _manifest(work):
    return GroundTruthStore(work / "ground_truth").read(runner.RUN_ID)


def test_detection_recall_and_delay(work, replay):
    test, result = replay
    rep = evaluate(_manifest(work), test, result, "stub-laya")
    assert rep["verdict"]["detected"] and rep["verdict"]["no_false_positives"]
    assert rep["detection_delay_s"] <= 90
    assert rep["verdict"]["chain_order_matches"]


def test_triage_routes_through_laya_with_severity_floor(replay):
    _, result = replay
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
    result = await run_pipeline(test, fit_detectors(train), laya, RulesProvider(), investigate)
    (dec,) = result.triage.values()
    assert dec.provider == "rules" and dec.category == "RESOURCE"


def test_rca_is_evidence_backed_and_matches_hidden_truth(work, replay):
    test, result = replay
    (inv,) = result.investigations.values()
    assert inv.outcome == "investigated" and len(inv.trace) <= 12
    ids = {e.id for e in inv.evidence}
    facts = [c for c in inv.report.claims if c.kind == "fact"]
    assert facts and all(c.evidence_ids and set(c.evidence_ids) <= ids for c in facts)
    rep = evaluate(_manifest(work), test, result, "stub-laya")
    assert rep["verdict"]["root_cause_match"] and rep["advisory"]["evidence_precision_1"]


def test_lifecycle_resolves_after_five_clean_minutes_and_closes(work, replay):
    test, result = replay
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
        res = await run_pipeline(test, fit_detectors(train), RulesProvider(), None, investigate)
        (inv,) = res.investigations.values()
        return inv.report.model_dump_json(), [e.model_dump_json() for e in inv.evidence]

    assert asyncio.run(once(work)) == asyncio.run(once(again))


async def test_ground_truth_read_only_after_investigation(work, monkeypatch):
    events: list[str] = []
    real_read = GroundTruthStore.read

    def read(self, run_id):
        events.append("manifest_read")
        return real_read(self, run_id)

    def spy_investigate(*a, **k):
        out = investigate(*a, **k)
        events.append("investigation_done")
        return out

    monkeypatch.setattr(GroundTruthStore, "read", read)
    monkeypatch.setattr(runner, "investigate", spy_investigate)
    out = work / "reports"
    report = await runner.run(work, out)
    assert events == ["investigation_done", "manifest_read"]
    assert report["verdict"]["all_pass"]
    assert json.loads((out / "f01_report.json").read_text())["verdict"]["detected"]
    assert (out / "f01_report.md").exists()


def test_time_boxed_investigation_is_inconclusive(work, replay):
    test, result = replay
    feats = next(iter(result.features.values()))
    ticks = iter([0.0] + [500.0] * 20)  # clock jumps past the 180 s cap after the first call
    out = investigate(test.at(test.end), "inc-1", feats, clock=lambda: next(ticks))
    assert out.outcome == "inconclusive" and out.report is None and "time budget" in out.error
