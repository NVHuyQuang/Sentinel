"""Evidence validator, tool budgets and the no-future-reads rule."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from services.decision_engine.providers.base import Claim, Evidence, RCAReport
from services.investigation.rca import ReportValidationError, finalize_report, validate_report
from services.investigation.toolbox import ToolBox, ToolBudgetExceeded
from services.telemetry.backend import QueryRejected, SnapshotBackend, parse_ts

NOW = datetime(2026, 11, 6, 3, 10, tzinfo=timezone.utc)
INV = uuid4()


def _ev(obs="order-service db_pool_utilization: mean 0.15 before, 1.00 over the last 30 s"):
    return Evidence(id=uuid4(), investigation_id=INV, source_type="query_metrics",
                    service="order-service", observation=obs, timestamp=NOW,
                    source_reference="snapshot://x", query_parameters={}, supports=[],
                    contradicts=[])


def _report(claims, supporting=()):
    return RCAReport(incident_id=uuid4(), summary="s", affected_services=["order-service"],
                     probable_root_cause="x", confidence_level="low", claims=claims,
                     supporting_evidence_ids=list(supporting), contradicting_evidence_ids=[],
                     timeline=[], alternative_hypotheses=[], recommendations=[],
                     unresolved_questions=[], investigation_complete=True)


def test_valid_report_passes():
    e = _ev()
    validate_report(_report([Claim(id="c1", text=e.observation, kind="fact",
                                   evidence_ids=[e.id])], [e.id]), [e], INV)


def test_fact_without_evidence_is_rejected():
    with pytest.raises(ReportValidationError, match="fact without evidence"):
        validate_report(_report([Claim(id="c1", text="pool full", kind="fact",
                                       evidence_ids=[])]), [_ev()], INV)


def test_fact_with_unknown_evidence_is_rejected():
    with pytest.raises(ReportValidationError, match="unknown evidence"):
        validate_report(_report([Claim(id="c1", text="x", kind="fact",
                                       evidence_ids=[uuid4()])]), [_ev()], INV)


def test_fact_with_unsupported_number_is_rejected():
    e = _ev()
    with pytest.raises(ReportValidationError, match="not in linked evidence"):
        validate_report(_report([Claim(id="c1", text="pool utilization was 0.42", kind="fact",
                                       evidence_ids=[e.id])]), [e], INV)


def test_hypothesis_may_have_no_evidence():
    validate_report(_report([Claim(id="c1", text="maybe", kind="hypothesis",
                                   evidence_ids=[])]), [_ev()], INV)


def test_finalize_retries_once_then_fails_explicitly():
    calls = []

    def bad():
        calls.append(1)
        return _report([Claim(id="c1", text="x", kind="fact", evidence_ids=[])])

    with pytest.raises(ReportValidationError, match="after 2 attempts"):
        finalize_report(bad, [_ev()], INV)
    assert len(calls) == 2


def test_toolbox_enforces_call_budget_and_deadline(work):
    be = SnapshotBackend(work / "test").at(parse_ts("2026-11-06T03:10:00Z"))
    box = ToolBox(be, INV, max_calls=2)
    for _ in range(2):
        box.get_service_dependencies("order-service")
    with pytest.raises(ToolBudgetExceeded, match="call budget"):
        box.get_service_dependencies("order-service")
    ticks = iter([0.0, 181.0])
    slow = ToolBox(be, INV, clock=lambda: next(ticks))
    with pytest.raises(ToolBudgetExceeded, match="time budget"):
        slow.get_service_dependencies("order-service")


def test_toolbox_has_no_remediation_tools():
    public = {n for n in dir(ToolBox) if not n.startswith("_")}
    assert public == {"query_metrics", "query_logs", "query_traces", "get_deployment_events",
                      "get_service_dependencies", "record", "now"}


def test_backend_never_reads_the_future_and_enforces_limits(work, monkeypatch):
    be = SnapshotBackend(work / "test").at(parse_ts("2026-11-06T03:10:00Z"))
    pts = be.query_metrics("order-service", "db_pool_utilization", be.start, be.end)
    assert pts and max(parse_ts(p["ts"]) for p in pts) <= be.now
    with pytest.raises(QueryRejected, match="after the investigation clock"):
        be.query_metrics("order-service", "error_rate", be.now + timedelta(minutes=1),
                         be.now + timedelta(minutes=2))
    with pytest.raises(QueryRejected, match="allowlisted"):
        be.query_metrics("fault-controller", "error_rate", be.start, be.now)
    with pytest.raises(QueryRejected, match="allowlisted"):
        be.query_metrics("order-service", "sum(rate(x[5m]))", be.start, be.now)
    assert len(be.query_logs("order-service", be.start, be.now, limit=10_000)) <= 100
    monkeypatch.setattr("services.telemetry.backend.MAX_RANGE", timedelta(minutes=5))
    with pytest.raises(QueryRejected, match="time range"):
        SnapshotBackend(work / "test").query_metrics("order-service", "error_rate", be.start,
                                                     be.end)
