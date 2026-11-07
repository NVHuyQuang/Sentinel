"""STUB investigation: a scripted Context Collector plus a rule-based RCA. No LLM is called.

This stands in for the System Two agents (Spec 6) so the pipeline can be proven end to end.
It is deterministic on purpose: every replay produces identical evidence and report.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from services.decision_engine.providers.base import (
    Claim,
    Evidence,
    IncidentFeatures,
    RCAReport,
)
from services.investigation.rca import ReportValidationError, finalize_report
from services.investigation.toolbox import ToolBox, ToolBudgetExceeded
from services.telemetry.backend import QueryRejected, SnapshotBackend

LOOKBACK = timedelta(minutes=10)
STUB_MARKER = "stub-rules-v0 (no LLM)"


@dataclass
class InvestigationResult:
    incident_id: str
    outcome: str  # investigated | inconclusive
    report: RCAReport | None
    evidence: list[Evidence]
    trace: list[dict]
    error: str | None = None
    marker: str = STUB_MARKER
    extras: dict = field(default_factory=dict)


def _shift(points: list) -> tuple[float, float] | None:
    """(mean before, mean over the last two scrapes). None when there is too little data."""
    vals = [p["value"] for p in points]
    if len(vals) < 6:
        return None
    return sum(vals[:-4]) / len(vals[:-4]), sum(vals[-2:]) / 2


def _metric_obs(service: str, metric: str, shift) -> str:
    if shift is None:
        return f"{service} {metric}: not enough samples"
    return f"{service} {metric}: mean {shift[0]:.2f} before, {shift[1]:.2f} over the last 30 s"


def _target_service(features: IncidentFeatures) -> str:
    pooled = [w for w in features.windows if w.db_pool_utilization is not None]
    if pooled:
        return max(pooled, key=lambda w: w.db_pool_utilization).service
    return features.primary_service


def _collect(box: ToolBox, target: str) -> dict:
    """Scripted plan, 10 tool calls. Returns evidence + shifts keyed by purpose."""
    end, start = box.now, box.now - LOOKBACK
    got: dict = {}
    deps = box.get_service_dependencies(target)
    got["deps"] = box.record(deps, target, f"{target} depends on {', '.join(deps.data)}", [], [])
    plan = [
        ("pool", target, "db_pool_utilization", ["db_pool_exhaustion"], []),
        ("wait", target, "db_query_wait_ms_p95", ["db_pool_exhaustion"], []),
        ("latency", target, "latency_p95_ms", ["db_pool_exhaustion"], []),
        ("errors", target, "error_rate", ["db_pool_exhaustion"], []),
        ("gateway", "api-gateway", "error_rate", ["db_pool_exhaustion"], []),
        ("cpu", target, "cpu_utilization", [], ["cpu_saturation"]),
    ]
    for key, svc, metric, sup, con in plan:
        res = box.query_metrics(svc, metric, start, end)
        shift = _shift(res.data)
        if key == "cpu":  # cpu evidence only contradicts when it stayed low
            con = con if shift and shift[1] < 0.8 else []
        got[key] = box.record(res, svc, _metric_obs(svc, metric, shift), sup, con)
        got[key + "_shift"] = shift
    res = box.query_logs(target, start, end, level="ERROR", limit=50)
    counts: dict[str, int] = {}
    for r in res.data:
        counts[r["event"]] = counts.get(r["event"], 0) + 1
    top = max(counts.items(), key=lambda kv: kv[1], default=None)
    obs = (f"{target} logged {len(res.data)} ERROR entries; most frequent event "
           f"{top[0]} ({top[1]})" if top else f"{target} logged 0 ERROR entries")
    got["logs_top"] = top[0] if top else None
    got["logs"] = box.record(res, target, obs,
                             ["db_pool_exhaustion"] if top and "pool" in top[0] else [], [])
    res = box.query_traces(target, start, end, errors_only=True)
    names: dict[str, int] = {}
    for s in res.data:
        names[s["name"]] = names.get(s["name"], 0) + 1
    topn = max(names.items(), key=lambda kv: kv[1], default=None)
    obs = (f"{target} has {len(res.data)} error spans; most frequent span {topn[0]} ({topn[1]})"
           if topn else f"{target} has 0 error spans")
    got["traces"] = box.record(res, target, obs, [], [])
    res = box.get_deployment_events(target, start, end)
    got["deploy"] = box.record(
        res, target, f"{target} has {len(res.data)} releases in the last 10 minutes", [],
        ["application_regression"] if not res.data else [])
    return got


def _report(incident_uuid: UUID, target: str, got: dict) -> RCAReport:
    pool, wait, lat = got["pool_shift"], got["wait_shift"], got["latency_shift"]
    gw = got["gateway_shift"]
    saturated = bool(pool and pool[1] >= 0.9 and pool[0] < 0.5)
    waiting = bool(wait and wait[1] >= 500 and wait[1] > 3 * wait[0])
    slow = bool(lat and lat[1] >= 2 * lat[0])
    failing = bool(gw and gw[1] >= 0.05)
    claims: list[Claim] = []

    def fact(key: str) -> str:
        cid = f"c{len(claims) + 1}"
        claims.append(Claim(id=cid, text=got[key].observation, kind="fact",
                            evidence_ids=[got[key].id]))
        return cid

    chain = [fact(k) for k in ("pool", "wait", "latency", "errors", "gateway")]
    fact("logs"), fact("deploy"), fact("cpu")
    root = None
    if saturated and waiting and (slow or failing):
        root = f"Database connection pool exhaustion on {target}"
        claims.append(Claim(
            id=f"c{len(claims) + 1}", kind="inference",
            text=("Pool saturation made requests wait for a connection, raising latency; "
                  "acquire timeouts surfaced as 5xx at api-gateway"),
            evidence_ids=[got[k].id for k in ("pool", "wait", "latency", "gateway")]))
    supporting = [got[k].id for k in ("pool", "wait", "latency", "errors", "gateway", "logs")
                  if "db_pool_exhaustion" in got[k].supports] if root else []
    contradicting = [got[k].id for k in ("cpu", "deploy") if got[k].contradicts]
    return RCAReport(
        incident_id=incident_uuid,
        summary=(f"{root}; latency and errors followed pool saturation." if root
                 else "Evidence did not establish a root cause."),
        affected_services=sorted({target, "api-gateway"}),
        probable_root_cause=root,
        confidence_level="high" if root and saturated and waiting and slow and failing
        else ("medium" if root else "low"),
        claims=claims,
        supporting_evidence_ids=supporting,
        contradicting_evidence_ids=contradicting,
        timeline=[{"order": i + 1, "claim_ids": [c]} for i, c in enumerate(chain)],
        alternative_hypotheses=[
            {"hypothesis": "CPU saturation", "status": "contradicted",
             "evidence_ids": [str(got["cpu"].id)]},
            {"hypothesis": "Application regression after a release",
             "status": "no release in window", "evidence_ids": [str(got["deploy"].id)]},
        ],
        # Text only. No remediation tool exists, so nothing here can be executed.
        recommendations=[
            f"Inspect sessions holding {target} database connections for stuck transactions.",
            f"Review recent changes to the {target} connection pool configuration.",
        ],
        unresolved_questions=[] if root else ["Which component is the bottleneck?"],
        investigation_complete=root is not None,
    )


def investigate(backend: SnapshotBackend, incident_id: str, features: IncidentFeatures,
                clock: Callable[[], float] = time.monotonic) -> InvestigationResult:
    inv_id = uuid5(NAMESPACE_URL, f"{backend.snapshot_id}/investigation/{incident_id}")
    inc_uuid = uuid5(NAMESPACE_URL, f"{backend.snapshot_id}/incident/{incident_id}")
    box = ToolBox(backend, inv_id, clock=clock)
    target = _target_service(features)
    try:
        got = _collect(box, target)
        report = finalize_report(lambda: _report(inc_uuid, target, got), box.evidence, inv_id)
    except (ToolBudgetExceeded, QueryRejected, ReportValidationError) as exc:
        return InvestigationResult(incident_id, "inconclusive", None, box.evidence, box.trace,
                                   error=f"{type(exc).__name__}: {exc}")
    outcome = "investigated" if report.investigation_complete else "inconclusive"
    return InvestigationResult(incident_id, outcome, report, box.evidence, box.trace)
