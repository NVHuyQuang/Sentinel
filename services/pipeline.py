"""Detection -> triage -> investigation -> lifecycle driver (Spec 4.1-4.5, 5.3, 10).

Walks a recorded run at the 30 s evaluation cadence. At each instant the backend clock is set
to that instant, so nothing after "now" is ever visible (no future reads). Per tick:
windows -> 3 detectors -> ensemble -> hysteresis gate -> grouper -> IncidentFeatures ->
policy.triage_with_fallback -> should_investigate -> investigation -> auto-resolve -> close.

This module never imports the simulator or the ground-truth store.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from services.decision_engine.policy import should_investigate, triage_with_fallback
from services.decision_engine.providers.base import (
    DecisionProvider,
    IncidentFeatures,
    ServiceSignalWindow,
    TriageDecision,
)
from services.investigation.stub import InvestigationResult
from services.signal_processor.detectors import (
    CusumDetector,
    IsolationForestDetector,
    RobustBaselineDetector,
)
from services.signal_processor.ensemble import (
    CandidateGate,
    compute_ensemble_score,
    critical_rule_fires,
)
from services.signal_processor.features.windows import (
    EVAL_CADENCE_SECONDS,
    build_evaluation_windows,
    evaluation_times,
)
from services.signal_processor.incident_grouper import AnomalySignal, IncidentGrouper
from services.telemetry.backend import SnapshotBackend

SETTLE = timedelta(seconds=60)  # triage waits so the 1-min window is not half pre-onset
WARMUP = timedelta(seconds=60)  # first 1-min window needs a full minute of scrapes
CONTEXT = timedelta(minutes=5)


@dataclass
class ServiceDetectors:
    robust: RobustBaselineDetector
    isolation: IsolationForestDetector
    change: CusumDetector


def fit_detectors(training: SnapshotBackend) -> dict[str, ServiceDetectors]:
    """Fit per-service detectors on a normal-only run (Training split), never on the test run."""
    scrapes = training.scrapes(training.end)
    out: dict[str, ServiceDetectors] = {}
    for svc in sorted(training.services):
        wins = [build_evaluation_windows(svc, scrapes, t)[0]
                for t in evaluation_times(training.start + WARMUP, training.end,
                                          EVAL_CADENCE_SECONDS)]
        change = CusumDetector(drift=1.0, cap=6.0).fit(wins)
        change.sigma = max(change.sigma, 0.01)  # floor: error-rate wobble under 1 pp is noise
        out[svc] = ServiceDetectors(RobustBaselineDetector().fit(wins),
                                    IsolationForestDetector(random_state=0).fit(wins), change)
    return out


def _tags(short: ServiceSignalWindow, det: ServiceDetectors) -> list[str]:
    tags = []
    base = det.robust.baseline.get("latency_p95_ms")
    if base and short.latency_p95_ms >= 2 * base[0]:
        tags.append("latency_up")
    if short.error_rate >= 0.05:
        tags.append("error_up")
    if short.db_pool_utilization is not None and short.db_pool_utilization >= 0.85:
        tags.append("pool_saturated")
    if short.dependency_timeout_rate is not None and short.dependency_timeout_rate >= 0.05:
        tags.append("dependency_timeout")
    return tags


@dataclass
class RunResult:
    incidents: list = field(default_factory=list)
    triage: dict[str, TriageDecision] = field(default_factory=dict)
    features: dict[str, IncidentFeatures] = field(default_factory=dict)
    investigations: dict[str, InvestigationResult] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)  # execution trace, append-only


async def run_pipeline(
    backend: SnapshotBackend,
    detectors: dict[str, ServiceDetectors],
    primary: DecisionProvider,
    fallback: DecisionProvider | None,
    investigate: Callable[[SnapshotBackend, str, IncidentFeatures], InvestigationResult],
) -> RunResult:
    grouper = IncidentGrouper()
    gate = CandidateGate()
    res = RunResult()
    pending: dict[str, datetime] = {}  # incident id -> triage due time

    def log(t: datetime, event: str, **kw) -> None:
        res.trace.append({"t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "event": event, **kw})

    for now in evaluation_times(backend.start + WARMUP, backend.end, EVAL_CADENCE_SECONDS):
        view = backend.at(now)
        scrapes = view.scrapes(now)
        wins: dict[str, tuple[ServiceSignalWindow, ServiceSignalWindow]] = {}
        scores: dict[str, tuple[float, float, float, float]] = {}
        signals: list[AnomalySignal] = []
        for svc in sorted(view.services):
            short, ctx = build_evaluation_windows(svc, scrapes, now)
            d = detectors[svc]
            stat, iso = d.robust.score_window(short), d.isolation.score_window(short)
            chg = d.change.update_window(short)
            score = compute_ensemble_score(stat, iso, chg)
            crit = critical_rule_fires(short)
            wins[svc], scores[svc] = (short, ctx), (score, stat, iso, chg)
            if gate.evaluate(svc, score, crit, now):
                signals.append(AnomalySignal(svc, now, score, stat, iso, chg, crit,
                                             symptom_tags=tuple(_tags(short, d))))
        for sig in sorted(signals, key=lambda s: -s.score):
            log(now, "anomaly_candidate", service=sig.service, score=round(sig.score, 3),
                critical=sig.critical_rule_triggered)
            inc = grouper.ingest(sig)
            if inc.incident_id not in res.features and inc.incident_id not in pending:
                pending[inc.incident_id] = now + SETTLE
                log(now, "incident_opened", incident=inc.incident_id, service=sig.service)
        for inc in grouper.list_incidents():
            if inc.status == "closed":
                continue
            worst = max(scores[s][0] for s in inc.affected_services)
            crit = any(critical_rule_fires(wins[s][0]) for s in inc.affected_services)
            was_resolved = inc.resolved_at is not None
            grouper.record_evaluation(inc.incident_id, now, worst, crit)
            if inc.resolved_at is not None and not was_resolved:
                log(now, "incident_resolved", incident=inc.incident_id)
            if pending.get(inc.incident_id, now + SETTLE) > now:
                continue
            del pending[inc.incident_id]
            feats = _features(inc, wins, scores, view, grouper, now, detectors)
            res.features[inc.incident_id] = feats
            dec = await triage_with_fallback(feats, primary, fallback)
            res.triage[inc.incident_id] = dec
            grouper.transition(inc.incident_id, "triaged", now)
            log(now, "triaged", incident=inc.incident_id, services=feats.affected_services,
                provider=dec.provider, category=dec.category, severity=dec.severity)
            if should_investigate(dec, feats.anomaly_score, feats.critical_rule_triggered):
                grouper.transition(inc.incident_id, "investigating", now)
                log(now, "investigation_started", incident=inc.incident_id)
                out = investigate(view, inc.incident_id, feats)
                res.investigations[inc.incident_id] = out
                grouper.complete_investigation(inc.incident_id, out.outcome)
                log(now, "investigation_done", incident=inc.incident_id, outcome=out.outcome,
                    tool_calls=len(out.trace), error=out.error)
            else:
                grouper.complete_investigation(inc.incident_id, skipped=True)
                log(now, "investigation_skipped", incident=inc.incident_id)
        for inc in grouper.list_incidents():
            if inc.status != "closed" and grouper.can_close(inc.incident_id):
                grouper.close(inc.incident_id, now)
                log(now, "incident_closed", incident=inc.incident_id)
    res.incidents = grouper.list_incidents()
    return res


def _features(inc, wins, scores, view: SnapshotBackend, grouper: IncidentGrouper,
              now: datetime, detectors: dict[str, ServiceDetectors]) -> IncidentFeatures:
    svcs = list(inc.affected_services)
    sig = {s.service: s for s in inc.signals}
    edges = [(a, b) for a in svcs for b in svcs if b in grouper.dependencies.get(a, ())]
    return IncidentFeatures(
        candidate_id=inc.incident_id,
        opened_at=inc.opened_at,
        primary_service=inc.primary_service,
        affected_services=svcs,
        windows=[w for s in svcs for w in wins[s]],
        detector_scores={
            "statistical": max(scores[s][1] for s in svcs),
            "isolation": max(scores[s][2] for s in svcs),
            "change_point": max(scores[s][3] for s in svcs),
        },
        anomaly_score=max(scores[s][0] for s in svcs),
        critical_rule_triggered=any(sig[s].critical_rule_triggered for s in sig),
        deployment_services=[s for s in view.recent_deployments(now, CONTEXT) if s in svcs],
        dependency_edges=edges,
        # tags are re-read from the windows at triage time, not frozen at first detection
        symptom_tags=sorted(inc.symptom_tags | {t for s in svcs for t in _tags(wins[s][0],
                                                                               detectors[s])}),
    )
