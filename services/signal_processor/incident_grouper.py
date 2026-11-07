"""Grouping + lifecycle (Spec 4.4, 4.5).

Correlated candidates collapse into one incident. Signals join the
same incident when they fall within the 3-minute grouping window and
share at least one grouping cue: same service, a dependency link, a
shared trace id, or an overlapping symptom tag. When neither side
carries trace or symptom detail, temporal proximity alone groups, so
an information-free burst prefers one incident over a flood of
duplicates.

Lifecycle: open -> triaged -> investigating -> investigated or
inconclusive -> closed. An incident auto-resolves after 10
consecutive evaluations (5 minutes at 30s cadence) with no active
anomaly below the clear level. An anomaly within 10 minutes of the
resolve instant re-opens the same incident; a later one opens a new
incident. Closing needs the resolve instant plus a finished or
skipped investigation.
"""

from dataclasses import dataclass
from datetime import datetime

GROUPING_WINDOW_SECONDS = 180
REOPEN_WINDOW_SECONDS = 600
CLEAN_EVALS_TO_RESOLVE = 10
CLEAN_EVAL_INTERVAL_SECONDS = 30
CLEAR_THRESHOLD = 0.45

_ALLOWED_NEXT = {
    "open": {"triaged", "investigating", "investigated", "inconclusive", "closed"},
    "triaged": {"investigating", "investigated", "inconclusive", "closed"},
    "investigating": {"investigated", "inconclusive", "closed"},
    "investigated": {"closed"},
    "inconclusive": {"closed"},
    "closed": set(),
}

_BUILTIN_DEPENDENCIES = {
    "api-gateway": ("auth-service", "order-service"),
    "auth-service": ("postgres",),
    "order-service": ("payment-service", "postgres", "redis"),
    "payment-service": ("redis", "provider-simulator"),
}


@dataclass
class AnomalySignal:
    """One detector emission for one service at one instant."""

    service: str
    timestamp: datetime
    score: float
    statistical: float = 0.0
    isolation: float = 0.0
    change: float = 0.0
    critical_rule_triggered: bool = False
    trace_ids: tuple = ()
    symptom_tags: tuple = ()


@dataclass
class GroupedIncident:
    """One incident holding every correlated signal that joined it."""

    incident_id: str
    primary_service: str
    affected_services: list
    signals: list
    symptom_tags: set
    trace_ids: set
    status: str = "open"
    opened_at: datetime | None = None
    updated_at: datetime | None = None
    resolved_at: datetime | None = None
    last_signal_at: datetime | None = None
    clean_eval_count: int = 0
    investigation_finished: bool = False
    investigation_skipped: bool = False

    @property
    def investigation_done(self):
        """True once the investigation finished or was skipped."""
        return bool(self.investigation_finished or self.investigation_skipped)


class IncidentGrouper:
    """Ingests AnomalySignal items and owns lifecycle bookkeeping."""

    def __init__(
        self,
        dependencies=None,
        grouping_window_seconds=GROUPING_WINDOW_SECONDS,
        reopen_window_seconds=REOPEN_WINDOW_SECONDS,
        clean_evals_to_resolve=CLEAN_EVALS_TO_RESOLVE,
        clear_threshold=CLEAR_THRESHOLD,
    ):
        graph = {key: set(value) for key, value in _BUILTIN_DEPENDENCIES.items()}
        if dependencies is not None:
            for source, targets in dependencies.items():
                if source not in graph:
                    graph[source] = set()
                graph[source].update(targets)
        self.dependencies = graph
        self.grouping_window_seconds = int(grouping_window_seconds)
        self.reopen_window_seconds = int(reopen_window_seconds)
        self.clean_evals_to_resolve = int(clean_evals_to_resolve)
        self.clear_threshold = float(clear_threshold)
        self._incidents = []
        self._next_number = 1

    def _reachable(self, start):
        seen = set()
        stack = [start]
        while stack:
            current = stack.pop()
            for target in self.dependencies.get(current, ()):
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return seen

    def _related(self, first, second):
        if first == second:
            return True
        if second in self._reachable(first):
            return True
        return first in self._reachable(second)

    def _shares_cue(self, signal, incident):
        if signal.service in incident.affected_services:
            return True
        for member in incident.affected_services:
            if self._related(signal.service, member):
                return True
        if (
            signal.trace_ids
            and incident.trace_ids
            and set(signal.trace_ids).intersection(incident.trace_ids)
        ):
            return True
        if (
            signal.symptom_tags
            and incident.symptom_tags
            and set(signal.symptom_tags).intersection(incident.symptom_tags)
        ):
            return True
        if not signal.trace_ids and not signal.symptom_tags:
            return True
        return bool(not incident.trace_ids and not incident.symptom_tags)

    def _find_match(self, signal):
        best = None
        for incident in self._incidents:
            if incident.status == "closed":
                continue
            if incident.resolved_at is not None:
                delta = (signal.timestamp - incident.resolved_at).total_seconds()
                if not 0 <= delta <= self.reopen_window_seconds:
                    continue
            else:
                delta = abs((signal.timestamp - incident.last_signal_at).total_seconds())
                if delta > self.grouping_window_seconds:
                    continue
            if not self._shares_cue(signal, incident):
                continue
            if best is None:
                best = incident
                continue
            if incident.last_signal_at > best.last_signal_at:
                best = incident
        return best

    def ingest(self, signal):
        """Add one signal, creating or merging into an incident."""
        match = self._find_match(signal)
        if match is None:
            number = self._next_number
            self._next_number = number + 1
            incident = GroupedIncident(
                incident_id=f"inc-{number}",
                primary_service=signal.service,
                affected_services=[signal.service],
                signals=[signal],
                status="open",
                opened_at=signal.timestamp,
                updated_at=signal.timestamp,
                resolved_at=None,
                last_signal_at=signal.timestamp,
                clean_eval_count=0,
                symptom_tags=set(signal.symptom_tags),
                trace_ids=set(signal.trace_ids),
            )
            self._incidents.append(incident)
            return incident
        match.signals.append(signal)
        if signal.service not in match.affected_services:
            match.affected_services.append(signal.service)
        match.symptom_tags.update(signal.symptom_tags)
        match.trace_ids.update(signal.trace_ids)
        match.last_signal_at = signal.timestamp
        match.updated_at = signal.timestamp
        match.clean_eval_count = 0
        if match.resolved_at is not None:
            match.resolved_at = None
            match.status = "open"
        return match

    def get(self, incident_id):
        """Look up an incident by id. Raises KeyError when unknown."""
        for incident in self._incidents:
            if incident.incident_id == incident_id:
                return incident
        raise KeyError(incident_id)

    def list_incidents(self):
        """All incidents in creation order."""
        return list(self._incidents)

    def transition(self, incident_id, new_status, now=None):
        """Move an incident along the allowed lifecycle step."""
        incident = self.get(incident_id)
        allowed = _ALLOWED_NEXT.get(incident.status, set())
        if new_status not in allowed:
            raise ValueError(f"cannot move {incident_id} from {incident.status} to {new_status}")
        incident.status = new_status
        if now is not None:
            incident.updated_at = now
        return incident

    def complete_investigation(self, incident_id, outcome="investigated", skipped=False):
        """Mark the investigation finished, or skipped by policy."""
        incident = self.get(incident_id)
        if skipped:
            incident.investigation_skipped = True
            incident.investigation_finished = True
            return incident
        if outcome not in ("investigated", "inconclusive"):
            raise ValueError("outcome must be investigated or inconclusive")
        incident.investigation_finished = True
        if incident.status in ("open", "triaged", "investigating"):
            incident.status = outcome
        return incident

    def can_close(self, incident_id):
        """True only when resolved and investigation done or skipped."""
        incident = self.get(incident_id)
        return incident.resolved_at is not None and incident.investigation_done

    def close(self, incident_id, now=None):
        """Set status to closed. Raises ValueError when gated."""
        incident = self.get(incident_id)
        if incident.resolved_at is None:
            raise ValueError("incident has no resolved_at; cannot close")
        if not incident.investigation_done:
            raise ValueError("investigation not finished; cannot close")
        incident.status = "closed"
        if now is not None:
            incident.updated_at = now
        return incident

    def record_evaluation(self, incident_id, now, score, critical, clear_threshold=None):
        """Record one 30s evaluation for auto-resolve bookkeeping.

        A clean evaluation has the score below the clear level with
        no critical rule firing. Ten consecutive clean evaluations
        set resolved_at to `now`; any active anomaly resets the
        consecutive counter.
        """
        incident = self.get(incident_id)
        level = self.clear_threshold if clear_threshold is None else float(clear_threshold)
        is_clean = (not critical) and float(score) < level
        if is_clean:
            incident.clean_eval_count += 1
            should_resolve = incident.clean_eval_count >= self.clean_evals_to_resolve
            if should_resolve and incident.resolved_at is None:
                incident.resolved_at = now
        else:
            incident.clean_eval_count = 0
        incident.updated_at = now
        return incident
