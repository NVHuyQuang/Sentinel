"""Allowlisted, budgeted read-only tools over a TelemetryBackend.

Hard caps (Spec 6.4): 12 tool calls per round, 30 total, 180 s wall clock. Every call yields a
ToolResult with a checksum and an immutable snapshot reference; the graph turns results into
Evidence. Agents never see backend internals or submit raw query languages: only
validated-parameter tools exist, and none of them writes.
"""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid5

from services.decision_engine.providers.base import Evidence
from services.telemetry.backend import SnapshotBackend

MAX_CALLS = 30
ROUND_MAX = 12
DEADLINE_S = 180.0


class ToolBudgetExceeded(RuntimeError):
    pass


class RoundBudgetExceeded(ToolBudgetExceeded):
    """Per-round cap hit; the next round may still run."""


@dataclass
class ToolResult:
    tool: str
    params: dict
    data: list
    checksum: str
    source_reference: str


class ToolBox:
    def __init__(self, backend: SnapshotBackend, investigation_id: UUID,
                 clock: Callable[[], float] = time.monotonic, max_calls: int = MAX_CALLS,
                 deadline_s: float = DEADLINE_S, round_max: int = ROUND_MAX,
                 history: list[dict] | None = None) -> None:
        self._backend = backend
        self.investigation_id = investigation_id
        self._clock = clock
        self._t0 = clock()
        self.max_calls, self.deadline_s, self.round_max = max_calls, deadline_s, round_max
        self._history = history or []  # past incidents; empty until a store exists
        self.round = 0
        self._round_calls = 0
        self.evidence: list[Evidence] = []
        self.trace: list[dict] = []  # execution trace: one entry per tool call

    @property
    def now(self) -> datetime:
        return self._backend.now

    def new_round(self) -> None:
        self.round += 1
        self._round_calls = 0

    @property
    def calls(self) -> int:
        return len(self.trace)

    def time_left(self) -> float:
        return self.deadline_s - (self._clock() - self._t0)

    def _call(self, tool: str, params: dict, fn: Callable[[], list]) -> ToolResult:
        if len(self.trace) >= self.max_calls:
            raise ToolBudgetExceeded(f"tool call budget of {self.max_calls} reached")
        if self.time_left() < 0:
            raise ToolBudgetExceeded(f"time budget of {self.deadline_s}s reached")
        if self._round_calls >= self.round_max:
            raise RoundBudgetExceeded(f"round budget of {self.round_max} calls reached")
        self._round_calls += 1
        data = fn()
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        q = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        res = ToolResult(tool, params, data, digest,
                         f"snapshot://{self._backend.snapshot_id[:16]}/{tool}?q={q}&sha256={digest}")
        self.trace.append({"round": self.round, "tool": tool, "params": params, "rows": len(data),
                           "sha256": digest})
        return res

    def query_metrics(self, service: str, metric: str, start: datetime, end: datetime):
        p = {"service": service, "metric": metric, "start": start.isoformat(),
             "end": end.isoformat()}
        return self._call("query_metrics", p,
                          lambda: self._backend.query_metrics(service, metric, start, end))

    def query_logs(self, service: str, start: datetime, end: datetime, level: str | None = None,
                   limit: int = 100):
        p = {"service": service, "start": start.isoformat(), "end": end.isoformat(),
             "level": level, "limit": limit}
        return self._call("query_logs", p,
                          lambda: self._backend.query_logs(service, start, end, level, limit))

    def query_traces(self, service: str, start: datetime, end: datetime,
                     errors_only: bool = False):
        p = {"service": service, "start": start.isoformat(), "end": end.isoformat(),
             "errors_only": errors_only}
        return self._call("query_traces", p,
                          lambda: self._backend.query_traces(service, start, end, errors_only))

    def get_deployment_events(self, service: str, start: datetime, end: datetime):
        p = {"service": service, "start": start.isoformat(), "end": end.isoformat()}
        return self._call("get_deployment_events", p,
                          lambda: self._backend.get_deployment_events(service, start, end))

    def get_service_dependencies(self, service: str):
        return self._call("get_service_dependencies", {"service": service},
                          lambda: self._backend.get_service_dependencies(service))

    def search_historical_incidents(self, query: str, limit: int = 5):
        q = query.lower()[:100]
        hits = [h for h in self._history if q in json.dumps(h).lower()][:max(1, min(limit, 5))]
        return self._call("search_historical_incidents", {"query": q, "limit": limit},
                          lambda: hits)

    def record(self, res: ToolResult, service: str, observation: str, supports: list[str],
               contradicts: list[str]) -> Evidence:
        """Store one evidence record. Id is deterministic so replays are identical."""
        ev = Evidence(
            id=uuid5(self.investigation_id, res.source_reference),
            investigation_id=self.investigation_id,
            source_type=res.tool,
            service=service,
            observation=observation,
            timestamp=self.now,
            source_reference=res.source_reference,
            query_parameters=res.params,
            supports=supports,
            contradicts=contradicts,
        )
        self.evidence.append(ev)
        return ev
