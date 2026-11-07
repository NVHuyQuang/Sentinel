"""Allowlisted, budgeted read-only tools over a TelemetryBackend.

Hard caps: 12 tool calls and 180 s wall clock per investigation. Every call yields a
ToolResult with a checksum and an immutable snapshot reference; the collector turns results
into Evidence. The agent never sees backend internals, only validated-parameter tools.
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

MAX_CALLS = 12
DEADLINE_S = 180.0


class ToolBudgetExceeded(RuntimeError):
    pass


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
                 deadline_s: float = DEADLINE_S) -> None:
        self._backend = backend
        self.investigation_id = investigation_id
        self._clock = clock
        self._t0 = clock()
        self.max_calls, self.deadline_s = max_calls, deadline_s
        self.evidence: list[Evidence] = []
        self.trace: list[dict] = []  # execution trace: one entry per tool call

    @property
    def now(self) -> datetime:
        return self._backend.now

    def _call(self, tool: str, params: dict, fn: Callable[[], list]) -> ToolResult:
        if len(self.trace) >= self.max_calls:
            raise ToolBudgetExceeded(f"tool call budget of {self.max_calls} reached")
        if self._clock() - self._t0 > self.deadline_s:
            raise ToolBudgetExceeded(f"time budget of {self.deadline_s}s reached")
        data = fn()
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        q = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        res = ToolResult(tool, params, data, digest,
                         f"snapshot://{self._backend.snapshot_id[:16]}/{tool}?q={q}&sha256={digest}")
        self.trace.append({"tool": tool, "params": params, "rows": len(data), "sha256": digest})
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
