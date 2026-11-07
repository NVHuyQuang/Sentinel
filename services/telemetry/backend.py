"""TelemetryBackend interface, LiveBackend stub, SnapshotBackend (recorded replay).

SnapshotBackend enforces the same limits a live backend must: service allowlist, max time
range, max rows, and a hard "now" so no query can read data after the investigation's clock.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from services.signal_processor.features.windows import RawScrape

MAX_RANGE = timedelta(minutes=30)
MAX_METRIC_POINTS = 500
MAX_LOGS = 100
MAX_SPANS = 50
METRIC_ALLOWLIST = frozenset(
    {
        "request_rate", "error_rate", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "cpu_utilization", "memory_utilization", "db_pool_utilization",
        "db_query_wait_ms_p95", "dependency_timeout_rate",
    }
)


class QueryRejected(ValueError):
    """A tool query broke an allowlist or limit."""


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc) \
        if "." in value else datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)


class TelemetryBackend(Protocol):
    def query_metrics(self, service: str, metric: str, start: datetime, end: datetime) -> list: ...

    def query_logs(self, service: str, start: datetime, end: datetime, level: str | None = None,
                   limit: int = 100) -> list: ...

    def query_traces(self, service: str, start: datetime, end: datetime,
                     errors_only: bool = False) -> list: ...

    def get_deployment_events(self, service: str, start: datetime, end: datetime) -> list: ...

    def get_service_dependencies(self, service: str) -> list: ...


class LiveBackend:
    """Prometheus/Loki/Tempo adapter. Out of scope for this slice."""

    def __getattr__(self, name: str):
        raise NotImplementedError("LiveBackend is not implemented; use SnapshotBackend")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class SnapshotBackend:
    def __init__(self, root: Path, now: datetime | None = None, verify: bool = True) -> None:
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        sums = json.loads((self.root / "checksums.json").read_text())
        if verify:
            for name, digest in sums.items():
                actual = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(f"snapshot checksum mismatch: {name}")
        self.snapshot_id = hashlib.sha256(json.dumps(sums, sort_keys=True).encode()).hexdigest()
        self.services = frozenset(self.meta["services"])
        self.start = parse_ts(self.meta["start"])
        self.end = parse_ts(self.meta["end"])
        self._deps = json.loads((self.root / "dependencies.json").read_text())
        self.now = now or self.end

    def at(self, now: datetime) -> "SnapshotBackend":
        """Same snapshot with a new clock. Cheap: shares nothing mutable."""
        clone = object.__new__(SnapshotBackend)
        clone.__dict__.update(self.__dict__)
        clone.now = min(now, self.end)
        return clone

    # -- guards ---------------------------------------------------------------------------
    def _window(self, service: str, start: datetime, end: datetime) -> tuple[datetime, datetime]:
        if service not in self.services:
            raise QueryRejected(f"service {service!r} is not allowlisted")
        end = min(end, self.now)
        if start > end:
            raise QueryRejected("start is after the investigation clock")
        if end - start > MAX_RANGE:
            raise QueryRejected(f"time range exceeds {MAX_RANGE}")
        return start, end

    # -- tools ----------------------------------------------------------------------------
    def query_metrics(self, service: str, metric: str, start: datetime, end: datetime) -> list:
        if metric not in METRIC_ALLOWLIST:
            raise QueryRejected(f"metric {metric!r} is not allowlisted")
        start, end = self._window(service, start, end)
        rows = _jsonl(self.root / "metrics" / f"{metric}.jsonl")
        out = [{"ts": r["ts"], "value": r["value"]} for r in rows
               if r["service"] == service and start <= parse_ts(r["ts"]) <= end]
        return out[-MAX_METRIC_POINTS:]

    def query_logs(self, service: str, start: datetime, end: datetime, level: str | None = None,
                   limit: int = 100) -> list:
        start, end = self._window(service, start, end)
        limit = max(1, min(limit, MAX_LOGS))
        rows = [r for r in _jsonl(self.root / "logs.jsonl")
                if r["service"] == service and start <= parse_ts(r["timestamp"]) <= end
                and (level is None or r["level"] == level)]
        return rows[-limit:]

    def query_traces(self, service: str, start: datetime, end: datetime,
                     errors_only: bool = False) -> list:
        start, end = self._window(service, start, end)
        rows = [r for r in _jsonl(self.root / "traces.jsonl")
                if r["service"] == service and start <= parse_ts(r["start"]) <= end
                and (not errors_only or r["status"] == "ERROR")]
        return rows[-MAX_SPANS:]

    def get_deployment_events(self, service: str, start: datetime, end: datetime) -> list:
        start, end = self._window(service, start, end)
        return [r for r in _jsonl(self.root / "deployments.jsonl")
                if r["service"] == service and start <= parse_ts(r["timestamp"]) <= end]

    def get_service_dependencies(self, service: str) -> list:
        if service not in self.services:
            raise QueryRejected(f"service {service!r} is not allowlisted")
        return list(self._deps.get(service, []))

    # -- observability plane (signal processor), not an agent tool -------------------------
    def scrapes(self, until: datetime) -> list[RawScrape]:
        """15 s scrapes up to min(until, now). Missing metrics stay None."""
        until = min(until, self.now)
        by_key: dict[tuple[str, str], RawScrape] = {}
        for path in sorted((self.root / "metrics").glob("*.jsonl")):
            name = path.stem
            for r in _jsonl(path):
                ts = parse_ts(r["ts"])
                if ts > until:
                    continue
                s = by_key.setdefault((r["service"], r["ts"]),
                                      RawScrape(service=r["service"], timestamp=ts))
                if name in ("log_error_count", "trace_error_count"):
                    setattr(s, name, int(r["value"]))
                elif hasattr(s, name):
                    setattr(s, name, r["value"])
        return sorted(by_key.values(), key=lambda s: (s.timestamp, s.service))

    def recent_deployments(self, until: datetime, within: timedelta) -> list[str]:
        until = min(until, self.now)
        return sorted({r["service"] for r in _jsonl(self.root / "deployments.jsonl")
                       if until - within <= parse_ts(r["timestamp"]) <= until})
