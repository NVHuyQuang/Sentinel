"""In-process telemetry: 15 s scrapes, JSON logs with trace ids, spans.

Emits only what a real stack would: request counts, latency, pool gauges, error logs.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

SCRAPE_SECONDS = 15
TRACE_SAMPLE = 0.05  # error traces are always kept


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


@dataclass
class _Acc:
    durations: list[float] = field(default_factory=list)
    errors: int = 0
    dep_timeouts: int = 0
    log_errors: int = 0
    trace_errors: int = 0


@dataclass
class Trace:
    trace_id: str
    spans: list[dict] = field(default_factory=list)
    error: bool = False


class Collector:
    """Per-service accumulators; `scrape()` turns them into one row per metric."""

    def __init__(self, services: list[str], rng: random.Random, epoch: datetime) -> None:
        self.rng = rng
        self.epoch = epoch  # datetime of virtual t=0
        self.acc = {s: _Acc() for s in services}
        self.metrics: list[dict] = []
        self.logs: list[dict] = []
        self.spans: list[dict] = []
        self.pool_probe = None  # callable(now) -> (utilization, wait_ms_list), order DB pool

    def iso(self, t: float) -> str:
        return (self.epoch + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def new_trace(self) -> Trace:
        return Trace(trace_id=f"{self.rng.getrandbits(64):016x}")

    def span(
        self, tr: Trace, service: str, name: str, start: float, dur: float, status="OK", **attrs
    ) -> None:
        tr.spans.append(
            {
                "trace_id": tr.trace_id,
                "span_id": f"{self.rng.getrandbits(32):08x}",
                "service": service,
                "name": name,
                "start": self.iso(start),
                "duration_ms": round(dur * 1000, 2),
                "status": status,
                "attributes": attrs,
            }
        )
        if status == "ERROR":
            tr.error = True

    def request(self, service: str, dur: float, ok: bool, dep_timeout: bool = False) -> None:
        a = self.acc[service]
        a.durations.append(dur * 1000)
        a.errors += 0 if ok else 1
        a.dep_timeouts += 1 if dep_timeout else 0

    def log(self, t: float, service: str, level: str, tr: Trace, event: str, dur: float, msg: str):
        self.logs.append(
            {
                "timestamp": self.iso(t),
                "service": service,
                "level": level,
                "trace_id": tr.trace_id,
                "span_id": tr.spans[-1]["span_id"] if tr.spans else "",
                "event": event,
                "duration_ms": round(dur * 1000, 2),
                "message": msg,
            }
        )
        if level == "ERROR":
            self.acc[service].log_errors += 1

    def finish(self, tr: Trace) -> None:
        if tr.error or self.rng.random() < TRACE_SAMPLE:
            self.spans.extend(tr.spans)
        if tr.error:
            for s in {sp["service"] for sp in tr.spans if sp["status"] == "ERROR"}:
                self.acc[s].trace_errors += 1

    def scrape(self, now: float, base: dict[str, dict]) -> None:
        ts = self.iso(now)
        for svc, a in self.acc.items():
            n = len(a.durations)
            rate = n / SCRAPE_SECONDS
            b = base[svc]
            # CPU/memory are modeled from load plus noise; not measured container stats.
            cpu = b["cpu"] + b["cpu_k"] * rate + self.rng.gauss(0, 0.01)
            mem = b["mem"] + self.rng.gauss(0, 0.005)
            row = {
                "request_rate": rate,
                "error_rate": a.errors / n if n else 0.0,
                "latency_p50_ms": _pct(a.durations, 0.50),
                "latency_p95_ms": _pct(a.durations, 0.95),
                "latency_p99_ms": _pct(a.durations, 0.99),
                "cpu_utilization": min(1.0, max(0.0, cpu)),
                "memory_utilization": min(1.0, max(0.0, mem)),
                "log_error_count": a.log_errors,
                "trace_error_count": a.trace_errors,
            }
            if b.get("deps"):
                row["dependency_timeout_rate"] = a.dep_timeouts / n if n else 0.0
            if svc == "order-service" and self.pool_probe is not None:
                util, waits = self.pool_probe(now)
                row["db_pool_utilization"] = util
                row["db_query_wait_ms_p95"] = _pct(waits, 0.95)
            for name, value in row.items():
                self.metrics.append({"metric": name, "ts": ts, "service": svc, "value": value})
            self.acc[svc] = _Acc()
