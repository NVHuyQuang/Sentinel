"""Window aggregation over 15s scrapes, evaluated every 30s (Spec 4.1, 4.2).

Each evaluation yields one ServiceSignalWindow per service. The short
window spans 1 minute, the context window spans 5 minutes. Aggregation
averages the scrapes whose timestamp falls in [window_start, window_end).
Metrics absent from every scrape in the window stay None; absent
samples are skipped, never replaced with zero. Log and trace counts
are summed, recent_deployment is true when any scrape in the window
carries it.

Required metrics per Spec 4.2 (request_rate, error_rate, latencies,
cpu, memory) must have at least one sample in the window. When a
required metric has no sample the builder raises ValueError with an
explicit message rather than emitting a zero-filled value, so missing
telemetry is never silently coerced.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from services.decision_engine.providers.base import ServiceSignalWindow

SCRAPE_INTERVAL_SECONDS = 15
EVAL_CADENCE_SECONDS = 30
SHORT_WINDOW_SECONDS = 60
CONTEXT_WINDOW_SECONDS = 300

_METRICS = (
    "request_rate",
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "cpu_utilization",
    "memory_utilization",
    "db_pool_utilization",
    "dependency_timeout_rate",
)
_REQUIRED = (
    "request_rate",
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "cpu_utilization",
    "memory_utilization",
)


@dataclass
class RawScrape:
    """One 15s scrape sample. Metrics are None when the scrape lacks them."""

    service: str
    timestamp: datetime
    request_rate: float | None = None
    error_rate: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    cpu_utilization: float | None = None
    memory_utilization: float | None = None
    db_pool_utilization: float | None = None
    dependency_timeout_rate: float | None = None
    log_error_count: int = 0
    trace_error_count: int = 0
    recent_deployment: bool = False


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _in_window(scrapes, service, window_start, window_end):
    return [s for s in scrapes if s.service == service and window_start <= s.timestamp < window_end]


def aggregate_window(service, scrapes, window_start, window_end):
    """Build one ServiceSignalWindow for `service` over a time range."""
    in_window = _in_window(scrapes, service, window_start, window_end)
    averaged = {}
    for name in _METRICS:
        present = [getattr(s, name) for s in in_window if getattr(s, name) is not None]
        averaged[name] = _mean(present)
    for name in _REQUIRED:
        if averaged[name] is None:
            raise ValueError(
                f"missing required metric {name} for service {service} in [{window_start}, {window_end}); "
                "refusing to zero-fill"
            )
    return ServiceSignalWindow(
        service=service,
        window_start=window_start,
        window_end=window_end,
        request_rate=averaged["request_rate"],
        error_rate=averaged["error_rate"],
        latency_p50_ms=averaged["latency_p50_ms"],
        latency_p95_ms=averaged["latency_p95_ms"],
        latency_p99_ms=averaged["latency_p99_ms"],
        cpu_utilization=averaged["cpu_utilization"],
        memory_utilization=averaged["memory_utilization"],
        db_pool_utilization=averaged["db_pool_utilization"],
        dependency_timeout_rate=averaged["dependency_timeout_rate"],
        recent_deployment=any(s.recent_deployment for s in in_window),
        log_error_count=sum(s.log_error_count for s in in_window),
        trace_error_count=sum(s.trace_error_count for s in in_window),
    )


def build_evaluation_windows(service, scrapes, eval_time):
    """Return (short 1-min, context 5-min) windows ending at `eval_time`."""
    short = aggregate_window(
        service,
        scrapes,
        eval_time - timedelta(seconds=SHORT_WINDOW_SECONDS),
        eval_time,
    )
    context = aggregate_window(
        service,
        scrapes,
        eval_time - timedelta(seconds=CONTEXT_WINDOW_SECONDS),
        eval_time,
    )
    return short, context


def build_all_service_windows(scrapes, eval_time):
    """Return {service: (short, context)} for every service in `scrapes`."""
    services = sorted({s.service for s in scrapes})
    result = {}
    for service in services:
        result[service] = build_evaluation_windows(service, scrapes, eval_time)
    return result


def evaluation_times(start, end, cadence_seconds=EVAL_CADENCE_SECONDS):
    """Yield 30s-cadence evaluation instants in [start, end]."""
    instant = start
    step = timedelta(seconds=cadence_seconds)
    while instant <= end:
        yield instant
        instant = instant + step
