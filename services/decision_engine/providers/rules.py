"""Deterministic rule fallback provider T0 (Spec Section 5.1 severity table).

Production fallback behind the same DecisionProvider seam as the model
providers. Inputs are IncidentFeatures only (Spec Section 5.2): no ground
truth, no scenario manifest fields (Spec Section 3.3, Section 7.1).

Severity rules (Spec Section 5.1), evaluated on the latest (shortest)
window per affected service:

- SEV1: API Gateway error_rate >= 0.50, or any core service failing /ready
  for >= 60 s. The /ready probe is not part of IncidentFeatures, so only
  the gateway-error path is evaluated here.
- SEV2: any core service error_rate >= 0.20, or P95 latency >= 5x its
  baseline median.
- SEV3: any service error_rate >= 0.05, or P95 latency >= 2x its baseline
  median.
- SEV4: anomaly candidate that meets none of the rules above.

Baseline approximation: IncidentFeatures carries no baseline-median field.
The longest window per service (5-min) serves as the baseline proxy and
the shortest (1-min) as the current window; ratio = current p95 / proxy
p95. With a single window per service there is no proxy and the ratio is
1.0, so the latency path cannot fire. The spec also requires the rule to
hold for at least 2 consecutive evaluations; a single snapshot cannot
show that, so sustainment stays with the caller loop.
"""

import time
from typing import Literal

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ServiceSignalWindow,
    TriageDecision,
)

CORE_SERVICES = frozenset(
    {"api-gateway", "auth-service", "order-service", "payment-service"}
)

SEV1_GATEWAY_ERROR_RATE = 0.50
SEV2_ERROR_RATE = 0.20
SEV3_ERROR_RATE = 0.05
SEV2_LATENCY_RATIO = 5.0
SEV3_LATENCY_RATIO = 2.0
RESOURCE_UTILIZATION = 0.85
DEPENDENCY_TIMEOUT_RATE = 0.05

_PRIORITY_BY_SEVERITY = {"SEV1": 0.95, "SEV2": 0.80, "SEV3": 0.55, "SEV4": 0.25}


def _per_service(
    features: IncidentFeatures,
) -> list[tuple[str, ServiceSignalWindow, float]]:
    """One (service, current window, latency ratio) entry per affected service.

    Current window is the shortest (latest 1-min); the ratio compares its
    p95 against the longest (5-min) window used as the baseline proxy.
    """
    by_service: dict[str, list[ServiceSignalWindow]] = {}
    for window in features.windows:
        by_service.setdefault(window.service, []).append(window)
    result = []
    for service, windows in by_service.items():
        ordered = sorted(windows, key=lambda w: w.window_end - w.window_start)
        current = ordered[0]
        ratio = 1.0
        if len(ordered) > 1 and ordered[-1].latency_p95_ms > 0:
            ratio = current.latency_p95_ms / ordered[-1].latency_p95_ms
        result.append((service, current, ratio))
    return result


def _severity(per_service: list[tuple[str, ServiceSignalWindow, float]]) -> str:
    gateway_error = max(
        (w.error_rate for s, w, _ in per_service if s == "api-gateway"),
        default=0.0,
    )
    if gateway_error >= SEV1_GATEWAY_ERROR_RATE:
        return "SEV1"
    if any(
        s in CORE_SERVICES and w.error_rate >= SEV2_ERROR_RATE
        for s, w, _ in per_service
    ) or any(ratio >= SEV2_LATENCY_RATIO for _, _, ratio in per_service):
        return "SEV2"
    if any(w.error_rate >= SEV3_ERROR_RATE for _, w, _ in per_service) or any(
        ratio >= SEV3_LATENCY_RATIO for _, _, ratio in per_service
    ):
        return "SEV3"
    return "SEV4"


def rule_severity(features: IncidentFeatures) -> Literal["SEV1", "SEV2"] | None:
    """Deterministic Section 5.1 severity when it is SEV1/SEV2, else None.

    feeds enforce_severity_floor: a SEV1/SEV2 rule firing raises any model
    decision to at least that level and never lowers it.
    """
    severity = _severity(_per_service(features))
    if severity in ("SEV1", "SEV2"):
        return severity  # type: ignore[return-value]
    return None


def _category(
    per_service: list[tuple[str, ServiceSignalWindow, float]],
    features: IncidentFeatures,
) -> str:
    currents = [w for _, w, _ in per_service]
    max_error = max((w.error_rate for w in currents), default=0.0)
    max_cpu = max((w.cpu_utilization for w in currents), default=0.0)
    max_mem = max((w.memory_utilization for w in currents), default=0.0)
    max_pool = max(
        (w.db_pool_utilization for w in currents if w.db_pool_utilization is not None),
        default=0.0,
    )
    max_dep_timeout = max(
        (
            w.dependency_timeout_rate
            for w in currents
            if w.dependency_timeout_rate is not None
        ),
        default=0.0,
    )
    max_ratio = max((ratio for _, _, ratio in per_service), default=1.0)
    tags = set(features.symptom_tags)
    if (
        max_cpu >= RESOURCE_UTILIZATION
        or max_mem >= RESOURCE_UTILIZATION
        or max_pool >= RESOURCE_UTILIZATION
        or tags & {"pool_saturated", "cpu_high", "memory_high"}
    ):
        return "RESOURCE"
    if max_dep_timeout >= DEPENDENCY_TIMEOUT_RATE or "dependency_timeout" in tags:
        return "DEPENDENCY"
    if "error_up" in tags or max_error >= SEV2_ERROR_RATE:
        return "ERROR_SPIKE"
    if "latency_up" in tags or max_ratio >= SEV3_LATENCY_RATIO:
        return "LATENCY"
    if max_error >= SEV3_ERROR_RATE:
        return "ERROR_SPIKE"
    return "UNKNOWN"


class RulesProvider:
    """T0 deterministic fallback: Spec Section 5.1 rules over IncidentFeatures."""

    async def triage(self, features: IncidentFeatures) -> TriageDecision:
        start = time.perf_counter()
        per_service = _per_service(features)
        severity = _severity(per_service)
        currents = [w for _, w, _ in per_service]
        stats = {
            "max_error_rate": max((w.error_rate for w in currents), default=0.0),
            "max_latency_ratio": max(
                (ratio for _, _, ratio in per_service), default=1.0
            ),
            "max_cpu_utilization": max(
                (w.cpu_utilization for w in currents), default=0.0
            ),
            "max_memory_utilization": max(
                (w.memory_utilization for w in currents), default=0.0
            ),
        }
        return TriageDecision(
            category=_category(per_service, features),  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            priority_score=_PRIORITY_BY_SEVERITY[severity],
            investigate=severity in ("SEV1", "SEV2")
            or features.critical_rule_triggered
            or features.anomaly_score >= 0.85,
            deployment_relevant=bool(features.deployment_services)
            or any(w.recent_deployment for w in features.windows),
            provider="rules",
            model_version=None,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            raw_scores=stats,
        )
