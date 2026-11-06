"""Laya adapter T2 (docs/laya-contract.md Section 2, verified Spike 0).

Quality caveat (laya-contract Section 3): the checkpoint's priority_score
clusters at 0.498-0.509 across all 50 spike samples (expected level ~4.5/9
with near-uniform probabilities), so it carries almost no discriminative
signal, and severity understates: s05 (db_pool 0.92, p95 3480 ms,
critical_rule_triggered true) classified SEV4. This adapter does not hide
that: policy.enforce_severity_floor and policy.should_investigate stay on
top of the decision, and emitted probabilities are uncalibrated (the
checkpoint warns about invalid temperatures), so they must not be
presented as statistics.

Laya is an in-process library (Router(device) + predict(text, questions)),
not an HTTP endpoint, so there is no process to kill: every failure is
surfaced as ProviderError and the policy layer falls back to rules.
"""

import asyncio
import sys
import time

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    TriageDecision,
)

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "What best describes the incident?",
        "criteria": {
            "LATENCY": "high latency timeouts slowness degraded performance",
            "ERROR_SPIKE": "5xx errors failure rate high",
            "RESOURCE": "cpu memory resource saturation",
            "DEPENDENCY": "downstream dependency slow or failing",
            "UNKNOWN": "cannot determine",
        },
    },
    "severity": {
        "type": "choice",
        "instructions": "How severe?",
        "criteria": {
            "SEV1": "critical unavailability",
            "SEV2": "major degradation",
            "SEV3": "partial degradation",
            "SEV4": "low impact",
        },
    },
    "priority_score": {
        "type": "score",
        "instructions": "Score investigation priority from 0 to 9 inclusive.",
        "criteria": [str(i) for i in range(10)],
    },
    "investigate": {"type": "noul", "instructions": "Is investigation required?"},
    "deployment_relevant": {
        "type": "noul",
        "instructions": "Is deployment relevant?",
    },
}

_CATEGORIES = {"LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN"}
_SEVERITIES = {"SEV1", "SEV2", "SEV3", "SEV4"}

# Production timeout must exceed the device P95 (laya-contract Section 3:
# GPU >= ~70 ms, CPU cold >= ~1.8 s) or the policy layer falls back
# spuriously. The spike's 10 ms is a test budget, not a production default.
_DEFAULT_TIMEOUT_MS = {"cuda": 2000.0, "cpu": 10000.0}


def render_features(features: IncidentFeatures) -> str:
    """Render IncidentFeatures as key/value text, byte-identical to the spike.

    Token order and formatting match scripts/spike_laya.py make_samples so
    spike/model comparisons stay fair. Metrics come from the primary
    service's latest (shortest) window; the baseline copies its longest
    window p95 (falling back to the current window p50 when only one
    window exists, and to 0 with no windows at all).
    """
    windows = [w for w in features.windows if w.service == features.primary_service]
    if not windows:
        windows = list(features.windows)
    windows = sorted(windows, key=lambda w: w.window_end - w.window_start)
    current = windows[0] if windows else None
    reference = windows[-1] if len(windows) > 1 else current

    error_rate = current.error_rate if current else 0.0
    p95 = current.latency_p95_ms if current else 0.0
    baseline = (
        reference.latency_p95_ms if reference else (current.latency_p50_ms if current else 0.0)
    )
    cpu = current.cpu_utilization if current else 0.0
    mem = current.memory_utilization if current else 0.0
    pool = current.db_pool_utilization if current else None
    pool_part = f"db_pool {pool}" if pool is not None else "db_pool n/a"
    deploy = bool(features.deployment_services) or any(
        w.recent_deployment for w in features.windows
    )
    return (
        f"primary_service {features.primary_service} "
        f"affected {'/'.join(features.affected_services)} "
        f"error_rate {error_rate} p95_latency_ms {p95} "
        f"baseline_p95_ms {baseline} "
        f"cpu_utilization {cpu} memory_utilization {mem} {pool_part} "
        f"anomaly_score {features.anomaly_score} "
        f"critical_rule_triggered {str(features.critical_rule_triggered).lower()} "
        f"recent_deployment {str(deploy).lower()} "
        f"symptom_tags {','.join(features.symptom_tags)}"
    )


def parse_result(raw, latency_ms=None, device=None) -> TriageDecision:
    """Parse one raw Laya payload into the 5 Sentinel decisions.

    Verified parser rules (laya-contract Section 2): priority_score is the
    expected level normalized by 9.0 (never the argmax); noul booleans use
    bool(float(noul) >= 0.5). Any missing key, out-of-vocab choice, bad
    number or out-of-range priority raises ProviderError.
    """
    try:
        answers = raw["answers"]
        cat_ans = answers["category"]
        sev_ans = answers["severity"]
        pri_ans = answers["priority_score"]
        inv_ans = answers["investigate"]
        dep_ans = answers["deployment_relevant"]
    except (KeyError, TypeError) as exc:
        raise ProviderError(f"malformed: missing decision key {exc}") from exc

    try:
        category = cat_ans.get("choice")
        severity = sev_ans.get("choice")
    except AttributeError as exc:
        raise ProviderError(f"malformed: choice answer is not a mapping {exc}") from exc
    if category not in _CATEGORIES or severity not in _SEVERITIES:
        raise ProviderError(
            f"out-of-vocab choice: category={category!r} severity={severity!r}"
        )
    try:
        score_expected = float(pri_ans["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"malformed: bad priority_score {exc}") from exc
    priority_score = score_expected / 9.0
    if not 0.0 <= priority_score <= 1.0:
        raise ProviderError(
            f"malformed: priority_score out of range {priority_score}"
        )
    try:
        inv_noul = float(inv_ans["noul"])
        dep_noul = float(dep_ans["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"malformed: bad noul {exc}") from exc

    routing = raw.get("routing", {}) if isinstance(raw, dict) else {}
    model_version = routing.get("repo", "unknown") if isinstance(routing, dict) else "unknown"
    raw_scores = {
        "score_expected": score_expected,
        "noul_investigate": inv_noul,
        "noul_deployment": dep_noul,
    }
    for key, answer, label in (
        ("category_probability", cat_ans, category),
        ("severity_probability", sev_ans, severity),
    ):
        probs = answer.get("probabilities") if isinstance(answer, dict) else None
        if isinstance(probs, dict):
            try:
                raw_scores[key] = float(probs[label])
            except (KeyError, TypeError, ValueError):
                pass
    if device is not None:
        raw_scores["device_cuda"] = 1.0 if device == "cuda" else 0.0
    return TriageDecision(
        category=category,
        severity=severity,
        priority_score=priority_score,
        investigate=bool(inv_noul >= 0.5),
        deployment_relevant=bool(dep_noul >= 0.5),
        provider="laya",
        model_version=model_version,
        latency_ms=latency_ms,
        raw_scores=raw_scores,
    )


class LayaAdapter:
    """DecisionProvider over the in-process Laya Router."""

    def __init__(self, device="cuda", timeout_ms=None, router=None):
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")
        self.device = device
        self.timeout_ms = (
            float(timeout_ms)
            if timeout_ms is not None
            else _DEFAULT_TIMEOUT_MS[device]
        )
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms!r}")
        if router is None:
            try:
                router = self._build_router()
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(
                    f"laya construction failed: {type(exc).__name__}: {exc}"
                ) from exc
        self._router = router
        # Untimed warmup: triggers the lazy checkpoint load so the first
        # timed triage does not pay the ~60-75 s load (laya-contract S1).
        try:
            self._router.predict("warmup", QUESTIONS)
        except Exception as exc:
            raise ProviderError(
                f"laya warmup failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _build_router(self):
        try:
            from laya import Router
        except Exception:  # noqa: BLE001 - import failure must become ProviderError
            # Spike workaround for the torchaudio DLL load failure on Windows.
            sys.modules.setdefault("torchaudio", None)
            from laya import Router

        return Router(device=self.device)

    async def triage(self, features: IncidentFeatures) -> TriageDecision:
        text = render_features(features)
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._router.predict(text, QUESTIONS)
                ),
                timeout=self.timeout_ms / 1000.0,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            # The in-flight predict is not cancellable; its executor thread
            # finishes in the background after the ProviderError is raised.
            raise ProviderError(
                f"timeout: laya predict exceeded {self.timeout_ms} ms"
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"laya predict failed: {type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        return parse_result(raw, latency_ms=latency_ms, device=self.device)
