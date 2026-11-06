"""Random Forest provider T1 (Spec Section 9.4).

Placeholder caveat: this module ships without a production model artifact.
Tests train a tiny synthetic smoke model in memory (20 rows) only to prove
the plumbing. Before any T1 evaluation, retrain on the real Training split
(Spec Section 9.2: 48 positive + 12 normal, disjoint seeds; Validation for
thresholds; Test never tuned) and load the serialized bundle via
SKLEARN_MODEL_PATH. Do not claim T1 quality from placeholder data.

Train time lives in fit_bundle (called by an offline script, never by
triage). Triage only extracts FEATURE_ORDER features and predicts.

Feature vector: FEATURE_ORDER is the single source of truth for train and
inference. Extractors always return one float per entry in that order.
Missing db_pool, dependency_timeout_rate, and detector scores stay NaN
(RandomForest handles NaN natively since sklearn 1.4); they are never
silently filled with 0. Latency values stay in raw milliseconds (no
scaling; trees do not need it).

Priority head: no regression head. priority_score is the maximum category
class probability (max of predict_proba for the category model), which is
in [0, 1] by construction. Severity probabilities are still stored in
raw_scores so Brier score remains computable.
"""

import os
import time
from pathlib import Path

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    TriageDecision,
)

ENV_MODEL_PATH = "SKLEARN_MODEL_PATH"

CATEGORIES = ("LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN")
SEVERITIES = ("SEV1", "SEV2", "SEV3", "SEV4")

# Exact feature order for train and inference. Do not re-declare elsewhere.
#
# primary_* come from the primary service latest (shortest) window, falling
# back to the shortest window overall when the primary service has no window.
# max_* are maxima over the latest window per affected service. A missing
# db_pool / dependency_timeout_rate contributes no value to the max; if every
# affected service is missing the field the max stays NaN.
#
# Remaining entries: anomaly_score, the three detector scores (missing key
# stays NaN), critical_flag (1.0/0.0), affected_service_count (services in
# IncidentFeatures.affected_services), and window_count (total windows).
FEATURE_ORDER = (
    "primary_error_rate",
    "primary_latency_p50_ms",
    "primary_latency_p95_ms",
    "primary_latency_p99_ms",
    "primary_cpu_utilization",
    "primary_memory_utilization",
    "primary_db_pool_utilization",
    "primary_dependency_timeout_rate",
    "max_error_rate",
    "max_latency_p50_ms",
    "max_latency_p95_ms",
    "max_latency_p99_ms",
    "max_cpu_utilization",
    "max_memory_utilization",
    "max_db_pool_utilization",
    "max_dependency_timeout_rate",
    "anomaly_score",
    "detector_statistical",
    "detector_isolation",
    "detector_change_point",
    "critical_flag",
    "affected_service_count",
    "window_count",
)

_PER_SERVICE_FIELDS = (
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "cpu_utilization",
    "memory_utilization",
    "db_pool_utilization",
    "dependency_timeout_rate",
)


def _latest_per_service(features: IncidentFeatures) -> dict:
    by_service = {}
    for window in features.windows:
        by_service.setdefault(window.service, []).append(window)
    latest = {}
    for service, windows in by_service.items():
        latest[service] = min(windows, key=lambda w: w.window_end - w.window_start)
    return latest


def _primary_window(features: IncidentFeatures, latest: dict):
    if features.primary_service in latest:
        return latest[features.primary_service]
    if latest:
        return min(
            latest.values(), key=lambda w: w.window_end - w.window_start
        )
    return None


def _field(window, name: str) -> float:
    if window is None:
        return float("nan")
    value = getattr(window, name)
    return float("nan") if value is None else float(value)


def extract_features(features: IncidentFeatures) -> list[float]:
    """Flatten IncidentFeatures into a fixed numeric vector in FEATURE_ORDER."""
    latest = _latest_per_service(features)
    primary = _primary_window(features, latest)
    by_name = {}
    for name in _PER_SERVICE_FIELDS:
        by_name[f"primary_{name}"] = _field(primary, name)
    for name in _PER_SERVICE_FIELDS:
        present = [
            _field(window, name)
            for window in latest.values()
            if getattr(window, name) is not None
        ]
        by_name[f"max_{name}"] = max(present) if present else float("nan")
    by_name["anomaly_score"] = float(features.anomaly_score)
    by_name["detector_statistical"] = float(
        features.detector_scores.get("statistical", float("nan"))
    )
    by_name["detector_isolation"] = float(
        features.detector_scores.get("isolation", float("nan"))
    )
    by_name["detector_change_point"] = float(
        features.detector_scores.get("change_point", float("nan"))
    )
    by_name["critical_flag"] = 1.0 if features.critical_rule_triggered else 0.0
    by_name["affected_service_count"] = float(len(features.affected_services))
    by_name["window_count"] = float(len(features.windows))
    return [by_name[name] for name in FEATURE_ORDER]


def fit_bundle(X, y_category, y_severity, model_version="placeholder"):
    """Train the two RF heads on the Training split only (Spec Section 9.2).

    X is an iterable of rows already in FEATURE_ORDER (build it with
    extract_features). No hyperparameter tuning is done here: fixed
    random_state and a small fixed tree count. The returned dict is the
    serialized bundle: keys category_model, severity_model, feature_order,
    category_classes, severity_classes, model_version.

    Default model_version "placeholder" is the smoke-test marker; pass a
    commit or dataset version for a real Training-split bundle.
    """
    from sklearn.ensemble import RandomForestClassifier

    rows = [list(row) for row in X]
    categories = list(y_category)
    severities = list(y_severity)
    if not rows or len(rows) != len(categories) or len(rows) != len(severities):
        raise ValueError("X, y_category, y_severity must be non-empty and aligned")
    if len(rows[0]) != len(FEATURE_ORDER):
        raise ValueError(
            f"each row must have {len(FEATURE_ORDER)} features in FEATURE_ORDER"
        )
    category_model = RandomForestClassifier(n_estimators=20, random_state=0)
    severity_model = RandomForestClassifier(n_estimators=20, random_state=0)
    category_model.fit(rows, categories)
    severity_model.fit(rows, severities)
    return {
        "category_model": category_model,
        "severity_model": severity_model,
        "feature_order": list(FEATURE_ORDER),
        "category_classes": [str(c) for c in category_model.classes_],
        "severity_classes": [str(c) for c in severity_model.classes_],
        "model_version": str(model_version),
    }


def _validate_bundle(bundle) -> None:
    if not isinstance(bundle, dict):
        raise ProviderError("malformed: model bundle is not a mapping")
    for key in ("category_model", "severity_model", "feature_order"):
        if key not in bundle:
            raise ProviderError(f"malformed: bundle missing {key}")
    if list(bundle["feature_order"]) != list(FEATURE_ORDER):
        raise ProviderError(
            "malformed: bundle feature_order does not match FEATURE_ORDER"
        )
    if "category_classes" in bundle and set(bundle["category_classes"]) != set(
        CATEGORIES
    ):
        raise ProviderError("malformed: bundle category_classes mismatch")
    if "severity_classes" in bundle and set(bundle["severity_classes"]) != set(
        SEVERITIES
    ):
        raise ProviderError("malformed: bundle severity_classes mismatch")
    for key in ("category_model", "severity_model"):
        if not hasattr(bundle[key], "predict_proba"):
            raise ProviderError(f"malformed: bundle {key} has no predict_proba")


class RandomForestProvider:
    """DecisionProvider over a serialized two-head Random Forest bundle."""

    def __init__(self, model_path=None, bundle=None, model_version=None):
        if bundle is not None:
            self._bundle = bundle
            self.model_version = (
                model_version
                or bundle.get("model_version", "in-memory")
                if isinstance(bundle, dict)
                else "in-memory"
            )
        else:
            path_value = (
                str(model_path) if model_path is not None else os.getenv(ENV_MODEL_PATH)
            )
            if not path_value:
                raise ProviderError(
                    "missing model file: set SKLEARN_MODEL_PATH or pass model_path"
                )
            path = Path(path_value)
            if not path.is_file():
                raise ProviderError(f"model file not found: {path}")
            try:
                import joblib

                self._bundle = joblib.load(str(path))
            except Exception as exc:
                raise ProviderError(
                    f"model load failed: {type(exc).__name__}: {exc}"
                ) from exc
            stored = (
                self._bundle.get("model_version")
                if isinstance(self._bundle, dict)
                else None
            )
            self.model_version = model_version or stored or path.stem
        _validate_bundle(self._bundle)

    async def triage(self, features: IncidentFeatures) -> TriageDecision:
        start = time.perf_counter()
        try:
            vector = extract_features(features)
            if len(vector) != len(FEATURE_ORDER):
                raise ProviderError(
                    "malformed: feature vector length mismatch"
                )
            bundle = self._bundle
            category_model = bundle["category_model"]
            severity_model = bundle["severity_model"]
            category_proba = list(category_model.predict_proba([vector])[0])
            severity_proba = list(severity_model.predict_proba([vector])[0])
            category_classes = [str(c) for c in category_model.classes_]
            severity_classes = [str(c) for c in severity_model.classes_]
            category_index = max(range(len(category_proba)), key=category_proba.__getitem__)
            severity_index = max(range(len(severity_proba)), key=severity_proba.__getitem__)
            category = category_classes[category_index]
            severity = severity_classes[severity_index]
            if category not in CATEGORIES or severity not in SEVERITIES:
                raise ProviderError(
                    f"out-of-vocab choice: category={category!r} severity={severity!r}"
                )
            priority_score = float(max(category_proba))
            if not 0.0 <= priority_score <= 1.0:
                raise ProviderError(
                    f"malformed: priority_score out of range {priority_score}"
                )
            raw_scores = {}
            for label, prob in zip(category_classes, category_proba):
                raw_scores[f"category_{label}"] = float(prob)
            for label, prob in zip(severity_classes, severity_proba):
                raw_scores[f"severity_{label}"] = float(prob)
            raw_scores["category_confidence"] = priority_score
            latency_ms = (time.perf_counter() - start) * 1000.0
            return TriageDecision(
                category=category,
                severity=severity,
                priority_score=priority_score,
                investigate=severity in ("SEV1", "SEV2")
                or features.critical_rule_triggered
                or features.anomaly_score >= 0.85,
                deployment_relevant=bool(features.deployment_services)
                or any(w.recent_deployment for w in features.windows),
                provider="sklearn",
                model_version=self.model_version,
                latency_ms=latency_ms,
                raw_scores=raw_scores,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"sklearn predict failed: {type(exc).__name__}: {exc}"
            ) from exc
