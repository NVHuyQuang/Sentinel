"""Deterministic policy layer (Spec §5.3, §5.1 severity override).

Runs on top of every provider — model output never bypasses it.
"""

from typing import Literal

from services.decision_engine.providers.base import (
    DecisionProvider,
    IncidentFeatures,
    ProviderError,
    TriageDecision,
)
from services.decision_engine.providers.rules import RulesProvider, rule_severity

_SEVERITY_RANK = {"SEV1": 4, "SEV2": 3, "SEV3": 2, "SEV4": 1}


def should_investigate(
    decision: TriageDecision,
    anomaly_score: float,
    critical_rule_triggered: bool,
) -> bool:
    if critical_rule_triggered:
        return True
    if decision.severity in {"SEV1", "SEV2"}:
        return True
    if anomaly_score >= 0.85:
        return True
    return decision.investigate and decision.priority_score >= 0.65


def enforce_severity_floor(
    decision: TriageDecision,
    rule_severity: Literal["SEV1", "SEV2"] | None,
) -> TriageDecision:
    """A SEV1/SEV2 rule firing raises severity to at least that level; it never lowers it."""
    if rule_severity is None:
        return decision
    if _SEVERITY_RANK[rule_severity] > _SEVERITY_RANK[decision.severity]:
        return decision.model_copy(update={"severity": rule_severity})
    return decision


async def triage_with_fallback(
    features: IncidentFeatures,
    primary: DecisionProvider,
    fallback: DecisionProvider | None = None,
) -> TriageDecision:
    """Try the primary provider (Laya); on ProviderError return rules.

    No manual step: timeout, malformed or any other ProviderError from the
    primary transparently returns fallback.triage (rules by default).
    enforce_severity_floor applies after whichever provider won, and rule
    severity can only raise. should_investigate stays with the caller.
    """
    if fallback is None:
        fallback = RulesProvider()
    try:
        decision = await primary.triage(features)
    except ProviderError:
        decision = await fallback.triage(features)
    return enforce_severity_floor(decision, rule_severity(features))
