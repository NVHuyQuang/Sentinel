"""Unit tests: Spec §5.3 should_investigate + §5.1 severity override (no I/O)."""

from services.decision_engine.policy import enforce_severity_floor, should_investigate
from services.decision_engine.providers.base import TriageDecision


def _decision(**overrides) -> TriageDecision:
    base = {
        "category": "UNKNOWN",
        "severity": "SEV4",
        "priority_score": 0.0,
        "investigate": False,
        "provider": "rules",
    }
    base.update(overrides)
    return TriageDecision(**base)


def test_critical_rule_always_investigates():
    assert should_investigate(_decision(), anomaly_score=0.0, critical_rule_triggered=True)


def test_sev1_and_sev2_always_investigate():
    for severity in ("SEV1", "SEV2"):
        assert should_investigate(
            _decision(severity=severity), anomaly_score=0.0, critical_rule_triggered=False
        )


def test_high_anomaly_investigates():
    assert should_investigate(_decision(), anomaly_score=0.85, critical_rule_triggered=False)
    assert should_investigate(_decision(), anomaly_score=0.99, critical_rule_triggered=False)


def test_low_signal_does_not_investigate():
    assert not should_investigate(_decision(), anomaly_score=0.84, critical_rule_triggered=False)


def test_model_investigate_with_high_priority_investigates():
    decision = _decision(investigate=True, priority_score=0.65)
    assert should_investigate(decision, anomaly_score=0.1, critical_rule_triggered=False)


def test_rule_severity_raises_never_lowers():
    raised = enforce_severity_floor(_decision(severity="SEV4"), "SEV2")
    assert raised.severity == "SEV2"

    raised_to_sev1 = enforce_severity_floor(_decision(severity="SEV3"), "SEV1")
    assert raised_to_sev1.severity == "SEV1"

    # A SEV2 rule must not lower a SEV1 decision.
    assert enforce_severity_floor(_decision(severity="SEV1"), "SEV2").severity == "SEV1"
    # No rule firing: decision unchanged.
    assert enforce_severity_floor(_decision(severity="SEV3"), None).severity == "SEV3"
