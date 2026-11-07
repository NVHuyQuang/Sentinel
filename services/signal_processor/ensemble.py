"""Ensemble: S = 0.35*statistical + 0.40*isolation + 0.25*change.

The weights and the formula above are a starting hypothesis from the
Spec 4.3 table, not a probability of an incident. The score is an
experimental baseline and must not be read as a calibrated chance
that an incident is under way. Weights are expressed as a tuple so
callers and tests can verify they sum to 1.0.

A candidate fires when the ensemble score exceeds the trigger level
or a deterministic critical rule fires. Trigger and clear levels form
a hysteresis pair with clear below trigger, and a per-service
cooldown suppresses repeat emissions until the candidate clears and
the cooldown ends. All three levels are function parameters with sane
starting values, meant to be chosen on the Validation split only.
Until a Validation run exists the values here are placeholder inputs
for plumbing checks, not tuned thresholds.
"""

from datetime import timedelta

WEIGHT_STATISTICAL = 0.35
WEIGHT_ISOLATION = 0.40
WEIGHT_CHANGE = 0.25
ENSEMBLE_WEIGHTS = (WEIGHT_STATISTICAL, WEIGHT_ISOLATION, WEIGHT_CHANGE)

TRIGGER_THRESHOLD = 0.70
CLEAR_THRESHOLD = 0.45
COOLDOWN_SECONDS = 180

CRITICAL_ERROR_RATE = 0.50
CRITICAL_CPU_UTILIZATION = 0.95
CRITICAL_MEMORY_UTILIZATION = 0.95


def compute_ensemble_score(statistical, isolation, change, weights=ENSEMBLE_WEIGHTS):
    """Return the weighted ensemble score in [0, 1].

    Each detector score must already be in [0, 1]. Weights must sum
    to 1.0 within 1e-9; any other tuple raises ValueError so a mis-
    typed weight vector cannot silently skew the blend.
    """
    weight_total = sum(weights)
    if abs(weight_total - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {weight_total!r}")
    for label, value in (
        ("statistical", statistical),
        ("isolation", isolation),
        ("change", change),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label} score out of [0, 1]: {value!r}")
    total = (
        float(weights[0]) * float(statistical)
        + float(weights[1]) * float(isolation)
        + float(weights[2]) * float(change)
    )
    return min(1.0, max(0.0, total))


def critical_rule_fires(
    window,
    error_rate_level=CRITICAL_ERROR_RATE,
    cpu_level=CRITICAL_CPU_UTILIZATION,
    memory_level=CRITICAL_MEMORY_UTILIZATION,
):
    """True when a deterministic availability rule fires on `window`.

    Missing (None) metrics never fire the rule. The levels are
    parameters so callers can tune them on the Validation split
    without editing this module.
    """
    if window is None:
        return False
    error_value = getattr(window, "error_rate", None)
    if error_value is not None and float(error_value) >= error_rate_level:
        return True
    cpu_value = getattr(window, "cpu_utilization", None)
    if cpu_value is not None and float(cpu_value) >= cpu_level:
        return True
    memory_value = getattr(window, "memory_utilization", None)
    return bool(memory_value is not None and float(memory_value) >= memory_level)


def is_candidate(score, critical, trigger_threshold=TRIGGER_THRESHOLD):
    """Stateless candidate check: above trigger or critical."""
    if critical:
        return True
    return float(score) > float(trigger_threshold)


class CandidateGate:
    """Per-service hysteresis + cooldown gate.

    Inactive services emit only when the score exceeds the trigger
    level (or critical fires) and the cooldown instant has passed.
    Active services stay active until the score drops below the clear
    level with no critical rule firing; repeat emissions are
    suppressed while active and during cooldown. clear_threshold
    must be strictly below trigger_threshold.
    """

    def __init__(
        self,
        trigger_threshold=TRIGGER_THRESHOLD,
        clear_threshold=CLEAR_THRESHOLD,
        cooldown_seconds=COOLDOWN_SECONDS,
    ):
        if clear_threshold >= trigger_threshold:
            raise ValueError("clear_threshold must be below trigger_threshold")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        self.trigger_threshold = float(trigger_threshold)
        self.clear_threshold = float(clear_threshold)
        self.cooldown_seconds = int(cooldown_seconds)
        self._state = {}

    def _state_for(self, service):
        if service not in self._state:
            self._state[service] = {
                "active": False,
                "last_trigger_at": None,
                "cooldown_until": None,
            }
        return self._state[service]

    def is_active(self, service):
        """Whether `service` currently holds an uncleared candidate."""
        if service not in self._state:
            return False
        return bool(self._state[service]["active"])

    def evaluate(self, service, score, critical, now):
        """Return True when a fresh candidate should be emitted now."""
        state = self._state_for(service)
        if state["active"]:
            if not critical and float(score) < self.clear_threshold:
                state["active"] = False
            return False
        cooldown_until = state["cooldown_until"]
        if cooldown_until is not None and now < cooldown_until:
            return False
        if critical or float(score) > self.trigger_threshold:
            state["active"] = True
            state["last_trigger_at"] = now
            state["cooldown_until"] = now + timedelta(seconds=self.cooldown_seconds)
            return True
        return False
