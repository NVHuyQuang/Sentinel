"""Robust statistical baseline: Median/MAD deviation score (Spec 4.3, D1).

Fits the per-metric median and MAD on the Training split only.
Scoring takes the largest |robust z| across metrics and maps it to a
0-1 score with a capped linear ramp. z_threshold is a constructor
parameter with a sane starting value; tuning it belongs on the
Validation split, not inside this module.
"""

import math

_METRICS = (
    "request_rate",
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "cpu_utilization",
    "memory_utilization",
)


def _median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return None
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


class RobustBaselineDetector:
    """Per-metric Median/MAD baseline with a max-z 0-1 score.

    z_threshold scales the ramp: a z of z_threshold maps to 0.5 and
    2 * z_threshold maps to 1.0. Lowering or raising it changes how
    fast moderate deviations saturate. mad_floor guards a constant
    baseline where MAD is 0.
    """

    def __init__(self, z_threshold=3.5, mad_floor=1e-9):
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if mad_floor <= 0:
            raise ValueError("mad_floor must be positive")
        self.z_threshold = float(z_threshold)
        self.mad_floor = float(mad_floor)
        self.baseline = {}

    def fit(self, windows):
        """Estimate per-metric (median, MAD) from Training windows only."""
        collected = {}
        for name in _METRICS:
            collected[name] = [
                float(getattr(w, name)) for w in windows if getattr(w, name) is not None
            ]
        self.baseline = {}
        for name in _METRICS:
            values = collected[name]
            if not values:
                continue
            median = _median(values)
            deviations = [abs(v - median) for v in values]
            mad = _median(deviations)
            self.baseline[name] = (median, mad)
        return self

    def _robust_z(self, value, median, mad):
        if mad < self.mad_floor:
            return 0.0 if value == median else 10.0
        return abs(0.6745 * (value - median) / mad)

    def score_window(self, window):
        """Return a 0-1 deviation score for one ServiceSignalWindow."""
        if not self.baseline:
            return 0.0
        best = 0.0
        for name in _METRICS:
            entry = self.baseline.get(name)
            if entry is None:
                continue
            raw = getattr(window, name)
            if raw is None:
                continue
            median, mad = entry
            best = max(best, self._robust_z(float(raw), median, mad))
        scale = 2.0 * self.z_threshold
        score = best / scale
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        if math.isnan(score):
            return 0.0
        return score

    def set_calibration(self, lo, hi):
        """Record a Calibration-split range for later normalization.

        Stored for callers that normalize raw z values with
        calibration.normalize_to_unit. The linear ramp in
        score_window stays the working mapping until measured
        Calibration data replaces it.
        """
        self.calibration = (float(lo), float(hi))
        return self
