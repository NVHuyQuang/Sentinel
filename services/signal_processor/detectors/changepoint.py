"""Change-point detector: one-sided CUSUM over error_rate (Spec 4.3, D3).

Why CUSUM and not PELT: PELT needs the ruptures package, which is not
in pyproject and would add install weight for a first slice. CUSUM
needs only the standard library, runs in O(1) per 30s evaluation, and
is adequate for a single-metric mean shift. PELT tuning stays out of
scope until measured runs show drift detection is the bottleneck.
drift and threshold are constructor parameters for Validation tuning.
The reference mean and scale come from the Training split only, and
score normalization to [0, 1] is meant to be calibrated on the
Calibration split only.
"""

import math


class CusumDetector:
    """Streaming CUSUM. update() scores one value, score_series batches."""

    def __init__(self, drift=0.5, threshold=5.0, cap=None):
        if drift < 0:
            raise ValueError("drift must be >= 0")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.drift = float(drift)
        self.threshold = float(threshold)
        # cap bounds the statistic so a long shift decays within minutes after recovery.
        self.cap = None if cap is None else float(cap)
        self.mu = None
        self.sigma = 1.0
        self.g_pos = 0.0
        self.g_neg = 0.0
        self.calibration = None

    def fit(self, windows):
        """Estimate reference mu/sigma from Training-split error_rate."""
        values = [float(w.error_rate) for w in windows if w.error_rate is not None]
        if values:
            self.mu = sum(values) / len(values)
            variance = sum((v - self.mu) ** 2 for v in values) / len(values)
            spread = math.sqrt(variance)
            self.sigma = spread if spread > 0 else 1.0
        self.reset()
        return self

    def reset(self):
        self.g_pos = 0.0
        self.g_neg = 0.0
        return self

    def set_calibration(self, lo, hi):
        """Record a Calibration-split S-statistic range for mapping."""
        self.calibration = (float(lo), float(hi))
        return self

    def _score_from_statistic(self, statistic):
        if self.calibration is not None:
            lo, hi = self.calibration
            span = hi - lo
            if span <= 0:
                return 0.0 if statistic <= lo else 1.0
            score = (statistic - lo) / span
            return min(1.0, max(0.0, score))
        score = statistic / self.threshold
        return min(1.0, max(0.0, score))

    def update(self, value):
        """Advance CUSUM by one observation and return the 0-1 score."""
        if self.mu is None:
            return 0.0
        if value is None:
            current = max(self.g_pos, self.g_neg)
            return self._score_from_statistic(current)
        standardized = (float(value) - self.mu) / self.sigma
        self.g_pos = max(0.0, self.g_pos + standardized - self.drift)
        self.g_neg = max(0.0, self.g_neg - standardized - self.drift)
        if self.cap is not None:
            self.g_pos = min(self.g_pos, self.cap)
            self.g_neg = min(self.g_neg, self.cap)
        current = max(self.g_pos, self.g_neg)
        return self._score_from_statistic(current)

    def update_window(self, window):
        """Score one ServiceSignalWindow via its error_rate."""
        if window is None:
            current = max(self.g_pos, self.g_neg)
            return self._score_from_statistic(current)
        return self.update(window.error_rate)

    def score_series(self, values):
        """Replay a numeric series and return the final 0-1 score.

        When unfitted, the reference level comes from the series
        median so a constant series scores near 0 and a shifted series
        scores high without requiring a prior fit call.
        """
        series = [float(v) for v in values if v is not None]
        if not series:
            return 0.0
        fitted_before = self.mu is not None
        if not fitted_before:
            ordered = sorted(series)
            mid = len(ordered) // 2
            if len(ordered) % 2 == 1:
                self.mu = ordered[mid]
            else:
                self.mu = (ordered[mid - 1] + ordered[mid]) / 2.0
            self.sigma = 1.0
            self.reset()
        else:
            self.reset()
        score = 0.0
        for item in series:
            score = self.update(item)
        if not fitted_before:
            self.mu = None
        return score
