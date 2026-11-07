"""Isolation Forest detector: multivariate anomaly score (Spec 4.3, D2).

n_estimators = 200 and contamination = 0.03 are a starting hypothesis
only, copied from the Spec 4.3 table. They are constructor parameters
so Validation tuning can change them without editing this file. The
model fits on the Training split only. Score normalization to [0, 1]
is calibrated on the Calibration split only; until measured
Calibration data exists, a sigmoid mapping on the raw decision value
stands in and is marked placeholder in the report.
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
        return 0.0
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


class IsolationForestDetector:
    """Thin wrapper around sklearn IsolationForest with 0-1 scoring."""

    def __init__(self, n_estimators=200, contamination=0.03, random_state=0):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not 0.0 < contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        self.n_estimators = int(n_estimators)
        self.contamination = float(contamination)
        self.random_state = random_state
        self.model = None
        self.medians = None
        self.calibration = None

    def _matrix(self, windows, medians):
        rows = []
        for window in windows:
            row = []
            for name in _METRICS:
                raw = getattr(window, name)
                row.append(float(raw) if raw is not None else medians[name])
            rows.append(row)
        return rows

    def _vector(self, window):
        row = []
        for name in _METRICS:
            raw = getattr(window, name)
            row.append(float(raw) if raw is not None else self.medians[name])
        return row

    def fit(self, windows):
        """Fit on Training-split windows only. Small inputs do not raise.

        With fewer than 2 rows the model stays unfitted and scoring
        returns 0.0, so a short input cannot abort the pipeline.
        """
        rows_in = list(windows)
        if len(rows_in) < 2:
            self.medians = {name: 0.0 for name in _METRICS}
            self.model = None
            return self
        medians = {}
        for name in _METRICS:
            present = [float(getattr(w, name)) for w in rows_in if getattr(w, name) is not None]
            medians[name] = _median(present) if present else 0.0
        self.medians = medians
        from sklearn.ensemble import IsolationForest

        matrix = self._matrix(rows_in, medians)
        sample_count = len(matrix)
        max_samples = min(256, sample_count)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples=max_samples,
        )
        self.model.fit(matrix)
        return self

    def set_calibration(self, lo, hi):
        """Record a Calibration-split raw-decision range for mapping."""
        self.calibration = (float(lo), float(hi))
        return self

    def score_window(self, window):
        """Return a 0-1 anomaly score, 1 = most anomalous."""
        if self.model is None or self.medians is None:
            return 0.0
        vector = self._vector(window)
        raw = float(self.model.decision_function([vector])[0])
        if self.calibration is not None:
            lo, hi = self.calibration
            span = hi - lo
            if span <= 0:
                return 0.0 if raw >= hi else 1.0
            score = (hi - raw) / span
            return min(1.0, max(0.0, score))
        # Sigmoid on the raw decision value: negative raw means more
        # anomalous, so a negative slope pushes the score toward 1.
        score = 1.0 / (1.0 + math.exp(raw * 4.0))
        return min(1.0, max(0.0, score))
