"""Calibration helpers. Normalization uses the Calibration split only.

PLACEHOLDER = True: no measured Calibration runs exist on disk yet, so
an inline synthetic normal-only set stands in. It is labeled
placeholder here and in the Day 4 report. Do not read calibrated
weights from it; it only proves the normalization plumbing until
measured runs replace it.
"""

import random

PLACEHOLDER = True
PLACEHOLDER_MARKER = "placeholder"

_NOMINAL = {
    "request_rate": 120.0,
    "error_rate": 0.02,
    "latency_p50_ms": 80.0,
    "latency_p95_ms": 180.0,
    "latency_p99_ms": 320.0,
    "cpu_utilization": 0.40,
    "memory_utilization": 0.50,
}
_SPREAD = {
    "request_rate": 5.0,
    "error_rate": 0.005,
    "latency_p50_ms": 5.0,
    "latency_p95_ms": 10.0,
    "latency_p99_ms": 15.0,
    "cpu_utilization": 0.05,
    "memory_utilization": 0.05,
}


def placeholder_calibration_set(num_runs=8, windows_per_run=20, seed=42):
    """Synthetic normal-only calibration data (placeholder, inline).

    Each entry is a dict with the 7 window metrics plus db_pool and
    dependency values. Values jitter around the nominal normal level
    with a seeded generator. The set is normal-only by construction.
    """
    rng = random.Random(seed)
    rows = []
    for _run in range(num_runs):
        for _window in range(windows_per_run):
            row = {}
            for name, center in _NOMINAL.items():
                jitter = rng.gauss(0.0, _SPREAD[name])
                row[name] = center + jitter
            rows.append(row)
    return rows


def calibration_range(scores):
    """Return (min, max) over Calibration-split scores only."""
    values = [float(v) for v in scores]
    if not values:
        raise ValueError("calibration needs at least one score")
    return min(values), max(values)


def normalize_to_unit(value, lo, hi):
    """Map `value` to [0, 1] via (value - lo) / (hi - lo), clipped.

    When hi == lo the mapping collapses: values at or below the level
    map to 0, above map to 1. Inputs outside [lo, hi] clip to the ends.
    """
    span = hi - lo
    if span <= 0:
        return 0.0 if value <= lo else 1.0
    return min(1.0, max(0.0, (value - lo) / span))
