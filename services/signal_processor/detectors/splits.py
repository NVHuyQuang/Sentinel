"""Dataset splits per Spec 9.2. Splits are disjoint by config + seed.

Counts mirror the Spec 9.2 table. Isolation Forest fitting uses the
Training split only. Score normalization to a 0-1 range uses the
Calibration split (normal runs only). Trigger and clear thresholds are
chosen on the Validation split only. The Test split is never touched
during tuning.

There is no measured data on disk for this service yet. An inline
synthetic normal-only set (detectors.calibration) stands in for
Calibration and is marked placeholder wherever it is used.
"""

TRAINING_POSITIVE = 48
TRAINING_NORMAL = 12
CALIBRATION_POSITIVE = 0
CALIBRATION_NORMAL = 8
DEVELOPMENT_POSITIVE = 12
DEVELOPMENT_NORMAL = 4
VALIDATION_POSITIVE = 12
VALIDATION_NORMAL = 4
TEST_POSITIVE = 12
TEST_NORMAL = 4

SPLITS = {
    "training": {
        "positive": TRAINING_POSITIVE,
        "normal": TRAINING_NORMAL,
        "purpose": "fit Isolation Forest and baselines",
    },
    "calibration": {
        "positive": CALIBRATION_POSITIVE,
        "normal": CALIBRATION_NORMAL,
        "purpose": "detector score normalization, normal runs only",
    },
    "development": {
        "positive": DEVELOPMENT_POSITIVE,
        "normal": DEVELOPMENT_NORMAL,
        "purpose": "implementation and debugging",
    },
    "validation": {
        "positive": VALIDATION_POSITIVE,
        "normal": VALIDATION_NORMAL,
        "purpose": "threshold tuning",
    },
    "test": {
        "positive": TEST_POSITIVE,
        "normal": TEST_NORMAL,
        "purpose": "final evaluation, never tuned",
    },
}


def split_config(split, seed):
    """Return the config record for one split at one seed."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    entry = SPLITS[split]
    return {
        "split": split,
        "seed": seed,
        "positive": entry["positive"],
        "normal": entry["normal"],
        "purpose": entry["purpose"],
    }


def check_disjoint(configs):
    """True when every record carries a distinct (split, seed) pair.

    Disjointness is defined on the pair, never on individual telemetry
    windows. Accepts an iterable of records from split_config.
    """
    seen = set()
    for record in configs:
        key = (record["split"], record["seed"])
        if key in seen:
            return False
        seen.add(key)
    return True
