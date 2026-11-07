"""Detector exports. See Spec 4.3 and 9.2 for split discipline."""

from services.signal_processor.detectors.calibration import (
    PLACEHOLDER,
    PLACEHOLDER_MARKER,
    calibration_range,
    normalize_to_unit,
    placeholder_calibration_set,
)
from services.signal_processor.detectors.changepoint import CusumDetector
from services.signal_processor.detectors.isolation import IsolationForestDetector
from services.signal_processor.detectors.robust import RobustBaselineDetector
from services.signal_processor.detectors.splits import (
    CALIBRATION_NORMAL,
    SPLITS,
    check_disjoint,
    split_config,
)

__all__ = [
    "CALIBRATION_NORMAL",
    "PLACEHOLDER",
    "PLACEHOLDER_MARKER",
    "SPLITS",
    "CusumDetector",
    "IsolationForestDetector",
    "RobustBaselineDetector",
    "calibration_range",
    "check_disjoint",
    "normalize_to_unit",
    "placeholder_calibration_set",
    "split_config",
]
