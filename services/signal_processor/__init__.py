"""Signal processor: window aggregation, anomaly detectors, ensemble, grouping.

Spec tree (section 11):
    services/signal_processor/
    |-- features/   1-min + 5-min window aggregation
    |-- detectors/  statistical / isolation / change-point
    |-- ensemble.py weighted score + hysteresis + cooldown
    `-- incident_grouper.py grouping + lifecycle

Output per service per evaluation is a ServiceSignalWindow (see
services.decision_engine.providers.base) plus detector scores. The
grouper turns correlated candidates into one incident and ultimately
feeds IncidentFeatures. Telemetry stays sanitized: no labels that a
running environment would never emit appear in names or values.
"""

__all__ = []
