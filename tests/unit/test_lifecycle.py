"""Unit tests: lifecycle auto-resolve and close gating (Spec 4.5)."""

from datetime import datetime, timedelta, timezone

import pytest

from services.signal_processor.incident_grouper import (
    AnomalySignal,
    IncidentGrouper,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _open_incident(grouper):
    return grouper.ingest(
        AnomalySignal(
            service="order-service",
            timestamp=NOW,
            score=0.9,
            symptom_tags=("error_up",),
        )
    )


def test_nine_clean_then_tenth_resolves():
    grouper = IncidentGrouper(clear_threshold=0.45)
    incident = _open_incident(grouper)
    for index in range(9):
        grouper.record_evaluation(
            incident.incident_id,
            NOW + timedelta(seconds=30 * (index + 1)),
            score=0.1,
            critical=False,
        )
    assert incident.resolved_at is None
    assert incident.clean_eval_count == 9
    tenth_at = NOW + timedelta(seconds=300)
    grouper.record_evaluation(incident.incident_id, tenth_at, score=0.1, critical=False)
    assert incident.resolved_at == tenth_at
    assert incident.clean_eval_count == 10


def test_anomaly_resets_consecutive_clean_run():
    grouper = IncidentGrouper(clear_threshold=0.45)
    incident = _open_incident(grouper)
    for index in range(5):
        grouper.record_evaluation(
            incident.incident_id,
            NOW + timedelta(seconds=30 * (index + 1)),
            score=0.1,
            critical=False,
        )
    grouper.record_evaluation(
        incident.incident_id,
        NOW + timedelta(seconds=180),
        score=0.9,
        critical=False,
    )
    assert incident.clean_eval_count == 0
    assert incident.resolved_at is None
    for index in range(9):
        grouper.record_evaluation(
            incident.incident_id,
            NOW + timedelta(seconds=210 + 30 * index),
            score=0.1,
            critical=False,
        )
    assert incident.resolved_at is None


def test_close_needs_resolved_and_finished_investigation():
    grouper = IncidentGrouper(clear_threshold=0.45)
    incident = _open_incident(grouper)
    with pytest.raises(ValueError):
        grouper.close(incident.incident_id)
    grouper.complete_investigation(incident.incident_id)
    with pytest.raises(ValueError):
        grouper.close(incident.incident_id)
    for index in range(10):
        grouper.record_evaluation(
            incident.incident_id,
            NOW + timedelta(seconds=30 * (index + 1)),
            score=0.1,
            critical=False,
        )
    assert grouper.can_close(incident.incident_id) is True
    closed = grouper.close(incident.incident_id, now=NOW + timedelta(seconds=330))
    assert closed.status == "closed"


def test_skipped_investigation_counts_as_done():
    grouper = IncidentGrouper(clear_threshold=0.45)
    incident = _open_incident(grouper)
    grouper.complete_investigation(incident.incident_id, skipped=True)
    assert incident.investigation_done is True
    for index in range(10):
        grouper.record_evaluation(
            incident.incident_id,
            NOW + timedelta(seconds=30 * (index + 1)),
            score=0.05,
            critical=False,
        )
    closed = grouper.close(incident.incident_id)
    assert closed.status == "closed"
