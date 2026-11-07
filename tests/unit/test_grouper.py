"""Unit tests: incident grouping (Spec 4.4, 4.5)."""

from datetime import datetime, timedelta, timezone

from services.signal_processor.incident_grouper import (
    AnomalySignal,
    IncidentGrouper,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _signal(service, offset_seconds, tags=("latency_up",), traces=("t-1",)):
    return AnomalySignal(
        service=service,
        timestamp=NOW + timedelta(seconds=offset_seconds),
        score=0.9,
        statistical=0.9,
        isolation=0.9,
        change=0.9,
        trace_ids=traces,
        symptom_tags=tags,
    )


def test_related_within_window_merges():
    grouper = IncidentGrouper(
        dependencies={"order-service": ["payment-service"]},
        grouping_window_seconds=180,
    )
    first = grouper.ingest(_signal("order-service", 0))
    second = grouper.ingest(_signal("payment-service", 120))
    assert len(grouper.list_incidents()) == 1
    assert first.incident_id == second.incident_id
    assert len(second.signals) == 2
    assert set(second.affected_services) == {"order-service", "payment-service"}


def test_far_apart_opens_new_incident():
    grouper = IncidentGrouper(
        dependencies={"order-service": ["payment-service"]},
        grouping_window_seconds=180,
    )
    grouper.ingest(_signal("order-service", 0))
    grouper.ingest(_signal("payment-service", 120))
    later = grouper.ingest(_signal("order-service", 120 + 20 * 60))
    incidents = grouper.list_incidents()
    assert len(incidents) == 2
    assert later.incident_id != incidents[0].incident_id


def test_resolved_reopens_within_ten_minutes():
    grouper = IncidentGrouper(
        dependencies={"order-service": ["payment-service"]},
        grouping_window_seconds=180,
        reopen_window_seconds=600,
        clear_threshold=0.45,
    )
    incident = grouper.ingest(_signal("order-service", 0))
    resolved_at = NOW + timedelta(seconds=300)
    for index in range(10):
        grouper.record_evaluation(
            incident.incident_id,
            resolved_at + timedelta(seconds=30 * index),
            score=0.1,
            critical=False,
        )
    assert incident.resolved_at == resolved_at + timedelta(seconds=30 * 9)
    reopened = grouper.ingest(_signal("order-service", 630))
    assert reopened.incident_id == incident.incident_id
    assert reopened.resolved_at is None
    assert reopened.status == "open"
    assert len(grouper.list_incidents()) == 1
