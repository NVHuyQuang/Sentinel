"""Anti-leakage (Spec 3.3): no forbidden term in anything an agent can read."""

import re
from pathlib import Path

from simulator.recorder import forbidden_terms

ROOT = Path(__file__).resolve().parents[2]
SERVICE_SIDE = ["simulator/services.py", "simulator/telemetry.py", "simulator/app.py"]


def test_detector_catches_terms():
    assert forbidden_terms("injected fault F03 in a chaos scenario") == [
        "chaos", "f03", "fault", "injected", "scenario"]
    assert forbidden_terms("default pool, trace_id abc01f9 and f01c0ffee") == []


def test_service_side_source_is_clean():
    for rel in SERVICE_SIDE:
        assert forbidden_terms((ROOT / rel).read_text()) == [], rel


def test_every_recorded_snapshot_file_is_clean(work):
    files = [p for d in ("train", "test") for p in (work / d).rglob("*") if p.is_file()]
    assert len(files) > 10
    for p in files:
        assert forbidden_terms(p.read_text()) == [], p.name


def test_agent_plane_never_imports_harness():
    pat = re.compile(r"^\s*(from|import)\s+(simulator|evaluation)\b", re.MULTILINE)
    for p in (ROOT / "services").rglob("*.py"):
        assert not pat.search(p.read_text()), p
