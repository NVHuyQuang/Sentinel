"""RCA report validation (Spec 7.4): fact claims must link to real, supporting evidence."""

import re
from collections.abc import Callable
from uuid import UUID

from services.decision_engine.providers.base import Evidence, RCAReport

_NUM = re.compile(r"\d+(?:\.\d+)?")


class ReportValidationError(ValueError):
    pass


def validate_report(report: RCAReport, evidence: list[Evidence], investigation_id: UUID) -> None:
    """Raise ReportValidationError listing every violation; return None when clean."""
    known = {e.id: e for e in evidence if e.investigation_id == investigation_id}
    errors: list[str] = []
    for claim in report.claims:
        if claim.kind != "fact":
            continue
        if not claim.evidence_ids:
            errors.append(f"claim {claim.id}: fact without evidence")
            continue
        missing = [str(i) for i in claim.evidence_ids if i not in known]
        if missing:
            errors.append(f"claim {claim.id}: unknown evidence {missing}")
            continue
        seen = {n for i in claim.evidence_ids for n in _NUM.findall(known[i].observation)}
        stray = [n for n in _NUM.findall(claim.text) if n not in seen]
        if stray:
            errors.append(f"claim {claim.id}: numbers {stray} not in linked evidence")
    for eid in report.supporting_evidence_ids + report.contradicting_evidence_ids:
        if eid not in known:
            errors.append(f"report cites unknown evidence {eid}")
    if errors:
        raise ReportValidationError("; ".join(errors))


def finalize_report(generate: Callable[[], RCAReport], evidence: list[Evidence],
                    investigation_id: UUID, retries: int = 1) -> RCAReport:
    """Validate; retry generation once; then fail explicitly (never persist a bad report)."""
    last: ReportValidationError | None = None
    for _ in range(retries + 1):
        report = generate()
        try:
            validate_report(report, evidence, investigation_id)
            return report
        except ReportValidationError as exc:
            last = exc
    raise ReportValidationError(f"report rejected after {retries + 1} attempts: {last}")
