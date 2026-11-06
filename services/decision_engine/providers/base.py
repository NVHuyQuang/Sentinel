"""Day-1 contracts for the decision engine (Spec §5.2, §7.2–§7.4).

Provider seam (Spec §5.2 tree) — Day-2 providers plug in behind ``DecisionProvider``
without touching these contracts:

    services/decision_engine/providers/
    ├── base.py      # this file: DecisionProvider, TriageDecision, IncidentFeatures, ProviderError
    ├── laya.py      # Day-2: Laya adapter (see docs/laya-contract.md §2)
    ├── sklearn.py   # Day-2: Random Forest baseline (T1)
    ├── llm.py       # Day-2: OmniRoute structured output (T3, see docs/omniroute-contract.md)
    └── rules.py     # Day-2: deterministic rule fallback (T0)

Ground-truth fields must never enter these contracts: fault manifests
(``scenario_ground_truth``) are visible only to the benchmark evaluator
after an investigation completes (Spec §3.3, §7.1).  If a field records
the actual fault, its onset, or the expected causal chain, it does not
belong in ``IncidentFeatures``.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

# --- Spec §4.2 ---------------------------------------------------------------


class ServiceSignalWindow(BaseModel):
    service: str
    window_start: datetime
    window_end: datetime
    request_rate: float = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    latency_p99_ms: float = Field(ge=0)
    cpu_utilization: float = Field(ge=0, le=1)
    memory_utilization: float = Field(ge=0, le=1)
    db_pool_utilization: float | None = None
    dependency_timeout_rate: float | None = None
    recent_deployment: bool = False
    log_error_count: int = 0
    trace_error_count: int = 0


# --- Spec §5.2 ---------------------------------------------------------------


class IncidentFeatures(BaseModel):
    """Aggregated input for triage. Built from detector output only; never contains ground truth."""

    candidate_id: str
    opened_at: datetime
    primary_service: str
    affected_services: list[str]
    windows: list[ServiceSignalWindow]  # latest 1-min and 5-min window per affected service
    detector_scores: dict[str, float]  # statistical / isolation / change_point
    anomaly_score: float = Field(ge=0, le=1)
    critical_rule_triggered: bool = False
    deployment_services: list[str] = []  # services with a deployment in the context window
    dependency_edges: list[tuple[str, str]] = []  # observed caller -> callee among affected services
    symptom_tags: list[str] = []  # e.g. "latency_up", "error_up", "pool_saturated"


class TriageDecision(BaseModel):
    category: Literal["LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN"]
    severity: Literal["SEV1", "SEV2", "SEV3", "SEV4"]
    priority_score: float = Field(ge=0, le=1)
    investigate: bool
    deployment_relevant: bool | None = None
    provider: str
    model_version: str | None = None
    latency_ms: float | None = None
    raw_scores: dict[str, float] = {}


class ProviderError(Exception):
    """Timeout / malformed output / out-of-vocab choice → policy layer falls back to rules."""


class DecisionProvider(Protocol):
    async def triage(self, features: IncidentFeatures) -> TriageDecision:
        ...  # raise ProviderError on timeout/invalid output so the policy layer can fall back


# --- Spec §7.2 ---------------------------------------------------------------


class IncidentStatus(str, Enum):
    OPEN = "open"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    INVESTIGATED = "investigated"
    INCONCLUSIVE = "inconclusive"
    CLOSED = "closed"


class Incident(BaseModel):
    id: UUID
    service_id: UUID
    status: IncidentStatus
    category: str
    severity: str
    anomaly_score: float
    opened_at: datetime
    updated_at: datetime
    triage_decision_id: UUID | None = None
    investigation_id: UUID | None = None


# --- Spec §7.3 ---------------------------------------------------------------


class Evidence(BaseModel):
    id: UUID
    investigation_id: UUID
    source_type: str
    service: str
    observation: str
    timestamp: datetime
    source_reference: str
    query_parameters: dict
    supports: list[str]
    contradicts: list[str]


# --- Spec §7.4 ---------------------------------------------------------------


class Claim(BaseModel):
    id: str
    text: str
    kind: Literal["fact", "inference", "hypothesis"]
    evidence_ids: list[UUID]  # required non-empty when kind == "fact"


class RCAReport(BaseModel):
    incident_id: UUID
    summary: str
    affected_services: list[str]
    probable_root_cause: str | None
    confidence_level: Literal["low", "medium", "high"]
    claims: list[Claim]  # every statement in summary/timeline/root cause is expressed as a claim
    supporting_evidence_ids: list[UUID]
    contradicting_evidence_ids: list[UUID]
    timeline: list[dict]  # each entry references claim ids
    alternative_hypotheses: list[dict]
    recommendations: list[str]
    unresolved_questions: list[str]
    investigation_complete: bool
