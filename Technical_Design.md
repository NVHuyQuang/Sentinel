# SENTINEL — Technical Design Specification v1.0

Hybrid AI System for Autonomous Incident Detection, Triage & Root Cause Investigation

Implementation-ready design

14-day MVP

Research prototype

This specification defines Sentinel as a hybrid AIOps platform combining conventional anomaly detection, Laya-based System One decisions (running on localhost), and LLM-based System Two investigation through the self-hosted OmniRoute gateway.

The objective is to produce a technically credible, reproducible AI engineering portfolio project—not a generic monitoring dashboard with an LLM attached.

## 1. Final architectural decisions

| Dimension                     | Decision                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| Infrastructure                | Docker Compose simulated microservices                       |
| Autonomy                      | Autonomous detection and investigation; recommendations only |
| Research focus                | Hybrid ML + System One + System Two                          |
| Fast decision model           | Laya (running on localhost, behind a provider abstraction)   |
| Reasoning model               | OmniRoute local gateway with configurable routed models      |
| Agent orchestration           | LangGraph                                                    |
| Backend                       | Python 3.12 + FastAPI                                        |
| Frontend                      | Next.js + TypeScript                                         |
| Metrics                       | Prometheus                                                   |
| Logs                          | Grafana Loki                                                 |
| Traces                        | OpenTelemetry + Grafana Tempo                                |
| State database                | PostgreSQL                                                   |
| Event processing              | Redis Streams                                                |
| Historical incident retrieval | PostgreSQL + pgvector                                        |
| Local deployment              | Docker Compose                                               |
| Experiment tracking           | MLflow or structured experiment artifacts                    |
| Execution policy              | Read-only investigation tools; no autonomous remediation     |

### Explicit non-goals for v1

- No automatic Kubernetes changes or rollbacks.
- No autonomous deployment modifications.
- No production credential access.
- No online model retraining.
- No enterprise multi-tenancy.
- No requirement to train a foundation model.
- No claim that the agent can prove causality from correlations alone.

The key architectural principle is:

Detection identifies unusual behavior. Decision models determine triage priority. Reasoning agents investigate possible causes. Deterministic policies enforce operational boundaries.

# 2. End-to-end system architecture

Simulated Microservices Environment

API Gateway · Auth · Orders · Payments · PostgreSQL · Redis

Prometheus

Metrics

Loki

Logs

Tempo

Traces

Signal Processing Layer

Window aggregation · Feature engineering · Baselines · Anomaly detection

System One Decision Layer

Laya (localhost) · ML baseline · Rule fallback

Incident classification · Severity · Investigation priority

Policy & Workflow Router

Deduplication · Escalation · Budget · Queue

System Two Investigation

LangGraph Supervisor

Context Agent

Hypothesis Agent

Evidence Agent

RCA Agent

Incident Intelligence Output

Timeline · Evidence · Ranked hypotheses · RCA · Recommendations

Sentinel Dashboard

Live incidents · Investigation traces · Replay · Benchmark comparison

### Runtime boundaries

There are three separate control planes.

Observability plane: Collects telemetry and exposes read-only queries. This plane does not invoke LLMs.

Intelligence plane: Processes signals, calls Laya (localhost) or alternative decision providers, and performs investigation through the local OmniRoute OpenAI-compatible gateway.

Application plane: Manages incident state, API requests, users, audit trails, dashboards, and experiment results.

Separating these planes prevents the agent workflow from becoming tightly coupled to telemetry ingestion.

# 3. Simulated infrastructure specification

The simulated environment should resemble a small e-commerce backend, because it naturally produces distributed incidents involving application services, databases, dependencies, and deployments.

## 3.1 Services

| Service            | Responsibility                          | Dependencies                       |
| ------------------ | --------------------------------------- | ---------------------------------- |
| API Gateway        | Routes incoming requests                | Auth, Orders                       |
| Auth Service       | Authentication and token validation     | PostgreSQL                         |
| Order Service      | Order creation and retrieval            | PostgreSQL, Payment, Redis (read cache) |
| Payment Service    | Simulated payment processing            | Redis, External Provider Simulator |
| Provider Simulator | Configurable downstream API             | None                               |
| PostgreSQL         | Application persistence                 | None                               |
| Redis              | Cache and simulated queue               | None                               |
| Traffic Generator  | Produces reproducible workloads         | API Gateway                        |
| Fault Controller   | Activates predefined incident scenarios | Demo services                      |

All services should expose `/health`, `/ready`, and `/metrics` where appropriate.

Use OpenTelemetry propagation for requests crossing service boundaries.

The Fault Controller and Traffic Generator are test harness components, not monitored services. They must be excluded from the service allowlist, from `get_service_dependencies`, and from every telemetry stream the agent can query (see 3.3 sanitization rules).

Container metrics: CPU and memory utilization (F04/F05) come from cAdvisor, added to the Compose stack and scraped by Prometheus, with application-level process metrics (for example `process_cpu_seconds_total`, `process_resident_memory_bytes`) as a cross-check. Each simulated service must run with explicit CPU and memory limits so that utilization is a well-defined ratio. On Docker Desktop (Windows/WSL2), container metrics are noisier than on native Linux; calibrate CPU thresholds on the same host that records the benchmark snapshots and record the host configuration with each dataset version.

## 3.2 Observability requirements

Metrics

Collect these core signals at 15-second intervals:

- Request rate.
- HTTP error rate.
- P50/P95/P99 latency.
- CPU and memory utilization.
- Database connection pool utilization.
- Database query latency.
- Dependency timeout rate.
- Queue depth and processing lag.

Logs

Use structured JSON logs containing:

```
{
  "timestamp": "2026-11-06T03:00:00Z",
  "service": "payment-service",
  "level": "ERROR",
  "trace_id": "abc123",
  "span_id": "def456",
  "event": "dependency_timeout",
  "duration_ms": 3000,
  "message": "Payment provider request timed out"
}
```

Traces

Capture:

- Service-to-service calls.
- Database operations.
- Dependency requests.
- Span status.
- Duration.
- Error attributes.

Ensure the agent can correlate logs with traces using trace IDs.

## 3.3 Fault injection catalog

Implement six deterministic scenarios.

| ID  | Fault                               | Primary target (default)          | Allowed alternate targets | Expected observable symptoms                    |
| --- | ----------------------------------- | --------------------------------- | ------------------------- | ----------------------------------------------- |
| F01 | Database connection pool exhaustion | Order Service DB pool             | Auth Service DB pool      | Higher query wait time, API latency, timeouts   |
| F02 | Downstream provider slowdown        | Provider Simulator (seen via Payment) | none                  | Payment latency, dependency timeouts            |
| F03 | Application regression              | Order Service (new release)       | Payment Service           | Increased 500 errors after a deployment event   |
| F04 | CPU saturation                      | Payment Service                   | Order Service             | Higher CPU, increased response latency          |
| F05 | Memory pressure                     | Auth Service                      | Order Service             | Memory growth, degraded service responsiveness  |
| F06 | Cache unavailability                | Redis (seen via Order Service read cache) | Redis (seen via Payment) | Cache errors, database load increase       |

The "Affected service" run variation in 9.2 chooses only among the listed primary and alternate targets, so every fault has a defined, physically meaningful causal chain. Example chain for F01: pool max lowered or connections held, then Order Service pool utilization at 100%, then query wait time rises, then Order latency and timeouts, then Gateway 5xx.

Fault injection must operate only on the isolated demo environment. No real production endpoints or external services should be affected.

For each scenario, store a private ground-truth manifest containing the actual fault, onset time, affected services, expected causal chain, and recovery time.

Do not expose that manifest to the investigation agents. The benchmark runner uses it only after an investigation is complete.

### Telemetry sanitization rules (anti-leakage)

Telemetry must contain only what a real environment would emit. Fault injection must never label itself in anything the agents can read.

- No log line, metric name, label, span attribute, container name, environment variable, or service name may contain terms such as `fault`, `inject`, `chaos`, `scenario`, `F01`-`F06`, `ground_truth`, or `simulated_fault`.
- Faults act through realistic mechanisms (lowered pool size, held connections, CPU burn, memory leak, stopped Redis, latency added in the Provider Simulator) and produce only the natural symptoms of those mechanisms.
- Deployment events come from a release-pipeline record (service, version, timestamp, change summary written as a normal release note). F03 is an actual behavior change behind that new version, not a flag called "regression".
- The Fault Controller writes its manifest to a store the agent tools cannot reach, and emits no telemetry into Prometheus, Loki, or Tempo.
- CI check: grep every recorded snapshot and every tool response fixture for the forbidden terms above and fail the build on any hit.

# 4. Data processing and anomaly detection

## 4.1 Processing windows

Use two window sizes:

- Short window: 1 minute, for rapid changes.
- Context window: 5 minutes, for baseline comparison and feature aggregation.

Prometheus scrapes every 15 seconds. The signal processor evaluates every 30 seconds.

Each evaluation produces a normalized feature vector.

## 4.2 Feature schema

```python
from datetime import datetime

from pydantic import BaseModel, Field


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
```

Missing telemetry must remain explicitly missing. Do not silently replace unavailable metrics with zero.

## 4.3 Detection algorithms

Use three complementary detectors.

| Detector                    | Purpose                         | Initial configuration         |
| --------------------------- | ------------------------------- | ----------------------------- |
| Robust statistical baseline | Detect sudden metric deviations | Median/MAD                    |
| Isolation Forest            | Detect multivariate anomalies   | 200 trees, contamination 0.03 |
| Change-point detector       | Detect distribution shifts      | PELT or CUSUM                 |

The configurations above are starting hypotheses, not empirically optimized settings.

### Ensemble score

For the MVP, normalize each detector output to a comparable 0–1 scale using the dedicated Calibration split (normal-only runs, see 9.2). Isolation Forest is fit and baselines are estimated on the Training split only; the Validation split is used only to choose thresholds, and the Test split is never touched during tuning.

Then compute:

\\[ S\_{\text{anomaly}} = 0.35S\_{\text{statistical}} + 0.40S\_{\text{isolation}} + 0.25S\_{\text{change}} \\]

This weighted score is an experimental baseline.

It should not be interpreted as a calibrated probability of an incident.

Trigger an anomaly candidate when:

- The ensemble score exceeds a calibrated threshold; or
- A deterministic critical safety/availability rule fires.

Add hysteresis and cooldown windows to reduce repeated alerts.

## 4.4 Incident grouping

Group anomalies by:

- Service identity.
- Temporal proximity.
- Shared trace/dependency context.
- Similarity of observed symptoms.

A suggested initial grouping window is 3 minutes.

The system should create one incident containing multiple correlated signals rather than opening separate incidents for every abnormal metric.

## 4.5 Incident lifecycle and closing

Status flow: `open` → `triaged` → `investigating` → `investigated` | `inconclusive` → `closed`.

- Auto-resolve: an incident gets `resolved_at` when no member service has an active anomaly (ensemble score below the clear threshold and no critical rule firing) for 5 consecutive minutes (10 evaluations at 30 seconds). The clear threshold is lower than the trigger threshold (hysteresis).
- Re-open: a new anomaly on a member service within the 3-minute grouping window of the last signal merges into the existing incident instead of opening a new one. After `resolved_at`, a new anomaly within 10 minutes re-opens the same incident; later anomalies open a new one.
- `closed` is set only after `resolved_at` is set and the investigation has finished (or been skipped by policy). Closing never triggers any operational action.
- An investigation running when the incident resolves still runs to completion; its report records that the incident resolved during investigation.

# 5. System One — Laya integration specification

The System One layer should be a replaceable decision subsystem, not a hard dependency embedded throughout the application.

## 5.1 Decision responsibilities

Laya (running locally on localhost) receives an `IncidentFeatures` object (an incident candidate aggregated across all correlated signals and services, see 5.2) and makes structured triage decisions. Laya and the cloud LLM baseline (T3) are evaluated zero-shot with the same choice/criteria schema; any few-shot or prompt variant must be applied identically to both and recorded in the experiment configuration.

| Decision               | Type             | Output                                                        |
| ---------------------- | ---------------- | ------------------------------------------------------------- |
| Incident category      | Choice           | `LATENCY`, `ERROR_SPIKE`, `RESOURCE`, `DEPENDENCY`, `UNKNOWN` |
| Severity               | Choice           | `SEV1`, `SEV2`, `SEV3`, `SEV4`                                |
| Investigation priority | Score            | Normalized score                                              |
| Investigation required | Boolean decision | Yes/No with supporting score where available                  |
| Deployment relevance   | Boolean decision | Whether deployment deserves investigation                     |

Treat these as predictions, not authoritative facts.

For severity, define operational rules first. Core services are API Gateway, Auth, Order, and Payment. Thresholds are provisional and are tuned only on the Validation split.

| Severity | Meaning                                     | Deterministic rule (evaluated on the 1-minute window, sustained for at least 2 consecutive evaluations) |
| -------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| SEV1     | Critical simulated service unavailability   | API Gateway error_rate ≥ 50%, or any core service failing `/ready` for ≥ 60 s                          |
| SEV2     | Major degradation affecting core operations | Any core service error_rate ≥ 20%, or P95 latency ≥ 5× its baseline median                             |
| SEV3     | Partial degradation or elevated errors      | Any service error_rate ≥ 5%, or P95 latency ≥ 2× its baseline median                                   |
| SEV4     | Low-impact anomaly requiring observation    | Anomaly candidate that meets none of the rules above                                                    |

Severity ground-truth labels for the benchmark are assigned from an `impact_level` field in the scenario manifest, set when the scenario is designed. They are never computed from the observed metrics by these same rules; otherwise the rule-based baseline (T0) would score perfectly by construction.

The policy engine must be able to override a model decision when deterministic evidence indicates a critical incident (a SEV1 or SEV2 rule firing raises the severity to at least that level; it never lowers it).

## 5.2 Provider abstraction

Use a typed interface:

```
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field


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


class DecisionProvider(Protocol):
    async def triage(self, features: IncidentFeatures) -> TriageDecision:
        ...  # raise ProviderError on timeout/invalid output so the policy layer can fall back
```

Implement four providers:

```
services/decision_engine/providers/
├── base.py      # DecisionProvider, TriageDecision, IncidentFeatures, ProviderError
├── laya.py
├── sklearn.py
├── llm.py       # OmniRoute structured output (T3)
└── rules.py
```

All providers must return the same `TriageDecision` schema.

### Laya adapter implementation

The Laya adapter should:

1. Receive a validated `IncidentFeatures` object.
2. Format the feature vector into structured incident text and question schemas (choices and criteria for category, severity, and urgency).
3. Query the local Laya model instance running on localhost (via in-process router or localhost inference endpoint).
4. Parse the returned decisions and scores.
5. Map outputs to Sentinel's internal `TriageDecision` schema.
6. Record inference latency (ms), device mode (CPU/GPU), and provider metadata.
7. Handle timeouts and invalid responses cleanly.
8. Return an explicit error so the policy layer can select a fallback.

Laya runs locally on localhost to achieve ultra-low triage latency with zero external API dependencies or cloud cost for high-frequency System One decisions.

Critical fallback rule: If Laya is unavailable on localhost, Sentinel must remain operational using the baseline provider.

## 5.3 Decision policy

A suggested policy:

```
def should_investigate(
    decision: TriageDecision,
    anomaly_score: float,
    critical_rule_triggered: bool,
) -> bool:
    if critical_rule_triggered:
        return True
    if decision.severity in {"SEV1", "SEV2"}:
        return True
    if anomaly_score >= 0.85:
        return True
    return decision.investigate and decision.priority_score >= 0.65
```

Thresholds are provisional and should be tuned on validation scenarios.

The policy should additionally enforce deduplication, cooldowns, and maximum concurrent investigations.

### 5.4 Laya integration contract (Spike 0, blocking)

The adapter details above are intent, not a verified contract. Before any Laya code beyond a stub, produce `docs/laya-contract.md` from a real call against the local instance and answer:

- How Laya is invoked (in-process library or localhost HTTP endpoint), its version, and how it is started in Docker Compose or on the host.
- The exact request format (input text, choice/criteria schema, maximum input length) and the exact response format (labels, scores, whether scores are probabilities, logits, or uncalibrated).
- Whether output is deterministic for the same input, and the measured P50/P95 latency on CPU and, if available, GPU.
- Failure modes (timeout, malformed output, out-of-vocabulary choices) and how each maps to `ProviderError`.

Pass criteria: 50 representative `IncidentFeatures` samples return parseable output for all five decisions, latency is recorded, and killing the Laya process makes the system fall back to `rules.py` with no manual action. Until this passes, T2 and H2 are marked blocked in the experiment plan; the provider interface stays unchanged and the other providers continue to work.

# 6. System Two — Multi-agent investigation

## 6.1 Architecture choice

Use LangGraph with a deterministic supervisor and four specialized reasoning nodes.

This is preferable to an unrestricted swarm of agents because the investigation needs:

- Explicit state transitions.
- Controlled tool access.
- Bounded iterations.
- Persistent checkpoints.
- Auditable evidence.
- Reproducible evaluation.

The supervisor should be application code. Individual reasoning agents use OmniRoute through its local OpenAI-compatible endpoint for structured reasoning, hypothesis formation, evidence verification, and RCA synthesis. OmniRoute handles the configured upstream providers and routing strategy; Sentinel must not call an upstream model API directly.

The default local configuration is:

```
LLM_PROVIDER=omniroute
LLM_BASE_URL=http://localhost:20128/v1
LLM_MODEL=auto/deepseek
LLM_API_KEY=local-omniroute
```

`LLM_MODEL` may be changed to another OmniRoute route such as `auto/best-free` for development or `auto/cheap` for cost-sensitive runs. The selected route and resolved upstream model must be recorded for every experiment.

## 6.2 Agent contracts

Agent A — Context Collector

Input: Incident ID, affected service, anomaly signals.

Tools: `query_metrics`, `query_logs`, `query_traces`, `get_service_dependencies`, `get_deployment_events`.

Output: Incident context containing symptoms, affected services, timeline, and evidence references.

Restriction: Cannot propose remediation or change infrastructure.

Agent B — Hypothesis Generator

Input: Incident context.

Task: Generate up to three distinct, testable root-cause hypotheses.

Output: Hypotheses with supporting signals, contradicting signals, and proposed verification queries.

Restriction: Cannot declare a root cause confirmed.

Agent C — Evidence Investigator

Input: Ranked hypotheses and available tool catalog.

Task: Execute bounded, read-only queries to gather evidence for and against each hypothesis.

Output: Evidence records and hypothesis assessments.

Restriction: Maximum 12 tool calls per investigation round.

Agent D — RCA Synthesizer

Input: All validated evidence, hypotheses, and incident timeline.

Task: Produce an evidence-backed incident analysis and remediation recommendations.

Output: Structured RCA report.

Restriction: Must distinguish verified facts from hypotheses and acknowledge uncertainty.

## 6.3 Investigation state

```python
from typing import TypedDict


class InvestigationState(TypedDict):
    incident_id: str
    service: str
    severity: str
    context: dict
    hypotheses: list[dict]
    evidence: list[dict]
    iteration: int
    tool_calls: int
    selected_hypothesis_id: str | None
    rca_report: dict | None
    status: str
    errors: list[str]
```

For production-quality implementation, replace generic dictionaries with dedicated Pydantic models at the API and persistence boundaries.

## 6.4 LangGraph execution flow

START — Incident accepted

Context Collector

Hypothesis Generator

Evidence Investigator

Evidence Sufficiency Gate

Enough evidence, or budget exhausted?

Sufficient / budget exhausted

RCA Synthesizer

More evidence needed

Refine Hypotheses

END — RCA Report / Inconclusive

Set these initial execution limits:

| Limit                               | Value       |
| ----------------------------------- | ----------- |
| Maximum investigation rounds        | 3           |
| Maximum tool calls per round        | 12          |
| Maximum total tool calls            | 30          |
| Maximum investigation duration      | 180 seconds |
| Maximum hypotheses                  | 3           |
| Maximum simultaneous investigations | 3           |

The orchestrator should stop when a limit is reached and produce an incomplete or inconclusive report rather than inventing a confident explanation.

## 6.5 Read-only investigation tools

Define the following tools:

```python
def query_metrics(
    service: str, metric: str, start: datetime, end: datetime
): ...
def query_logs(
    service: str, start: datetime, end: datetime,
    level: str | None = None, limit: int = 100
): ...
def query_traces(
    service: str, start: datetime, end: datetime, errors_only: bool = False
): ...
def get_deployment_events(service: str, start: datetime, end: datetime): ...
def get_service_dependencies(service: str): ...
def search_historical_incidents(query: str, limit: int = 5): ...
```

Every tool must enforce server-side service allowlists, time-range limits, query limits, and timeouts.

Do not let the LLM submit arbitrary PromQL, LogQL, SQL, or shell commands directly. Build queries from validated parameters in trusted application code.

All tools read through a `TelemetryBackend` interface with two implementations: `LiveBackend` (Prometheus, Loki, Tempo) and `SnapshotBackend` (a recorded, timestamped snapshot on disk, see 10). The agent code and tool code are identical in both modes; only the backend is injected. This is what makes recorded replay and fair model comparison possible.

# 7. Data contracts and database schema

## 7.1 Core entities

Use UUIDs for internal identifiers and UTC timestamps throughout.

| Entity                  | Key fields                                                   |
| ----------------------- | ------------------------------------------------------------ |
| `services`              | id, name, environment, metadata                              |
| `signal_windows`        | id, service_id, start, end, features                         |
| `anomaly_events`        | id, service_id, score, detectors, timestamp                  |
| `incidents`             | id, status, severity, category, opened_at, resolved_at       |
| `triage_decisions`      | id, incident_id, provider, model_version, decision, latency  |
| `investigations`        | id, incident_id, status, iteration, started_at, completed_at |
| `hypotheses`            | id, investigation_id, description, status, rank              |
| `evidence`              | id, investigation_id, source_type, reference, payload        |
| `agent_events`          | id, investigation_id, agent, event_type, timestamp, payload  |
| `rca_reports`           | id, investigation_id, conclusion, recommendations            |
| `experiment_runs`       | id, configuration, dataset_version, metrics                  |
| `scenario_ground_truth` | scenario_id, fault_type, root_cause, affected_services       |

The ground-truth table should be accessible only to the benchmark evaluator, not to the investigation API or agent tools.

## 7.2 Incident API schema

```python
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


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
```

## 7.3 Evidence schema

Every factual claim in the RCA must point to evidence.

```python
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
```

`source_reference` should be an immutable reference to the retrieved result or stored snapshot, not an invented citation.

For reproducibility, store the query parameters, retrieval timestamp, result checksum, and a bounded sanitized snapshot.

## 7.4 RCA report schema

```
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
```

Validation before persistence: reject (retry once, then fail explicitly) any report where a `fact` claim has no `evidence_ids`, an evidence ID does not exist for this investigation, or the claim text is not supported by the referenced evidence observation. "100% of factual claims linked to evidence" is measured on `claims` of kind `fact`; prose outside `claims` is not allowed in the persisted report.

Use `confidence_level` values such as `low`, `medium`, and `high` initially. Do not present uncalibrated LLM self-assessment as a statistically meaningful probability.

# 8. Backend API specification

All endpoints prefixed with `/api/v1`.

| Method | Endpoint                        | Purpose                   |
| ------ | ------------------------------- | ------------------------- |
| GET    | `/services`                     | List monitored services   |
| GET    | `/services/{id}/health`         | Service health summary    |
| GET    | `/incidents`                    | List and filter incidents |
| GET    | `/incidents/{id}`               | Incident details          |
| POST   | `/incidents/{id}/investigate`   | Start an investigation    |
| GET    | `/investigations/{id}`          | Investigation state       |
| GET    | `/investigations/{id}/events`   | Execution event stream    |
| GET    | `/investigations/{id}/evidence` | Evidence records          |
| GET    | `/incidents/{id}/report`        | RCA report                |
| GET    | `/decisions/{id}`               | Decision model output     |
| POST   | `/scenarios/{id}/replay`        | Start controlled replay   |
| GET    | `/experiments`                  | Experiment history        |
| POST   | `/experiments/run`              | Start benchmark run       |
| GET    | `/experiments/{id}/results`     | Evaluation results        |

Use Server-Sent Events for live investigation updates in the MVP.

All state-changing endpoints should require authentication, even in the demo environment, and replay operations should only affect isolated simulated services.

# 9. Experimental design

This is the research component that will distinguish Sentinel from a typical AI agent demo.

## 9.1 Research questions

RQ1 — Detection: Does a hybrid anomaly detector improve incident detection compared with simple statistical thresholds?

RQ2 — Triage: Can Laya (running locally on localhost) provide competitive incident classification and prioritization compared with traditional ML and cloud LLM-based classification?

RQ3 — Investigation: Does structured multi-agent investigation improve RCA accuracy compared with a single LLM agent?

RQ4 — Efficiency: Does the System One + System Two architecture reduce LLM invocations, investigation latency, or cost without materially reducing RCA quality?

RQ5 — Reliability: Does evidence validation reduce unsupported RCA claims?

## 9.2 Dataset generation

For the MVP, generate 36 held-out incident runs across six fault types.

Planned evaluation dataset composition.

02468DB poolDependencyRegressionCPUMemoryCache

Planned benchmark: 36 positive incident runs, six per fault category.

Also generate normal or non-incident runs (32 in total across the splits below), including workload spikes that should not automatically become incidents. The Training split adds 48 positive runs on top of the 36 evaluation runs; the "36 held-out" figure refers to the Development, Validation, and Test splits only.

Each run should vary:

- Fault onset.
- Fault duration.
- Request load.
- Fault intensity.
- Background noise.
- Affected service.

### Dataset split

To avoid leakage, separate scenarios by configuration and random seed, not by individual telemetry windows.

| Split       | Positive runs | Normal runs | Purpose                                                          |
| ----------- | ------------- | ----------- | ---------------------------------------------------------------- |
| Training    | 48            | 12          | Fit Isolation Forest/baselines and the Random Forest triage (T1) |
| Calibration | 0             | 8           | Detector score normalization (normal-only)                       |
| Development | 12            | 4           | Implementation and debugging                                     |
| Validation  | 12            | 4           | Threshold tuning                                                 |
| Test        | 12            | 4           | Final evaluation                                                 |

The Training split exists because 12 positive runs are too few to train T1 for five categories and four severities without overfitting. Runs are generated from the scenario generator, so extra runs are cheap; all splits use disjoint configurations and seeds. Laya and the cloud LLM baseline use no training data (zero-shot, see 5.1).

For stronger evidence, repeat the final evaluation across multiple seeds after the MVP. The initial test set is small, so results should be reported with uncertainty rather than presented as definitive superiority.

The agents must not receive scenario IDs, fault labels, ground-truth files, or benchmark-specific hints.

## 9.3 Detection experiments

| Experiment | Detection method            |
| ---------- | --------------------------- |
| D0         | Static thresholds           |
| D1         | Robust statistical baseline |
| D2         | Isolation Forest            |
| D3         | Change-point detector       |
| D4         | Hybrid ensemble             |

Metrics:

\\[ \text{Precision}=\frac{TP}{TP+FP} \\]

\\[ \text{Recall}=\frac{TP}{TP+FN} \\]

\\[ F_1=\frac{2PR}{P+R} \\]

Also report false alerts per hour and detection delay:

\\[ T\_{\text{delay}}=T\_{\text{detected}}-T\_{\text{onset}} \\]

Evaluate at the incident-event level, not only per telemetry window. Define a matching tolerance and count repeated alerts for the same incident separately when measuring alert noise.

## 9.4 System One triage experiments

Use the same normalized incident features for all decision providers.

| Variant | Provider                                        | Purpose                                         |
| ------- | ----------------------------------------------- | ----------------------------------------------- |
| T0      | Rule-based                                      | Deterministic baseline                          |
| T1      | Random Forest                                   | Conventional supervised ML                      |
| T2      | Laya (localhost) **[BLOCKED — pending Spike 0, §5.4]** | System One fast decision model                  |
| T3      | OmniRoute `auto/deepseek` structured output     | Routed LLM classification baseline              |

Measure:

- Incident category macro-F1.
- Severity accuracy and macro-F1.
- Investigation routing precision/recall.
- P50/P95 decision latency.
- Cost per 1,000 decisions.
- Invalid output rate.
- Provider failure rate.

For probabilistic outputs, additionally measure Brier score or calibration error where appropriate.

Important: Laya's suitability is an empirical question. The evaluation should be designed to allow Laya to perform better, similarly, or worse than the baselines.

## 9.5 RCA experiments

Compare four configurations:

| Variant | Architecture                                |
| ------- | ------------------------------------------- |
| R0      | Single LLM, fixed incident summary          |
| R1      | Single LLM with read-only tools             |
| R2      | Multi-agent investigation                   |
| R3      | Multi-agent + historical incident retrieval |

Keep the OmniRoute route, resolved model, telemetry snapshot, available evidence, and total execution budget comparable.

LLM reproducibility: use the lowest-variance sampling settings the API offers (temperature 0 or the provider's recommended deterministic setting), record the exact model ID, prompt version, and parameters in `experiment_runs`, and cache every LLM response keyed by hash(model, parameters, prompt) so a benchmark can be re-run offline. Because low temperature does not guarantee identical output, run each variant on each test incident at least 3 times and report mean and spread (for example bootstrap confidence intervals), not single scores. With a 12-incident test set, differences between variants are reported as indicative, not conclusive.

Evaluate:

- Root-cause category accuracy.
- Exact fault identification.
- Evidence precision.
- Unsupported claim rate.
- Alternative hypothesis coverage.
- Investigation duration.
- Token consumption.
- Total cost.

Root-cause correctness should be judged against hidden scenario ground truth, while evidence quality should be checked against retrieved telemetry records.

## 9.6 End-to-end hybrid ablation

This experiment tests the actual architecture.

| Variant | Detection | Triage           | Investigation                  |
| ------- | --------- | ---------------- | ------------------------------ |
| H0      | Rules     | Rules            | Single LLM (OmniRoute `auto/deepseek`)  |
| H1      | Hybrid ML | Rules            | Multi-agent (OmniRoute `auto/deepseek`) |
| H2      | Hybrid ML | Laya (localhost) **[BLOCKED — pending Spike 0, §5.4]** | Multi-agent (OmniRoute `auto/deepseek`) |
| H3      | Hybrid ML | Cloud LLM        | Multi-agent (OmniRoute `auto/deepseek`) |

For each variant, measure end-to-end detection-to-report time, RCA success rate, false investigations, LLM calls, and cost.

To separate model quality from routing effects, also run a fixed-incident experiment where every variant receives the same incident candidates.

Otherwise, a system that misses difficult incidents could appear cheaper simply because it investigates fewer cases.

## 9.7 Acceptance criteria

These are engineering targets, not expected research results.

| Criterion                        | MVP target                                    |
| -------------------------------- | --------------------------------------------- |
| End-to-end execution             | All six fault categories supported            |
| Incident detection recall        | ≥ 80% on test incidents                       |
| False alert rate                 | Reported and compared with baseline           |
| RCA category accuracy            | ≥ 70% on investigated test incidents          |
| Evidence traceability            | 100% of `fact` claims in `RCAReport.claims` link to existing, supporting evidence |
| Invalid structured outputs       | Under 2% after validation/retry               |
| Investigation timeout compliance | 100%                                          |
| Unauthorized remediation actions | 0                                             |
| Reproducible benchmark           | Same config and seed reproduce scenarios      |

If results fall below these targets, report them rather than tuning on the held-out test set.

# 10. Incident replay engine

The replay engine is one of Sentinel's most valuable portfolio features.

It allows you to demonstrate and evaluate investigations without waiting for real incidents.

## Replay workflow

1. Select a scenario.
2. Initialize the isolated demo environment.
3. Start a controlled traffic workload.
4. Activate the predefined fault.
5. Collect telemetry.
6. Run detection and triage.
7. Trigger investigation.
8. Generate the RCA report.
9. Compare results against hidden ground truth.
10. Store metrics and execution traces.

Support two modes:

Live simulation: Recreates the fault in Docker and captures telemetry in real time.

Recorded replay: Loads a previously captured, timestamped telemetry snapshot. This mode is essential for fair model comparisons because every model sees identical evidence.

Snapshot format: one directory per run containing metrics (Parquet, one file per metric with timestamp, labels, value), logs (JSON Lines), traces (JSON Lines of spans), deployment events, the service dependency map, a `meta.json` (dataset version, generator config hash, seed, host configuration, schema version), and a checksum manifest. The ground-truth manifest is stored outside the snapshot directory. `SnapshotBackend` serves the tool queries from these files, applying the same allowlists, time-range limits, and result limits as `LiveBackend`, and is only allowed to return data up to the investigation's "now" so the agent cannot read the future. The recorder writes snapshots from the live Prometheus/Loki/Tempo stack after the sanitization check in 3.3 passes.

For the initial benchmark, prioritize recorded replay. Live simulation is useful for demonstrations but introduces environmental variance.

# 11. Implementation repository structure

```
sentinel/
├── apps/
│   ├── dashboard/
│   └── api/
│
├── packages/
│   ├── contracts/
│   └── shared/
│
├── services/
│   ├── signal_processor/
│   │   ├── features/
│   │   ├── detectors/
│   │   ├── ensemble.py
│   │   └── incident_grouper.py
│   │
│   ├── decision_engine/
│   │   ├── providers/
│   │   │   ├── laya.py
│   │   │   ├── sklearn.py
│   │   │   ├── llm.py
│   │   │   └── rules.py
│   │   ├── policy.py
│   │   └── calibration.py
│   │
│   ├── telemetry/
│   │   ├── backend.py        # TelemetryBackend interface
│   │   ├── live.py           # Prometheus / Loki / Tempo
│   │   └── snapshot.py       # recorded replay
│   │
│   ├── investigation/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── agents/
│   │   │   ├── context.py
│   │   │   ├── hypothesis.py
│   │   │   ├── evidence.py
│   │   │   └── rca.py
│   │   ├── tools/
│   │   └── prompts/
│   │
│   └── replay/
│       ├── controller.py
│       ├── scenarios/
│       └── recorder.py
│
├── simulator/
│   ├── gateway/
│   ├── auth/
│   ├── orders/
│   ├── payments/
│   ├── provider/
│   ├── traffic/
│   └── faults/
│
├── observability/
│   ├── prometheus/
│   ├── loki/
│   ├── tempo/
│   └── otel-collector/
│
├── evaluation/
│   ├── datasets/
│   ├── benchmarks/
│   ├── metrics/
│   └── reports/
│
├── infrastructure/
│   ├── docker/
│   └── migrations/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/
│   ├── architecture.md
│   ├── api-spec.md
│   ├── agent-design.md
│   ├── experiments.md
│   └── security.md
│
├── CLAUDE.md
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

# 12. 14-day implementation plan

The schedule assumes a focused research-prototype effort. The full design is larger than a guaranteed two-week delivery, so each day has a primary deliverable and a clear scope boundary.

IMPLEMENTATION TIMELINE

## Day 1 of 14

Foundation

1

2

3

4

5

6

7

8

9

10

11

12

13

14

## Architecture and contracts

- Initialize monorepo and Docker Compose
- Define Pydantic contracts and API boundaries
- Create database migrations and service interfaces

Definition of done

API health endpoint, PostgreSQL and Redis operational

&#x20;PreviousNext day&#x20;

## Pre-implementation spikes (before Day 1 coding)

Half a day, blocking. Each spike ends with a short note in `docs/` and a pass/fail result.

1. Laya contract: see 5.4.
2. OmniRoute contract: confirm the local endpoint (`http://localhost:20128/v1`), configured route (default `auto/deepseek`), model discovery behavior, JSON/structured output behavior and failure rate, tool-calling support and its limits, routing fallbacks, rate limits and timeouts, context window, token/cost reporting, and which sampling parameters are honored.
3. Telemetry round-trip: write 10 sample metrics, logs, and traces through the stack, record them with the snapshot format from 10, and read them back through `SnapshotBackend` with identical results.
4. Container metrics: confirm cAdvisor reports per-container CPU and memory under Docker Desktop on this host, and that F04/F05 produce a visible signal.

## Scope control and fallback plan

If implementation falls behind schedule, prioritize the following order:

1. End-to-end telemetry and incident detection.
2. One working decision provider with Laya integration on localhost.
3. Evidence-backed investigation.
4. Recorded replay and evaluation.
5. Dashboard and visual polish.

Defer pgvector, sophisticated service dependency graphs, extensive ML hyperparameter tuning, and additional fault categories if needed.

A working, measured three-scenario pipeline is more valuable than a six-scenario architecture that cannot execute end to end.

# 13. Testing, security and reliability

## Testing strategy

| Test level  | Focus                                                                |
| ----------- | -------------------------------------------------------------------- |
| Unit        | Feature aggregation, anomaly scoring, decision mapping, policy logic |
| Contract    | Laya adapter, OmniRoute structured outputs, telemetry tools           |
| Integration | Prometheus/Loki/Tempo queries, PostgreSQL, Redis                     |
| Agent       | State transitions, tool budgets, malformed responses                 |
| End-to-end  | Fault injection → detection → triage → RCA                           |
| Evaluation  | Accuracy, latency, cost and reproducibility                          |

Use `pytest`, `pytest-asyncio`, `httpx`, and `testcontainers` where appropriate.

Every external model call should have a mocked test path so the core test suite does not require paid API access.

## Security requirements

- All agent tools are read-only.
- No shell execution exposed to the LLM.
- No deployment, rollback, or infrastructure mutation tools in the agent runtime.
- Secrets are excluded from logs, traces, and prompts.
- Telemetry query parameters are validated server-side.
- Tool responses are treated as untrusted data, not instructions.
- Each investigation has strict execution and cost budgets.
- Agent outputs are validated before persistence.
- Fault injection is restricted to the isolated simulator.
- Incident reports cannot trigger operational changes.

The application may generate remediation recommendations, but recommendations are never automatically executed.

## Failure handling

| Failure                       | Required behavior                           |
| ----------------------------- | ------------------------------------------- |
| Laya unavailable (localhost)  | Use configured fallback provider            |
| LLM timeout                   | Retry within budget, then mark inconclusive |
| Missing telemetry             | Report missing evidence                     |
| Invalid model output          | Validate, retry once, then fail explicitly  |
| Redis interruption            | Preserve persisted incident state           |
| Duplicate anomaly             | Merge into existing incident                |
| Contradictory evidence        | Retain alternative hypotheses               |
| Investigation budget exceeded | Stop and produce partial findings           |

# 14. Required deliverables

By the end of the project, the repository should contain these artifacts.

Project completion checklist

0/14

Working Docker Compose environment

Instrumented microservices simulator

Six reproducible incident scenarios

Telemetry ingestion and feature pipeline

Hybrid anomaly detection module

Laya decision provider (localhost) with fallback

LangGraph investigation workflow

Evidence-backed RCA generation

Recorded incident replay engine

Benchmark and ablation results

Functional investigation dashboard

Architecture and API documentation

Automated tests and CI workflow

Demo video and portfolio README

# 15. Claude Code execution strategy

Use Claude Code as the development environment, not as the production investigation engine.

The first task should establish the architecture and contracts before generating application code.

# Sentinel — Claude Code Implementation Instructions

You are the Principal AI Engineer and Software Architect responsible for implementing Sentinel.

## Project

Sentinel is a hybrid AIOps research prototype combining:

- Traditional ML for anomaly detection.
- Laya (running on localhost) as a System One fast decision provider.
- OmniRoute local gateway with `auto/deepseek` as the default System Two reasoning route.
- LangGraph for autonomous incident investigation.
- OpenTelemetry, Prometheus, Loki, and Tempo for observability.
- Docker Compose for reproducible simulated infrastructure.

## Non-negotiable constraints

1. The system must be runnable locally.
2. Agents have read-only investigation permissions.
3. No autonomous remediation.
4. All RCA claims require evidence references.
5. All agent loops have execution limits.
6. Model providers must be replaceable.
7. Experiments must be reproducible.
8. Benchmark ground truth must never enter agent context.
9. All external API contracts must be verified before implementation.
10. Never claim that a feature works without executing relevant tests.

## Implementation approach

Use incremental vertical slices.

First implement a single incident scenario through the entire pipeline:

Simulator → Telemetry → Detection → Triage → Investigation → RCA.

Then expand functionality.

## Initial tasks

1. Create `docs/architecture.md`.
2. Create `docs/api-spec.md`.
3. Create `docs/agent-design.md`.
4. Create `docs/experiments.md`.
5. Create `docs/security.md`.
6. Define Pydantic contracts.
7. Define the database schema.
8. Define API contracts.
9. Run Spike 0: verify the Laya localhost runtime and write `docs/laya-contract.md` (see 5.4), and verify the OmniRoute contract (see the spike list in section 12).
10. Produce a dependency-aware implementation plan.

Use `CLAUDE.md` to document architectural rules, repository conventions, commands, and testing requirements.

## Engineering standards

- Python 3.12 with type annotations.
- Pydantic v2.
- FastAPI.
- SQLAlchemy and Alembic.
- Pytest.
- Ruff.
- Structured JSON logging.
- Typed agent state.
- Explicit error handling.
- Small, independently testable modules.

## First milestone

Implement a reproducible database connection pool exhaustion incident.

The system must:

1. Generate a controlled fault.
2. Capture metrics, logs, and traces.
3. Detect the anomaly.
4. Triage the incident.
5. Investigate using read-only tools.
6. Generate an evidence-backed RCA.
7. Compare the result against hidden ground truth.

Do not start implementing unrelated features before this vertical slice works.

Before coding, inspect the repository, document architectural decisions, identify unknowns, and produce a concrete implementation plan with testable acceptance criteria.

## Final engineering recommendation

I would position Sentinel as an experimental hybrid decision-and-reasoning architecture for AIOps, with three contributions:

1. Hybrid intelligence: Traditional ML, System One decisions, and System Two reasoning working together.
2. Evidence-grounded autonomous investigation: Agents that investigate incidents through structured tools and verifiable evidence.
3. Reproducible evaluation: Controlled incident scenarios, model comparisons, and measurable system behavior.

The most important experiment is whether adding Laya (running on localhost) improves the overall system compared with a simpler ML + LLM architecture. That's a hypothesis worth testing, not an assumption.

For the first development milestone, build just one complete scenario—database connection pool exhaustion—from fault injection through an evidence-backed RCA report. Once that works reliably, the rest of Sentinel becomes a sequence of controlled extensions rather than a collection of disconnected AI components.