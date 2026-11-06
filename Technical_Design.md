# SENTINEL — Technical Design Specification v1.0

Hybrid AI System for Autonomous Incident Detection, Triage & Root Cause Investigation

Implementation-ready design

14-day MVP

Research prototype

This specification defines Sentinel as a hybrid AIOps platform combining conventional anomaly detection, Laya-based System One decisions (running on localhost), and LLM-based System Two investigation (DeepSeek via proxy server).

The objective is to produce a technically credible, reproducible AI engineering portfolio project—not a generic monitoring dashboard with an LLM attached.

## 1. Final architectural decisions

| Dimension                     | Decision                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| Infrastructure                | Docker Compose simulated microservices                       |
| Autonomy                      | Autonomous detection and investigation; recommendations only |
| Research focus                | Hybrid ML + System One + System Two                          |
| Fast decision model           | Laya (running on localhost, behind a provider abstraction)   |
| Reasoning model               | DeepSeek via proxy server                                    |
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

Intelligence plane: Processes signals, calls Laya (localhost) or alternative decision providers, and performs investigation with DeepSeek via proxy server.

Application plane: Manages incident state, API requests, users, audit trails, dashboards, and experiment results.

Separating these planes prevents the agent workflow from becoming tightly coupled to telemetry ingestion.

# 3. Simulated infrastructure specification

The simulated environment should resemble a small e-commerce backend, because it naturally produces distributed incidents involving application services, databases, dependencies, and deployments.

## 3.1 Services

| Service            | Responsibility                          | Dependencies                       |
| ------------------ | --------------------------------------- | ---------------------------------- |
| API Gateway        | Routes incoming requests                | Auth, Orders                       |
| Auth Service       | Authentication and token validation     | PostgreSQL                         |
| Order Service      | Order creation and retrieval            | PostgreSQL, Payment                |
| Payment Service    | Simulated payment processing            | Redis, External Provider Simulator |
| Provider Simulator | Configurable downstream API             | None                               |
| PostgreSQL         | Application persistence                 | None                               |
| Redis              | Cache and simulated queue               | None                               |
| Traffic Generator  | Produces reproducible workloads         | API Gateway                        |
| Fault Controller   | Activates predefined incident scenarios | Demo services                      |

All services should expose `/health`, `/ready`, and `/metrics` where appropriate.

Use OpenTelemetry propagation for requests crossing service boundaries.

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

| ID  | Fault                               | Expected observable symptoms                    |
| --- | ----------------------------------- | ----------------------------------------------- |
| F01 | Database connection pool exhaustion | Higher query wait time, API latency, timeouts   |
| F02 | Downstream provider slowdown        | Payment latency, dependency timeouts            |
| F03 | Application regression              | Increased 500 errors after simulated deployment |
| F04 | CPU saturation                      | Higher CPU, increased response latency          |
| F05 | Memory pressure                     | Memory growth, degraded service responsiveness  |
| F06 | Cache unavailability                | Cache errors, database load increase            |

Fault injection must operate only on the isolated demo environment. No real production endpoints or external services should be affected.

For each scenario, store a private ground-truth manifest containing the actual fault, onset time, affected services, expected causal chain, and recovery time.

Do not expose that manifest to the investigation agents. The benchmark runner uses it only after an investigation is complete.

# 4. Data processing and anomaly detection

## 4.1 Processing windows

Use two window sizes:

- Short window: 1 minute, for rapid changes.
- Context window: 5 minutes, for baseline comparison and feature aggregation.

Prometheus scrapes every 15 seconds. The signal processor evaluates every 30 seconds.

Each evaluation produces a normalized feature vector.

## 4.2 Feature schema

```
from pydantic import BaseModel, Fieldfrom datetime import datetimeclass ServiceSignalWindow(BaseModel):    service: str    window_start: datetime    window_end: datetime    request_rate: float = Field(ge=0)    error_rate: float = Field(ge=0, le=1)    latency_p50_ms: float = Field(ge=0)    latency_p95_ms: float = Field(ge=0)    latency_p99_ms: float = Field(ge=0)    cpu_utilization: float = Field(ge=0, le=1)    memory_utilization: float = Field(ge=0, le=1)    db_pool_utilization: float | None = None    dependency_timeout_rate: float | None = None    recent_deployment: bool = False    log_error_count: int = 0    trace_error_count: int = 0
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

For the MVP, normalize each detector output to a comparable 0–1 scale using a held-out calibration dataset.

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

# 5. System One — Laya integration specification

The System One layer should be a replaceable decision subsystem, not a hard dependency embedded throughout the application.

## 5.1 Decision responsibilities

Laya (running locally on localhost) receives an incident candidate and makes structured triage decisions.

| Decision               | Type             | Output                                                        |
| ---------------------- | ---------------- | ------------------------------------------------------------- |
| Incident category      | Choice           | `LATENCY`, `ERROR_SPIKE`, `RESOURCE`, `DEPENDENCY`, `UNKNOWN` |
| Severity               | Choice           | `SEV1`, `SEV2`, `SEV3`, `SEV4`                                |
| Investigation priority | Score            | Normalized score                                              |
| Investigation required | Boolean decision | Yes/No with supporting score where available                  |
| Deployment relevance   | Boolean decision | Whether deployment deserves investigation                     |

Treat these as predictions, not authoritative facts.

For severity, define operational rules first:

| Severity | Meaning                                     |
| -------- | ------------------------------------------- |
| SEV1     | Critical simulated service unavailability   |
| SEV2     | Major degradation affecting core operations |
| SEV3     | Partial degradation or elevated errors      |
| SEV4     | Low-impact anomaly requiring observation    |

The policy engine must be able to override a model decision when deterministic evidence indicates a critical incident.

## 5.2 Provider abstraction

Use a typed interface:

```
from typing import Protocolfrom pydantic import BaseModel, Fieldclass TriageDecision(BaseModel):    category: str    severity: str    priority_score: float = Field(ge=0, le=1)    investigate: bool    deployment_relevant: bool | None = None    provider: str    model_version: str | None = None    raw_scores: dict[str, float] = {}class DecisionProvider(Protocol):    async def triage(        self,        features: ServiceSignalWindow    ) -> TriageDecision:        ...
```

Implement four providers:

```
decision_providers/
├── base.py
├── laya.py
├── sklearn_classifier.py
├── llm_classifier.py
└── rule_based.py
```

All providers must return the same `TriageDecision` schema.

### Laya adapter implementation

The Laya adapter should:

1. Receive a validated feature vector.
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
def should_investigate(    decision: TriageDecision,    anomaly_score: float,    critical_rule_triggered: bool,) -> bool:    if critical_rule_triggered:        return True    if decision.severity in {"SEV1", "SEV2"}:        return True    if anomaly_score >= 0.85:        return True    return (        decision.investigate        and decision.priority_score >= 0.65    )
```

Thresholds are provisional and should be tuned on validation scenarios.

The policy should additionally enforce deduplication, cooldowns, and maximum concurrent investigations.

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

The supervisor should be application code. Individual reasoning agents use DeepSeek (connected via the proxy server) for structured reasoning, hypothesis formation, evidence verification, and RCA synthesis.

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

```
from typing import TypedDictclass InvestigationState(TypedDict):    incident_id: str    service: str    severity: str    context: dict    hypotheses: list[dict]    evidence: list[dict]    iteration: int    tool_calls: int    selected_hypothesis_id: str | None    rca_report: dict | None    status: str    errors: list[str]
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

```
query_metrics(    service: str,    metric: str,    start: datetime,    end: datetime)query_logs(    service: str,    start: datetime,    end: datetime,    level: str | None = None,    limit: int = 100)query_traces(    service: str,    start: datetime,    end: datetime,    errors_only: bool = False)get_deployment_events(    service: str,    start: datetime,    end: datetime)get_service_dependencies(    service: str)search_historical_incidents(    query: str,    limit: int = 5)
```

Every tool must enforce server-side service allowlists, time-range limits, query limits, and timeouts.

Do not let the LLM submit arbitrary PromQL, LogQL, SQL, or shell commands directly. Build queries from validated parameters in trusted application code.

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

```
from enum import Enumfrom uuid import UUIDfrom datetime import datetimefrom pydantic import BaseModelclass IncidentStatus(str, Enum):    OPEN = "open"    TRIAGED = "triaged"    INVESTIGATING = "investigating"    INVESTIGATED = "investigated"    INCONCLUSIVE = "inconclusive"    CLOSED = "closed"class Incident(BaseModel):    id: UUID    service_id: UUID    status: IncidentStatus    category: str    severity: str    anomaly_score: float    opened_at: datetime    updated_at: datetime    triage_decision_id: UUID | None = None    investigation_id: UUID | None = None
```

## 7.3 Evidence schema

Every factual claim in the RCA must point to evidence.

```
class Evidence(BaseModel):    id: UUID    investigation_id: UUID    source_type: str    service: str    observation: str    timestamp: datetime    source_reference: str    query_parameters: dict    supports: list[str]    contradicts: list[str]
```

`source_reference` should be an immutable reference to the retrieved result or stored snapshot, not an invented citation.

For reproducibility, store the query parameters, retrieval timestamp, result checksum, and a bounded sanitized snapshot.

## 7.4 RCA report schema

```
class RCAReport(BaseModel):    incident_id: UUID    summary: str    affected_services: list[str]    probable_root_cause: str | None    confidence_level: str    supporting_evidence_ids: list[UUID]    contradicting_evidence_ids: list[UUID]    timeline: list[dict]    alternative_hypotheses: list[dict]    recommendations: list[str]    unresolved_questions: list[str]    investigation_complete: bool
```

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

Also generate at least 12 normal or non-incident runs, including workload spikes that should not automatically become incidents.

Each run should vary:

- Fault onset.
- Fault duration.
- Request load.
- Fault intensity.
- Background noise.
- Affected service.

### Dataset split

To avoid leakage, separate scenarios by configuration and random seed, not by individual telemetry windows.

| Split       | Positive runs | Normal runs | Purpose                      |
| ----------- | ------------- | ----------- | ---------------------------- |
| Development | 12            | 4           | Implementation and debugging |
| Validation  | 12            | 4           | Threshold tuning             |
| Test        | 12            | 4           | Final evaluation             |

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
| T2      | Laya (localhost)                                | System One fast decision model                  |
| T3      | DeepSeek structured output (via proxy)          | Cloud LLM classification baseline               |

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

Keep the model (DeepSeek via proxy), telemetry snapshot, available evidence, and total execution budget comparable.

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
| H0      | Rules     | Rules            | Single LLM (DeepSeek via proxy)  |
| H1      | Hybrid ML | Rules            | Multi-agent (DeepSeek via proxy) |
| H2      | Hybrid ML | Laya (localhost) | Multi-agent (DeepSeek via proxy) |
| H3      | Hybrid ML | Cloud LLM        | Multi-agent (DeepSeek via proxy) |

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
| Evidence traceability            | 100% of factual RCA claims linked to evidence |
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
| Contract    | Laya adapter, DeepSeek proxy structured outputs, telemetry tools     |
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
- DeepSeek (via proxy server) as a System Two reasoning model.
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
2. Create `docs/technical-spec.md`.
3. Create `docs/agent-design.md`.
4. Create `docs/experiments.md`.
5. Create `docs/security.md`.
6. Define Pydantic contracts.
7. Define the database schema.
8. Define API contracts.
9. Verify Laya local runtime / localhost endpoint availability and integration requirements.
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