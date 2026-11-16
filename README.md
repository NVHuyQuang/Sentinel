# Sentinel: Autonomous Incident Detection, Triage & Root Cause Investigation

Sentinel is a hybrid AIOps research prototype combining statistical and machine learning anomaly detection, fast local Small Language Model (SLM) triage (**System One** via Laya), and multi-agent Large Language Model (LLM) root cause investigation (**System Two** via OmniRoute gateway and LangGraph).

> **Core Architectural Principle:**  
> *Detection identifies unusual behavior. Decision models determine triage priority. Reasoning agents investigate possible causes. Deterministic policies enforce operational boundaries.*

---

## Architecture Overview

```
                      +------------------------------------------+
                      |   Simulated Microservices Environment    |
                      |  API Gateway · Auth · Order · Payment    |
                      +------------------------------------------+
                                           |
                                [Telemetry Scrapes]
                                (Metrics, Logs, Traces)
                                           v
+-----------------------------------------------------------------------------------+
| 1. Signal Processing Layer                                                        |
|    - Sliding window aggregation (1m / 5m / 15m)                                   |
|    - 3-Detector Ensemble: Robust Baseline (MAD) + Isolation Forest + CUSUM         |
|    - Candidate hysteresis gate + Topology-aware Incident Grouper                 |
+-----------------------------------------------------------------------------------+
                                           |
                                [IncidentFeatures]
                                           v
+-----------------------------------------------------------------------------------+
| 2. System One Triage Engine (Fast Decision Plane)                                 |
|    - Primary: Laya SLM (localhost in-process, CPU/CUDA)                           |
|    - Fallbacks: Scikit-learn baseline / Deterministic Rules Engine                |
|    - Strict Policy Guardrails: Severity floors & automated kill-switch            |
+-----------------------------------------------------------------------------------+
                                           |
                              [SEV1/SEV2 or High Priority]
                                           v
+-----------------------------------------------------------------------------------+
| 3. System Two Investigation Engine (Deep Reasoning Plane)                        |
|    - Orchestration: LangGraph Multi-Agent Workflow                                |
|    - Agent Roles: Context Collector -> Hypothesis Generator                       |
|                   -> Evidence Investigator -> Gatekeeper -> RCA Synthesizer       |
|    - Budgeted Read-Only Toolbox (Strict quotas & monotonic deadline)              |
|    - Evidence Traceability Validator (Every factual claim linked to telemetry)    |
+-----------------------------------------------------------------------------------+
                                           |
                                  [Verified RCA Report]
                                (Root cause + Evidence links)
```

---

## Key Subsystems

### 1. Signal Processing & Anomaly Detection
Sentinel continuously extracts telemetry across all microservices using fixed time-slice snapshots (`SnapshotBackend`):
- **Robust Baseline Detector:** Computes median and Median Absolute Deviation (MAD) across baseline traffic to flag statistical variance without sensitivity to outliers.
- **Isolation Forest Detector:** Detects multivariate metric anomalies across latency percentiles (P50, P95, P99), error rates, and resource utilization.
- **CUSUM (Cumulative Sum) Detector:** Detects step changes and persistent drift with capped accumulation.
- **Calibrated Ensemble & Hysteresis Gate:** Aggregates individual detector scores with weights and requires persistent anomalous signal ticks before promoting to an incident candidate.
- **Incident Grouper:** Uses service dependency topology to cluster correlated alerts into unified `IncidentFeatures`.

### 2. System One Triage Engine
Fast classification into incident category (`RESOURCE`, `NETWORK`, `DEPLOYMENT`, `UNKNOWN`), severity (`SEV1` through `SEV4`), and priority score:
- **Tier 0 — Deterministic Rules Provider:** Strict heuristic baseline ensuring system continuity.
- **Tier 1 — Scikit-Learn Provider:** Classical machine learning classification trained on historical features.
- **Tier 2 — Laya SLM Adapter:** Local in-process SLM router optimized for sub-100ms latency on GPU and low-overhead CPU execution.
- **Tier 3 — LLM Provider:** Routed model provider via the OmniRoute gateway.
- **Policy Layer & Guardrails:**
  - `enforce_severity_floor`: Deterministic safety rules (e.g., error rate $\ge 20\%$ or database pool $\ge 85\%$ automatically forces $\ge$ `SEV2`).
  - Graceful fallback: If the primary SLM times out or errors, the policy router automatically falls back to deterministic rules.

### 3. System Two Root Cause Investigation
Deep reasoning orchestrated through a structured LangGraph state graph:
- **Agent Roles:**
  1. **Context Collector:** Initializes incident context, service topology, and symptom tags.
  2. **Hypothesis Generator:** Generates up to 3 candidate hypotheses (e.g., database pool exhaustion, thread starvation, network partition).
  3. **Evidence Investigator:** Executes targeted queries against the read-only telemetry backend.
  4. **Gatekeeper:** Evaluates whether gathered evidence is sufficient or budget has been reached.
  5. **RCA Synthesizer:** Produces a structured `RCAReport` containing verified claims, probable root cause, confidence, and contradicted hypotheses.
- **Budgeted Read-Only Toolbox:**
  - `query_metrics`, `query_logs`, `query_traces`, `get_deployment_events`, `get_service_dependencies`, `search_historical_incidents`.
  - Hard operational limits: Max 12 calls/round, max 30 calls total, strict monotonic timeout.
- **Evidence Traceability Validator:**
  - Enforces that every `fact` claim links to real telemetry evidence retrieved during investigation.
  - Reject reports containing ungrounded claims or invalid evidence IDs.

### 4. Evaluation & R0–R3 Ablation Framework
Evaluates root cause investigation performance across four rigorous experimental variants against hidden ground truth manifests (Spec §9.5, RQ3):
- **R0 (Zero Tools):** Single LLM prompt with detector incident summary only.
- **R1 (ReAct Baseline):** Single LLM with access to the budgeted read-only toolbox and RCA validator.
- **R2 (Multi-Agent Graph):** LangGraph multi-agent workflow (Context, Hypothesis, Investigator, Gatekeeper, RCA).
- **R3 (Multi-Agent + Memory):** R2 workflow augmented with historical incident search (`search_historical_incidents`).

---

## Core Safety & Research Rules

1. **Read-Only Agents (No Autonomous Remediation):**  
   Investigation agents have zero access to mutation tools. Recommendations are emitted in reports, never executed autonomously. Fault injection remains strictly quarantined in the simulator.
2. **Telemetry Sanitization:**  
   No test harness artifacts, fault manifests, scenario IDs (`F01`–`F06`), or ground-truth keywords (`fault`, `chaos`, `inject`) ever leak into agent context or tool outputs.
3. **Ground Truth Isolation:**  
   Ground truth manifests are read solely by the post-investigation benchmark scoring engine (`evaluation/score.py`, `evaluation/ablation.py`).
4. **Evidence Traceability:**  
   Every claim marked as `fact` in an RCA report must trace directly to verified telemetry records (`supporting_evidence_ids`).

---

## Repository Structure

```
Sentinel/
├── apps/
│   └── api/                # FastAPI application (health check, service catalog)
├── docs/                   # Benchmark reports, contracts (Laya, GLiDE, OmniRoute), specs
├── evaluation/             # Evaluation harness, F01 replay benchmark, R0-R3 ablation
│   ├── ablation.py         # R0-R3 ablation framework (Spec §9.5, RQ3)
│   ├── f01_runner.py       # Deterministic F01 replay execution
│   └── score.py            # Precision, recall, and root-cause accuracy scoring
├── services/
│   ├── decision_engine/    # System One triage providers (rules, sklearn, laya, llm) and policy
│   ├── investigation/      # System Two LangGraph multi-agent workflow, toolbox, RCA validator
│   ├── signal_processor/   # Robust, Isolation Forest, CUSUM detectors, ensemble, grouper
│   ├── telemetry/          # Snapshot backend for metrics, logs, and traces
│   └── pipeline.py         # End-to-end pipeline driver (detection -> triage -> investigation)
├── simulator/              # Microservice topology, synthetic fault controller, replay recorder
├── tests/
│   ├── integration/        # End-to-end replay, fallback kill-tests, real provider tests
│   └── unit/               # Detector, policy, contract, graph, and ablation unit tests
├── Technical_Design.md     # Full Sentinel v1.0 architecture specification
├── CLAUDE.md               # Developer guidelines, constraints, and contracts
├── pyproject.toml          # Project metadata and dependencies
└── docker-compose.yml      # Local PostgreSQL and Redis services
```

---

## Quickstart

### Prerequisites
- Python 3.12+ (`py -3.12` on Windows)
- Docker & Docker Compose (for PostgreSQL and Redis)

### 1. Environment Setup
Clone the repository and configure your environment:
```powershell
cp .env.example .env
```
Configure your `.env` with your OmniRoute gateway credentials:
```ini
OMNIROUTE_API_KEY=your-client-api-key
LLM_BASE_URL=http://localhost:20128/v1
LLM_MODEL=auto/deepseek
DATABASE_URL=postgresql://sentinel:sentinel@localhost:5432/sentinel
REDIS_URL=redis://localhost:6379/0
```

### 2. Infrastructure Services
Start PostgreSQL and Redis:
```powershell
docker compose up -d postgres redis
```

### 3. Run Test Suite
Run the full test suite (contracts, detectors, multi-agent graph, ablation):
```powershell
py -3.12 -m pytest
```
Run only unit tests:
```powershell
py -3.12 -m pytest tests/unit
```

### 4. Start the API
Launch the FastAPI development server:
```powershell
py -3.12 -m uvicorn apps.api.main:app --reload --port 8000
```
- Health Check: `http://localhost:8000/health`
- Service Allowlist: `http://localhost:8000/api/v1/services`

### 5. Run the R0–R3 RCA Ablation Benchmark
Execute the ablation study on the recorded F01 replay:
```powershell
py -3.12 -m evaluation.ablation --repeats 3 --deadline 600
```

---

## License

This project is developed as an AI engineering research prototype. All rights reserved.
