# Sentinel — CLAUDE.md

Hybrid AIOps research prototype: anomaly detection + Laya System One triage +
OmniRoute System Two investigation. Spec: `Technical_Design.md`.

## Architecture rules

- **Three control planes** (Spec §2): observability (read-only telemetry, no LLMs),
  intelligence (Laya localhost + OmniRoute gateway), application (state, API, audit).
  Keep them separate — the agent workflow never couples directly to telemetry ingestion.
- **Read-only agents, no autonomous remediation.** Agent tools are read-only
  (Spec §6.5); recommendations are never executed. Fault injection stays in the
  isolated simulator (Spec §3.3, §13).
- **Evidence traceability.** Every `fact` claim in the RCA report must link to
  existing evidence (`RCAReport.claims`, Spec §7.4). Uncalibrated LLM scores are
  never presented as statistics.
- **Ground truth never enters agent context.** `scenario_ground_truth` is visible
  only to the benchmark evaluator after an investigation completes (Spec §3.3, §7.1).
  Fault manifests, scenario IDs, and fault labels must not appear in telemetry,
  prompts, or tool responses.
- **Provider seam only.** Decision providers implement `DecisionProvider`
  (`services/decision_engine/providers/base.py`); policy sits on top of every
  provider and a SEV1/SEV2 rule firing always raises severity, never lowers it
  (Spec §5.1, `enforce_severity_floor`).

## Provider contracts

- **System One = Laya** (in-process `Router`, CUDA default). Timeout must exceed P95:
  **GPU ≥ ~70 ms, CPU ≥ ~1.8 s** (cold CPU tail), else the policy layer falls back
  spuriously. See `docs/laya-contract.md` §3.
- **System Two = OmniRoute** (`auto/deepseek` via `http://localhost:20128/v1`).
  **Auth requires a real `OMNIROUTE_API_KEY`** — `local-omniroute` is rejected with
  401 (docs/omniroute-contract.md §1). Copy `.env.example`; never commit `.env`.
  **Enum-guarded prompt is mandatory** (omniroute-contract §2): without the literal
  category/severity enums and the 0–1 priority range in the system message, output
  is unparseable (0/20). Cold latency 39–72 s → **timeout ≥ 75 s**; the semantic
  cache makes latency **bimodal** (cold ~60 s vs cached ~2.1 s, same response id) —
  never average it away. `auto/deepseek` returned **no `tool_calls` array** (§2) —
  don't promise structured tool use on that route. Cost is **not reported** by the
  gateway; do not fabricate per-token prices.
- **GLiDE rejected — keep Laya** (`docs/glide-contract.md`): 98% SEV1, collapsed
  categories, `priority_score` systematically below Laya's already-dead cluster.

## Known caveats

- **Python 3.10 vs 3.12.** Spec targets 3.12. This host's default `python` is
  3.10.11, but 3.12 is installed (`py -3.12`) and has the deps — `pyproject.toml`
  declares `requires-python = ">=3.12"`. **Run everything with `py -3.12 …`**.
  Code stays 3.10-compatible (`X | None` annotations) but do not claim the spec
  was silently downgraded.
- **Laya `priority_score` is dead at ~0.5** (0.497–0.509 over 50 samples,
  near-uniform probabilities) — the §5.3 `priority_score >= 0.65` branch never fires
  from that signal alone (laya-contract §3). Treat scores as predictions, not facts.
- Laya understates severity (s05 `db_pool 0.92` + `critical_rule_triggered` → SEV4)
  and its probabilities are uncalibrated — both recorded in the contracts, don't re-derive them.

## Repo conventions

- Pydantic v2, `X | None` annotations, no `Any`-typed dicts at API boundaries.
- Ruff: line length 100, target `py312` (`[tool.ruff]` in `pyproject.toml`).
- Contracts live in `services/decision_engine/providers/base.py` (Spec §5.2 models
  verbatim + §7.2–§7.4). No Laya/OmniRoute adapter code yet — Day-2 files
  (`laya.py`, `llm.py`, `rules.py`, `sklearn.py`) plug in behind `DecisionProvider`.
- Telemetry sanitization (Spec §3.3): no `fault`/`inject`/`chaos`/`scenario`/
  `F01`–`F06`/`ground_truth` terms in anything the agents can read.
- Never print or log secrets; `OMNIROUTE_API_KEY` comes from the environment only.

## Commands

```powershell
py -3.12 -m pytest                      # full suite (unit + integration)
py -3.12 -m pytest tests/unit           # contracts + policy only
py -3.12 -m uvicorn apps.api.main:app   # API on http://localhost:8000 (/health, /api/v1/services)
docker compose up -d postgres redis     # state + event bus only (no Prom/Loki/Tempo yet)
```
