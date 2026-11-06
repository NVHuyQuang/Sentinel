# Laya Contract (Spike 0, blocking — Spec §5.4)

Status: **PASS** (both devices). T2/H2 are **unblocked** on contract grounds; quality remains an empirical question per Spec §9.4.

Every number below is a real measured call from `scripts/spike_laya.py`, not a spec guess.
Reproduce with:

```powershell
python scripts/spike_laya.py --device cuda   # production default
python scripts/spike_laya.py --device cpu    # comparison
```

Raw per-sample evidence lives in `docs/spike_laya_samples-<device>.jsonl` (git-ignored,
see `.gitignore` `docs/spike_laya_*`). This file is the committed summary.

## 1. How Laya is invoked

- **In-process library**, not a localhost HTTP endpoint.
  `Router(device="cuda"|"cpu")` + `router.predict(text, questions)` (same call as
  `run_laya_cpu.py`; `run_laya_gpu.py` was deleted as a pure device-copy).
- **Version:** `laya.__version__` → **0.4.26** (printed by the spike script itself).
- **Python:** 3.10.11 on this host. Spec targets 3.12 — see CLAUDE.md note; do not
  silently downgrade the spec.
- **Model resolved per call:** `"multilingual"`, repo `convaiinnovations/laya/multilingual`,
  model id `laya-rl-agent` (in every `raw_response.routing`). No GPU/CUDA version pin
  recorded — host has 1 CUDA device (`torch.cuda.is_available() == True`).
- **Startup model:** `router.predict("warmup", QUESTIONS)` once after construction
  (identical to `run_laya_cpu.py`). The first *untimed* warmup call is what triggers
  lazy checkpoint load; without it the first timed call includes load (~60–75 s).
- **Docker Compose:** not started in this spike. The adapter contract below is
  device-parameterised so Compose can start with either device once specified.

## 2. Exact request / response contract

Questions dict — one `predict` answers all five decisions in a single forward pass:

```python
QUESTIONS = {
  "category":    {"type": "choice", "instructions": "What best describes the incident?",
                  "criteria": {"LATENCY": "...", "ERROR_SPIKE": "...", "RESOURCE": "...",
                               "DEPENDENCY": "...", "UNKNOWN": "..."}},
  "severity":    {"type": "choice", "instructions": "How severe?",
                  "criteria": {"SEV1": "...", "SEV2": "...", "SEV3": "...", "SEV4": "..."}},
  "priority_score": {"type": "score", "instructions": "Score investigation priority from 0 to 9 inclusive.",
                     "criteria": ["0","1","2","3","4","5","6","7","8","9"]},   # list, not dict
  "investigate":     {"type": "noul", "instructions": "Is investigation required?"},  # no criteria key
  "deployment_relevant": {"type": "noul", "instructions": "Is deployment relevant?"},
}
```

- `type` ∈ {`choice`, `score`, `noul`} (`laya.QTYPES`). Choice criteria are
  `{label: description}`; score criteria must be a **list** of up to 10 levels
  (`laya.structured.MAX_SCORE_LEVELS == 10` — a 0–100 scale is rejected);
  `noul` takes **no** criteria.
- **Input:** plain incident text (this spike renders `IncidentFeatures` fields as
  key/value text, e.g. `primary_service order-service affected order-service/auth-service
  error_rate 0.082 p95_latency_ms 3480 … db_pool 0.92 …`). Observed `usage`:
  `input_tokens` ≈ 434–439 (includes per-question overhead),
  `state_tokens` ≈ 38–39, `truncated: false`. Agent default token budget is
  `max_len == 512` (`self.cfg.get("max_len", 512)` in laya source); longer states are
  truncated (`state_tokens_dropped`). The production adapter must fit/summarise
  `IncidentFeatures` into budget — long `windows` lists will not fit.
- **Response:** `result["answers"]` keyed by question name.
  - `choice`: `{type, choice, probabilities (sums ≈ 1.0), confidence, answer_confidence, action}`.
  - `score`: `{type, score (probability-weighted expected level), legend, probabilities
    (per level "0".."9"), confidence, answer_confidence, action}`.
  - `noul`: `{type, noul (float float), confidence, answer_confidence, action}` where
    `noul` = P(True). Parser rule verified in `laya.structured._project`:
    `bool(float(noul) >= 0.5)`. Spike parser uses the same rule.
  - Sentinel `priority_score` normalisation: `float(raw["score"]) / 9.0` (expected
    level over max level). Argmax level is **not** used.
  - **Uncalibrated:** the checkpoint emits
    `RuntimeWarning: … invalid temperatures … Treat confidence from the affected entries
    as uncalibrated`. `probabilities` exist and sum to ~1.0, but the spike does **not**
    treat them as calibrated probabilities — Spec §5.1/§7.4 forbids presenting
    uncalibrated outputs as statistics.

### What Laya cannot give (do not invent)

- No REST endpoint, no per-question latency split, no logits, no temperature control on
  `Router.predict`. Latency is measured client-side (`time.perf_counter()` around predict).
- `score` resolution is limited to 10 levels; finer granularity needs a different schema
  (e.g. a `choice` over bins) or a different provider.
- Out-of-vocab answers cannot be produced by a healthy checkpoint — labels are constrained
  to `criteria`. The parser still guards against them (stale/mocked checkpoints).

## 3. Measured results (50 IncidentFeatures-shaped samples, seed 0)

Sample generator (`make_samples(50)` in the spike script, deterministic): 5 kinds
(latency / error_spike / resource / dependency / low) × 4 core services, with realistic
metric ranges per kind. **6/50 samples carry a real `db_pool` value** (0.85–1.0 on the
latency branch) so F01 (DB pool exhaustion) is exercised — earlier generator versions
had `pool=None` everywhere; rerun after the patch shows all six parse (e.g. `s05
db_pool 0.92` → `LATENCY / SEV4`).

| Device | Parseable | Latency P50 | Latency P95 | Deterministic (same input ×2) |
|---|---|---|---|---|
| `cuda` | **50/50** | **50.2 ms** | **52.9 ms** | **true** (full parsed-dict equality) |
| `cpu`  | **50/50** | **703.0 ms** | **1 708.9 ms**¹ | **true** |

¹ CPU tail: the first 8 timed calls after cold start measure 1.47–1.83 s; steady state
from call 9 onward is ~0.63–0.75 s. GPU shows no such tail (min 48.7 ms, max 60.9 ms).
**Production timeout must exceed P95 per device** (CPU cold ≥ ~1.8 s; GPU ≥ ~70 ms) or
the policy layer will fall back spuriously on its first calls. The spike’s timeout test
deliberately uses **10 ms** to force the failure path — it is a test budget, not a
production recommendation.

Spot evidence (CUDA, `parsed` after spike-local stub parser — real `base.py` in Day 1):

```json
// s05 (db_pool 0.92, F01-like)
{"category": "LATENCY", "severity": "SEV4", "priority_score": 0.5015,
 "investigate": false, "deployment_relevant": false}
```

Observations (evidence, not quality claims — T2/H2 evaluation decides):

- `priority_score` clusters at **0.497–0.509** (expected level ≈ 4.5/9, probabilities
  near-uniform ≈ 0.1/level) across all 50 samples on both devices. The Laya score as
  prompted here has **almost no discriminative power**; the policy threshold
  `priority_score >= 0.65` (Spec §5.3) would never fire from this signal alone.
- `category`/`severity` do vary (all of `ERROR_SPIKE` / `LATENCY` / `RESOURCE` /
  `DEPENDENCY` / `UNKNOWN`, `SEV2`–`SEV4`), but Laya **understates severity**: e.g.
  `s05` with `db_pool 0.92` + `p95 3480 ms` + `critical_rule_triggered true` is
  classified `SEV4`. The deterministic policy override (SEV1/SEV2 never lowered)
  must stay on top of the adapter.

## 4. Failure modes → `ProviderError` mapping

Spike-local `ProviderError` (stub; real class in Day-1 `base.py`) is raised by the parser
and by the timeout guard. Verified results (both devices):

| Failure | How triggered in spike | Maps to |
|---|---|---|
| **Timeout** | `ThreadPoolExecutor.future.result(timeout=0.01)` around `predict` | `ProviderError("timeout")` — wired (GPU 50 ms > 10 ms). Client-side enforcement; Router `hooks_timeout` exists but was not exercised. |
| **Malformed output** | `parse_result` given a truncated payload (`{answers: {category: …}}`) | `ProviderError("malformed: missing decision key 'severity'")` — parser-level, not model-level. |
| **Out-of-vocab choice** | `parse_result` given `choice="NOT_A_CHOICE" / "SEV9"` with otherwise valid payload | `ProviderError("out-of-vocab choice: …")` — parser-level guard. Real checkpoint cannot produce this by construction. |

## 5. Fallback — proven against a stub, real kill-test deferred

What the spike proves: `triage_with_fallback()` catches `ProviderError` from the parser
(and would catch it from the timeout guard) and returns a `rules` decision with
`provider == "rules"` and **no manual step** (`fallback_ok: true` on both devices).

What it does **not** prove: a real process kill. Laya here is in-process, so the test
injects `_DeadRouter.predict` raising `ProviderError("simulated Laya process death
(in-process)")`. There is no `rules.py` yet — `rules_fallback()` in the spike script is
an explicit stub (deny-if-error_rate≥0.2: `SEV2/ERROR_SPIKE` else `SEV3/UNKNOWN`).

**Day-1 obligation:** replace the stub with the real `rules.py`, keep the same
try/`except ProviderError → rules` seam, and add a real kill-test (start the decision
engine, kill the Laya provider, assert the next triage returns `provider == "rules"`).
Until then this contract claims stub-level fallback only.

## 6. Verdict vs Spec §5.4 pass criteria

- [x] 50/50 parseable for all five decisions (CUDA **and** CPU)
- [x] Latency recorded (P50/P95 per device, plus cold-start tail note)
- [x] Fallback proven against stub `ProviderError` → `rules` (no manual action)
  — real process-kill test deferred to Day 1, stated explicitly above

**PASS.** Adapter may be implemented in Day 1 against §2 exactly as measured. Any few-shot
or prompt variant must be applied identically to the T3 LLM baseline (Spec §5.1).
