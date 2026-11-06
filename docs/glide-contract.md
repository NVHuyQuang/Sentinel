# GLiDE Contract (Spike 0 re-run, GLiDE candidate — Spec §5.4)

Status: **PASS** on contract grounds (50/50 parseable, latency recorded, stub fallback) — **replacement of Laya is NOT justified on this evidence.** T2/H2 evaluation decides; this spike only clears the contract gate.
`docs/laya-contract.md` (Laya PASS) is preserved unchanged; this file proposes, it does not replace, until the experiment decides.

Every number below is a real measured call from `scripts/spike_glide.py` on the exact same 50 inputs as Spike 0 (`make_samples(50)`, seed 0, `request_text` strings verified identical to the Laya run 50/50), not a spec guess. Reproduce with (CUDA primary; CPU spot-checked only, see below):

```powershell
python scripts/spike_glide.py --device cuda   # production default
```

Raw per-sample evidence lives in `docs/spike_glide_samples-cuda.jsonl` (git-ignored,
see `.gitignore` `docs/spike_glide_*`). This file is the committed summary.
Raw summary (same run): `docs/spike_glide_summary-cuda.json` (also git-ignored).

**Scope note:** this contract is **GPU-only** per task instruction (the full 50-row CPU rerun was deferred — see §3 CPU spot note). The PASS verdict below is a GPU-contract PASS.

## 1. How GLiDE is invoked

- **In-process library**, not a localhost HTTP endpoint.
  `model = Glide.from_pretrained("GLiDE-0.4B", device="cuda")` + `model.decide(state, questions)`
  (verified against `GLiDE-0.4B/glide/model.py::Glide.decide` and `README.md`; `decide` raises
  `glide.RequestError` for unservable requests, `decide_batch` returns it per item instead).
- **Version:** `glide.__version__` → **1.2.0** (printed by the spike script itself).
- **Model:** local directory `GLiDE-0.4B/` (fastino/GLiDE-0.4B, `model-00001.safetensors` ~789 MB,
  downloaded via `hf download fastino/GLiDE-0.4B --local-dir GLiDE-0.4B`; **never committed** —
  `.gitignore` `GLiDE-0.4B/` verified with `git check-ignore`).
- **Model resolved per call:** `raw_response.model == "glide-0.4b"` in every `raw_response`
  (`decision_config.json: "model_name": "glide-0.4b"`). Host has 1 CUDA device:
  `torch.cuda.is_available() == True`, `torch 2.10.0+cu128`, `NVIDIA GeForce RTX 3050 Ti Laptop GPU`.
- **Python:** 3.10.11 on this host (same as Spike 0).
- **VRAM:** **not measured** in `scripts/spike_glide.py` on this host. Upstream README claims
  0.79 GB after loading / 1.77 GB peak on an RTX PRO 6000 — treated as unverified on this host,
  not as evidence. Weights are bf16, 0.81 GB on disk (`README.md` table).
- **Context:** `decision_config.json: max_len 7999, trunc_head 1024` (long prompts keep first 1024
  tokens + end; per-question token budget 7,999). Our states are ~90 tokens — no truncation
  (`usage.input_tokens` 897–962 per call, see §2).
- **Startup model:** `model.decide("warmup", QUESTIONS)` once after loading (same warmup pattern as
  Spike 0). First timed call (s00) measured 120 ms vs steady-state min 86 ms — no Laya-style
  multi-second tail on GPU.
- **CPU note (spot, not part of the GPU contract):** 3 fresh CPU samples measured
  **11.5–12.1 s/sample** (`--n_samples 3`, parseable 3/3, deterministic 2× true). A full 50-row CPU
  run (≈10 min at that rate) was deferred by instruction; do not read a CPU P50/P95 from this spot.

## 2. Exact request / response contract

- **Input (`state` mapping — declared methodology difference):** GLiDE calls `decide(state, questions)`
  where `state` is the **same rendered string** as Laya’s `text`:
  `primary_service <svc> affected <a/b> error_rate <e> p95_latency_ms <p> baseline_p95_ms <b>
  cpu_utilization <c> memory_utilization <m> db_pool <v|n/a> anomaly_score <a>
  critical_rule_triggered <true|false> recent_deployment <true|false> symptom_tags <t>`.
  README allows string **or** any JSON value as `state` (JSON values are serialized by the server
  path); this spike passes the string form, byte-identical to Laya’s `predict(text, …)`.
  `request_text` is stored per row in `docs/spike_glide_samples-cuda.jsonl` so an auditor can diff
  against `docs/spike_laya_samples-cuda.jsonl` (verified: 50/50 identical `request_text`).
- **Questions dict — byte-identical to `docs/laya-contract.md` §2** (verified by diffing the
  `QUESTIONS` literals in `scripts/spike_glide.py` vs `scripts/spike_laya.py`; no GLiDE-specific
  wording added — Spec §5.1 satisfied, run is comparable):

```python
QUESTIONS = {
  "category":    {"type": "choice", "instructions": "What best describes the incident?",
                  "criteria": {"LATENCY": "high latency timeouts slowness degraded performance",
                               "ERROR_SPIKE": "5xx errors failure rate high", "RESOURCE": "cpu memory resource saturation",
                               "DEPENDENCY": "downstream dependency slow or failing", "UNKNOWN": "cannot determine"}},
  "severity":    {"type": "choice", "instructions": "How severe?",
                  "criteria": {"SEV1": "critical unavailability", "SEV2": "major degradation",
                               "SEV3": "partial degradation", "SEV4": "low impact"}},
  "priority_score": {"type": "score", "instructions": "Score investigation priority from 0 to 9 inclusive.",
                     "criteria": ["0","1","2","3","4","5","6","7","8","9"]},   # list, not dict
  "investigate":     {"type": "noul", "instructions": "Is investigation required?"},  # no criteria key
  "deployment_relevant": {"type": "noul", "instructions": "Is deployment relevant?"},
}
```

  (`glide/prompt.py::options_for` confirms `choice` needs a criteria object 1–255 options,
  `score` a list of level descriptions, `noul` optional criteria; `QUESTION_TYPES` matches.)
- **Response:** `result["answers"]` keyed by question name (same envelope as Laya, **different
  probabilities** — see differences below).
  - `choice`: `{type, choice (argmax), probabilities (sums ≈ 1.0 over all options),
    confidence (top − second)}`.
  - `score`: `{type, score (argmax level index), expected_level (Σ index×p),
    probabilities (per level "0".."9"), confidence}`.
  - `noul`: `{type, noul (P(True)), confidence (|2P−1|)}`. Parser rule: `bool(float(noul) >= 0.5)`
    (same cutoff as Spike 0).
  - **Score normalisation — explicitly the same as Laya:** `priority_score = expected_level / 9.0`
    (probability-weighted expected level over max level, **not** argmax index). `glide/model.py::_answer`
    defines `expected_level` identically; `raw_response` carries both `score` (index) and
    `expected_level` — the parser prefers `expected_level` and keeps `raw_score_index` for audit.
  - Sentinel `usage`: `{"input_tokens": 897–962 (summed over the 5 questions), "output_tokens": 0}`
    across this run (s00 observed 917). GLiDE runs **one forward pass per question** (5× prompt
    rendering, cf. README “One forward pass per question”; token count is ~2× Laya’s ~697), which
    is why input_tokens is higher despite the same state string.
  - **Probabilities are model estimates** (README Limitations), `temperature` in decision_config
    is 1.0 and not exposed per call. Same caution as Laya: do not treat as calibrated without
    Brier/calibration measurement (T2 evaluation stage). No logits, no temperature control.

### What GLiDE cannot give (do not invent)

- No REST endpoint is used (HTTP server exists in-repo but untested here); no per-question latency
  split; no temperature control on `decide`. Latency is client-side `time.perf_counter()`.
- `score` resolution is 10 levels here (list of 10); README notes up to 255 options per choice —
  finer granularity would need a wider `criteria` list or another provider.
- Out-of-vocab answers cannot be produced while `criteria` are set (argmax over given keys);
  the parser still guards (stale/mocked payloads).

## 3. Measured results (same 50 IncidentFeatures-shaped samples, seed 0)

| Device | Parseable | Latency P50 | Latency P95 | Deterministic (same input ×2) |
|---|---|---|---|---|
| `cuda` (RTX 3050 Ti) | **50/50** | **89.6 ms** | **94.0 ms** | **true** (full parsed-dict equality) |
| `cpu` (spot, n=3) | 3/3 | **11 562 ms** (3 rows) | 11 829 ms | true (2×) |

**Production timeout must exceed P95** (GPU ≥ ~120 ms observed max; CPU spot ≥ ~12 s) or the policy
layer will fall back spuriously on its first calls. The spike’s timeout test uses **10 ms** to force
the path — a test budget, not a production recommendation. CPU full-50 P50/P95 were not measured
(by instruction); the 3-row spot above is not a CPU contract claim.

**Quality distributions (CUDA, 50 parsed — evidence, not quality claims; T2/H2 decides):**

- `priority_score`: **min 0.338, max 0.499** (expected levels 3.04–4.49). Histogram (0.1 bins):
  0.3–0.4 = 14, 0.4–0.5 = 36, all other bins 0 — **every value is below 0.5**, i.e. below Laya’s
  already-dead cluster and far below the policy threshold `priority_score >= 0.65` (Spec §5.3).
- `category`: **LATENCY 41, RESOURCE 9, ERROR_SPIKE 0, DEPENDENCY 0, UNKNOWN 0** — collapsed to
  **≤2 categories** (verification criterion in task: yes, collapsed).
- `severity`: **SEV1 49, SEV2 1, SEV3 0, SEV4 0** — **98% SEV1 (≥90% one severity: yes)**,
  and it fires even on the 5 `symptom_tags normal` samples (error_rate 0–0.03, p95 300–500 ms)
  and all 15 `error_up` samples — none of which it called `ERROR_SPIKE`. Opposite failure to Laya
  (which understated to SEV4); GLiDE **overstates severity** (SEV1 for near-normal states).
- `investigate`: 50/50 `true`; `deployment_relevant`: 12 `true`.

Spot evidence (CUDA, `parsed` after spike-local stub parser):

```json
// s05 (db_pool 0.92, F01-like — same request_text as Laya s05)
{"category": "LATENCY", "severity": "SEV1", "priority_score": 0.458,
 "investigate": true, "deployment_relevant": false}
```

### Laya vs GLiDE head-to-head (same 50 inputs, CUDA)

| Dimension | Laya | GLiDE | Winner |
|---|---|---|---|
| Parseable | 50/50 | 50/50 | draw (contract) |
| Latency P50 | 50.2 ms | 89.6 ms | **Laya** (+78% slower) |
| Latency P95 | 52.9 ms | 94.0 ms | **Laya** |
| CPU (perception) | P50 703 ms / P95 1 709 ms | spot ~11.6 s/sample | **Laya** (~17× faster on CPU spot) |
| `priority_score` range | 0.497–0.509 (dead at 0.5) | 0.338–0.499 (dead **below** 0.5) | neither — both non-discriminative |
| Category spread | 5/5 present (22/10/9/5/4) | 2/5 present (41/9/0/0/0) | **Laya** |
| Severity spread | SEV2/SEV4 (understates; s05 SEV4) | 98% SEV1 (overstates; s05 SEV1) | **Laya** (still wrong, less degenerate) |
| s05 F01-like | `LATENCY/SEV4/0.501/investigate false` | `LATENCY/SEV1/0.458/investigate true` | GLiDE less wrong on `investigate`, worse on `severity` |
| Input tokens | ~697 | 897–962 (5 forward passes) | **Laya** (cheaper prompt) |
| Context headroom | 512 max_len (Laya budget) | 7,999/question (GLiDE) | **GLiDE** (only clear win) |

**Reading:** on the one dimension a System One provider lives or dies by — the decisions it actually
emits on identical inputs — GLiDE is **worse**: it collapses category, stamps SEV1 on near-normal
states, and its priority signal is systematically below the policy threshold. Its solo win
(context headroom) does not repair a degenerate decision distribution. (Laya’s numbers are from
`docs/laya-contract.md` §3 CUDA table + re-derived distributions over its 50 parsed CUDA samples.)

## 4. Failure modes → `ProviderError` mapping

Spike-local `ProviderError` (stub; real class in Day-1 `base.py`) is raised by the parser
and by the timeout guard. Verified results (CUDA; parser-level tests are device-independent):

| Failure | How triggered in spike | Maps to |
|---|---|---|
| **Timeout** | `ThreadPoolExecutor.future.result(timeout=0.01)` around `decide` | `ProviderError("timeout")` — wired (GPU ~90 ms > 10 ms). Client-side enforcement. |
| **Malformed output** | `parse_result` given a truncated payload (`{answers: {category: …}}`) | `ProviderError("malformed: missing decision key 'severity'")` — parser-level, not model-level. |
| **Out-of-vocab choice** | `parse_result` given `choice="NOT_A_CHOICE" / "SEV9"` with otherwise valid payload | `ProviderError("out-of-vocab choice: …")` — parser-level guard. Real checkpoint cannot produce this by construction (argmax over `criteria`). |

## 5. Fallback — stub-level only, real kill-test deferred to Day 1

What the spike proves: `triage_with_fallback()` (same seam as `spike_laya.py`, with `_DeadRouter.decide`
raising `ProviderError("simulated GLiDE process death (in-process)")`) catches `ProviderError` and
returns a `rules` decision with `provider == "rules"` and **no manual step** (`fallback_ok: true`).

What it does **not** prove: a real process kill. GLiDE here is in-process, so the test is injection-only.
There is no `rules.py` yet — `rules_fallback()` is the same explicit stub as Spike 0
(deny-if-error_rate≥0.2: `SEV2/ERROR_SPIKE` else `SEV3/UNKNOWN`).

**Day-1 obligation (if GLiDE were adopted):** replace the stub with the real `rules.py`, keep the same
try/`except ProviderError → rules` seam, and add a real kill-test (start the decision engine, kill the
GLiDE provider, assert the next triage returns `provider == "rules"`). Until then this contract claims
stub-level fallback only. (This applies identically to Laya — neither spike proves a real kill.)

## 6. Verdict vs Spec §5.4 pass criteria (for GLiDE as T2 candidate)

- [x] 50/50 parseable for all five decisions (CUDA; CPU spot 3/3 — full CPU not run)
- [x] Latency recorded (P50/P95 CUDA; CPU spot noted explicitly, not a claim)
- [x] Fallback proven against stub `ProviderError` → `rules` (no manual action)
  — real process-kill test deferred to Day 1, stated explicitly above

**GLiDE contract: PASS** (GPU scope; adapter wiring is contractually sound).
**Replacement recommendation: KEEP LAYA — do not replace on this evidence.** GLiDE is slower on both
devices the spot checked, collapses to 2 categories and 98% SEV1, and pushes `priority_score`
systematically *below* Laya’s dead-at-0.5 cluster. If replacement were still pursued, the burden of
proof flips: T2 must show GLiDE’s macro-F1 / severity / routing metrics beat Laya’s on the held-out
splits, not merely that it parses. **Quality risk — not this spike — decides T2/H2** (Spec §9.4):
the priority signal’s lack of discriminative power (Laya dead-at-0.5; GLiDE dead-below-0.5) plus
GLiDE’s severity overstatement are the risks T2 must measure with ground truth, and they are exactly
why Spike 0 kept Laya despite its own caveats.
