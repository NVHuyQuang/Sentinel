# OmniRoute Contract (Spike 1, blocking — Spec §12 spike 2)

Status: **PASS** (authenticated local gateway, structured output, failure paths demonstrated). The T3 LLM baseline and System Two investigation wiring are **unblocked on contract grounds**; semantic-cache behaviour and the `local-omniroute` auth discrepancy below must be handled by the Day-1 `llm.py` adapter.

Every number below is a real measured call from `scripts/spike_omniroute.py`, not a spec guess.
Reproduce with (after starting OmniRoute, see §1):

```powershell
omniroute serve --daemon --no-open
$env:OMNIROUTE_API_KEY = "<client API key from OmniRoute dashboard / api_keys store>"
python scripts/spike_omniroute.py --model auto/deepseek --n 20
```

Raw per-sample evidence lives in `docs/spike_omniroute_samples.jsonl` (git-ignored,
see `.gitignore` `docs/spike_omniroute_*`). This file is the committed summary.
Raw summary (same run): `docs/spike_omniroute_summary.json` (also git-ignored).

## 1. How OmniRoute is invoked

- **Base URL:** `http://localhost:20128/v1` (spec default `LLM_BASE_URL`; confirmed live on this host).
- **Route:** `POST /v1/chat/completions` (OpenAI-compatible). Discovery: `GET /v1/models`.
- **How started on this host:** `omniroute serve --daemon --no-open` (OmniRoute CLI **3.8.52**).
  Server listens on `0.0.0.0:20128` by default; dashboard at `http://localhost:20128`.
  Server was stopped (`omniroute stop`) after the spike. This is a local Node.js daemon,
  not Docker Compose.
- **Auth — spec default does NOT work on this host.** Spec §6.1 says `LLM_API_KEY=local-omniroute`,
  but unauthenticated `GET /v1/models` returns `401 {"error":{"message":"Authentication required",
  "type":"invalid_api_key","code":"invalid_api_key"}}`, and `Bearer local-omniroute` returns
  `401 {"error":{"message":"Invalid API key",…}}`. Real authenticated calls require the
  OmniRoute client API key (`OMNIROUTE_API_KEY`, `Authorization: Bearer <key>`). The Day-1 `llm.py`
  adapter must read a real `OMNIROUTE_API_KEY` (or `LLM_API_KEY` **value** containing it) from env —
  do not bake in `local-omniroute`. The repo `.env` HF token was not used for OmniRoute calls and
  was never printed.
- **Resolved upstream model per call:** route `auto/deepseek` → `deepseek-ai/deepseek-v4-pro-0813`
  (all 20 triage calls; `raw_response.model` field; server log shows
  `cloudflare-playground/deepseek-ai/deepseek-v4-pro-0813` via combo routing,
  `cfp/deepseek-ai/deepseek-v4-pro-0813`). This is a free/no-auth upstream; a paid configured
  provider (`moonshot`) exists but returned `429 … account … suspended due to insufficient balance`
  when tried directly and is not part of the `auto/deepseek` pool. No combos are user-configured
  (`omniroute combo list` → “No combos configured”); `auto/*` names are built-in virtual combos.
- **Python:** 3.10.11 on this host (`scripts/spike_omniroute.py` uses stdlib `urllib` only; no new deps).
- **Semantic cache:** identical prompts return the **same response `id`** (e.g. triage
  `chatcmpl-cfp-9f9b6afd-dfd` for t01–t19 + det0–det2) in ~2.1 s after the first cold call (~61 s).
  Latency tables below report client-measured `time.perf_counter()` around `urlopen`, so cold
  (~39–72 s across runs) and cached (~2.1 s) distributions are visible, not averaged away.

### Baseline health

- `GET /v1/models` (authenticated) → **status 200**, live total **473** models at spike time.
- Required routes present live: `auto/deepseek` ✅, `auto/cheap` ✅, `auto/best-free` ✅.
- Dump comparison (`omniroute-models.json`, git-ignored, read with `utf-8-sig` for BOM):
  dump total **440**, same three routes present. **Drift: live 473 vs dump 440 (+33)** —
  the catalog is dynamic (free-pool membership changes between runs; earlier in this session
  live was 454, then 473). `context_length` agreed on the three routes (see §6);
  do not treat the dump as a stable pin.

## 2. Exact request / response contract

Exact triage request shape (also stored per-row as `request_shape` in raw samples):

```json
{
  "model": "auto/deepseek",
  "messages": [
    {"role": "system", "content": "You must return exactly one JSON object and nothing else (no markdown, no extra text). Required keys: category, severity, priority_score, investigate, deployment_relevant. category MUST be exactly one of these 5 strings: LATENCY, ERROR_SPIKE, RESOURCE, DEPENDENCY, UNKNOWN. severity MUST be exactly one of these 4 strings: SEV1, SEV2, SEV3, SEV4. priority_score MUST be a number between 0 and 1 inclusive (for example 0.85, never 85 or 95). investigate and deployment_relevant MUST be JSON true or false. Example valid output: {\"category\":\"LATENCY\",\"severity\":\"SEV3\",\"priority_score\":0.5,\"investigate\":true,\"deployment_relevant\":false}. Never use human labels like 'Performance Degradation', 'High', or 'Critical'."},
    {"role": "user", "content": "Classify this incident and output only the JSON object. Incident: primary_service order-service error_rate 0.24 p95_latency_ms 3420 cpu_utilization 0.55 memory_utilization 0.50 db_pool 0.92 anomaly_score 0.91 critical_rule_triggered true recent_deployment true symptom_tags latency_up,pool_saturated. JSON keys: category, severity, priority_score, investigate, deployment_relevant. category in [LATENCY,ERROR_SPIKE,RESOURCE,DEPENDENCY,UNKNOWN]. severity in [SEV1,SEV2,SEV3,SEV4]. priority_score 0-1."}
  ],
  "temperature": 0,
  "response_format": {"type": "json_object"}
}
```

- `temperature: 0` and `response_format: {"type": "json_object"}` were both sent and honored
  syntactically (responses are JSON-parseable objects in `choices[0].message.content`; see §3 caveat).
  The upstream is a reasoning model (`message.reasoning_content` present in every triage + tool response);
  `temperature` does not suppress it. No other sampling params were sent.
- **Response:** `choices[0].message.content` is a JSON **string** (not a pre-parsed object).
  Parser rule (spike-local, mirrors Day-1 `llm.py` obligation): `json.loads(content.strip())`, then
  validate all five keys, `category ∈ {LATENCY,ERROR_SPIKE,RESOURCE,DEPENDENCY,UNKNOWN}`,
  `severity ∈ {SEV1,SEV2,SEV3,SEV4}`, `priority_score ∈ [0,1]`, booleans for the other two.
  Any miss → `ProviderError` (see §5). Prior prompt wording without the enum/example guard produced
  `{"category":"Performance Degradation","severity":"High","priority_score":95,…}` (0/20 parse) —
  the enum guard above is contractually required, not optional polish.

Spot evidence (`t00` cold parsed after spike-local parser — real `llm.py` in Day 1):

```json
{"category": "LATENCY", "severity": "SEV1", "priority_score": 0.95,
 "investigate": true, "deployment_relevant": true}
```

with `raw_response.model == "deepseek-ai/deepseek-v4-pro-0813"` and
`raw_response.usage == {"prompt_tokens": 453, "completion_tokens": 540, "total_tokens": 993, …}`.

### Tool calling

Probe: `tools: [{"type":"function","function":{"name":"query_metrics",…}}]`, user asked to “Call the
read-only tool query_metrics for service order-service metric error_rate”.
Response **200 with no `tool_calls` structure** (no `message.tool_calls` key). Instead,
`message.content` is free text beginning `"tool_call:\n- name: query_metrics\n  arguments: …"`.
From this measurement the Day-1 adapter **cannot rely on a structured `tool_calls` array** for
`auto/deepseek` — the Evidence Investigator loop must either (a) walk free-text the way the model
emitted it (fragile, keep behind `ProviderError` on no-match), or (b) re-probe against a different
route with measured `tool_calls` output before promising tool use. `/v1/models` still advertises
`"tool_calling": true` for `auto/deepseek` — the advertisement did not match the observed wire format.

## 3. Measured results (20 identical triage prompts, `auto/deepseek`, temp 0)

| Metric | Measured value |
|---|---|
| Parseable (all 5 decisions) | **20/20** (cold `t00` + 19 cached repeats, full parsed-dict equality across repeats) |
| Latency P50 | **2 081 ms** (dominated by semantic-cache hits) |
| Latency P95 | **5 053 ms**¹ |
| Deterministic (same input ×3) | **true** (identical parsed dicts; identical response `id` — cache-assisted, see below) |
| Input tokens per call | **453** (`prompt_tokens`, all 20; includes system prompt) |
| Output tokens per call | **540** (`completion_tokens`, all 20; includes reasoning + JSON) |
| Resolved upstream | `deepseek-ai/deepseek-v4-pro-0813` (20/20) |
| Cost field | **not reported** (no `cost`/`total_cost`/`price` key in any response or usage) |

¹ P95 is bimodal, not a Gaussian tail: `t00` cold = **60 634 ms**, the other 19 = **2 055–2 128 ms**
(cache hits, same `id`). Across this session cold `auto/deepseek` calls measured **38.8 s, 60.6 s,
71.7 s**; cached repeats **~2.1 s**. **Production timeout must exceed the cold path by a wide margin**
(≥ 75 s suggested starting point) or the adapter will fall back spuriously on every first-seen prompt —
the spike’s 2 s timeout test in §5 is a test budget, not a production recommendation. Note also that
output tokens (540) are inflated by `reasoning_content`-adjacent verbosity; budget Sentinel
token accounting from `usage` as returned, not from content length.

## 4. What OmniRoute cannot give (do not invent)

- **No cost / pricing field.** `usage` carries only token counts (`prompt_tokens`,
  `completion_tokens`, `total_tokens`, `cached_tokens: 0` in this run, etc.). No per-call USD cost
  was returned. Sentinel’s “cost per 1,000 decisions” (Spec §9.4) **cannot be computed from `/v1/chat/completions`
  responses on this host** — it must come from OmniRoute’s cost report (`omniroute cost`) or be reported
  as unavailable. Do not fabricate a per-token price.
- **No rate-limit headers observed on triage responses** (none recorded in raw bodies/headers probed;
  parallel test in §5 showed no 429 for 10× parallel). The absence of a 429 in a 10-call parallel burst
  is not a limit — the adapter must still handle 429 if it appears (see §5).
- **No guaranteed determinism beyond the cache.** 3× determinism held via identical response `id`
  (semantic cache hit). A truly fresh prompt has no determinism guarantee measured here
  (different prompts retried upstream: e.g. `auto/best-free` cold resolved to `zai-org/glm-5.2` at
  ~70 s and asked for clarification instead of emitting JSON). Spec §9.5’s
  “run each variant 3× and report mean/spread” still applies.
- **No `tool_calls` array from `auto/deepseek` in this probe** (§2). `capabilities.tool_calling: true`
  in `/v1/models` did not produce the structure.
- **No stable model catalog.** Live total moved 454 → 473 within this session; `auto/*` pool members
  are dynamic. `context_length` values below are point-in-time, not pins.

## 5. Failure modes → `ProviderError` mapping

Spike-local `ProviderError` (stub; real class in Day-1 `base.py`) is raised by the parser and by the
transport/timeout guard. Adapter rule (mirrors Spike 0 / Spec §13): **validate, retry once, then
`ProviderError` → `rules` fallback** (no manual action). Verified results:

| Failure | How triggered in spike | Observed | Adapter action |
|---|---|---|---|
| **Unauthenticated baseline** | `GET /v1/models` with no key and with `Bearer local-omniroute` | `401 Authentication required` / `401 Invalid API key` | Fail fast (no retry — config error). Sentinel startup must require a real `OMNIROUTE_API_KEY`; otherwise every LLM call is a `ProviderError` and triage falls back to `rules`. |
| **Invalid model name** | `model: "auto/does-not-exist-zzz"` | `400 {"error":{"message":"Model 'auto/does-not-exist-zzz' is not a valid combo or provider. Unknown built-in auto combo.","type":"invalid_request_error","code":"bad_request"}}` | `ProviderError("invalid_model")` — no retry (deterministic 400), fall back to `rules`. |
| **Malformed JSON request** | POST body `{not valid json` | `400 {"error":{"message":"Invalid JSON body","type":"invalid_request_error","code":"bad_request"}}` | `ProviderError("malformed_request")` — no retry (client bug), fall back. |
| **Malformed model output** | Parser given `content` with wrong enum (first prompt wording: `Performance Degradation`/`High`/`95`) | Parser raises `ProviderError("out-of-vocab choice: category=…")` | Retry once with the §2 enum-guarded prompt (as in this spike), then `ProviderError("malformed_output")` → `rules`. Real `llm.py` must keep the enum guard — prompt wording is part of the contract. |
| **Out-of-vocab choice** | Same as above (parser-level) | Parser-level `ProviderError`, not a transport error | Same as malformed-output retry path. |
| **Timeout (2 s client)** | Fresh (uncached) incident, `timeout=2` | Transport `TimeoutError: timed out` (no HTTP status; measured ~4.0 s wall for the 2 s socket timeout) | `ProviderError("timeout")`. Retry once within budget, then mark inconclusive / fall back. Cold upstream latency is 39–72 s — a 2 s budget only ever demonstrates the path. |
| **10× parallel** | 10 concurrent `POST /v1/chat/completions` (small payload) | **10/10 `200`**, no 429 (latency ~2.2–2.3 s — payload already semantically cached from earlier in the run; cold parallel behaviour not separately timed) | No special handling needed for this burst; still implement 429 → retry-once-with-backoff → `ProviderError("rate_limited")` so a real limit degrades to `rules` instead of hammering. Do not claim “no rate limit” from this test. |

Fallback note: this spike proves parser- and transport-level `ProviderError` mapping only, using the same
stub seam as Spike 0 (`triage_with_fallback` equivalent to be re-implemented in Day-1 `llm.py` +
`rules.py`). A real OmniRoute outage kill-test (`omniroute stop` then assert next triage returns
`provider == "rules"` with no manual action) is a **Day-1 obligation**, same as Spike 0’s deferred
kill-test. The server was in fact stopped after this spike (`omniroute stop` succeeded), but no
Sentinel fallback path was wired yet to observe. Until then this contract claims mapping-level fallback only.

## 6. Context / token / cost reporting (actually returned)

From live `GET /v1/models` (point-in-time at spike run):

| Route | `context_length` | `max_input_tokens` | `max_output_tokens` | Tool-calling advertised |
|---|---|---|---|---|
| `auto/deepseek` | **1 000 000** | 1 000 000 | 384 000 | true (wire format not observed — §2) |
| `auto/cheap` | **400 000** | 400 000 | 8 192 | true |
| `auto/best-free` | **400 000** | 400 000 | 8 192 | true |

Dump (`omniroute-models.json`) `context_length` matched on all three routes. Token usage per triage
call: `usage.prompt_tokens 453`, `usage.completion_tokens 540`, `usage.total_tokens 993`,
`usage.cached_tokens 0` (plus `prompt_tokens_details.cached_tokens 0`). No cost/price field in any
response body or usage object — **cost: not reported**. The adapter must record `model` (route) +
`raw_response.model` (resolved upstream) per experiment run (Spec §6.1) and must not sum upstream
reasoning tokens into Sentinel-visible figures without labelling them.

## 7. Verdict vs Spec §12 spike list (spike 2)

§12 spike 2 asks to confirm: local endpoint, configured route, model discovery, JSON/structured
output behavior and failure rate, tool-calling support and limits, routing fallbacks, rate limits
and timeouts, context window, token/cost reporting, and which sampling parameters are honored.

- [x] Local endpoint `http://localhost:20128/v1` answers (`GET /v1/models` 200, authenticated)
- [x] Configured route `auto/deepseek` resolves (`deepseek-ai/deepseek-v4-pro-0813`) and serves structured output
- [x] **20/20** structured outputs parse for all five decisions (enum-guarded prompt, temp 0, JSON mode)
- [x] Failure paths demonstrated (invalid model 400, malformed JSON 400, 2 s timeout → transport error)
- [~] Tool-calling: API returns 200 but **no `tool_calls` array** — free-text instead (limit documented in §2)
- [~] Cost/token: tokens reported per call; **cost not reported** (documented, not invented)
- [~] Sampling: `temperature 0` + `response_format json_object` sent and JSON-shaped responses returned;
  reasoning verbosity and cache-assisted determinism mean temp-0 is not a determinism guarantee

**PASS.** `llm.py` may be implemented in Day 1 against §2 exactly as measured, with the
authenticated-key requirement, enum-guarded prompt, cold-latency timeout (≥ 75 s), cache-aware
benchmarking, and free-text tool-call handling above. Task 3 (Day 1 skeleton) **can start**;
`llm.py` itself is still Day-1 work, not done here.
