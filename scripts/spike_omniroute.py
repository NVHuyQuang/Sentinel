"""Spike 1: OmniRoute contract probe (spec section 12 spike 2, blocking).

Run: python scripts/spike_omniroute.py [--base-url http://localhost:20128/v1] [--model auto/deepseek]
Requires: OMNIROUTE_API_KEY in environment (OmniRoute client API key; never printed or logged).
Writes: docs/spike_omniroute_samples.jsonl (git-ignored raw per-sample evidence).
Prints: JSON summary to stdout and docs/spike_omniroute_summary.json (git-ignored raw summary).

Real calls only. Uses stdlib urllib (no new deps).

Stub ProviderError below mirrors Spike 0 naming (real class comes in Day 1 base.py).
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TRIAGE_SYSTEM = (
    "You must return exactly one JSON object and nothing else (no markdown, no extra text). "
    "Required keys: category, severity, priority_score, investigate, deployment_relevant. "
    "category MUST be exactly one of these 5 strings: LATENCY, ERROR_SPIKE, RESOURCE, DEPENDENCY, UNKNOWN. "
    "severity MUST be exactly one of these 4 strings: SEV1, SEV2, SEV3, SEV4. "
    "priority_score MUST be a number between 0 and 1 inclusive (for example 0.85, never 85 or 95). "
    "investigate and deployment_relevant MUST be JSON true or false. "
    "Example valid output: {\"category\":\"LATENCY\",\"severity\":\"SEV3\",\"priority_score\":0.5,"
    "\"investigate\":true,\"deployment_relevant\":false}. "
    "Never use human labels like 'Performance Degradation', 'High', or 'Critical'."
)
TRIAGE_USER = (
    "Classify this incident and output only the JSON object. "
    "Incident: primary_service order-service error_rate 0.24 p95_latency_ms 3420 "
    "cpu_utilization 0.55 memory_utilization 0.50 db_pool 0.92 anomaly_score 0.91 "
    "critical_rule_triggered true recent_deployment true symptom_tags latency_up,pool_saturated. "
    "JSON keys: category, severity, priority_score, investigate, deployment_relevant. "
    "category in [LATENCY,ERROR_SPIKE,RESOURCE,DEPENDENCY,UNKNOWN]. "
    "severity in [SEV1,SEV2,SEV3,SEV4]. priority_score 0-1."
)

CATEGORIES = {"LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN"}
SEVERITIES = {"SEV1", "SEV2", "SEV3", "SEV4"}


class ProviderError(Exception):
    """Stub mapping for Sentinel adapter: any transport or parse failure."""


def http_get_json(url, api_key, timeout=30):
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = None
            return resp.status, data, body, None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = None
        return e.code, data, body, None
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"


def http_post_json(url, api_key, payload, timeout=120, raw_body=False):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    if raw_body:
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
    else:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            lat = (time.perf_counter() - t0) * 1000
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = None
            return resp.status, data, body, lat, None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        lat = (time.perf_counter() - t0) * 1000
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = None
        return e.code, data, body, lat, None
    except Exception as e:
        # socket timeouts surface as TimeoutError / URLError / socket.timeout
        lat = (time.perf_counter() - t0) * 1000
        return None, None, None, lat, f"{type(e).__name__}: {e}"


def parse_triage(raw):
    """Parse raw chat completion into the 5 triage decisions, or raise ProviderError."""
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"malformed: missing choices[0].message.content {exc}") from exc
    if not isinstance(content, str):
        raise ProviderError("malformed: content is not a string")
    text = content.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = text.strip("`").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"malformed: content not JSON {exc}; content={content[:200]!r}") from exc
    try:
        category = obj["category"]
        severity = obj["severity"]
        priority_score = float(obj["priority_score"])
        investigate = bool(obj["investigate"])
        deployment_relevant = bool(obj["deployment_relevant"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"malformed: missing or bad triage field {exc}") from exc
    if category not in CATEGORIES:
        raise ProviderError(f"out-of-vocab choice: category={category!r}")
    if severity not in SEVERITIES:
        raise ProviderError(f"out-of-vocab choice: severity={severity!r}")
    if not 0.0 <= priority_score <= 1.0:
        raise ProviderError(f"malformed: priority_score out of range {priority_score}")
    return {
        "category": category,
        "severity": severity,
        "priority_score": priority_score,
        "investigate": investigate,
        "deployment_relevant": deployment_relevant,
    }


def percentile(data, p):
    s = sorted(data)
    if not s:
        return None
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    d = k - f
    return s[f] * (1 - d) + s[c] * d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:20128/v1")
    parser.add_argument("--model", default="auto/deepseek")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--out", default="docs/spike_omniroute_samples.jsonl")
    args = parser.parse_args()

    api_key = os.environ.get("OMNIROUTE_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("BLOCKED: OMNIROUTE_API_KEY not set in environment", file=sys.stderr)
        sys.exit(2)

    base = args.base_url.rstrip("/")
    models_url = f"{base}/models"
    chat_url = f"{base}/chat/completions"

    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Base URL: {base} Model: {args.model}", flush=True)

    # 1. Baseline health: GET /v1/models
    status, models_data, models_body, transport_err = http_get_json(models_url, api_key, timeout=30)
    live_ids = []
    context_by_id = {}
    if transport_err:
        print(f"GET /v1/models transport error: {transport_err}", flush=True)
        live = {"status": None, "total": 0, "transport_error": transport_err}
    elif status == 200 and isinstance(models_data, dict) and isinstance(models_data.get("data"), list):
        live_ids = [m.get("id") for m in models_data["data"] if isinstance(m, dict)]
        for m in models_data["data"]:
            if isinstance(m, dict) and m.get("id"):
                context_by_id[m["id"]] = m.get("context_length")
        live = {"status": status, "total": len(live_ids)}
    else:
        live = {"status": status, "total": 0, "error_body": (models_body or "")[:500]}

    # Compare against omniroute-models.json dump (BOM-safe).
    dump_path = Path("omniroute-models.json")
    dump_ids = []
    dump_ctx = {}
    dump_error = None
    if dump_path.exists():
        try:
            dump = json.loads(dump_path.read_text(encoding="utf-8-sig"))
            rows = dump.get("data", []) if isinstance(dump, dict) else []
            dump_ids = [m.get("id") for m in rows if isinstance(m, dict)]
            dump_ctx = {m.get("id"): m.get("context_length") for m in rows if isinstance(m, dict)}
        except Exception as e:
            dump_error = f"{type(e).__name__}: {e}"
    else:
        dump_error = "omniroute-models.json not found in cwd"

    wanted = ["auto/deepseek", "auto/cheap", "auto/best-free"]
    live_present = {w: (w in live_ids) for w in wanted}
    dump_present = {w: (w in dump_ids) for w in wanted}
    drift = None
    if live_ids and dump_ids:
        drift = {
            "live_total": len(live_ids),
            "dump_total": len(dump_ids),
            "live_context_deepseek": context_by_id.get("auto/deepseek"),
            "dump_context_deepseek": dump_ctx.get("auto/deepseek"),
            "live_context_cheap": context_by_id.get("auto/cheap"),
            "dump_context_cheap": dump_ctx.get("auto/cheap"),
            "live_context_bestfree": context_by_id.get("auto/best-free"),
            "dump_context_bestfree": dump_ctx.get("auto/best-free"),
        }

    print(f"GET /v1/models status={live.get('status')} live_total={live.get('total')} dump_total={len(dump_ids)}", flush=True)
    print(f"live_present={live_present} dump_present={dump_present} dump_error={dump_error}", flush=True)

    # Triage payload (exact request shape recorded for contract).
    triage_payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": TRIAGE_SYSTEM},
            {"role": "user", "content": TRIAGE_USER},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    rows = []
    latencies = []
    parsed_ok = 0
    usages = []
    resolved_models = {}
    cost_present = False
    cost_values = []

    for i in range(args.n):
        sid = f"t{i:02d}"
        status, raw, body, lat, terr = http_post_json(chat_url, api_key, triage_payload, timeout=args.timeout)
        if terr:
            print(f"{sid} transport error after {lat:.0f}ms: {terr}", flush=True)
            rows.append({"sample_id": sid, "request_shape": triage_payload, "raw_response": None,
                         "parsed": None, "latency_ms": lat, "transport_error": terr,
                         "http_status": status, "error_body": body})
            continue
        latencies.append(lat)
        parsed = None
        parse_error = None
        try:
            parsed = parse_triage(raw)
            parsed_ok += 1
        except ProviderError as exc:
            parse_error = str(exc)
            print(f"{sid} parse failed: {parse_error}", flush=True)
        if isinstance(raw, dict):
            usage = raw.get("usage")
            if isinstance(usage, dict):
                usages.append(usage)
            resolved = raw.get("model")
            if resolved:
                resolved_models[resolved] = resolved_models.get(resolved, 0) + 1
            # Cost fields: inspect top level and usage for any cost/price key.
            for k in ("cost", "total_cost", "cost_usd", "price"):
                if k in raw:
                    cost_present = True
                    cost_values.append(raw[k])
            if isinstance(usage, dict):
                for k in ("cost", "total_cost", "cost_usd", "price"):
                    if k in usage:
                        cost_present = True
                        cost_values.append(usage[k])
        rows.append({"sample_id": sid, "request_shape": triage_payload, "raw_response": raw,
                     "parsed": parsed, "latency_ms": lat, "parse_error": parse_error,
                     "http_status": status, "error_body": body if parsed is None and raw is None else None})
        print(f"{sid} status={status} latency_ms={lat:.0f} parsed={parsed}", flush=True)

    # Determinism: same prompt 3x (reuse first call result + 2 fresh calls if budget allows).
    det_parsed = []
    det_lat = []
    det_raw_ids = []
    # Reuse first successful parsed as call 1 if available to save latency, but still do 3 fresh
    # calls so evidence is explicit; cache hits are fast (measured below).
    for j in range(3):
        status, raw, body, lat, terr = http_post_json(chat_url, api_key, triage_payload, timeout=args.timeout)
        det_lat.append(lat)
        try:
            p = parse_triage(raw)
        except ProviderError as exc:
            p = None
            print(f"det{j} parse failed: {exc}", flush=True)
        det_parsed.append(p)
        if isinstance(raw, dict):
            det_raw_ids.append(raw.get("id"))
        rows.append({"sample_id": f"det{j}", "request_shape": triage_payload, "raw_response": raw,
                     "parsed": p, "latency_ms": lat, "http_status": status})
    deterministic = len(det_parsed) == 3 and det_parsed[0] is not None and det_parsed[0] == det_parsed[1] == det_parsed[2]
    print(f"deterministic={deterministic} det_parsed={det_parsed} det_raw_ids={det_raw_ids}", flush=True)

    # Failure modes.
    failure = {}

    # a) invalid model name
    p_invalid = {"model": "auto/does-not-exist-zzz", "messages": [{"role": "user", "content": "hello"}], "temperature": 0}
    failure["invalid_model"] = http_post_json(chat_url, api_key, p_invalid, timeout=30)

    # b) malformed JSON request (invalid JSON body)
    failure["malformed_json"] = http_post_json(chat_url, api_key, "{not valid json", timeout=30, raw_body=True)

    # c) 10x parallel calls (small payload to limit upstream blast radius)
    from concurrent.futures import ThreadPoolExecutor

    p_small = {"model": args.model, "messages": [{"role": "user", "content": "Reply with JSON object with keys category,severity,priority_score,investigate,deployment_relevant exactly. Incident: order-service error_rate 0.05."}], "temperature": 0}

    def one_call(_):
        return http_post_json(chat_url, api_key, p_small, timeout=args.timeout)

    with ThreadPoolExecutor(max_workers=10) as pool:
        parallel_results = list(pool.map(one_call, range(10)))
    failure["parallel_10"] = parallel_results

    # d) tiny client timeout (2s) — use a fresh (uncached) incident so it does not hit semantic cache.
    timeout_payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": TRIAGE_SYSTEM},
            {"role": "user", "content": "Classify this incident and output only the JSON object. Incident: primary_service payment-service error_rate 0.11 p95_latency_ms 2100 cpu_utilization 0.81 memory_utilization 0.77 db_pool n/a anomaly_score 0.77 critical_rule_triggered false recent_deployment false symptom_tags cpu_high,memory_high. JSON keys: category, severity, priority_score, investigate, deployment_relevant. category in [LATENCY,ERROR_SPIKE,RESOURCE,DEPENDENCY,UNKNOWN]. severity in [SEV1,SEV2,SEV3,SEV4]. priority_score 0-1."},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    failure["timeout_2s"] = http_post_json(chat_url, api_key, timeout_payload, timeout=2)

    # Tool calling probe
    tool_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Call the read-only tool query_metrics for service order-service metric error_rate."}],
        "temperature": 0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "query_metrics",
                    "description": "Read-only: query service metrics.",
                    "parameters": {
                        "type": "object",
                        "properties": {"service": {"type": "string"}, "metric": {"type": "string"}},
                        "required": ["service", "metric"],
                    },
                },
            }
        ],
    }
    failure["tool_call"] = http_post_json(chat_url, api_key, tool_payload, timeout=args.timeout)

    # Context / cost summary
    context_report = {w: context_by_id.get(w) for w in wanted}
    input_tokens = [u.get("prompt_tokens") for u in usages if isinstance(u, dict) and "prompt_tokens" in u]
    output_tokens = [u.get("completion_tokens") for u in usages if isinstance(u, dict) and "completion_tokens" in u]

    p50 = percentile(latencies, 50) if latencies else None
    p95 = percentile(latencies, 95) if latencies else None

    summary = {
        "base_url": base,
        "model_route": args.model,
        "python": sys.version.split()[0],
        "models_status": live.get("status"),
        "models_live_total": live.get("total"),
        "models_live_present": live_present,
        "models_dump_total": len(dump_ids),
        "models_dump_present": dump_present,
        "models_dump_error": dump_error,
        "drift": drift,
        "context_live": context_report,
        "n_triage": args.n,
        "parseable": parsed_ok,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "latencies_ms": latencies,
        "deterministic_3x": deterministic,
        "det_parsed": det_parsed,
        "det_response_ids": det_raw_ids,
        "det_latency_ms": det_lat,
        "resolved_models": resolved_models,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_reported": cost_present,
        "cost_values": cost_values,
        "failure_invalid_model": {"status": failure["invalid_model"][0], "body": (failure["invalid_model"][2] or "")[:1000], "transport": failure["invalid_model"][4]},
        "failure_malformed_json": {"status": failure["malformed_json"][0], "body": (failure["malformed_json"][2] or "")[:1000], "transport": failure["malformed_json"][4]},
        "failure_parallel_10": [{"status": r[0], "lat_ms": r[3], "transport": r[4]} for r in failure["parallel_10"]],
        "failure_timeout_2s": {"status": failure["timeout_2s"][0], "lat_ms": failure["timeout_2s"][3], "transport": failure["timeout_2s"][4]},
        "tool_call": {"status": failure["tool_call"][0], "body": (failure["tool_call"][2] or "")[:3000], "transport": failure["tool_call"][4]},
    }

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    with open("docs/spike_omniroute_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
