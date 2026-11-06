"""Spike 0 (GLiDE variant): real GLiDE-0.4B contract probe, parallel to spike_laya.py / spike_joey.py.

Run: python scripts/spike_glide.py [--device cuda|cpu] (default cuda) [--model GLiDE-0.4B]
Writes: docs/spike_glide_samples-<device>.jsonl (one line per sample) and prints a summary.

Contract under test (from glide package in repo fastino/GLiDE-0.4B):
  Glide.from_pretrained(path) -> model.decide(state, questions)
  questions = {"name": {"type": "choice|score|noul", "instructions": str, "criteria": dict|list}}
  choice criteria: {label: description}; score criteria: ["0".."9"] (up to 255 levels);
  noul has no criteria. Result["answers"][name] holds choice/probabilities or
  score/probabilities/expected_level or noul float.

Native GLiDE encoder 0.4B bf16 ~0.81GB fits 4GB VRAM easily. No quantization needed.
Requires PYTHONPATH to include model dir (glide package lives in repo), so script adds it.

Stub ProviderError / rules fallback below are explicit stubs for this spike only.
Real services/decision_engine/providers/base.py + rules.py come in Day 1.
"""

import sys

# Bypass torchaudio DLL loading issue on Windows (as in run_laya_cpu.py)
sys.modules["torchaudio"] = None

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import pathlib


class ProviderError(Exception):
    """Stub for Day-1 base.py ProviderError (spike-local, do not import elsewhere)."""


QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "What best describes the incident?",
        "criteria": {
            "LATENCY": "high latency timeouts slowness degraded performance",
            "ERROR_SPIKE": "5xx errors failure rate high",
            "RESOURCE": "cpu memory resource saturation",
            "DEPENDENCY": "downstream dependency slow or failing",
            "UNKNOWN": "cannot determine",
        },
    },
    "severity": {
        "type": "choice",
        "instructions": "How severe?",
        "criteria": {
            "SEV1": "critical unavailability",
            "SEV2": "major degradation",
            "SEV3": "partial degradation",
            "SEV4": "low impact",
        },
    },
    "priority_score": {
        "type": "score",
        "instructions": "Score investigation priority from 0 to 9 inclusive.",
        "criteria": [str(i) for i in range(10)],
    },
    "investigate": {"type": "noul", "instructions": "Is investigation required?"},
    "deployment_relevant": {
        "type": "noul",
        "instructions": "Is deployment relevant?",
    },
}

CATEGORIES = {"LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN"}
SEVERITIES = {"SEV1", "SEV2", "SEV3", "SEV4"}
SERVICES = ["api-gateway", "auth-service", "order-service", "payment-service"]


def make_samples(n=50):
    rng = random.Random(0)
    samples = []
    for i in range(n):
        service = rng.choice(SERVICES)
        affected = [service]
        if rng.random() < 0.5:
            other = rng.choice([s for s in SERVICES if s != service])
            affected.append(other)
        kind = rng.choice(["latency", "error_spike", "resource", "dependency", "low"])
        if kind == "latency":
            err = round(rng.uniform(0.0, 0.15), 3)
            p95 = int(rng.uniform(1200, 4000))
            base = int(rng.uniform(300, 600))
            cpu, mem = round(rng.uniform(0.4, 0.7), 2), round(rng.uniform(0.4, 0.6), 2)
            pool = round(rng.uniform(0.85, 1.0), 2) if rng.random() < 0.6 else None
            anomaly = round(rng.uniform(0.65, 0.95), 3)
            crit = rng.random() < 0.5
            deploy = rng.random() < 0.4
            tags = ["latency_up"]
        elif kind == "error_spike":
            err = round(rng.uniform(0.2, 0.65), 3)
            p95 = int(rng.uniform(800, 2500))
            base = int(rng.uniform(300, 600))
            cpu, mem, pool = round(rng.uniform(0.4, 0.7), 2), round(rng.uniform(0.4, 0.6), 2), None
            anomaly = round(rng.uniform(0.75, 0.98), 3)
            crit = rng.random() < 0.6
            deploy = rng.random() < 0.3
            tags = ["error_up"]
        elif kind == "resource":
            err = round(rng.uniform(0.0, 0.2), 3)
            p95 = int(rng.uniform(900, 3000))
            base = int(rng.uniform(300, 600))
            cpu = round(rng.uniform(0.9, 0.99), 2)
            mem = round(rng.uniform(0.85, 0.98), 2)
            pool = None
            anomaly = round(rng.uniform(0.7, 0.95), 3)
            crit = False
            deploy = rng.random() < 0.2
            tags = ["cpu_high", "memory_high"]
        elif kind == "dependency":
            err = round(rng.uniform(0.05, 0.3), 3)
            p95 = int(rng.uniform(1500, 5000))
            base = int(rng.uniform(300, 600))
            cpu, mem, pool = round(rng.uniform(0.4, 0.7), 2), round(rng.uniform(0.4, 0.6), 2), None
            anomaly = round(rng.uniform(0.7, 0.97), 3)
            crit = False
            deploy = False
            tags = ["dependency_timeout"]
        else:
            err = round(rng.uniform(0.0, 0.03), 3)
            p95 = int(rng.uniform(300, 500))
            base = int(rng.uniform(300, 500))
            cpu, mem, pool = round(rng.uniform(0.2, 0.4), 2), round(rng.uniform(0.2, 0.4), 2), None
            anomaly = round(rng.uniform(0.1, 0.4), 3)
            crit = False
            deploy = rng.random() < 0.1
            tags = ["normal"]
        pool_s = f"db_pool {pool}" if pool is not None else "db_pool n/a"
        text = (
            f"primary_service {service} affected {'/'.join(affected)} "
            f"error_rate {err} p95_latency_ms {p95} baseline_p95_ms {base} "
            f"cpu_utilization {cpu} memory_utilization {mem} {pool_s} "
            f"anomaly_score {anomaly} critical_rule_triggered {str(crit).lower()} "
            f"recent_deployment {str(deploy).lower()} symptom_tags {','.join(tags)}"
        )
        samples.append({"sample_id": f"s{i:02d}", "request_text": text})
    return samples


def parse_result(raw):
    """Parse raw GLiDE payload into the 5 Sentinel decisions, or raise ProviderError."""
    try:
        answers = raw["answers"]
        cat_ans = answers["category"]
        sev_ans = answers["severity"]
        pri_ans = answers["priority_score"]
        inv_ans = answers["investigate"]
        dep_ans = answers["deployment_relevant"]
    except (KeyError, TypeError) as exc:
        raise ProviderError(f"malformed: missing decision key {exc}") from exc
    category = cat_ans.get("choice")
    severity = sev_ans.get("choice")
    if category not in CATEGORIES or severity not in SEVERITIES:
        raise ProviderError(
            f"out-of-vocab choice: category={category!r} severity={severity!r}"
        )
    try:
        # GLiDE score returns most-probable level index; expected_level is float.
        # Use expected_level if present (better), else score index.
        if "expected_level" in pri_ans:
            score_expected = float(pri_ans["expected_level"])
        else:
            score_expected = float(pri_ans["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"malformed: bad priority_score {exc}") from exc
    priority_score = score_expected / 9.0
    if not 0.0 <= priority_score <= 1.0:
        raise ProviderError(f"malformed: priority_score out of range {priority_score}")
    try:
        inv_noul = float(inv_ans["noul"])
        dep_noul = float(dep_ans["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"malformed: bad noul {exc}") from exc
    return {
        "category": category,
        "severity": severity,
        "priority_score": priority_score,
        "investigate": inv_noul >= 0.5,
        "deployment_relevant": dep_noul >= 0.5,
        "raw_score_expected": score_expected,
        "raw_score_index": pri_ans.get("score"),
        "raw_noul": {"investigate": inv_noul, "deployment_relevant": dep_noul},
        "provider": "glide",
        "model_version": raw.get("model", "unknown"),
    }


def rules_fallback(text):
    """Explicit STUB rules fallback (Day-1 rules.py replaces this)."""
    err = 0.0
    for tok in text.split():
        if tok.startswith("error_rate"):
            try:
                err = float(tok.split("=")[-1] if "=" in tok else 0.0)
            except ValueError:
                err = 0.0
    severity = "SEV2" if err >= 0.2 else "SEV3"
    return {
        "category": "ERROR_SPIKE" if err >= 0.2 else "UNKNOWN",
        "severity": severity,
        "priority_score": 0.8 if err >= 0.2 else 0.5,
        "investigate": True,
        "deployment_relevant": None,
        "provider": "rules",
        "model_version": None,
    }


def triage_with_fallback(model, text):
    try:
        t0 = time.perf_counter()
        raw = model.decide(text, QUESTIONS)
        lat = (time.perf_counter() - t0) * 1000
        return parse_result(raw), raw, lat
    except ProviderError:
        return rules_fallback(text), None, None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--model", default="GLiDE-0.4B")
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Loading GLiDE model on {args.device} (model={args.model})...", flush=True)
    # glide package lives in repo root; add to path
    sys.path.insert(0, str(pathlib.Path(args.model).resolve()))
    from glide import Glide, __version__ as glide_version

    print(f"glide version: {glide_version}", flush=True)
    device = "cuda" if args.device == "cuda" else "cpu"
    # Try cuda, fallback cpu if CUDA unavailable
    try:
        model = Glide.from_pretrained(args.model, device=device)
    except Exception as e:
        print(f"cuda load failed ({e}), trying cpu", flush=True)
        model = Glide.from_pretrained(args.model, device="cpu")
        device = "cpu"
    print("Warmup (decide 'warmup')...", flush=True)
    model.decide("warmup", QUESTIONS)

    samples = make_samples(args.n_samples)
    rows = []
    latencies = []
    parseable = 0
    for s in samples:
        t0 = time.perf_counter()
        try:
            raw = model.decide(s["request_text"], QUESTIONS)
        except Exception as exc:
            lat = (time.perf_counter() - t0) * 1000
            print(f"{s['sample_id']} decide failed: {type(exc).__name__}: {exc}", flush=True)
            rows.append(
                {
                    "sample_id": s["sample_id"],
                    "request_text": s["request_text"],
                    "questions": QUESTIONS,
                    "raw_response": None,
                    "parsed": None,
                    "latency_ms": lat,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        try:
            parsed = parse_result(raw)
            parseable += 1
        except ProviderError as exc:
            parsed = None
            print(f"{s['sample_id']} parse failed: {exc}", flush=True)
        rows.append(
            {
                "sample_id": s["sample_id"],
                "request_text": s["request_text"],
                "questions": QUESTIONS,
                "raw_response": raw,
                "parsed": parsed,
                "latency_ms": lat,
            }
        )
        print(f"{s['sample_id']} latency_ms={lat:.1f} parsed={parsed}", flush=True)

    out = pathlib.Path(f"docs/spike_glide_samples-{device}.jsonl")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def percentile(data, p):
        s = sorted(data)
        k = (len(s) - 1) * (p / 100)
        f, c = int(k), min(int(k) + 1, len(s) - 1)
        d = k - f
        return s[f] * (1 - d) + s[c] * d

    p50 = percentile(latencies, 50) if latencies else None
    p95 = percentile(latencies, 95) if latencies else None

    det_text = samples[0]["request_text"]
    det_raw1 = model.decide(det_text, QUESTIONS)
    det_raw2 = model.decide(det_text, QUESTIONS)
    det_p1 = parse_result(det_raw1)
    det_p2 = parse_result(det_raw2)
    deterministic = det_p1 == det_p2

    timeout_result = "not-run"
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(model.decide, det_text, QUESTIONS)
            try:
                fut.result(timeout=0.01)
                timeout_result = "unexpectedly completed within 0.01s (no timeout)"
            except FuturesTimeout:
                timeout_result = "ProviderError('timeout'): decide exceeded 0.01s client timeout"
    except Exception as exc:
        timeout_result = f"ProviderError: {type(exc).__name__}: {exc}"

    try:
        parse_result({"answers": {"category": {"choice": 123}}})
        malformed_result = "unexpectedly parsed"
    except ProviderError as exc:
        malformed_result = f"ProviderError: {exc}"

    try:
        parse_result(
            {
                "answers": {
                    "category": {"choice": "NOT_A_CHOICE"},
                    "severity": {"choice": "SEV9"},
                    "priority_score": {"score": 4, "expected_level": 4.5},
                    "investigate": {"noul": 0.9},
                    "deployment_relevant": {"noul": 0.1},
                }
            }
        )
        oov_result = "unexpectedly parsed"
    except ProviderError as exc:
        oov_result = f"ProviderError: {exc}"

    class _DeadRouter:
        def decide(self, *a, **k):
            raise ProviderError("simulated GLiDE process death (in-process)")

    fb_decision, fb_raw, fb_lat = triage_with_fallback(
        _DeadRouter(), samples[0]["request_text"]
    )
    fallback_ok = (fb_raw is None and fb_decision["provider"] == "rules" and "severity" in fb_decision)

    summary = {
        "glide_model": args.model,
        "glide_version": glide_version,
        "provider": "glide",
        "device": f"{device} (Glide device='{device}')",
        "python": sys.version.split()[0],
        "n_samples": len(samples),
        "parseable": parseable,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "deterministic_same_input_2x": deterministic,
        "failure_timeout": timeout_result,
        "failure_malformed": malformed_result,
        "failure_out_of_vocab": oov_result,
        "fallback_ok": fallback_ok,
        "fallback_decision": fb_decision,
        "samples_file": str(out),
    }
    print(json.dumps(summary, indent=2), flush=True)
    with pathlib.Path(f"docs/spike_glide_summary-{device}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
