"""RCA ablation R0-R3 on the F01 recorded replay (Spec 9.5, RQ3).

    py -3.12 -m evaluation.ablation --repeats 3 --deadline 600

R0 single LLM, fixed incident summary, no tools, no evidence validator.
R1 single LLM with the same budgeted read-only toolbox; RCA validator on.
R2 multi-agent graph (context, hypotheses, investigator, gate, RCA).
R3 R2 plus search_historical_incidents. The historical store is EMPTY today, so R3 gets no
   past incidents; nothing is fabricated to make it differ.

Fairness: one pipeline run captures the identical incident features and snapshot view; every
variant starts from the same summary (evidence E1), the same OmniRoute route, budgets
(12 calls/round, 30 total, one deadline) and prompts, and temperature 0. Each (variant, repeat)
gets its own freshly emptied llm cache dir, so R2 replies cannot leak into R1. The hidden
manifest is read only after every variant has finished. No raw prompts are written to reports.
"""

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import httpx

from evaluation.f01_runner import RUN_ID, make_primary, record_all
from services.decision_engine.providers.rules import RulesProvider
from services.investigation.graph import VARIANTS, InvestigationResult, SystemTwo
from services.investigation.llm import LLM, ChatClient
from services.investigation.rca import ReportValidationError, validate_report
from services.pipeline import fit_detectors, run_pipeline
from services.telemetry.backend import SnapshotBackend
from simulator.fault_controller import GroundTruthStore

# Evaluator-only answer key, keyed by the manifest fault name. Never shown to any agent.
TRUTH = {"db_connection_pool_exhaustion": {
    "fault_id": "F01", "category": "RESOURCE_EXHAUSTION",
    "alternatives": [("cpu",), ("regression", "release", "deploy")]}}
NUMERIC = ("duration_s", "tokens", "llm_calls", "tool_calls", "evidence_precision",
           "evidence_recall", "unsupported_claim_rate", "alt_coverage", "cache_hit_rate")
FLAGS = ("exact_fault", "category_correct", "completed")


def metrics_for(res: InvestigationResult, manifest: dict) -> dict:
    exp, truth = manifest["expected"], TRUTH[manifest["fault"]]
    rep, llm = res.report, res.extras.get("llm", {})
    known = {e.id: e for e in res.evidence}
    tele = [e for e in res.evidence if e.source_type != "incident_summary"]
    rel = {tuple(x) for x in exp["relevant_signals"]}

    def key(e):
        return (e.service, e.query_parameters.get("metric") or e.source_type)

    m: dict = {"outcome": res.outcome, "error": res.error, "completed": res.outcome == "investigated",
               "exact_fault": False, "category_correct": False, "evidence_precision": None,
               "unsupported_claim_rate": None, "alt_coverage": None,
               "evidence_recall": len({key(e) for e in tele} & rel) / len(rel)}
    if rep:
        text = (rep.probable_root_cause or "").lower()
        m["exact_fault"] = bool(text) and all(k.lower() in text for k in exp["root_cause_keywords"])
        m["category_correct"] = bool(text) and res.extras.get(
            "root_cause_category") == truth["category"]
        facts = [c for c in rep.claims if c.kind == "fact"]
        inv_id = res.evidence[0].investigation_id
        bad = 0
        for c in facts:  # judged per claim against retrieved records, same rule as the validator
            one = rep.model_copy(update={"claims": [c], "supporting_evidence_ids": [],
                                         "contradicting_evidence_ids": []})
            try:
                validate_report(one, res.evidence, inv_id)
            except ReportValidationError:
                bad += 1
        m["unsupported_claim_rate"] = bad / len(facts) if facts else None
        cited = {e for c in facts for e in c.evidence_ids
                 if e in known and known[e].source_type != "incident_summary"}
        m["evidence_precision"] = (sum(key(known[e]) in rel for e in cited) / len(cited)
                                   if cited else None)
        alt = json.dumps(rep.alternative_hypotheses).lower()
        m["alt_coverage"] = sum(any(k in alt for k in kws)
                                for kws in truth["alternatives"]) / len(truth["alternatives"])
    calls, hits = llm.get("calls", 0), llm.get("cache_hits", 0)
    m.update(
        duration_s=res.extras.get("duration_s"), llm_calls=calls, tool_calls=len(res.trace),
        tokens=llm.get("prompt_tokens", 0) + llm.get("completion_tokens", 0),
        prompt_tokens=llm.get("prompt_tokens", 0),
        completion_tokens=llm.get("completion_tokens", 0), cost="not reported",
        calls_per_round=res.extras.get("calls_per_round"),
        cache_hit_rate=hits / calls if calls else None,
        cache_state="n/a (no LLM call)" if not calls else (
            "cold" if hits == 0 else f"warm ({hits}/{calls} hits)"),
        route=llm.get("route"), resolved_upstream=llm.get("resolved_upstream"),
        validator_rejections=res.extras.get("validator_rejections"),
        response_ids=llm.get("response_ids", []), history=res.extras.get("history"))
    return m


def _stats(values: list) -> dict | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": statistics.fmean(v), "min": min(v), "max": max(v),
            "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}


def summarize(runs: list[dict]) -> dict:
    out: dict = {k: _stats([r[k] for r in runs]) for k in NUMERIC}
    for k in FLAGS:
        out[k] = f"{sum(bool(r[k]) for r in runs)}/{len(runs)}"
    ids = [i for r in runs for i in r["response_ids"] if i]
    out["same_response_id_calls"] = len(ids) - len(set(ids))
    out["cache_states"] = sorted({r["cache_state"] for r in runs})
    return out


def verdict(summary: dict, real: bool, repeats: int) -> str:
    if not real:
        return ("No real model evidence: no OmniRoute run happened (missing OMNIROUTE_API_KEY or "
                "gateway), so every variant ended inconclusive. RQ3 cannot be answered from this "
                "report; rerun with a key.")
    def rate(v, k):
        return int(summary[v][k].split("/")[0]) / repeats
    r1, r2 = rate("R1", "exact_fault"), rate("R2", "exact_fault")
    spread = max((summary[v]["evidence_precision"] or {"sd": 0})["sd"] for v in ("R1", "R2"))
    beats = r2 > r1 and (r2 - r1) > spread
    return (f"On this single F01 incident, exact-fault rate was R0 {rate('R0', 'exact_fault'):.2f}, "
            f"R1 {r1:.2f}, R2 {r2:.2f}, R3 {rate('R3', 'exact_fault'):.2f} over {repeats} cold "
            f"repeats. Multi-agent {'beats' if beats else 'does not clearly beat'} the single "
            "tool-using LLM by more than the repeat spread. One incident and "
            f"{repeats} repeats make every difference indicative, not conclusive. R3 had an "
            "empty history store, so it tests only whether enabling the tool changes behavior.")


async def capture(work: Path) -> list:
    """Run detection and triage once; keep the exact (view, incident, features) per incident."""
    got: list = []

    async def grab(view, incident_id, feats):
        got.append((view, incident_id, feats))
        return InvestigationResult(incident_id, "inconclusive", None, [], [], error="captured")

    await run_pipeline(SnapshotBackend(work / "test"),
                       fit_detectors(SnapshotBackend(work / "train")),
                       make_primary("stub-laya", "cpu"), RulesProvider(), grab)
    return got


async def run_ablation(captured: list, manifest_reader: Callable[[], dict],
                       llm_factory: Callable[[str, int], LLM], variants: tuple = VARIANTS,
                       repeats: int = 3, deadline_s: float = 600.0, real: bool = False) -> dict:
    (view, incident_id, feats), = captured
    raw: dict[str, list[InvestigationResult]] = {v: [] for v in variants}
    budgets: dict = {}
    for v in variants:
        for rep in range(repeats):
            s2 = SystemTwo(llm_factory(v, rep), deadline_s=deadline_s, variant=v)
            budgets[v] = s2.budget()
            raw[v].append(await s2.investigate(view, incident_id, feats))
    if len({json.dumps(b, sort_keys=True) for b in budgets.values()}) != 1:
        raise AssertionError(f"variants got different budgets: {budgets}")
    manifest = manifest_reader()  # only now, after every investigation finished
    per = {v: [metrics_for(r, manifest) for r in rs] for v, rs in raw.items()}
    summary = {v: summarize(runs) for v, runs in per.items()}
    return {
        "notice": ("No real OmniRoute run existed before this ablation. "
                   + ("This is the first real-model comparison." if real else
                      "THIS REPORT CONTAINS NO REAL-MODEL EVIDENCE (no key or gateway).")),
        "real_model": real, "incident": "F01 (1 incident; indicative only)", "repeats": repeats,
        "budget": budgets[variants[0]], "history_store": "empty: R3 returns no past incidents",
        "cache": "one fresh cache dir per (variant, repeat); see cache_state per run",
        "runs": per, "summary": summary, "verdict": verdict(summary, real, repeats)
        if set(variants) == set(VARIANTS) else "partial variant set: no verdict"}


def _fmt(s: dict | None, pct: bool = False) -> str:
    if not s:
        return "n/a"
    f = (lambda x: f"{x:.2f}")
    return f"{f(s['mean'])} ({f(s['min'])}-{f(s['max'])})"


def render_markdown(rep: dict) -> str:
    out = ["# R0-R3 RCA ablation (F01 replay)", "", f"**{rep['notice']}**", "",
           f"- incident: {rep['incident']}; repeats: {rep['repeats']}; budget: {rep['budget']}",
           f"- history: {rep['history_store']}", f"- cache: {rep['cache']}", "",
           ("| variant | exact F01 | category | precision | recall | unsupported | alt cov | "
           "duration s | tokens | LLM calls | tool calls | same-id calls | cache |"),
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, s in rep["summary"].items():
        out.append(f"| {v} | {s['exact_fault']} | {s['category_correct']} | "
                   f"{_fmt(s['evidence_precision'])} | {_fmt(s['evidence_recall'])} | "
                   f"{_fmt(s['unsupported_claim_rate'])} | {_fmt(s['alt_coverage'])} | "
                   f"{_fmt(s['duration_s'])} | {_fmt(s['tokens'])} | {_fmt(s['llm_calls'])} | "
                   f"{_fmt(s['tool_calls'])} | {s['same_response_id_calls']} | "
                   f"{', '.join(s['cache_states'])} |")
    out += ["", "Cells show mean (min-max) over repeats. Cost: not reported by the gateway.", "",
            "## Runs"]
    for v, runs in rep["runs"].items():
        for i, r in enumerate(runs):
            out.append(f"- {v} #{i + 1}: {r['outcome']}"
                       + (f" ({r['error']})" if r["error"] else "")
                       + f"; upstream {r['resolved_upstream']}; calls/round {r['calls_per_round']}; "
                       f"validator rejections {r['validator_rejections']}")
    out += ["", "## Verdict (RQ3, F01 only)", "", rep["verdict"], ""]
    return "\n".join(out)


def _gateway_up(key: str | None) -> bool:
    base = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    try:
        return bool(key) and httpx.get(f"{base}/models", timeout=3,
                                       headers={"Authorization": f"Bearer {key}"}).status_code == 200
    except httpx.HTTPError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("evaluation/work"))
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--deadline", type=float, default=600.0,
                    help="seconds per investigation; cold OmniRoute calls take 39-72 s each")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args(argv)
    if not (a.work / "test" / "meta.json").exists():
        record_all(a.work)
    key = os.getenv("OMNIROUTE_API_KEY")
    real = _gateway_up(key)
    cache_root = a.work / "llm_cache" / "ablation"

    def factory(variant: str, rep: int) -> LLM:
        d = cache_root / f"{variant}-rep{rep + 1}"
        shutil.rmtree(d, ignore_errors=True)  # cold: nothing from any other run can leak in
        return ChatClient(api_key=key, cache_dir=d, timeout_s=120.0)

    async def go() -> dict:
        captured = await capture(a.work)
        reader = GroundTruthStore(a.work / "ground_truth").read
        return await run_ablation(captured, lambda: reader(RUN_ID), factory,
                                  tuple(a.variants.split(",")), a.repeats, a.deadline, real)

    rep = asyncio.run(go())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "r0_r3_ablation.json").write_text(json.dumps(rep, indent=2, default=str))
    (a.out / "r0_r3_ablation.md").write_text(render_markdown(rep))
    print(render_markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
