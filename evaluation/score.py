"""Compare a finished run against the hidden manifest. Evaluator-only; agents never import this."""

from datetime import datetime

from services.pipeline import RunResult
from services.telemetry.backend import SnapshotBackend, parse_ts

_RANK = {"SEV1": 4, "SEV2": 3, "SEV3": 2, "SEV4": 1}


def _secs(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds()


def _chain(manifest: dict, backend: SnapshotBackend, onset: datetime) -> list[dict]:
    full = backend.at(backend.end)
    out = []
    for st in manifest["causal_chain"]:
        pts = full.query_metrics(st["service"], st["metric"], backend.start, backend.end)
        hit = next((parse_ts(p["ts"]) for p in pts
                    if parse_ts(p["ts"]) >= onset and p["value"] >= st["min"]), None)
        out.append({"stage": st["stage"], "first_seen": hit.isoformat() if hit else None,
                    "after_onset_s": _secs(hit, onset) if hit else None})
    return out


def evaluate(manifest: dict, backend: SnapshotBackend, run: RunResult, primary: str) -> dict:
    onset, recovery = parse_ts(manifest["onset"]), parse_ts(manifest["recovery"])
    exp = manifest["expected"]
    affected = set(manifest["affected_services"])
    pre = [i for i in run.incidents if i.opened_at < onset]
    hit = next((i for i in run.incidents if i.opened_at >= onset
                and affected & set(i.affected_services)), None)
    chain = _chain(manifest, backend, onset)
    seen = [c["after_onset_s"] for c in chain]
    ordered = None not in seen and seen == sorted(seen)
    rep: dict = {"primary": primary, "onset": manifest["onset"], "recovery": manifest["recovery"],
                 "false_positive_incidents_before_onset": len(pre),
                 "incident_count": len(run.incidents), "chain": chain,
                 "chain_order_matches": ordered}
    verdict = {"detected": hit is not None, "no_false_positives": not pre,
               "chain_order_matches": ordered}
    advisory: dict[str, bool] = {}  # reported, not gating: depends on which provider answered
    if hit:
        dec, inv = run.triage[hit.incident_id], run.investigations.get(hit.incident_id)
        rep["detection_delay_s"] = _secs(hit.opened_at, onset)
        rep["incident"] = {"id": hit.incident_id, "affected": hit.affected_services,
                           "status": hit.status,
                           "resolved_at": hit.resolved_at.isoformat() if hit.resolved_at else None}
        rep["triage"] = dec.model_dump()
        advisory["category_match"] = dec.category == exp["category"]
        verdict["severity_ok"] = _RANK[dec.severity] >= _RANK[exp["min_severity"]]
        verdict["resolved_after_recovery"] = bool(hit.resolved_at and hit.resolved_at > recovery)
        verdict["closed"] = hit.status == "closed"
        if hit.resolved_at:
            rep["resolve_delay_after_recovery_s"] = _secs(hit.resolved_at, recovery)
        if inv:
            rep["investigation"] = {"outcome": inv.outcome, "error": inv.error,
                                    "tool_calls": len(inv.trace), **inv.extras}
        rca = inv.report if inv else None
        verdict["rca_produced"] = rca is not None
        if rca:
            text = (rca.probable_root_cause or "").lower()
            facts = [c for c in rca.claims if c.kind == "fact"]
            cited = {e for c in facts for e in c.evidence_ids}
            rel = {tuple(x) for x in exp["relevant_signals"]}
            by_id = {e.id: e for e in inv.evidence}
            def key(e):
                return (e.service, e.query_parameters.get("metric") or e.source_type)

            good = [e for e in cited if key(by_id[e]) in rel]
            retrieved = {key(e) for e in inv.evidence}
            rep["rca"] = {"root_cause": rca.probable_root_cause, "confidence": rca.confidence_level,
                          "claims": len(rca.claims), "fact_claims": len(facts),
                          "evidence_precision": len(good) / len(cited) if cited else 0.0,
                          "evidence_recall": len(retrieved & rel) / len(rel),
                          "evidence_retrieved": len(inv.evidence),
                          "tool_calls": len(inv.trace), "marker": inv.marker,
                          "recommendations": rca.recommendations}
            verdict["root_cause_match"] = all(k.lower() in text for k in exp["root_cause_keywords"])
            verdict["facts_all_linked"] = all(c.evidence_ids for c in facts)
            advisory["evidence_precision_1"] = rep["rca"]["evidence_precision"] == 1.0
    verdict["all_pass"] = all(verdict.values())
    rep["verdict"] = verdict
    rep["advisory"] = advisory
    rep["execution_trace"] = run.trace
    return rep


def render_markdown(r: dict) -> str:
    v = r["verdict"]
    lines = ["# F01 vertical slice report", "",
             f"- primary provider requested: {r['primary']}",
             f"- onset {r['onset']}, recovery {r['recovery']}",
             (f"- incidents: {r['incident_count']} (before onset: "
              f"{r['false_positive_incidents_before_onset']})"),
             f"- detection delay: {r.get('detection_delay_s')} s", ""]
    if "triage" in r:
        t = r["triage"]
        lines.append(f"- triage: {t['category']} / {t['severity']} via {t['provider']} "
                     f"(investigate={t['investigate']})")
    if "rca" in r:
        a = r["rca"]
        lines.append(f"- RCA ({a['marker']}): {a['root_cause']} [{a['confidence']}], "
                     f"{a['fact_claims']} fact claims, "
                     f"evidence precision {a['evidence_precision']:.2f} recall "
                     f"{a['evidence_recall']:.2f}, "
                     f"{a['tool_calls']} tool calls")
    if "investigation" in r:
        i = r["investigation"]
        lines.append(f"- investigation: {i['outcome']}" + (f" ({i['error']})" if i["error"] else "")
                     + f", {i.get('duration_s')} s, calls/round {i.get('calls_per_round')}, "
                     f"llm {i.get('llm')}")
    lines += ["", "## Causal chain (observed, seconds after onset)"]
    lines += [f"- {c['stage']}: {c['after_onset_s']}" for c in r["chain"]]
    lines += ["", "## Verdict (gating)"] + [f"- {k}: {'PASS' if ok else 'FAIL'}" for k, ok in v.items()]
    lines += ["", "## Advisory (provider-dependent)"]
    lines += [f"- {k}: {'PASS' if ok else 'FAIL'}" for k, ok in r["advisory"].items()]
    return "\n".join(lines) + "\n"
