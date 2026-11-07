"""Turn raw tool results into short, sanitized observation strings (trusted code, no LLM).

Everything an agent reads about telemetry passes through here. Free text from logs and traces is
attacker-influenced data: it is cut to printable ASCII, one line, 60 chars, with data-fence markers removed, and only event and
span names are quoted, never log messages.
"""

import re

from services.investigation.toolbox import ToolResult

_UNSAFE = re.compile(r"[^ -~]")


def clean(text: object, limit: int = 60) -> str:
    return _UNSAFE.sub("?", str(text)).replace("<<<", "").replace(">>>", "")[:limit]


def _shift(points: list) -> tuple[float, float] | None:
    """(mean before, mean over the last two scrapes). None when there is too little data."""
    vals = [p["value"] for p in points]
    if len(vals) < 6:
        return None
    return sum(vals[:-4]) / len(vals[:-4]), sum(vals[-2:]) / 2


def _top(names: list[str]) -> tuple[str, int] | None:
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]), default=None)


def observe(res: ToolResult, service: str) -> str:
    p = res.params
    if res.tool == "query_metrics":
        shift = _shift(res.data)
        head = f"{service} {p['metric']}"
        if shift is None:
            return f"{head}: not enough samples"
        return f"{head}: mean {shift[0]:.2f} before, {shift[1]:.2f} over the last 30 s"
    if res.tool == "query_logs":
        top = _top([r["event"] for r in res.data])
        lvl = p.get("level") or "any-level"
        if not top:
            return f"{service} logged 0 {lvl} entries"
        return f"{service} logged {len(res.data)} {lvl} entries; most frequent event " \
               f"{clean(top[0])} ({top[1]})"
    if res.tool == "query_traces":
        top = _top([s["name"] for s in res.data])
        kind = "error spans" if p.get("errors_only") else "spans"
        if not top:
            return f"{service} has 0 {kind}"
        return f"{service} has {len(res.data)} {kind}; most frequent span {clean(top[0])} ({top[1]})"
    if res.tool == "get_deployment_events":
        return f"{service} has {len(res.data)} releases in the window"
    if res.tool == "get_service_dependencies":
        return f"{service} depends on {', '.join(clean(d) for d in res.data) or 'nothing'}"
    return f"{len(res.data)} similar past incidents found"
