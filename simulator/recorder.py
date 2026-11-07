"""Run the demo environment in virtual time and record a snapshot (Spec section 10 format).

Snapshot dir: metrics/<name>.jsonl (one file per metric), logs.jsonl, traces.jsonl,
deployments.jsonl, dependencies.json, meta.json, checksums.json. Parquet is skipped for this
slice (no pyarrow); same row shape. Output is byte-identical for the same seed and config.
"""

import asyncio
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from simulator.fault_controller import FaultController, GroundTruthStore
from simulator.services import Gateway, OrderService, PaymentService
from simulator.telemetry import SCRAPE_SECONDS, Collector
from simulator.vtime import run_virtual

EPOCH = datetime(2026, 11, 6, 3, 0, 0, tzinfo=timezone.utc)
SERVICES = ["api-gateway", "order-service", "payment-service"]
DEPENDENCIES = {
    "api-gateway": ["auth-service", "order-service"],
    "auth-service": ["postgres"],
    "order-service": ["payment-service", "postgres", "redis"],
    "payment-service": ["redis", "provider-simulator"],
}
BASE = {
    "api-gateway": {"cpu": 0.10, "cpu_k": 0.006, "mem": 0.35, "deps": True},
    "order-service": {"cpu": 0.12, "cpu_k": 0.012, "mem": 0.48, "deps": True},
    "payment-service": {"cpu": 0.08, "cpu_k": 0.008, "mem": 0.40, "deps": False},
}
# Release record written like a normal release note; old enough to be outside every window.
RELEASES = [
    {"service": "payment-service", "version": "1.8.3", "timestamp": "2026-11-05T21:40:00Z",
     "change_summary": "Bump HTTP client library to 2.31.0 and refresh retry documentation."}
]
SCHEMA_VERSION = 1
_FORBIDDEN = re.compile(r"\b(fault\w*|inject\w*|chaos|scenario\w*|F0[1-6]|ground_truth|"
                        r"simulated_fault)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SimConfig:
    seed: int
    duration_s: int = 1500
    rate_rps: float = 20.0
    write_ratio: float = 0.15
    keyspace: int = 400
    pool_size: int = 10
    # Pool exhaustion activation; onset_s None means a normal-only run.
    onset_s: int | None = None
    hold_s: int = 360
    reduced_pool: int = 3
    held: int = 2


def forbidden_terms(text: str) -> list[str]:
    """Sanitization check (Spec 3.3): leakage terms found in `text`."""
    return sorted({m.group(0).lower() for m in _FORBIDDEN.finditer(text)})


async def _simulate(cfg: SimConfig, store: GroundTruthStore | None, run_id: str) -> Collector:
    rng = random.Random(cfg.seed)
    tel = Collector(SERVICES, rng, EPOCH)
    payment = PaymentService(tel, rng)
    order = OrderService(tel, rng, payment, cfg.pool_size)
    gateway = Gateway(tel, rng, order)
    tel.pool_probe = order.pool.drain
    loop = asyncio.get_running_loop()
    tasks: set[asyncio.Task] = set()

    async def scraper() -> None:
        while True:
            await asyncio.sleep(SCRAPE_SECONDS)
            tel.scrape(loop.time(), BASE)

    async def traffic() -> None:
        while True:
            await asyncio.sleep(rng.expovariate(cfg.rate_rps))
            t = asyncio.create_task(
                gateway.handle(rng.random() < cfg.write_ratio, rng.randrange(cfg.keyspace)))
            tasks.add(t)
            t.add_done_callback(tasks.discard)

    async def controller() -> None:
        if cfg.onset_s is None or store is None:
            return
        fc = FaultController(order, store, run_id, EPOCH)
        await asyncio.sleep(cfg.onset_s)
        fc.activate_pool_exhaustion("order-service", cfg.reduced_pool, cfg.held, cfg.hold_s,
                                    loop.time())
        await asyncio.sleep(cfg.hold_s)
        fc.recover()

    background = [asyncio.create_task(c()) for c in (scraper, traffic, controller)]
    await asyncio.sleep(cfg.duration_s + 0.001)
    for t in background:
        t.cancel()
    for t in list(tasks):
        t.cancel()
    await asyncio.gather(*background, *tasks, return_exceptions=True)
    return tel


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)


def record_snapshot(out_dir: Path, cfg: SimConfig, store: GroundTruthStore | None = None,
                    run_id: str = "run") -> dict:
    """Simulate, sanitize-check, write the snapshot. Returns the checksum manifest."""
    tel = run_virtual(_simulate(cfg, store, run_id))
    files: dict[str, str] = {}
    by_metric: dict[str, list[dict]] = {}
    for row in tel.metrics:
        by_metric.setdefault(row["metric"], []).append(row)
    for name, rows in sorted(by_metric.items()):
        files[f"metrics/{name}.jsonl"] = _jsonl(rows)
    files["logs.jsonl"] = _jsonl(tel.logs)
    files["traces.jsonl"] = _jsonl(tel.spans)
    files["deployments.jsonl"] = _jsonl(RELEASES)
    files["dependencies.json"] = json.dumps(DEPENDENCIES, sort_keys=True, indent=1)
    cfg_dict = asdict(cfg)
    cfg_dict.pop("onset_s"), cfg_dict.pop("hold_s"), cfg_dict.pop("reduced_pool")
    cfg_dict.pop("held")  # activation parameters are not part of what an agent may see
    files["meta.json"] = json.dumps(
        {
            "dataset_version": "demo-v0",
            "schema_version": SCHEMA_VERSION,
            "seed": cfg.seed,
            "generator_config_hash": hashlib.sha256(
                json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:16],
            "host": {"clock": "virtual-time", "traffic": "poisson"},
            "start": tel.iso(0),
            "end": tel.iso(cfg.duration_s),
            "services": SERVICES,
            "scrape_interval_s": SCRAPE_SECONDS,
        },
        sort_keys=True, indent=1)
    bad = {n: forbidden_terms(c) for n, c in files.items() if forbidden_terms(c)}
    if bad:
        raise ValueError(f"sanitization check failed, refusing to write snapshot: {bad}")
    out_dir = Path(out_dir)
    for name, content in files.items():
        p = out_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("ascii"))
    sums = {n: hashlib.sha256(c.encode("ascii")).hexdigest() for n, c in sorted(files.items())}
    (out_dir / "checksums.json").write_text(json.dumps(sums, sort_keys=True, indent=1))
    return sums
