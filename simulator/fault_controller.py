"""Fault Controller (harness, not a monitored service): activates the DB pool exhaustion scenario.

Mechanism only, no labelling: the pool max is lowered (a config change) and a few connections
are held (a stuck batch job / leaked transactions). The service then shows its natural
symptoms. The manifest goes to a GroundTruthStore that no agent tool is given. The controller
emits no telemetry of its own.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from simulator.services import OrderService

# Expected symptom chain, used only by the evaluator after the investigation is done.
CAUSAL_CHAIN = [
    {"stage": "pool_at_capacity", "service": "order-service", "metric": "db_pool_utilization",
     "min": 0.9},
    {"stage": "query_wait_rises", "service": "order-service", "metric": "db_query_wait_ms_p95",
     "min": 500.0},
    {"stage": "order_latency_up", "service": "order-service", "metric": "latency_p95_ms",
     "min": 1000.0},
    {"stage": "gateway_5xx", "service": "api-gateway", "metric": "error_rate", "min": 0.05},
]


class GroundTruthStore:
    """Private manifest store. Lives outside any snapshot directory and outside tool reach."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write(self, run_id: str, manifest: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{run_id}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def read(self, run_id: str) -> dict:
        return json.loads((self.root / f"{run_id}.json").read_text())


class FaultController:
    TARGETS = ("order-service",)  # Auth DB pool is the allowed alternate; not simulated here

    def __init__(self, order: OrderService, store: GroundTruthStore, run_id: str,
                 epoch: datetime) -> None:
        self.order, self.store, self.run_id, self.epoch = order, store, run_id, epoch
        self._saved_size: int | None = None
        self._holders: list[asyncio.Task] = []
        self._manifest: dict = {}

    def _at(self, t: float) -> str:
        return (self.epoch + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")

    async def _hold(self, seconds: float) -> None:
        await self.order.pool.acquire()  # connection held by a stuck session
        try:
            await asyncio.sleep(seconds)
        finally:
            self.order.pool.release()

    def activate_pool_exhaustion(self, target: str, pool_size: int, hold: int,
                                 hold_seconds: float, now: float) -> None:
        if target not in self.TARGETS:
            raise ValueError(f"target {target!r} not supported in this slice")
        self._saved_size = self.order.pool.max_size
        self.order.pool.resize(pool_size)
        self._holders = [asyncio.create_task(self._hold(hold_seconds)) for _ in range(hold)]
        self._manifest = {
            "fault": "db_connection_pool_exhaustion",
            "target": target,
            "onset": self._at(now),
            "recovery": self._at(now + hold_seconds),
            "affected_services": ["order-service", "api-gateway"],
            "root_cause_service": target,
            "causal_chain": CAUSAL_CHAIN,
            "impact_level": "major",
            "expected": {
                "category": "RESOURCE",
                "min_severity": "SEV2",
                "root_cause_keywords": ["connection pool", target],
                "relevant_signals": [
                    ["order-service", "db_pool_utilization"],
                    ["order-service", "db_query_wait_ms_p95"],
                    ["order-service", "latency_p95_ms"],
                    ["order-service", "error_rate"],
                    ["api-gateway", "error_rate"],
                    # Evidence that rules out alternatives or corroborates also counts.
                    ["order-service", "cpu_utilization"],
                    ["order-service", "query_logs"],
                    ["order-service", "query_traces"],
                    ["order-service", "get_deployment_events"],
                ],
            },
        }
        self.store.write(self.run_id, self._manifest)

    def recover(self) -> None:
        if self._saved_size is not None:
            self.order.pool.resize(self._saved_size)
            self._saved_size = None


def create_controller_app(fc: FaultController):
    """HTTP wrapper for live mode. Separate process/port from every monitored service."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Body(BaseModel):
        target: str = "order-service"
        pool_size: int = 3
        hold: int = 2
        hold_seconds: float = 360.0

    app = FastAPI(title="controller")

    @app.post("/activate")
    async def activate(body: Body) -> dict:
        loop = asyncio.get_running_loop()
        try:
            fc.activate_pool_exhaustion(body.target, body.pool_size, body.hold,
                                        body.hold_seconds, loop.time())
        except ValueError as exc:
            from fastapi import HTTPException

            raise HTTPException(422, str(exc)) from None
        return {"ok": True}

    @app.post("/recover")
    async def recover() -> dict:
        fc.recover()
        return {"ok": True}

    return app
