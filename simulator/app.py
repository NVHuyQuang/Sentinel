"""Order Service HTTP app (live mode): /health, /ready, /metrics, /orders/{id}.

    py -3.12 -m uvicorn simulator.app:app --port 8001

Postgres and Redis are in-process stand-ins here; the compose stack's real ones are not wired.
"""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Response

from simulator.services import ConnectionPool, Gateway, OrderService, PaymentService, PoolTimeout
from simulator.telemetry import Collector


@dataclass
class Env:
    tel: Collector
    order: OrderService
    gateway: Gateway

    @property
    def pool(self) -> ConnectionPool:
        return self.order.pool


def build_env(seed: int = 0, pool_size: int = 10) -> Env:
    rng = random.Random(seed)
    tel = Collector(["api-gateway", "order-service", "payment-service"], rng,
                    datetime.now(timezone.utc))
    order = OrderService(tel, rng, PaymentService(tel, rng), pool_size)
    return Env(tel, order, Gateway(tel, rng, order))


def create_order_app(env: Env) -> FastAPI:
    app = FastAPI(title="order-service")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict:
        try:
            await asyncio.wait_for(env.pool.acquire(), 0.25)  # ready = can check out a connection
        except (PoolTimeout, asyncio.TimeoutError):
            raise HTTPException(503, "database connection unavailable") from None
        env.pool.release()
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics() -> Response:
        p = env.pool
        lines = [
            f"db_pool_max_size {p.max_size}",
            f"db_pool_in_use {p.in_use}",
            f"db_pool_utilization {min(1.0, p.in_use / p.max_size):.4f}",
            f"db_pool_waiters {len(p._waiters)}",
        ]
        return Response("\n".join(lines) + "\n", media_type="text/plain")

    @app.get("/orders/{order_id}")
    async def get_order(order_id: int) -> dict:
        tr = env.tel.new_trace()
        if not await env.order.handle(tr, False, order_id % 400):
            raise HTTPException(503, "order lookup failed")
        return {"id": order_id}

    return app


app = create_order_app(build_env())
