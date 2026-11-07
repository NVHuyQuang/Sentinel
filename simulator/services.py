"""Gateway, Order and Payment services over a Postgres-like pool and a Redis-like cache.

Postgres and Redis are in-process stand-ins (latency only); real drivers are out of this slice.
Order reads hit the cache first and fall back to the DB, which is where pool pressure shows up.
"""

import asyncio
import random
from collections import deque

from simulator.telemetry import Collector, Trace

DB_QUERY_S = 0.100
ACQUIRE_TIMEOUT_S = 2.0
CACHE_TTL_S = 8.0
BACKGROUND_ERROR = 0.003


class PoolTimeout(Exception):
    pass


class ConnectionPool:
    """FIFO connection pool. Utilization is time-averaged between scrapes, like a gauge average."""

    def __init__(self, max_size: int, acquire_timeout: float = ACQUIRE_TIMEOUT_S) -> None:
        self.max_size = max_size
        self.acquire_timeout = acquire_timeout
        self.in_use = 0
        self._waiters: deque = deque()
        self._area = 0.0
        self._last = 0.0
        self._since = 0.0
        self.waits_ms: list[float] = []

    def _tick(self) -> None:
        now = asyncio.get_running_loop().time()
        self._area += min(1.0, self.in_use / self.max_size) * (now - self._last)
        self._last = now

    def drain(self, now: float) -> tuple[float, list[float]]:
        self._tick()
        span = max(now - self._since, 1e-9)
        util, waits = self._area / span, self.waits_ms
        self._area, self._since, self.waits_ms = 0.0, now, []
        return util, waits

    async def acquire(self) -> float:
        loop = asyncio.get_running_loop()
        start = loop.time()
        self._tick()
        if self.in_use < self.max_size:
            self.in_use += 1
            self.waits_ms.append(0.0)
            return 0.0
        fut = loop.create_future()
        self._waiters.append(fut)
        try:
            await asyncio.wait_for(fut, self.acquire_timeout)
        except asyncio.TimeoutError:
            if fut in self._waiters:
                self._waiters.remove(fut)
            self.waits_ms.append((loop.time() - start) * 1000)
            raise PoolTimeout from None
        waited = loop.time() - start
        self.waits_ms.append(waited * 1000)
        return waited

    def release(self) -> None:
        self._tick()
        self.in_use -= 1
        self._wake()

    def resize(self, max_size: int) -> None:
        self._tick()
        self.max_size = max_size
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self.in_use < self.max_size:
            fut = self._waiters.popleft()
            if not fut.done():
                self.in_use += 1
                fut.set_result(None)


class Cache:
    def __init__(self) -> None:
        self._d: dict[int, float] = {}

    def get(self, key: int, now: float) -> bool:
        return key in self._d and now - self._d[key] < CACHE_TTL_S

    def set(self, key: int, now: float) -> None:
        self._d[key] = now


class PaymentService:
    name = "payment-service"

    def __init__(self, tel: Collector, rng: random.Random) -> None:
        self.tel, self.rng = tel, rng

    async def charge(self, tr: Trace) -> bool:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.sleep(0.04 * self.rng.lognormvariate(0, 0.2))
        ok = self.rng.random() > BACKGROUND_ERROR
        dur = loop.time() - t0
        self.tel.span(tr, self.name, "POST /charge", t0, dur, "OK" if ok else "ERROR")
        if not ok:
            self.tel.log(loop.time(), self.name, "ERROR", tr, "provider_declined", dur,
                         "payment provider declined the charge")
        self.tel.request(self.name, dur, ok)
        return ok


class OrderService:
    name = "order-service"

    def __init__(self, tel: Collector, rng: random.Random, payment: PaymentService,
                 pool_size: int = 10) -> None:
        self.tel, self.rng, self.payment = tel, rng, payment
        self.pool = ConnectionPool(pool_size)
        self.cache = Cache()

    async def _db(self, tr: Trace) -> tuple[bool, bool]:
        """Returns (ok, dependency_timeout)."""
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            waited = await self.pool.acquire()
        except PoolTimeout:
            dur = loop.time() - t0
            self.tel.span(tr, self.name, "postgres.acquire", t0, dur, "ERROR",
                          pool_wait_ms=round(dur * 1000))
            self.tel.log(loop.time(), self.name, "ERROR", tr, "db_pool_acquire_timeout", dur,
                         "timed out waiting for a database connection")
            return False, True
        try:
            q0 = loop.time()
            await asyncio.sleep(DB_QUERY_S * self.rng.lognormvariate(0, 0.25))
            self.tel.span(tr, self.name, "postgres.query", q0, loop.time() - q0,
                          pool_wait_ms=round(waited * 1000))
        finally:
            self.pool.release()
        if waited > 0.5:
            self.tel.log(loop.time(), self.name, "WARN", tr, "slow_db_checkout", waited,
                         "database connection checkout was slow")
        return True, False

    async def handle(self, tr: Trace, write: bool, key: int) -> bool:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        ok, dep_timeout = True, False
        if write:
            ok, dep_timeout = await self._db(tr)
            if ok:
                ok = await self.payment.charge(tr)
        else:
            await asyncio.sleep(0.001)
            if self.cache.get(key, t0):
                self.tel.span(tr, self.name, "redis.get", t0, 0.001, hit=True)
            else:
                ok, dep_timeout = await self._db(tr)
                if ok:
                    self.cache.set(key, loop.time())
        if ok and self.rng.random() < BACKGROUND_ERROR:
            ok = False
            self.tel.log(loop.time(), self.name, "ERROR", tr, "constraint_violation", 0.0,
                         "database constraint violation")
        dur = loop.time() - t0
        self.tel.span(tr, self.name, "POST /orders" if write else "GET /orders/{id}", t0, dur,
                      "OK" if ok else "ERROR")
        self.tel.request(self.name, dur, ok, dep_timeout)
        return ok


class Gateway:
    name = "api-gateway"
    UPSTREAM_TIMEOUT_S = 3.0

    def __init__(self, tel: Collector, rng: random.Random, order: OrderService) -> None:
        self.tel, self.rng, self.order = tel, rng, order

    async def handle(self, write: bool, key: int) -> None:
        loop = asyncio.get_running_loop()
        tr = self.tel.new_trace()
        t0 = loop.time()
        try:
            ok = await asyncio.wait_for(self.order.handle(tr, write, key), self.UPSTREAM_TIMEOUT_S)
        except asyncio.TimeoutError:
            ok = False
        dur = loop.time() - t0
        self.tel.span(tr, self.name, "POST /orders" if write else "GET /orders/{id}", t0, dur,
                      "OK" if ok else "ERROR")
        if not ok:
            self.tel.log(loop.time(), self.name, "ERROR", tr, "upstream_error", dur,
                         "upstream order-service returned 5xx")
        self.tel.request(self.name, dur, ok)
        self.tel.finish(tr)
