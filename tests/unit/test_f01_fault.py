"""F01 mechanism: lowered pool + held connections produce the pool-pressure symptoms."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from simulator.app import build_env, create_order_app
from simulator.fault_controller import FaultController, GroundTruthStore, create_controller_app
from simulator.recorder import EPOCH
from simulator.services import ConnectionPool, PoolTimeout
from simulator.vtime import run_virtual


async def _worker(pool: ConnectionPool) -> None:
    while True:
        try:
            await pool.acquire()
        except PoolTimeout:
            continue
        await asyncio.sleep(0.1)
        pool.release()


def _run(pool_size: int, held: int, workers: int, seconds: float):
    async def main():
        pool = ConnectionPool(pool_size)
        for _ in range(held):  # connections held by stuck sessions
            await pool.acquire()
        tasks = [asyncio.create_task(_worker(pool)) for _ in range(workers)]
        await asyncio.sleep(seconds)
        out = pool.drain(asyncio.get_running_loop().time())
        for t in tasks:
            t.cancel()
        return out

    return run_virtual(main())


def test_healthy_pool_is_mostly_idle_with_no_wait():
    util, waits = _run(10, 0, 3, 20)
    assert util < 0.5 and max(waits) == 0.0


def test_pool_pressure_symptoms():
    util, waits = _run(3, 2, 6, 30)
    assert util >= 0.95  # pool utilization symptom
    assert sorted(waits)[int(0.95 * len(waits))] >= 500  # query wait symptom (ms)


def test_activation_lowers_pool_and_holds_connections(tmp_path):
    async def main():
        pool = ConnectionPool(10)
        fc = FaultController(SimpleNamespace(pool=pool), GroundTruthStore(tmp_path), "r", EPOCH)
        fc.activate_pool_exhaustion("order-service", 3, 2, 1000.0, 0.0)
        await asyncio.sleep(0)  # holders check out their connections
        state = (pool.max_size, pool.in_use)
        for t in fc._holders:
            t.cancel()
        await asyncio.gather(*fc._holders, return_exceptions=True)
        return state

    assert run_virtual(main()) == (3, 2)


def test_unsupported_target_rejected(tmp_path):
    fc = FaultController(SimpleNamespace(pool=None), GroundTruthStore(tmp_path), "r", EPOCH)
    with pytest.raises(ValueError):
        fc.activate_pool_exhaustion("auth-service", 3, 2, 10.0, 0.0)


async def test_http_controller_drives_order_service_symptoms(tmp_path):
    env = build_env(pool_size=10)
    fc = FaultController(env.order, GroundTruthStore(tmp_path / "gt"), "r", EPOCH)
    order = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_order_app(env)),
                              base_url="http://o")
    ctl = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_controller_app(fc)),
                            base_url="http://c")
    assert (await order.get("/health")).status_code == 200
    assert (await order.get("/ready")).status_code == 200
    r = await ctl.post("/activate", json={"pool_size": 2, "hold": 2, "hold_seconds": 0.5})
    assert r.status_code == 200
    await asyncio.sleep(0.05)
    assert "db_pool_utilization 1.0000" in (await order.get("/metrics")).text
    assert (await order.get("/ready")).status_code == 503  # cannot check out a connection
    assert (await ctl.post("/activate", json={"target": "auth-service"})).status_code == 422
    await asyncio.sleep(0.6)
    await ctl.post("/recover")
    assert (await order.get("/ready")).status_code == 200
    assert "db_pool_utilization 0.0000" in (await order.get("/metrics")).text


def test_manifest_goes_to_private_store(work):
    m = GroundTruthStore(work / "ground_truth").read("f01-demo")
    assert m["fault"] == "db_connection_pool_exhaustion"
    assert m["onset"] < m["recovery"] and m["causal_chain"] and m["impact_level"]
    assert not any("ground_truth" in str(p) for p in (work / "test").rglob("*"))
