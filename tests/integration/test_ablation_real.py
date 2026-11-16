"""One real OmniRoute R0 run on the F01 replay. Skipped without a key or gateway."""

import os

import httpx
import pytest

from evaluation.ablation import capture, metrics_for
from evaluation.f01_runner import RUN_ID
from services.investigation.graph import SystemTwo
from services.investigation.llm import ChatClient
from simulator.fault_controller import GroundTruthStore

KEY = os.getenv("OMNIROUTE_API_KEY")
BASE = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")


def _gateway_up() -> bool:
    try:
        return httpx.get(f"{BASE}/models", headers={"Authorization": f"Bearer {KEY}"},
                         timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not (KEY and _gateway_up()),
                                reason="needs a real OMNIROUTE_API_KEY and a running gateway")


async def test_real_r0_single_call_no_tools(work, tmp_path):
    ((view, incident_id, feats),) = await capture(work)
    s2 = SystemTwo(ChatClient(cache_dir=tmp_path / "cold"), deadline_s=600.0, variant="R0")
    res = await s2.investigate(view, incident_id, feats)
    m = metrics_for(res, GroundTruthStore(work / "ground_truth").read(RUN_ID))
    assert m["llm_calls"] == 1 and m["tool_calls"] == 0 and m["cache_state"] == "cold"
    assert m["resolved_upstream"] and m["cost"] == "not reported"
