"""One real OmniRoute run of System Two on the F01 replay. Skipped without a key or gateway.

Cold calls take 39-72 s each (omniroute-contract section 3), so this can take several minutes
on the first run; the response cache (evaluation work dir) makes repeats fast and identical.
"""

import os

import httpx
import pytest

import evaluation.f01_runner as runner

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


async def test_real_llm_investigation_on_f01_replay(work, tmp_path):
    report = await runner.run(work, tmp_path / "r")
    inv = report["investigation"]
    assert inv["llm"]["calls"] >= 1 and inv["llm"]["resolved_upstream"]
    if inv["outcome"] == "investigated":  # a real model may legitimately end inconclusive
        assert report["verdict"]["root_cause_match"] and report["verdict"]["facts_all_linked"]
