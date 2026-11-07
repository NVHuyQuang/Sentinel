"""OmniRoute chat client for the System Two agents, with a response cache and run metadata.

One JSON-object chat call per agent node. temperature 0. Any transport, auth or output problem
raises ProviderError; the graph then ends the investigation as inconclusive (no canned answer).

Cache: replies are stored on disk keyed by sha256(model, params, messages). A cache hit
returns the same text with the original resolved model and token usage, so repeated runs are
identical. That is cache-assisted determinism, not model determinism: benchmark each variant
3x and read the spread only on cold (uncached) runs. Cold upstream latency is 39-72 s
(omniroute-contract section 3), so the default timeout is 120 s. Cost is not reported by the
gateway and is never estimated here.
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from services.decision_engine.providers.base import ProviderError

DEFAULT_BASE_URL = "http://localhost:20128/v1"
DEFAULT_MODEL = "auto/deepseek"
DEFAULT_TIMEOUT_S = 120.0


@dataclass
class LLMReply:
    text: str
    model: str  # route asked for
    resolved_model: str  # upstream that answered (raw_response.model)
    usage: dict
    latency_ms: float
    cached: bool = False


class LLM(Protocol):
    async def complete(self, system: str, user: str) -> LLMReply: ...


@dataclass
class RunMeta:
    """Experiment metadata (Spec 6.1): route, resolved upstreams, tokens, cache hits."""

    route: str = ""
    resolved: set = field(default_factory=set)
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_ms: float = 0.0

    def add(self, r: LLMReply) -> None:
        self.route = r.model
        self.resolved.add(r.resolved_model)
        self.calls += 1
        self.cache_hits += int(r.cached)
        self.prompt_tokens += int(r.usage.get("prompt_tokens", 0))
        self.completion_tokens += int(r.usage.get("completion_tokens", 0))
        self.llm_latency_ms += r.latency_ms

    def as_dict(self) -> dict:
        return {"route": self.route, "resolved_upstream": sorted(self.resolved),
                "temperature": 0, "calls": self.calls, "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "llm_latency_ms": round(self.llm_latency_ms, 1), "cost": "not reported"}


def parse_json_object(text: str) -> dict:
    """The reply must be one JSON object. Fenced or chatty replies are tolerated, nothing more."""
    body = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise ProviderError(f"malformed: reply is not JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ProviderError("malformed: reply JSON is not an object")
    return data


class ChatClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S,
                 cache_dir: Path | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OMNIROUTE_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def complete(self, system: str, user: str) -> LLMReply:
        payload = {"model": self.model, "temperature": 0,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        path = self.cache_dir / f"{self._key(payload)}.json" if self.cache_dir else None
        if path and path.exists():
            hit = json.loads(path.read_text())
            return LLMReply(**{**hit, "latency_ms": 0.0, "cached": True})
        if not self.api_key:
            raise ProviderError("missing OMNIROUTE_API_KEY: config error, no retry")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        start = time.perf_counter()
        raw = None
        for attempt in range(2):
            try:
                r = await self._client.post(url, json=payload,
                                            headers={"Authorization": f"Bearer {self.api_key}"})
                if r.status_code in (400, 401, 403, 404):
                    raise ProviderError(f"llm http {r.status_code}, no retry")
                r.raise_for_status()
                raw = r.json()
                break
            except ProviderError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                if attempt == 1:
                    raise ProviderError(f"llm unavailable: {type(exc).__name__}: {exc}") from exc
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed: missing choices content ({exc})") from exc
        if not isinstance(text, str):
            raise ProviderError("malformed: content is not a string")
        reply = LLMReply(text=text, model=self.model, resolved_model=str(raw.get("model", "unknown")),
                         usage=raw.get("usage") or {},
                         latency_ms=(time.perf_counter() - start) * 1000)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**reply.__dict__, "cached": False}))
        return reply
