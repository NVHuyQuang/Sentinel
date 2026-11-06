"""OmniRoute LLM provider T3 (docs/omniroute-contract.md Section 2).

Structured-output baseline over the local OmniRoute gateway. The system
prompt below is the contract enum-guarded prompt verbatim: without it the
upstream returns human labels like "Performance Degradation"/"High"/95 and
0/20 parse (contract Section 2). temperature is 0 and response_format is
json_object. Cold upstream latency measured 39-72 s, so the default client
timeout is 120 s (above cold P99); the spike 2 s budget only demonstrates
the timeout path.

Auth requires a real OMNIROUTE_API_KEY from the OmniRoute dashboard;
"local-omniroute" is rejected by the gateway (contract Section 1) and is
never hard-coded here. Missing key fails at triage time with ProviderError
and no retry (config error, contract Section 5).

OmniRoute returns no structured tool_calls array for the auto/deepseek
route (free text instead, contract Section 2) and no cost field in usage
(contract Section 4): cost is recorded as not reported via cost_reported
0.0. Never promise tool use in this provider.
"""

import json
import os
import time

import httpx

from services.decision_engine.providers.base import (
    IncidentFeatures,
    ProviderError,
    TriageDecision,
)
from services.decision_engine.providers.laya import render_features

# Verbatim enum-guarded system prompt (omniroute-contract Section 2).
SYSTEM_PROMPT = (
    "You must return exactly one JSON object and nothing else (no markdown, "
    "no extra text). "
    "Required keys: category, severity, priority_score, investigate, "
    "deployment_relevant. "
    "category MUST be exactly one of these 5 strings: "
    "LATENCY, ERROR_SPIKE, RESOURCE, DEPENDENCY, UNKNOWN. "
    "severity MUST be exactly one of these 4 strings: "
    "SEV1, SEV2, SEV3, SEV4. "
    "priority_score MUST be a number between 0 and 1 inclusive "
    "(for example 0.85, never 85 or 95). "
    "investigate and deployment_relevant MUST be JSON true or false. "
    'Example valid output: {"category":"LATENCY","severity":"SEV3",'
    '"priority_score":0.5,"investigate":true,"deployment_relevant":false}. '
    "Never use human labels like 'Performance Degradation', 'High', or "
    "'Critical'."
)

# Verbatim user wrapper (omniroute-contract Section 2); the incident text
# between prefix and suffix is render_features() imported from laya.py.
_USER_PREFIX = (
    "Classify this incident and output only the JSON object. Incident: "
)
_USER_SUFFIX = (
    " JSON keys: category, severity, priority_score, investigate, "
    "deployment_relevant. "
    "category in [LATENCY,ERROR_SPIKE,RESOURCE,DEPENDENCY,UNKNOWN]. "
    "severity in [SEV1,SEV2,SEV3,SEV4]. priority_score 0-1."
)

_DEFAULT_BASE_URL = "http://localhost:20128/v1"
_DEFAULT_MODEL = "auto/deepseek"
_DEFAULT_TIMEOUT_S = 120.0

_CATEGORIES = {"LATENCY", "ERROR_SPIKE", "RESOURCE", "DEPENDENCY", "UNKNOWN"}
_SEVERITIES = {"SEV1", "SEV2", "SEV3", "SEV4"}
_NO_RETRY_STATUS = (400, 401, 403, 404)
_RETRY_STATUS = (429, 500, 502, 503, 504)


def parse_llm_response(raw, latency_ms=None) -> TriageDecision:
    """Parse one OmniRoute chat-completions payload into the 5 decisions.

    Parser rule (omniroute-contract Section 2): json.loads(content.strip()),
    then validate all five keys, Literal sets, priority in [0, 1], and
    booleans for investigate/deployment_relevant. Any miss raises
    ProviderError. Token counts go into raw_scores; cost is not reported
    (contract Section 4), recorded as cost_reported 0.0. model_version is
    the resolved upstream model from raw_response.model.
    """
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"malformed: missing choices content {exc}") from exc
    if not isinstance(content, str):
        raise ProviderError("malformed: message content is not a string")
    try:
        data = json.loads(content.strip())
    except ValueError as exc:
        raise ProviderError(f"malformed: invalid JSON content {exc}") from exc
    if not isinstance(data, dict):
        raise ProviderError("malformed: content JSON is not an object")
    missing = [
        key
        for key in (
            "category",
            "severity",
            "priority_score",
            "investigate",
            "deployment_relevant",
        )
        if key not in data
    ]
    if missing:
        raise ProviderError(f"malformed: missing decision key {missing}")
    category = data["category"]
    severity = data["severity"]
    priority = data["priority_score"]
    investigate = data["investigate"]
    deployment_relevant = data["deployment_relevant"]
    if category not in _CATEGORIES or severity not in _SEVERITIES:
        raise ProviderError(
            f"out-of-vocab choice: category={category!r} severity={severity!r}"
        )
    if isinstance(priority, bool) or not isinstance(priority, (int, float)):
        raise ProviderError(f"malformed: bad priority_score {priority!r}")
    priority_score = float(priority)
    if not 0.0 <= priority_score <= 1.0:
        raise ProviderError(
            f"malformed: priority_score out of range {priority!r}"
        )
    if not isinstance(investigate, bool) or not isinstance(
        deployment_relevant, bool
    ):
        raise ProviderError(
            "malformed: investigate/deployment_relevant must be bool"
        )
    usage = raw.get("usage") if isinstance(raw, dict) else None
    raw_scores = {}
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                raw_scores[key] = float(value)
    has_cost = isinstance(usage, dict) and (
        "cost" in usage or "total_cost" in usage or "price" in usage
    )
    raw_scores["cost_reported"] = 1.0 if has_cost else 0.0
    resolved = raw.get("model") if isinstance(raw, dict) else None
    model_version = resolved if isinstance(resolved, str) and resolved else "unknown"
    return TriageDecision(
        category=category,
        severity=severity,
        priority_score=priority_score,
        investigate=investigate,
        deployment_relevant=deployment_relevant,
        provider="llm",
        model_version=model_version,
        latency_ms=latency_ms,
        raw_scores=raw_scores,
    )


class LLMProvider:
    """DecisionProvider over the OmniRoute /v1/chat/completions gateway."""

    def __init__(
        self,
        api_key=None,
        base_url=None,
        model=None,
        timeout_s=_DEFAULT_TIMEOUT_S,
        client=None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("OMNIROUTE_API_KEY")
        self.base_url = (
            base_url if base_url is not None else os.getenv("LLM_BASE_URL") or _DEFAULT_BASE_URL
        )
        self.model = (
            model if model is not None else os.getenv("LLM_MODEL") or _DEFAULT_MODEL
        )
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        if client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout_s
            )
        else:
            self._client = client

    async def triage(self, features: IncidentFeatures) -> TriageDecision:
        if not self.api_key:
            raise ProviderError(
                "missing OMNIROUTE_API_KEY: config error, no retry"
            )
        user_content = _USER_PREFIX + render_features(features) + _USER_SUFFIX
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        start = time.perf_counter()
        last_error = None
        for attempt in range(2):
            try:
                response = await self._client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _NO_RETRY_STATUS:
                    kind = "invalid_model" if status == 400 else "auth"
                    raise ProviderError(
                        f"llm http {status}: {kind}, no retry"
                    ) from exc
                if status in _RETRY_STATUS and attempt == 0:
                    last_error = exc
                    continue
                raise ProviderError(f"llm http {status}") from exc
            except httpx.TimeoutException as exc:
                if attempt == 0:
                    last_error = exc
                    continue
                raise ProviderError("timeout") from exc
            except httpx.TransportError as exc:
                if attempt == 0:
                    last_error = exc
                    continue
                raise ProviderError(
                    f"llm transport failed: {type(exc).__name__}: {exc}"
                ) from exc
            latency_ms = (time.perf_counter() - start) * 1000.0
            try:
                raw = response.json()
            except ValueError as exc:
                if attempt == 0:
                    last_error = exc
                    continue
                raise ProviderError(
                    f"malformed: invalid JSON body {exc}"
                ) from exc
            try:
                return parse_llm_response(raw, latency_ms=latency_ms)
            except ProviderError as exc:
                if attempt == 1:
                    raise
                last_error = exc
                continue
        raise ProviderError(f"llm failed after retry: {last_error}")
