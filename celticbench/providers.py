"""Hosted chat adapters and live model-list doctor probes.

Adapters are stateless and safe to call from the runner's thread pool. Retries,
pacing, and caching live in ``runner.py``; every adapter raises ``ProviderError``
on failure so the runner owns retry policy.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import certifi
import requests

from .prompt import clean_output
from .registry import credential_envs

TIMEOUT = 90


class ProviderError(Exception):
    """One failed provider call; message is safe to print."""


@dataclasses.dataclass(frozen=True)
class Translation:
    """One provider answer plus auditable receipt metadata."""

    text: str
    model_reported: str
    decoding_used: dict[str, Any]
    usage: dict[str, int]
    deviations: tuple[str, ...] = ()


def _key(entry: dict[str, Any]) -> str:
    env = entry["key_env"]
    key = os.environ.get(env, "")
    if not key:
        raise ProviderError(f"missing API key: set {env} in .env")
    return key


def _post(url: str, *, headers: dict[str, str] | None = None,
          params: dict[str, str] | None = None, body: Any = None) -> Any:
    try:
        response = requests.post(
            url, headers=headers, params=params, json=body,
            timeout=TIMEOUT, verify=certifi.where(),
        )
    except requests.RequestException as exc:
        raise ProviderError(f"network: {type(exc).__name__}: {str(exc)[:120]}") from exc
    if response.status_code != 200:
        raise ProviderError(f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response: {response.text[:120]}") from exc


def _get(url: str, *, headers: dict[str, str] | None = None,
         params: dict[str, str] | None = None) -> Any:
    try:
        response = requests.get(
            url, headers=headers, params=params,
            timeout=TIMEOUT, verify=certifi.where(),
        )
    except requests.RequestException as exc:
        raise ProviderError(f"network: {type(exc).__name__}: {str(exc)[:120]}") from exc
    if response.status_code != 200:
        raise ProviderError(f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response: {response.text[:120]}") from exc


def _decoding_sent(entry: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Apply a declared provider deviation without concealing it from receipts."""
    decoding = dict(entry["decoding"])
    deviation = entry.get("decoding_deviation")
    if not deviation:
        return decoding, ()
    omitted = tuple(deviation["omits"])
    for name in omitted:
        decoding.pop(name, None)
    return decoding, (f"omitted {', '.join(omitted)}: {deviation['reason']}",)


def _tokens(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _usage(prompt: Any, completion: Any, total: Any = None) -> dict[str, int]:
    prompt_tokens = _tokens(prompt)
    completion_tokens = _tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": _tokens(total) or prompt_tokens + completion_tokens,
    }


_SAMPLING_FIELDS = ("temperature", "top_p", "max_tokens")


def translate_openai_compat(entry: dict[str, Any], prompt: str) -> Translation:
    decoding, deviations = _decoding_sent(entry)
    body: dict[str, Any] = {
        "model": entry["model"],
        "messages": [{"role": "user", "content": prompt}],
        **decoding,
    }
    headers = {"Authorization": f"Bearer {_key(entry)}"}
    url = entry["base_url"].rstrip("/") + "/chat/completions"
    try:
        data = _post(url, headers=headers, body=body)
    except ProviderError as exc:
        message = str(exc)
        if "HTTP 400" not in message or not any(field in message for field in _SAMPLING_FIELDS):
            raise
        dropped = [field for field in ("temperature", "top_p")
                   if body.pop(field, None) is not None]
        if "max_tokens" in message and "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
            decoding = {key: value for key, value in decoding.items() if key != "max_tokens"}
            dropped.append("max_tokens (resent as max_completion_tokens)")
        decoding = {key: value for key, value in decoding.items() if key in body}
        deviations += (f"vendor rejected {', '.join(dropped)} with HTTP 400; "
                       "retried without them",)
        data = _post(url, headers=headers, body=body)
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"empty choices: {json.dumps(data)[:200]}")
    content = (choices[0].get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    return Translation(
        text=clean_output(content),
        model_reported=str(data.get("model", entry["model"])),
        decoding_used=decoding,
        usage=_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                     usage.get("total_tokens")),
        deviations=deviations,
    )


def translate_anthropic(entry: dict[str, Any], prompt: str) -> Translation:
    decoding, deviations = _decoding_sent(entry)
    body = {
        "model": entry["model"],
        "messages": [{"role": "user", "content": prompt}],
        **decoding,
    }
    data = _post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": _key(entry),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        body=body,
    )
    blocks = data.get("content") or []
    text = "".join(block.get("text", "") for block in blocks
                   if block.get("type") == "text")
    if not text and blocks:
        text = blocks[-1].get("text", "")
    usage = data.get("usage") or {}
    return Translation(
        text=clean_output(text),
        model_reported=str(data.get("model", entry["model"])),
        decoding_used=decoding,
        usage=_usage(usage.get("input_tokens"), usage.get("output_tokens")),
        deviations=deviations,
    )


_GEMINI_CONFIG = {
    "temperature": "temperature",
    "top_p": "topP",
    "max_tokens": "maxOutputTokens",
}


def translate_gemini(entry: dict[str, Any], prompt: str) -> Translation:
    decoding, deviations = _decoding_sent(entry)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{entry['model']}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            _GEMINI_CONFIG[key]: value for key, value in decoding.items()
            if key in _GEMINI_CONFIG
        },
    }
    data = _post(
        url, params={"key": _key(entry)}, body=body,
        headers={"Content-Type": "application/json"},
    )
    candidate = (data.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    usage = data.get("usageMetadata") or {}
    return Translation(
        text=clean_output(text),
        model_reported=str(data.get("modelVersion", entry["model"])),
        decoding_used=decoding,
        usage=_usage(usage.get("promptTokenCount"), usage.get("candidatesTokenCount"),
                     usage.get("totalTokenCount")),
        deviations=deviations,
    )


_ADAPTERS = {
    "openai_compat": translate_openai_compat,
    "anthropic": translate_anthropic,
    "gemini": translate_gemini,
}


def translate_chat(entry: dict[str, Any], prompt: str) -> Translation:
    adapter = _ADAPTERS.get(entry["provider"])
    if adapter is None:
        raise ProviderError(f"{entry['provider']!r} is not a hosted chat provider")
    return adapter(entry, prompt)


def list_models(entry: dict[str, Any]) -> list[str]:
    """Return live model IDs from one hosted chat provider."""
    provider = entry["provider"]
    if provider == "openai_compat":
        data = _get(
            entry["base_url"].rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {_key(entry)}"},
        )
        return sorted(str(item.get("id", "")) for item in data.get("data", []))
    if provider == "anthropic":
        data = _get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": _key(entry), "anthropic-version": "2023-06-01"},
        )
        return sorted(str(item.get("id", "")) for item in data.get("data", []))
    if provider == "gemini":
        data = _get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": _key(entry), "pageSize": "1000"},
        )
        return sorted(
            str(item.get("name", "")).removeprefix("models/")
            for item in data.get("models", [])
        )
    raise ProviderError(f"no model-list doctor probe for provider {provider!r}")


def doctor_check(system_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Check credentials and the pinned hosted model ID; never raise."""
    result: dict[str, Any] = {
        "system": system_id,
        "provider": entry["provider"],
        "model": entry["model"],
        "pin_status": entry["pin_status"],
    }
    missing_envs = [env for env in credential_envs(entry) if not os.environ.get(env, "")]
    if missing_envs:
        result["status"] = "MISSING_KEY"
        result["detail"] = f"set {', '.join(missing_envs)} in .env"
        return result
    try:
        models = list_models(entry)
    except ProviderError as exc:
        result["status"] = "AUTH_OR_NETWORK_FAIL"
        result["detail"] = str(exc)
        return result
    if entry["model"] in models:
        result["status"] = "OK"
        result["detail"] = "pinned model ID confirmed against live model list"
        return result
    hint = entry.get("family_hint", "")
    suggestions = [model for model in models if hint and hint in model.lower()][:8]
    result["status"] = "MODEL_NOT_FOUND"
    result["detail"] = (
        f"pinned ID not in live list; similar: {suggestions}"
        if suggestions else "pinned ID not in live list; inspect full list"
    )
    return result
