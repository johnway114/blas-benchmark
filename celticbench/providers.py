"""Hosted-API provider adapters and doctor probes.

Plain REST via requests; no vendor SDKs. Each translate_* function takes a
registry entry and returns (text, model_reported, decoding_used). Retries and
pacing live in runner.py; adapters raise ProviderError on any failure so the
runner can decide.
"""
from __future__ import annotations

import html
import json
import os
from typing import Any

import certifi
import requests

from .prompt import clean_output

TIMEOUT = 90


class ProviderError(Exception):
    """One failed provider call; message is safe to print."""


def _key(entry: dict[str, Any]) -> str:
    env = entry.get("key_env")
    key = os.environ.get(env or "", "")
    if not key:
        raise ProviderError(f"missing API key: set {env} in .env")
    return key


def _post(url: str, *, headers: dict[str, str] | None = None,
          params: dict[str, str] | None = None, body: dict[str, Any]) -> dict[str, Any]:
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
         params: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        response = requests.get(url, headers=headers, params=params,
                                timeout=TIMEOUT, verify=certifi.where())
    except requests.RequestException as exc:
        raise ProviderError(f"network: {type(exc).__name__}: {str(exc)[:120]}") from exc
    if response.status_code != 200:
        raise ProviderError(f"HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


# ---------------------------------------------------------------------------
# Translation calls
# ---------------------------------------------------------------------------

def translate_openai_compat(entry: dict[str, Any], prompt: str) -> tuple[str, str, dict[str, Any]]:
    decoding = dict(entry["decoding"])
    body: dict[str, Any] = {
        "model": entry["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "max_tokens": decoding["max_tokens"],
    }
    headers = {"Authorization": f"Bearer {_key(entry)}"}
    url = entry["base_url"].rstrip("/") + "/chat/completions"
    try:
        data = _post(url, headers=headers, body=body)
    except ProviderError as exc:
        # Some reasoning-tier models reject sampling parameters outright.
        # Retry once without them and record the deviation in the receipt.
        message = str(exc)
        if "HTTP 400" in message and ("temperature" in message or "top_p" in message or "max_tokens" in message):
            for field in ("temperature", "top_p"):
                body.pop(field, None)
            if "max_tokens" in message:
                body["max_completion_tokens"] = body.pop("max_tokens")
            data = _post(url, headers=headers, body=body)
            decoding = {k: v for k, v in body.items() if k not in ("model", "messages")}
            decoding["adjusted"] = True
        else:
            raise
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"empty choices: {json.dumps(data)[:200]}")
    content = (choices[0].get("message") or {}).get("content") or ""
    return clean_output(content), str(data.get("model", entry["model"])), decoding


def translate_anthropic(entry: dict[str, Any], prompt: str) -> tuple[str, str, dict[str, Any]]:
    decoding = dict(entry["decoding"])
    body = {
        "model": entry["model"],
        "max_tokens": decoding["max_tokens"],
        "temperature": decoding["temperature"],
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": _key(entry),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = _post("https://api.anthropic.com/v1/messages", headers=headers, body=body)
    blocks = data.get("content") or []
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    if not text and blocks:
        text = blocks[-1].get("text", "")
    return clean_output(text), str(data.get("model", entry["model"])), decoding


def translate_gemini(entry: dict[str, Any], prompt: str) -> tuple[str, str, dict[str, Any]]:
    decoding = dict(entry["decoding"])
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{entry['model']}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": decoding["temperature"],
            "topP": decoding["top_p"],
            "maxOutputTokens": decoding["max_tokens"],
        },
    }
    data = _post(url, params={"key": _key(entry)}, body=body,
                 headers={"Content-Type": "application/json"})
    candidate = (data.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    reported = str(data.get("modelVersion", entry["model"]))
    return clean_output(text), reported, decoding


def translate_google_v2(entry: dict[str, Any], text: str, source: str, target: str) -> tuple[str, str, dict[str, Any]]:
    body = {"q": text, "source": source, "target": target, "format": "text"}
    data = _post(entry["endpoint"], params={"key": _key(entry)}, body=body)
    translations = ((data.get("data") or {}).get("translations")) or []
    if not translations:
        raise ProviderError(f"no translations field: {json.dumps(data)[:200]}")
    raw = translations[0].get("translatedText", "")
    return clean_output(html.unescape(raw)), "google-translate-v2/nmt", {}


# ---------------------------------------------------------------------------
# Doctor probes: is the key valid, does the pinned model exist, and if not,
# what similar IDs does the vendor list?
# ---------------------------------------------------------------------------

def list_models(entry: dict[str, Any]) -> list[str]:
    provider = entry["provider"]
    if provider == "openai_compat":
        data = _get(entry["base_url"].rstrip("/") + "/models",
                    headers={"Authorization": f"Bearer {_key(entry)}"})
        return sorted(str(item.get("id", "")) for item in data.get("data", []))
    if provider == "anthropic":
        data = _get("https://api.anthropic.com/v1/models",
                    headers={"x-api-key": _key(entry), "anthropic-version": "2023-06-01"})
        return sorted(str(item.get("id", "")) for item in data.get("data", []))
    if provider == "gemini":
        data = _get("https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": _key(entry), "pageSize": "1000"})
        return sorted(str(item.get("name", "")).removeprefix("models/")
                      for item in data.get("models", []))
    if provider == "google_translate_v2":
        data = _get("https://translation.googleapis.com/language/translate/v2/languages",
                    params={"key": _key(entry), "target": "en"})
        langs = ((data.get("data") or {}).get("languages")) or []
        return sorted(str(item.get("language", "")) for item in langs)
    raise ProviderError(f"no doctor probe for provider {provider!r}")


def doctor_check(system_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """One registry entry -> status dict. Never raises."""
    result: dict[str, Any] = {"system": system_id, "provider": entry["provider"],
                              "model": entry["model"], "pin_status": entry["pin_status"]}
    if entry["provider"] == "hf_local":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            result["status"] = "OK"
            result["detail"] = "local anchor; torch+transformers importable, no key needed"
        except ImportError:
            result["status"] = "NEEDS_DEPS"
            result["detail"] = "pip install -r requirements-local.txt to run this anchor"
        return result
    env = entry.get("key_env")
    if not os.environ.get(env or "", ""):
        result["status"] = "MISSING_KEY"
        result["detail"] = f"set {env} in .env"
        return result
    try:
        models = list_models(entry)
    except ProviderError as exc:
        result["status"] = "AUTH_OR_NETWORK_FAIL"
        result["detail"] = str(exc)
        return result
    if entry["provider"] == "google_translate_v2":
        from .lib import LANGS
        missing = [k for k, v in LANGS.items() if v["google"] and v["google"] not in models]
        result["status"] = "OK" if not missing else "LANGS_MISSING"
        result["detail"] = ("all registered languages available" if not missing
                            else f"languages missing from paid API: {missing}")
        return result
    model = entry["model"]
    if model in models:
        result["status"] = "OK"
        result["detail"] = "pinned model ID confirmed against live model list"
    else:
        hint = entry.get("family_hint", "")
        suggestions = [m for m in models if hint and hint in m.lower()][:8]
        result["status"] = "MODEL_NOT_FOUND"
        result["detail"] = (f"pinned ID not in live list; similar: {suggestions}"
                            if suggestions else "pinned ID not in live list; inspect full list")
    return result
