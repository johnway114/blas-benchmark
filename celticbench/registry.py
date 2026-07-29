"""Fail-closed registry for the six hosted frontier systems.

Every system is run over all six Celtic languages in both directions with the
same published chat prompt. Hosted model IDs are pinned here; ``doctor`` checks
credentials and confirms those IDs against each provider's live model list.
"""
from __future__ import annotations

from typing import Any

from .lib import ALL_LANGS
from .prompt import CHAT_DECODING

_ALL_BOTH = {"en-xx": ALL_LANGS, "xx-en": ALL_LANGS}

SYSTEMS: dict[str, dict[str, Any]] = {
    "gpt-5.6-sol": {
        "label": "GPT-5.6 Sol",
        "vendor": "OpenAI",
        "provider": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.6-sol",
        "pin_status": "verified",
        "key_env": "OPENAI_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "gpt",
    },
    "claude-opus-5": {
        "label": "Claude Opus 5",
        "vendor": "Anthropic",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "pin_status": "verified",
        "key_env": "ANTHROPIC_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "opus",
    },
    "gemini-3.6-flash": {
        "label": "Gemini 3.6 Flash",
        "vendor": "Google",
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "pin_status": "verified",
        "key_env": "GEMINI_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "decoding_deviation": {
            "omits": ("temperature", "top_p"),
            "reason": (
                "Gemini deprecated temperature/top_p; the API ignores them and "
                "will error on them in future models, so they are not sent"
            ),
        },
        "family_hint": "gemini",
    },
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "vendor": "DeepSeek",
        "provider": "openai_compat",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-pro",
        "pin_status": "verified",
        "key_env": "DEEPSEEK_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "reasoning": "provider-default reasoning model; cannot be disabled",
        "family_hint": "deepseek",
    },
    "kimi-k3": {
        "label": "Kimi K3",
        "vendor": "Moonshot AI",
        "provider": "openai_compat",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k3",
        "pin_status": "verified",
        "key_env": "MOONSHOT_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "reasoning": "always-on reasoning; cannot be disabled",
        "family_hint": "kimi",
    },
    "qwen3.7-max": {
        "label": "Qwen3.7 Max",
        "vendor": "Alibaba",
        "provider": "openai_compat",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-max",
        "pin_status": "provisional",
        "key_env": "DASHSCOPE_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "qwen",
    },
}

PROVIDERS = ("openai_compat", "anthropic", "gemini")


def get_system(system_id: str) -> dict[str, Any]:
    if system_id not in SYSTEMS:
        known = ", ".join(sorted(SYSTEMS))
        raise SystemExit(f"unregistered system {system_id!r}; registered: {known}")
    return SYSTEMS[system_id]


def supported(system_id: str, lang: str, direction: str) -> tuple[bool, str | None]:
    entry = get_system(system_id)
    if lang in entry["supported"].get(direction, ()):
        return True, None
    return False, f"{system_id} does not support {lang} {direction}"


def credential_envs(entry: dict[str, Any]) -> tuple[str, ...]:
    """Every environment variable required before this hosted system can run."""
    return (entry["key_env"],)


def matrix() -> list[dict[str, Any]]:
    """Every benchmark combination, including fail-closed support status."""
    import os

    from .lib import all_corpora, manifest_path

    rows: list[dict[str, Any]] = []
    for system_id, entry in SYSTEMS.items():
        for corpus in all_corpora():
            for lang in ALL_LANGS:
                if not os.path.exists(manifest_path(corpus, lang)):
                    continue
                for direction in ("en-xx", "xx-en"):
                    ok, reason = supported(system_id, lang, direction)
                    rows.append({
                        "system": system_id,
                        "corpus": corpus,
                        "lang": lang,
                        "direction": direction,
                        "supported": ok,
                        "reason": reason,
                        "key_env": entry["key_env"],
                        "provider": entry["provider"],
                    })
    return rows
