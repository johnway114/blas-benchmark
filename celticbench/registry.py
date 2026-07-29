"""Fail-closed system registry.

Every benchmarked system is pinned here. `run` refuses anything not
registered, and refuses registered systems on unsupported language/direction
combinations. Hosted model IDs carry a pin_status:

  verified     ID confirmed against vendor documentation at pin time
  provisional  best-known ID at pin time; `bench.py doctor` confirms it
               against the live model list once an API key exists and
               suggests the correct ID if it drifted
  alias        vendor-documented stable alias that tracks their current model

Local anchor systems are pinned to immutable Hugging Face revisions (same
pins as first measured on 2026-06-25) and need no credentials. They anchor
the leaderboard across years: hosted models churn, anchors do not.

Pins were last reviewed 2026-07-29.
"""
from __future__ import annotations

from typing import Any

from .lib import ALL_LANGS, LANGS
from .prompt import CHAT_DECODING

_ALL_BOTH = {"en-xx": ALL_LANGS, "xx-en": ALL_LANGS}
_GOOGLE_LANGS = tuple(k for k in ALL_LANGS if LANGS[k]["google"])   # no kw
_NLLB_LANGS = tuple(k for k in ALL_LANGS if LANGS[k]["nllb"])       # ga cy gd br

SYSTEMS: dict[str, dict[str, Any]] = {
    # ---- Dedicated MT, paid API -------------------------------------------
    "google-translate-v2": {
        "label": "Google Cloud Translation (paid API)",
        "vendor": "Google",
        "provider": "google_translate_v2",
        "model": "nmt",
        "pin_status": "verified",
        "key_env": "GOOGLE_TRANSLATE_API_KEY",
        "endpoint": "https://translation.googleapis.com/language/translate/v2",
        "license": "proprietary API",
        "tier": "dedicated-mt",
        "supported": {"en-xx": _GOOGLE_LANGS, "xx-en": _GOOGLE_LANGS},
        "unsupported_note": "Cornish (kw) is not offered by Google Translate",
        "decoding": {},  # no decoding parameters exposed
    },

    # ---- Hosted frontier chat models --------------------------------------
    "gpt-5.6-sol": {
        "label": "GPT-5.6 Sol (flagship)",
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
    "gpt-5.6-luna": {
        "label": "GPT-5.6 Luna (efficient tier)",
        "vendor": "OpenAI",
        "provider": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.6-luna",
        "pin_status": "provisional",
        "key_env": "OPENAI_API_KEY",
        "license": "proprietary API",
        "tier": "efficient-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "gpt",
    },
    "claude-opus-5": {
        "label": "Claude Opus 5 (flagship)",
        "vendor": "Anthropic",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "pin_status": "provisional",
        "key_env": "ANTHROPIC_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "opus",
    },
    "claude-haiku-4-5": {
        "label": "Claude Haiku 4.5 (efficient tier)",
        "vendor": "Anthropic",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "pin_status": "provisional",
        "key_env": "ANTHROPIC_API_KEY",
        "license": "proprietary API",
        "tier": "efficient-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "haiku",
    },
    "gemini-3.1-pro": {
        "label": "Gemini 3.1 Pro (flagship)",
        "vendor": "Google",
        "provider": "gemini",
        "model": "gemini-3.1-pro",
        "pin_status": "verified",
        "key_env": "GEMINI_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "gemini",
    },
    "gemini-3.5-flash": {
        "label": "Gemini 3.5 Flash (efficient tier)",
        "vendor": "Google",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "pin_status": "provisional",
        "key_env": "GEMINI_API_KEY",
        "license": "proprietary API",
        "tier": "efficient-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "gemini",
    },
    "deepseek-chat": {
        "label": "DeepSeek chat (V4 family)",
        "vendor": "DeepSeek",
        "provider": "openai_compat",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "pin_status": "alias",
        "key_env": "DEEPSEEK_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "deepseek",
    },
    "mistral-large-latest": {
        "label": "Mistral Large (latest alias)",
        "vendor": "Mistral",
        "provider": "openai_compat",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-large-latest",
        "pin_status": "alias",
        "key_env": "MISTRAL_API_KEY",
        "license": "proprietary API",
        "tier": "flagship-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "mistral",
    },

    # ---- Open-weight anchors (local, keyless, immutable revisions) --------
    "nllb-600m": {
        "label": "NLLB-200 distilled 600M (anchor)",
        "vendor": "Meta",
        "provider": "hf_local",
        "family": "nllb",
        "model": "facebook/nllb-200-distilled-600M",
        "revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "pin_status": "verified",
        "key_env": None,
        "license": "CC-BY-NC-4.0",
        "benchmark_only": True,  # non-commercial licence: scored, never shipped by anyone
        "tier": "open-anchor",
        "supported": {"en-xx": _NLLB_LANGS, "xx-en": _NLLB_LANGS},
        "unsupported_note": "NLLB-200 does not cover Manx or Cornish",
        "decoding": {"num_beams": 4, "max_new_tokens": 256, "do_sample": False},
        "context_tokens": 1024,
    },
    "opus-mt-cel": {
        "label": "Opus-MT en-cel / cel-en (anchor)",
        "vendor": "Helsinki-NLP",
        "provider": "hf_local",
        "family": "opus_cel",
        "model": {"en-xx": "Helsinki-NLP/opus-mt-en-cel", "xx-en": "Helsinki-NLP/opus-mt-cel-en"},
        "revision": {"en-xx": "e79438534e0be0f4efa2585e3b24393bf40def95", "xx-en": "f536f454b1320ff4bfed9435845bd960390c81cf"},
        "pin_status": "verified",
        "key_env": None,
        "license": "Apache-2.0",
        "tier": "open-anchor",
        "supported": _ALL_BOTH,
        "decoding": {"num_beams": 4, "max_new_tokens": 256, "do_sample": False},
        "context_tokens": 512,
    },
    "madlad400-3b": {
        "label": "MADLAD-400 3B MT (anchor)",
        "vendor": "Google",
        "provider": "hf_local",
        "family": "madlad",
        "model": "google/madlad400-3b-mt",
        "revision": "fa184c675da0b5c9e1c8694fccd4e12e2d422094",
        "pin_status": "verified",
        "key_env": None,
        "license": "Apache-2.0",
        "tier": "open-anchor",
        "supported": _ALL_BOTH,
        "decoding": {"num_beams": 2, "max_new_tokens": 256, "do_sample": False},
        "context_tokens": 512,
    },
}

CHAT_PROVIDERS = ("openai_compat", "anthropic", "gemini")


def get_system(system_id: str) -> dict[str, Any]:
    if system_id not in SYSTEMS:
        known = ", ".join(sorted(SYSTEMS))
        raise SystemExit(f"unregistered system {system_id!r}; registered: {known}")
    return SYSTEMS[system_id]


def supported(system_id: str, lang: str, direction: str) -> tuple[bool, str | None]:
    entry = get_system(system_id)
    langs = entry["supported"].get(direction, ())
    if lang in langs:
        return True, None
    note = entry.get("unsupported_note")
    return False, note or f"{system_id} does not support {lang} {direction}"


def is_chat(entry: dict[str, Any]) -> bool:
    return entry["provider"] in CHAT_PROVIDERS


def resolve_model(entry: dict[str, Any], direction: str) -> str:
    model = entry["model"]
    return model[direction] if isinstance(model, dict) else model


def resolve_revision(entry: dict[str, Any], direction: str) -> str | None:
    revision = entry.get("revision")
    if isinstance(revision, dict):
        return revision[direction]
    return revision


def matrix() -> list[dict[str, Any]]:
    """Every runnable (system, corpus, lang, direction) combination."""
    from .lib import CORPORA, manifest_path
    import os

    rows: list[dict[str, Any]] = []
    for system_id, entry in SYSTEMS.items():
        for corpus in CORPORA:
            for lang in ALL_LANGS:
                if not os.path.exists(manifest_path(corpus, lang)):
                    continue  # corpus does not exist for this language (e.g. flores br/gv/kw)
                for direction in ("en-xx", "xx-en"):
                    ok, reason = supported(system_id, lang, direction)
                    rows.append({
                        "system": system_id,
                        "corpus": corpus,
                        "lang": lang,
                        "direction": direction,
                        "supported": ok,
                        "reason": reason,
                        "key_env": entry.get("key_env"),
                        "provider": entry["provider"],
                    })
    return rows
