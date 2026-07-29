"""Fail-closed system registry.

Every benchmarked system is pinned here. `run` refuses anything not
registered, and refuses registered systems on unsupported language/direction
combinations. Hosted model IDs carry a pin_status:

  verified     ID confirmed against vendor documentation at pin time
  provisional  best-known ID at pin time; `bench.py doctor` confirms it
               against the live model list once an API key exists and
               suggests the correct ID if it drifted
  alias        vendor-documented stable alias that tracks their current model

Aliases are a last resort: an alias that silently moves under us produces two
incomparable rows with the same name. Receipts record the model the vendor
reported, so a swap is at least detectable after the fact.

Local systems are pinned to immutable Hugging Face revisions and need no
credentials. They anchor the leaderboard across years: hosted models churn,
anchors do not.

Entry fields
------------
Required in every entry:
  label vendor provider model pin_status key_env license tier supported decoding

Optional, by provider:
  base_url            openai_compat: OpenAI-compatible root, no trailing slash
  endpoint            dedicated MT: absolute REST endpoint
  extra_env           further credential/config env vars doctor must find
  family_hint         doctor: substring used to suggest replacement model IDs
  api_version         dedicated MT: version string the API requires
  family              hf_local: seq2seq family (nllb/opus_cel/madlad) or "causal"
  revision            hf_local: immutable commit SHA (str, or per-direction dict)
  context_tokens      hf_local: input truncation length
  gated / token_env   hf_local: weights need an accepted licence + HF token
  benchmark_only      non-commercial licence: scored, shippable by nobody
  unsupported_note    why the uncovered languages are refused
  experimental_langs  provider ships these but labels them experimental
  decoding_deviation  provider forces a departure from the fixed decoding
  reasoning           provider-default reasoning that cannot be turned off

Roster policy (see METHODOLOGY.md "Panel"): general-purpose models are run on
all six languages because no vendor publishes a Celtic support contract for
them, so their coverage is the empirical question. Dedicated MT services are
run only on the languages their own current language list contains; anything
else is refused, never attempted and scored as a failure.

Pins were last reviewed 2026-07-29.
"""
from __future__ import annotations

from typing import Any

from .lib import ALL_LANGS, LANGS
from .prompt import CAUSAL_DECODING, CHAT_DECODING

_ALL_BOTH = {"en-xx": ALL_LANGS, "xx-en": ALL_LANGS}


def _covered(column: str) -> tuple[str, ...]:
    """Languages a provider's own published language list contains."""
    return tuple(k for k in ALL_LANGS if LANGS[k][column])


def _both(column: str) -> dict[str, tuple[str, ...]]:
    langs = _covered(column)
    return {"en-xx": langs, "xx-en": langs}


def _pairs(*langs: str) -> dict[str, tuple[str, ...]]:
    return {"en-xx": langs, "xx-en": langs}


SYSTEMS: dict[str, dict[str, Any]] = {
    # ---- Dedicated MT services --------------------------------------------
    "google-translate-v2": {
        "label": "Google Cloud Translation NMT",
        "vendor": "Google",
        "provider": "google_translate_v2",
        "model": "nmt",
        "pin_status": "verified",
        "key_env": "GOOGLE_TRANSLATE_API_KEY",
        "endpoint": "https://translation.googleapis.com/language/translate/v2",
        "license": "proprietary API",
        "tier": "dedicated-mt",
        "supported": _both("google"),
        "unsupported_note": (
            "Google's current NMT language list has no Manx or Cornish; the "
            "consumer web UI is not the API contract"
        ),
        "decoding": {},  # no decoding parameters exposed
    },
    "google-translation-llm": {
        "label": "Google Cloud Translation LLM",
        "vendor": "Google",
        "provider": "google_translation_llm",
        "model": "general/translation-llm",
        "pin_status": "verified",
        "key_env": "GOOGLE_CLOUD_ACCESS_TOKEN",
        "extra_env": ("GOOGLE_CLOUD_PROJECT",),
        "endpoint": "https://translate.googleapis.com/v3",
        "license": "proprietary API",
        "tier": "dedicated-mt",
        "supported": _both("google_tllm"),
        "experimental_langs": ("ga", "gd"),
        "unsupported_note": "Translation LLM lists only cy (plus experimental ga/gd)",
        "decoding": {},
    },
    "deepl": {
        "label": "DeepL API",
        "vendor": "DeepL",
        "provider": "deepl",
        "model": "deepl-translate-v2",
        "pin_status": "verified",
        "key_env": "DEEPL_API_KEY",
        "license": "proprietary API",
        "tier": "dedicated-mt",
        "supported": _both("deepl"),
        "unsupported_note": "DeepL lists no Scottish Gaelic, Manx or Cornish",
        "decoding": {},
    },
    "azure-translator": {
        "label": "Azure AI Translator",
        "vendor": "Microsoft",
        "provider": "azure_translator",
        "model": "azure-translator-nmt",
        "pin_status": "verified",
        "key_env": "AZURE_TRANSLATOR_KEY",
        "extra_env": ("AZURE_TRANSLATOR_REGION",),
        "endpoint": "https://api.cognitive.microsofttranslator.com/translate",
        "api_version": "3.0",
        "license": "proprietary API",
        "tier": "dedicated-mt",
        "supported": _both("azure"),
        "unsupported_note": "Azure lists only Irish and Welsh of the six",
        "decoding": {},
    },
    "aws-translate": {
        "label": "Amazon Translate",
        "vendor": "Amazon",
        "provider": "aws_translate",
        "model": "aws-translate",
        "pin_status": "verified",
        "key_env": "AWS_ACCESS_KEY_ID",
        "extra_env": ("AWS_SECRET_ACCESS_KEY", "AWS_REGION"),
        "license": "proprietary API",
        "tier": "dedicated-mt",
        "supported": _both("aws"),
        "unsupported_note": "Amazon Translate lists only Irish and Welsh of the six",
        "decoding": {},
    },
    "alibaba-mt": {
        "label": "Alibaba Cloud Machine Translation (general)",
        "vendor": "Alibaba",
        "provider": "alibaba_mt",
        "model": "TranslateGeneral",
        "pin_status": "verified",
        "key_env": "ALIBABA_ACCESS_KEY_ID",
        "extra_env": ("ALIBABA_ACCESS_KEY_SECRET",),
        "endpoint": "https://mt.aliyuncs.com/",
        "api_version": "2018-10-12",
        "license": "proprietary API",
        "tier": "dedicated-mt",
        "supported": _both("alibaba"),
        "unsupported_note": (
            "Alibaba's list has no Scottish Gaelic; its `sco` is Scots, a "
            "different language, and is never substituted for gd"
        ),
        "decoding": {},
    },

    # ---- Hosted general-purpose models, flagship tier ----------------------
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

    # ---- Hosted general-purpose models, efficient tier ---------------------
    "gpt-5.6-luna": {
        "label": "GPT-5.6 Luna",
        "vendor": "OpenAI",
        "provider": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.6-luna",
        "pin_status": "verified",
        "key_env": "OPENAI_API_KEY",
        "license": "proprietary API",
        "tier": "efficient-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "gpt",
    },
    "claude-haiku-4-5": {
        "label": "Claude Haiku 4.5",
        "vendor": "Anthropic",
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "pin_status": "verified",
        "key_env": "ANTHROPIC_API_KEY",
        "license": "proprietary API",
        "tier": "efficient-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "haiku",
    },
    "deepseek-v4-flash": {
        "label": "DeepSeek V4 Flash",
        "vendor": "DeepSeek",
        "provider": "openai_compat",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
        "pin_status": "verified",
        "key_env": "DEEPSEEK_API_KEY",
        "license": "proprietary API",
        "tier": "efficient-chat",
        "supported": _ALL_BOTH,
        "decoding": CHAT_DECODING,
        "family_hint": "deepseek",
    },

    # ---- Open-weight MT anchors (local, keyless, immutable revisions) -----
    "opus-mt-cel": {
        "label": "Opus-MT en-cel / cel-en (anchor)",
        "vendor": "Helsinki-NLP",
        "provider": "hf_local",
        "family": "opus_cel",
        "model": {"en-xx": "Helsinki-NLP/opus-mt-en-cel", "xx-en": "Helsinki-NLP/opus-mt-cel-en"},
        "revision": {"en-xx": "e79438534e0be0f4efa2585e3b24393bf40def95",
                     "xx-en": "f536f454b1320ff4bfed9435845bd960390c81cf"},
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
        "benchmark_only": True,  # non-commercial: scored, shippable by nobody
        "tier": "open-anchor",
        "supported": _both("nllb"),
        "unsupported_note": "NLLB-200 does not cover Manx or Cornish",
        "decoding": {"num_beams": 4, "max_new_tokens": 256, "do_sample": False},
        "context_tokens": 1024,
    },
    "translategemma-4b": {
        "label": "TranslateGemma 4B IT",
        "vendor": "Google",
        "provider": "hf_local",
        "family": "causal",
        "model": "google/translategemma-4b-it",
        "revision": "10042cb0e6e7fdce748996a71dc3dc432a4e0c89",
        "pin_status": "verified",
        "key_env": None,
        "gated": True,
        "token_env": "HF_TOKEN",
        "license": "Gemma Terms of Use",
        "tier": "open-mt",
        # Google's report documents synthetic en->ga and en->gd, and Breton and
        # Manx paired with English in both directions. Nothing for Cornish.
        "supported": {"en-xx": ("ga", "gd", "br", "gv"), "xx-en": ("br", "gv")},
        "unsupported_note": "TranslateGemma documents no Cornish, and no ga/gd into English",
        # No `template`: the model card that would document its inference format
        # is behind the same licence gate as the weights, so nothing can be
        # pinned yet. It cannot run without a token either, so no number can be
        # published off an unverified format. Pin the template when you accept
        # the licence.
        "decoding": CAUSAL_DECODING,
        "context_tokens": 2048,
    },
    "salamandrata-7b": {
        "label": "SalamandraTA 7B Instruct",
        "vendor": "Barcelona Supercomputing Center",
        "provider": "hf_local",
        "family": "causal",
        "model": "BSC-LT/salamandraTA-7b-instruct",
        "revision": "67e0f70c3a058b0f18381d421084d06fc7edbc3e",
        "pin_status": "verified",
        "key_env": None,
        "license": "GPL-3.0",
        "tier": "open-mt",
        "supported": _pairs("ga", "cy"),
        "unsupported_note": "SalamandraTA documents Irish and Welsh, not the other four",
        # Its own published general-translation template, verbatim from the model
        # card. The card warns the model "lacks chat capabilities and has not
        # been trained with any chat instructions": given the shared benchmark
        # prompt it translates the instruction and then the sentence.
        # https://huggingface.co/BSC-LT/salamandraTA-7b-instruct
        "template": ("Translate the following text from {src_name} into {tgt_name}.\n"
                     "{src_name}: {text} \n{tgt_name}:"),
        # Beam width 5 is the card's own recommendation, as with the other
        # dedicated MT models' documented beams.
        "decoding": {"num_beams": 5, "max_new_tokens": 256, "do_sample": False},
        "context_tokens": 2048,
    },
    "tiny-aya-water": {
        "label": "Tiny Aya Water",
        "vendor": "Cohere Labs",
        "provider": "hf_local",
        "family": "causal",
        "model": "CohereLabs/tiny-aya-water",
        "revision": "3696a27d9f455538e50887778122cc898b6370b8",
        "pin_status": "verified",
        "key_env": None,
        "gated": True,
        "token_env": "HF_TOKEN",
        "license": "CC-BY-NC-4.0",
        "benchmark_only": True,
        "tier": "open-mt",
        "supported": _pairs("ga", "cy"),
        "unsupported_note": "Tiny Aya's published Celtic evaluation covers Irish and Welsh",
        "decoding": CAUSAL_DECODING,
        "context_tokens": 2048,
    },

    # ---- Open-weight general model (local control) ------------------------
    "qwen3.5-9b": {
        "label": "Qwen3.5 9B",
        "vendor": "Alibaba",
        "provider": "hf_local",
        "family": "causal",
        "model": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "pin_status": "verified",
        "key_env": None,
        "license": "Apache-2.0",
        "tier": "open-general",
        "supported": _ALL_BOTH,
        "decoding": CAUSAL_DECODING,
        "context_tokens": 2048,
    },
}

CHAT_PROVIDERS = ("openai_compat", "anthropic", "gemini")
MT_PROVIDERS = ("google_translate_v2", "google_translation_llm", "deepl",
                "azure_translator", "aws_translate", "alibaba_mt")
LOCAL_PROVIDERS = ("hf_local",)


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
    """Hosted conversational model reached through a chat-completions API."""
    return entry["provider"] in CHAT_PROVIDERS


def uses_prompt(entry: dict[str, Any]) -> bool:
    """True when the system is driven by the published prompt.

    Hosted chat models and local causal models both consume it, so both bind
    their receipts to its hash. Sequence-to-sequence MT models and dedicated
    MT services take raw text and no prompt.
    """
    return is_chat(entry) or entry.get("family") == "causal"


def credential_envs(entry: dict[str, Any]) -> tuple[str, ...]:
    """Every environment variable this system needs before it can run."""
    envs = [entry["key_env"]] if entry.get("key_env") else []
    envs.extend(entry.get("extra_env", ()))
    return tuple(envs)


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
    import os

    from .lib import all_corpora, manifest_path

    rows: list[dict[str, Any]] = []
    for system_id, entry in SYSTEMS.items():
        for corpus in all_corpora():
            for lang in ALL_LANGS:
                if not os.path.exists(manifest_path(corpus, lang)):
                    continue  # corpus does not cover this language (e.g. flores br/gv/kw)
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
