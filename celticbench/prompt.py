"""The fixed translation prompt and fixed decoding parameters.

One prompt for every chat model, published verbatim in METHODOLOGY.md. No
per-model prompt tuning: the benchmark measures the model, not our prompting.
The template hash and decoding hash are recorded in every receipt; changing
either creates a new, incomparable leaderboard version by construction.
"""
from __future__ import annotations

from typing import Any

from .lib import lang_pair_names, sha256_json, sha256_text

PROMPT_TEMPLATE = (
    "Translate the following {src_name} sentence into {tgt_name}. "
    "Output only the {tgt_name} translation as plain text: no quotes, no "
    "notes, no explanation, no alternatives.\n\n{text}"
)

PROMPT_SHA256 = sha256_text(PROMPT_TEMPLATE)

# Fixed decoding for every hosted chat system. max_tokens is deliberately
# generous so reasoning-heavy models are not truncated mid-answer; sentence
# outputs are short regardless.
CHAT_DECODING: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 2048,
}

CHAT_DECODING_SHA256 = sha256_json(CHAT_DECODING)


def render(lang: str, direction: str, text: str) -> str:
    src_name, tgt_name = lang_pair_names(lang, direction)
    return PROMPT_TEMPLATE.format(src_name=src_name, tgt_name=tgt_name, text=text)


def clean_output(raw: str) -> str:
    """Single-line normalization applied to every system's output."""
    return raw.replace("\n", " ").strip()
