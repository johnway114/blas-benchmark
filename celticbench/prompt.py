"""The shared hosted translation prompt and fixed decoding contract."""
from __future__ import annotations

from typing import Any

from .lib import lang_pair_names, sha256_json, sha256_text

PROMPT_TEMPLATE = (
    "Translate the following {src_name} sentence into {tgt_name}. "
    "Output only the {tgt_name} translation as plain text: no quotes, no "
    "notes, no explanation, no alternatives.\n\n{text}"
)
PROMPT_SHA256 = sha256_text(PROMPT_TEMPLATE)

CHAT_DECODING: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 2048,
}
CHAT_DECODING_SHA256 = sha256_json(CHAT_DECODING)


def render(lang: str, direction: str, text: str) -> str:
    """Render the one published prompt used by every system."""
    src_name, tgt_name = lang_pair_names(lang, direction)
    return PROMPT_TEMPLATE.format(src_name=src_name, tgt_name=tgt_name, text=text)


def clean_output(raw: str) -> str:
    """Normalize every hosted answer to one stripped line."""
    return raw.replace("\n", " ").strip()
