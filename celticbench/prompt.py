"""The fixed translation prompt, per-model MT templates, and fixed decoding.

One prompt for every general-purpose model, published verbatim in
METHODOLOGY.md. No per-model prompt tuning: the benchmark measures the model,
not our prompting.

A *dedicated* translation model is the documented exception, and for the same
reason the sequence-to-sequence anchors get their own control tokens: it was
never trained to follow instructions, so handing it ours measures our misuse
rather than its translation. SalamandraTA's card is blunt about it -- "it lacks
chat capabilities and has not been trained with any chat instructions" -- and
with the shared prompt it dutifully translates the instruction into Irish
before translating the sentence. Such a system declares its published template
in the registry, and its receipt records that template and its hash, so what
the model was actually given is always on the record.
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

# Fixed decoding for local causal (general-purpose / instruction-tuned) models
# reached through the same published prompt. Greedy, so a rerun of the same
# weights on the same machine reproduces the same bytes; 256 new tokens is far
# more than a sentence needs and stops a degenerate model from running away.
CAUSAL_DECODING: dict[str, Any] = {
    "num_beams": 1,
    "max_new_tokens": 256,
    "do_sample": False,
}

CAUSAL_DECODING_SHA256 = sha256_json(CAUSAL_DECODING)


def template_for(entry: dict[str, Any]) -> str:
    """The template this system is driven by: its own, or the published one."""
    return entry.get("template") or PROMPT_TEMPLATE


def template_sha256(entry: dict[str, Any]) -> str:
    return sha256_text(template_for(entry))


def render_with(template: str, lang: str, direction: str, text: str) -> str:
    src_name, tgt_name = lang_pair_names(lang, direction)
    return template.format(src_name=src_name, tgt_name=tgt_name, text=text)


def render(lang: str, direction: str, text: str) -> str:
    """The shared published prompt, for every system that has no template."""
    return render_with(PROMPT_TEMPLATE, lang, direction, text)


def clean_output(raw: str) -> str:
    """Single-line normalization applied to every system's output."""
    return raw.replace("\n", " ").strip()
