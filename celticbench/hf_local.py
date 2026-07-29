"""Local open-weight anchor inference (NLLB, Opus-MT Celtic, MADLAD-400).

Heavy imports (torch, transformers) happen only inside translate_batch, so
API-only users never need them. Revisions are immutable pins from the
registry; the model_reported string embeds model@revision.
"""
from __future__ import annotations

import os
from typing import Any

from .lib import LANGS
from .registry import resolve_model, resolve_revision

_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
BATCH_SIZE = int(os.environ.get("CELTICBENCH_BATCH", "8"))


def _runtime():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "local anchors need torch+transformers: pip install -r requirements-local.txt"
        ) from exc
    device = os.environ.get("CELTICBENCH_DEVICE") or (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return torch, transformers, device


def _load(model_id: str, revision: str):
    torch, transformers, device = _runtime()
    key = (model_id, revision)
    if key not in _CACHE:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_id, revision=revision
        ).to(device).eval()
        _CACHE[key] = (tokenizer, model)
    return torch, device, *_CACHE[key]


def _batched(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def translate_batch(entry: dict[str, Any], texts: list[str], lang: str, direction: str) -> tuple[list[str], str]:
    """Translate texts; returns (outputs, model_reported)."""
    family = entry["family"]
    model_id = resolve_model(entry, direction)
    revision = resolve_revision(entry, direction)
    torch, device, tokenizer, model = _load(model_id, revision)
    decoding = dict(entry["decoding"])
    generate_kwargs: dict[str, Any] = {
        "num_beams": decoding["num_beams"],
        "max_new_tokens": decoding["max_new_tokens"],
        "do_sample": decoding["do_sample"],
    }
    tokenize_kwargs: dict[str, Any] = {}

    if family == "nllb":
        src_code = "eng_Latn" if direction == "en-xx" else LANGS[lang]["nllb"]
        tgt_code = LANGS[lang]["nllb"] if direction == "en-xx" else "eng_Latn"
        tokenizer.src_lang = src_code
        bos = tokenizer.convert_tokens_to_ids(tgt_code)
        generate_kwargs["forced_bos_token_id"] = bos
        inputs = list(texts)
    elif family == "opus_cel":
        if direction == "en-xx":
            code = LANGS[lang]["opus_cel"]
            inputs = [f">>{code}<< {text}" for text in texts]
        else:
            inputs = list(texts)
    elif family == "madlad":
        target = LANGS[lang]["madlad"] if direction == "en-xx" else "en"
        inputs = [f"<2{target}> {text}" for text in texts]
    else:
        raise SystemExit(f"unknown hf_local family {family!r}")

    outputs: list[str] = []
    context = int(entry.get("context_tokens", 512))
    with torch.inference_mode():
        for batch in _batched(inputs, BATCH_SIZE):
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=context, **tokenize_kwargs).to(device)
            generated = model.generate(**encoded, **generate_kwargs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(text.replace("\n", " ").strip() for text in decoded)
    return outputs, f"{model_id}@{revision[:12]}"
