"""Local open-weight inference: seq2seq MT anchors and causal instruct models.

Heavy imports (torch, transformers) happen only inside translate_batch, so
API-only users never need them. Revisions are immutable pins from the
registry; the model_reported string embeds model@revision. Seq2seq anchors are
driven by their family's own control tokens, causal models by the same
published prompt the hosted chat systems receive.
"""
from __future__ import annotations

import datetime
import os
from typing import Any

from . import prompt
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


def _gated_token(entry: dict[str, Any], model_id: str) -> str | None:
    """Gated weights are fetched with a licence-accepted token or not at all."""
    if not entry.get("gated"):
        return None
    variable = entry["token_env"]
    token = (os.environ.get(variable) or "").strip()
    if not token:
        raise SystemExit(
            f"{model_id} is gated: accept the {entry['license']} licence at "
            f"https://huggingface.co/{model_id} with your Hugging Face account, "
            f"then export {variable}=<token>. Refusing to retry anonymously."
        )
    return token


def _prepare_causal_tokenizer(tokenizer: Any, model_id: str) -> None:
    """Left-pad, so every prompt in a batch ends at the same column.

    Decoder-only generation continues from the final position and the caller
    slices the prompt off by width; right padding would leave the continuation
    at a different offset in every row. Mutates the cached instance only.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise SystemExit(f"{model_id} tokenizer has no pad or eos token to pad batches with")
        tokenizer.pad_token = tokenizer.eos_token


def _causal_model(transformers: Any, model_id: str, revision: str,
                  hub_kwargs: dict[str, Any], dtype: Any):
    """Build the class the checkpoint declares itself to be.

    Qwen3.5-9B and TranslateGemma are image-text-to-text repos whose decoder
    hangs off a nested text_config. For such a checkpoint AutoModelForCausalLM
    can resolve to the text-only class, at which point transformers silently
    substitutes config.text_config and loads a fragment of the checkpoint; so
    composite configs try the image-text-to-text loader first, and a class that
    contradicts the declared architecture is rejected rather than benchmarked.
    """
    config = transformers.AutoConfig.from_pretrained(model_id, **hub_kwargs)
    declared = next(iter(getattr(config, "architectures", None) or ()), None)
    chain = ("AutoModelForCausalLM", "AutoModelForImageTextToText")
    if "text_config" in (getattr(config, "sub_configs", None) or {}):
        chain = tuple(reversed(chain))
    failures: list[str] = []
    for name in chain:
        loader = getattr(transformers, name, None)
        if loader is None:
            failures.append(f"{name}: absent from transformers {transformers.__version__}")
            continue
        try:
            model = loader.from_pretrained(model_id, dtype=dtype, **hub_kwargs)
        except Exception as exc:  # any loader failure is just a rejected candidate
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        built = type(model).__name__
        if declared and built != declared:
            failures.append(f"{name}: built {built}, checkpoint declares {declared}")
            continue
        return model
    raise SystemExit(f"cannot load causal model {model_id}@{revision[:12]}: " + "; ".join(failures))


def _load(entry: dict[str, Any], model_id: str, revision: str):
    torch, transformers, device = _runtime()
    key = (model_id, revision)
    if key not in _CACHE:
        hub_kwargs: dict[str, Any] = {"revision": revision}
        token = _gated_token(entry, model_id)
        if token:
            hub_kwargs["token"] = token
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **hub_kwargs)
        if entry["family"] == "causal":
            _prepare_causal_tokenizer(tokenizer, model_id)
            # bfloat16: this benchmark runs on 24 GiB of unified memory and a 9B
            # in float32 does not fit. The small seq2seq anchors keep their own dtype.
            model = _causal_model(transformers, model_id, revision, hub_kwargs, torch.bfloat16)
        else:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id, **hub_kwargs)
        _CACHE[key] = (tokenizer, model.to(device).eval())
    return torch, device, *_CACHE[key]


def _causal_inputs(tokenizer: Any, entry: dict[str, Any], texts: list[str],
                   lang: str, direction: str) -> tuple[list[str], bool]:
    """Render this system's own prompt in the model's own chat format.

    A dedicated MT model is bound to the template it publishes rather than the
    shared benchmark prompt: SalamandraTA is explicitly not chat-trained and
    answers the shared prompt by translating the instruction along with the
    sentence. Returns (inputs, templated); templated inputs already carry the
    special tokens the template emitted.
    """
    template = prompt.template_for(entry)
    rendered = [prompt.render_with(template, lang, direction, text) for text in texts]
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return rendered, False
    template_kwargs: dict[str, Any] = {}
    source = ("".join(chat_template.values()) if isinstance(chat_template, dict)
              else str(chat_template))
    if "date_string" in source:
        # A template that reads date_string otherwise substitutes a date frozen
        # into the checkpoint; SalamandraTA's model card passes the run date.
        template_kwargs["date_string"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    return [tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                          tokenize=False, add_generation_prompt=True,
                                          **template_kwargs)
            for text in rendered], True


def _batched(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def translate_batch(entry: dict[str, Any], texts: list[str], lang: str, direction: str) -> tuple[list[str], str]:
    """Translate texts; returns (outputs, model_reported)."""
    family = entry["family"]
    model_id = resolve_model(entry, direction)
    revision = resolve_revision(entry, direction)
    torch, device, tokenizer, model = _load(entry, model_id, revision)
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
    elif family == "causal":
        inputs, templated = _causal_inputs(tokenizer, entry, texts, lang, direction)
        if templated:
            tokenize_kwargs["add_special_tokens"] = False  # the template emits its own
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    else:
        raise SystemExit(f"unknown hf_local family {family!r}")

    outputs: list[str] = []
    context = int(entry.get("context_tokens", 512))
    with torch.inference_mode():
        for batch in _batched(inputs, BATCH_SIZE):
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=context, **tokenize_kwargs).to(device)
            generated = model.generate(**encoded, **generate_kwargs)
            if family == "causal":
                # left padding gave every row the same prompt width, so the
                # continuation begins at one offset for the whole batch
                generated = generated[:, encoded["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(prompt.clean_output(text) for text in decoded)
    return outputs, f"{model_id}@{revision[:12]}"
