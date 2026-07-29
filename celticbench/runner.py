"""Run one system over one corpus/language/direction, producing a hypothesis
file and its receipt.

Behavior:
  - fail-closed: unregistered systems, unsupported combinations, and eval
    files that do not match their committed manifest all refuse to run;
  - resumable: every (system, model, prompt, decoding, text) result is cached
    line-level in out/cache/{system}.jsonl and reused across reruns;
  - honest failures: a line that still fails after 4 attempts becomes an
    empty hypothesis line and is counted in the receipt, never dropped.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from typing import Any

from . import hf_local, providers
from .lib import (
    CACHE_DIR, SCHEMA_RECEIPT, direction_io, harness_version,
    hyp_path, lang_pair_names, read_lines, receipt_path, sha256_json, file_sha256,
    verify_manifest, write_lines, LANGS,
)
from .prompt import PROMPT_SHA256, PROMPT_TEMPLATE, render
from .registry import get_system, is_chat, resolve_model, resolve_revision, supported

RETRIES = 4
DEFAULT_SLEEP = 0.15


def _cache_file(system_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{system_id}.jsonl")


def _load_cache(system_id: str) -> dict[str, str]:
    cache: dict[str, str] = {}
    path = _cache_file(system_id)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                    cache[entry["k"]] = entry["v"]
                except (json.JSONDecodeError, KeyError):
                    continue  # torn tail line from a crash; harmless
    return cache


def _append_cache(system_id: str, key: str, value: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_file(system_id), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


def _cache_key(entry: dict[str, Any], direction: str, lang: str, text: str) -> str:
    return sha256_json({
        "model": entry["model"] if not isinstance(entry["model"], dict) else entry["model"][direction],
        "prompt_sha256": PROMPT_SHA256 if is_chat(entry) else None,
        "decoding": entry["decoding"],
        "direction": direction,
        "lang": lang,
        "text": text,
    })


def _translate_one(entry: dict[str, Any], lang: str, direction: str, text: str) -> tuple[str, str, dict[str, Any]]:
    provider = entry["provider"]
    if provider == "openai_compat":
        return providers.translate_openai_compat(entry, render(lang, direction, text))
    if provider == "anthropic":
        return providers.translate_anthropic(entry, render(lang, direction, text))
    if provider == "gemini":
        return providers.translate_gemini(entry, render(lang, direction, text))
    if provider == "google_translate_v2":
        source = "en" if direction == "en-xx" else LANGS[lang]["google"]
        target = LANGS[lang]["google"] if direction == "en-xx" else "en"
        return providers.translate_google_v2(entry, text, source, target)
    raise SystemExit(f"provider {provider!r} has no per-line path")


def run_system(system_id: str, corpus: str, lang: str, direction: str,
               limit: int | None = None, sleep: float = DEFAULT_SLEEP) -> str:
    entry = get_system(system_id)
    ok, reason = supported(system_id, lang, direction)
    if not ok:
        raise SystemExit(f"refusing: {reason}")
    manifest = verify_manifest(corpus, lang)

    input_path, _ = direction_io(corpus, lang, direction)
    sources = read_lines(input_path)
    if limit is not None:
        sources = sources[:limit]

    src_name, tgt_name = lang_pair_names(lang, direction)
    print(f"run {system_id}: {corpus}.{lang} {src_name} -> {tgt_name}, n={len(sources)}")

    outputs: list[str] = []
    fails = 0
    model_reported: str | None = None
    decoding_used: dict[str, Any] = dict(entry["decoding"])

    if entry["provider"] == "hf_local":
        outputs, model_reported = hf_local.translate_batch(entry, sources, lang, direction)
    else:
        cache = _load_cache(system_id)
        for index, text in enumerate(sources):
            key = _cache_key(entry, direction, lang, text)
            if key in cache:
                outputs.append(cache[key])
                continue
            result = ""
            for attempt in range(RETRIES):
                try:
                    result, model_reported, decoding_used = _translate_one(entry, lang, direction, text)
                    break
                except providers.ProviderError as exc:
                    if "missing API key" in str(exc):
                        raise SystemExit(str(exc)) from exc
                    if attempt == RETRIES - 1:
                        print(f"  FAIL line {index}: {str(exc)[:140]}")
                        fails += 1
                    else:
                        time.sleep(1.5 * (attempt + 1))
            outputs.append(result)
            if result:
                cache[key] = result
                _append_cache(system_id, key, result)
            time.sleep(sleep)
            if index and index % 50 == 0:
                print(f"  {index}/{len(sources)}")

    out_path = hyp_path(corpus, lang, direction, system_id)
    write_lines(out_path, outputs)

    receipt = {
        "schema": SCHEMA_RECEIPT,
        "system": system_id,
        "provider": entry["provider"],
        "vendor": entry["vendor"],
        "model_requested": resolve_model(entry, direction),
        "model_reported": model_reported or resolve_model(entry, direction),
        "revision": resolve_revision(entry, direction),
        "pin_status": entry["pin_status"],
        "license": entry["license"],
        "benchmark_only": bool(entry.get("benchmark_only", False)),
        "corpus": corpus,
        "lang": lang,
        "direction": direction,
        "n": len(sources),
        "limit": limit,
        "fails": fails,
        "prompt_template": PROMPT_TEMPLATE if is_chat(entry) else None,
        "prompt_sha256": PROMPT_SHA256 if is_chat(entry) else None,
        "decoding": decoding_used,
        "decoding_sha256": sha256_json(decoding_used),
        "corpus_manifest_sha256": manifest["contract_sha256"],
        "hypothesis_sha256": file_sha256(out_path),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "harness_version": harness_version(),
    }
    with open(receipt_path(out_path), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"  wrote {out_path} (+ receipt), fails={fails}")
    return out_path
