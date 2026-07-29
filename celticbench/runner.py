"""Run one system over one corpus/language/direction, producing a hypothesis
file and its receipt.

Behavior:
  - fail-closed: unregistered systems, unsupported combinations, and eval
    files that do not match their committed manifest all refuse to run;
  - resumable: every (system, model, prompt, decoding, text) result is cached
    line-level in out/cache/{system}.jsonl and reused across reruns. The cache
    stores what the vendor reported alongside the text, so a run served
    entirely from cache still writes a receipt about the model that actually
    produced those bytes rather than the model we asked for;
  - honest failures: a line that still fails after 4 attempts becomes an
    empty hypothesis line and is counted in the receipt, never dropped;
  - concurrent when asked: --workers fans lines out across threads. Order is
    preserved by index, so the hypothesis file is identical either way.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import hf_local, providers
from .lib import (
    CACHE_DIR, METHOD_VERSION, SCHEMA_RECEIPT, canonical_json, direction_io,
    file_sha256, harness_version, hyp_path, lang_pair_names, read_lines,
    receipt_path, runtime_versions, sha256_json, verify_manifest, write_lines,
)
from .prompt import render_with, template_for, template_sha256
from .registry import (
    get_system, resolve_model, resolve_revision, supported, uses_prompt,
)

RETRIES = 4
DEFAULT_SLEEP = 0.15

_CACHE_LOCK = threading.Lock()


def _cache_file(system_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{system_id}.jsonl")


def _load_cache(system_id: str) -> dict[str, dict[str, Any]]:
    """key -> {v, m, d, u}. Records without provenance are ignored.

    A pre-v2 cache line carries only the text. Reusing it would produce a
    receipt that cannot say which model wrote the line, so it is treated as a
    miss; the cache is a local optimisation, never published state.
    """
    cache: dict[str, dict[str, Any]] = {}
    path = _cache_file(system_id)
    if not os.path.exists(path):
        return cache
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if "m" not in record:
                    continue
                cache[record["k"]] = record
            except (json.JSONDecodeError, KeyError):
                continue  # torn tail line from a crash; harmless
    return cache


def _append_cache(system_id: str, key: str, translation: providers.Translation) -> dict[str, Any]:
    record = {
        "k": key,
        "v": translation.text,
        "m": translation.model_reported,
        "d": translation.decoding_used,
        "u": translation.usage,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with _CACHE_LOCK:
        with open(_cache_file(system_id), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _cache_key(entry: dict[str, Any], direction: str, lang: str, text: str) -> str:
    return sha256_json({
        "model": resolve_model(entry, direction),
        "prompt_sha256": template_sha256(entry) if uses_prompt(entry) else None,
        "decoding": entry["decoding"],
        "direction": direction,
        "lang": lang,
        "text": text,
    })


def _translate_one(entry: dict[str, Any], lang: str, direction: str, text: str) -> providers.Translation:
    if uses_prompt(entry):
        prompt = render_with(template_for(entry), lang, direction, text)
        return providers.translate_chat(entry, prompt)
    source, target = providers.mt_codes(entry, lang, direction)
    return providers.translate_mt(entry, text, source, target)


class _Provenance:
    """What the vendor actually did, accumulated across a run's lines."""

    def __init__(self, declared_decoding: dict[str, Any]) -> None:
        self.models: set[str] = set()
        self.decodings: dict[str, dict[str, Any]] = {}
        self.deviations: set[str] = set()
        self.declared_decoding = declared_decoding
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.requests = 0
        self.cache_hits = 0
        self.lock = threading.Lock()

    def record(self, model: str, decoding: dict[str, Any], usage: dict[str, int],
               deviations: tuple[str, ...] = (), *, cached: bool = False) -> None:
        with self.lock:
            self.models.add(model)
            self.decodings[canonical_json(decoding)] = decoding
            self.deviations.update(deviations)
            for field in self.usage:
                self.usage[field] += int(usage.get(field, 0) or 0)
            if cached:
                self.cache_hits += 1
            else:
                self.requests += 1

    def record_local(self, model: str, decoding: dict[str, Any]) -> None:
        """Local inference: provenance without usage.

        `requests` counts hosted-API calls, which is what `plan` estimates and
        what a bill is made of. A local batch is neither a request nor a cache
        hit, and counting it as one would misstate both.
        """
        self.models.add(model)
        self.decodings[canonical_json(decoding)] = decoding

    def model_reported(self, fallback: str) -> str:
        if not self.models:
            return fallback
        if len(self.models) == 1:
            return next(iter(self.models))
        return "mixed"

    def decoding_used(self) -> dict[str, Any]:
        if len(self.decodings) == 1:
            return next(iter(self.decodings.values()))
        return dict(self.declared_decoding)


def _run_hosted(entry: dict[str, Any], system_id: str, sources: list[str], lang: str,
                direction: str, sleep: float, workers: int,
                provenance: _Provenance) -> tuple[list[str], int]:
    cache = _load_cache(system_id)
    outputs: list[str] = [""] * len(sources)
    pending: list[tuple[int, str, str]] = []

    for index, text in enumerate(sources):
        key = _cache_key(entry, direction, lang, text)
        record = cache.get(key)
        if record is None:
            pending.append((index, key, text))
            continue
        outputs[index] = record["v"]
        provenance.record(record["m"], record.get("d") or {}, record.get("u") or {}, cached=True)

    if not pending:
        print(f"  {len(sources)} lines served from cache")
        return outputs, 0

    done = [0]
    fails = [0]

    def work(item: tuple[int, str, str]) -> tuple[int, str]:
        index, key, text = item
        for attempt in range(RETRIES):
            try:
                translation = _translate_one(entry, lang, direction, text)
            except providers.ProviderError as exc:
                if "missing API key" in str(exc) or "missing credential" in str(exc):
                    raise SystemExit(str(exc)) from exc
                if attempt == RETRIES - 1:
                    with provenance.lock:
                        fails[0] += 1
                    print(f"  FAIL line {index}: {str(exc)[:140]}")
                    return index, ""
                time.sleep(1.5 * (attempt + 1))
                continue
            provenance.record(translation.model_reported, translation.decoding_used,
                              translation.usage, translation.deviations)
            if translation.text:
                _append_cache(system_id, key, translation)
            time.sleep(sleep)
            with provenance.lock:
                done[0] += 1
                if done[0] % 50 == 0:
                    print(f"  {done[0]}/{len(pending)} (+{len(sources) - len(pending)} cached)")
            return index, translation.text
        return index, ""

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, text in pool.map(work, pending):
                outputs[index] = text
    else:
        for item in pending:
            index, text = work(item)
            outputs[index] = text
    return outputs, fails[0]


def run_system(system_id: str, corpus: str, lang: str, direction: str,
               limit: int | None = None, sleep: float = DEFAULT_SLEEP,
               workers: int = 1) -> str:
    entry = get_system(system_id)
    ok, reason = supported(system_id, lang, direction)
    if not ok:
        raise SystemExit(f"refusing: {reason}")
    manifest = verify_manifest(corpus, lang)

    input_path, reference_path = direction_io(corpus, lang, direction)
    sources = read_lines(input_path)
    if limit is not None:
        sources = sources[:limit]

    src_name, tgt_name = lang_pair_names(lang, direction)
    print(f"run {system_id}: {corpus}.{lang} {src_name} -> {tgt_name}, n={len(sources)}")

    provenance = _Provenance(dict(entry["decoding"]))
    if entry["provider"] == "hf_local":
        outputs, model_reported = hf_local.translate_batch(entry, sources, lang, direction)
        fails = 0
        provenance.record_local(model_reported, dict(entry["decoding"]))
    else:
        outputs, fails = _run_hosted(entry, system_id, sources, lang, direction,
                                     sleep, workers, provenance)

    out_path = hyp_path(corpus, lang, direction, system_id, limit)
    write_lines(out_path, outputs)

    decoding_used = provenance.decoding_used()
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "method_version": METHOD_VERSION,
        "system": system_id,
        "provider": entry["provider"],
        "vendor": entry["vendor"],
        "tier": entry["tier"],
        "model_requested": resolve_model(entry, direction),
        "model_reported": provenance.model_reported(resolve_model(entry, direction)),
        "model_reported_variants": sorted(provenance.models),
        "revision": resolve_revision(entry, direction),
        "pin_status": entry["pin_status"],
        "license": entry["license"],
        "benchmark_only": bool(entry.get("benchmark_only", False)),
        "reasoning": entry.get("reasoning"),
        "corpus": corpus,
        "lang": lang,
        "direction": direction,
        "n": len(sources),
        "limit": limit,
        "partial_slice": limit is not None,
        "fails": fails,
        "prompt_template": template_for(entry) if uses_prompt(entry) else None,
        "prompt_sha256": template_sha256(entry) if uses_prompt(entry) else None,
        "decoding_declared": dict(entry["decoding"]),
        "decoding_declared_sha256": sha256_json(entry["decoding"]),
        "decoding": decoding_used,
        "decoding_sha256": sha256_json(decoding_used),
        "decoding_deviations": sorted(provenance.deviations),
        "corpus_manifest_sha256": manifest["contract_sha256"],
        "eval_input_sha256": file_sha256(input_path),
        "eval_reference_sha256": file_sha256(reference_path),
        "hypothesis_file": os.path.basename(out_path),
        "hypothesis_sha256": file_sha256(out_path),
        "usage": {
            **provenance.usage,
            "requests": provenance.requests,
            "cache_hits": provenance.cache_hits,
            "source_chars": sum(len(text) for text in sources),
        },
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "harness_version": harness_version(),
        "runtime": runtime_versions(),
    }
    with open(receipt_path(out_path), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"  wrote {out_path} (+ receipt), fails={fails}")
    return out_path
