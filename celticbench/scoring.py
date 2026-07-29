"""Score every hypothesis that carries a verifiable receipt.

Fail-closed: a hypothesis without a receipt, with a tampered receipt, whose
bytes do not match the receipt hash, or whose corpus manifest no longer
matches the committed contract is excluded from scores.json and listed under
`excluded` with its reason. Nothing is silently dropped or silently included.

Metrics:
  Track A (reference): chrF++ (word_order=2) and BLEU via sacrebleu.
  Track C (reference-free): off_target_rate, blank_rate, copy_rate,
  repetition_rate, plus a length-ratio diagnostic.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
from typing import Any

from .lib import (
    LANGS, OUT, SCHEMA_SCORES, SCORES, direction_io, expected_target_lang,
    file_sha256, load_manifest, read_lines, receipt_path,
)
from .registry import SYSTEMS

_HYP_RE = re.compile(
    r"^(?P<corpus>[a-z0-9]+)\.(?P<lang>[a-z]{2})\.(?P<direction>en-xx|xx-en)\.(?P<system>.+)\.hyp$"
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_repetitive(line: str) -> bool:
    """Degenerate output: a repeated 4-gram loop or near-total token collapse."""
    tokens = line.split()
    if len(tokens) < 8:
        return False
    if len(set(tokens)) / len(tokens) < 0.3:
        return True
    grams = [" ".join(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    for index in range(len(grams) - 8):
        if grams[index] == grams[index + 4] == grams[index + 8]:
            return True
    return False


def track_c_metrics(sources: list[str], hyps: list[str], expected_lang: str) -> dict[str, Any]:
    from .langid import CONFIDENCE_THRESHOLD, is_off_target

    n = len(hyps)
    if n == 0:
        return {}
    blank = sum(1 for h in hyps if not h.strip())
    copied = sum(1 for s, h in zip(sources, hyps) if h.strip() and _norm(s) == _norm(h))
    repetitive = sum(1 for h in hyps if _is_repetitive(h))
    off_target = sum(1 for h in hyps if is_off_target(h, expected_lang))
    return {
        "blank_rate": round(blank / n, 4),
        "copy_rate": round(copied / n, 4),
        "repetition_rate": round(repetitive / n, 4),
        "off_target_rate": round(off_target / n, 4),
        "off_target_expected_lang": expected_lang,
        "off_target_confidence_threshold": CONFIDENCE_THRESHOLD,
    }


def score_pair(hyps: list[str], refs: list[str]) -> tuple[float, float]:
    import sacrebleu

    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return round(chrf.score, 2), round(bleu.score, 2)


def verify_receipt(hyp_file: str) -> tuple[dict[str, Any] | None, str | None]:
    """(receipt, None) when verifiable, (None, reason) otherwise."""
    rpath = receipt_path(hyp_file)
    if not os.path.exists(rpath):
        return None, "no receipt"
    with open(rpath, encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("hypothesis_sha256") != file_sha256(hyp_file):
        return None, "hypothesis bytes do not match receipt hash"
    if receipt.get("system") not in SYSTEMS:
        return None, f"receipt system {receipt.get('system')!r} is not registered"
    try:
        manifest = load_manifest(receipt["corpus"], receipt["lang"])
    except (FileNotFoundError, KeyError) as exc:
        return None, f"manifest unavailable: {exc}"
    if receipt.get("corpus_manifest_sha256") != manifest.get("contract_sha256"):
        return None, "corpus manifest changed since this run; re-run the system"
    return receipt, None


def score_all() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for hyp_file in sorted(glob.glob(os.path.join(OUT, "*.hyp"))):
        name = os.path.basename(hyp_file)
        match = _HYP_RE.match(name)
        if not match:
            excluded.append({"file": name, "reason": "unrecognized filename"})
            continue
        receipt, reason = verify_receipt(hyp_file)
        if receipt is None:
            excluded.append({"file": name, "reason": reason or "unverifiable"})
            continue

        corpus, lang, direction, system_id = (
            match["corpus"], match["lang"], match["direction"], match["system"],
        )
        input_path, ref_path_ = direction_io(corpus, lang, direction)
        sources = read_lines(input_path)
        refs = read_lines(ref_path_)
        hyps = read_lines(hyp_file)
        limit = receipt.get("limit")
        if limit:
            sources, refs = sources[:limit], refs[:limit]
        if len(hyps) != len(refs):
            excluded.append({
                "file": name,
                "reason": f"hyp/ref length mismatch {len(hyps)} vs {len(refs)}",
            })
            continue

        chrf, bleu = score_pair(hyps, refs)
        entry = SYSTEMS[system_id]
        row: dict[str, Any] = {
            "system": system_id,
            "label": entry["label"],
            "vendor": entry["vendor"],
            "tier": entry["tier"],
            "license": entry["license"],
            "benchmark_only": bool(entry.get("benchmark_only", False)),
            "pin_status": receipt["pin_status"],
            "model_reported": receipt["model_reported"],
            "corpus": corpus,
            "lang": lang,
            "language_name": LANGS[lang]["name"],
            "direction": direction,
            "n": len(hyps),
            "partial_slice": bool(limit),
            "fails": receipt.get("fails", 0),
            "chrf_pp": chrf,
            "bleu": bleu,
            "receipt": os.path.basename(receipt_path(hyp_file)),
            "created_utc": receipt.get("created_utc"),
            "harness_version": receipt.get("harness_version"),
        }
        row.update(track_c_metrics(sources, hyps, expected_target_lang(lang, direction)))
        results.append(row)

    results.sort(key=lambda r: (r["direction"], r["lang"], r["corpus"], -r["chrf_pp"]))
    payload = {
        "schema": SCHEMA_SCORES,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "results": results,
        "excluded": excluded,
    }
    os.makedirs(SCORES, exist_ok=True)
    out_path = os.path.join(SCORES, "scores.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"scored {len(results)} runs, excluded {len(excluded)} -> {out_path}")
    for item in excluded:
        print(f"  EXCLUDED {item['file']}: {item['reason']}")
    return payload
