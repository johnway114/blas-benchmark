"""Score every hypothesis that carries a verifiable receipt.

Fail-closed. A run is scored only when every claim binding it to a published
number still holds: the hypothesis bytes match the receipt hash, the filename
matches the receipt, the eval files still hash to the committed manifest, the
line counts agree, and the prompt and decoding contracts are the ones the run
recorded. Anything else is listed under `excluded` with its reason. Nothing is
silently dropped and nothing is silently included.

The strictness is the point: a published leaderboard row must be
reconstructible from the committed corpus contract, the committed hypothesis,
and its receipt, without trusting the machine that produced it.

Metrics:
  Track A (reference): chrF++ (word_order=2) and BLEU via sacrebleu, with the
  sacrebleu signatures recorded next to the numbers.
  Track C (reference-free): off_target_rate over lines that could be detected
  at all, plus blank, copy and repetition rates and a length-ratio diagnostic.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import statistics
from typing import Any

from .lib import (
    ALL_LANGS, LANGS, METHOD_VERSION, OUT, SCHEMA_RECEIPT, SCHEMA_SCORES, SCORES,
    all_corpora, direction_io, expected_target_lang, file_sha256, load_manifest,
    parse_hyp_name, read_lines, receipt_path, runtime_versions, sha256_json,
    verify_manifest,
)
from .prompt import PROMPT_SHA256
from .registry import SYSTEMS, supported

# A hypothesis whose length is wildly out of step with its source is broken
# even when no reference exists: truncation on one side, a bolted-on
# explanation on the other. The band is deliberately wide.
LENGTH_RATIO_BAND = (0.5, 2.0)


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


def _length_ratios(sources: list[str], hyps: list[str]) -> dict[str, Any]:
    ratios = [len(h) / len(s) for s, h in zip(sources, hyps) if s.strip() and h.strip()]
    if not ratios:
        return {}
    low, high = LENGTH_RATIO_BAND
    ordered = sorted(ratios)
    return {
        "length_ratio_median": round(statistics.median(ordered), 3),
        "length_ratio_p10": round(ordered[int(0.10 * (len(ordered) - 1))], 3),
        "length_ratio_p90": round(ordered[int(0.90 * (len(ordered) - 1))], 3),
        "length_ratio_band": [low, high],
        "length_ratio_outlier_rate": round(
            sum(1 for r in ratios if r < low or r > high) / len(ratios), 4),
        "length_ratio_n": len(ratios),
    }


def track_c_metrics(sources: list[str], hyps: list[str], expected_lang: str) -> dict[str, Any]:
    """Reference-free integrity metrics.

    off_target_rate is a share of the lines the detector could actually judge:
    a blank line is a blank, not a wrong language, and counting it in the
    denominator silently dilutes off-target as blank_rate rises. blank_rate,
    copy_rate and repetition_rate stay over all lines, where they belong.
    """
    from .langid import CONFIDENCE_THRESHOLD, is_off_target

    n = len(hyps)
    if n == 0:
        return {}
    blank = sum(1 for h in hyps if not h.strip())
    copied = sum(1 for s, h in zip(sources, hyps) if h.strip() and _norm(s) == _norm(h))
    repetitive = sum(1 for h in hyps if _is_repetitive(h))
    judged = n - blank
    off_target = sum(1 for h in hyps if is_off_target(h, expected_lang))
    metrics = {
        "blank_rate": round(blank / n, 4),
        "copy_rate": round(copied / n, 4),
        "repetition_rate": round(repetitive / n, 4),
        "off_target_rate": round(off_target / judged, 4) if judged else None,
        "off_target_n": judged,
        "off_target_expected_lang": expected_lang,
        "off_target_confidence_threshold": CONFIDENCE_THRESHOLD,
    }
    metrics.update(_length_ratios(sources, hyps))
    return metrics


def _metrics():
    import sacrebleu

    return sacrebleu.CHRF(word_order=2), sacrebleu.BLEU()


def score_pair(hyps: list[str], refs: list[str]) -> tuple[float, float, dict[str, str]]:
    """chrF++, BLEU, and sacrebleu's own signatures for both.

    The signature is only available from a metric that has actually scored
    something (it reports the reference count), which is the right coupling:
    we publish the signature of the computation that produced these numbers,
    not of a metric object we constructed to look at.
    """
    chrf_metric, bleu_metric = _metrics()
    chrf = chrf_metric.corpus_score(hyps, [refs])
    bleu = bleu_metric.corpus_score(hyps, [refs])
    # sacrebleu keeps the metric name on the score and the parameters on the
    # signature; a published signature needs both to identify the metric.
    signatures = {
        "chrf_pp": f"{chrf.name}|{chrf_metric.get_signature()}",
        "bleu": f"{bleu.name}|{bleu_metric.get_signature()}",
    }
    return round(chrf.score, 2), round(bleu.score, 2), signatures


def _lid_provenance() -> dict[str, Any]:
    """Which detector build produced the off-target numbers."""
    from .lib import HERE
    from .langid import CONFIDENCE_THRESHOLD, MODEL_PATH

    provenance: dict[str, Any] = {"confidence_threshold": CONFIDENCE_THRESHOLD}
    path = os.path.join(HERE, "manifests", "lid-validation.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        provenance["model"] = report.get("model")
        provenance["model_sha256"] = report.get("model_sha256")
    elif os.path.exists(MODEL_PATH):
        provenance["model"] = os.path.basename(MODEL_PATH)
        provenance["model_sha256"] = file_sha256(MODEL_PATH)
    return provenance


def verify_receipt(hyp_file: str) -> tuple[dict[str, Any] | None, str | None]:
    """(receipt, None) when the run is still publishable, (None, reason) otherwise."""
    name = os.path.basename(hyp_file)
    parsed = parse_hyp_name(name)
    if parsed is None:
        return None, "unrecognized filename"

    rpath = receipt_path(hyp_file)
    if not os.path.exists(rpath):
        return None, "no receipt"
    with open(rpath, encoding="utf-8") as handle:
        receipt = json.load(handle)

    if receipt.get("schema") != SCHEMA_RECEIPT:
        return None, f"receipt schema {receipt.get('schema')!r} is not {SCHEMA_RECEIPT}"
    if receipt.get("method_version") != METHOD_VERSION:
        return None, (f"run was made under method {receipt.get('method_version')!r}, "
                      f"the published method is {METHOD_VERSION}; re-run the system")
    if receipt.get("hypothesis_sha256") != file_sha256(hyp_file):
        return None, "hypothesis bytes do not match receipt hash"

    system_id = receipt.get("system")
    if system_id not in SYSTEMS:
        return None, f"receipt system {system_id!r} is not registered"

    # The filename is what the leaderboard groups on; the receipt is what it
    # trusts. A disagreement means one of them was moved or edited.
    for field in ("corpus", "lang", "direction", "system"):
        if receipt.get(field) != parsed[field]:
            return None, (f"filename says {field}={parsed[field]!r} but receipt says "
                          f"{receipt.get(field)!r}")
    if receipt.get("limit") != parsed["limit"]:
        return None, "filename slice does not match the receipt limit"

    entry = SYSTEMS[system_id]
    ok, reason = supported(system_id, receipt["lang"], receipt["direction"])
    if not ok:
        return None, f"registry no longer supports this combination: {reason}"

    try:
        manifest = load_manifest(receipt["corpus"], receipt["lang"])
    except (FileNotFoundError, KeyError) as exc:
        return None, f"manifest unavailable: {exc}"
    if receipt.get("corpus_manifest_sha256") != manifest.get("contract_sha256"):
        return None, "corpus manifest changed since this run; re-run the system"
    try:
        verify_manifest(receipt["corpus"], receipt["lang"])
    except (FileNotFoundError, ValueError) as exc:
        return None, f"eval files no longer match the manifest: {exc}"

    input_path, reference_path = direction_io(receipt["corpus"], receipt["lang"],
                                              receipt["direction"])
    if receipt.get("eval_input_sha256") != file_sha256(input_path):
        return None, "eval input file changed since this run"
    if receipt.get("eval_reference_sha256") != file_sha256(reference_path):
        return None, "eval reference file changed since this run"

    if receipt.get("prompt_sha256") != PROMPT_SHA256:
        return None, "prompt changed since this run; that is a new leaderboard version"
    if receipt.get("decoding_declared_sha256") != sha256_json(entry["decoding"]):
        return None, "declared decoding changed since this run; re-run the system"
    if receipt.get("decoding_sha256") != sha256_json(receipt.get("decoding")):
        return None, "receipt decoding hash does not match its own decoding block"

    return receipt, None


def coverage() -> list[dict[str, Any]]:
    """Every (system, lang, direction) with whether it can be run at all.

    Published tables need to distinguish "this system scored badly" from
    "this system does not offer this language", and both from "not run yet".
    """
    rows: list[dict[str, Any]] = []
    for system_id in SYSTEMS:
        for lang in ALL_LANGS:
            for direction in ("en-xx", "xx-en"):
                ok, reason = supported(system_id, lang, direction)
                rows.append({
                    "system": system_id,
                    "lang": lang,
                    "direction": direction,
                    "supported": ok,
                    "reason": reason,
                })
    return rows


def score_all() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    signatures: dict[str, str] = {}

    for hyp_file in sorted(glob.glob(os.path.join(OUT, "*.hyp"))):
        name = os.path.basename(hyp_file)
        receipt, reason = verify_receipt(hyp_file)
        if receipt is None:
            excluded.append({"file": name, "reason": reason or "unverifiable"})
            continue

        corpus, lang, direction = receipt["corpus"], receipt["lang"], receipt["direction"]
        system_id = receipt["system"]
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
        if receipt.get("n") != len(hyps):
            excluded.append({
                "file": name,
                "reason": f"receipt claims n={receipt.get('n')} but file has {len(hyps)} lines",
            })
            continue

        chrf, bleu, signatures = score_pair(hyps, refs)
        entry = SYSTEMS[system_id]
        row: dict[str, Any] = {
            "system": system_id,
            "label": entry["label"],
            "vendor": entry["vendor"],
            "tier": entry["tier"],
            "license": entry["license"],
            "benchmark_only": bool(entry.get("benchmark_only", False)),
            "pin_status": receipt["pin_status"],
            "model_requested": receipt.get("model_requested"),
            "model_reported": receipt["model_reported"],
            "reasoning": receipt.get("reasoning"),
            "decoding_deviations": receipt.get("decoding_deviations") or [],
            "corpus": corpus,
            "lang": lang,
            "language_name": LANGS[lang]["name"],
            "direction": direction,
            "n": len(hyps),
            "partial_slice": bool(limit),
            "fails": receipt.get("fails", 0),
            "chrf_pp": chrf,
            "bleu": bleu,
            "usage": receipt.get("usage", {}),
            "receipt": os.path.basename(receipt_path(hyp_file)),
            "hypothesis_file": name,
            "created_utc": receipt.get("created_utc"),
            "harness_version": receipt.get("harness_version"),
            "runtime": receipt.get("runtime", {}),
        }
        row.update(track_c_metrics(sources, hyps, expected_target_lang(lang, direction)))
        results.append(row)

    results.sort(key=lambda r: (r["direction"], r["lang"], r["corpus"], -r["chrf_pp"]))
    payload = {
        "schema": SCHEMA_SCORES,
        "method_version": METHOD_VERSION,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "metric_signatures": signatures,
        "langid": _lid_provenance(),
        "runtime": runtime_versions(),
        "corpora": list(all_corpora()),
        "results": results,
        "coverage": coverage(),
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
