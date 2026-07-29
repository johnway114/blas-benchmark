"""Offline scoring E2E: manifests + hypotheses + receipts -> scores.json.

Uses real sacrebleu, real receipt verification, real files in tmp dirs.
The 'system under test' is a handcrafted hypothesis file with a receipt in
exactly the shape runner.py writes, which is precisely what score_all trusts.

Most of these tests are about refusal. A leaderboard row is a claim that
specific bytes were produced by a specific model against a specific corpus
under a specific prompt; every test below breaks exactly one link in that
chain and asserts the row disappears with a reason.
"""
import json
import os

import pytest
import sacrebleu

import celticbench.lib as lib
import celticbench.scoring as scoring
from celticbench.lib import (
    METHOD_VERSION, SCHEMA_RECEIPT, direction_io, file_sha256, hyp_path,
    receipt_path, sha256_json, write_lines, write_manifest,
)
from celticbench.prompt import CHAT_DECODING, PROMPT_SHA256, PROMPT_TEMPLATE
from celticbench.registry import SYSTEMS

SRC = ["Good morning.", "The sea is cold today.", "I saw three birds."]
REF = ["Myttin da.", "Yeyn yw an mor hedhyw.", "My a welas teyr edhen."]
HYP = ["Myttin da.", "Yeyn yw an mor.", "My a welas teyr edhen."]


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "EVAL", str(tmp_path / "eval"))
    monkeypatch.setattr(lib, "MANIFESTS", str(tmp_path / "manifests"))
    monkeypatch.setattr(lib, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(scoring, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(scoring, "SCORES", str(tmp_path / "scores"))
    write_lines(lib.eval_src_path("tatoeba", "kw"), SRC)
    write_lines(lib.eval_ref_path("tatoeba", "kw"), REF)
    manifest = write_manifest("tatoeba", "kw", len(SRC), {"dataset": "test-fixture"})
    return tmp_path, manifest


def _write_run(manifest, system_id="gpt-5.6-sol", hyps=HYP, direction="en-xx",
               lang="kw", limit=None, **overrides):
    """Write a hypothesis plus the receipt runner.py would have written."""
    entry = SYSTEMS[system_id]
    out = hyp_path("tatoeba", lang, direction, system_id, limit)
    write_lines(out, hyps)
    input_path, reference_path = direction_io("tatoeba", lang, direction)
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "method_version": METHOD_VERSION,
        "system": system_id,
        "provider": entry["provider"],
        "vendor": entry["vendor"],
        "tier": entry["tier"],
        "model_requested": "test-model",
        "model_reported": "test-model@1",
        "model_reported_variants": ["test-model@1"],
        "revision": None,
        "pin_status": entry["pin_status"],
        "license": entry["license"],
        "reasoning": None,
        "corpus": "tatoeba",
        "lang": lang,
        "direction": direction,
        "n": len(hyps),
        "limit": limit,
        "partial_slice": limit is not None,
        "fails": 0,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_sha256": PROMPT_SHA256,
        "decoding_declared": dict(entry["decoding"]),
        "decoding_declared_sha256": sha256_json(entry["decoding"]),
        "decoding": dict(entry["decoding"]),
        "decoding_sha256": sha256_json(entry["decoding"]),
        "decoding_deviations": [],
        "corpus_manifest_sha256": manifest["contract_sha256"],
        "eval_input_sha256": file_sha256(input_path),
        "eval_reference_sha256": file_sha256(reference_path),
        "hypothesis_file": os.path.basename(out),
        "hypothesis_sha256": file_sha256(out),
        "usage": {"requests": len(hyps), "cache_hits": 0, "total_tokens": 0},
        "created_utc": "2026-07-29T00:00:00+00:00",
        "harness_version": "test",
        "runtime": {"python": "3.14.0"},
    }
    receipt.update(overrides)
    _save(out, receipt)
    return out, receipt


def _save(hyp_file, receipt):
    with open(receipt_path(hyp_file), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle)


def _only_exclusion(payload):
    assert payload["results"] == [], payload["results"]
    assert len(payload["excluded"]) == 1, payload["excluded"]
    return payload["excluded"][0]["reason"]


def _lid_available():
    from celticbench.langid import MODEL_PATH
    return os.path.exists(MODEL_PATH)


@pytest.mark.skipif(not _lid_available(), reason="lid model not downloaded; run bench.py prepare")
def test_score_all_end_to_end(workspace):
    _, manifest = workspace
    _write_run(manifest)
    payload = scoring.score_all()
    assert payload["excluded"] == []
    assert len(payload["results"]) == 1
    row = payload["results"][0]

    expected_chrf = round(sacrebleu.corpus_chrf(HYP, [REF], word_order=2).score, 2)
    expected_bleu = round(sacrebleu.corpus_bleu(HYP, [REF]).score, 2)
    assert row["chrf_pp"] == expected_chrf
    assert row["bleu"] == expected_bleu
    assert row["n"] == 3
    assert row["blank_rate"] == 0.0
    assert row["copy_rate"] == 0.0
    assert row["language_name"] == "Cornish"
    assert os.path.exists(os.path.join(scoring.SCORES, "scores.json"))


def test_scores_record_metric_and_detector_provenance(workspace):
    _, manifest = workspace
    _write_run(manifest)
    payload = scoring.score_all()
    assert payload["method_version"] == "v3"
    assert len(payload["coverage"]) == 72
    assert all(cell["supported"] and cell["reason"] is None
               for cell in payload["coverage"])
    # A chrF++ number without its sacrebleu signature is not reproducible.
    assert "chrF" in payload["metric_signatures"]["chrf_pp"]
    assert "BLEU" in payload["metric_signatures"]["bleu"]
    assert payload["runtime"]["python"]
    assert payload["coverage"], "coverage must say which cells are even offered"


def test_tampered_hypothesis_is_excluded(workspace):
    _, manifest = workspace
    out, _ = _write_run(manifest)
    with open(out, "a", encoding="utf-8") as handle:
        handle.write("injected line\n")
    assert "do not match receipt hash" in _only_exclusion(scoring.score_all())


def test_missing_receipt_is_excluded(workspace):
    write_lines(hyp_path("tatoeba", "kw", "en-xx", "gpt-5.6-sol"), HYP)
    assert _only_exclusion(scoring.score_all()) == "no receipt"


def test_old_schema_receipt_is_excluded(workspace):
    _, manifest = workspace
    _write_run(manifest, schema="celticbench.receipt.v1")
    assert "is not celticbench.receipt.v2" in _only_exclusion(scoring.score_all())


def test_manifest_drift_since_run_is_excluded(workspace):
    _, manifest = workspace
    _write_run(manifest)
    write_lines(lib.eval_ref_path("tatoeba", "kw"),
                ["Myttin da.", "CHANGED.", "My a welas teyr edhen."])
    write_manifest("tatoeba", "kw", 3, {"dataset": "test-fixture"}, force=True)
    assert "manifest changed since this run" in _only_exclusion(scoring.score_all())


def test_edited_eval_file_without_manifest_rewrite_is_excluded(workspace):
    _, manifest = workspace
    _write_run(manifest)
    # The manifest still claims the old hash, so the corpus contract itself is
    # violated: scoring must refuse rather than score against edited gold.
    write_lines(lib.eval_ref_path("tatoeba", "kw"),
                ["Myttin da.", "EDITED IN PLACE.", "My a welas teyr edhen."])
    assert "no longer match the manifest" in _only_exclusion(scoring.score_all())


def test_unregistered_system_receipt_is_excluded(workspace):
    _, manifest = workspace
    out, receipt = _write_run(manifest)
    receipt["system"] = "mystery-model"
    renamed = out.replace("gpt-5.6-sol", "mystery-model")
    os.rename(out, renamed)
    os.remove(receipt_path(out))
    _save(renamed, receipt)
    assert "not registered" in _only_exclusion(scoring.score_all())


def test_filename_that_disagrees_with_receipt_is_excluded(workspace):
    _, manifest = workspace
    out, receipt = _write_run(manifest)
    renamed = out.replace("en-xx", "xx-en")
    os.rename(out, renamed)
    os.remove(receipt_path(out))
    _save(renamed, receipt)
    reason = _only_exclusion(scoring.score_all())
    assert "filename says direction='xx-en'" in reason


def test_partial_slice_does_not_collide_with_the_full_run(workspace):
    _, manifest = workspace
    full, _ = _write_run(manifest)
    partial, _ = _write_run(manifest, hyps=HYP[:2], limit=2)
    assert full != partial and os.path.exists(full) and os.path.exists(partial)
    payload = scoring.score_all()
    assert payload["excluded"] == []
    slices = sorted(row["partial_slice"] for row in payload["results"])
    assert slices == [False, True]


def test_slice_marker_must_match_the_receipt(workspace):
    _, manifest = workspace
    # A full-run receipt renamed to look like a smoke slice, or vice versa.
    out, receipt = _write_run(manifest, hyps=HYP[:2], limit=2)
    renamed = out.replace(".limit2", "")
    os.rename(out, renamed)
    os.remove(receipt_path(out))
    _save(renamed, receipt)
    assert "slice does not match" in _only_exclusion(scoring.score_all())


def test_prompt_change_since_run_is_excluded(workspace):
    _, manifest = workspace
    _write_run(manifest, system_id="gpt-5.6-sol", prompt_sha256="0" * 64)
    assert "prompt changed since this run" in _only_exclusion(scoring.score_all())


def test_decoding_contract_change_since_run_is_excluded(workspace):
    _, manifest = workspace
    stale = dict(CHAT_DECODING, temperature=0.7)
    _write_run(manifest, system_id="gpt-5.6-sol",
               decoding_declared=stale, decoding_declared_sha256=sha256_json(stale))
    assert "declared decoding changed" in _only_exclusion(scoring.score_all())


def test_receipt_that_contradicts_itself_is_excluded(workspace):
    _, manifest = workspace
    _write_run(manifest, decoding_sha256="0" * 64)
    assert "does not match its own decoding block" in _only_exclusion(scoring.score_all())


def test_receipt_line_count_must_match_the_file(workspace):
    _, manifest = workspace
    _write_run(manifest, n=99)
    assert "receipt claims n=99" in _only_exclusion(scoring.score_all())




def test_v2_receipt_is_excluded_from_method_v3(workspace):
    _, manifest = workspace
    _write_run(manifest, method_version="v2")
    reason = _only_exclusion(scoring.score_all())
    assert "made under method 'v2'" in reason
    assert "published method is v3" in reason
