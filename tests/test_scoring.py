"""Offline scoring E2E: manifests + hypotheses + receipts -> scores.json.

Uses real sacrebleu, real receipt verification, real files in tmp dirs.
The 'system under test' is a handcrafted hypothesis file with a receipt in
exactly the shape runner.py writes, which is precisely what score_all trusts.
"""
import json
import os

import pytest
import sacrebleu

import celticbench.lib as lib
import celticbench.scoring as scoring
from celticbench.lib import (
    SCHEMA_RECEIPT, file_sha256, hyp_path, receipt_path, sha256_json, write_lines,
    write_manifest,
)

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


def _write_run(manifest, system_id="opus-mt-cel", hyps=HYP, direction="en-xx"):
    out = hyp_path("tatoeba", "kw", direction, system_id)
    write_lines(out, hyps)
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "system": system_id,
        "provider": "hf_local",
        "vendor": "Helsinki-NLP",
        "model_requested": "Helsinki-NLP/opus-mt-en-cel",
        "model_reported": "Helsinki-NLP/opus-mt-en-cel@e794385",
        "revision": "e79438534e0be0f4efa2585e3b24393bf40def95",
        "pin_status": "verified",
        "license": "Apache-2.0",
        "benchmark_only": False,
        "corpus": "tatoeba",
        "lang": "kw",
        "direction": direction,
        "n": len(hyps),
        "limit": None,
        "fails": 0,
        "prompt_template": None,
        "prompt_sha256": None,
        "decoding": {"num_beams": 4},
        "decoding_sha256": sha256_json({"num_beams": 4}),
        "corpus_manifest_sha256": manifest["contract_sha256"],
        "hypothesis_sha256": file_sha256(out),
        "created_utc": "2026-07-29T00:00:00+00:00",
        "harness_version": "test",
    }
    with open(receipt_path(out), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle)
    return out, receipt


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


def test_tampered_hypothesis_is_excluded(workspace):
    _, manifest = workspace
    out, _ = _write_run(manifest)
    with open(out, "a", encoding="utf-8") as handle:
        handle.write("injected line\n")
    payload = scoring.score_all()
    assert payload["results"] == []
    assert len(payload["excluded"]) == 1
    assert "do not match receipt hash" in payload["excluded"][0]["reason"]


def test_missing_receipt_is_excluded(workspace):
    write_lines(hyp_path("tatoeba", "kw", "en-xx", "opus-mt-cel"), HYP)
    payload = scoring.score_all()
    assert payload["results"] == []
    assert payload["excluded"][0]["reason"] == "no receipt"


def test_manifest_drift_since_run_is_excluded(workspace):
    tmp_path, manifest = workspace
    _write_run(manifest)
    write_lines(lib.eval_ref_path("tatoeba", "kw"), ["Myttin da.", "CHANGED.", "My a welas teyr edhen."])
    write_manifest("tatoeba", "kw", 3, {"dataset": "test-fixture"}, force=True)
    payload = scoring.score_all()
    assert payload["results"] == []
    assert "manifest changed since this run" in payload["excluded"][0]["reason"]


def test_unregistered_system_receipt_is_excluded(workspace):
    _, manifest = workspace
    out, receipt = _write_run(manifest)
    receipt["system"] = "mystery-model"
    with open(receipt_path(out), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle)
    renamed = out.replace("opus-mt-cel", "mystery-model")
    os.rename(out, renamed)
    os.rename(receipt_path(out), receipt_path(renamed))
    # hash still matches the moved file, so the registry check is what fires
    payload = scoring.score_all()
    assert payload["results"] == []
    assert "not registered" in payload["excluded"][0]["reason"]
