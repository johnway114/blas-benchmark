"""Runner behaviour: what ends up in the hypothesis file and its receipt.

No network. `_translate_one` is replaced by a fake vendor, which is the only
thing in the loop that would talk to one.
"""
import json

import pytest

import celticbench.lib as lib
import celticbench.runner as runner
from celticbench.lib import (
    SCHEMA_RECEIPT, file_sha256, hyp_path, read_lines, receipt_path, write_lines,
    write_manifest,
)
from celticbench.providers import ProviderError, Translation

SRC = ["Good morning.", "The sea is cold today.", "I saw three birds.", "Come here."]
REF = ["Myttin da.", "Yeyn yw an mor hedhyw.", "My a welas teyr edhen.", "Deus omma."]


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "EVAL", str(tmp_path / "eval"))
    monkeypatch.setattr(lib, "MANIFESTS", str(tmp_path / "manifests"))
    monkeypatch.setattr(lib, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "out" / "cache"))
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    write_lines(lib.eval_src_path("tatoeba", "kw"), SRC)
    write_lines(lib.eval_ref_path("tatoeba", "kw"), REF)
    write_manifest("tatoeba", "kw", len(SRC), {"dataset": "test-fixture"})
    return tmp_path


def _fake_vendor(monkeypatch, *, model="gpt-5.6-sol-2026-07-01", fail_on=(), calls=None):
    def translate(entry, lang, direction, text):
        if calls is not None:
            calls.append(text)
        if text in fail_on:
            raise ProviderError("HTTP 500: upstream exploded")
        return Translation(
            text=f"[{lang}] {text}",
            model_reported=model,
            decoding_used=dict(entry["decoding"]),
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )

    monkeypatch.setattr(runner, "_translate_one", translate)


def _receipt_for(out_path):
    with open(receipt_path(out_path), encoding="utf-8") as handle:
        return json.load(handle)


def test_run_writes_hypotheses_and_a_binding_receipt(workspace, monkeypatch):
    _fake_vendor(monkeypatch)
    out = runner.run_system("gpt-5.6-sol", "tatoeba", "kw", "en-xx")
    assert read_lines(out) == [f"[kw] {line}" for line in SRC]

    receipt = _receipt_for(out)
    assert receipt["schema"] == SCHEMA_RECEIPT
    assert receipt["hypothesis_sha256"] == file_sha256(out)
    assert receipt["eval_input_sha256"] == file_sha256(lib.eval_src_path("tatoeba", "kw"))
    assert receipt["eval_reference_sha256"] == file_sha256(lib.eval_ref_path("tatoeba", "kw"))
    assert receipt["n"] == len(SRC) and receipt["fails"] == 0
    assert receipt["model_reported"] == "gpt-5.6-sol-2026-07-01"
    assert receipt["usage"]["requests"] == len(SRC)
    assert receipt["usage"]["total_tokens"] == 10 * len(SRC)
    assert receipt["usage"]["source_chars"] == sum(len(s) for s in SRC)
    assert receipt["runtime"]["python"]


def test_second_run_is_served_from_cache_without_losing_provenance(workspace, monkeypatch):
    calls: list[str] = []
    _fake_vendor(monkeypatch, calls=calls)
    runner.run_system("gpt-5.6-sol", "tatoeba", "kw", "en-xx")
    assert len(calls) == len(SRC)

    calls.clear()
    out = runner.run_system("gpt-5.6-sol", "tatoeba", "kw", "en-xx")
    assert calls == [], "a cached line must not be re-requested"
    receipt = _receipt_for(out)
    assert receipt["usage"]["cache_hits"] == len(SRC)
    assert receipt["usage"]["requests"] == 0
    # The whole point of caching provenance: this receipt still names the model
    # that produced the bytes, even though this run called nobody.
    assert receipt["model_reported"] == "gpt-5.6-sol-2026-07-01"


def test_failed_line_becomes_blank_and_is_counted(workspace, monkeypatch):
    _fake_vendor(monkeypatch, fail_on={SRC[1]})
    out = runner.run_system("gpt-5.6-sol", "tatoeba", "kw", "en-xx")
    lines = read_lines(out)
    assert lines[1] == ""
    assert lines[0] and lines[2] and lines[3]
    receipt = _receipt_for(out)
    assert receipt["fails"] == 1
    assert receipt["n"] == len(SRC), "a failed line is blank, never dropped"


def test_workers_preserve_line_order(workspace, monkeypatch):
    _fake_vendor(monkeypatch)
    out = runner.run_system("gpt-5.6-sol", "tatoeba", "kw", "en-xx", workers=4)
    assert read_lines(out) == [f"[kw] {line}" for line in SRC]


def test_partial_slice_gets_its_own_file(workspace, monkeypatch):
    _fake_vendor(monkeypatch)
    full = runner.run_system("gpt-5.6-sol", "tatoeba", "kw", "en-xx")
    partial = runner.run_system("gpt-5.6-sol", "tatoeba", "kw", "en-xx", limit=2)
    assert partial.endswith(".limit2.hyp") and partial != full
    assert len(read_lines(partial)) == 2
    assert len(read_lines(full)) == len(SRC), "the smoke run must not clobber the full run"
    assert _receipt_for(partial)["partial_slice"] is True
    assert _receipt_for(full)["partial_slice"] is False


def test_vendor_switching_model_mid_run_is_recorded(workspace, monkeypatch):
    """A silent alias swap must be visible in the receipt, not averaged away."""
    seen: list[str] = []

    def translate(entry, lang, direction, text):
        model = "gpt-5.6-sol-2026-07-01" if len(seen) < 2 else "gpt-5.6-sol-2026-08-01"
        seen.append(model)
        return Translation(text=text, model_reported=model,
                           decoding_used=dict(entry["decoding"]), usage={})

    monkeypatch.setattr(runner, "_translate_one", translate)
    out = runner.run_system("gpt-5.6-sol", "tatoeba", "kw", "en-xx")
    receipt = _receipt_for(out)
    assert receipt["model_reported"] == "mixed"
    assert receipt["model_reported_variants"] == [
        "gpt-5.6-sol-2026-08-01", "gpt-5.6-sol-2026-07-01",
    ] or receipt["model_reported_variants"] == [
        "gpt-5.6-sol-2026-07-01", "gpt-5.6-sol-2026-08-01",
    ]


def test_unsupported_combination_refuses_before_any_request(workspace, monkeypatch):
    calls: list[str] = []
    _fake_vendor(monkeypatch, calls=calls)
    with pytest.raises(SystemExit, match="refusing"):
        runner.run_system("google-translate-v2", "tatoeba", "kw", "en-xx")
    assert calls == []


def test_local_run_is_not_counted_as_hosted_requests(workspace, monkeypatch):
    """`requests` is what a bill is made of; local inference costs none."""
    monkeypatch.setattr(runner.hf_local, "translate_batch",
                        lambda entry, texts, lang, direction:
                        ([f"<{t}>" for t in texts], "Helsinki-NLP/opus-mt-en-cel@e79438534e0b"))
    out = runner.run_system("opus-mt-cel", "tatoeba", "kw", "en-xx")
    receipt = _receipt_for(out)
    assert receipt["usage"]["requests"] == 0
    assert receipt["usage"]["cache_hits"] == 0
    assert receipt["model_reported"] == "Helsinki-NLP/opus-mt-en-cel@e79438534e0b"
    assert receipt["prompt_sha256"] is None, "a seq2seq anchor takes no prompt"
