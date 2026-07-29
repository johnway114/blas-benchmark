"""Manifest contract: eval files are pinned; drift is a hard error."""
import pytest

import celticbench.lib as lib
from celticbench.lib import verify_manifest, write_lines, write_manifest


@pytest.fixture()
def tiny_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "EVAL", str(tmp_path / "eval"))
    monkeypatch.setattr(lib, "MANIFESTS", str(tmp_path / "manifests"))
    src = ["Good morning.", "The sea is cold."]
    ref = ["Myttin da.", "Yeyn yw an mor."]
    write_lines(lib.eval_src_path("tatoeba", "kw"), src)
    write_lines(lib.eval_ref_path("tatoeba", "kw"), ref)
    manifest = write_manifest("tatoeba", "kw", len(src), {"dataset": "test-fixture"})
    return manifest


def test_verify_roundtrip(tiny_corpus):
    manifest = verify_manifest("tatoeba", "kw")
    assert manifest["n"] == 2
    assert manifest["contract_sha256"] == tiny_corpus["contract_sha256"]


def test_edited_eval_file_is_refused(tiny_corpus):
    with open(lib.eval_ref_path("tatoeba", "kw"), "a", encoding="utf-8") as handle:
        handle.write("tampered line\n")
    with pytest.raises(ValueError, match="does not match committed manifest|line count"):
        verify_manifest("tatoeba", "kw")


def test_edited_manifest_is_refused(tiny_corpus, tmp_path):
    path = lib.manifest_path("tatoeba", "kw")
    text = open(path, encoding="utf-8").read().replace('"n":2', '"n":3')
    open(path, "w", encoding="utf-8").write(text)
    with pytest.raises(ValueError, match="contract hash mismatch"):
        verify_manifest("tatoeba", "kw")


def test_rebuild_drift_requires_force(tiny_corpus):
    write_lines(lib.eval_ref_path("tatoeba", "kw"), ["Myttin da.", "CHANGED."])
    with pytest.raises(ValueError, match="do not match the committed manifest"):
        write_manifest("tatoeba", "kw", 2, {"dataset": "test-fixture"})
    manifest = write_manifest("tatoeba", "kw", 2, {"dataset": "test-fixture"}, force=True)
    assert verify_manifest("tatoeba", "kw")["contract_sha256"] == manifest["contract_sha256"]


def test_missing_manifest_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "MANIFESTS", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="no committed manifest"):
        verify_manifest("flores", "ga")
