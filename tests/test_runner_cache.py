"""Line-level cache: deterministic keys, provenance, crash-safe roundtrip."""
import json

import celticbench.runner as runner
from celticbench.providers import Translation
from celticbench.registry import SYSTEMS


def _translation(text, model="gpt-5.6-sol-2026-07-01"):
    return Translation(text=text, model_reported=model,
                       decoding_used={"temperature": 0.0},
                       usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})


def test_cache_roundtrip_keeps_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    runner._append_cache("test-system", "k1", _translation("Myttin da."))
    runner._append_cache("test-system", "k2", _translation("Bore da."))
    cache = runner._load_cache("test-system")
    assert [record["v"] for record in cache.values()] == ["Myttin da.", "Bore da."]
    # Without the reported model and usage, a fully cached rerun would have to
    # invent the provenance in its receipt.
    assert cache["k1"]["m"] == "gpt-5.6-sol-2026-07-01"
    assert cache["k1"]["u"]["total_tokens"] == 14


def test_torn_tail_line_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    runner._append_cache("test-system", "k1", _translation("value"))
    with open(runner._cache_file("test-system"), "a", encoding="utf-8") as handle:
        handle.write('{"k": "k2", "v": "trunc')  # simulated crash mid-write
    assert list(runner._load_cache("test-system")) == ["k1"]


def test_cache_entry_without_provenance_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    runner._append_cache("test-system", "k1", _translation("fresh"))
    with open(runner._cache_file("test-system"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"k": "legacy", "v": "old text"}) + "\n")
    cache = runner._load_cache("test-system")
    assert "legacy" not in cache, "a line we cannot attribute must be re-fetched"
    assert "k1" in cache


def test_cache_key_binds_model_prompt_decoding_direction():
    entry = SYSTEMS["gpt-5.6-sol"]
    key_a = runner._cache_key(entry, "en-xx", "ga", "Hello.")
    assert key_a == runner._cache_key(entry, "en-xx", "ga", "Hello.")
    assert key_a != runner._cache_key(entry, "xx-en", "ga", "Hello.")
    assert key_a != runner._cache_key(entry, "en-xx", "cy", "Hello.")
    assert key_a != runner._cache_key(entry, "en-xx", "ga", "Hello!")
    other = SYSTEMS["claude-opus-5"]
    assert key_a != runner._cache_key(other, "en-xx", "ga", "Hello.")
