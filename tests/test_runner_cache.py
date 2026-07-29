"""Line cache keys and records remain bound to the complete v3 method."""
import json

import celticbench.runner as runner
from celticbench.providers import Translation
from celticbench.registry import SYSTEMS


def _translation(text, model="gpt-5.6-sol-2026-07-01", deviations=()):
    return Translation(
        text=text,
        model_reported=model,
        decoding_used={"temperature": 0.0},
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        deviations=deviations,
    )


def test_cache_roundtrip_keeps_provenance_and_deviations(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    deviation = "omitted temperature, top_p: provider rejected fixed sampling"
    runner._append_cache("test-system", "k1", _translation(
        "Myttin da.", deviations=(deviation,),
    ))
    runner._append_cache("test-system", "k2", _translation("Bore da."))

    cache = runner._load_cache("test-system")

    assert [record["v"] for record in cache.values()] == ["Myttin da.", "Bore da."]
    assert cache["k1"]["m"] == "gpt-5.6-sol-2026-07-01"
    assert cache["k1"]["u"]["total_tokens"] == 14
    assert cache["k1"]["method_version"] == "v3"
    assert cache["k1"]["deviations"] == [deviation]


def test_cache_is_bound_to_method_version_in_key_and_record(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    entry = SYSTEMS["gpt-5.6-sol"]

    monkeypatch.setattr(runner, "METHOD_VERSION", "v2")
    v2_key = runner._cache_key(entry, "en-xx", "ga", "Hello.")
    runner._append_cache("gpt-5.6-sol", v2_key, _translation("Dia dhuit."))

    monkeypatch.setattr(runner, "METHOD_VERSION", "v3")
    v3_key = runner._cache_key(entry, "en-xx", "ga", "Hello.")

    assert v3_key != v2_key
    assert runner._load_cache("gpt-5.6-sol") == {}


def test_torn_tail_line_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    runner._append_cache("test-system", "k1", _translation("value"))
    with open(runner._cache_file("test-system"), "a", encoding="utf-8") as handle:
        handle.write('{"k": "k2", "v": "trunc')
    assert list(runner._load_cache("test-system")) == ["k1"]


def test_legacy_and_incomplete_cache_records_are_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    runner._append_cache("test-system", "fresh", _translation("fresh text"))
    incomplete = [
        {"k": "legacy", "v": "old text", "m": "old-model", "d": {}, "u": {}},
        {
            "k": "no-deviations",
            "v": "old text",
            "m": "old-model",
            "d": {},
            "u": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "method_version": "v3",
        },
        {
            "k": "incomplete-usage",
            "v": "old text",
            "m": "old-model",
            "d": {},
            "u": {"total_tokens": 2},
            "method_version": "v3",
            "deviations": [],
        },
    ]
    with open(runner._cache_file("test-system"), "a", encoding="utf-8") as handle:
        for record in incomplete:
            handle.write(json.dumps(record) + "\n")

    cache = runner._load_cache("test-system")

    assert set(cache) == {"fresh"}


def test_cache_key_binds_model_prompt_decoding_direction_language_and_text(monkeypatch):
    entry = SYSTEMS["gpt-5.6-sol"]
    key = runner._cache_key(entry, "en-xx", "ga", "Hello.")
    assert key == runner._cache_key(entry, "en-xx", "ga", "Hello.")
    assert key != runner._cache_key(entry, "xx-en", "ga", "Hello.")
    assert key != runner._cache_key(entry, "en-xx", "cy", "Hello.")
    assert key != runner._cache_key(entry, "en-xx", "ga", "Hello!")
    assert key != runner._cache_key(SYSTEMS["claude-opus-5"], "en-xx", "ga", "Hello.")

    changed_decoding = {**entry, "decoding": {**entry["decoding"], "max_tokens": 1024}}
    assert key != runner._cache_key(changed_decoding, "en-xx", "ga", "Hello.")

    monkeypatch.setattr(runner, "PROMPT_SHA256", "0" * 64)
    assert key != runner._cache_key(entry, "en-xx", "ga", "Hello.")
