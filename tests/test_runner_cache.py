"""Line-level cache: deterministic keys, crash-safe append/load roundtrip."""
import celticbench.runner as runner
from celticbench.registry import SYSTEMS


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    runner._append_cache("test-system", "k1", "Myttin da.")
    runner._append_cache("test-system", "k2", "Bore da.")
    cache = runner._load_cache("test-system")
    assert cache == {"k1": "Myttin da.", "k2": "Bore da."}


def test_torn_tail_line_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", str(tmp_path / "cache"))
    runner._append_cache("test-system", "k1", "value")
    with open(runner._cache_file("test-system"), "a", encoding="utf-8") as handle:
        handle.write('{"k": "k2", "v": "trunc')  # simulated crash mid-write
    cache = runner._load_cache("test-system")
    assert cache == {"k1": "value"}


def test_cache_key_binds_model_prompt_decoding_direction():
    entry = SYSTEMS["gpt-5.6-sol"]
    key_a = runner._cache_key(entry, "en-xx", "ga", "Hello.")
    assert key_a == runner._cache_key(entry, "en-xx", "ga", "Hello.")
    assert key_a != runner._cache_key(entry, "xx-en", "ga", "Hello.")
    assert key_a != runner._cache_key(entry, "en-xx", "cy", "Hello.")
    assert key_a != runner._cache_key(entry, "en-xx", "ga", "Hello!")
    other = SYSTEMS["claude-opus-5"]
    assert key_a != runner._cache_key(other, "en-xx", "ga", "Hello.")
