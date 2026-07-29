"""Registry invariants for the hosted frontier-only v3 benchmark."""
import pytest

from celticbench.lib import ALL_LANGS
from celticbench.prompt import CHAT_DECODING
from celticbench.registry import (
    PROVIDERS, SYSTEMS, credential_envs, get_system, matrix, supported,
)

SYSTEM_IDS = {
    "gpt-5.6-sol",
    "claude-opus-5",
    "gemini-3.6-flash",
    "deepseek-v4-pro",
    "kimi-k3",
    "qwen3.7-max",
}
REQUIRED_FIELDS = (
    "label", "vendor", "provider", "model", "pin_status", "key_env",
    "license", "tier", "supported", "decoding",
)
DIRECTIONS = ("en-xx", "xx-en")


def test_registry_is_exactly_the_six_frontier_systems():
    assert set(SYSTEMS) == SYSTEM_IDS


def test_unregistered_system_is_rejected():
    with pytest.raises(SystemExit, match="unregistered system"):
        get_system("gpt-99-imaginary")


def test_every_entry_is_a_complete_hosted_flagship():
    for system_id, entry in SYSTEMS.items():
        assert all(field in entry for field in REQUIRED_FIELDS), system_id
        assert entry["provider"] in PROVIDERS, system_id
        assert entry["tier"] == "flagship-chat", system_id
        assert entry["pin_status"] in ("verified", "provisional"), system_id
        assert entry["model"] and not entry["model"].endswith("-latest"), system_id
        assert entry.get("revision") is None, system_id
        assert entry["decoding"] == CHAT_DECODING, system_id


def test_registry_uses_exactly_three_hosted_providers():
    assert set(PROVIDERS) == {"openai_compat", "anthropic", "gemini"}
    assert {entry["provider"] for entry in SYSTEMS.values()} == set(PROVIDERS)


def test_every_entry_requires_exactly_one_key():
    envs = []
    for system_id, entry in SYSTEMS.items():
        required = credential_envs(entry)
        assert required == (entry["key_env"],), system_id
        assert required[0], system_id
        envs.append(required[0])
    assert len(set(envs)) == len(SYSTEMS)


def test_every_system_covers_all_languages_in_both_directions():
    expected = set(ALL_LANGS)
    for system_id, entry in SYSTEMS.items():
        assert set(entry["supported"]) == set(DIRECTIONS), system_id
        for direction in DIRECTIONS:
            assert set(entry["supported"][direction]) == expected, (system_id, direction)
            for lang in ALL_LANGS:
                assert supported(system_id, lang, direction) == (True, None)


def test_declared_decoding_deviations_are_complete_and_auditable():
    for system_id, entry in SYSTEMS.items():
        deviation = entry.get("decoding_deviation")
        if deviation is None:
            continue
        assert deviation["omits"], system_id
        assert deviation["reason"], system_id
        assert set(deviation["omits"]) <= set(entry["decoding"]), system_id


def test_v3_matrix_has_144_supported_runs_and_72_coverage_cells():
    rows = matrix()
    assert len(rows) == 144
    assert all(row["supported"] and row["reason"] is None for row in rows)
    assert {row["system"] for row in rows} == SYSTEM_IDS
    coverage_cells = {
        (row["system"], row["lang"], row["direction"])
        for row in rows
    }
    assert len(coverage_cells) == 72
    assert coverage_cells == {
        (system_id, lang, direction)
        for system_id in SYSTEM_IDS
        for lang in ALL_LANGS
        for direction in DIRECTIONS
    }
