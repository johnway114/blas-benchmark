"""Registry invariants: the fail-closed contract everything else relies on."""
import pytest

from celticbench.lib import ALL_LANGS, LANGS
from celticbench.prompt import CHAT_DECODING
from celticbench.registry import (
    CHAT_PROVIDERS, SYSTEMS, get_system, is_chat, resolve_model,
    resolve_revision, supported,
)

REQUIRED_FIELDS = ("label", "vendor", "provider", "model", "pin_status",
                   "license", "tier", "supported", "decoding")
VALID_PIN_STATUS = ("verified", "provisional", "alias")


def test_unregistered_system_is_rejected():
    with pytest.raises(SystemExit, match="unregistered system"):
        get_system("gpt-99-imaginary")


def test_every_entry_is_complete():
    for system_id, entry in SYSTEMS.items():
        for field in REQUIRED_FIELDS:
            assert field in entry, f"{system_id} missing {field}"
        assert entry["pin_status"] in VALID_PIN_STATUS, system_id
        assert "key_env" in entry, f"{system_id} must declare key_env (None for local)"
        for direction, langs in entry["supported"].items():
            assert direction in ("en-xx", "xx-en"), system_id
            for lang in langs:
                assert lang in ALL_LANGS, f"{system_id}: unknown lang {lang}"


def test_chat_systems_share_the_fixed_decoding():
    for system_id, entry in SYSTEMS.items():
        if is_chat(entry):
            assert entry["decoding"] == CHAT_DECODING, (
                f"{system_id} deviates from the fixed chat decoding; "
                "that would break cross-system comparability"
            )


def test_hosted_systems_have_key_env_and_local_ones_do_not():
    for system_id, entry in SYSTEMS.items():
        if entry["provider"] == "hf_local":
            assert entry["key_env"] is None, system_id
            assert entry.get("revision"), f"{system_id} local anchor must pin a revision"
        else:
            assert entry["key_env"], f"{system_id} hosted system must name its key env"


def test_nc_licensed_anchor_is_flagged_benchmark_only():
    nllb = SYSTEMS["nllb-600m"]
    assert "NC" in nllb["license"]
    assert nllb["benchmark_only"] is True


def test_supported_reflects_documented_coverage():
    ok, _ = supported("google-translate-v2", "kw", "en-xx")
    assert not ok, "Google does not offer Cornish; registry must refuse it"
    ok, reason = supported("nllb-600m", "gv", "en-xx")
    assert not ok and reason
    for lang in ALL_LANGS:
        for direction in ("en-xx", "xx-en"):
            ok, _ = supported("opus-mt-cel", lang, direction)
            assert ok, f"opus-mt-cel covers all six ({lang} {direction})"


def test_direction_dependent_model_resolution():
    entry = SYSTEMS["opus-mt-cel"]
    assert resolve_model(entry, "en-xx") == "Helsinki-NLP/opus-mt-en-cel"
    assert resolve_model(entry, "xx-en") == "Helsinki-NLP/opus-mt-cel-en"
    assert resolve_revision(entry, "en-xx") != resolve_revision(entry, "xx-en")
    sol = SYSTEMS["gpt-5.6-sol"]
    assert resolve_model(sol, "en-xx") == resolve_model(sol, "xx-en") == "gpt-5.6-sol"
    assert resolve_revision(sol, "en-xx") is None


def test_google_language_map_matches_registry_support():
    google_langs = SYSTEMS["google-translate-v2"]["supported"]["en-xx"]
    assert set(google_langs) == {k for k in ALL_LANGS if LANGS[k]["google"]}


def test_chat_providers_constant_matches_entries():
    providers_in_use = {e["provider"] for e in SYSTEMS.values()}
    assert set(CHAT_PROVIDERS) <= providers_in_use
