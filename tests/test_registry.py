"""Registry invariants: the fail-closed contract everything else relies on."""
import pytest

from celticbench.lib import ALL_LANGS, LANGS
from celticbench.prompt import CAUSAL_DECODING, CHAT_DECODING
from celticbench.registry import (
    CHAT_PROVIDERS, MT_PROVIDERS, SYSTEMS, credential_envs, get_system, is_chat,
    resolve_model, resolve_revision, supported, uses_prompt,
)

REQUIRED_FIELDS = ("label", "vendor", "provider", "model", "pin_status",
                   "license", "tier", "supported", "decoding")
VALID_PIN_STATUS = ("verified", "provisional", "alias")
VALID_TIERS = ("flagship-chat", "efficient-chat", "dedicated-mt",
               "open-mt", "open-anchor", "open-general")

# Which LANGS column each dedicated MT service's coverage must be derived from.
MT_LANG_COLUMN = {
    "google-translate-v2": "google",
    "google-translation-llm": "google_tllm",
    "deepl": "deepl",
    "azure-translator": "azure",
    "aws-translate": "aws",
    "alibaba-mt": "alibaba",
}


def test_unregistered_system_is_rejected():
    with pytest.raises(SystemExit, match="unregistered system"):
        get_system("gpt-99-imaginary")


def test_every_entry_is_complete():
    for system_id, entry in SYSTEMS.items():
        for field in REQUIRED_FIELDS:
            assert field in entry, f"{system_id} missing {field}"
        assert entry["pin_status"] in VALID_PIN_STATUS, system_id
        assert entry["tier"] in VALID_TIERS, f"{system_id} has tier {entry['tier']}"
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


def test_prompted_general_models_share_the_fixed_decoding():
    """A model given the shared prompt is also given the shared decoding.

    A dedicated MT model driven by its own published template may also use the
    decoding that template's card documents, exactly as the sequence-to-sequence
    anchors use their own beams. It must still be deterministic.
    """
    for system_id, entry in SYSTEMS.items():
        if entry.get("family") != "causal":
            continue
        if entry.get("template"):
            assert entry["decoding"]["do_sample"] is False, system_id
            assert entry["decoding"]["num_beams"] >= 1, system_id
        else:
            assert entry["decoding"] == CAUSAL_DECODING, system_id


def test_a_template_is_only_ever_a_dedicated_mt_models_own_format():
    """The shared prompt is the default; a template is a documented exception."""
    for system_id, entry in SYSTEMS.items():
        if not entry.get("template"):
            continue
        assert entry["tier"] == "open-mt", f"{system_id} must not tune the shared prompt"
        for field in ("{src_name}", "{tgt_name}", "{text}"):
            assert field in entry["template"], f"{system_id} template lacks {field}"


def test_hosted_systems_have_key_env_and_local_ones_do_not():
    for system_id, entry in SYSTEMS.items():
        if entry["provider"] == "hf_local":
            assert entry["key_env"] is None, system_id
            assert entry.get("revision"), f"{system_id} local system must pin a revision"
        else:
            assert entry["key_env"], f"{system_id} hosted system must name its key env"


def test_pinned_revisions_are_full_commit_shas():
    for system_id, entry in SYSTEMS.items():
        revision = entry.get("revision")
        revisions = revision.values() if isinstance(revision, dict) else [revision]
        for value in revisions:
            if value is None:
                continue
            assert len(value) == 40 and all(c in "0123456789abcdef" for c in value), (
                f"{system_id} pins {value!r}, which is not an immutable commit SHA"
            )


def test_no_hosted_model_id_is_a_moving_alias():
    """A `-latest` pin silently changes model mid-edition and voids the receipts."""
    for system_id, entry in SYSTEMS.items():
        if entry["provider"] == "hf_local":
            continue
        model = entry["model"]
        assert not model.endswith("-latest"), f"{system_id} pins a moving alias"


def test_gated_local_weights_declare_their_token():
    for system_id, entry in SYSTEMS.items():
        if entry.get("gated"):
            assert entry.get("token_env"), f"{system_id} is gated but names no token env"


def test_credential_envs_covers_multi_secret_providers():
    assert credential_envs(SYSTEMS["opus-mt-cel"]) == ()
    assert credential_envs(SYSTEMS["gpt-5.6-sol"]) == ("OPENAI_API_KEY",)
    assert credential_envs(SYSTEMS["aws-translate"]) == (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
    )


def test_uses_prompt_matches_how_the_system_is_driven():
    assert uses_prompt(SYSTEMS["gpt-5.6-sol"])        # hosted chat
    assert uses_prompt(SYSTEMS["qwen3.5-9b"])         # local causal, same prompt
    assert not uses_prompt(SYSTEMS["opus-mt-cel"])    # seq2seq MT, raw text
    assert not uses_prompt(SYSTEMS["deepl"])          # MT service, raw text


def test_nc_licensed_systems_are_flagged_benchmark_only():
    for system_id in ("nllb-600m", "tiny-aya-water"):
        entry = SYSTEMS[system_id]
        assert "NC" in entry["license"]
        assert entry["benchmark_only"] is True


def test_supported_reflects_documented_coverage():
    ok, _ = supported("google-translate-v2", "kw", "en-xx")
    assert not ok, "Google does not offer Cornish; registry must refuse it"
    ok, _ = supported("google-translate-v2", "gv", "en-xx")
    assert not ok, "Google's NMT list has no Manx either; the web UI is not the API"
    ok, reason = supported("nllb-600m", "gv", "en-xx")
    assert not ok and reason
    ok, _ = supported("deepl", "gd", "en-xx")
    assert not ok, "DeepL lists no Scottish Gaelic"
    for lang in ALL_LANGS:
        for direction in ("en-xx", "xx-en"):
            ok, _ = supported("opus-mt-cel", lang, direction)
            assert ok, f"opus-mt-cel covers all six ({lang} {direction})"


def test_alibaba_is_the_only_service_offering_manx_and_cornish():
    for lang in ("gv", "kw"):
        offering = [
            system_id for system_id in MT_LANG_COLUMN
            if supported(system_id, lang, "en-xx")[0]
        ]
        assert offering == ["alibaba-mt"], f"{lang}: {offering}"


def test_mt_coverage_is_derived_from_the_language_table():
    """Hand-typed coverage drifts from the vendor's list; derived coverage cannot."""
    for system_id, column in MT_LANG_COLUMN.items():
        expected = {k for k in ALL_LANGS if LANGS[k][column]}
        for direction in ("en-xx", "xx-en"):
            assert set(SYSTEMS[system_id]["supported"][direction]) == expected, system_id


def test_dedicated_mt_services_expose_no_decoding():
    for system_id, entry in SYSTEMS.items():
        if entry["provider"] in MT_PROVIDERS:
            assert entry["decoding"] == {}, f"{system_id} claims decoding it cannot control"


def test_declared_decoding_deviations_are_explained():
    for system_id, entry in SYSTEMS.items():
        deviation = entry.get("decoding_deviation")
        if deviation is None:
            continue
        assert deviation["omits"], system_id
        assert deviation["reason"], f"{system_id} must say why it departs from the contract"
        for field in deviation["omits"]:
            assert field in entry["decoding"], (
                f"{system_id} claims to omit {field}, which is not in its decoding"
            )


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


def test_provider_constants_match_entries():
    providers_in_use = {e["provider"] for e in SYSTEMS.values()}
    assert set(CHAT_PROVIDERS) <= providers_in_use
    assert set(MT_PROVIDERS) <= providers_in_use
