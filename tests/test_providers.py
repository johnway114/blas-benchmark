"""Provider adapters: request shape, receipt contents, signing, language codes.

No network: the module-level _post/_get are monkeypatched. The signing tests are
known-answer tests against the worked examples Amazon and Alibaba publish, so a
regression in canonicalization fails here rather than as an opaque 403 in a run.
"""
import hashlib

import pytest

from celticbench import providers, registry
from celticbench.prompt import CHAT_DECODING
from celticbench.providers import ProviderError

GEMINI_DEVIATION = {
    "omits": ("temperature", "top_p"),
    "reason": "Gemini deprecated temperature/top_p",
}

OPENAI_ENTRY = {
    "provider": "openai_compat",
    "base_url": "https://api.example.com/v1",
    "model": "gpt-test",
    "key_env": "TEST_OPENAI_KEY",
    "decoding": CHAT_DECODING,
}
ANTHROPIC_ENTRY = {
    "provider": "anthropic",
    "model": "claude-test",
    "key_env": "TEST_ANTHROPIC_KEY",
    "decoding": CHAT_DECODING,
}
GEMINI_ENTRY = {
    "provider": "gemini",
    "model": "gemini-test",
    "key_env": "TEST_GEMINI_KEY",
    "decoding": CHAT_DECODING,
}


class Calls:
    """Records every intercepted request and replays canned responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen = []

    def __call__(self, url, *, headers=None, params=None, body=None, data=None):
        self.seen.append({"url": url, "headers": headers or {}, "params": params or {},
                          "body": body, "data": data})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def last(self):
        return self.seen[-1]


def openai_response(usage=None):
    payload = {"model": "gpt-test-0931",
               "choices": [{"message": {"content": "Dia dhuit."}}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def anthropic_response(usage=None):
    payload = {"model": "claude-test-20260101",
               "content": [{"type": "text", "text": "Dia dhuit."}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def gemini_response(usage=None):
    payload = {"modelVersion": "gemini-test-002",
               "candidates": [{"content": {"parts": [{"text": "Dia dhuit."}]}}]}
    if usage is not None:
        payload["usageMetadata"] = usage
    return payload


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    for env in ("TEST_OPENAI_KEY", "TEST_ANTHROPIC_KEY", "TEST_GEMINI_KEY",
                "TEST_MT_KEY"):
        monkeypatch.setenv(env, "secret")


# --- forced departures from the fixed decoding -----------------------------

def test_declared_deviation_omits_parameters_and_records_the_vendor_reason(monkeypatch):
    calls = Calls(gemini_response())
    monkeypatch.setattr(providers, "_post", calls)
    entry = {**GEMINI_ENTRY, "decoding_deviation": GEMINI_DEVIATION}

    result = providers.translate_chat(entry, "Hello.")

    config = calls.last["body"]["generationConfig"]
    assert config == {"maxOutputTokens": 2048}
    assert result.decoding_used == {"max_tokens": 2048}
    assert len(result.deviations) == 1
    assert "temperature, top_p" in result.deviations[0]
    assert GEMINI_DEVIATION["reason"] in result.deviations[0]


def test_without_a_declared_deviation_the_full_fixed_decoding_is_sent(monkeypatch):
    calls = Calls(gemini_response())
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_chat(GEMINI_ENTRY, "Hello.")

    assert calls.last["body"]["generationConfig"] == {
        "temperature": 0.0, "topP": 1.0, "maxOutputTokens": 2048}
    assert result.decoding_used == dict(CHAT_DECODING)
    assert result.deviations == ()


def test_deviation_applies_to_openai_compat_bodies_too(monkeypatch):
    calls = Calls(openai_response())
    monkeypatch.setattr(providers, "_post", calls)
    entry = {**OPENAI_ENTRY, "decoding_deviation": GEMINI_DEVIATION}

    result = providers.translate_chat(entry, "Hello.")

    assert "temperature" not in calls.last["body"]
    assert "top_p" not in calls.last["body"]
    assert calls.last["body"]["max_tokens"] == 2048
    assert result.decoding_used == {"max_tokens": 2048}


def test_sampling_rejection_is_retried_and_surfaced_as_a_deviation(monkeypatch):
    rejection = ProviderError(
        "HTTP 400: {'error': {'message': \"Unsupported value: 'temperature' \""
        "'top_p' and 'max_tokens' are not supported with this model\"}}")
    calls = Calls(rejection, openai_response())
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_chat(OPENAI_ENTRY, "Hello.")

    retried = calls.last["body"]
    assert "temperature" not in retried and "top_p" not in retried
    assert retried["max_completion_tokens"] == 2048
    assert "max_tokens" not in retried
    assert result.decoding_used == {}
    assert "adjusted" not in result.decoding_used
    assert len(result.deviations) == 1
    assert "temperature" in result.deviations[0] and "HTTP 400" in result.deviations[0]


def test_a_400_that_is_not_about_sampling_is_not_retried(monkeypatch):
    calls = Calls(ProviderError("HTTP 400: model not found"))
    monkeypatch.setattr(providers, "_post", calls)

    with pytest.raises(ProviderError, match="model not found"):
        providers.translate_chat(OPENAI_ENTRY, "Hello.")
    assert len(calls.seen) == 1


# --- token usage, one shape per vendor -------------------------------------

def test_openai_usage_is_read_from_the_reported_totals(monkeypatch):
    monkeypatch.setattr(providers, "_post", Calls(openai_response(
        {"prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 48})))

    result = providers.translate_chat(OPENAI_ENTRY, "Hello.")

    assert result.usage == {"prompt_tokens": 41, "completion_tokens": 7,
                            "total_tokens": 48}
    assert result.model_reported == "gpt-test-0931"


def test_anthropic_usage_sums_the_halves_because_no_total_is_reported(monkeypatch):
    monkeypatch.setattr(providers, "_post", Calls(anthropic_response(
        {"input_tokens": 41, "output_tokens": 7, "cache_read_input_tokens": 0})))

    result = providers.translate_chat(ANTHROPIC_ENTRY, "Hello.")

    assert result.usage == {"prompt_tokens": 41, "completion_tokens": 7,
                            "total_tokens": 48}
    assert result.model_reported == "claude-test-20260101"


def test_anthropic_sends_the_fixed_decoding_and_the_pinned_api_version(monkeypatch):
    calls = Calls(anthropic_response())
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_chat(ANTHROPIC_ENTRY, "Hello.")

    assert calls.last["url"] == "https://api.anthropic.com/v1/messages"
    assert calls.last["headers"]["anthropic-version"] == "2023-06-01"
    assert calls.last["headers"]["x-api-key"] == "secret"
    for name, value in CHAT_DECODING.items():
        assert calls.last["body"][name] == value
    assert result.decoding_used == dict(CHAT_DECODING)


def test_anthropic_honours_a_declared_deviation(monkeypatch):
    calls = Calls(anthropic_response())
    monkeypatch.setattr(providers, "_post", calls)
    entry = {**ANTHROPIC_ENTRY, "decoding_deviation": GEMINI_DEVIATION}

    result = providers.translate_chat(entry, "Hello.")

    assert "temperature" not in calls.last["body"]
    assert "top_p" not in calls.last["body"]
    assert calls.last["body"]["max_tokens"] == 2048
    assert result.deviations == (
        f"omitted temperature, top_p: {GEMINI_DEVIATION['reason']}",)


def test_gemini_total_wins_over_the_sum_because_it_includes_thinking(monkeypatch):
    monkeypatch.setattr(providers, "_post", Calls(gemini_response(
        {"promptTokenCount": 41, "candidatesTokenCount": 7,
         "thoughtsTokenCount": 120, "totalTokenCount": 168})))

    result = providers.translate_chat(GEMINI_ENTRY, "Hello.")

    assert result.usage == {"prompt_tokens": 41, "completion_tokens": 7,
                            "total_tokens": 168}
    assert result.model_reported == "gemini-test-002"


def test_absent_usage_records_zeros_rather_than_a_guess(monkeypatch):
    monkeypatch.setattr(providers, "_post", Calls(openai_response()))

    result = providers.translate_chat(OPENAI_ENTRY, "Hello.")

    assert result.usage == {"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0}


# --- per-provider language codes -------------------------------------------

@pytest.mark.parametrize("provider,lang,expected", [
    ("google_translate_v2", "ga", ("en", "ga")),
    ("google_translation_llm", "cy", ("en", "cy")),
    ("azure_translator", "cy", ("en", "cy")),
    ("aws_translate", "ga", ("en", "ga")),
    ("alibaba_mt", "kw", ("en", "kw")),
    ("deepl", "br", ("EN", "BR")),
])
def test_mt_codes_en_to_celtic(provider, lang, expected):
    assert providers.mt_codes({"provider": provider}, lang, "en-xx") == expected


@pytest.mark.parametrize("provider,lang,expected", [
    ("google_translate_v2", "gd", ("gd", "en")),
    ("aws_translate", "cy", ("cy", "en")),
    ("alibaba_mt", "gv", ("gv", "en")),
    # DeepL's bare EN is source-only; a target must name a regional variant.
    ("deepl", "ga", ("GA", "EN-GB")),
])
def test_mt_codes_celtic_to_en(provider, lang, expected):
    assert providers.mt_codes({"provider": provider}, lang, "xx-en") == expected


@pytest.mark.parametrize("provider,lang", [
    ("deepl", "gd"), ("azure_translator", "gv"), ("aws_translate", "kw"),
    ("google_translate_v2", "gv"), ("google_translation_llm", "br"),
    ("alibaba_mt", "gd"),
])
def test_mt_codes_refuses_a_language_the_provider_does_not_publish(provider, lang):
    with pytest.raises(ProviderError, match="refusing to substitute"):
        providers.mt_codes({"provider": provider}, lang, "en-xx")


def test_mt_codes_rejects_an_unknown_direction():
    with pytest.raises(ProviderError, match="unknown direction"):
        providers.mt_codes({"provider": "deepl"}, "ga", "xx-xx")


def test_mt_codes_rejects_a_chat_provider():
    with pytest.raises(ProviderError, match="no language mapping"):
        providers.mt_codes({"provider": "gemini"}, "ga", "en-xx")


def test_every_registered_mt_system_resolves_codes_for_everything_it_claims():
    for system_id, entry in registry.SYSTEMS.items():
        if entry["provider"] not in registry.MT_PROVIDERS:
            continue
        for direction, langs in entry["supported"].items():
            for lang in langs:
                source, target = providers.mt_codes(entry, lang, direction)
                assert source and target, (system_id, lang, direction)


# --- request signing, against the vendors' own worked examples --------------

# https://docs.aws.amazon.com/amazonglacier/latest/dev/amazon-glacier-signing-requests.html
GLACIER_HEADERS = {
    "host": "glacier.us-east-1.amazonaws.com",
    "x-amz-date": "20120525T002453Z",
    "x-amz-glacier-version": "2012-06-01",
}
GLACIER_CANONICAL = (
    "PUT\n"
    "/-/vaults/examplevault\n"
    "\n"
    "host:glacier.us-east-1.amazonaws.com\n"
    "x-amz-date:20120525T002453Z\n"
    "x-amz-glacier-version:2012-06-01\n"
    "\n"
    "host;x-amz-date;x-amz-glacier-version\n"
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
GLACIER_STRING_TO_SIGN = (
    "AWS4-HMAC-SHA256\n"
    "20120525T002453Z\n"
    "20120525/us-east-1/glacier/aws4_request\n"
    "5f1da1a2d0feb614dd03d71e87928b8e449ac87614479332aced3a701f916743"
)
GLACIER_SIGNATURE = "3ce5b2f2fffac9262b4da9256f8d086b4aaf42eba5f111c21681a65a127b7c2a"
GLACIER_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


def test_sigv4_canonical_request_matches_the_documented_example():
    canonical, signed_headers = providers._canonical_request(
        "PUT", "/-/vaults/examplevault", "", GLACIER_HEADERS, b"")

    assert canonical == GLACIER_CANONICAL
    assert signed_headers == "host;x-amz-date;x-amz-glacier-version"


def test_sigv4_canonicalization_lowercases_sorts_and_trims_headers():
    canonical, signed_headers = providers._canonical_request(
        "PUT", "/-/vaults/examplevault", "",
        {"X-Amz-Glacier-Version": " 2012-06-01 ", "Host": GLACIER_HEADERS["host"],
         "x-amz-date": "20120525T002453Z"}, b"")

    assert canonical == GLACIER_CANONICAL
    assert signed_headers == "host;x-amz-date;x-amz-glacier-version"


def test_sigv4_string_to_sign_matches_the_documented_example():
    canonical, _ = providers._canonical_request(
        "PUT", "/-/vaults/examplevault", "", GLACIER_HEADERS, b"")

    assert providers._sigv4_string_to_sign(
        "20120525T002453Z", "20120525/us-east-1/glacier/aws4_request",
        canonical) == GLACIER_STRING_TO_SIGN


def test_sigv4_signature_matches_the_documented_example():
    assert providers._sigv4_signature(
        GLACIER_SECRET, "20120525", "us-east-1", "glacier",
        GLACIER_STRING_TO_SIGN) == GLACIER_SIGNATURE


def test_sigv4_authorization_header_matches_the_documented_example():
    assert providers._sigv4_authorization(
        "AKIAIOSFODNN7EXAMPLE", GLACIER_SECRET, "us-east-1", "glacier",
        "20120525T002453Z", "PUT", "/-/vaults/examplevault", "",
        GLACIER_HEADERS, b"") == (
        "AWS4-HMAC-SHA256 "
        "Credential=AKIAIOSFODNN7EXAMPLE/20120525/us-east-1/glacier/aws4_request, "
        "SignedHeaders=host;x-amz-date;x-amz-glacier-version, "
        f"Signature={GLACIER_SIGNATURE}")


def test_sigv4_signature_is_bound_to_the_payload():
    canonical, _ = providers._canonical_request(
        "PUT", "/-/vaults/examplevault", "", GLACIER_HEADERS, b"tampered")

    assert providers._sigv4_signature(
        GLACIER_SECRET, "20120525", "us-east-1", "glacier",
        providers._sigv4_string_to_sign(
            "20120525T002453Z", "20120525/us-east-1/glacier/aws4_request",
            canonical)) != GLACIER_SIGNATURE


# https://www.alibabacloud.com/help/en/sdk/product-overview/v3-request-structure-and-signature
def test_acs3_signature_matches_the_documented_example():
    empty = hashlib.sha256(b"").hexdigest()
    canonical, signed_headers = providers._canonical_request(
        "POST", "/",
        "ImageId=win2019_1809_x64_dtc_zh-cn_40G_alibase_20230811.vhd&RegionId=cn-shanghai",
        {"host": "ecs.cn-shanghai.aliyuncs.com",
         "x-acs-action": "RunInstances",
         "x-acs-content-sha256": empty,
         "x-acs-date": "2023-10-26T10:22:32Z",
         "x-acs-signature-nonce": "3156853299f313e23d1673dc12e1703d",
         "x-acs-version": "2014-05-26"}, b"")

    assert signed_headers == ("host;x-acs-action;x-acs-content-sha256;x-acs-date;"
                             "x-acs-signature-nonce;x-acs-version")
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        "7ea06492da5221eba5297e897ce16e55f964061054b7695beedaac1145b1e259")
    assert providers._acs3_signature("YourAccessKeySecret", canonical) == (
        "06563a9e1b43f5dfe96b81484da74bceab24a1d853912eee15083a6f0f3283c0")


# --- dedicated MT adapters -------------------------------------------------

DEEPL_ENTRY = {"provider": "deepl", "model": "deepl-translate-v2",
               "key_env": "TEST_MT_KEY", "decoding": {}}


@pytest.mark.parametrize("key,host", [
    ("aaaa-bbbb:fx", "https://api-free.deepl.com"),
    ("aaaa-bbbb", "https://api.deepl.com"),
])
def test_deepl_host_switches_on_the_free_key_suffix(key, host):
    assert providers._deepl_host(key) == host


@pytest.mark.parametrize("key,host", [
    ("aaaa-bbbb:fx", "https://api-free.deepl.com/v2/translate"),
    ("aaaa-bbbb", "https://api.deepl.com/v2/translate"),
])
def test_deepl_translate_posts_to_the_host_the_key_implies(monkeypatch, key, host):
    monkeypatch.setenv("TEST_MT_KEY", key)
    calls = Calls({"translations": [{"detected_source_language": "EN",
                                     "text": "Dia dhuit."}]})
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_mt(DEEPL_ENTRY, "Hello.", "EN", "GA")

    assert calls.last["url"] == host
    assert calls.last["headers"]["Authorization"] == f"DeepL-Auth-Key {key}"
    assert calls.last["body"] == {"text": ["Hello."], "source_lang": "EN",
                                  "target_lang": "GA"}
    assert result.text == "Dia dhuit."
    assert result.decoding_used == {}
    assert result.usage == {"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0}


def test_google_v2_unescapes_the_html_entities_it_always_returns(monkeypatch):
    monkeypatch.setattr(providers, "_post", Calls(
        {"data": {"translations": [{"translatedText": "Tom &amp; Máire&#39;s"}]}}))
    entry = {"provider": "google_translate_v2", "model": "nmt",
             "key_env": "TEST_MT_KEY", "decoding": {},
             "endpoint": "https://translation.googleapis.com/language/translate/v2"}

    result = providers.translate_mt(entry, "Tom & Mary's", "en", "ga")

    assert result.text == "Tom & Máire's"
    assert result.model_reported == "nmt"


def test_google_translation_llm_names_the_model_resource(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "celticbench")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    calls = Calls({"translations": [
        {"translatedText": "Bore da.",
         "model": "projects/1234/locations/global/models/general/translation-llm"}]})
    monkeypatch.setattr(providers, "_post", calls)
    entry = {"provider": "google_translation_llm", "model": "general/translation-llm",
             "key_env": "TEST_MT_KEY", "decoding": {},
             "endpoint": "https://translate.googleapis.com/v3"}

    result = providers.translate_mt(entry, "Good morning.", "en", "cy")

    assert calls.last["url"] == (
        "https://translate.googleapis.com/v3/projects/celticbench/locations/global"
        ":translateText")
    assert calls.last["body"]["model"] == (
        "projects/celticbench/locations/global/models/general/translation-llm")
    assert calls.last["headers"]["Authorization"] == "Bearer secret"
    # The vendor normalizes the resource to the project number; that is what the
    # receipt must carry, not the resource we asked for.
    assert result.model_reported == (
        "projects/1234/locations/global/models/general/translation-llm")


def test_google_translation_llm_refuses_without_a_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(providers, "_post", Calls())
    entry = {"provider": "google_translation_llm", "model": "general/translation-llm",
             "key_env": "TEST_MT_KEY", "decoding": {},
             "endpoint": "https://translate.googleapis.com/v3"}

    with pytest.raises(ProviderError, match="GOOGLE_CLOUD_PROJECT"):
        providers.translate_mt(entry, "Good morning.", "en", "cy")


@pytest.mark.parametrize("missing", ["GOOGLE_CLOUD_PROJECT", "AZURE_TRANSLATOR_REGION",
                                     "AWS_REGION", "ALIBABA_ACCESS_KEY_SECRET"])
def test_a_missing_required_variable_uses_the_phrase_the_runner_fail_fasts_on(
        monkeypatch, missing):
    # runner.py aborts the whole run on "missing API key"/"missing credential"
    # rather than retrying something no retry can fix.
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(ProviderError, match="missing credential"):
        providers._env(missing)


def test_azure_sends_the_array_body_and_the_region_header(monkeypatch):
    monkeypatch.setenv("AZURE_TRANSLATOR_REGION", "northeurope")
    calls = Calls([{"translations": [{"to": "cy", "text": "Bore da."}]}])
    monkeypatch.setattr(providers, "_post", calls)
    entry = {"provider": "azure_translator", "model": "azure-translator-nmt",
             "key_env": "TEST_MT_KEY", "api_version": "3.0", "decoding": {},
             "endpoint": "https://api.cognitive.microsofttranslator.com/translate"}

    result = providers.translate_mt(entry, "Good morning.", "en", "cy")

    assert calls.last["body"] == [{"text": "Good morning."}]
    assert calls.last["params"] == {"api-version": "3.0", "from": "en", "to": "cy"}
    assert calls.last["headers"]["Ocp-Apim-Subscription-Key"] == "secret"
    assert calls.last["headers"]["Ocp-Apim-Subscription-Region"] == "northeurope"
    assert result.text == "Bore da."


AWS_ENTRY = {"provider": "aws_translate", "model": "aws-translate",
             "key_env": "TEST_MT_KEY", "decoding": {}}


def test_aws_translate_signs_a_json11_request_at_the_documented_target(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", GLACIER_SECRET)
    calls = Calls({"TranslatedText": "Dia dhuit.", "TargetLanguageCode": "ga"})
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_mt(AWS_ENTRY, "Hello.", "en", "ga")

    assert calls.last["url"] == "https://translate.eu-west-1.amazonaws.com/"
    headers = calls.last["headers"]
    assert headers["x-amz-target"] == "AWSShineFrontendService_20170701.TranslateText"
    assert headers["content-type"] == "application/x-amz-json-1.1"
    assert "host" not in headers  # requests derives Host; only the signature needs it
    assert headers["Authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=secret/")
    assert "/eu-west-1/translate/aws4_request" in headers["Authorization"]
    assert calls.last["data"] == (
        b'{"Text":"Hello.","SourceLanguageCode":"en","TargetLanguageCode":"ga"}')
    assert result.text == "Dia dhuit."


def test_aws_translate_refuses_without_a_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", GLACIER_SECRET)
    monkeypatch.setattr(providers, "_post", Calls())

    with pytest.raises(ProviderError, match="AWS_REGION"):
        providers.translate_mt(AWS_ENTRY, "Hello.", "en", "ga")


ALIBABA_ENTRY = {"provider": "alibaba_mt", "model": "TranslateGeneral",
                 "key_env": "TEST_MT_KEY", "api_version": "2018-10-12",
                 "decoding": {}, "endpoint": "https://mt.aliyuncs.com/"}


def test_alibaba_signs_the_form_body_and_records_the_endpoint(monkeypatch):
    monkeypatch.setenv("ALIBABA_ACCESS_KEY_SECRET", "shh")
    calls = Calls({"Code": 200, "Message": "success",
                   "Data": {"Translated": "Kernewek", "WordCount": "1"}})
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_mt(ALIBABA_ENTRY, "Cornish", "en", "kw")

    sent = calls.last
    assert sent["url"] == "https://mt.aliyuncs.com/"
    assert sent["data"] == (b"FormatType=text&Scene=general&SourceLanguage=en"
                            b"&SourceText=Cornish&TargetLanguage=kw")
    assert sent["headers"]["x-acs-content-sha256"] == hashlib.sha256(
        sent["data"]).hexdigest()
    assert sent["headers"]["x-acs-action"] == "TranslateGeneral"
    assert sent["headers"]["x-acs-version"] == "2018-10-12"
    assert sent["headers"]["Authorization"].startswith(
        "ACS3-HMAC-SHA256 Credential=secret,SignedHeaders="
        "content-type;host;x-acs-action;x-acs-content-sha256;x-acs-date;"
        "x-acs-signature-nonce;x-acs-version,Signature=")
    assert result.text == "Kernewek"
    assert result.decoding_used == {"endpoint_host": "mt.aliyuncs.com",
                                    "region": "international"}


def test_alibaba_regional_host_is_recorded_as_that_region(monkeypatch):
    monkeypatch.setenv("ALIBABA_ACCESS_KEY_SECRET", "shh")
    monkeypatch.setattr(providers, "_post", Calls(
        {"Code": 200, "Data": {"Translated": "Kernewek"}}))
    entry = {**ALIBABA_ENTRY, "endpoint": "https://mt.ap-southeast-1.aliyuncs.com/"}

    result = providers.translate_mt(entry, "Cornish", "en", "kw")

    assert result.decoding_used == {"endpoint_host": "mt.ap-southeast-1.aliyuncs.com",
                                    "region": "ap-southeast-1"}


def test_alibaba_application_level_error_code_is_a_provider_error(monkeypatch):
    monkeypatch.setenv("ALIBABA_ACCESS_KEY_SECRET", "shh")
    monkeypatch.setattr(providers, "_post", Calls(
        {"Code": 10005, "Message": "The specified language pair is not supported."}))

    with pytest.raises(ProviderError, match="10005"):
        providers.translate_mt(ALIBABA_ENTRY, "Cornish", "en", "kw")


def test_mt_usage_dicts_are_not_shared_between_translations(monkeypatch):
    monkeypatch.setattr(providers, "_post", Calls(
        {"translations": [{"text": "a"}]}, {"translations": [{"text": "b"}]}))

    first = providers.translate_mt(DEEPL_ENTRY, "Hello.", "EN", "GA")
    first.usage["prompt_tokens"] = 99
    second = providers.translate_mt(DEEPL_ENTRY, "Hello.", "EN", "GA")

    assert second.usage["prompt_tokens"] == 0


# --- dispatch ---------------------------------------------------------------

def test_translate_chat_refuses_an_mt_provider():
    with pytest.raises(ProviderError, match="not a chat provider"):
        providers.translate_chat({"provider": "deepl"}, "Hello.")


def test_translate_mt_refuses_a_chat_provider():
    with pytest.raises(ProviderError, match="not a dedicated MT provider"):
        providers.translate_mt({"provider": "anthropic"}, "Hello.", "en", "ga")


# --- doctor ----------------------------------------------------------------

def test_missing_key_lists_every_missing_credential(monkeypatch):
    entry = registry.get_system("aws-translate")
    for env in registry.credential_envs(entry):
        monkeypatch.delenv(env, raising=False)

    result = providers.doctor_check("aws-translate", entry)

    assert result["status"] == "MISSING_KEY"
    for env in registry.credential_envs(entry):
        assert env in result["detail"]


def test_missing_key_lists_only_the_ones_actually_absent(monkeypatch):
    entry = registry.get_system("aws-translate")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "present")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    result = providers.doctor_check("aws-translate", entry)

    assert result["status"] == "MISSING_KEY"
    assert "AWS_ACCESS_KEY_ID" not in result["detail"]
    assert "AWS_SECRET_ACCESS_KEY" in result["detail"]
    assert "AWS_REGION" in result["detail"]


def test_gated_local_weights_ask_for_the_hf_token(monkeypatch):
    entry = registry.get_system("translategemma-4b")
    monkeypatch.delenv(entry["token_env"], raising=False)

    result = providers.doctor_check("translategemma-4b", entry)

    assert result["status"] == "NEEDS_HF_TOKEN"
    assert entry["token_env"] in result["detail"]


def test_gated_local_weights_with_a_token_fall_through_to_the_dependency_check(monkeypatch):
    entry = registry.get_system("translategemma-4b")
    monkeypatch.setenv(entry["token_env"], "hf_token")

    result = providers.doctor_check("translategemma-4b", entry)

    assert result["status"] in ("OK", "NEEDS_DEPS")


def test_ungated_local_weights_never_ask_for_a_token(monkeypatch):
    entry = registry.get_system("nllb-600m")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = providers.doctor_check("nllb-600m", entry)

    assert result["status"] in ("OK", "NEEDS_DEPS")


def test_mt_language_probe_names_the_missing_codes(monkeypatch):
    entry = registry.get_system("deepl")
    monkeypatch.setenv(entry["key_env"], "aaaa:fx")
    monkeypatch.setattr(providers, "list_models", lambda _entry: ["ga", "en", "en-GB"])

    result = providers.doctor_check("deepl", entry)

    assert result["status"] == "LANGS_MISSING"
    assert "CY" in result["detail"] and "BR" in result["detail"]
    assert "GA" not in result["detail"]


def test_mt_language_probe_passes_when_the_vendor_covers_everything(monkeypatch):
    entry = registry.get_system("azure-translator")
    for env in registry.credential_envs(entry):
        monkeypatch.setenv(env, "present")
    monkeypatch.setattr(providers, "list_models",
                        lambda _entry: ["en", "ga", "cy", "fr"])

    result = providers.doctor_check("azure-translator", entry)

    assert result["status"] == "OK"


def test_mt_language_probe_requires_the_english_target_code(monkeypatch):
    entry = registry.get_system("deepl")
    monkeypatch.setenv(entry["key_env"], "aaaa:fx")
    # DeepL offering bare EN but no regional variant breaks every xx-en row.
    monkeypatch.setattr(providers, "list_models",
                        lambda _entry: ["ga", "cy", "br", "en"])

    result = providers.doctor_check("deepl", entry)

    assert result["status"] == "LANGS_MISSING"
    assert "EN-GB" in result["detail"]


def test_a_failed_probe_is_reported_not_raised(monkeypatch):
    entry = registry.get_system("azure-translator")
    for env in registry.credential_envs(entry):
        monkeypatch.setenv(env, "present")

    def boom(_entry):
        raise ProviderError("HTTP 401: invalid key")

    monkeypatch.setattr(providers, "list_models", boom)

    result = providers.doctor_check("azure-translator", entry)

    assert result["status"] == "AUTH_OR_NETWORK_FAIL"
    assert "401" in result["detail"]


def test_alibaba_reports_that_no_language_endpoint_exists(monkeypatch):
    entry = registry.get_system("alibaba-mt")
    for env in registry.credential_envs(entry):
        monkeypatch.setenv(env, "present")

    result = providers.doctor_check("alibaba-mt", entry)

    assert result["status"] == "NO_LANGUAGE_PROBE"
    assert "no list-languages API operation" in result["detail"]


def test_list_models_refuses_to_invent_an_alibaba_language_probe():
    with pytest.raises(ProviderError, match="no list-languages API operation"):
        providers.list_models(registry.get_system("alibaba-mt"))
