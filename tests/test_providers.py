"""Hosted adapters and live model-list doctor outcomes, without network access."""
import pytest

from celticbench import providers
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
    "pin_status": "verified",
    "decoding": CHAT_DECODING,
    "family_hint": "gpt",
}
ANTHROPIC_ENTRY = {
    "provider": "anthropic",
    "model": "claude-test",
    "key_env": "TEST_ANTHROPIC_KEY",
    "pin_status": "verified",
    "decoding": CHAT_DECODING,
    "family_hint": "claude",
}
GEMINI_ENTRY = {
    "provider": "gemini",
    "model": "gemini-test",
    "key_env": "TEST_GEMINI_KEY",
    "pin_status": "verified",
    "decoding": CHAT_DECODING,
    "family_hint": "gemini",
}


class Calls:
    """Record intercepted requests and replay canned responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen = []

    def __call__(self, url, *, headers=None, params=None, body=None, data=None):
        self.seen.append({
            "url": url,
            "headers": headers or {},
            "params": params or {},
            "body": body,
            "data": data,
        })
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def last(self):
        return self.seen[-1]


def openai_response(usage=None):
    payload = {
        "model": "gpt-test-0931",
        "choices": [{"message": {"content": "Dia dhuit."}}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def anthropic_response(usage=None):
    payload = {
        "model": "claude-test-20260101",
        "content": [{"type": "text", "text": "Dia dhuit."}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def gemini_response(usage=None):
    payload = {
        "modelVersion": "gemini-test-002",
        "candidates": [{"content": {"parts": [{"text": "Dia dhuit."}]}}],
    }
    if usage is not None:
        payload["usageMetadata"] = usage
    return payload


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    for env in ("TEST_OPENAI_KEY", "TEST_ANTHROPIC_KEY", "TEST_GEMINI_KEY"):
        monkeypatch.setenv(env, "secret")


def test_declared_deviation_omits_parameters_and_records_the_vendor_reason(monkeypatch):
    calls = Calls(gemini_response())
    monkeypatch.setattr(providers, "_post", calls)
    entry = {**GEMINI_ENTRY, "decoding_deviation": GEMINI_DEVIATION}

    result = providers.translate_chat(entry, "Hello.")

    assert calls.last["body"]["generationConfig"] == {"maxOutputTokens": 2048}
    assert result.decoding_used == {"max_tokens": 2048}
    assert result.deviations == (
        f"omitted temperature, top_p: {GEMINI_DEVIATION['reason']}",
    )


def test_without_a_declared_deviation_gemini_sends_the_full_decoding(monkeypatch):
    calls = Calls(gemini_response())
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_chat(GEMINI_ENTRY, "Hello.")

    assert calls.last["body"]["generationConfig"] == {
        "temperature": 0.0,
        "topP": 1.0,
        "maxOutputTokens": 2048,
    }
    assert result.decoding_used == dict(CHAT_DECODING)
    assert result.deviations == ()


def test_deviation_applies_to_openai_compatible_bodies(monkeypatch):
    calls = Calls(openai_response())
    monkeypatch.setattr(providers, "_post", calls)
    entry = {**OPENAI_ENTRY, "decoding_deviation": GEMINI_DEVIATION}

    result = providers.translate_chat(entry, "Hello.")

    assert "temperature" not in calls.last["body"]
    assert "top_p" not in calls.last["body"]
    assert calls.last["body"]["max_tokens"] == 2048
    assert result.decoding_used == {"max_tokens": 2048}


def test_sampling_rejection_is_retried_and_recorded_as_a_deviation(monkeypatch):
    rejection = ProviderError(
        "HTTP 400: Unsupported 'temperature', 'top_p', and 'max_tokens'",
    )
    calls = Calls(rejection, openai_response())
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_chat(OPENAI_ENTRY, "Hello.")

    retried = calls.last["body"]
    assert "temperature" not in retried and "top_p" not in retried
    assert "max_tokens" not in retried
    assert retried["max_completion_tokens"] == 2048
    assert result.decoding_used == {}
    assert len(result.deviations) == 1
    assert "temperature" in result.deviations[0] and "HTTP 400" in result.deviations[0]


def test_non_sampling_400_is_not_retried(monkeypatch):
    calls = Calls(ProviderError("HTTP 400: model not found"))
    monkeypatch.setattr(providers, "_post", calls)

    with pytest.raises(ProviderError, match="model not found"):
        providers.translate_chat(OPENAI_ENTRY, "Hello.")
    assert len(calls.seen) == 1


def test_openai_request_shape_and_reported_usage(monkeypatch):
    calls = Calls(openai_response(
        {"prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 48},
    ))
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_chat(OPENAI_ENTRY, "Hello.")

    assert calls.last["url"] == "https://api.example.com/v1/chat/completions"
    assert calls.last["headers"]["Authorization"] == "Bearer secret"
    assert calls.last["body"] == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Hello."}],
        **CHAT_DECODING,
    }
    assert result.usage == {
        "prompt_tokens": 41,
        "completion_tokens": 7,
        "total_tokens": 48,
    }
    assert result.model_reported == "gpt-test-0931"


def test_anthropic_request_shape_and_usage_sum(monkeypatch):
    calls = Calls(anthropic_response(
        {"input_tokens": 41, "output_tokens": 7, "cache_read_input_tokens": 0},
    ))
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_chat(ANTHROPIC_ENTRY, "Hello.")

    assert calls.last["url"] == "https://api.anthropic.com/v1/messages"
    assert calls.last["headers"]["anthropic-version"] == "2023-06-01"
    assert calls.last["headers"]["x-api-key"] == "secret"
    assert calls.last["body"] == {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "Hello."}],
        **CHAT_DECODING,
    }
    assert result.decoding_used == dict(CHAT_DECODING)
    assert result.usage == {
        "prompt_tokens": 41,
        "completion_tokens": 7,
        "total_tokens": 48,
    }
    assert result.model_reported == "claude-test-20260101"


def test_anthropic_honours_a_declared_deviation(monkeypatch):
    calls = Calls(anthropic_response())
    monkeypatch.setattr(providers, "_post", calls)
    entry = {**ANTHROPIC_ENTRY, "decoding_deviation": GEMINI_DEVIATION}

    result = providers.translate_chat(entry, "Hello.")

    assert "temperature" not in calls.last["body"]
    assert "top_p" not in calls.last["body"]
    assert calls.last["body"]["max_tokens"] == 2048
    assert result.deviations == (
        f"omitted temperature, top_p: {GEMINI_DEVIATION['reason']}",
    )


def test_gemini_request_shape_and_reported_total(monkeypatch):
    calls = Calls(gemini_response({
        "promptTokenCount": 41,
        "candidatesTokenCount": 7,
        "thoughtsTokenCount": 120,
        "totalTokenCount": 168,
    }))
    monkeypatch.setattr(providers, "_post", calls)

    result = providers.translate_chat(GEMINI_ENTRY, "Hello.")

    assert calls.last["url"].endswith("/models/gemini-test:generateContent")
    assert calls.last["params"] == {"key": "secret"}
    assert calls.last["body"]["contents"] == [{"parts": [{"text": "Hello."}]}]
    assert result.usage == {
        "prompt_tokens": 41,
        "completion_tokens": 7,
        "total_tokens": 168,
    }
    assert result.model_reported == "gemini-test-002"


def test_absent_usage_records_zeros_rather_than_a_guess(monkeypatch):
    monkeypatch.setattr(providers, "_post", Calls(openai_response()))
    result = providers.translate_chat(OPENAI_ENTRY, "Hello.")
    assert result.usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_translate_chat_refuses_an_unknown_provider():
    with pytest.raises(ProviderError, match="not a hosted chat provider"):
        providers.translate_chat({"provider": "unknown"}, "Hello.")


@pytest.mark.parametrize("entry,response,expected_url", [
    (OPENAI_ENTRY, {"data": [{"id": "gpt-test"}]}, "https://api.example.com/v1/models"),
    (ANTHROPIC_ENTRY, {"data": [{"id": "claude-test"}]}, "https://api.anthropic.com/v1/models"),
    (GEMINI_ENTRY, {"models": [{"name": "models/gemini-test"}]},
     "https://generativelanguage.googleapis.com/v1beta/models"),
])
def test_doctor_confirms_pinned_ids_from_each_live_model_list(
        monkeypatch, entry, response, expected_url):
    calls = Calls(response)
    monkeypatch.setattr(providers, "_get", calls)

    result = providers.doctor_check("fixture-system", entry)

    assert result["status"] == "OK"
    assert result["model"] == entry["model"]
    assert calls.last["url"] == expected_url


def test_doctor_reports_every_live_outcome_without_raising(monkeypatch):
    monkeypatch.delenv("TEST_OPENAI_KEY")
    missing = providers.doctor_check("fixture-system", OPENAI_ENTRY)
    assert missing["status"] == "MISSING_KEY"
    assert "TEST_OPENAI_KEY" in missing["detail"]

    monkeypatch.setenv("TEST_OPENAI_KEY", "secret")
    monkeypatch.setattr(providers, "_get", Calls(ProviderError("HTTP 401: unauthorized")))
    failed = providers.doctor_check("fixture-system", OPENAI_ENTRY)
    assert failed["status"] == "AUTH_OR_NETWORK_FAIL"
    assert "401" in failed["detail"]

    monkeypatch.setattr(providers, "_get", Calls({"data": [{"id": "gpt-test-new"}]}))
    absent = providers.doctor_check("fixture-system", OPENAI_ENTRY)
    assert absent["status"] == "MODEL_NOT_FOUND"
    assert "gpt-test-new" in absent["detail"]
