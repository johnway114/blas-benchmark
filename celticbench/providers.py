"""Hosted-API provider adapters and doctor probes.

Plain REST via requests; no vendor SDKs, no boto3. The runner reaches every
hosted system through exactly three entry points -- translate_chat,
translate_mt and mt_codes -- so endpoints, auth, request signing and per-vendor
language codes stay behind this module's door. Adapters hold no mutable state
and are safe to call from the runner's thread pool. Retries, pacing and caching
live in runner.py; adapters raise ProviderError on any failure so the runner can
decide.

Every non-obvious endpoint, header and signing step below cites the vendor
documentation it was read from.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import hmac
import html
import json
import os
import urllib.parse
import uuid
from typing import Any

import certifi
import requests

from .lib import LANGS
from .prompt import clean_output
from .registry import credential_envs

TIMEOUT = 90


class ProviderError(Exception):
    """One failed provider call; message is safe to print."""


@dataclasses.dataclass(frozen=True)
class Translation:
    """One provider's answer, plus everything the receipt needs to be audited."""

    text: str
    model_reported: str
    decoding_used: dict[str, Any]
    usage: dict[str, int]
    deviations: tuple[str, ...] = ()


def _key(entry: dict[str, Any]) -> str:
    env = entry.get("key_env")
    key = os.environ.get(env or "", "")
    if not key:
        raise ProviderError(f"missing API key: set {env} in .env")
    return key


def _env(name: str) -> str:
    """A required non-key credential or scope (region, project, secret).

    The wording matters: runner.py fail-fasts the whole run on "missing API key"
    or "missing credential" instead of burning retries on something no retry
    can fix.
    """
    value = os.environ.get(name, "")
    if not value:
        raise ProviderError(f"missing credential: set {name} in .env")
    return value


def _post(url: str, *, headers: dict[str, str] | None = None,
          params: dict[str, str] | None = None, body: Any = None,
          data: bytes | None = None) -> Any:
    """POST and return parsed JSON.

    `body` is serialized by requests; `data` sends exact bytes, which the signed
    providers need because their signature commits to the payload hash.
    """
    try:
        response = requests.post(
            url, headers=headers, params=params,
            json=body if data is None else None, data=data,
            timeout=TIMEOUT, verify=certifi.where(),
        )
    except requests.RequestException as exc:
        raise ProviderError(f"network: {type(exc).__name__}: {str(exc)[:120]}") from exc
    if response.status_code != 200:
        raise ProviderError(f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response: {response.text[:120]}") from exc


def _get(url: str, *, headers: dict[str, str] | None = None,
         params: dict[str, str] | None = None) -> Any:
    try:
        response = requests.get(url, headers=headers, params=params,
                                timeout=TIMEOUT, verify=certifi.where())
    except requests.RequestException as exc:
        raise ProviderError(f"network: {type(exc).__name__}: {str(exc)[:120]}") from exc
    if response.status_code != 200:
        raise ProviderError(f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response: {response.text[:120]}") from exc


# ---------------------------------------------------------------------------
# Decoding and usage: what we actually sent, and what the vendor actually billed
# ---------------------------------------------------------------------------

def _decoding_sent(entry: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Fixed decoding minus any parameter the vendor forces us to drop.

    A registry entry declares `decoding_deviation` when the vendor no longer
    accepts part of the published decoding contract; the omission is recorded as
    a deviation string so the receipt shows the departure instead of hiding it.
    """
    decoding = dict(entry["decoding"])
    deviation = entry.get("decoding_deviation")
    if not deviation:
        return decoding, ()
    omitted = tuple(deviation["omits"])
    for name in omitted:
        decoding.pop(name, None)
    return decoding, (f"omitted {', '.join(omitted)}: {deviation['reason']}",)


def _tokens(value: Any) -> int:
    """A token count the vendor did not report is 0 in the receipt, never a guess."""
    return int(value) if isinstance(value, (int, float)) else 0


def _usage(prompt: Any, completion: Any, total: Any = None) -> dict[str, int]:
    """Normalize one vendor's token counts.

    `total` is summed from the halves only when the vendor reports no total of
    its own (Anthropic); vendors whose total includes hidden reasoning tokens
    report a larger number than the sum and that reported number wins.
    """
    prompt_tokens = _tokens(prompt)
    completion_tokens = _tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": _tokens(total) or prompt_tokens + completion_tokens,
    }


# ---------------------------------------------------------------------------
# Hosted chat models
# ---------------------------------------------------------------------------

_SAMPLING_FIELDS = ("temperature", "top_p", "max_tokens")


def translate_openai_compat(entry: dict[str, Any], prompt: str) -> Translation:
    decoding, deviations = _decoding_sent(entry)
    body: dict[str, Any] = {
        "model": entry["model"],
        "messages": [{"role": "user", "content": prompt}],
        **decoding,
    }
    headers = {"Authorization": f"Bearer {_key(entry)}"}
    url = entry["base_url"].rstrip("/") + "/chat/completions"
    try:
        data = _post(url, headers=headers, body=body)
    except ProviderError as exc:
        message = str(exc)
        if "HTTP 400" not in message or not any(f in message for f in _SAMPLING_FIELDS):
            raise
        # Reasoning tiers on the OpenAI-compatible surface reject the sampling
        # parameters outright and rename the length cap to max_completion_tokens.
        # Retry once without them so the row is scored, and name the departure.
        dropped = [f for f in ("temperature", "top_p") if body.pop(f, None) is not None]
        if "max_tokens" in message and "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
            decoding = {k: v for k, v in decoding.items() if k != "max_tokens"}
            dropped.append("max_tokens (resent as max_completion_tokens)")
        decoding = {k: v for k, v in decoding.items() if k in body}
        deviations += (f"vendor rejected {', '.join(dropped)} with HTTP 400; "
                       "retried without them",)
        data = _post(url, headers=headers, body=body)
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"empty choices: {json.dumps(data)[:200]}")
    content = (choices[0].get("message") or {}).get("content") or ""
    # usage.prompt_tokens / completion_tokens / total_tokens (CompletionUsage):
    # https://github.com/openai/openai-openapi/blob/master/openapi.yaml
    usage = data.get("usage") or {}
    return Translation(
        text=clean_output(content),
        model_reported=str(data.get("model", entry["model"])),
        decoding_used=decoding,
        usage=_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                     usage.get("total_tokens")),
        deviations=deviations,
    )


def translate_anthropic(entry: dict[str, Any], prompt: str) -> Translation:
    decoding, deviations = _decoding_sent(entry)
    body: dict[str, Any] = {
        "model": entry["model"],
        "messages": [{"role": "user", "content": prompt}],
        # The Messages API spells max_tokens/temperature/top_p exactly as the
        # fixed decoding contract does. https://platform.claude.com/docs/en/api/messages
        **decoding,
    }
    headers = {
        "x-api-key": _key(entry),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = _post("https://api.anthropic.com/v1/messages", headers=headers, body=body)
    blocks = data.get("content") or []
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    if not text and blocks:
        text = blocks[-1].get("text", "")
    # usage.input_tokens / usage.output_tokens; the Messages API reports no
    # total, so the receipt's total is their sum.
    # https://platform.claude.com/docs/en/api/messages
    usage = data.get("usage") or {}
    return Translation(
        text=clean_output(text),
        model_reported=str(data.get("model", entry["model"])),
        decoding_used=decoding,
        usage=_usage(usage.get("input_tokens"), usage.get("output_tokens")),
        deviations=deviations,
    )


# generationConfig spells the fixed decoding differently:
# https://ai.google.dev/api/generate-content#GenerationConfig
_GEMINI_CONFIG = {"temperature": "temperature", "top_p": "topP",
                  "max_tokens": "maxOutputTokens"}


def translate_gemini(entry: dict[str, Any], prompt: str) -> Translation:
    decoding, deviations = _decoding_sent(entry)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{entry['model']}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {_GEMINI_CONFIG[k]: v for k, v in decoding.items()
                             if k in _GEMINI_CONFIG},
    }
    data = _post(url, params={"key": _key(entry)}, body=body,
                 headers={"Content-Type": "application/json"})
    candidate = (data.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    # usageMetadata.promptTokenCount / candidatesTokenCount / totalTokenCount;
    # the total also covers thinking tokens, so it is not the sum of the halves.
    # https://ai.google.dev/api/generate-content#UsageMetadata
    usage = data.get("usageMetadata") or {}
    return Translation(
        text=clean_output(text),
        model_reported=str(data.get("modelVersion", entry["model"])),
        decoding_used=decoding,
        usage=_usage(usage.get("promptTokenCount"), usage.get("candidatesTokenCount"),
                     usage.get("totalTokenCount")),
        deviations=deviations,
    )


# ---------------------------------------------------------------------------
# Request signing shared by Amazon SigV4 and Alibaba ACS3
# ---------------------------------------------------------------------------

def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def _canonical_request(method: str, uri: str, query: str,
                       headers: dict[str, str], payload: bytes) -> tuple[str, str]:
    """Canonical request text plus the signed-header list it commits to.

    Amazon and Alibaba specify byte-identical canonicalization: method, URI,
    canonical query string, lowercased and sorted `name:trimmed-value` lines, a
    blank line, the semicolon-joined signed-header names, then the payload hash.
    Callers pass an already-encoded uri and canonical query string.
    https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-create-signed-request.html
    https://www.alibabacloud.com/help/en/sdk/product-overview/v3-request-structure-and-signature
    """
    canonical_headers = "".join(
        f"{name.lower()}:{value.strip()}\n"
        for name, value in sorted(headers.items(), key=lambda item: item[0].lower())
    )
    signed_headers = ";".join(sorted(name.lower() for name in headers))
    request = "\n".join([method, uri, query, canonical_headers, signed_headers,
                         _sha256_hex(payload)])
    return request, signed_headers


_SIGV4_ALGORITHM = "AWS4-HMAC-SHA256"


def _sigv4_string_to_sign(amz_date: str, scope: str, canonical_request: str) -> str:
    """Algorithm, timestamp, credential scope, hash of the canonical request."""
    return "\n".join([_SIGV4_ALGORITHM, amz_date, scope,
                      _sha256_hex(canonical_request.encode())])


def _sigv4_signature(secret: str, date_stamp: str, region: str, service: str,
                     string_to_sign: str) -> str:
    """HMAC chain over date, region, service and the aws4_request terminator."""
    key = _hmac_sha256(f"AWS4{secret}".encode(), date_stamp)
    key = _hmac_sha256(key, region)
    key = _hmac_sha256(key, service)
    key = _hmac_sha256(key, "aws4_request")
    return hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()


def _sigv4_authorization(access_key: str, secret: str, region: str, service: str,
                         amz_date: str, method: str, uri: str, query: str,
                         headers: dict[str, str], payload: bytes) -> str:
    canonical_request, signed_headers = _canonical_request(
        method, uri, query, headers, payload)
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    signature = _sigv4_signature(secret, amz_date[:8], region, service,
                                 _sigv4_string_to_sign(amz_date, scope, canonical_request))
    return (f"{_SIGV4_ALGORITHM} Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}")


_ACS3_ALGORITHM = "ACS3-HMAC-SHA256"


def _acs3_signature(secret: str, canonical_request: str) -> str:
    """Alibaba signs `ACS3-HMAC-SHA256\\n<hash>` with the raw AccessKey secret.

    No date/region key derivation, unlike SigV4.
    https://www.alibabacloud.com/help/en/sdk/product-overview/v3-request-structure-and-signature
    """
    string_to_sign = f"{_ACS3_ALGORITHM}\n{_sha256_hex(canonical_request.encode())}"
    return hmac.new(secret.encode(), string_to_sign.encode(), hashlib.sha256).hexdigest()


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


# ---------------------------------------------------------------------------
# Dedicated MT services
# ---------------------------------------------------------------------------

# Dedicated MT services bill characters, not tokens: the receipt says so with
# zeros rather than inventing a token count from the text.
_MT_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def translate_google_v2(entry: dict[str, Any], text: str, source: str,
                        target: str) -> Translation:
    """Cloud Translation v2 (NMT), API-key auth, HTML-escaped output.

    https://cloud.google.com/translate/docs/reference/rest/v2/translate
    """
    body = {"q": text, "source": source, "target": target, "format": "text"}
    data = _post(entry["endpoint"], params={"key": _key(entry)}, body=body)
    translations = ((data.get("data") or {}).get("translations")) or []
    if not translations:
        raise ProviderError(f"no translations field: {json.dumps(data)[:200]}")
    raw = translations[0].get("translatedText", "")
    return Translation(text=clean_output(html.unescape(raw)), model_reported="nmt",
                       decoding_used={}, usage=_MT_USAGE.copy())


def _google_v3_target(entry: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    """(parent, model resource, headers) for one Cloud Translation v3 call.

    `global` is the documented location for non-regionalized calls on the
    general models, so it is the default when GOOGLE_CLOUD_LOCATION is unset.
    https://docs.cloud.google.com/translate/docs/reference/rest/v3/projects.locations/translateText
    """
    project = _env("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "") or "global"
    parent = f"projects/{project}/locations/{location}"
    headers = {"Authorization": f"Bearer {_key(entry)}",
               "Content-Type": "application/json"}
    return parent, f"{parent}/models/{entry['model']}", headers


def translate_google_translation_llm(entry: dict[str, Any], text: str, source: str,
                                     target: str) -> Translation:
    """Cloud Translation v3 :translateText against the Translation LLM model.

    OAuth bearer token, not an API key. The response echoes the model resource
    normalized to the project *number*, which is what the receipt records.
    https://docs.cloud.google.com/translate/docs/reference/rest/v3/projects.locations/translateText
    """
    parent, model, headers = _google_v3_target(entry)
    body = {
        "contents": [text],
        "mimeType": "text/plain",
        "sourceLanguageCode": source,
        "targetLanguageCode": target,
        "model": model,
    }
    url = f"{entry['endpoint'].rstrip('/')}/{parent}:translateText"
    data = _post(url, headers=headers, body=body)
    translations = data.get("translations") or []
    if not translations:
        raise ProviderError(f"no translations field: {json.dumps(data)[:200]}")
    return Translation(
        text=clean_output(translations[0].get("translatedText", "")),
        model_reported=str(translations[0].get("model") or model),
        decoding_used={}, usage=_MT_USAGE.copy(),
    )


def _deepl_host(key: str) -> str:
    """DeepL API Free keys carry the documented ":fx" suffix and a separate host.

    https://developers.deepl.com/docs/getting-started/auth
    """
    return "https://api-free.deepl.com" if key.endswith(":fx") else "https://api.deepl.com"


def translate_deepl(entry: dict[str, Any], text: str, source: str,
                    target: str) -> Translation:
    """POST /v2/translate, DeepL-Auth-Key scheme, one text per request.

    https://developers.deepl.com/api-reference/translate/request-translation
    """
    key = _key(entry)
    headers = {"Authorization": f"DeepL-Auth-Key {key}",
               "Content-Type": "application/json"}
    body = {"text": [text], "source_lang": source, "target_lang": target}
    data = _post(f"{_deepl_host(key)}/v2/translate", headers=headers, body=body)
    translations = data.get("translations") or []
    if not translations:
        raise ProviderError(f"no translations field: {json.dumps(data)[:200]}")
    return Translation(text=clean_output(translations[0].get("text", "")),
                       model_reported=entry["model"], decoding_used={},
                       usage=_MT_USAGE.copy())


def translate_azure_translator(entry: dict[str, Any], text: str, source: str,
                               target: str) -> Translation:
    """POST /translate?api-version=3.0; request and response are JSON arrays.

    Ocp-Apim-Subscription-Region is required for regional and multi-service
    resources, so the registry demands it rather than assuming a global one.
    https://learn.microsoft.com/en-us/azure/ai-services/translator/text-translation/reference/v3/translate
    https://learn.microsoft.com/en-us/azure/ai-services/translator/text-translation/reference/authentication
    """
    headers = {
        "Ocp-Apim-Subscription-Key": _key(entry),
        "Ocp-Apim-Subscription-Region": _env("AZURE_TRANSLATOR_REGION"),
        "Content-Type": "application/json; charset=UTF-8",
    }
    params = {"api-version": entry["api_version"], "from": source, "to": target}
    data = _post(entry["endpoint"], headers=headers, params=params,
                 body=[{"text": text}])
    results = data if isinstance(data, list) else []
    translations = (results[0].get("translations") if results else None) or []
    if not translations:
        raise ProviderError(f"no translations field: {json.dumps(data)[:200]}")
    return Translation(text=clean_output(translations[0].get("text", "")),
                       model_reported=entry["model"], decoding_used={},
                       usage=_MT_USAGE.copy())


# Amazon Translate is a JSON 1.1 service: one POST to /, the operation named in
# X-Amz-Target as <targetPrefix>.<operation>, SigV4 signing name "translate".
# https://github.com/boto/botocore/blob/develop/botocore/data/translate/2017-07-01/service-2.json
_AWS_TARGET_PREFIX = "AWSShineFrontendService_20170701"
_AWS_SERVICE = "translate"


def _aws_call(entry: dict[str, Any], operation: str, payload: dict[str, Any]) -> Any:
    region = _env("AWS_REGION")
    host = f"{_AWS_SERVICE}.{region}.amazonaws.com"
    body = json.dumps(payload, separators=(",", ":")).encode()
    amz_date = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    signed = {
        "content-type": "application/x-amz-json-1.1",
        "host": host,
        "x-amz-date": amz_date,
        "x-amz-target": f"{_AWS_TARGET_PREFIX}.{operation}",
    }
    authorization = _sigv4_authorization(
        _key(entry), _env("AWS_SECRET_ACCESS_KEY"), region, _AWS_SERVICE,
        amz_date, "POST", "/", "", signed, body)
    headers = {k: v for k, v in signed.items() if k != "host"}
    headers["Authorization"] = authorization
    return _post(f"https://{host}/", headers=headers, data=body)


def translate_aws_translate(entry: dict[str, Any], text: str, source: str,
                            target: str) -> Translation:
    """TranslateText over the JSON 1.1 protocol, signed with local SigV4.

    https://docs.aws.amazon.com/translate/latest/APIReference/API_TranslateText.html
    """
    data = _aws_call(entry, "TranslateText", {
        "Text": text, "SourceLanguageCode": source, "TargetLanguageCode": target,
    })
    translated = data.get("TranslatedText")
    if translated is None:
        raise ProviderError(f"no TranslatedText field: {json.dumps(data)[:200]}")
    return Translation(text=clean_output(translated), model_reported=entry["model"],
                       decoding_used={}, usage=_MT_USAGE.copy())


def translate_alibaba_mt(entry: dict[str, Any], text: str, source: str,
                         target: str) -> Translation:
    """TranslateGeneral, an RPC-style operation signed with ACS3-HMAC-SHA256.

    RPC style fixes the canonical URI at "/" and carries every operation
    parameter as form data, so the payload hash covers the encoded form body and
    the canonical query string is empty.
    https://api.alibabacloud.com/api/alimt/2018-10-12/TranslateGeneral
    https://www.alibabacloud.com/help/en/sdk/product-overview/v3-request-structure-and-signature
    """
    host = urllib.parse.urlsplit(entry["endpoint"]).netloc
    form = {
        "FormatType": "text",
        "Scene": "general",
        "SourceLanguage": source,
        "SourceText": text,
        "TargetLanguage": target,
    }
    body = urllib.parse.urlencode(sorted(form.items()),
                                  quote_via=urllib.parse.quote).encode()
    signed = {
        "content-type": "application/x-www-form-urlencoded",
        "host": host,
        "x-acs-action": entry["model"],
        "x-acs-content-sha256": _sha256_hex(body),
        "x-acs-date": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "x-acs-signature-nonce": uuid.uuid4().hex,
        "x-acs-version": entry["api_version"],
    }
    canonical_request, signed_headers = _canonical_request(
        "POST", "/", "", signed, body)
    headers = {k: v for k, v in signed.items() if k != "host"}
    headers["Authorization"] = (
        f"{_ACS3_ALGORITHM} Credential={_key(entry)},SignedHeaders={signed_headers},"
        f"Signature={_acs3_signature(_env('ALIBABA_ACCESS_KEY_SECRET'), canonical_request)}"
    )
    data = _post(entry["endpoint"], headers=headers, data=body)
    if str(data.get("Code", "")) not in ("200", ""):
        raise ProviderError(f"TranslateGeneral {data.get('Code')}: {data.get('Message')}")
    translated = (data.get("Data") or {}).get("Translated")
    if translated is None:
        raise ProviderError(f"no Data.Translated field: {json.dumps(data)[:200]}")
    return Translation(
        text=clean_output(translated), model_reported=entry["model"],
        # mt.aliyuncs.com is the shared international host the official SDK maps
        # every non-Chinese region to; only mt.<region>.aliyuncs.com names one.
        # https://github.com/aliyun/alibabacloud-python-sdk/blob/master/alimt-20181012/alibabacloud_alimt20181012/client.py
        decoding_used={"endpoint_host": host, "region": _alibaba_region(host)},
        usage=_MT_USAGE.copy(),
    )


def _alibaba_region(host: str) -> str:
    labels = host.split(".")
    return labels[1] if len(labels) == 4 else "international"


# ---------------------------------------------------------------------------
# Dispatch: the runner's whole view of this module
# ---------------------------------------------------------------------------

_CHAT_ADAPTERS = {
    "openai_compat": translate_openai_compat,
    "anthropic": translate_anthropic,
    "gemini": translate_gemini,
}

_MT_ADAPTERS = {
    "google_translate_v2": translate_google_v2,
    "google_translation_llm": translate_google_translation_llm,
    "deepl": translate_deepl,
    "azure_translator": translate_azure_translator,
    "aws_translate": translate_aws_translate,
    "alibaba_mt": translate_alibaba_mt,
}

# The language-table column and the provider's own English codes, as (source,
# target). Only DeepL splits the two: bare EN is source-only and a regional
# variant is required as a target, so EN-GB is pinned for these references.
# https://developers.deepl.com/docs/getting-started/supported-languages
_MT_LANGS = {
    "google_translate_v2": ("google", ("en", "en")),
    "google_translation_llm": ("google_tllm", ("en", "en")),
    "deepl": ("deepl", ("EN", "EN-GB")),
    "azure_translator": ("azure", ("en", "en")),
    "aws_translate": ("aws", ("en", "en")),
    "alibaba_mt": ("alibaba", ("en", "en")),
}


def translate_chat(entry: dict[str, Any], prompt: str) -> Translation:
    adapter = _CHAT_ADAPTERS.get(entry["provider"])
    if adapter is None:
        raise ProviderError(f"{entry['provider']!r} is not a chat provider")
    return adapter(entry, prompt)


def translate_mt(entry: dict[str, Any], text: str, source: str,
                 target: str) -> Translation:
    adapter = _MT_ADAPTERS.get(entry["provider"])
    if adapter is None:
        raise ProviderError(f"{entry['provider']!r} is not a dedicated MT provider")
    return adapter(entry, text, source, target)


def mt_codes(entry: dict[str, Any], lang: str, direction: str) -> tuple[str, str]:
    """(source, target) codes for one MT provider, in that provider's vocabulary."""
    mapping = _MT_LANGS.get(entry["provider"])
    if mapping is None:
        raise ProviderError(f"no language mapping for provider {entry['provider']!r}")
    if direction not in ("en-xx", "xx-en"):
        raise ProviderError(f"unknown direction {direction!r}")
    column, (english_source, english_target) = mapping
    celtic = LANGS[lang][column]
    if not celtic:
        # The registry refuses uncovered combinations before a run starts, so
        # this fires only if the registry and the language table disagree. Never
        # substitute a neighbouring language to make the call succeed.
        raise ProviderError(
            f"{entry['provider']} publishes no code for {lang}; refusing to substitute")
    if direction == "en-xx":
        return english_source, celtic
    return celtic, english_target


# ---------------------------------------------------------------------------
# Doctor probes: are the credentials usable, does the pinned model still exist,
# and does the vendor's own language list still cover what we registered?
# ---------------------------------------------------------------------------

# Alibaba's alimt product publishes no list-languages operation -- only
# GetDetectLanguage, which detects rather than enumerates -- so there is nothing
# honest to probe against.
# https://api.alibabacloud.com/meta/v1/products/alimt/versions/2018-10-12/api-docs.json
_NO_LANGUAGE_PROBE = {
    "alibaba_mt": ("Alibaba publishes no list-languages API operation for alimt "
                   "2018-10-12; coverage is taken from its published language "
                   "code list and cannot be probed live"),
}


def list_models(entry: dict[str, Any]) -> list[str]:
    """Model IDs for chat providers, language codes for MT providers."""
    provider = entry["provider"]
    if provider == "openai_compat":
        data = _get(entry["base_url"].rstrip("/") + "/models",
                    headers={"Authorization": f"Bearer {_key(entry)}"})
        return sorted(str(item.get("id", "")) for item in data.get("data", []))
    if provider == "anthropic":
        data = _get("https://api.anthropic.com/v1/models",
                    headers={"x-api-key": _key(entry), "anthropic-version": "2023-06-01"})
        return sorted(str(item.get("id", "")) for item in data.get("data", []))
    if provider == "gemini":
        data = _get("https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": _key(entry), "pageSize": "1000"})
        return sorted(str(item.get("name", "")).removeprefix("models/")
                      for item in data.get("models", []))
    if provider == "google_translate_v2":
        data = _get("https://translation.googleapis.com/language/translate/v2/languages",
                    params={"key": _key(entry), "target": "en"})
        langs = ((data.get("data") or {}).get("languages")) or []
        return sorted(str(item.get("language", "")) for item in langs)
    if provider == "google_translation_llm":
        # supportedLanguages takes the model resource, so this is the Translation
        # LLM's own coverage rather than the default NMT model's.
        # https://docs.cloud.google.com/translate/docs/reference/rest/v3/projects.locations/getSupportedLanguages
        parent, model, headers = _google_v3_target(entry)
        data = _get(f"{entry['endpoint'].rstrip('/')}/{parent}/supportedLanguages",
                    headers=headers, params={"model": model})
        return sorted(str(item.get("languageCode", ""))
                      for item in data.get("languages") or [])
    if provider == "deepl":
        # /v3/languages replaced the deprecated /v2/languages; resource picks the
        # endpoint whose coverage we care about.
        # https://developers.deepl.com/docs/languages/using-the-languages-api
        key = _key(entry)
        data = _get(f"{_deepl_host(key)}/v3/languages",
                    headers={"Authorization": f"DeepL-Auth-Key {key}"},
                    params={"resource": "translate_text"})
        return sorted(str(item.get("lang", "")) for item in data or [])
    if provider == "azure_translator":
        # /languages needs no authentication; scope=translation is the group the
        # translate endpoint uses.
        # https://learn.microsoft.com/en-us/azure/ai-services/translator/text-translation/reference/v3/languages
        root = entry["endpoint"].rsplit("/", 1)[0]
        data = _get(f"{root}/languages",
                    params={"api-version": entry["api_version"], "scope": "translation"})
        return sorted(str(code) for code in (data.get("translation") or {}))
    if provider == "aws_translate":
        # ListLanguages caps MaxResults at 500, above the current language count.
        # https://docs.aws.amazon.com/translate/latest/APIReference/API_ListLanguages.html
        data = _aws_call(entry, "ListLanguages", {"MaxResults": 500})
        return sorted(str(item.get("LanguageCode", ""))
                      for item in data.get("Languages") or [])
    raise ProviderError(_NO_LANGUAGE_PROBE.get(provider)
                        or f"no doctor probe for provider {provider!r}")


def _doctor_local(result: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    token_env = entry.get("token_env")
    if entry.get("gated") and not os.environ.get(token_env or "", ""):
        result["status"] = "NEEDS_HF_TOKEN"
        result["detail"] = (f"gated weights: accept the licence on the model page "
                            f"and set {token_env} in .env")
        return result
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        result["status"] = "NEEDS_DEPS"
        result["detail"] = "pip install -r requirements-local.txt to run this anchor"
        return result
    result["status"] = "OK"
    result["detail"] = "local anchor; torch+transformers importable, no key needed"
    return result


def _doctor_mt(result: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    detail = _NO_LANGUAGE_PROBE.get(entry["provider"])
    if detail:
        result["status"] = "NO_LANGUAGE_PROBE"
        result["detail"] = detail
        return result
    try:
        published = list_models(entry)
    except ProviderError as exc:
        result["status"] = "AUTH_OR_NETWORK_FAIL"
        result["detail"] = str(exc)
        return result
    column, english = _MT_LANGS[entry["provider"]]
    registered = {lang for langs in entry["supported"].values() for lang in langs}
    expected = sorted({LANGS[lang][column] for lang in registered} | set(english))
    available = {code.lower() for code in published}
    missing = [code for code in expected if code.lower() not in available]
    result["status"] = "OK" if not missing else "LANGS_MISSING"
    result["detail"] = (f"vendor language list covers all {len(expected)} registered codes"
                        if not missing
                        else f"codes absent from the vendor language list: {missing}")
    return result


def doctor_check(system_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """One registry entry -> status dict. Never raises."""
    result: dict[str, Any] = {"system": system_id, "provider": entry["provider"],
                              "model": entry["model"], "pin_status": entry["pin_status"]}
    if entry["provider"] == "hf_local":
        return _doctor_local(result, entry)
    missing_envs = [env for env in credential_envs(entry) if not os.environ.get(env, "")]
    if missing_envs:
        result["status"] = "MISSING_KEY"
        result["detail"] = f"set {', '.join(missing_envs)} in .env"
        return result
    if entry["provider"] in _MT_ADAPTERS:
        return _doctor_mt(result, entry)
    try:
        models = list_models(entry)
    except ProviderError as exc:
        result["status"] = "AUTH_OR_NETWORK_FAIL"
        result["detail"] = str(exc)
        return result
    model = entry["model"]
    if model in models:
        result["status"] = "OK"
        result["detail"] = "pinned model ID confirmed against live model list"
    else:
        hint = entry.get("family_hint", "")
        suggestions = [m for m in models if hint and hint in m.lower()][:8]
        result["status"] = "MODEL_NOT_FOUND"
        result["detail"] = (f"pinned ID not in live list; similar: {suggestions}"
                            if suggestions else "pinned ID not in live list; inspect full list")
    return result
