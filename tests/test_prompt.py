"""The fixed prompt is a published contract; lock its shape and hashes."""
from celticbench.lib import sha256_json, sha256_text
from celticbench.prompt import (
    CHAT_DECODING, CHAT_DECODING_SHA256, PROMPT_SHA256, PROMPT_TEMPLATE,
    clean_output, render,
)


def test_render_en_to_celtic():
    prompt = render("kw", "en-xx", "Good morning.")
    assert "English sentence into Cornish" in prompt
    assert prompt.endswith("Good morning.")
    assert "only the Cornish translation" in prompt


def test_render_celtic_to_english():
    prompt = render("gd", "xx-en", "Madainn mhath.")
    assert "Scottish Gaelic sentence into English" in prompt
    assert prompt.endswith("Madainn mhath.")


def test_hashes_are_derived_not_frozen_constants():
    assert PROMPT_SHA256 == sha256_text(PROMPT_TEMPLATE)
    assert CHAT_DECODING_SHA256 == sha256_json(CHAT_DECODING)


def test_decoding_is_deterministic_greedy():
    assert CHAT_DECODING["temperature"] == 0.0
    assert CHAT_DECODING["top_p"] == 1.0


def test_clean_output_flattens_newlines():
    assert clean_output(" Bore da.\nNodyn. ") == "Bore da. Nodyn."
