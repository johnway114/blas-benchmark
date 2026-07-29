"""Track C reference-free metrics: the contamination-immune measurements."""
import os

import pytest

from celticbench.scoring import _is_repetitive, track_c_metrics


def _lid_available():
    from celticbench.langid import MODEL_PATH
    return os.path.exists(MODEL_PATH)


def test_repetition_detector():
    loop = "the cat sat on " * 12
    assert _is_repetitive(loop.strip())
    collapse = "da da da da da da da da da da"
    assert _is_repetitive(collapse)
    assert not _is_repetitive("Myttin da, fatla genes hedhyw?")
    assert not _is_repetitive("short line")


@pytest.mark.skipif(not _lid_available(), reason="lid model not downloaded; run bench.py prepare")
def test_blank_copy_and_off_target_rates():
    sources = [
        "Good morning to you.",
        "The weather is fine today.",
        "I need to keep going.",
        "This line was not translated.",
    ]
    hyps = [
        "",                                     # blank
        "The weather is fine today.",           # copied English, expected Cornish
        "Devam etmem gerekiyordu.",              # Turkish: the measured failure class
        "This line was not translated.",        # copied
    ]
    metrics = track_c_metrics(sources, hyps, expected_lang="kw")
    assert metrics["blank_rate"] == 0.25
    assert metrics["copy_rate"] == 0.5
    # Blanks are never off-target; copies and Turkish should be flagged as
    # confidently non-Cornish. Allow the copy lines or Turkish to individually
    # fall under the confidence threshold, but at least one must fire.
    assert metrics["off_target_rate"] >= 0.25
    assert metrics["off_target_expected_lang"] == "kw"


@pytest.mark.skipif(not _lid_available(), reason="lid model not downloaded; run bench.py prepare")
def test_correct_language_is_not_off_target():
    from celticbench.langid import is_off_target

    assert not is_off_target("Tha an latha brèagha an-diugh.", "gd")
    assert not is_off_target("Mae'r tywydd yn braf heddiw.", "cy")
    assert not is_off_target("I would like a cup of tea, please.", "en")
    assert is_off_target("I would like a cup of tea, please.", "ga")


def test_empty_input_returns_empty_metrics():
    assert track_c_metrics([], [], "ga") == {}
