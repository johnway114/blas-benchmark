"""Language identification for the off-target metric (Track C).

Backend: fastText lid.176.ftz (CC-BY-SA), 176 languages including all six
Celtic languages (ga cy gd br gv kw). Loaded lazily; works with either the
`fasttext` or the lighter `fasttext-predict` package.

Reliability is measured, not assumed: `bench.py prepare` runs the detector
over the gold references and commits per-language recognition rates to
manifests/lid-validation.json. Languages where gold text itself is poorly
recognized get their off-target rate labelled advisory in the scores.
"""
from __future__ import annotations

import os
from typing import Any

from .lib import DATA

MODEL_PATH = os.path.join(DATA, "lid.176.ftz")
CONFIDENCE_THRESHOLD = 0.5

_model: Any = None


def ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"langid model missing at {MODEL_PATH}; run `python bench.py prepare`"
        )
    return MODEL_PATH


def _load() -> Any:
    global _model
    if _model is None:
        ensure_model()
        try:
            from fasttext import load_model  # type: ignore
        except ImportError:
            try:
                from fasttext_predict import load_model  # type: ignore
            except ImportError as exc:
                raise SystemExit(
                    "no fasttext backend: pip install fasttext-predict"
                ) from exc
        _model = load_model(MODEL_PATH)
    return _model


def detect(text: str) -> tuple[str, float]:
    """(iso_639_1_label, probability) for one line; ('', 0.0) for blank."""
    text = text.strip()
    if not text:
        return "", 0.0
    labels, probs = _load().predict(text.replace("\n", " "), k=1)
    if not labels:
        return "", 0.0
    return labels[0].replace("__label__", ""), float(probs[0])


def is_off_target(text: str, expected: str) -> bool:
    """True when the line is confidently written in the wrong language.

    Blank lines are never off-target (they are counted by blank_rate), and
    low-confidence detections are not counted against the system.
    """
    if not text.strip():
        return False
    label, prob = detect(text)
    return label != expected and prob >= CONFIDENCE_THRESHOLD
