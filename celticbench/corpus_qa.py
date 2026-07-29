"""Measure defects in the built eval sets and commit the measurements.

The manifests prove an eval file has not changed; they say nothing about
whether it was fit to publish in the first place. This module counts the
defects that actually move a translation score -- duplicate rows (a repeated
sentence is a repeated vote), untranslated pairs (a copy-through system scores
"correct"), invisible control characters (they change tokenisation and
character counts), non-NFC text (two encodings of the same accent are two
different strings to chrF and to a tokeniser), and wild length ratios (a
truncated or garbage reference) -- into manifests/corpus-qa.json.

Nothing here mutates the eval files: a defect is reported, never silently
repaired, because repairing it would invalidate every published manifest. The
report contains no timestamps and iterates in a fixed order, so two runs over
identical eval files produce byte-identical output.
"""
from __future__ import annotations

import collections
import math
import os
import re
import unicodedata
from typing import Any

from . import lib

QA_FILENAME = "corpus-qa.json"

# Reference-over-source character ratio band. Chosen from the built Track A
# corpora: FLORES sits inside 0.68-1.86 for all three languages and Tatoeba's
# bulk inside 0.6-2.2, so this band flags a handful of rows per corpus rather
# than a tail of ordinary sentences.
LENGTH_RATIO_BAND = (0.5, 2.0)

# Below this many source characters the ratio is noise -- "Welcome." -> "Do
# bheatha dhan duthaich!" is a ratio of 3.1 and a perfectly good translation --
# so the band is only applied to sentences long enough for it to mean anything.
LENGTH_RATIO_MIN_SOURCE_CHARS = 20

# Cc C0/C1 controls, Cf format characters (U+200B and friends, which really do
# occur in FLORES), Cs lone surrogates, Co private use. None of these belong in
# a Latin-script Celtic or English sentence.
CONTROL_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co"})

_WHITESPACE = re.compile(r"\s+")

DEFINITIONS = {
    "duplicate_rows": "occurrences of a text that already appeared on an earlier row of the same file (n minus distinct texts)",
    "duplicate_groups": "distinct texts occurring on more than one row",
    "blank_rows": "rows that are empty or whitespace-only",
    "control_char_rows": "rows containing a character in Unicode general category Cc, Cf, Cs or Co",
    "non_nfc_rows": "rows that are not already in Unicode NFC",
    "untranslated_rows": "rows whose source and reference are equal after NFC, whitespace collapsing and casefolding; punctuation is not stripped, so this is a lower bound",
    "blank_pair_rows": "rows where either side is empty or whitespace-only",
    "length_ratio": "reference characters over source characters, both NFC-normalised and stripped; rows with a blank side are excluded",
    "length_ratio_percentiles": "nearest-rank on the ascending ratio list, no interpolation, so every reported value is a ratio actually observed in the corpus",
    "length_ratio_outside_band": f"rows with at least {LENGTH_RATIO_MIN_SOURCE_CHARS} source characters whose ratio falls outside {list(LENGTH_RATIO_BAND)}",
    "ids_rows": "line count of the parallel eval/{corpus}.{lang}.ids attribution file, or null when the corpus has no per-row upstream id",
}


def report_path() -> str:
    return os.path.join(lib.MANIFESTS, QA_FILENAME)


def _compare_key(text: str) -> str:
    """Form in which source and reference are compared for untranslated pairs."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text).strip()).casefold()


def _visible_length(text: str) -> int:
    """Character count used for the length ratio.

    NFC first so that a decomposed and a precomposed accent are one character
    either way; the ratio must not depend on which encoding upstream chose.
    """
    return len(unicodedata.normalize("NFC", text).strip())


def _has_control(text: str) -> bool:
    return any(unicodedata.category(char) in CONTROL_CATEGORIES for char in text)


def _percentile(ascending: list[float], quantile: float) -> float:
    """Nearest-rank percentile: no interpolation, no float drift, always observed."""
    index = math.ceil(quantile * len(ascending)) - 1
    return ascending[min(len(ascending) - 1, max(0, index))]


def _side_stats(rows: list[str]) -> dict[str, int]:
    counts = collections.Counter(rows)
    return {
        "duplicate_rows": len(rows) - len(counts),
        "duplicate_groups": sum(1 for count in counts.values() if count > 1),
        "blank_rows": sum(1 for row in rows if not row.strip()),
        "control_char_rows": sum(1 for row in rows if _has_control(row)),
        "non_nfc_rows": sum(1 for row in rows if not unicodedata.is_normalized("NFC", row)),
    }


def analyse_pair(corpus: str, lang: str) -> dict[str, Any]:
    """QA numbers for one built (corpus, lang) pair. Fails closed on a skew."""
    src = lib.read_lines(lib.eval_src_path(corpus, lang))
    ref = lib.read_lines(lib.eval_ref_path(corpus, lang))
    if len(src) != len(ref):
        raise ValueError(
            f"{corpus}.{lang}: src has {len(src)} rows, ref has {len(ref)}; "
            f"the eval files are not parallel and cannot be scored"
        )
    ids_file = lib.eval_ids_path(corpus, lang)
    ids_rows = len(lib.read_lines(ids_file)) if os.path.exists(ids_file) else None

    low, high = LENGTH_RATIO_BAND
    ratios: list[float] = []
    band_rows = 0
    outside_band = 0
    untranslated = 0
    blank_pairs = 0
    for source, reference in zip(src, ref):
        source_len = _visible_length(source)
        reference_len = _visible_length(reference)
        if not source_len or not reference_len:
            blank_pairs += 1
            continue
        if _compare_key(source) == _compare_key(reference):
            untranslated += 1
        ratio = reference_len / source_len
        ratios.append(ratio)
        if source_len >= LENGTH_RATIO_MIN_SOURCE_CHARS:
            band_rows += 1
            if not low <= ratio <= high:
                outside_band += 1
    ratios.sort()

    return {
        "n": len(src),
        "ids_rows": ids_rows,
        "src": _side_stats(src),
        "ref": _side_stats(ref),
        "untranslated_rows": untranslated,
        "blank_pair_rows": blank_pairs,
        "length_ratio": {
            "measured_rows": len(ratios),
            "min": round(ratios[0], 4) if ratios else None,
            "p01": round(_percentile(ratios, 0.01), 4) if ratios else None,
            "median": round(_percentile(ratios, 0.5), 4) if ratios else None,
            "p99": round(_percentile(ratios, 0.99), 4) if ratios else None,
            "max": round(ratios[-1], 4) if ratios else None,
            "band_measured_rows": band_rows,
            "outside_band": outside_band,
        },
    }


def summary_lines(report: dict[str, Any]) -> list[str]:
    """One human-readable line per (corpus, lang), in report order."""
    lines = []
    for corpus, by_lang in report["corpora"].items():
        for lang, entry in by_lang.items():
            ratio = entry["length_ratio"]
            lines.append(
                f"qa {corpus}.{lang}: n={entry['n']}"
                f" dup src {entry['src']['duplicate_rows']} ref {entry['ref']['duplicate_rows']}"
                f" untranslated {entry['untranslated_rows']}"
                f" blank {entry['blank_pair_rows']}"
                f" control {entry['src']['control_char_rows'] + entry['ref']['control_char_rows']}"
                f" non-NFC {entry['src']['non_nfc_rows'] + entry['ref']['non_nfc_rows']}"
                f" ratio {ratio['p01']}/{ratio['median']}/{ratio['p99']}"
                f" outside band {ratio['outside_band']}/{ratio['band_measured_rows']}"
            )
    return lines


def build_corpus_qa() -> dict[str, Any]:
    """Write manifests/corpus-qa.json for every built (corpus, lang) pair."""
    report: dict[str, Any] = {
        "schema": lib.SCHEMA_QA,
        "definitions": DEFINITIONS,
        "length_ratio_band": list(LENGTH_RATIO_BAND),
        "length_ratio_min_source_chars": LENGTH_RATIO_MIN_SOURCE_CHARS,
        "control_categories": sorted(CONTROL_CATEGORIES),
        "corpora": {},
    }
    for corpus in lib.all_corpora():
        for lang in lib.LANGS:
            if not os.path.exists(lib.eval_src_path(corpus, lang)):
                continue
            if not os.path.exists(lib.eval_ref_path(corpus, lang)):
                raise FileNotFoundError(
                    f"{lib.eval_ref_path(corpus, lang)} missing while the src side exists"
                )
            report["corpora"].setdefault(corpus, {})[lang] = analyse_pair(corpus, lang)
    os.makedirs(lib.MANIFESTS, exist_ok=True)
    with open(report_path(), "w", encoding="utf-8") as handle:
        handle.write(lib.canonical_json(report) + "\n")
    return report
