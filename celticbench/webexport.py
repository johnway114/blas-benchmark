"""Render scores/scores.json into the published web data contract.

The website must never re-derive method logic. Anything that decides how a
number is read -- whether an off-target rate is authoritative, whether a row
is rankable, which flags a row carries, whether the edition is even complete
-- is decided here, once, by the same code that renders LEADERBOARD.md, and
shipped as data. A renderer that reimplements a rule will eventually disagree
with the leaderboard, and then two published numbers mean two different
things.

Output: `scores/web.json`, schema `celticbench.web.v1`.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .leaderboard import SMALL_N, _flags, _lid_reliability
from .lib import (
    HERE,
    LANGS,
    MANIFESTS,
    METHOD_VERSION,
    SCORES,
    TRACK_B_PREFIX,
    manifest_path,
)
from .prompt import PROMPT_TEMPLATE
from .registry import SYSTEMS, matrix

SCHEMA_WEB = "celticbench.web.v1"

REPO_URL = "https://github.com/johnway114/blas-benchmark"

DIRECTION_LABELS = {
    "en-xx": "English into Celtic",
    "xx-en": "Celtic into English",
}

# Why a corpus exists and how far its numbers carry. The site prints this
# verbatim next to every table so a reader never has to find METHODOLOGY.md to
# learn that a Tatoeba row is not a FLORES row.
CORPUS_NOTES = {
    "flores": {
        "track": "A",
        "label": "FLORES-200 devtest",
        "provenance": "professional human translations, CC-BY-SA 4.0",
        "comparability": "comparable",
        "note": (
            "The public comparability anchor. Old enough to be in the training "
            "data of every system measured here, so it proves comparability, "
            "not generalization."
        ),
    },
    "tatoeba": {
        "track": "A",
        "label": "Tatoeba EN-pairs",
        "provenance": "community sentences, CC-BY 2.0 FR",
        "comparability": "coverage-only",
        "note": (
            "Coverage-only. Public, widely redistributed, and short; scores "
            "here are potentially contamination-inflated and do not transfer "
            "to document-length translation."
        ),
    },
}

TRACK_B_NOTE = {
    "track": "B",
    "provenance": "official publishers, sealed before any model saw it",
    "comparability": "fresh",
    "note": (
        "Bilingual text published after the newest benchmarked training "
        "cutoff, hash-committed before any model saw it. Institutional "
        "register only: it says nothing about conversational or literary "
        "translation."
    ),
}

# v3 computes one corpus-level score per run. There is no bootstrap, no
# confidence interval and no significance test, so a gap of a point or two
# between two systems is not a finding. Saying that in the data contract is
# the only way to stop a renderer implying precision the method does not have.
STATISTICS = {
    "confidence_intervals": False,
    "runs_per_cell": 1,
    "significance_testing": False,
    "note": (
        "Single-run point estimates. v3 does not bootstrap, publishes no "
        "confidence intervals, and runs no significance test, so adjacent "
        "scores are not distinguishable and the ordering within a table is "
        "presentation, not a ranking claim. Treat only large, consistent gaps "
        "as real."
    ),
    "planned": (
        "Paired bootstrap resampling with 95% intervals and significance-"
        "tested tie groups is the next method change; it will bump the method "
        "version because it changes how every row is read."
    ),
}

# A high chrF++ can hide a system that answered in English, answered nothing,
# or echoed the source back. Ranking on overlap alone lets that outrank an
# honest attempt, so a row has to clear these rates before it is allowed to
# win anything. The thresholds are published with the data rather than left
# implicit in a renderer: an unstated "good zone" is decoration, and a reader
# who cannot see the boundary cannot dispute it.
INTEGRITY_FLOOR = {
    "off_target_rate": 0.01,
    "blank_rate": 0.005,
    "copy_rate": 0.02,
    "note": (
        "All three are lower-is-better. A row that exceeds any of them is "
        "still published with its score, and is excluded from winning its "
        "cell: an overlap metric cannot tell the difference between a good "
        "translation and fluent output in the wrong language, so the "
        "integrity rates gate the comparison rather than sitting in an "
        "appendix."
    ),
}


def _load_scores() -> dict[str, Any]:
    path = os.path.join(SCORES, "scores.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run bench.py score first")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_manifest(name: str) -> dict[str, Any] | None:
    path = os.path.join(MANIFESTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _corpus_n(corpus: str, lang: str) -> int:
    """Rows in a built eval set, read from its pinned manifest.

    Mirrors `bench.py plan` so the request total published on the site is the
    same number the operator saw before spending.
    """
    path = manifest_path(corpus, lang)
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        return int(json.load(handle)["n"])


def _panel() -> list[dict[str, Any]]:
    """The frozen roster, with every disclosure the site has to print."""
    panel = []
    for system_id, entry in SYSTEMS.items():
        deviation = entry.get("decoding_deviation")
        panel.append({
            "id": system_id,
            "label": entry["label"],
            "vendor": entry["vendor"],
            "provider": entry["provider"],
            "model": entry["model"],
            "pin_status": entry["pin_status"],
            "license": entry["license"],
            "tier": entry["tier"],
            "reasoning": bool(entry.get("reasoning")),
            "decoding_deviation": (
                {"omits": list(deviation["omits"]), "reason": deviation["reason"]}
                if deviation else None
            ),
        })
    panel.sort(key=lambda item: item["label"])
    return panel


def _trackb_slices() -> dict[str, dict[str, Any]]:
    """Sealed Track B slices, keyed by corpus id, with per-language status.

    The languages with no viable publisher are carried through with their
    reasons. Their absence is a published finding about parallel publishing in
    Breton, Manx and Cornish, so a consumer that only reads `languages` would
    render a gap where the benchmark has an answer.
    """
    slices: dict[str, dict[str, Any]] = {}
    if not os.path.isdir(MANIFESTS):
        return slices
    for name in sorted(os.listdir(MANIFESTS)):
        if not (name.startswith(TRACK_B_PREFIX) and name.endswith(".slice.json")):
            continue
        record = _read_manifest(name)
        if not record:
            continue
        corpus = name.split(".", 1)[0]
        languages: dict[str, dict[str, Any]] = {}
        for lang, detail in (record.get("languages") or {}).items():
            languages[lang] = {
                "status": "sealed",
                "n": detail.get("n"),
                "documents": detail.get("documents"),
                "sources": detail.get("sources") or [],
                "rejected": detail.get("rejected") or {},
                "reason": None,
            }
        for entry in record.get("unavailable") or []:
            languages[entry["lang"]] = {
                "status": "unavailable",
                "n": None,
                "documents": None,
                "sources": [],
                "rejected": {},
                "reason": entry.get("reason"),
            }
        slices[corpus] = {
            "slice": record.get("slice", corpus.removeprefix(TRACK_B_PREFIX)),
            "cutoff": record.get("cutoff"),
            "harvest_date": record.get("harvest_date"),
            "sealed_utc": record.get("harvested_at"),
            "extractor_version": record.get("extractor_version"),
            "limit_per_lang": record.get("limit_per_lang"),
            "languages": languages,
        }
    return slices


def _corpora(corpora: list[str], slices: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for corpus in corpora:
        if corpus in CORPUS_NOTES:
            out.append({"id": corpus, **CORPUS_NOTES[corpus]})
        elif corpus.startswith(TRACK_B_PREFIX):
            record = slices.get(corpus, {})
            out.append({
                "id": corpus,
                "label": f"Fresh harvest {record.get('slice', corpus)}",
                "cutoff": record.get("cutoff"),
                "sealed_utc": record.get("sealed_utc"),
                **TRACK_B_NOTE,
            })
        else:
            out.append({
                "id": corpus,
                "label": corpus,
                "track": "?",
                "comparability": "unknown",
                "note": "Unrecognised corpus; no published reading guidance.",
            })
    return out


def _reliability_label(
    advisory: dict[tuple[str, str, str], bool],
    expected: str,
    lang: str,
    corpus: str,
) -> str:
    """authoritative | advisory | unmeasured for one off-target rate.

    No measurement is not the same as a good measurement, so a missing entry
    becomes `unmeasured` rather than silently passing as reliable.
    """
    flag = advisory.get((expected, lang, corpus))
    if flag is None:
        return "unmeasured"
    return "advisory" if flag else "authoritative"


def _language_table(payload: dict[str, Any], slices: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scored = payload.get("results") or []
    langs = []
    for code, meta in LANGS.items():
        available = sorted({row["corpus"] for row in scored if row["lang"] == code})
        trackb = {
            corpus: record["languages"].get(code)
            for corpus, record in slices.items()
            if record["languages"].get(code)
        }
        langs.append({
            "code": code,
            "name": meta["name"],
            "has_flores": meta["flores"] is not None,
            "scored_corpora": available,
            "trackb": trackb or None,
        })
    return langs


def _integrity_failures(row: dict[str, Any], reliability: str | None) -> list[str]:
    """Which integrity rates this row exceeds, in published order.

    Off-target only gates where the detector is authoritative for that
    language and corpus. On Tatoeba the detector's own false-positive rate on
    gold Celtic text runs to 5-9%, well above the 1% floor, so gating there
    would fail every system for the detector's errors rather than their own.
    Blank and copy involve no detector and always gate.

    A rate that should be gateable and is missing is not a pass: the row
    cannot demonstrate it cleared a floor nobody measured.
    """
    failures = []
    off_target = row.get("off_target_rate")
    if reliability == "authoritative":
        if off_target is None:
            failures.append("off_target_rate unmeasured")
        elif off_target > INTEGRITY_FLOOR["off_target_rate"]:
            failures.append("off_target_rate")
    for metric in ("blank_rate", "copy_rate"):
        value = row.get(metric)
        if value is None:
            failures.append(f"{metric} unmeasured")
        elif value > INTEGRITY_FLOOR[metric]:
            failures.append(metric)
    return failures


def _result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Score rows, each carrying its own reading rules.

    Two flags decide what a renderer may do with a row. `rankable` says the
    row is big enough and complete enough to order at all -- Manx at n=18 is
    published because the scarcity is the finding, and excluded from ordering
    for the same reason. `integrity_ok` says the output was actually in the
    requested language often enough for an overlap score to mean anything.
    Both must hold before a row can win its cell.
    """
    advisory = _lid_reliability()
    rows = []
    for row in payload.get("results") or []:
        expected = row.get("off_target_expected_lang") or row["lang"]
        reliability = (
            _reliability_label(advisory, expected, row["lang"], row["corpus"])
            if row.get("off_target_rate") is not None else None
        )
        failures = _integrity_failures(row, reliability)
        rows.append({
            "system": row["system"],
            "label": row["label"],
            "vendor": row.get("vendor"),
            "lang": row["lang"],
            "language_name": row.get("language_name") or LANGS[row["lang"]]["name"],
            "direction": row["direction"],
            "corpus": row["corpus"],
            "n": row["n"],
            "chrf_pp": row["chrf_pp"],
            "bleu": row["bleu"],
            "off_target_rate": row.get("off_target_rate"),
            "off_target_n": row.get("off_target_n"),
            "off_target_reliability": reliability,
            "blank_rate": row.get("blank_rate"),
            "copy_rate": row.get("copy_rate"),
            "repetition_rate": row.get("repetition_rate"),
            "length_ratio_median": row.get("length_ratio_median"),
            "length_ratio_outlier_rate": row.get("length_ratio_outlier_rate"),
            "fails": row.get("fails", 0),
            "model_reported": row.get("model_reported"),
            "pin_status": row.get("pin_status"),
            "decoding_deviations": row.get("decoding_deviations") or [],
            "reasoning": row.get("reasoning"),
            "partial_slice": bool(row.get("partial_slice")),
            "rankable": row["n"] >= SMALL_N and not row.get("partial_slice"),
            "integrity_failures": failures,
            "integrity_ok": not failures,
            "flags": [f for f in _flags(row).split("; ") if f],
            "receipt": row.get("receipt"),
            "hypothesis_file": row.get("hypothesis_file"),
        })
    rows.sort(key=lambda r: (r["direction"], r["lang"], r["corpus"], -r["chrf_pp"]))
    return rows


def _edition(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Edition completeness, stated as a number rather than implied.

    A partly-run edition that renders like a finished one is the single
    easiest way to publish a wrong ranking, so the count of scored runs
    against the planned matrix is part of the contract.
    """
    planned = [row for row in matrix() if row.get("supported")]
    expected_runs = len(planned)
    expected_requests = sum(_corpus_n(row["corpus"], row["lang"]) for row in planned)
    scored_runs = len(rows)
    if scored_runs == 0:
        status = "not-started"
    elif scored_runs < expected_runs:
        status = "in-progress"
    else:
        status = "complete"
    return {
        "method_version": payload.get("method_version", METHOD_VERSION),
        "generated_utc": payload.get("generated_utc"),
        "status": status,
        "runs_scored": scored_runs,
        "runs_expected": expected_runs,
        "requests_expected": expected_requests,
        "systems": len(SYSTEMS),
        "languages": len(LANGS),
        "directions": list(DIRECTION_LABELS),
    }


def build() -> dict[str, Any]:
    payload = _load_scores()
    slices = _trackb_slices()
    rows = _result_rows(payload)
    qa = _read_manifest("corpus-qa.json")
    lid = _read_manifest("lid-validation.json")

    return {
        "schema": SCHEMA_WEB,
        "edition": _edition(payload, rows),
        "provenance": {
            "repo": REPO_URL,
            "runtime": payload.get("runtime") or {},
            "metric_signatures": payload.get("metric_signatures") or {},
            "langid": payload.get("langid") or {},
            "prompt_template": PROMPT_TEMPLATE,
            "harness_versions": sorted({
                row["harness_version"]
                for row in (payload.get("results") or [])
                if row.get("harness_version")
            }),
        },
        "statistics": STATISTICS,
        "integrity_floor": INTEGRITY_FLOOR,
        "direction_labels": DIRECTION_LABELS,
        "panel": _panel(),
        "languages": _language_table(payload, slices),
        "corpora": _corpora(list(payload.get("corpora") or []), slices),
        "trackb": slices,
        "results": rows,
        "coverage": payload.get("coverage") or [],
        "excluded": payload.get("excluded") or [],
        "corpus_qa": {
            "measured_utc": (qa or {}).get("generated_utc"),
            "languages": (qa or {}).get("languages"),
        } if qa else None,
        "lid_validation": {
            "model": (lid or {}).get("model"),
            "model_sha256": (lid or {}).get("model_sha256"),
            "threshold": (lid or {}).get("advisory_false_positive_threshold"),
            "languages": (lid or {}).get("languages"),
            "english": (lid or {}).get("english"),
        } if lid else None,
    }


def write(path: str | None = None) -> str:
    """Write scores/web.json and report what a consumer will see."""
    document = build()
    target = path or os.path.join(SCORES, "web.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    edition = document["edition"]
    print(f"wrote {os.path.relpath(target, HERE)}")
    print(
        f"  edition {edition['method_version']} {edition['status']}: "
        f"{edition['runs_scored']}/{edition['runs_expected']} runs scored"
    )
    if document["excluded"]:
        print(f"  {len(document['excluded'])} excluded runs carried through")
    return target
