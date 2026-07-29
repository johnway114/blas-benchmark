"""Render scores/scores.json into LEADERBOARD.md.

The renderer never computes a metric. Everything it prints was verified and
measured by scoring.py; its whole job is to present the numbers with the
caveats attached, so a row can never be read as more than it is: which
detector produced the off-target rate, which pins were provisional, which
systems were forced off the common decoding, and which cells are empty
because the vendor does not offer the language at all.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any

from .lib import HERE, LANGS, SCORES

SMALL_N = 50  # below this a score is directional, not comparable

TIER_LABELS = {
    "flagship-chat": "General-purpose models (flagship tier)",
}


def _load() -> dict[str, Any]:
    path = os.path.join(SCORES, "scores.json")
    if not os.path.exists(path):
        raise SystemExit("no scores/scores.json yet; run `python bench.py score` first")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _flags(row: dict[str, Any]) -> str:
    flags = []
    if row.get("benchmark_only"):
        flags.append("NC licence, benchmark-only")
    if row.get("pin_status") in ("provisional", "alias"):
        flags.append(f"pin: {row['pin_status']}")
    if row.get("partial_slice"):
        flags.append("partial slice")
    if row["n"] < SMALL_N:
        flags.append(f"directional (n={row['n']})")
    if row.get("fails"):
        flags.append(f"{row['fails']} failed lines")
    if row.get("model_reported") == "mixed":
        flags.append("vendor reported more than one model ID")
    if row.get("reasoning"):
        flags.append("reasoning model")
    if row.get("decoding_deviations"):
        flags.append("decoding deviation")
    return "; ".join(flags)


def _lid_reliability() -> dict[tuple[str, str, str], bool]:
    """(expected_lang, lang, corpus) -> True when off-target is advisory there.

    Advisory means the detector confidently mislabels more than the allowed
    share of *gold* lines of that language in that corpus, so a system's
    off-target rate has a measurable false-positive floor. The expected
    language is whatever the hypothesis was supposed to be written in, which
    is English for XX -> EN: judging an English hypothesis by the detector's
    Manx error rate would advertise the wrong measurement. Rates and
    threshold come from manifests/lid-validation.json (built by prepare).
    """
    path = os.path.join(HERE, "manifests", "lid-validation.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    threshold = float(report.get("advisory_false_positive_threshold", 0.05))
    advisory: dict[tuple[str, str, str], bool] = {}
    for lang, by_corpus in report.get("languages", {}).items():
        for corpus, stats in by_corpus.items():
            if isinstance(stats, dict):
                advisory[(lang, lang, corpus)] = stats.get("false_positive_rate", 1.0) > threshold
    for corpus, by_lang in report.get("english", {}).items():
        for lang, stats in by_lang.items():
            if isinstance(stats, dict):
                advisory[("en", lang, corpus)] = stats.get("false_positive_rate", 1.0) > threshold
    return advisory


def _provenance_lines(payload: dict[str, Any]) -> list[str]:
    lines = []
    signatures = payload.get("metric_signatures") or {}
    if signatures:
        lines.append(f"- Metrics: `{signatures.get('chrf_pp', '?')}`, `{signatures.get('bleu', '?')}`")
    langid = payload.get("langid") or {}
    if langid.get("model"):
        sha = (langid.get("model_sha256") or "")[:12]
        lines.append(f"- Off-target detector: `{langid['model']}` sha256 `{sha}`, "
                     f"confidence >= {langid.get('confidence_threshold')}")
    runtime = payload.get("runtime") or {}
    if runtime:
        rendered = ", ".join(f"{k} {v}" for k, v in sorted(runtime.items()))
        lines.append(f"- Runtime: {rendered}")
    return lines


def _coverage_table(payload: dict[str, Any]) -> list[str]:
    """Render system x language coverage separately for each direction."""
    coverage = payload.get("coverage") or []
    if not coverage:
        return []
    scored = {(r["system"], r["lang"], r["direction"]) for r in payload["results"]}
    supported = {(r["system"], r["lang"], r["direction"]): r["supported"] for r in coverage}
    systems = sorted({row["system"] for row in coverage})

    lines = ["## Coverage", ""]
    for direction, label in (("en-xx", "English -> Celtic"), ("xx-en", "Celtic -> English")):
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| System | " + " | ".join(LANGS) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in LANGS) + " |")
        for system_id in systems:
            cells = []
            for lang in LANGS:
                key = (system_id, lang, direction)
                if not supported.get(key):
                    cells.append("n/a")
                elif key in scored:
                    cells.append("ok")
                else:
                    cells.append(".")
            lines.append(f"| {system_id} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend([
        "`ok` scored, `.` runnable but not run yet, `n/a` the vendor's own "
        "language list does not offer it. An `n/a` is a coverage fact, not a "
        "score of zero.",
        "",
    ])
    return lines


def render() -> str:
    payload = _load()
    lid_note = _lid_reliability()
    lines: list[str] = []
    lines.append("# Celtic Translation Benchmark - leaderboard")
    lines.append("")
    version = payload.get("method_version", "unversioned")
    lines.append(f"Method **{version}**, generated from `scores/scores.json` "
                 f"({payload['generated_utc']}). Every row is backed by a verified "
                 "receipt in `out/`; excluded runs are listed at the bottom with "
                 "reasons. Rows from another method version are refused, not ranked: "
                 "`METHODOLOGY.md`, `CHANGELOG.md`.")
    lines.append("")
    provenance = _provenance_lines(payload)
    if provenance:
        lines.extend(provenance)
        lines.append("")

    results = payload["results"]
    for direction, direction_label in (("en-xx", "English -> Celtic"), ("xx-en", "Celtic -> English")):
        rows_dir = [r for r in results if r["direction"] == direction]
        if not rows_dir:
            continue
        lines.append(f"## {direction_label}")
        lines.append("")
        for lang in LANGS:
            rows = [r for r in rows_dir if r["lang"] == lang]
            if not rows:
                continue
            lines.append(f"### {LANGS[lang]['name']} ({lang})")
            lines.append("")
            lines.append("| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |")
            lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            for row in sorted(rows, key=lambda r: (r["corpus"], -r["chrf_pp"])):
                off = row.get("off_target_rate")
                off_str = f"{off:.1%}" if off is not None else "-"
                expected = row.get("off_target_expected_lang", lang)
                if off is not None:
                    # No measurement is not the same as a good measurement.
                    note = lid_note.get((expected, lang, row["corpus"]), "unmeasured")
                    if note is True:
                        off_str += " (advisory)"
                    elif note == "unmeasured":
                        off_str += " (unmeasured)"
                tier = TIER_LABELS.get(row["tier"], row["tier"]).split(" (")[0]
                lines.append(
                    f"| {row['label']} | {tier} | {row['corpus']} | {row['n']} "
                    f"| {row['chrf_pp']:.1f} | {row['bleu']:.1f} | {off_str} "
                    f"| {row.get('blank_rate', 0):.1%} | {row.get('copy_rate', 0):.1%} "
                    f"| {_flags(row)} |"
                )
            lines.append("")

    lines.extend(_coverage_table(payload))

    if payload.get("excluded"):
        lines.append("## Excluded runs (fail-closed)")
        lines.append("")
        for item in payload["excluded"]:
            lines.append(f"- `{item['file']}`: {item['reason']}")
        lines.append("")

    lines.append("## Reading the numbers")
    lines.append("")
    lines.append("- chrF++/BLEU: 0-100, higher is better; chrF++ is the headline metric.")
    lines.append("- Off-target: share of *non-blank* lines confidently detected in the "
                 "wrong language. Blank output is counted as blank, never as off-target.")
    lines.append("- FLORES rows are the comparable public benchmark; Tatoeba rows are "
                 "coverage-only and contamination-inflated for systems trained on it. "
                 "`trackb-*` rows are the fresh, post-cutoff harvest.")
    lines.append("- `directional (n<50)` rows (Manx) are reported for honesty, not ranking.")
    lines.append("- A `decoding deviation` note means the vendor refused part of the "
                 "common decoding contract; the receipt records exactly what was sent.")
    lines.append("")
    return "\n".join(lines)


def write(archive: bool = False) -> str:
    """Write LEADERBOARD.md; with archive=True also freeze a dated copy.

    A published edition must stay readable after the live board moves on, and
    a corrected number must be visibly a correction rather than a quiet edit.
    """
    payload = _load()
    content = render()
    path = os.path.join(HERE, "LEADERBOARD.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"wrote {path}")

    if archive:
        stamp = str(payload["generated_utc"]).replace(":", "").replace("-", "")
        archive_dir = os.path.join(HERE, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        board_copy = os.path.join(archive_dir, f"LEADERBOARD-{stamp}.md")
        with open(board_copy, "w", encoding="utf-8") as handle:
            handle.write(content)
        scores_copy = os.path.join(archive_dir, f"scores-{stamp}.json")
        shutil.copyfile(os.path.join(SCORES, "scores.json"), scores_copy)
        print(f"archived {board_copy} and {scores_copy}")
    return path
