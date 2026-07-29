"""Render scores/scores.json into LEADERBOARD.md."""
from __future__ import annotations

import json
import os
from typing import Any

from .lib import HERE, LANGS, SCORES

SMALL_N = 50  # below this a score is directional, not comparable


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
    return "; ".join(flags)


def render() -> str:
    payload = _load()
    lid_note = _lid_reliability()
    lines: list[str] = []
    lines.append("# Celtic Translation Benchmark — leaderboard")
    lines.append("")
    lines.append(f"Generated from `scores/scores.json` ({payload['generated_utc']}). "
                 "Every row is backed by a verified receipt in `out/`; excluded runs "
                 "are listed at the bottom with reasons. Method: `METHODOLOGY.md`.")
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
            lines.append("| System | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            for row in sorted(rows, key=lambda r: (r["corpus"], -r["chrf_pp"])):
                off = row.get("off_target_rate")
                off_str = f"{off:.1%}" if off is not None else "-"
                if off is not None and lid_note.get((lang, row["corpus"])):
                    off_str += " (advisory)"
                lines.append(
                    f"| {row['label']} | {row['corpus']} | {row['n']} | {row['chrf_pp']:.1f} "
                    f"| {row['bleu']:.1f} | {off_str} | {row.get('blank_rate', 0):.1%} "
                    f"| {row.get('copy_rate', 0):.1%} | {_flags(row)} |"
                )
            lines.append("")

    if payload.get("excluded"):
        lines.append("## Excluded runs (fail-closed)")
        lines.append("")
        for item in payload["excluded"]:
            lines.append(f"- `{item['file']}`: {item['reason']}")
        lines.append("")

    lines.append("## Reading the numbers")
    lines.append("")
    lines.append("- chrF++/BLEU: 0-100, higher is better; chrF++ is the headline metric.")
    lines.append("- Off-target: share of lines confidently detected in the wrong language.")
    lines.append("- FLORES rows are the comparable public benchmark; Tatoeba rows are "
                 "coverage-only and contamination-inflated for systems trained on it.")
    lines.append("- `directional (n<50)` rows (Manx) are reported for honesty, not ranking.")
    lines.append("")
    return "\n".join(lines)


def _lid_reliability() -> dict[tuple[str, str], bool]:
    """(lang, corpus) -> True when off-target is advisory there.

    Advisory means the detector confidently mislabels more than the allowed
    share of gold reference lines for that language/corpus, so a system's
    off-target rate has a measurable false-positive floor. The rates and
    threshold come from manifests/lid-validation.json (built by prepare).
    """
    path = os.path.join(HERE, "manifests", "lid-validation.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    threshold = float(report.get("advisory_false_positive_threshold", 0.05))
    advisory: dict[tuple[str, str], bool] = {}
    for lang, by_corpus in report.get("languages", {}).items():
        for corpus, stats in by_corpus.items():
            advisory[(lang, corpus)] = stats.get("false_positive_rate", 1.0) > threshold
    return advisory


def write() -> str:
    content = render()
    path = os.path.join(HERE, "LEADERBOARD.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"wrote {path}")
    return path
