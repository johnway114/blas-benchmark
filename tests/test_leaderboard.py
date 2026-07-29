"""Leaderboard rendering: the caveats must survive the trip to markdown."""
import json

import pytest

import celticbench.leaderboard as leaderboard

LID_REPORT = {
    "model": "lid.176.ftz",
    "model_sha256": "abc123",
    "advisory_false_positive_threshold": 0.05,
    # The detector barely recognises gold Manx, but is near-perfect on the
    # English side of the same pairs.
    "languages": {"gv": {"tatoeba": {"n": 18, "recognized": 0.06, "false_positive_rate": 0.278}}},
    "english": {"tatoeba": {"gv": {"n": 18, "recognized": 0.99, "false_positive_rate": 0.006}}},
}


def _row(**overrides):
    row = {
        "system": "opus-mt-cel",
        "label": "Opus-MT en-cel / cel-en (anchor)",
        "vendor": "Helsinki-NLP",
        "tier": "open-anchor",
        "license": "Apache-2.0",
        "benchmark_only": False,
        "pin_status": "verified",
        "model_reported": "Helsinki-NLP/opus-mt-en-cel@e794385",
        "corpus": "tatoeba",
        "lang": "gv",
        "language_name": "Manx",
        "direction": "en-xx",
        "n": 18,
        "partial_slice": False,
        "fails": 0,
        "chrf_pp": 40.9,
        "bleu": 14.1,
        "off_target_rate": 0.333,
        "off_target_expected_lang": "gv",
        "blank_rate": 0.0,
        "copy_rate": 0.0,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(leaderboard, "HERE", str(tmp_path))
    monkeypatch.setattr(leaderboard, "SCORES", str(tmp_path / "scores"))
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "lid-validation.json").write_text(json.dumps(LID_REPORT))
    (tmp_path / "scores").mkdir()

    def write_scores(**payload_overrides):
        payload = {
            "generated_utc": "2026-07-29T18:00:00+00:00",
            "metric_signatures": {"chrf_pp": "chrF2++|version:2.6.0", "bleu": "BLEU|version:2.6.0"},
            "langid": {"model": "lid.176.ftz", "model_sha256": "abc123def456",
                       "confidence_threshold": 0.5},
            "runtime": {"python": "3.14.3"},
            "results": [_row()],
            "coverage": [],
            "excluded": [],
        }
        payload.update(payload_overrides)
        (tmp_path / "scores" / "scores.json").write_text(json.dumps(payload))
        return payload

    return write_scores


def test_advisory_follows_the_language_being_detected(board):
    board(results=[
        _row(),
        _row(direction="xx-en", off_target_expected_lang="en", off_target_rate=0.0),
    ])
    rendered = leaderboard.render()
    into_manx, into_english = [
        line for line in rendered.splitlines() if line.startswith("| Opus-MT")
    ]
    # Manx: the detector's own false-positive floor is 27.8%, so the rate is advisory.
    assert "33.3% (advisory)" in into_manx
    # English: measured at 0.6% on the same pairs, so it is authoritative and
    # must not inherit the Manx caveat.
    assert "0.0%" in into_english and "advisory" not in into_english


def test_unmeasured_corpus_is_not_passed_off_as_reliable(board):
    """A sealed Track B slice nobody has measured the detector on says so."""
    board(results=[_row(corpus="trackb-2026q3", off_target_rate=0.04)])
    row_line = next(line for line in leaderboard.render().splitlines()
                    if line.startswith("| Opus-MT"))
    assert "4.0% (unmeasured)" in row_line


def test_detector_and_metric_provenance_reach_the_page(board):
    board()
    rendered = leaderboard.render()
    assert "chrF2++|version:2.6.0" in rendered
    assert "abc123def456"[:12] in rendered
    assert "python 3.14.3" in rendered


def test_coverage_distinguishes_unsupported_from_unrun(board):
    board(coverage=[
        {"system": "deepl", "lang": "ga", "direction": "en-xx", "supported": True, "reason": None},
        {"system": "deepl", "lang": "gv", "direction": "en-xx", "supported": False,
         "reason": "DeepL lists no Manx"},
        {"system": "opus-mt-cel", "lang": "gv", "direction": "en-xx", "supported": True,
         "reason": None},
    ])
    rendered = leaderboard.render()
    deepl_row = next(line for line in rendered.splitlines() if line.startswith("| deepl |"))
    cells = [cell.strip() for cell in deepl_row.strip("|").split("|")]
    header = ["system"] + list(leaderboard.LANGS)
    assert cells[header.index("ga")] == ".", "supported but not run"
    assert cells[header.index("gv")] == "n/a", "a coverage fact, not a score of zero"
    anchor_row = next(line for line in rendered.splitlines() if line.startswith("| opus-mt-cel |"))
    anchor_cells = [cell.strip() for cell in anchor_row.strip("|").split("|")]
    assert anchor_cells[header.index("gv")] == "ok", "scored"


def test_caveat_flags_are_printed(board):
    board(results=[_row(
        benchmark_only=True,
        pin_status="provisional",
        partial_slice=True,
        fails=3,
        model_reported="mixed",
        reasoning="always-on reasoning; cannot be disabled",
        decoding_deviations=["omitted temperature, top_p: vendor deprecated them"],
    )])
    row_line = next(line for line in leaderboard.render().splitlines()
                    if line.startswith("| Opus-MT"))
    for expected in ("NC licence", "pin: provisional", "partial slice", "3 failed lines",
                     "more than one model ID", "reasoning model", "decoding deviation",
                     "directional (n=18)"):
        assert expected in row_line, expected


def test_excluded_runs_are_published_with_their_reasons(board):
    board(excluded=[{"file": "tatoeba.gv.en-xx.mystery.hyp", "reason": "no receipt"}])
    rendered = leaderboard.render()
    assert "## Excluded runs" in rendered
    assert "tatoeba.gv.en-xx.mystery.hyp`: no receipt" in rendered


def test_archive_freezes_the_board_and_the_scores_together(board, tmp_path):
    """A published edition has to stay readable after the live board moves on."""
    board()
    leaderboard.write(archive=True)
    frozen = sorted(p.name for p in (tmp_path / "archive").iterdir())
    assert frozen == ["LEADERBOARD-20260729T180000+0000.md",
                      "scores-20260729T180000+0000.json"]
    # The frozen board must be the board, not a re-render of whatever comes next.
    assert (tmp_path / "archive" / frozen[0]).read_text() == (tmp_path / "LEADERBOARD.md").read_text()
