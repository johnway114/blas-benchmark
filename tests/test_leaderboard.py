"""Leaderboard rendering preserves v3 provenance, caveats, and coverage."""
import json

import pytest

import celticbench.leaderboard as leaderboard

SYSTEM_IDS = (
    "gpt-5.6-sol",
    "claude-opus-5",
    "gemini-3.6-flash",
    "deepseek-v4-pro",
    "kimi-k3",
    "qwen3.7-max",
)
LID_REPORT = {
    "model": "lid.176.ftz",
    "model_sha256": "abc123",
    "advisory_false_positive_threshold": 0.05,
    "languages": {
        "gv": {
            "tatoeba": {
                "n": 18,
                "recognized": 0.06,
                "false_positive_rate": 0.278,
            },
        },
    },
    "english": {
        "tatoeba": {
            "gv": {
                "n": 18,
                "recognized": 0.99,
                "false_positive_rate": 0.006,
            },
        },
    },
}


def _row(**overrides):
    row = {
        "system": "gpt-5.6-sol",
        "label": "GPT-5.6 Sol",
        "vendor": "OpenAI",
        "tier": "flagship-chat",
        "license": "proprietary API",
        "benchmark_only": False,
        "pin_status": "verified",
        "model_reported": "gpt-5.6-sol-2026-07-01",
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


def _coverage():
    return [
        {
            "system": system_id,
            "lang": lang,
            "direction": direction,
            "supported": True,
            "reason": None,
        }
        for system_id in SYSTEM_IDS
        for lang in leaderboard.LANGS
        for direction in ("en-xx", "xx-en")
    ]


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(leaderboard, "HERE", str(tmp_path))
    monkeypatch.setattr(leaderboard, "SCORES", str(tmp_path / "scores"))
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "lid-validation.json").write_text(json.dumps(LID_REPORT))
    (tmp_path / "scores").mkdir()

    def write_scores(**payload_overrides):
        payload = {
            "method_version": "v3",
            "generated_utc": "2026-07-29T18:00:00+00:00",
            "metric_signatures": {
                "chrf_pp": "chrF2++|version:2.6.0",
                "bleu": "BLEU|version:2.6.0",
            },
            "langid": {
                "model": "lid.176.ftz",
                "model_sha256": "abc123def456",
                "confidence_threshold": 0.5,
            },
            "runtime": {"python": "3.14.3"},
            "results": [_row()],
            "coverage": _coverage(),
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
        line for line in rendered.splitlines() if line.startswith("| GPT-5.6")
    ]
    assert "33.3% (advisory)" in into_manx
    assert "0.0%" in into_english and "advisory" not in into_english


def test_unmeasured_corpus_is_not_passed_off_as_reliable(board):
    board(results=[_row(corpus="trackb-2026q3", off_target_rate=0.04)])
    row_line = next(
        line for line in leaderboard.render().splitlines()
        if line.startswith("| GPT-5.6")
    )
    assert "4.0% (unmeasured)" in row_line


def test_v3_metric_detector_and_runtime_provenance_reach_the_page(board):
    board()
    rendered = leaderboard.render()
    assert "Method **v3**" in rendered
    assert "chrF2++|version:2.6.0" in rendered
    assert "abc123def456"[:12] in rendered
    assert "python 3.14.3" in rendered


def test_coverage_has_six_frontier_rows_per_direction_and_preserves_the_split(board):
    board()
    rendered = leaderboard.render()
    coverage_rows = [
        line for line in rendered.splitlines()
        if any(line.startswith(f"| {system_id} |") for system_id in SYSTEM_IDS)
    ]
    assert len(coverage_rows) == 12
    assert {line.split("|")[1].strip() for line in coverage_rows} == set(SYSTEM_IDS)

    gpt_rows = [line for line in coverage_rows if line.startswith("| gpt-5.6-sol |")]
    assert len(gpt_rows) == 2
    header = ["system", *leaderboard.LANGS]
    into_manx = [cell.strip() for cell in gpt_rows[0].strip("|").split("|")]
    into_english = [cell.strip() for cell in gpt_rows[1].strip("|").split("|")]
    assert into_manx[header.index("gv")] == "ok"
    assert into_english[header.index("gv")] == "."
    assert "n/a" not in gpt_rows[0] and "n/a" not in gpt_rows[1]


def test_frontier_caveat_flags_are_printed(board):
    board(results=[_row(
        pin_status="provisional",
        partial_slice=True,
        fails=3,
        model_reported="mixed",
        reasoning="always-on reasoning; cannot be disabled",
        decoding_deviations=["omitted temperature, top_p: vendor deprecated them"],
    )])
    row_line = next(
        line for line in leaderboard.render().splitlines()
        if line.startswith("| GPT-5.6")
    )
    for expected in (
        "pin: provisional",
        "partial slice",
        "3 failed lines",
        "more than one model ID",
        "reasoning model",
        "decoding deviation",
        "directional (n=18)",
    ):
        assert expected in row_line


def test_excluded_runs_are_published_with_their_reasons(board):
    board(excluded=[{
        "file": "tatoeba.gv.en-xx.gpt-5.6-sol.hyp",
        "reason": "run was made under method 'v2'",
    }])
    rendered = leaderboard.render()
    assert "## Excluded runs" in rendered
    assert "gpt-5.6-sol.hyp`: run was made under method 'v2'" in rendered


def test_archive_freezes_the_board_and_scores_together(board, tmp_path):
    board()
    leaderboard.write(archive=True)
    frozen = sorted(path.name for path in (tmp_path / "archive").iterdir())
    assert frozen == [
        "LEADERBOARD-20260729T180000+0000.md",
        "scores-20260729T180000+0000.json",
    ]
    assert (tmp_path / "archive" / frozen[0]).read_text() == (
        tmp_path / "LEADERBOARD.md"
    ).read_text()
