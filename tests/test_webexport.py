"""The web data contract carries method decisions, not raw rows.

The website renders whatever this export says. Every test here guards a place
where a renderer would otherwise have to re-derive a rule and could disagree
with LEADERBOARD.md: edition completeness, off-target reliability, whether a
row may be ordered, and whether an absence is a gap or a finding.
"""
import json

import pytest

import celticbench.leaderboard as leaderboard
import celticbench.webexport as webexport

LID_REPORT = {
    "model": "lid.176.ftz",
    "model_sha256": "abc123def456",
    "advisory_false_positive_threshold": 0.05,
    "languages": {
        "gv": {"tatoeba": {"n": 18, "false_positive_rate": 0.278}},
        "ga": {"flores": {"n": 1012, "false_positive_rate": 0.0}},
    },
    "english": {
        "tatoeba": {"gv": {"n": 18, "false_positive_rate": 0.006}},
    },
}

SLICE = {
    "schema": "celticbench.trackb-slice.v1",
    "corpus": "trackb-2026q3",
    "slice": "2026q3",
    "cutoff": "2026-02-16",
    "harvest_date": "2026-07-29",
    "harvested_at": "2026-07-29T19:43:53+00:00",
    "extractor_version": "trackb-extract-1",
    "limit_per_lang": 300,
    "languages": {
        "cy": {
            "n": 300,
            "documents": 32,
            "language_name": "Welsh",
            "sources": ["gov-wales-announcements"],
            "rejected": {"duplicate": 5},
        },
    },
    "unavailable": [
        {"lang": "gv", "language_name": "Manx",
         "reason": "gov.im has no Manx language section to mirror."},
    ],
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
        "corpus": "flores",
        "lang": "ga",
        "language_name": "Irish",
        "direction": "en-xx",
        "n": 1012,
        "partial_slice": False,
        "fails": 0,
        "chrf_pp": 55.4,
        "bleu": 28.1,
        # Inside the published integrity floor, so tests about anything else
        # do not trip the gate by accident.
        "off_target_rate": 0.004,
        "off_target_n": 1012,
        "off_target_expected_lang": "ga",
        "blank_rate": 0.0,
        "copy_rate": 0.0,
        "harness_version": "abc1234",
    }
    row.update(overrides)
    return row


@pytest.fixture()
def export(tmp_path, monkeypatch):
    """Build the export against a scratch repo layout.

    `webexport` reuses `leaderboard._lid_reliability`, which reads HERE, so
    both modules are redirected: a test that only moved one would silently
    exercise the real manifests.
    """
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "lid-validation.json").write_text(json.dumps(LID_REPORT))
    (manifests / "trackb-2026q3.slice.json").write_text(json.dumps(SLICE))
    (tmp_path / "scores").mkdir()

    monkeypatch.setattr(leaderboard, "HERE", str(tmp_path))
    monkeypatch.setattr(webexport, "HERE", str(tmp_path))
    monkeypatch.setattr(webexport, "MANIFESTS", str(manifests))
    monkeypatch.setattr(webexport, "SCORES", str(tmp_path / "scores"))

    def build(**payload_overrides):
        payload = {
            "schema": "celticbench.scores.v2",
            "method_version": "v3",
            "generated_utc": "2026-07-29T22:43:37+00:00",
            "corpora": ["flores", "tatoeba", "trackb-2026q3"],
            "metric_signatures": {"chrf_pp": "chrF2++|nw:2", "bleu": "BLEU|13a"},
            "langid": {"model": "lid.176.ftz", "confidence_threshold": 0.5},
            "runtime": {"python": "3.14.3", "sacrebleu": "2.6.0"},
            "results": [_row()],
            "coverage": [],
            "excluded": [],
        }
        payload.update(payload_overrides)
        (tmp_path / "scores" / "scores.json").write_text(json.dumps(payload))
        return webexport.build()

    return build


# ── edition completeness ────────────────────────────────────────────────────
# A half-run edition that renders like a finished one publishes a ranking
# nobody checked. Status is derived from counted runs, never asserted.

def test_no_results_is_not_started(export):
    document = export(results=[])
    assert document["edition"]["status"] == "not-started"
    assert document["edition"]["runs_scored"] == 0


def test_partial_matrix_is_in_progress(export):
    document = export()
    assert document["edition"]["status"] == "in-progress"
    assert document["edition"]["runs_scored"] == 1
    assert document["edition"]["runs_expected"] > 1


def test_complete_matrix_is_complete(export, monkeypatch):
    monkeypatch.setattr(webexport, "matrix", lambda: [
        {"system": "gpt-5.6-sol", "corpus": "flores", "lang": "ga",
         "direction": "en-xx", "supported": True},
        {"system": "gpt-5.6-sol", "corpus": "flores", "lang": "ga",
         "direction": "xx-en", "supported": False},
    ])
    document = export()
    assert document["edition"]["status"] == "complete"
    # The unsupported combination is a coverage fact, not a missing run.
    assert document["edition"]["runs_expected"] == 1


# ── off-target reliability ──────────────────────────────────────────────────
# The site must never decide for itself whether a detector number can be
# trusted, and "nobody measured it" must not read as "it is fine".

def test_low_false_positive_rate_is_authoritative(export):
    document = export()
    assert document["results"][0]["off_target_reliability"] == "authoritative"


def test_high_false_positive_rate_is_advisory(export):
    document = export(results=[_row(
        corpus="tatoeba", lang="gv", n=18,
        off_target_expected_lang="gv", off_target_rate=0.33,
    )])
    assert document["results"][0]["off_target_reliability"] == "advisory"


def test_unmeasured_pair_is_labelled_unmeasured(export):
    document = export(results=[_row(corpus="tatoeba", lang="kw",
                                    off_target_expected_lang="kw")])
    assert document["results"][0]["off_target_reliability"] == "unmeasured"


def test_into_english_uses_the_english_side_rate(export):
    """An English hypothesis is judged against English.

    Labelling it with the Manx detector's error rate would advertise a caveat
    that does not apply to the measurement being reported.
    """
    document = export(results=[_row(
        corpus="tatoeba", lang="gv", direction="xx-en", n=18,
        off_target_expected_lang="en", off_target_rate=0.0,
    )])
    assert document["results"][0]["off_target_reliability"] == "authoritative"


def test_absent_off_target_rate_has_no_reliability_label(export):
    document = export(results=[_row(off_target_rate=None)])
    assert document["results"][0]["off_target_reliability"] is None


# ── rankability ─────────────────────────────────────────────────────────────

def test_small_n_row_is_published_but_not_rankable(export):
    document = export(results=[_row(lang="gv", corpus="tatoeba", n=18)])
    row = document["results"][0]
    assert row["rankable"] is False
    assert any("directional" in flag for flag in row["flags"])


def test_partial_slice_is_not_rankable(export):
    document = export(results=[_row(partial_slice=True)])
    assert document["results"][0]["rankable"] is False


def test_full_run_is_rankable(export):
    assert export()["results"][0]["rankable"] is True


def test_statistics_block_denies_confidence_intervals(export):
    """v3 has no CIs, so the contract says so rather than letting a
    renderer imply precision by ordering rows."""
    stats = export()["statistics"]
    assert stats["confidence_intervals"] is False
    assert stats["runs_per_cell"] == 1
    assert stats["significance_testing"] is False


# ── disclosures that must survive the export ────────────────────────────────

def test_decoding_deviation_reaches_the_panel(export):
    panel = {entry["id"]: entry for entry in export()["panel"]}
    deviation = panel["gemini-3.6-flash"]["decoding_deviation"]
    assert deviation is not None
    assert set(deviation["omits"]) == {"temperature", "top_p"}
    assert deviation["reason"]
    assert panel["claude-opus-5"]["decoding_deviation"] is None


def test_every_panel_system_carries_its_pin_and_licence(export):
    for entry in export()["panel"]:
        assert entry["pin_status"] in {"verified", "provisional", "alias"}
        assert entry["license"]
        assert entry["model"]


def test_prompt_template_is_published_verbatim(export):
    """A benchmark whose prompt is unknown is not a benchmark."""
    template = export()["provenance"]["prompt_template"]
    assert "{text}" in template
    assert "Output only" in template


def test_excluded_runs_are_carried_through(export):
    document = export(excluded=[{"file": "x.hyp", "reason": "prompt changed"}])
    assert document["excluded"][0]["reason"] == "prompt changed"


# ── Track B absences are findings ───────────────────────────────────────────

def test_unavailable_language_carries_its_reason(export):
    slice_record = export()["trackb"]["trackb-2026q3"]
    manx = slice_record["languages"]["gv"]
    assert manx["status"] == "unavailable"
    assert "gov.im" in manx["reason"]
    assert manx["n"] is None


def test_sealed_language_carries_provenance(export):
    welsh = export()["trackb"]["trackb-2026q3"]["languages"]["cy"]
    assert welsh["status"] == "sealed"
    assert welsh["n"] == 300
    assert welsh["sources"] == ["gov-wales-announcements"]


def test_slice_seal_timestamp_comes_from_the_manifest(export):
    """The seal time is the freshness claim; a null would erase it."""
    assert export()["trackb"]["trackb-2026q3"]["sealed_utc"] == "2026-07-29T19:43:53+00:00"


def test_language_table_surfaces_unavailability(export):
    langs = {entry["code"]: entry for entry in export()["languages"]}
    assert langs["gv"]["trackb"]["trackb-2026q3"]["status"] == "unavailable"
    assert langs["gv"]["has_flores"] is False
    assert langs["ga"]["has_flores"] is True


# ── corpus reading guidance ─────────────────────────────────────────────────

def test_every_corpus_declares_its_comparability(export):
    corpora = {entry["id"]: entry for entry in export()["corpora"]}
    assert corpora["flores"]["comparability"] == "comparable"
    assert corpora["tatoeba"]["comparability"] == "coverage-only"
    assert corpora["trackb-2026q3"]["comparability"] == "fresh"
    for entry in corpora.values():
        assert entry["note"]


def test_unknown_corpus_does_not_silently_pass_as_comparable(export):
    document = export(corpora=["flores", "mystery"])
    mystery = {entry["id"]: entry for entry in document["corpora"]}["mystery"]
    assert mystery["comparability"] == "unknown"


def test_trackb_corpus_carries_its_cutoff(export):
    corpora = {entry["id"]: entry for entry in export()["corpora"]}
    assert corpora["trackb-2026q3"]["cutoff"] == "2026-02-16"


# ── the whole document ──────────────────────────────────────────────────────

def test_export_is_json_serialisable_and_schema_stamped(export, tmp_path):
    document = export()
    assert document["schema"] == "celticbench.web.v1"
    target = tmp_path / "web.json"
    target.write_text(json.dumps(document, ensure_ascii=False))
    assert json.loads(target.read_text())["schema"] == "celticbench.web.v1"


def test_results_are_ordered_for_display(export):
    document = export(results=[
        _row(system="a", label="A", chrf_pp=40.0),
        _row(system="b", label="B", chrf_pp=60.0),
        _row(system="c", label="C", chrf_pp=50.0),
    ])
    scores = [row["chrf_pp"] for row in document["results"]]
    assert scores == sorted(scores, reverse=True)


# ── integrity gates the comparison ──────────────────────────────────────────
# A system that answered in English scores non-trivial overlap on an
# into-English direction and can outrank an honest attempt. The floor stops
# an overlap metric from crowning that row — but only where the measurement
# behind it is good enough to gate on.
#
# The default fixture row is FLORES Irish, where the detector's measured
# false-positive rate is 0.0%, so off-target is authoritative there.

def test_clean_row_clears_the_integrity_floor(export):
    row = export(results=[_row(off_target_rate=0.004)])["results"][0]
    assert row["integrity_ok"] is True
    assert row["integrity_failures"] == []


def test_off_target_above_floor_forfeits_the_win(export):
    result = export(results=[_row(off_target_rate=0.4)])["results"][0]
    assert result["integrity_ok"] is False
    assert "off_target_rate" in result["integrity_failures"]
    # Published, not hidden: the score is still there to be read.
    assert result["chrf_pp"] == 55.4


def test_off_target_does_not_gate_where_the_detector_is_advisory(export):
    """Tatoeba Manx: the detector misfires on 27.8% of gold Manx.

    Gating a 1% floor there would fail every system for the detector's errors
    rather than their own, which would quietly turn a measurement problem into
    a verdict about the models.
    """
    result = export(results=[_row(
        corpus="tatoeba", lang="gv", n=18,
        off_target_expected_lang="gv", off_target_rate=0.33,
    )])["results"][0]
    assert result["off_target_reliability"] == "advisory"
    assert result["integrity_failures"] == []
    assert result["integrity_ok"] is True


def test_blank_and_copy_gate_even_when_off_target_cannot(export):
    """Neither rate involves the detector, so detector reliability is
    irrelevant to them."""
    result = export(results=[_row(
        corpus="tatoeba", lang="gv", n=18,
        off_target_expected_lang="gv", off_target_rate=0.33,
        blank_rate=0.2,
    )])["results"][0]
    assert result["integrity_failures"] == ["blank_rate"]


def test_blank_and_copy_rates_are_gated(export):
    blank = export(results=[_row(blank_rate=0.2)])["results"][0]
    assert blank["integrity_failures"] == ["blank_rate"]
    copied = export(results=[_row(copy_rate=0.5)])["results"][0]
    assert copied["integrity_failures"] == ["copy_rate"]


def test_unmeasured_blank_rate_is_not_a_pass(export):
    """A metric nobody computed cannot demonstrate the row cleared the floor."""
    result = export(results=[_row(blank_rate=None)])["results"][0]
    assert result["integrity_ok"] is False
    assert result["integrity_failures"] == ["blank_rate unmeasured"]


def test_rate_exactly_at_the_threshold_passes(export):
    """The floor is a ceiling on the rate, not a strict inequality; a row
    sitting exactly on the published number is inside it."""
    result = export(results=[_row(off_target_rate=0.01)])["results"][0]
    assert result["integrity_ok"] is True


def test_integrity_floor_is_published_with_numbers(export):
    """An unstated good zone is decoration. A reader must be able to dispute
    the boundary, which means seeing it."""
    floor = export()["integrity_floor"]
    assert floor["off_target_rate"] == 0.01
    assert floor["blank_rate"] == 0.005
    assert floor["copy_rate"] == 0.02
    assert "lower-is-better" in floor["note"]
