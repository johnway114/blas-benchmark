"""Offline corpus provenance and QA.

Two halves. First, handcrafted eval files -> manifests/corpus-qa.json: every
defect the report claims to detect is planted in a tmp_path eval file, one
defect per language so the counts cannot alias each other. The band and length
thresholds are read from the module, not hardcoded, so tuning them does not
break these tests -- only a detector that stops detecting does.

Second, build_tatoeba against synthetic bz2 exports in the upstream format, to
prove the CC-BY attribution ids line up row for row with the .src/.ref files
the harness actually scores.
"""
import bz2
import json
import os

import pytest

import celticbench.corpus_qa as corpus_qa
import celticbench.lib as lib
import celticbench.prepare as prepare
from celticbench.lib import read_lines, write_lines

LOW, HIGH = corpus_qa.LENGTH_RATIO_BAND
MIN_SRC = corpus_qa.LENGTH_RATIO_MIN_SOURCE_CHARS


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "EVAL", str(tmp_path / "eval"))
    monkeypatch.setattr(lib, "MANIFESTS", str(tmp_path / "manifests"))
    return tmp_path


def _pair(corpus, lang, src, ref, ids=None):
    write_lines(lib.eval_src_path(corpus, lang), src)
    write_lines(lib.eval_ref_path(corpus, lang), ref)
    if ids is not None:
        write_lines(lib.eval_ids_path(corpus, lang), ids)


def _text(n):
    """Exactly n visible characters, no leading or trailing whitespace."""
    filler = ("mor " * (n // 4 + 2))[:n]
    return filler[:-1] + "a" if filler.endswith(" ") else filler


def test_clean_pair_reports_no_defects(workspace):
    _pair("tatoeba", "kw",
          ["Good morning.", "The sea is cold today.", "I saw three birds."],
          ["Myttin da.", "Yeyn yw an mor hedhyw.", "My a welas teyr edhen."])
    entry = corpus_qa.build_corpus_qa()["corpora"]["tatoeba"]["kw"]
    assert entry["n"] == 3
    assert entry["untranslated_rows"] == 0
    assert entry["blank_pair_rows"] == 0
    for side in ("src", "ref"):
        assert entry[side] == {
            "duplicate_rows": 0, "duplicate_groups": 0, "blank_rows": 0,
            "control_char_rows": 0, "non_nfc_rows": 0,
        }


def test_duplicate_rows_counted_per_side(workspace):
    # Two English rows collapse onto one Welsh reference: three redundant src
    # rows in two groups, one redundant ref row.
    _pair("tatoeba", "cy",
          ["Hello.", "Hello.", "Hello.", "Goodbye.", "Goodbye.", "Only once."],
          ["Helo.", "Sut mae.", "Bore da.", "Hwyl.", "Hwyl fawr.", "Hwyl am nawr."])
    entry = corpus_qa.build_corpus_qa()["corpora"]["tatoeba"]["cy"]
    assert entry["src"]["duplicate_rows"] == 3
    assert entry["src"]["duplicate_groups"] == 2
    assert entry["ref"]["duplicate_rows"] == 0

    _pair("tatoeba", "gd", ["One.", "Two.", "Three."], ["Aon.", "Aon.", "Aon."])
    entry = corpus_qa.build_corpus_qa()["corpora"]["tatoeba"]["gd"]
    assert entry["src"]["duplicate_rows"] == 0
    assert entry["ref"]["duplicate_rows"] == 2
    assert entry["ref"]["duplicate_groups"] == 1


def test_untranslated_pairs_detected(workspace):
    _pair("tatoeba", "gv",
          ["Tom!", "Kernow  a  Gas", "The sea is cold.", "Moghrey mie."],
          ["tom!", "kernow a gas", "Ta'n keayn feayr.", "Moghrey mie."])
    entry = corpus_qa.build_corpus_qa()["corpora"]["tatoeba"]["gv"]
    # rows 0 (case), 1 (whitespace) and 3 (verbatim) are copy-throughs; row 2
    # is a real translation.
    assert entry["untranslated_rows"] == 3


def test_control_characters_detected(workspace):
    _pair("tatoeba", "kw",
          ["The AI \u200bsystem is used widely.", "Clean English row."],
          ["Myttin\tda.", "Lin glan."])
    entry = corpus_qa.build_corpus_qa()["corpora"]["tatoeba"]["kw"]
    assert entry["src"]["control_char_rows"] == 1
    assert entry["ref"]["control_char_rows"] == 1


def test_non_nfc_rows_detected(workspace):
    _pair("tatoeba", "ga",
          ["Se\u0301an went home.", "Nothing odd here."],
          ["Chuaigh Séan abhaile.", "Rud ar bith aisteach."])
    entry = corpus_qa.build_corpus_qa()["corpora"]["tatoeba"]["ga"]
    assert entry["src"]["non_nfc_rows"] == 1
    assert entry["ref"]["non_nfc_rows"] == 0


def test_blank_rows_detected_and_excluded_from_ratios(workspace):
    _pair("tatoeba", "br",
          ["Real row.", "", "Source only.", "   ", "Another real row."],
          ["Linenn wir.", "Linenn goll.", "", "\t", "Linenn wir all."])
    entry = corpus_qa.build_corpus_qa()["corpora"]["tatoeba"]["br"]
    assert entry["src"]["blank_rows"] == 2
    assert entry["ref"]["blank_rows"] == 2
    # one blank source, one blank reference, one row blank on both sides
    assert entry["blank_pair_rows"] == 3
    assert entry["length_ratio"]["measured_rows"] == 2


def test_length_ratio_distribution_and_band(workspace):
    inside = (_text(MIN_SRC + 10), _text(MIN_SRC + 10))
    above = (_text(MIN_SRC), _text(int(MIN_SRC * HIGH) + 5))
    below = (_text(MIN_SRC * 2), _text(max(1, int(MIN_SRC * 2 * LOW) - 2)))
    short = (_text(MIN_SRC - 1), _text(int((MIN_SRC - 1) * HIGH) + 10))
    rows = [inside, above, below, short]
    _pair("tatoeba", "kw", [s for s, _ in rows], [r for _, r in rows])

    ratio = corpus_qa.build_corpus_qa()["corpora"]["tatoeba"]["kw"]["length_ratio"]
    assert ratio["measured_rows"] == 4
    # the short row is measured but exempt from the band: one bad word swing
    # cannot make a five-character sentence a corpus defect
    assert ratio["band_measured_rows"] == 3
    assert ratio["outside_band"] == 2
    observed = sorted(round(len(r) / len(s), 4) for s, r in rows)
    assert ratio["min"] == observed[0]
    assert ratio["max"] == observed[-1]
    # nearest-rank percentiles are always ratios that exist in the corpus
    for key in ("p01", "median", "p99"):
        assert ratio[key] in observed


def test_ids_file_reported_only_when_present(workspace):
    _pair("tatoeba", "kw", ["A row.", "B row."], ["Lin A.", "Lin B."],
          ids=["12345\t67890", "12346\t67891"])
    _pair("flores", "ga", ["A row.", "B row."], ["Lin A.", "Lin B."])
    corpora = corpus_qa.build_corpus_qa()["corpora"]
    assert corpora["tatoeba"]["kw"]["ids_rows"] == 2
    assert corpora["flores"]["ga"]["ids_rows"] is None


def test_track_b_slice_is_reported(workspace):
    _pair("trackb-2026q3", "gd", ["Fresh row."], ["Sreath ùr."])
    lib.write_manifest("trackb-2026q3", "gd", 1, {"dataset": "test-fixture"})
    corpora = corpus_qa.build_corpus_qa()["corpora"]
    assert corpora["trackb-2026q3"]["gd"]["n"] == 1


def test_skewed_pair_is_fatal(workspace):
    _pair("tatoeba", "kw", ["One.", "Two.", "Three."], ["Onan.", "Dew."])
    with pytest.raises(ValueError, match="not parallel"):
        corpus_qa.build_corpus_qa()


def test_missing_reference_side_is_fatal(workspace):
    write_lines(lib.eval_src_path("tatoeba", "kw"), ["One."])
    with pytest.raises(FileNotFoundError):
        corpus_qa.build_corpus_qa()


def test_report_bytes_identical_across_runs(workspace):
    _pair("tatoeba", "kw",
          ["Good morning.", "Good morning.", "", _text(MIN_SRC * 2)],
          ["Myttin da.", "myttin  da.", "Nyns eus.", _text(MIN_SRC)])
    _pair("flores", "cy", ["A row.", "B row."], ["Lin A.", "Lin\u200b B."])

    corpus_qa.build_corpus_qa()
    with open(corpus_qa.report_path(), "rb") as handle:
        first = handle.read()
    corpus_qa.build_corpus_qa()
    with open(corpus_qa.report_path(), "rb") as handle:
        second = handle.read()
    assert first == second

    payload = json.loads(first)
    assert payload["schema"] == lib.SCHEMA_QA
    assert set(payload["corpora"]) == {"flores", "tatoeba"}
    assert len(corpus_qa.summary_lines(payload)) == 2


ENG_SENTENCES = [("101", "Good morning."), ("102", "The sea is cold today."),
                 ("103", "Hello."), ("104", "Thank you.")]

# (native_id, text) and (native_id, eng_id) in the upstream export shapes:
# sentences are "Sentence id [tab] Lang [tab] Text", links are
# "Sentence id [tab] Translation id" -- https://tatoeba.org/en/downloads
CORNISH_SENTENCES = [("203", "Dhe'n gwariva."), ("201", "Myttin da."),
                     ("202", "Yeyn yw an mor hedhyw.")]
CORNISH_LINKS = [("202", "102"), ("201", "101"), ("203", "101")]


def _write_bz2(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with bz2.open(path, "wt", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")


@pytest.fixture()
def tatoeba_exports(tmp_path, monkeypatch):
    """Synthetic per-language exports; no network, no real archives."""
    monkeypatch.setattr(lib, "EVAL", str(tmp_path / "eval"))
    monkeypatch.setattr(lib, "MANIFESTS", str(tmp_path / "manifests"))
    monkeypatch.setattr(prepare, "DATA", str(tmp_path / "data"))
    monkeypatch.setattr(prepare, "_download", lambda url, dest: None)

    _write_bz2(prepare._tatoeba_file("eng_sentences.tsv.bz2"),
               [(sid, "eng", text) for sid, text in ENG_SENTENCES])
    for index, (lang, meta) in enumerate(lib.LANGS.items()):
        iso3 = meta["tatoeba"]
        if lang == "kw":
            sentences, links = CORNISH_SENTENCES, CORNISH_LINKS
        else:
            # upstream sentence ids are integers; build_tatoeba sorts on them
            sentences = [(str(300 + index), f"{lang} greeting")]
            links = [(str(300 + index), "103")]
        _write_bz2(prepare._tatoeba_file(f"{iso3}_sentences.tsv.bz2"),
                   [(sid, iso3, text) for sid, text in sentences])
        _write_bz2(prepare._tatoeba_file(f"{iso3}-eng_links.tsv.bz2"), links)
    return tmp_path


def test_tatoeba_ids_line_up_with_eval_rows(tatoeba_exports):
    prepare.build_tatoeba()
    src = read_lines(lib.eval_src_path("tatoeba", "kw"))
    ref = read_lines(lib.eval_ref_path("tatoeba", "kw"))
    ids = read_lines(lib.eval_ids_path("tatoeba", "kw"))

    # ordered by native sentence id, deduplicated on the English side: 203 is
    # dropped because its English side (101) already came through with 201
    assert src == ["Good morning.", "The sea is cold today."]
    assert ref == ["Myttin da.", "Yeyn yw an mor hedhyw."]
    assert ids == ["101\t201", "102\t202"]

    eng_text = dict(ENG_SENTENCES)
    native_text = dict(CORNISH_SENTENCES)
    for row, id_line in enumerate(ids):
        eng_id, native_id = id_line.split("\t")
        assert eng_text[eng_id] == src[row]
        assert native_text[native_id] == ref[row]


def test_tatoeba_manifest_pins_the_ids_file(tatoeba_exports):
    prepare.build_tatoeba()
    manifest = lib.load_manifest("tatoeba", "kw")
    assert manifest["ids_sha256"] == lib.file_sha256(lib.eval_ids_path("tatoeba", "kw"))
    # fail-closed verification now covers the attribution file too
    lib.verify_manifest("tatoeba", "kw")
    write_lines(lib.eval_ids_path("tatoeba", "kw"), ["999\t999", "999\t999"])
    with pytest.raises(ValueError, match="tatoeba.kw.ids"):
        lib.verify_manifest("tatoeba", "kw")
