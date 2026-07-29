"""Track B harvest: freshness, conservative alignment, honest gaps.

Everything here runs offline. Sources reach the network only through the
`fetch` argument, so a dict of routes drives the real registered adapters as
well as the generic pipeline.
"""
import json
import os
import urllib.parse

import pytest

import celticbench.lib as lib
import celticbench.trackb as trackb
from celticbench.trackb import Candidate, Source


class StubFetch:
    """Resolves a URL by first matching substring; unknown URLs are unreachable."""

    def __init__(self, routes):
        self.routes = list(routes)
        self.seen = []

    def __call__(self, url):
        self.seen.append(url)
        for needle, body in self.routes:
            if needle in url:
                return body
        return None


@pytest.fixture()
def slice_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "EVAL", str(tmp_path / "eval"))
    monkeypatch.setattr(lib, "MANIFESTS", str(tmp_path / "manifests"))
    return tmp_path


def fake_source(candidates, source_id="stub", pairs_by_url=None):
    """A source with no network: candidates and their blocks are given inline."""
    def discover(_fetch, _cutoff, _harvest_date):
        yield from candidates

    def extract(_fetch, candidate):
        if pairs_by_url is not None:
            return list(pairs_by_url.get(candidate.url, ()))
        return list(candidate.payload or ())

    return Source(id=source_id, publisher="Stub Publisher", home="https://stub.example",
                  licence="stub", note="stub", discover=discover, extract=extract)


def en(i, extra=""):
    return f"Segment {i}: the committee published its findings this morning.{extra}"


def cy(i, extra=""):
    return f"Segment {i}: cyhoeddodd y pwyllgor ei chanfyddiadau y bore yma.{extra}"


def harvest(fetch, sources, **kwargs):
    kwargs.setdefault("cutoff", "2026-07-20")
    kwargs.setdefault("harvest_date", "2026-07-29")
    kwargs.setdefault("langs", ("cy",))
    kwargs.setdefault("limit_per_lang", 100)
    return trackb.harvest_slice("t1", fetch=fetch, sources=sources, **kwargs)


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------

def test_cutoff_drops_undated_and_too_old(slice_dirs):
    candidates = [
        Candidate("https://stub.example/fresh", "2026-07-25", [(en(1), cy(1))]),
        Candidate("https://stub.example/on-the-cutoff", "2026-07-20", [(en(2), cy(2))]),
        Candidate("https://stub.example/stale", "2026-01-04", [(en(3), cy(3))]),
        Candidate("https://stub.example/undated", "", [(en(4), cy(4))]),
    ]
    record = harvest(StubFetch([]), {"cy": (fake_source(candidates),)})
    row = record["languages"]["cy"]
    assert row["n"] == 1
    assert row["rejected"] == {"not_after_cutoff": 2, "undated": 1}
    src = lib.read_lines(lib.eval_src_path("trackb-t1", "cy"))
    assert src == [en(1)]


def test_cutoff_must_precede_harvest_date(slice_dirs):
    with pytest.raises(ValueError, match="not after cutoff"):
        harvest(StubFetch([]), {"cy": ()}, cutoff="2026-07-29", harvest_date="2026-07-20")


def test_slice_id_and_dates_are_validated(slice_dirs):
    with pytest.raises(ValueError, match="lowercase alphanumeric"):
        trackb.harvest_slice("2026 Q3", "2026-07-20", fetch=StubFetch([]))
    with pytest.raises(ValueError, match="cutoff must be an ISO date"):
        trackb.harvest_slice("t1", "last quarter", fetch=StubFetch([]))


# ---------------------------------------------------------------------------
# Conservative extraction
# ---------------------------------------------------------------------------

def test_identical_sides_are_rejected_as_untranslated(slice_dirs):
    same = en(1)
    candidates = [
        Candidate("https://stub.example/fallback", "2026-07-25",
                  [(same, same), (same, "  " + same.upper() + " "), (en(2), cy(2))]),
    ]
    record = harvest(StubFetch([]), {"cy": (fake_source(candidates),)})
    row = record["languages"]["cy"]
    assert row["n"] == 1
    assert row["rejected"]["identical"] == 2
    assert lib.read_lines(lib.eval_ref_path("trackb-t1", "cy")) == [cy(2)]


def test_length_ratio_and_size_outliers_are_rejected(slice_dirs):
    english = en(1)
    candidates = [
        Candidate("https://stub.example/a", "2026-07-25", [
            (english, "byr"),                        # far too short a translation
            (english, "gair " * 400),                # runaway duplication
            ("Too brief.", "Rhy fyr."),              # below MIN_CHARS
            ("A" * (trackb.MAX_CHARS + 1), "B" * (trackb.MAX_CHARS + 1)),
            (english, ""),                           # missing side
            (en(9), cy(9)),
        ]),
    ]
    record = harvest(StubFetch([]), {"cy": (fake_source(candidates),)})
    row = record["languages"]["cy"]
    assert row["n"] == 1
    assert row["rejected"] == {"empty": 1, "length_ratio": 2, "too_long": 1, "too_short": 1}


def test_reject_reason_boundaries():
    english = "x" * trackb.MIN_CHARS
    assert trackb.reject_reason(english, "y" * trackb.MIN_CHARS) is None
    assert trackb.reject_reason("x" * (trackb.MIN_CHARS - 1), "y" * 60) == "too_short"
    assert trackb.reject_reason(english, "y" * int(trackb.MIN_CHARS * trackb.MIN_RATIO)) is None
    assert trackb.reject_reason(english, "y" * (int(trackb.MIN_CHARS * trackb.MIN_RATIO) - 1)) == "length_ratio"
    assert trackb.reject_reason(english, "y" * int(trackb.MIN_CHARS * trackb.MAX_RATIO)) is None
    assert trackb.reject_reason(english, "y" * (int(trackb.MIN_CHARS * trackb.MAX_RATIO) + 1)) == "length_ratio"


def test_align_needs_identical_block_shape():
    left = [("p", "one"), ("h2", "two")]
    assert trackb.align(left, [("p", "un"), ("h2", "dau")]) == [("one", "un"), ("two", "dau")]
    assert trackb.align(left, [("p", "un"), ("p", "dau")]) == []       # tag sequence differs
    assert trackb.align(left, [("p", "un")]) == []                     # length differs
    assert trackb.align([], []) == []                                  # nothing is not a pair


def test_div_slice_tracks_nesting():
    document = ('<div class="chrome"><p>before</p></div>'
                '<div class="body" id="x"><div><p>inner</p></div><p>tail</p></div>'
                '<div class="chrome"><p>after</p></div>')
    inner = trackb._div_slice(document, '<div class="body"')
    assert trackb.blocks(inner, ("p",)) == [("p", "inner"), ("p", "tail")]
    assert trackb._div_slice(document, '<div class="absent"') is None


# ---------------------------------------------------------------------------
# Dedup, cap, determinism
# ---------------------------------------------------------------------------

def test_dedup_is_on_the_english_side(slice_dirs):
    candidates = [
        Candidate("https://stub.example/a", "2026-07-25", [(en(1), cy(1))]),
        Candidate("https://stub.example/b", "2026-07-24",
                  [("  " + en(1).upper() + "  ", cy(2)), (en(3), cy(3))]),
    ]
    record = harvest(StubFetch([]), {"cy": (fake_source(candidates),)})
    row = record["languages"]["cy"]
    assert row["n"] == 2
    assert row["rejected"]["duplicate"] == 1
    assert lib.read_lines(lib.eval_ref_path("trackb-t1", "cy")) == [cy(1), cy(3)]


def test_cap_takes_the_first_n_and_stops_fetching(slice_dirs):
    fetch = StubFetch([])
    first = fake_source([Candidate("https://stub.example/a", "2026-07-25",
                                   [(en(i), cy(i)) for i in range(5)])], source_id="first")
    second = fake_source([Candidate("https://stub.example/b", "2026-07-24",
                                    [(en(9), cy(9))])], source_id="second")
    record = harvest(fetch, {"cy": (first, second)}, limit_per_lang=3)
    assert lib.read_lines(lib.eval_src_path("trackb-t1", "cy")) == [en(0), en(1), en(2)]
    assert record["languages"]["cy"]["sources"] == ["first"]


def test_rebuild_reproduces_the_same_manifest(slice_dirs):
    sources = {"cy": (fake_source([
        Candidate("https://stub.example/a", "2026-07-25", [(en(i), cy(i)) for i in range(4)]),
    ]),)}
    first = harvest(StubFetch([]), sources)
    again = harvest(StubFetch([]), sources)
    assert first["languages"]["cy"]["contract_sha256"] == again["languages"]["cy"]["contract_sha256"]
    assert lib.verify_manifest("trackb-t1", "cy")["n"] == 4


# ---------------------------------------------------------------------------
# Eval file contract
# ---------------------------------------------------------------------------

def test_ids_stay_parallel_and_carry_url_and_date(slice_dirs):
    candidates = [
        Candidate("https://stub.example/one", "2026-07-25", [(en(1), cy(1)), (en(2), cy(2))]),
        Candidate("https://stub.example/two", "2026-07-24", [(en(3), cy(3))]),
    ]
    record = harvest(StubFetch([]), {"cy": (fake_source(candidates),)})
    corpus = record["corpus"]
    src = lib.read_lines(lib.eval_src_path(corpus, "cy"))
    ref = lib.read_lines(lib.eval_ref_path(corpus, "cy"))
    ids = lib.read_lines(lib.eval_ids_path(corpus, "cy"))
    assert len(src) == len(ref) == len(ids) == 3
    assert ids == ["https://stub.example/one\t2026-07-25",
                   "https://stub.example/one\t2026-07-25",
                   "https://stub.example/two\t2026-07-24"]
    manifest = lib.verify_manifest(corpus, "cy")
    assert manifest["ids_sha256"]
    provenance = manifest["provenance"]
    assert provenance["ids_format"] == "source_url\tpublication_date"
    assert provenance["cutoff"] == "2026-07-20"
    assert provenance["harvest_date"] == "2026-07-29"
    assert provenance["extractor_version"] == trackb.EXTRACTOR_VERSION
    assert provenance["items"] == [
        {"url": "https://stub.example/two", "date": "2026-07-24", "n": 1},
        {"url": "https://stub.example/one", "date": "2026-07-25", "n": 2},
    ]
    assert provenance["sources"][0]["publisher"] == "Stub Publisher"


def test_sealed_slice_becomes_a_runnable_corpus(slice_dirs):
    harvest(StubFetch([]), {"cy": (fake_source([
        Candidate("https://stub.example/a", "2026-07-25", [(en(1), cy(1))]),
    ]),)})
    assert "trackb-t1" in lib.all_corpora()
    summary = trackb.slice_summary("t1")
    assert summary["runnable"] is True
    assert summary["n_total"] == 1
    assert summary["languages"]["cy"]["manifest"] is True
    assert summary["cutoff"] == "2026-07-20"


def test_slice_summary_refuses_an_unsealed_slice(slice_dirs):
    with pytest.raises(FileNotFoundError, match="no sealed slice trackb-t1"):
        trackb.slice_summary("t1")


# ---------------------------------------------------------------------------
# Honest gaps
# ---------------------------------------------------------------------------

def test_languages_without_a_source_are_named_with_a_reason(slice_dirs):
    record = harvest(StubFetch([]), trackb.SOURCES, langs=("br", "gv", "kw"))
    assert record["languages"] == {}
    reasons = {row["lang"]: row["reason"] for row in record["unavailable"]}
    assert set(reasons) == {"br", "gv", "kw"}
    assert "Breton" in reasons["br"]
    assert "Manx" in reasons["gv"]
    assert "Cornish" in reasons["kw"]
    for row in record["unavailable"]:
        assert row["language_name"] == lib.LANGS[row["lang"]]["name"]
        assert not os.path.exists(lib.eval_src_path(record["corpus"], row["lang"]))
        assert not os.path.exists(lib.manifest_path(record["corpus"], row["lang"]))


def test_a_source_that_yields_nothing_is_unavailable_not_empty(slice_dirs):
    barren = fake_source([Candidate("https://stub.example/old", "2026-01-01", [(en(1), cy(1))])],
                         source_id="barren")
    record = harvest(StubFetch([]), {"cy": (barren,)})
    assert record["languages"] == {}
    reason = record["unavailable"][0]["reason"]
    assert "barren" in reason and "not_after_cutoff" in reason
    assert not os.path.exists(lib.manifest_path(record["corpus"], "cy"))


def test_slice_file_is_written_and_self_describing(slice_dirs):
    record = harvest(StubFetch([]), {**trackb.SOURCES, "cy": (fake_source([
        Candidate("https://stub.example/a", "2026-07-25", [(en(1), cy(1))]),
    ]),)}, langs=("cy", "kw"))
    with open(trackb.slice_path("t1"), encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["schema"] == trackb.SCHEMA_SLICE
    assert written["corpus"] == "trackb-t1"
    assert written["languages"]["cy"]["n"] == 1
    assert [row["lang"] for row in written["unavailable"]] == ["kw"]
    assert written["harvested_at"].startswith("20")
    assert written == record


def test_resealing_a_changed_slice_needs_force(slice_dirs):
    def slice_with(pairs):
        return {"cy": (fake_source([Candidate("https://stub.example/a", "2026-07-25", pairs)]),)}

    harvest(StubFetch([]), slice_with([(en(1), cy(1))]))
    with pytest.raises(ValueError, match="do not match the committed manifest"):
        harvest(StubFetch([]), slice_with([(en(1), cy(1)), (en(2), cy(2))]))
    record = harvest(StubFetch([]), slice_with([(en(1), cy(1)), (en(2), cy(2))]), force=True)
    assert record["languages"]["cy"]["n"] == 2


def test_slice_summary_refuses_an_unknown_schema(slice_dirs):
    harvest(StubFetch([]), {"cy": (fake_source([
        Candidate("https://stub.example/a", "2026-07-25", [(en(1), cy(1))]),
    ]),)})
    path = trackb.slice_path("t1")
    text = open(path, encoding="utf-8").read().replace(trackb.SCHEMA_SLICE, "something.else.v9")
    open(path, "w", encoding="utf-8").write(text)
    with pytest.raises(ValueError, match="unknown slice schema"):
        trackb.slice_summary("t1")


# ---------------------------------------------------------------------------
# Registered adapters, driven offline
# ---------------------------------------------------------------------------

GW_LISTING = """
<ul class="index-list__items">
<li class="index-list__item"><div class="index-list__title">
<a href="/written-statement-fresh"><span>Fresh statement</span></a></div>
<div class="index-list__meta"><span class="index-list__date">
<time datetime="2026-07-25T09:00:00Z">25 July 2026</time></span>
<span class="index-list__type">Cabinet statement</span></div></li>
<li class="index-list__item"><div class="index-list__title">
<a href="/written-statement-stale"><span>Stale statement</span></a></div>
<div class="index-list__meta"><span class="index-list__date">
<time datetime="2026-07-01T09:00:00Z">1 July 2026</time></span></div></li>
</ul>
"""

GW_ARTICLE_EN = """<html><head>
<link rel="alternate" hreflang="en" href="https://www.gov.wales/written-statement-fresh" />
<link rel="alternate" hreflang="cy" href="https://www.llyw.cymru/datganiad-ffres" />
</head><body>
<div class="announcement-item__article" id="ann">
<div class="inner"><p>The Cabinet Secretary confirmed the new inspection regime today.</p></div>
<h2>Next steps for local authorities</h2>
<p>Local authorities will report on progress before the end of the financial year.</p>
</div>
<div class="footer"><ul><li>Share this page via Email</li><li>Accessibility</li></ul>
<p>Copyright statement for the whole of this Welsh Government website.</p></div>
</body></html>
"""

GW_ARTICLE_CY = """<html><head>
<link rel="alternate" hreflang="cy" href="https://www.llyw.cymru/datganiad-ffres" />
</head><body>
<div class="announcement-item__article" id="ann">
<div class="inner"><p>Cadarnhaodd Ysgrifennydd y Cabinet y drefn arolygu newydd heddiw.</p></div>
<h2>Y camau nesaf i awdurdodau lleol</h2>
<p>Bydd awdurdodau lleol yn adrodd ar gynnydd cyn diwedd y flwyddyn ariannol.</p>
</div>
<div class="footer"><ul><li>Rhannwch y dudalen hon ar E-bost</li><li>Hygyrchedd</li></ul>
<p>Datganiad hawlfraint ar gyfer gwefan Llywodraeth Cymru gyfan.</p></div>
</body></html>
"""


def test_gov_wales_adapter_pairs_only_the_article_container(slice_dirs):
    fetch = StubFetch([
        ("/announcements/search?page=0", GW_LISTING),
        ("gov.wales/written-statement-fresh", GW_ARTICLE_EN),
        ("llyw.cymru/datganiad-ffres", GW_ARTICLE_CY),
    ])
    record = harvest(fetch, {"cy": trackb.SOURCES["cy"]})
    src = lib.read_lines(lib.eval_src_path("trackb-t1", "cy"))
    ref = lib.read_lines(lib.eval_ref_path("trackb-t1", "cy"))
    assert src == [
        "The Cabinet Secretary confirmed the new inspection regime today.",
        "Local authorities will report on progress before the end of the financial year.",
    ]
    assert ref[0].startswith("Cadarnhaodd Ysgrifennydd")
    # The h2 had to align for the document to be accepted at all, then fell to
    # the length floor: headings are structure, not translation evidence.
    assert record["languages"]["cy"]["rejected"] == {"too_short": 1}
    # Footer chrome translates perfectly and would align; scoping keeps it out.
    assert not any("Accessibility" in line or "hawlfraint" in line for line in src + ref)
    # The date-descending listing stops at the first item on or before the cutoff.
    assert "written-statement-stale" not in "".join(fetch.seen)


def test_gov_wales_adapter_drops_a_page_with_no_welsh_twin(slice_dirs):
    english = GW_ARTICLE_EN.replace(
        '<link rel="alternate" hreflang="cy" href="https://www.llyw.cymru/datganiad-ffres" />', "")
    fetch = StubFetch([
        ("/announcements/search?page=0", GW_LISTING),
        ("gov.wales/written-statement-fresh", english),
    ])
    record = harvest(fetch, {"cy": trackb.SOURCES["cy"]})
    assert record["languages"] == {}
    assert record["unavailable"][0]["lang"] == "cy"


EURLEX_ROWS = {"results": {"bindings": [
    {"celex": {"value": "32026R1894"}, "date": {"value": "2026-07-29"}},
    {"celex": {"value": "32023R2674R(01)"}, "date": {"value": "2026-07-28"}},
    {"celex": {"value": "32026R1813"}, "date": {"value": "2026-07-28"}},
]}}

EURLEX_EN = """<html><body>
<p>2026/1894</p>
<p>The Commission adopted the implementing regulation on port state control.</p>
<p>Member States shall bring into force the measures needed to comply by June.</p>
</body></html>
"""

EURLEX_GA = """<html><body>
<p>2026/1894</p>
<p>Ghlac an Coimisiún an rialachán cur chun feidhme maidir le rialú stáit poirt.</p>
<p>Cuirfidh na Ballstáit i bhfeidhm na bearta is g\u00e1 chun comhl\u00edonadh a dh\u00e9anamh.</p>
</body></html>
"""


def test_eurlex_adapter_dates_documents_from_the_cellar_graph(slice_dirs):
    fetch = StubFetch([
        ("webapi/rdf/sparql", json.dumps(EURLEX_ROWS)),
        ("EN/TXT/HTML/?uri=CELEX:32026R1894", EURLEX_EN),
        ("GA/TXT/HTML/?uri=CELEX:32026R1894", EURLEX_GA),
    ])
    record = harvest(fetch, {"ga": trackb.SOURCES["ga"]}, langs=("ga",))
    row = record["languages"]["ga"]
    assert row["n"] == 2
    # The bare document number is identical on both sides and below the floor.
    assert row["rejected"] == {"too_short": 1}
    ids = lib.read_lines(lib.eval_ids_path("trackb-t1", "ga"))
    assert ids[0] == ("https://eur-lex.europa.eu/legal-content/GA/TXT/?uri=CELEX:32026R1894"
                      "\t2026-07-29")
    # The date came from the graph, never from the URL we asked for.
    query = next(url for url in fetch.seen if "webapi/rdf/sparql" in url)
    assert "official-journal-act_date_publication" in urllib.parse.unquote_plus(query)


def test_eurlex_adapter_skips_corrigenda(slice_dirs):
    fetch = StubFetch([("webapi/rdf/sparql", json.dumps(EURLEX_ROWS))])
    record = harvest(fetch, {"ga": trackb.SOURCES["ga"]}, langs=("ga",))
    assert record["languages"] == {}
    requested = "".join(fetch.seen)
    # A corrigendum republishes older text: never even fetched.
    assert "32023R2674R(01)" not in requested
    assert "CELEX:32026R1894" in requested and "CELEX:32026R1813" in requested


def test_eurlex_adapter_stops_on_an_unusable_sparql_response(slice_dirs):
    for body in ("not json at all", json.dumps({"results": {"bindings": []}})):
        fetch = StubFetch([("webapi/rdf/sparql", body)])
        record = harvest(fetch, {"ga": trackb.SOURCES["ga"]}, langs=("ga",))
        assert record["languages"] == {}
        assert sum("legal-content" in url for url in fetch.seen) == 0


def _wp_post(post_id, lang, date, body, translations=None, link=None):
    post = {
        "id": post_id, "lang": lang, "date_gmt": date, "slug": f"post-{post_id}",
        "link": link or f"https://www.gaidhlig.scot/{lang}/post-{post_id}/",
        "content": {"rendered": body},
    }
    if translations is not None:
        post["translations"] = translations
    return post


WP_GD_BODY = ("<p>Tha Bòrd na Gàidhlig air maoineachadh ùr a chur air bhog airson sgoiltean.</p>"
              "<p>Bidh an sgeama fosgailte do thagraidhean gu deireadh na Sultaine.</p>")
WP_EN_BODY = ("<p>Bòrd na Gàidhlig has launched new funding for schools across Scotland.</p>"
              "<p>The scheme is open for applications until the end of September.</p>")


def test_wordpress_polylang_adapter_uses_the_translation_record(slice_dirs):
    fetch = StubFetch([
        ("gaidhlig.scot/wp-json/wp/v2/posts/41", json.dumps(_wp_post(41, "en", "2026-07-25T09:00:00", WP_EN_BODY))),
        ("gaidhlig.scot/wp-json/wp/v2/posts?lang=gd",
         json.dumps([_wp_post(40, "gd", "2026-07-25T09:00:00", WP_GD_BODY, {"en": 41, "gd": 40})])),
    ])
    record = harvest(fetch, {"gd": (trackb.SOURCES["gd"][0],)}, langs=("gd",))
    assert record["languages"]["gd"]["n"] == 2
    assert lib.read_lines(lib.eval_src_path("trackb-t1", "gd"))[0].startswith("Bòrd na Gàidhlig has launched")
    assert lib.read_lines(lib.eval_ref_path("trackb-t1", "gd"))[0].startswith("Tha Bòrd na Gàidhlig")


def test_wordpress_polylang_adapter_drops_an_untranslated_post(slice_dirs):
    fetch = StubFetch([
        ("gaidhlig.scot/wp-json/wp/v2/posts?lang=gd",
         json.dumps([_wp_post(40, "gd", "2026-07-25T09:00:00", WP_GD_BODY, {"gd": 40})])),
    ])
    record = harvest(fetch, {"gd": (trackb.SOURCES["gd"][0],)}, langs=("gd",))
    assert record["languages"] == {}


def test_wordpress_hreflang_adapter_resolves_the_english_slug(slice_dirs):
    gd_page = ('<html><head><link rel="alternate" hreflang="en" '
               'href="https://mgalba.com/new-funding-for-schools/?lang=en" />'
               '</head><body>rendered</body></html>')
    fetch = StubFetch([
        ("mgalba.com/wp-json/wp/v2/posts?slug=new-funding-for-schools",
         json.dumps([_wp_post(51, "en", "2026-07-25T09:00:00", WP_EN_BODY)])),
        ("mgalba.com/wp-json/wp/v2/posts?lang=gd",
         json.dumps([_wp_post(50, "gd", "2026-07-25T09:00:00", WP_GD_BODY,
                              link="https://mgalba.com/maoineachadh-ur/")])),
        ("mgalba.com/maoineachadh-ur/", gd_page),
    ])
    record = harvest(fetch, {"gd": (trackb.SOURCES["gd"][1],)}, langs=("gd",))
    assert record["languages"]["gd"]["n"] == 2
    assert record["languages"]["gd"]["sources"] == ["mg-alba-news"]


def test_registered_sources_are_only_the_verified_ones():
    assert sorted(trackb.SOURCES) == ["cy", "ga", "gd"]
    assert sorted(trackb.UNAVAILABLE) == ["br", "gv", "kw"]
    assert not set(trackb.SOURCES) & set(trackb.UNAVAILABLE)
    assert set(trackb.SOURCES) | set(trackb.UNAVAILABLE) == set(lib.ALL_LANGS)
    for sources in trackb.SOURCES.values():
        for source in sources:
            assert source.publisher and source.licence and source.home.startswith("https://")


def test_one_document_cannot_monopolise_a_slice(slice_dirs):
    huge = [(en(i), cy(i)) for i in range(trackb.MAX_BLOCKS_PER_DOCUMENT + 25)]
    sources = {"cy": (fake_source([
        Candidate("https://stub.example/act", "2026-07-25", huge),
        Candidate("https://stub.example/next", "2026-07-24", [(en(500), cy(500))]),
    ]),)}
    record = harvest(StubFetch([]), sources, limit_per_lang=100)
    row = record["languages"]["cy"]
    assert row["n"] == trackb.MAX_BLOCKS_PER_DOCUMENT + 1
    assert row["documents"] == 2
    # Truncation is not rejection: the surplus blocks were simply not needed.
    assert row["rejected"] == {}
