"""Track B: dated fresh-harvest slices of bilingual text from official publishers.

Track A is contaminated by construction: FLORES and Tatoeba predate every
benchmarked model. Track B answers that by harvesting text a publisher put out
*after* the newest benchmarked model's training cutoff, sealing it as a dated
slice, and publishing the manifest hash before any model sees it.

A slice is a corpus named trackb-<slice_id> (lib.TRACK_B_PREFIX), so
lib.all_corpora() picks it up from its committed manifests with no code change.

What makes an item eligible:
  * the publisher states a publication date, machine-readable, strictly after
    the cutoff. No date means dropped -- freshness is never assumed.
  * the same document exists in English and the Celtic language on that
    publisher's own site, linked by the publisher (hreflang, or a CMS
    translation record), never by us guessing a URL.
  * the two documents decompose into the same sequence of block tags. A
    different shape means we cannot claim which block translates which, so the
    whole document is dropped rather than aligned optimistically.

Every registered source in SOURCES was fetched successfully while writing this
module. Languages with no viable source are named in UNAVAILABLE with the
reason and stay empty; a slice never substitutes data for a missing language.

Network access is confined to http_fetch and reaches the harvester only through
the `fetch` argument, so tests drive the whole pipeline from a dict.
"""
from __future__ import annotations

import collections
import dataclasses
import datetime
import functools
import html
import json
import os
import re
import time
import urllib.parse
from typing import Any, Callable, Iterator

import certifi
import requests

from . import lib

EXTRACTOR_VERSION = "trackb-extract-1"
SCHEMA_SLICE = lib.SCHEMA_SLICE

# Self-identifying agent with a contact hint. Every registered source serves it
# a 200; sources that only answer a spoofed browser UA are deliberately not
# registered (see the module docstring's honesty requirement).
USER_AGENT = "celticbench-trackb/1.0 (Celtic translation benchmark corpus harvest)"
REQUEST_TIMEOUT = 45
REQUEST_PAUSE = 0.2  # a live harvest walks thousands of public-body pages

# Segment filters. Bounds were measured on real pairs from the registered
# sources: Welsh/English length ratios ran 0.88-1.41 (median 1.05) across
# gov.wales announcements; Irish/English 0.56-2.05 (median 1.09) across 3806
# segments of Official Journal acts, of which these bounds reject 2. Wide
# enough that a rejection means the pair is wrong, not that the language is
# verbose.
MIN_CHARS = 40
MAX_CHARS = 1000
MIN_RATIO = 0.6
MAX_RATIO = 2.0

# One document may contribute at most this many segments. An Official Journal
# act runs to hundreds of aligned paragraphs, so without a ceiling a single
# regulation would be the entire Irish slice and the corpus would measure one
# document's register. Small enough to spread a slice across tens of documents,
# large enough that short announcements are never truncated.
MAX_BLOCKS_PER_DOCUMENT = 40

# Walk limits. A slice is a quarter, so these bound a runaway harvest without
# truncating a legitimate one.
LISTING_MAX_PAGES = 400
WP_MAX_PAGES = 10
SPARQL_MAX_PAGES = 20
SPARQL_PAGE = 200

Fetch = Callable[[str], "str | None"]


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One document a source believes is a dated bilingual pair.

    `date` is the publisher-stated publication date (ISO, "" when the publisher
    gave none). `payload` carries whatever the extractor needs and never leaves
    the source that produced it.
    """
    url: str
    date: str
    payload: Any = None


@dataclasses.dataclass(frozen=True)
class Pair:
    english: str
    celtic: str
    url: str
    date: str


@dataclasses.dataclass(frozen=True)
class Source:
    """An official publisher of parallel English/Celtic text.

    discover(fetch, cutoff, harvest_date) yields candidates newest-first, so a
    capped harvest takes the freshest eligible text and stops fetching.
    extract(fetch, candidate) returns structurally aligned (english, celtic)
    blocks, or nothing at all when the two documents do not correspond.
    """
    id: str
    publisher: str
    home: str
    licence: str
    note: str
    discover: Callable[[Fetch, str, str], Iterator[Candidate]]
    extract: Callable[[Fetch, Candidate], list[tuple[str, str]]]


def http_fetch(url: str) -> str | None:
    """Fetch one URL. None means "not retrievable", which drops the document.

    Anything other than 200 -- a language version that does not exist, a
    withdrawn page, a rate limit -- must not become a half-built pair.

    Deliberately connection-per-request and cookie-free. Publisher listings that
    remember a selection in the session (EUR-Lex's Official Journal daily view
    does) will serve a stale document set under a new date parameter once a
    cookie jar is shared, which would attach the wrong publication date to real
    text. Statelessness here is a correctness property, not tidiness.
    """
    time.sleep(REQUEST_PAUSE)
    response = requests.get(url, headers={"User-Agent": USER_AGENT},
                            timeout=REQUEST_TIMEOUT, verify=certifi.where())
    if response.status_code != 200:
        return None
    return response.text


# ---------------------------------------------------------------------------
# HTML block extraction
#
# Deliberately regex-level and deliberately dumb: the only structure we trust
# is "the same tag sequence on both sides". Anything cleverer would be a
# heuristic aligner, and a heuristic aligner silently produces mismatched
# reference pairs, which is worse than a smaller corpus.
# ---------------------------------------------------------------------------

_DIV_EDGE = re.compile(r"<div\b|</div>", re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _text(fragment: str) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", fragment))).strip()


def _div_slice(document: str, opening: str) -> str | None:
    """Inner HTML of the div whose opening tag starts with `opening`.

    Depth-tracked rather than regex-matched: the article container nests, and a
    non-greedy match would stop at the first inner </div> and silently lose
    most of the body.
    """
    start = document.find(opening)
    if start < 0:
        return None
    body = document.index(">", start) + 1
    depth = 1
    for edge in _DIV_EDGE.finditer(document, body):
        depth += 1 if edge.group(0)[1] != "/" else -1
        if depth == 0:
            return document[body:edge.start()]
    return None


@functools.lru_cache(maxsize=8)
def _block_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"<(" + "|".join(tags) + r")\b[^>]*>(.*?)</\1>", re.S | re.I)


def blocks(document: str, tags: tuple[str, ...]) -> list[tuple[str, str]]:
    """(tag, text) for every non-empty block of an accepted tag, in order."""
    found = []
    for match in _block_pattern(tags).finditer(document):
        text = _text(match.group(2))
        if text:
            found.append((match.group(1).lower(), text))
    return found


def container_blocks(document: str, opening: str, tags: tuple[str, ...]) -> list[tuple[str, str]]:
    """Blocks inside one named container; empty when the container is absent.

    Scoping to the article container is what keeps site chrome -- share links,
    footer menus, cookie notices -- out of the corpus. Those are translated
    too, and they would align perfectly, and they are not translation data.
    """
    inner = _div_slice(document, opening)
    return blocks(inner, tags) if inner else []


def align(english: list[tuple[str, str]], celtic: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Pair blocks positionally, but only if the documents have one shape.

    Equal length *and* equal tag sequence. A translated page that gained a
    paragraph, lost a heading or reordered a list is not evidence about any
    individual block, so it contributes nothing.
    """
    if not english or len(english) != len(celtic):
        return []
    if [tag for tag, _ in english] != [tag for tag, _ in celtic]:
        return []
    return [(en_text, ce_text) for (_, en_text), (_, ce_text) in zip(english, celtic)]


# Attribute order is not fixed in the wild: gov.wales emits hreflang before
# href, gaidhlig.scot emits href first. Both orders, first declaration wins.
_HREFLANG = re.compile(r'<link[^>]+rel="alternate"[^>]+hreflang="([\w-]+)"[^>]+href="([^"]+)"', re.I)
_HREFLANG_SWAPPED = re.compile(r'<link[^>]+rel="alternate"[^>]+href="([^"]+)"[^>]+hreflang="([\w-]+)"', re.I)


def hreflang(document: str) -> dict[str, str]:
    """Publisher-declared language versions of this page, keyed by base code."""
    found: dict[str, str] = {}
    for code, url in _HREFLANG.findall(document):
        found.setdefault(code.split("-")[0].lower(), html.unescape(url))
    for url, code in _HREFLANG_SWAPPED.findall(document):
        found.setdefault(code.split("-")[0].lower(), html.unescape(url))
    return found


# ---------------------------------------------------------------------------
# Welsh: Welsh Government (gov.wales / llyw.cymru)
#
# Statutorily bilingual: every announcement is published in both languages and
# each English page declares its Welsh twin with hreflang. Announcements are
# one-off dated publications, which is what Track B needs -- the evergreen
# guidance pages on the same site are edited in place and their "last updated"
# date says nothing about when the text was written.
# Listing verified 2026-07-29: /announcements/search?page=N, 10 date-descending
# items per page, each with an ISO datetime attribute.
# ---------------------------------------------------------------------------

GOV_WALES = "https://www.gov.wales"
_GW_CONTAINER = '<div class="announcement-item__article"'
_GW_TAGS = ("p", "h2", "h3", "h4", "li")
_GW_ITEM = re.compile(r'<li class="index-list__item">.*?</li>', re.S)
_GW_HREF = re.compile(r'<a href="(/[^"]+)"')
_GW_DATE = re.compile(r'datetime="(\d{4}-\d{2}-\d{2})')


def _gov_wales_discover(fetch: Fetch, cutoff: str, harvest_date: str) -> Iterator[Candidate]:
    for page in range(LISTING_MAX_PAGES):
        document = fetch(f"{GOV_WALES}/announcements/search?page={page}")
        if document is None:
            return
        items = _GW_ITEM.findall(document)
        if not items:
            return
        for item in items:
            href, date = _GW_HREF.search(item), _GW_DATE.search(item)
            if not href:
                continue
            if not date:
                yield Candidate(url=GOV_WALES + href.group(1), date="")
                continue
            if date.group(1) <= cutoff:
                return  # date-descending listing: nothing later is eligible
            yield Candidate(url=GOV_WALES + href.group(1), date=date.group(1))


def _gov_wales_extract(fetch: Fetch, candidate: Candidate) -> list[tuple[str, str]]:
    english = fetch(candidate.url)
    if english is None:
        return []
    welsh_url = hreflang(english).get("cy")
    if not welsh_url:
        return []
    welsh = fetch(welsh_url)
    if welsh is None:
        return []
    return align(container_blocks(english, _GW_CONTAINER, _GW_TAGS),
                 container_blocks(welsh, _GW_CONTAINER, _GW_TAGS))


# ---------------------------------------------------------------------------
# Irish: Official Journal of the EU, secondary legislation (EUR-Lex / Cellar)
#
# Irish became a full working language of the EU on 2022-01-01, so acts are
# published in Irish and English as equally authentic texts. Both versions
# render from one Formex source, which is why the paragraph sequences match
# exactly rather than approximately: 31 of 31 sampled acts aligned, yielding
# 3806 usable segments over two publication days.
#
# Discovery goes through the Cellar SPARQL endpoint, NOT the Official Journal
# daily view. The daily view cannot date a document: checked 2026-07-29, it
# returned the same 17 ids for ojDate=20260722 and 20260728, and once a session
# cookie is held it returns that set for every date. Cellar instead states the
# date per document, in the publisher's own graph, and in the same query proves
# that both an English and an Irish expression exist -- so a language version we
# would otherwise discover by 404 is never requested at all.
#
# Scoped to CELEX sector 3 (secondary legislation). Corrigenda are excluded:
# their publication date is fresh but their text republishes an older act, which
# is exactly the contamination Track B exists to avoid.
# ---------------------------------------------------------------------------

EURLEX = "https://eur-lex.europa.eu"
CELLAR_SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"

# CELEX ids of corrigenda end in R(nn); 32023R2674R(01) is a 2026 corrigendum to
# a 2023 regulation, and its Irish text is not new.
_CORRIGENDUM = re.compile(r"R\(\d+\)$")

_OJ_QUERY = """PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX lang: <http://publications.europa.eu/resource/authority/language/>
SELECT DISTINCT ?celex ?date WHERE {
  ?work cdm:official-journal-act_date_publication ?date .
  ?work cdm:resource_legal_id_celex ?celex .
  ?en cdm:expression_belongs_to_work ?work ; cdm:expression_uses_language lang:ENG .
  ?ga cdm:expression_belongs_to_work ?work ; cdm:expression_uses_language lang:GLE .
  FILTER (?date > "%(cutoff)s"^^xsd:date && ?date <= "%(until)s"^^xsd:date)
  FILTER (STRSTARTS(str(?celex), "3"))
} ORDER BY DESC(?date) ASC(?celex) LIMIT %(limit)d OFFSET %(offset)d"""


def _sparql_url(query: str) -> str:
    return CELLAR_SPARQL + "?" + urllib.parse.urlencode(
        {"query": query, "format": "application/sparql-results+json"})


def _eurlex_discover(fetch: Fetch, cutoff: str, harvest_date: str) -> Iterator[Candidate]:
    for page in range(SPARQL_MAX_PAGES):
        raw = fetch(_sparql_url(_OJ_QUERY % {
            "cutoff": cutoff, "until": harvest_date,
            "limit": SPARQL_PAGE, "offset": page * SPARQL_PAGE}))
        if raw is None:
            return
        try:
            rows = json.loads(raw)["results"]["bindings"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return
        if not rows:
            return
        for row in rows:
            celex = row.get("celex", {}).get("value", "")
            if not celex or _CORRIGENDUM.search(celex):
                continue
            yield Candidate(url=f"{EURLEX}/legal-content/GA/TXT/?uri=CELEX:{celex}",
                            date=row.get("date", {}).get("value", ""), payload=celex)
        if len(rows) < SPARQL_PAGE:
            return


def _eurlex_extract(fetch: Fetch, candidate: Candidate) -> list[tuple[str, str]]:
    english = fetch(f"{EURLEX}/legal-content/EN/TXT/HTML/?uri=CELEX:{candidate.payload}")
    irish = fetch(f"{EURLEX}/legal-content/GA/TXT/HTML/?uri=CELEX:{candidate.payload}")
    if english is None or irish is None:
        return []
    return align(blocks(english, ("p",)), blocks(irish, ("p",)))


# ---------------------------------------------------------------------------
# Scottish Gaelic: WordPress public bodies
#
# Bòrd na Gàidhlig is the statutory Gaelic development body; MG ALBA is the
# statutory Gaelic Media Service. Both run WordPress with a translation plugin,
# so the REST API hands over the publication date and the post body as data
# instead of us scraping a theme.
#
# Two pairing mechanisms, named per source rather than tried in turn, because a
# fallback here would mean guessing which post is the translation:
#   polylang  the post's own `translations` map gives the English post id
#   hreflang  the rendered post declares the English URL; its slug is then
#             looked up in the REST collection scoped to lang=en
# Verified 2026-07-29 on both sites. Gaelic public-body output is thin -- eight
# Bòrd na Gàidhlig posts in 2026 to date -- and the slice records that honestly
# rather than padding it.
# ---------------------------------------------------------------------------

_WP_TAGS = ("p", "h2", "h3", "h4", "li")


def _wp_json(fetch: Fetch, url: str) -> Any:
    raw = fetch(url)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _wp_discover(fetch: Fetch, cutoff: str, harvest_date: str, *,
                 base: str, code: str) -> Iterator[Candidate]:
    for page in range(1, WP_MAX_PAGES + 1):
        posts = _wp_json(fetch, f"{base}/wp-json/wp/v2/posts?lang={code}&per_page=100"
                                f"&orderby=date&order=desc&page={page}&after={cutoff}T23:59:59")
        if not isinstance(posts, list) or not posts:
            return
        for post in posts:
            if not isinstance(post, dict):
                continue
            date = str(post.get("date_gmt") or post.get("date") or "")
            yield Candidate(url=str(post.get("link") or ""), date=date, payload=post)
        if len(posts) < 100:
            return


def _wp_body(post: Any) -> str:
    if not isinstance(post, dict):
        return ""
    return str(((post.get("content") or {}).get("rendered")) or "")


def _wp_polylang_extract(fetch: Fetch, candidate: Candidate, *, base: str) -> list[tuple[str, str]]:
    english_id = ((candidate.payload or {}).get("translations") or {}).get("en")
    if not english_id:
        return []
    english = _wp_json(fetch, f"{base}/wp-json/wp/v2/posts/{english_id}")
    if not isinstance(english, dict):
        return []
    return align(blocks(_wp_body(english), _WP_TAGS), blocks(_wp_body(candidate.payload), _WP_TAGS))


def _wp_slug(url: str) -> str:
    return url.split("?")[0].split("#")[0].rstrip("/").rsplit("/", 1)[-1]


def _wp_hreflang_extract(fetch: Fetch, candidate: Candidate, *, base: str) -> list[tuple[str, str]]:
    rendered = fetch(candidate.url)
    if rendered is None:
        return []
    english_url = hreflang(rendered).get("en")
    if not english_url:
        return []
    slug = _wp_slug(english_url)
    if not slug:
        return []
    found = _wp_json(fetch, f"{base}/wp-json/wp/v2/posts?slug={slug}&lang=en")
    if not isinstance(found, list) or not found:
        return []
    return align(blocks(_wp_body(found[0]), _WP_TAGS), blocks(_wp_body(candidate.payload), _WP_TAGS))


def _wp_source(source_id: str, publisher: str, base: str, licence: str, note: str,
               code: str, pairing: str) -> Source:
    extract = _wp_polylang_extract if pairing == "polylang" else _wp_hreflang_extract
    return Source(
        id=source_id, publisher=publisher, home=base, licence=licence, note=note,
        discover=functools.partial(_wp_discover, base=base, code=code),
        extract=functools.partial(extract, base=base),
    )


SOURCES: dict[str, tuple[Source, ...]] = {
    "cy": (
        Source(
            id="gov-wales-announcements",
            publisher="Welsh Government / Llywodraeth Cymru",
            home=GOV_WALES,
            licence="Open Government Licence v3.0",
            note="press releases, news stories and Cabinet written statements; "
                 "English page and its hreflang=cy twin",
            discover=_gov_wales_discover,
            extract=_gov_wales_extract,
        ),
    ),
    "ga": (
        Source(
            id="eurlex-oj-secondary-legislation",
            publisher="Publications Office of the European Union",
            home=EURLEX,
            licence="© European Union; reuse under Decision 2011/833/EU",
            note="Official Journal secondary legislation; Irish and English are "
                 "equally authentic texts rendered from one Formex source, dated "
                 "per document by the Cellar SPARQL endpoint",
            discover=_eurlex_discover,
            extract=_eurlex_extract,
        ),
    ),
    "gd": (
        _wp_source(
            "bord-na-gaidhlig-news", "Bòrd na Gàidhlig", "https://www.gaidhlig.scot",
            "© Bòrd na Gàidhlig; quoted for research",
            "news posts paired by the site's own Polylang translation record",
            code="gd", pairing="polylang",
        ),
        _wp_source(
            "mg-alba-news", "MG ALBA / Seirbheis nam Meadhanan Gàidhlig", "https://mgalba.com",
            "© MG ALBA; quoted for research",
            "news posts paired by the rendered page's hreflang=en declaration",
            code="gd", pairing="hreflang",
        ),
    ),
}

# Languages with no registered source, and why. Checked 2026-07-29 by fetching
# each candidate publisher named here.
UNAVAILABLE: dict[str, str] = {
    "br": "Breton public-sector bilingualism is French-Breton, not English-Breton. "
          "Ofis Publik ar Brezhoneg (brezhoneg.bzh) serves Breton and French only "
          "-- its /en/ path 404s -- and Region Bretagne (bretagne.bzh) refuses "
          "automated requests outright. No official publisher of parallel "
          "English/Breton text was found.",
    "gv": "The Isle of Man Government publishes in English; gov.im has no Manx "
          "language section to mirror. Culture Vannin, the statutory Manx culture "
          "body, publishes about Manx in English rather than issuing the same "
          "document in both languages. No dated parallel English/Manx stream exists.",
    "kw": "Cornwall Council's Cornish language pages are a small static set with no "
          "Cornish-language mirror of its dated publications, and no Cornish public "
          "body issues a parallel English/Cornish publication stream. Cornish "
          "revival material is produced as Cornish-only or English-only.",
}


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------

_SLICE_ID = re.compile(r"^[a-z0-9]+$")


def corpus_name(slice_id: str) -> str:
    return lib.TRACK_B_PREFIX + slice_id


def slice_path(slice_id: str) -> str:
    return os.path.join(lib.MANIFESTS, f"{corpus_name(slice_id)}.slice.json")


def _check_slice_id(slice_id: str) -> str:
    if not _SLICE_ID.match(slice_id):
        raise ValueError(f"slice id {slice_id!r} must be lowercase alphanumeric, e.g. '2026q3'")
    return slice_id


def _check_date(value: str, label: str) -> str:
    try:
        return datetime.date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}") from None


def _dedup_key(english: str) -> str:
    return " ".join(english.casefold().split())


def reject_reason(english: str, celtic: str) -> str | None:
    """Why this pair is not corpus material, or None if it is.

    Order matters, because the counters in the slice file are read as a
    diagnosis of a source: emptiness before length, length before sameness,
    sameness before ratio.
    """
    english, celtic = english.strip(), celtic.strip()
    if not english or not celtic:
        return "empty"
    if len(english) < MIN_CHARS:
        return "too_short"
    if len(english) > MAX_CHARS:
        return "too_long"
    if _dedup_key(english) == _dedup_key(celtic):
        return "identical"  # untranslated fallback page served under both URLs
    ratio = len(celtic) / len(english)
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return "length_ratio"
    return None


def _harvest_lang(fetch: Fetch, sources: tuple[Source, ...], cutoff: str,
                  harvest_date: str, limit: int) -> tuple[list[Pair], list[Source], dict[str, int]]:
    selected: list[Pair] = []
    seen: set[str] = set()
    rejected: collections.Counter[str] = collections.Counter()
    used: list[Source] = []
    for source in sources:
        if len(selected) >= limit:
            break
        before = len(selected)
        for candidate in source.discover(fetch, cutoff, harvest_date):
            if len(selected) >= limit:
                break
            if not candidate.date:
                rejected["undated"] += 1
                continue
            if candidate.date[:10] <= cutoff:
                rejected["not_after_cutoff"] += 1
                continue
            taken = 0
            for english, celtic in source.extract(fetch, candidate):
                if len(selected) >= limit or taken >= MAX_BLOCKS_PER_DOCUMENT:
                    break
                reason = reject_reason(english, celtic)
                if reason:
                    rejected[reason] += 1
                    continue
                key = _dedup_key(english)
                if key in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(key)
                taken += 1
                selected.append(Pair(english.strip(), celtic.strip(),
                                     candidate.url, candidate.date[:10]))
        if len(selected) > before:
            used.append(source)
    return selected, used, dict(sorted(rejected.items()))


def _item_provenance(pairs: list[Pair]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for pair in pairs:
        counts[(pair.date, pair.url)] = counts.get((pair.date, pair.url), 0) + 1
    return [{"url": url, "date": date, "n": n}
            for (date, url), n in sorted(counts.items())]


def harvest_slice(slice_id: str, cutoff: str, langs: tuple[str, ...] | None = None,
                  limit_per_lang: int = 500, force: bool = False,
                  fetch: Fetch | None = None, harvest_date: str | None = None,
                  sources: dict[str, tuple[Source, ...]] | None = None) -> dict[str, Any]:
    """Harvest and seal one Track B slice; returns the slice record.

    `cutoff` is the newest benchmarked model's training cutoff: only text a
    publisher dated strictly after it is eligible.

    `harvest_date` bounds the window and is an explicit argument, not a clock
    read, so re-running the same call against unchanged upstream pages rebuilds
    byte-identical eval files and the same manifest hashes. It defaults to
    today (UTC) for the one live harvest that seals the slice.

    Writes per language:
      eval/trackb-<slice>.<lang>.{src,ref,ids}   ids = url TAB publication date
      manifests/trackb-<slice>.<lang>.json       hash contract + provenance
    and one manifests/trackb-<slice>.slice.json describing the whole slice,
    including the languages that produced nothing and why.
    """
    _check_slice_id(slice_id)
    cutoff = _check_date(cutoff, "cutoff")
    harvest_date = _check_date(harvest_date or _today(), "harvest_date")
    if harvest_date <= cutoff:
        raise ValueError(f"harvest_date {harvest_date} is not after cutoff {cutoff}; "
                         f"a slice needs a window of text published after the cutoff")
    if limit_per_lang < 1:
        raise ValueError(f"limit_per_lang must be positive, got {limit_per_lang}")

    fetch = fetch or http_fetch
    table = SOURCES if sources is None else sources
    corpus = corpus_name(slice_id)
    wanted = tuple(langs) if langs else lib.ALL_LANGS
    unknown = [lang for lang in wanted if lang not in lib.LANGS]
    if unknown:
        raise ValueError(f"unknown languages {unknown}")

    languages: dict[str, Any] = {}
    unavailable: list[dict[str, str]] = []
    for lang in wanted:
        registered = table.get(lang, ())
        if not registered:
            unavailable.append({
                "lang": lang,
                "language_name": lib.LANGS[lang]["name"],
                "reason": UNAVAILABLE.get(lang, "no source registered for this language"),
            })
            continue
        pairs, used, rejected = _harvest_lang(fetch, registered, cutoff, harvest_date,
                                              limit_per_lang)
        if not pairs:
            detail = ", ".join(f"{name}={count}" for name, count in rejected.items())
            unavailable.append({
                "lang": lang,
                "language_name": lib.LANGS[lang]["name"],
                "reason": f"no pair from {', '.join(s.id for s in registered)} survived the "
                          f"cutoff {cutoff} and extraction checks "
                          f"({detail or 'no candidate offered'})",
            })
            continue
        items = _item_provenance(pairs)
        provenance = {
            "track": "B",
            "slice": slice_id,
            "cutoff": cutoff,
            "harvest_date": harvest_date,
            "extractor_version": EXTRACTOR_VERSION,
            "sources": [{"id": s.id, "publisher": s.publisher, "home": s.home,
                         "licence": s.licence, "note": s.note} for s in used],
            "selection": "blocks of structurally identical documents, newest publication "
                         "first, deduplicated on the English side, at most "
                         f"{MAX_BLOCKS_PER_DOCUMENT} per document, capped at {limit_per_lang}",
            "filters": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
                        "min_ratio": MIN_RATIO, "max_ratio": MAX_RATIO,
                        "max_blocks_per_document": MAX_BLOCKS_PER_DOCUMENT,
                        "identical_sides": "rejected", "block_alignment": "equal tag sequence"},
            "rejected": rejected,
            "ids_format": "source_url\tpublication_date",
            "items": items,
        }
        lib.write_lines(lib.eval_src_path(corpus, lang), [p.english for p in pairs])
        lib.write_lines(lib.eval_ref_path(corpus, lang), [p.celtic for p in pairs])
        lib.write_lines(lib.eval_ids_path(corpus, lang), [f"{p.url}\t{p.date}" for p in pairs])
        manifest = lib.write_manifest(corpus, lang, len(pairs), provenance, force=force)
        languages[lang] = {
            "language_name": lib.LANGS[lang]["name"],
            "n": len(pairs),
            "documents": len(items),
            "sources": [s.id for s in used],
            "rejected": rejected,
            "contract_sha256": manifest["contract_sha256"],
        }

    record = {
        "schema": SCHEMA_SLICE,
        "slice": slice_id,
        "corpus": corpus,
        "cutoff": cutoff,
        "harvest_date": harvest_date,
        "harvested_at": _now(),
        "extractor_version": EXTRACTOR_VERSION,
        "limit_per_lang": limit_per_lang,
        "languages": languages,
        "unavailable": unavailable,
    }
    os.makedirs(lib.MANIFESTS, exist_ok=True)
    with open(slice_path(slice_id), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return record


def slice_summary(slice_id: str) -> dict[str, Any]:
    """Committed record of a sealed slice, for the CLI. Never touches the network.

    Adds three keys the file cannot know: `runnable` (lib.all_corpora() sees the
    corpus), `n_total`, and per-language `manifest` presence, so the CLI can say
    whether the slice is actually usable rather than merely recorded.
    """
    _check_slice_id(slice_id)
    path = slice_path(slice_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no sealed slice {corpus_name(slice_id)}; "
                                f"run `python bench.py harvest {slice_id} --cutoff DATE`")
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema") != SCHEMA_SLICE:
        raise ValueError(f"{path}: unknown slice schema {record.get('schema')!r}")
    corpus = record.get("corpus", corpus_name(slice_id))
    record["runnable"] = corpus in lib.all_corpora()
    for lang, row in record.get("languages", {}).items():
        row["manifest"] = os.path.exists(lib.manifest_path(corpus, lang))
    record["n_total"] = sum(int(row["n"]) for row in record.get("languages", {}).values())
    return record


def _today() -> str:
    return datetime.datetime.now(datetime.UTC).date().isoformat()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
