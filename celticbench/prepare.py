"""Build Track A eval sets from upstream sources, pin them in manifests.

Downloads (cached under data/, gitignored, never redistributed):
  FLORES-200      https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
  Tatoeba         https://downloads.tatoeba.org/exports/per_language/...
  fastText lid    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

Outputs:
  eval/flores.{ga,cy,gd}.{src,ref}     full devtest, n=1012
  eval/tatoeba.{all six}.{src,ref}     every EN-paired sentence, deduped
  eval/tatoeba.{all six}.ids           attribution ids, one line per eval row
  manifests/{corpus}.{lang}.json       committed hash contracts
  manifests/lid-validation.json        langid reliability per language, measured on references
  manifests/corpus-qa.json             measured corpus defects (celticbench/corpus_qa.py)

The .ids file exists because Tatoeba is CC-BY: a published line has authors,
and the sentence id is what makes it traceable to them. Format, one line per
row of the parallel .src/.ref files, in that column order:

  <english_sentence_id> [tab] <native_sentence_id>

Both are upstream Tatoeba sentence ids as published in the per-language
`{iso3}_sentences.tsv.bz2` (`Sentence id [tab] Lang [tab] Text`) and
`{iso3}-eng_links.tsv.bz2` (`Sentence id [tab] Translation id`) exports; see
https://tatoeba.org/en/downloads. Each id resolves to
https://tatoeba.org/en/sentences/show/<id>, which names the contributor.
FLORES has no per-row upstream id, so it gets no .ids file.

Deterministic: Tatoeba pairs are ordered by (target sentence id) ascending and
deduplicated on the English side, so a rebuild from identical upstream bytes
reproduces identical eval files. If upstream drifted, write_manifest refuses
unless --force (a deliberate, published corpus change).
"""
from __future__ import annotations

import bz2
import csv
import os
import tarfile
from typing import Any, Iterator

import certifi
import requests

from .corpus_qa import build_corpus_qa, summary_lines
from .lib import (
    DATA, EVAL, LANGS, MANIFESTS, all_corpora, canonical_json, eval_ids_path,
    eval_ref_path, eval_src_path, file_sha256, read_lines, write_lines, write_manifest,
)

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
TATOEBA_BASE = "https://downloads.tatoeba.org/exports/per_language"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
LID_PATH = os.path.join(DATA, "lid.176.ftz")

FLORES_TAR = os.path.join(DATA, "flores200_dataset.tar.gz")
FLORES_DIR = os.path.join(DATA, "flores200_dataset")


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120, verify=certifi.where()) as response:
        response.raise_for_status()
        tmp = dest + ".part"
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
        os.replace(tmp, dest)
    print(f"  saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def build_flores(force: bool = False) -> None:
    _download(FLORES_URL, FLORES_TAR)
    tar_sha = file_sha256(FLORES_TAR)
    if not os.path.isdir(FLORES_DIR):
        print("  extracting flores200_dataset")
        with tarfile.open(FLORES_TAR, "r:gz") as tar:
            tar.extractall(DATA, filter="data")
    eng_path = os.path.join(FLORES_DIR, "devtest", "eng_Latn.devtest")
    eng = read_lines(eng_path)
    for lang, meta in LANGS.items():
        if not meta["flores"]:
            continue
        ref_upstream = os.path.join(FLORES_DIR, "devtest", f"{meta['flores']}.devtest")
        ref = read_lines(ref_upstream)
        if len(eng) != len(ref):
            raise SystemExit(f"flores {lang}: length mismatch {len(eng)} vs {len(ref)}")
        write_lines(eval_src_path("flores", lang), eng)
        write_lines(eval_ref_path("flores", lang), ref)
        provenance = {
            "dataset": "FLORES-200",
            "subset": "devtest",
            "license": "CC-BY-SA-4.0",
            "url": FLORES_URL,
            "archive_sha256": tar_sha,
            "source_member": "devtest/eng_Latn.devtest",
            "source_member_sha256": file_sha256(eng_path),
            "reference_member": f"devtest/{meta['flores']}.devtest",
            "reference_member_sha256": file_sha256(ref_upstream),
            "selection": "full devtest, upstream order",
        }
        write_manifest("flores", lang, len(eng), provenance, force=force)
        print(f"flores {lang}: n={len(eng)}")


def _tatoeba_file(name: str) -> str:
    return os.path.join(DATA, "tatoeba", name)


def _iter_tsv_bz2(path: str) -> Iterator[list[str]]:
    with bz2.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row:
                yield row


def build_tatoeba(force: bool = False) -> None:
    eng_archive = _tatoeba_file("eng_sentences.tsv.bz2")
    _download(f"{TATOEBA_BASE}/eng/eng_sentences.tsv.bz2", eng_archive)

    plans: dict[str, dict[str, Any]] = {}
    needed_eng_ids: set[str] = set()
    for lang, meta in LANGS.items():
        iso3 = meta["tatoeba"]
        sentences_archive = _tatoeba_file(f"{iso3}_sentences.tsv.bz2")
        links_archive = _tatoeba_file(f"{iso3}-eng_links.tsv.bz2")
        _download(f"{TATOEBA_BASE}/{iso3}/{iso3}_sentences.tsv.bz2", sentences_archive)
        _download(f"{TATOEBA_BASE}/{iso3}/{iso3}-eng_links.tsv.bz2", links_archive)
        native = {row[0]: row[2] for row in _iter_tsv_bz2(sentences_archive) if len(row) >= 3}
        links: list[tuple[int, str, str]] = []
        for row in _iter_tsv_bz2(links_archive):
            if len(row) >= 2 and row[0] in native:
                links.append((int(row[0]), row[0], row[1]))
                needed_eng_ids.add(row[1])
        links.sort()
        plans[lang] = {
            "native": native, "links": links,
            "archives": {
                "sentences": (f"{iso3}_sentences.tsv.bz2", file_sha256(sentences_archive)),
                "links": (f"{iso3}-eng_links.tsv.bz2", file_sha256(links_archive)),
            },
        }

    print(f"  scanning English sentences for {len(needed_eng_ids)} linked ids")
    english: dict[str, str] = {}
    for row in _iter_tsv_bz2(eng_archive):
        if len(row) >= 3 and row[0] in needed_eng_ids:
            english[row[0]] = row[2]

    eng_sha = file_sha256(eng_archive)
    for lang, plan in plans.items():
        seen: set[str] = set()
        src: list[str] = []
        ref: list[str] = []
        ids: list[str] = []
        for _, native_id, eng_id in plan["links"]:
            eng_text = english.get(eng_id, "").strip()
            native_text = plan["native"].get(native_id, "").strip()
            if not eng_text or not native_text or eng_text in seen:
                continue
            seen.add(eng_text)
            src.append(eng_text)
            ref.append(native_text)
            # CC-BY attribution: the row stays traceable to its contributors
            # via https://tatoeba.org/en/sentences/show/<id> on either side.
            ids.append(f"{eng_id}\t{native_id}")
        write_lines(eval_src_path("tatoeba", lang), src)
        write_lines(eval_ref_path("tatoeba", lang), ref)
        write_lines(eval_ids_path("tatoeba", lang), ids)
        archives = plan["archives"]
        provenance = {
            "dataset": "Tatoeba per-language exports",
            "license": "CC-BY-2.0-FR",
            "url_base": TATOEBA_BASE,
            "sentences_archive": archives["sentences"][0],
            "sentences_archive_sha256": archives["sentences"][1],
            "links_archive": archives["links"][0],
            "links_archive_sha256": archives["links"][1],
            "english_archive": "eng_sentences.tsv.bz2",
            "english_archive_sha256": eng_sha,
            "selection": "all EN-paired sentences, ordered by target sentence id, deduplicated on English side",
            "ids_format": "<english_sentence_id>\\t<native_sentence_id>, upstream Tatoeba sentence ids, one line per eval row",
            "ids_permalink": "https://tatoeba.org/en/sentences/show/{id}",
        }
        write_manifest("tatoeba", lang, len(src), provenance, force=force)
        print(f"tatoeba {lang}: n={len(src)}")


def _lid_side_stats(lines: list[str], expected: str) -> dict[str, Any]:
    from .langid import detect, is_off_target

    hits = sum(1 for line in lines if detect(line)[0] == expected)
    false_positives = sum(1 for line in lines if is_off_target(line, expected))
    return {
        "n": len(lines),
        "recognized": round(hits / len(lines), 4),
        "false_positive_rate": round(false_positives / len(lines), 4),
    }


def build_lid_validation() -> None:
    """Measure langid reliability on the gold text itself, both sides.

    Two numbers per side:
      recognized           share of gold lines detected as the right language
      false_positive_rate  share of gold lines the off-target metric would
                           wrongly flag (confidently detected as ANOTHER
                           language) -- this number decides whether
                           off-target is authoritative or advisory there

    The English side is measured too, and separately per language pair: an
    XX -> EN hypothesis is judged against English, so labelling it with the
    detector's Manx false-positive rate would describe the wrong measurement.
    """
    from .langid import CONFIDENCE_THRESHOLD, ensure_model

    # The off-target metric is only reproducible against a specific detector
    # build: lid.176.ftz is a mutable URL, so pin the bytes that produced
    # these rates, not just the filename.
    model_path = ensure_model()
    report: dict[str, Any] = {
        "model": os.path.basename(model_path),
        "model_sha256": file_sha256(model_path),
        "model_url": LID_URL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "advisory_false_positive_threshold": 0.05,
        "languages": {},
        "english": {},
    }
    # Every built corpus, Track B slices included: an unmeasured corpus would
    # otherwise publish off-target rates with no known false-positive floor.
    for corpus in all_corpora():
        for lang, meta in LANGS.items():
            ref_file = eval_ref_path(corpus, lang)
            if not os.path.exists(ref_file):
                continue
            celtic = [line for line in read_lines(ref_file) if line.strip()]
            english = [line for line in read_lines(eval_src_path(corpus, lang)) if line.strip()]
            if not celtic or not english:
                continue
            report["languages"].setdefault(lang, {})[corpus] = _lid_side_stats(celtic, meta["lid"])
            report["english"].setdefault(corpus, {})[lang] = _lid_side_stats(english, "en")
    path = os.path.join(MANIFESTS, "lid-validation.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(canonical_json(report) + "\n")
    for lang, by_corpus in report["languages"].items():
        summary = ", ".join(
            f"{c}: rec {v['recognized']:.0%} fp {v['false_positive_rate']:.1%}"
            for c, v in by_corpus.items()
        )
        print(f"lid {lang}: {summary}")
    for corpus, by_lang in report["english"].items():
        worst = max(by_lang.values(), key=lambda v: v["false_positive_rate"])
        print(f"lid en on {corpus}: worst fp {worst['false_positive_rate']:.1%}")


def prepare(force: bool = False) -> None:
    os.makedirs(EVAL, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    print("== FLORES-200 ==")
    build_flores(force=force)
    print("== Tatoeba ==")
    build_tatoeba(force=force)
    print("== langid model ==")
    _download(LID_URL, LID_PATH)
    print("== langid validation on references ==")
    build_lid_validation()
    print("== corpus QA ==")
    for line in summary_lines(build_corpus_qa()):
        print(line)
    print("prepare: done")
