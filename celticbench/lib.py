"""Core shared state: languages, paths, hashing, eval IO, manifests.

Nothing in this module imports anything heavier than the standard library.
Every downstream module (registry, providers, runner, scoring) builds on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from typing import Any

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
EVAL = os.path.join(HERE, "eval")
OUT = os.path.join(HERE, "out")
CACHE_DIR = os.path.join(OUT, "cache")
MANIFESTS = os.path.join(HERE, "manifests")
SCORES = os.path.join(HERE, "scores")

SCHEMA_MANIFEST = "celticbench.manifest.v2"
SCHEMA_RECEIPT = "celticbench.receipt.v2"
SCHEMA_SCORES = "celticbench.scores.v2"
SCHEMA_QA = "celticbench.corpus-qa.v1"
SCHEMA_SLICE = "celticbench.trackb-slice.v1"

# The published method version. Everything that decides whether two rows are
# comparable -- corpora, prompt, decoding, metric definitions -- is frozen
# inside one version, so a receipt from another version is refused rather than
# quietly ranked against this one. Bumping it means re-running every system;
# CHANGELOG.md says what each bump invalidated.
METHOD_VERSION = "v3"

# Canonical language table. Keys are ISO 639-1. Values retain only language
# names, corpus identifiers, and the language-ID code used by scoring/QA.
LANGS: dict[str, dict[str, Any]] = {
    "ga": {"name": "Irish", "flores": "gle_Latn", "tatoeba": "gle", "lid": "ga"},
    "cy": {"name": "Welsh", "flores": "cym_Latn", "tatoeba": "cym", "lid": "cy"},
    "gd": {"name": "Scottish Gaelic", "flores": "gla_Latn", "tatoeba": "gla", "lid": "gd"},
    "br": {"name": "Breton", "flores": None, "tatoeba": "bre", "lid": "br"},
    "gv": {"name": "Manx", "flores": None, "tatoeba": "glv", "lid": "gv"},
    "kw": {"name": "Cornish", "flores": None, "tatoeba": "cor", "lid": "kw"},
}
ALL_LANGS = tuple(LANGS)
DIRECTIONS = ("en-xx", "xx-en")

# Track A corpora: static public corpora, identical for every edition.
# Track B is a rolling fresh harvest; each sealed slice is its own corpus
# named trackb-<slice> and is discovered from its committed manifest, so a
# new slice needs no code change. Track C (reference-free integrity metrics)
# is computed on every hypothesis regardless of corpus.
CORPORA = ("flores", "tatoeba")
TRACK_B_PREFIX = "trackb-"


def all_corpora() -> tuple[str, ...]:
    """Track A corpora plus every Track B slice that has a committed manifest."""
    slices: set[str] = set()
    if os.path.isdir(MANIFESTS):
        for name in os.listdir(MANIFESTS):
            if name.startswith(TRACK_B_PREFIX) and name.endswith(".json"):
                slices.add(name.split(".", 1)[0])
    return CORPORA + tuple(sorted(slices))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_lines(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.replace("\n", " ") + "\n")


def harness_version() -> str:
    """Git describe of the harness itself, recorded in every receipt."""
    try:
        out = subprocess.run(
            ["git", "-C", HERE, "describe", "--always", "--dirty", "--tags"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def runtime_versions() -> dict[str, str]:
    """Versions of Python and sacrebleu, the runtime inputs to published scores."""
    import platform

    versions = {"python": platform.python_version()}
    try:
        versions["sacrebleu"] = __import__("sacrebleu").__version__
    except Exception:
        pass
    return versions


def load_env(path: str | None = None) -> None:
    """Tiny .env loader; never overrides variables already in the environment."""
    env_path = path or os.path.join(HERE, ".env")
    if not os.path.exists(env_path):
        return
    for line in read_lines(env_path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Eval file layout
#
# eval/{corpus}.{lang}.src   English side
# eval/{corpus}.{lang}.ref   Celtic side
# eval/{corpus}.{lang}.ids   upstream row ids, one per line (attribution)
#
# Direction decides which file is the model input:
#   en-xx: input = .src, references = .ref
#   xx-en: input = .ref, references = .src
# ---------------------------------------------------------------------------

def eval_src_path(corpus: str, lang: str) -> str:
    return os.path.join(EVAL, f"{corpus}.{lang}.src")


def eval_ref_path(corpus: str, lang: str) -> str:
    return os.path.join(EVAL, f"{corpus}.{lang}.ref")


def eval_ids_path(corpus: str, lang: str) -> str:
    """Upstream row identifiers, parallel to .src/.ref.

    Tatoeba sentences are CC-BY: the ids are what makes a published line
    traceable back to its author. Track B slices use them for the source URL
    of each segment. FLORES has no per-row id, so it has no ids file.
    """
    return os.path.join(EVAL, f"{corpus}.{lang}.ids")


def direction_io(corpus: str, lang: str, direction: str) -> tuple[str, str]:
    """(model_input_path, reference_path) for a corpus/lang/direction."""
    if direction == "en-xx":
        return eval_src_path(corpus, lang), eval_ref_path(corpus, lang)
    if direction == "xx-en":
        return eval_ref_path(corpus, lang), eval_src_path(corpus, lang)
    raise ValueError(f"unknown direction {direction!r}")


def lang_pair_names(lang: str, direction: str) -> tuple[str, str]:
    """(source_language_name, target_language_name) for prompts."""
    native = LANGS[lang]["name"]
    return ("English", native) if direction == "en-xx" else (native, "English")


def expected_target_lang(lang: str, direction: str) -> str:
    """ISO 639-1 code the hypothesis is expected to be written in."""
    return lang if direction == "en-xx" else "en"


def hyp_path(corpus: str, lang: str, direction: str, system_id: str,
             limit: int | None = None) -> str:
    """Output path for one run.

    A partial slice gets its own filename. A `--limit 8` smoke run must never
    overwrite (or be mistaken for) the published full run of the same
    combination, and both must be able to exist at once.
    """
    suffix = f".limit{limit}" if limit else ""
    return os.path.join(OUT, f"{corpus}.{lang}.{direction}.{system_id}{suffix}.hyp")


_HYP_NAME_RE = re.compile(
    r"^(?P<corpus>[a-z0-9-]+)\.(?P<lang>[a-z]{2})\.(?P<direction>en-xx|xx-en)"
    r"\.(?P<system>.+?)(?:\.limit(?P<limit>\d+))?\.hyp$"
)


def parse_hyp_name(name: str) -> dict[str, Any] | None:
    """Filename -> {corpus, lang, direction, system, limit} or None."""
    match = _HYP_NAME_RE.match(name)
    if not match:
        return None
    parsed = match.groupdict()
    parsed["limit"] = int(parsed["limit"]) if parsed["limit"] else None
    return parsed


def receipt_path(hypothesis_path: str) -> str:
    return hypothesis_path + ".receipt.json"


# ---------------------------------------------------------------------------
# Corpus manifests
#
# manifests/{corpus}.{lang}.json is committed; eval/ files are not. The
# manifest pins n and the SHA-256 of both eval files plus upstream
# provenance, GOLD-SETS style: a drifted or locally edited eval file is a
# hard error everywhere downstream, never a silent change. Manifests are
# deterministic (no timestamps).
# ---------------------------------------------------------------------------

def manifest_path(corpus: str, lang: str) -> str:
    return os.path.join(MANIFESTS, f"{corpus}.{lang}.json")


def load_manifest(corpus: str, lang: str) -> dict[str, Any]:
    path = manifest_path(corpus, lang)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no committed manifest for {corpus}.{lang}; run `python bench.py prepare` "
            f"(first build) or restore manifests/ from git"
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def manifest_contract_hash(manifest: dict[str, Any]) -> str:
    payload = {k: v for k, v in manifest.items() if k != "contract_sha256"}
    return sha256_json(payload)


def verify_manifest(corpus: str, lang: str) -> dict[str, Any]:
    """Fail-closed check that eval files match the committed manifest."""
    manifest = load_manifest(corpus, lang)
    if manifest.get("schema") != SCHEMA_MANIFEST:
        raise ValueError(f"{corpus}.{lang}: unknown manifest schema {manifest.get('schema')!r}")
    if manifest_contract_hash(manifest) != manifest.get("contract_sha256"):
        raise ValueError(f"{corpus}.{lang}: manifest contract hash mismatch (manifest edited?)")
    src, ref = eval_src_path(corpus, lang), eval_ref_path(corpus, lang)
    checks = [(src, "src_sha256"), (ref, "ref_sha256")]
    if manifest.get("ids_sha256"):
        checks.append((eval_ids_path(corpus, lang), "ids_sha256"))
    for path, key in checks:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing; run `python bench.py prepare` to rebuild eval files "
                f"(they must reproduce the committed manifest hashes)"
            )
        actual = file_sha256(path)
        if actual != manifest[key]:
            raise ValueError(
                f"{corpus}.{lang}: {os.path.basename(path)} sha256 {actual[:12]}… does not "
                f"match committed manifest {manifest[key][:12]}…; refusing to proceed"
            )
    n = len(read_lines(src))
    if n != manifest["n"] or n != len(read_lines(ref)):
        raise ValueError(f"{corpus}.{lang}: line count mismatch vs manifest n={manifest['n']}")
    if manifest.get("ids_sha256") and len(read_lines(eval_ids_path(corpus, lang))) != n:
        raise ValueError(f"{corpus}.{lang}: ids file line count does not match n={n}")
    return manifest


def write_manifest(corpus: str, lang: str, n: int, provenance: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Create or verify-against the committed manifest for freshly built eval files.

    If a manifest already exists and the rebuilt files do not match it, this
    raises unless force=True (explicit, deliberate corpus change).
    """
    manifest = {
        "schema": SCHEMA_MANIFEST,
        "corpus": corpus,
        "lang": lang,
        "language_name": LANGS[lang]["name"],
        "n": n,
        "src_sha256": file_sha256(eval_src_path(corpus, lang)),
        "ref_sha256": file_sha256(eval_ref_path(corpus, lang)),
        "provenance": provenance,
    }
    ids_file = eval_ids_path(corpus, lang)
    if os.path.exists(ids_file):
        manifest["ids_sha256"] = file_sha256(ids_file)
    manifest["contract_sha256"] = manifest_contract_hash(manifest)
    path = manifest_path(corpus, lang)
    if os.path.exists(path) and not force:
        existing = load_manifest(corpus, lang)
        if existing.get("contract_sha256") != manifest["contract_sha256"]:
            raise ValueError(
                f"{corpus}.{lang}: rebuilt eval files do not match the committed manifest. "
                f"Upstream data drifted or local build changed. Re-run with --force only "
                f"if the corpus change is deliberate and will be published as such."
            )
        return existing
    os.makedirs(MANIFESTS, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(canonical_json(manifest) + "\n")
    return manifest
