# Celtic Translation Benchmark

A published, recurring benchmark of how well AI systems translate the six
living Celtic languages:

| | Language | ISO | Reference sets |
| --- | --- | --- | --- |
| 🇮🇪 | Irish | ga | FLORES-200 devtest (n=1012), Tatoeba (n=2339), fresh harvest 2026q3 (n=300) |
| 🏴󠁧󠁢󠁷󠁬󠁳󠁿 | Welsh | cy | FLORES-200 devtest (n=1012), Tatoeba (n=1631), fresh harvest 2026q3 (n=300) |
| 🏴󠁧󠁢󠁳󠁣󠁴󠁿 | Scottish Gaelic | gd | FLORES-200 devtest (n=1012), Tatoeba (n=961), fresh harvest 2026q3 (n=122) |
| 🇫🇷 | Breton | br | Tatoeba (n=394) |
| 🇮🇲 | Manx | gv | Tatoeba (n=18, directional only) |
| 🏴 | Cornish | kw | Tatoeba (n=3402) |

Only Irish, Welsh and Scottish Gaelic have a fresh, post-training-cutoff
harvest. Nobody publishes parallel English/Breton, English/Manx or
English/Cornish text we can date: Breton public bilingualism is with French,
the Isle of Man Government has no Manx mirror, and no Cornish public body
issues a parallel stream. That absence is published with its reasons rather
than filled in.

Both directions (English -> Celtic and Celtic -> English), every system run
through the same harness, the same fixed prompt, the same frozen corpora, and
scored with the same metrics. Current results: [LEADERBOARD.md](LEADERBOARD.md).
Method and caveats: [METHODOLOGY.md](METHODOLOGY.md). What changed and what it
invalidates: [CHANGELOG.md](CHANGELOG.md).

## v3 scope

The v3 panel contains exactly six hosted frontier systems:

| System ID | Display name |
| --- | --- |
| `gpt-5.6-sol` | GPT-5.6 Sol |
| `claude-opus-5` | Claude Opus 5 |
| `gemini-3.6-flash` | Gemini 3.6 Flash |
| `deepseek-v4-pro` | DeepSeek V4 Pro |
| `kimi-k3` | Kimi K3 |
| `qwen3.7-max` | Qwen3.7 Max |

Every system runs every available corpus for all six languages in both
directions with one fixed prompt. That is 24 runs and 25,006 sentence
requests per system: 144 runs and 150,036 requests for the complete edition.

The roster is frozen for the edition. A model may be added, removed or
repinned only at an edition boundary: archive the outgoing leaderboard and
scores, bump the method version, record the change in CHANGELOG.md, and rerun
the complete panel. Results from different editions are never mixed.

## Why

Nobody measures Celtic MT seriously. WMT does not cover these languages,
FLORES covers only three of the six, and no leaderboard anywhere answers
"can this year's flagship model translate Manx?". Speakers of small languages
cannot easily check AI output themselves; this benchmark checks in public.

The benchmark makes that question measurable across all six languages,
including the three that lack FLORES coverage and the three that lack a fresh
parallel-English harvest. Sparse or imperfect public data is reported as a
caveat rather than hidden behind a narrower language roster.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add the six provider credentials

.venv/bin/python bench.py prepare        # download corpora, build + verify eval sets
.venv/bin/python bench.py doctor         # check all six credentials and hosted model pins
.venv/bin/python bench.py plan           # 144 runs, 150,036 requests
.venv/bin/python bench.py run --all-ready
.venv/bin/python bench.py score          # verify receipts, compute metrics
.venv/bin/python bench.py leaderboard    # render LEADERBOARD.md
.venv/bin/python bench.py export         # render scores/web.json, the published data contract
```

Because the hypotheses are committed, `prepare` followed by `score` re-derives
every published Track A number without credentials: the corpora come from
upstream and the outputs come from the repository. Track B is the exception,
and says so in METHODOLOGY.md.

Track B, the fresh harvest, is sealed quarterly and committed before any
model sees it:

```bash
.venv/bin/python bench.py harvest 2026q4 --cutoff 2026-02-16
.venv/bin/python bench.py qa             # measure the new slice
.venv/bin/python bench.py slice 2026q4
```

## Integrity rules

1. **We run everything ourselves.** No self-reported scores.
2. **Every registered system's result is published**, including embarrassing
   ones. Corrections get a changelog entry, never a quiet edit.
3. **Receipts or it did not happen.** Every score links a receipt binding the
   model ID reported by the API, decoding config hash, prompt hash, corpus
   manifest hash, and hypothesis hash. `score` refuses anything unverifiable.
4. **Corpora are hash-pinned.** `manifests/` commits the SHA-256 of every
   eval file; a drifted upstream download or edited local file is a hard
   error, not a silent change.
5. **Fixed prompt, fixed decoding, fixed slices for everyone** within a
   leaderboard version. The prompt is published verbatim in METHODOLOGY.md.
6. **Metrics are validated, not assumed.** The language-identification model
   behind the off-target metric is measured against the gold text itself
   (`manifests/lid-validation.json`, both sides, every corpus); where its
   false-positive rate exceeds 5% the leaderboard labels the metric
   *advisory*, and where nobody has measured it at all, *unmeasured*.
7. **Coverage is fixed within an edition.** Every one of the six systems runs
   all six Celtic languages in both directions. Missing credentials or an
   invalid hosted pin block completion; they never narrow the matrix.
8. **Corpus defects are published too.** `manifests/corpus-qa.json` counts
   the duplicates, control characters and length outliers in the corpora we
   score against, rather than quietly cleaning them.
9. **Failure modes gate the comparison.** A system that answers in English,
   answers nothing, or echoes the source still earns a respectable chrF++.
   A published floor - off-target below 1%, blank below 0.5%, copy below 2%,
   all lower-is-better - decides whether a row may be presented as the best
   in its cell. A row that exceeds it keeps its score and is marked, never
   quietly dropped. Off-target gates only where the detector is authoritative
   there, so no system is failed for the detector's own errors.
10. **Every number the website shows comes from `bench.py export`.** The
    renderer is handed the reading rules as data rather than re-deriving
    them, because two implementations of one method eventually disagree and
    then one published number means two different things.

## Licences

- Code: Apache-2.0 (`LICENSE`).
- FLORES-200: CC-BY-SA 4.0 (Meta AI). Downloaded from upstream at prepare
  time, never redistributed here.
- Tatoeba: CC-BY 2.0 FR. Same: downloaded from upstream, never redistributed.
- fastText lid.176: CC-BY-SA 3.0.
- Track B text is quoted from official publishers under their own terms
  (Welsh Government: OGL v3.0); only per-row URLs, dates and hashes are
  committed here, never the harvested text.
