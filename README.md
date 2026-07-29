# Celtic Translation Benchmark

A published, recurring benchmark of how well AI systems translate the six
living Celtic languages:

| | Language | ISO | Reference sets |
| --- | --- | --- | --- |
| 🇮🇪 | Irish | ga | FLORES-200 devtest (n=1012), Tatoeba (n=2339) |
| 🏴󠁧󠁢󠁷󠁬󠁳󠁿 | Welsh | cy | FLORES-200 devtest (n=1012), Tatoeba (n=1631) |
| 🏴󠁧󠁢󠁳󠁣󠁴󠁿 | Scottish Gaelic | gd | FLORES-200 devtest (n=1012), Tatoeba (n=961) |
| 🇫🇷 | Breton | br | Tatoeba (n=394) |
| 🇮🇲 | Manx | gv | Tatoeba (n=18, directional only) |
| 🏴 | Cornish | kw | Tatoeba (n=3402) |

Both directions (English -> Celtic and Celtic -> English), every system run
through the same harness, the same fixed prompt, the same frozen corpora, and
scored with the same metrics. Current results: [LEADERBOARD.md](LEADERBOARD.md).
Method and caveats: [METHODOLOGY.md](METHODOLOGY.md).

## Why

Nobody measures Celtic MT seriously. WMT does not cover these languages,
FLORES covers only three of the six, and no leaderboard anywhere answers
"can this year's flagship model translate Manx?". Speakers of small languages
cannot easily check AI output themselves; this benchmark checks in public.

Measured reality this harness has already reproduced: the only open model
that claims Cornish support answers in **Turkish** 75% of the time
(`opus-mt-cel`, see LEADERBOARD.md).

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# optional, for the local open-weight anchor systems:
.venv/bin/pip install -r requirements-local.txt

cp .env.example .env          # fill in whichever API keys you have

.venv/bin/python bench.py prepare        # download corpora, build + verify eval sets
.venv/bin/python bench.py doctor         # check keys and pinned model IDs
.venv/bin/python bench.py plan           # the full run matrix
.venv/bin/python bench.py run --all-ready
.venv/bin/python bench.py score          # verify receipts, compute metrics
.venv/bin/python bench.py leaderboard    # render LEADERBOARD.md
```

The local anchors (`opus-mt-cel`, `nllb-600m`, `madlad400-3b`) need no
credentials at all:

```bash
.venv/bin/python bench.py run opus-mt-cel tatoeba kw en-xx
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
   behind the off-target metric is measured against the gold references
   themselves (`manifests/lid-validation.json`); where its false-positive
   rate exceeds 5%, the leaderboard labels the metric *advisory*.

## Licences

- Code: Apache-2.0 (`LICENSE`).
- FLORES-200: CC-BY-SA 4.0 (Meta AI). Downloaded from upstream at prepare
  time, never redistributed here.
- Tatoeba: CC-BY 2.0 FR. Same: downloaded from upstream, never redistributed.
- fastText lid.176: CC-BY-SA 3.0.
- NLLB-200 weights are CC-BY-NC 4.0: scored as a reference anchor,
  flagged `benchmark-only` in every table, usable in no product.
