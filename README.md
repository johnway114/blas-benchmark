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

## What is measured

| Role | Systems |
| --- | --- |
| General-purpose, flagship | GPT-5.6 Sol, Claude Opus 5, Gemini 3.6 Flash, DeepSeek V4 Pro, Kimi K3, Qwen3.7 Max |
| General-purpose, efficient | GPT-5.6 Luna, Claude Haiku 4.5, DeepSeek V4 Flash |
| Dedicated MT services | Google Cloud Translation NMT, Google Cloud Translation LLM, DeepL, Azure AI Translator, Amazon Translate, Alibaba Cloud MT |
| Open-weight MT | Opus-MT en-cel/cel-en, MADLAD-400 3B, NLLB-200 600M, TranslateGemma 4B, SalamandraTA 7B, Tiny Aya Water |
| Open-weight general | Qwen3.5 9B |

General-purpose models are run on all six languages, because nobody publishes
a Celtic support contract for them and that is exactly the question. Dedicated
MT services are run only on the languages their own published language list
contains; the rest show as `n/a`, never as a score of zero. Only one
commercial API (Alibaba) lists Manx and Cornish at all.

## Why

Nobody measures Celtic MT seriously. WMT does not cover these languages,
FLORES covers only three of the six, and no leaderboard anywhere answers
"can this year's flagship model translate Manx?". Speakers of small languages
cannot easily check AI output themselves; this benchmark checks in public.

Measured reality this harness reproduces: the only open model that claims
Cornish support answers in **Turkish** on 77.9% of all 3402 Tatoeba lines
(`opus-mt-cel`, English -> Cornish; 4.2% are detected as Cornish, chrF++ 9.9).
That is the whole benchmark in one row: a model advertising a language it
cannot write, and nobody measuring it.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# optional, for the local open-weight models:
.venv/bin/pip install -r requirements-local.txt

cp .env.example .env          # fill in whichever credentials you have

.venv/bin/python bench.py prepare        # download corpora, build + verify eval sets
.venv/bin/python bench.py doctor         # check credentials and pinned model IDs
.venv/bin/python bench.py plan           # the full run matrix
.venv/bin/python bench.py run --all-ready
.venv/bin/python bench.py score          # verify receipts, compute metrics
.venv/bin/python bench.py leaderboard    # render LEADERBOARD.md
```

The local models (`opus-mt-cel`, `nllb-600m`, `madlad400-3b`, `qwen3.5-9b`,
`salamandrata-7b`) need no credentials at all:

```bash
.venv/bin/python bench.py run opus-mt-cel tatoeba kw en-xx
```

Because the hypotheses are committed, `prepare` followed by `score` re-derives
every published Track A number with no credentials and no model weights: the
corpora come from upstream, the outputs come from git. Track B is the
exception, and says so in METHODOLOGY.md.

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
7. **Coverage is a fact, not a score.** A dedicated MT service is only run on
   the languages its own published list contains; the rest render as `n/a`.
   Fabricating a zero for a language a vendor never offered would be a lie
   about the vendor and about the language.
8. **Corpus defects are published too.** `manifests/corpus-qa.json` counts
   the duplicates, control characters and length outliers in the corpora we
   score against, rather than quietly cleaning them.

## Licences

- Code: Apache-2.0 (`LICENSE`).
- FLORES-200: CC-BY-SA 4.0 (Meta AI). Downloaded from upstream at prepare
  time, never redistributed here.
- Tatoeba: CC-BY 2.0 FR. Same: downloaded from upstream, never redistributed.
- fastText lid.176: CC-BY-SA 3.0.
- Track B text is quoted from official publishers under their own terms
  (Welsh Government: OGL v3.0); only per-row URLs, dates and hashes are
  committed here, never the harvested text.
- Benchmark-only weights, scored as reference points and shippable by
  nobody: NLLB-200 and Tiny Aya Water (both CC-BY-NC 4.0). TranslateGemma is
  under the Gemma Terms and SalamandraTA under GPL-3.0; both are gated or
  copyleft and are flagged in every table they appear in.
