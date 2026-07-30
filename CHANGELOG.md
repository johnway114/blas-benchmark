# Changelog

Corrections get an entry here, never a quiet edit. Anything that can move a
published number - the corpus contract, the prompt, the decoding contract, a
metric definition, a model pin - is a versioned change and is listed below
with what it invalidates.

Dates are the date of the change, not of the data.

## v3 - 2026-07-29

v3 is a clean edition boundary. No v1 or v2 score is comparable with v3, and
every pre-v3 receipt is refused by v3 scoring rather than migrated.

### Added - published web data contract (`bench.py export`)

- `bench.py export` renders `scores/web.json`, schema `celticbench.web.v1`:
  the edition's completeness, the panel with its pins and decoding
  deviations, the prompt template, the metric signatures, the sealed Track B
  slices including the languages with no viable publisher and their reasons,
  and every scored row.
- The export carries reading rules as data, not just numbers. Each row states
  whether its off-target rate is `authoritative`, `advisory` or `unmeasured`,
  and whether it is `rankable`. A renderer that re-derived those rules would
  eventually disagree with `LEADERBOARD.md`, and then one number would mean
  two things.
- **Integrity now gates the comparison.** A published floor - off-target
  below 1%, blank below 0.5%, copy below 2%, all lower-is-better - decides
  whether a row may be presented as the best in its cell. A row that exceeds
  it keeps its score and is marked, because chrF++ cannot distinguish a good
  translation from fluent output in the wrong language. Off-target gates only
  where the detector is authoritative for that language and corpus: on
  Tatoeba its own false-positive rate on gold Celtic text reaches 5-9%, so
  gating there would fail systems for the detector's errors. This changes no
  score; it changes which score is allowed to be called a winner.
- The export declares that v3 has no confidence intervals, one run per cell
  and no significance test, so a consumer cannot present the ordering as a
  ranking claim without contradicting the data it is rendering.

Does not invalidate anything: `export` recomputes no metric and reads
`scores/scores.json` as scoring wrote it.

### Changed - frontier-only panel

- The live roster is now exactly `gpt-5.6-sol`, `claude-opus-5`,
  `gemini-3.6-flash`, `deepseek-v4-pro`, `kimi-k3`, and `qwen3.7-max`.
  Efficient variants, dedicated translation services and local models were
  removed from the current runtime and panel.
- Every system runs all six Celtic languages in both directions with the same
  fixed prompt: 24 runs and 25,006 requests per system, 144 runs and 150,036
  requests for the complete edition. Coverage exceptions no longer produce a
  smaller system matrix.
- Model roster and pin changes are edition changes. The outgoing leaderboard
  and scores must be archived before the method version changes, and the new
  edition requires a complete rerun.

### Archived and invalidated

- The v2 leaderboard is frozen at
  `archive/LEADERBOARD-20260729T214924+0000.md` and its scores at
  `archive/scores-20260729T214924+0000.json`.
- The 24 v2 Opus-MT hypothesis/receipt pairs are historical artifacts under
  `archive/v2/out/`; they are not live v3 results.
- All v2 results are invalid under the v3 method. Nothing is carried forward:
  every published v3 score requires a new v3 run and receipt.

## v2 - 2026-07-29

The v1 numbers are not comparable with v2 numbers. Every v1 receipt is
refused by `bench.py score` (schema, corpus contract and metric definitions
all moved), and the single v1 leaderboard row - an 8-line partial Cornish
Opus-MT slice - has been withdrawn rather than migrated.

### Corrected

- **Off-target denominator.** METHODOLOGY.md defined `off_target_rate` as a
  share of *non-blank* lines; the code divided by all lines, so a system that
  returned nothing was rewarded with a falling off-target rate. The code now
  matches the published definition and every row carries `off_target_n`, the
  number of lines the detector actually judged.
- **Google Cloud Translation coverage.** The registry offered Manx to
  `google-translate-v2`. Google's current NMT language list does not contain
  Manx (or Cornish); the consumer web UI is not the API contract. Manx runs
  of that system are refused, and Alibaba is now the only registered
  commercial API whose own list contains Manx and Cornish.
- **Length-ratio diagnostic.** METHODOLOGY.md promised one; there was none.
  Track C now reports median, p10, p90 and an out-of-band rate.
- **Advisory labelling applied the wrong language's error rate.** An XX -> EN
  row was labelled `(advisory)` from the detector's false-positive rate on
  *Celtic* gold text, although the hypothesis being judged is English.
  `prepare` now measures the English side of every pair too (worst case 0.6%
  on Tatoeba, 0.0% on FLORES), and the leaderboard looks up the language the
  hypothesis was actually expected to be written in. Manx EN -> XX stays
  advisory at 27.8%; Manx XX -> EN is authoritative and no longer carries a
  caveat it never earned.
- **Retired and misnamed model pins.** `deepseek-chat` was retired by the
  vendor on 2026-07-24; `gemini-3.1-pro` was never an API model ID (the
  documented ID is preview-only); `mistral-large-latest` was a moving alias
  that would silently change model mid-edition. All three are gone.

### Changed - corpus contract (manifest v1 -> v2)

- Tatoeba eval sets gained `eval/tatoeba.{lang}.ids`: the upstream sentence
  ids for both sides of every row, hashed into the manifest. Tatoeba is
  CC-BY, so a published line has authors and must stay traceable to them.
- Manifests are `celticbench.manifest.v2` and hash the ids file. Rebuilt with
  `prepare --force`; the eval `.src`/`.ref` bytes are unchanged, so only the
  contract hash moved.
- New `manifests/corpus-qa.json`: measured defects in the corpora we publish
  against, rebuilt by `prepare` and by `bench.py qa`. First measurement found
  317 duplicate Cornish references, 150 Welsh and 112 Irish (the English side
  is deduplicated, the Celtic side is not, so XX->EN has repeated inputs
  scored against different references), 11 FLORES Irish rows containing
  U+200B, and no untranslated pairs anywhere.
- `manifests/lid-validation.json` now pins the SHA-256 of the detector
  binary. An off-target rate is only reproducible against a specific
  fastText build, and `lid.176.ftz` is a mutable URL.

### Changed - receipts and scoring (receipt v1 -> v2, scores v1 -> v2)

- Receipts additionally bind: the eval input and reference file hashes, the
  declared decoding and its hash separately from the decoding actually sent,
  any provider-forced decoding deviation, the hypothesis filename, token
  usage and request/cache counts, and the versions of Python, sacrebleu,
  torch and transformers that produced the run.
- Scoring re-verifies the eval files against the committed manifest (it
  previously trusted the manifest's own hash), and now refuses a run whose
  filename disagrees with its receipt, whose line count disagrees with its
  receipt, whose prompt or declared decoding has changed since the run, whose
  receipt contradicts itself, or whose system/language/direction the registry
  no longer offers.
- `scores.json` records the sacrebleu signatures, the detector build, the
  runtime versions, and a coverage matrix so a table can distinguish "scored
  badly" from "the vendor does not offer this language".
- Hypotheses are committed alongside receipts. A receipt proves which bytes
  were scored; without those bytes a clean checkout could only take our word
  for it.
- Partial slices are written to their own `.limitN` filename, so a smoke run
  can no longer overwrite or be mistaken for a published full run.
- The line-level cache stores the reported model, decoding and usage next to
  the text, so a rerun served entirely from cache still writes a truthful
  receipt. Pre-v2 cache lines are ignored rather than trusted.

### Added - systems

- Dedicated MT services: Google Cloud Translation LLM (a different product
  from both the NMT model and the Gemini API), DeepL, Azure AI Translator,
  Amazon Translate, and Alibaba Cloud Machine Translation.
- General-purpose models: `deepseek-v4-pro`, `deepseek-v4-flash`, `kimi-k3`,
  `qwen3.7-max`, `gemini-3.6-flash`, and `claude-haiku-4-5` repinned to its
  exact dated snapshot.
- Local open-weight models: `qwen3.5-9b` (Apache-2.0 general control),
  `translategemma-4b`, `salamandrata-7b` and `tiny-aya-water`, run through
  the same published prompt as the hosted chat systems.
- Grok is deliberately absent: xAI's terms prohibit using their service to
  benchmark it. It will be added if and when xAI permits it in writing.

### Added - harness

- Track B is real and its first slice is sealed. `trackb-2026q3` (cutoff
  2026-02-16, harvested 2026-07-29) holds 300 Irish pairs from 11 EU Official
  Journal acts, 300 Welsh pairs from 32 Welsh Government announcements, and
  122 Scottish Gaelic pairs from 10 Bòrd na Gàidhlig and MG ALBA posts. Irish
  works because the publisher's Cellar SPARQL graph dates each document and
  proves both language versions exist before anything is fetched; the daily
  listing view does neither, and was caught filing a 2026-07-29 act under
  2026-07-28. Corrigenda are excluded and no single document may contribute
  more than 40 segments. Breton, Manx and Cornish have no parallel English
  publisher at all and are recorded as unavailable with reasons.
- A dedicated translation model is now driven by its own published template
  instead of the shared benchmark prompt, and its receipt records that
  template and hash. SalamandraTA's card states it has no chat capability;
  under the shared prompt it translated the instruction into Irish before the
  sentence, which measured our misuse rather than the model. Only `open-mt`
  systems may declare a template, it must be the one the model's card
  publishes, and general-purpose models are untouched.
- `scores.json` and every receipt now carry `method_version`, and scoring
  refuses a receipt from another method version instead of ranking it. The
  leaderboard prints the version it belongs to.
- `--workers` for concurrent hosted-API runs, order preserved.
- `bench.py qa`, `bench.py leaderboard --archive`, and per-run token and
  request accounting.

## v1 - 2026-07-29

First public harness: Track A corpora, Track C integrity metrics, receipts,
fail-closed registry and scoring, and one 8-line Cornish Opus-MT smoke run.
