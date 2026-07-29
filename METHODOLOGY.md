# Methodology

This document is the public contract for every number the benchmark
publishes. Anything not written here is not part of the method.

Method version: **v2** (2026-07-29). v1 numbers are not comparable with v2
numbers; `CHANGELOG.md` says exactly what moved and why.

## Languages and directions

Six target languages: Irish (ga), Welsh (cy), Scottish Gaelic (gd), Breton
(br), Manx (gv), Cornish (kw); English on the other side. Both directions are
scored separately:

- **EN -> XX** (into the Celtic language): the hard, quality-defining
  direction.
- **XX -> EN**: systematically easier (English is the high-resource target);
  reported because it is how most real users consume machine translation
  (reading help).

## Tracks

### Track A: static public corpora

| Corpus | Languages | n | Provenance |
| --- | --- | ---: | --- |
| FLORES-200 devtest | ga, cy, gd | 1012 each | professional human translations, CC-BY-SA 4.0 |
| Tatoeba EN-pairs | all six | 2339 / 1631 / 961 / 394 / 18 / 3402 (ga/cy/gd/br/gv/kw) | community sentences, CC-BY 2.0 FR |

Eval sets are built by `bench.py prepare` deterministically from upstream
downloads: FLORES devtest in upstream order, full set; Tatoeba pairs ordered
by target-sentence ID, deduplicated on the English side. `manifests/` pins
the SHA-256 of every built file; scoring refuses eval files that do not
match.

Tatoeba rows carry their upstream sentence ids in `eval/tatoeba.{lang}.ids`
(`<english_id>\t<native_id>`, one line per row, hashed into the manifest).
Tatoeba is CC-BY: every published line has authors, and
`https://tatoeba.org/en/sentences/show/<id>` is what makes it traceable to
them. FLORES has no per-row upstream id and therefore no ids file.

**Corpus defects are measured, not assumed.** `manifests/corpus-qa.json`
records, per corpus and language: duplicate rows on each side, untranslated
pairs, blank rows, control characters, non-NFC text, and the
reference-over-source length-ratio distribution. First measurement
(2026-07-29):

- **Duplicate references**: 317 Cornish, 150 Welsh, 112 Irish, 91 Scottish
  Gaelic, 6 Breton, 1 Manx. The English side is deduplicated and the Celtic
  side is not, so XX -> EN repeats some inputs against different references.
  Chosen deliberately: deduplicating the Celtic side would silently drop
  legitimate distinct English references.
- **11 FLORES Irish rows contain U+200B** (zero-width space). Left in place:
  the eval files must reproduce upstream bytes exactly, and a corpus we
  quietly clean is a corpus nobody else can rebuild.
- **No untranslated pairs** in any Track A corpus.

**Caveats, stated loudly:**

- Both corpora are public and old enough to be in the training data of every
  modern system. Treat Track A as a *comparability anchor*, not proof of
  generalization. FLORES rows are the trustworthy comparison; **Tatoeba rows
  are contamination-inflated** for any system trained on OPUS-family data
  (that includes the Opus-MT and NLLB anchors, and likely most hosted
  systems).
- **Manx has 18 usable pairs.** Its rows are labelled directional and are
  never ranked. That scarcity is a finding in itself.
- Tatoeba sentences are short and simple; scores there do not transfer to
  document-length translation.

### Track B: rolling fresh harvest

Bilingual text published *after* the newest benchmarked model's training
cutoff, harvested from public bodies that publish the same document in a
Celtic language and in English. Each harvest is a dated slice, becomes its
own corpus `trackb-<slice>`, and is contamination-free by construction at the
moment it is sealed.

Rules:

- **Freshness is proven, not assumed.** Every item carries a
  publisher-stated publication date strictly after the declared cutoff.
  An item whose date cannot be established is dropped, never assumed fresh.
- **Alignment is conservative.** Only structurally corresponding blocks are
  paired; pairs failing a length-ratio check, pairs whose two sides are
  identical (an untranslated fallback page), and duplicates are dropped.
- **Provenance is per row.** `eval/trackb-<slice>.{lang}.ids` carries the
  source URL and publication date of every segment; the slice manifest
  records the cutoff, the harvest date, the publishers, and the extractor
  version, and the manifest hash is committed **before** any model sees the
  slice (a public commit is a third-party-verifiable timestamp).
- **Missing sources are recorded, never filled.** A language with no viable
  publisher is listed in the slice file as unavailable with its reason. It is
  never substituted with other data.

**First slice, `trackb-2026q3`** (cutoff 2026-02-16, the newest training
cutoff any panel vendor documents; harvested 2026-07-29):

| Lang | n | Documents | Publisher | Status |
| --- | ---: | ---: | --- | --- |
| ga | 300 | 11 | EU Official Journal acts, EN and GA expressions (Publications Office) | sealed |
| cy | 300 | 32 | Welsh Government / Llywodraeth Cymru announcements (OGL v3.0) | sealed |
| gd | 122 | 10 | Bòrd na Gàidhlig and MG ALBA news | sealed, thin by nature |
| br | - | - | - | unavailable |
| gv | - | - | - | unavailable |
| kw | - | - | - | unavailable |

Irish comes from the EU Official Journal, where Irish has been a full working
language since 2022 and both language versions render from one source, so
paragraphs align exactly. Two things make it usable where the obvious route is
not: the publisher's own Cellar SPARQL graph states a publication date per
document (its daily-listing view does not - it returned the same document set
for different dates, and it filed an act published 2026-07-29 under
2026-07-28), and the same query proves an English *and* an Irish expression
exist before anything is fetched, so a missing translation can never be
discovered as a 404 and mistaken for content.

Two extra filters apply there. **Corrigenda are excluded**: a corrigendum
carries a fresh publication date but republishes older text, which is exactly
the contamination Track B exists to avoid. And **no document contributes more
than 40 segments**, because a single regulation runs to hundreds of aligned
paragraphs and would otherwise be the entire Irish slice; the cap is why 300
Irish segments span 11 documents rather than one.

The three unavailable languages are the finding, not a gap to paper over. In
full in `manifests/trackb-2026q3.slice.json`, in short: Breton public-sector
bilingualism is French-Breton, not English-Breton; the Isle of Man Government
has no Manx section to mirror; and no Cornish public body issues a parallel
English/Cornish stream. All three are absences of parallel publishing, not
absences of effort.

Sources that answer only a spoofed browser user agent are not harvested, even
where the text would be good: Sabhal Mòr Ostaig has the highest Gaelic volume
found and is excluded for exactly that reason.

Register caveat: institutional text measures formal-register translation
only, and says nothing about conversational or literary quality.

### Track C: reference-free integrity metrics

Computed on every hypothesis, immune to contamination, available even where
references barely exist:

| Metric | Definition |
| --- | --- |
| `off_target_rate` | share of **non-blank** lines confidently identified as a language other than the requested target (fastText lid.176, confidence >= 0.5). `off_target_n` reports that denominator |
| `blank_rate` | share of empty output lines |
| `copy_rate` | share of non-blank lines identical to the source after whitespace/case normalization |
| `repetition_rate` | share of lines with a repeated 4-gram loop or near-total token collapse (unique-token ratio < 0.3 over >= 8 tokens) |
| `length_ratio_*` | median, p10, p90 and out-of-band rate of hypothesis-over-source characters, band 0.5-2.0: catches truncation and runaway commentary without a reference |

Blank output is counted as blank and leaves the off-target denominator
entirely. Diluting off-target with blanks would reward a system for
answering nothing.

**Off-target reliability is measured, not assumed.** `prepare` runs the
detector over the gold text of both sides and commits per-language
false-positive rates, plus the SHA-256 of the detector binary itself, to
`manifests/lid-validation.json`. Measured on 2026-07-29 with `lid.176.ftz`
sha256 `8f3472cf…`:

| Lang | FLORES refs fp | Tatoeba refs fp | Off-target status for EN -> XX |
| --- | ---: | ---: | --- |
| ga | 0.0% | 8.5% | authoritative on FLORES, advisory on Tatoeba |
| cy | 0.0% | 9.4% | authoritative on FLORES, advisory on Tatoeba |
| gd | 0.4% | 6.6% | authoritative on FLORES, advisory on Tatoeba |
| br | - | 6.6% | advisory |
| gv | - | 27.8% | advisory (lid.176 recognizes 6% of gold Manx) |
| kw | - | 5.6% | advisory |

Rows above the 5% false-positive threshold are labelled `(advisory)` on the
leaderboard automatically. Short sentences are intrinsically hard to
identify; FLORES-length text is reliable.

The advisory decision follows the language actually being judged. An XX -> EN
hypothesis is judged against English, so it uses the English-side rate for
the same pairs (0.0% on FLORES, at worst 0.6% on Tatoeba) and is
authoritative; labelling it with the Celtic side's error rate would advertise
a caveat that does not apply to it.

Every built corpus is measured, Track B slices included: on `trackb-2026q3`
the detector recognises 97% of gold Welsh and 83% of gold Scottish Gaelic at
false-positive rates of 0.7% and 0.8%, so those rows are authoritative.
Institutional paragraphs are simply easier to identify than Tatoeba's short
sentences. A corpus/language pair with no committed measurement is labelled
`(unmeasured)` rather than silently passing as reliable.

## Reference metrics

- **chrF++** (sacrebleu, `word_order=2`), 0-100: the headline metric, robust
  for morphologically rich low-resource targets.
- **BLEU** (sacrebleu defaults): secondary, for comparability with older
  literature.
- sacrebleu's own signatures for both metrics are recorded in
  `scores/scores.json` and printed on the leaderboard, along with the Python
  and library versions that produced the run. A metric without its signature
  is not reproducible.
- **No COMET**: its encoder does not cover Cornish or Manx; a metric that
  silently drops half the matrix is worse than chrF++ everywhere.
- **No human preference panel** in the current version; nothing here claims
  human-judged quality.

## The fixed prompt

Every *general-purpose* system - hosted chat models and the local open-weight
general model alike - gets exactly this prompt, no system message,
temperature 0, top_p 1, max_tokens 2048 (generous so reasoning-style models
are not truncated):

```
Translate the following {source language} sentence into {target language}. Output only the {target language} translation as plain text: no quotes, no notes, no explanation, no alternatives.

{text}
```

No per-model prompt tuning, ever: the benchmark measures the model, not our
prompting. The local general model is greedy (`num_beams=1`,
`do_sample=false`, `max_new_tokens=256`) so a rerun of the same weights
reproduces the same bytes.

**A dedicated translation model gets its own published template instead**, for
the same reason the sequence-to-sequence anchors get their own control tokens
(NLLB forced-BOS target code, Opus-MT `>>xxx<<`, MADLAD `<2xx>`): it was never
trained to follow instructions, so handing it ours measures our misuse. This
is not a loophole for tuning - the template must be the one the model's own
card publishes, it is committed in the registry, and each receipt records the
exact template and its hash.

The case that forced the rule: SalamandraTA's card says outright that it
"lacks chat capabilities and has not been trained with any chat instructions".
Given the shared prompt it translated the instruction into Irish and then
translated the sentence, so every line began with "níl aon luachana, níl aon
nótaí" ("no quotes, no notes"). That is a measurement of the harness, not of
the model. It now receives its documented format
(`Translate the following text from {source} into {target}.` then the labelled
source and target lines) at its card's beam width.

Dedicated MT *services* are called through their native interface with no
prompt and no decoding parameters, because they expose none.

**Forced deviations are declared, not hidden.** Where a provider refuses part
of the decoding contract, the registry declares the deviation and its reason,
the adapter omits exactly those parameters, and the receipt records what was
actually sent. Current deviations:

- `gemini-3.6-flash`: Google deprecated `temperature` and `top_p`; the API
  ignores them and will reject them in future models, so they are not sent.
- `deepseek-v4-pro` and `kimi-k3` are reasoning models whose reasoning cannot
  be disabled. Their rows are flagged; they are not decoding-identical to a
  non-reasoning model and no amount of parameter-setting makes them so.

## The panel

Systems are chosen by role, before any Celtic score is seen, and frozen for
the edition:

| Role | Systems |
| --- | --- |
| General-purpose, flagship | GPT-5.6 Sol, Claude Opus 5, Gemini 3.6 Flash, DeepSeek V4 Pro, Kimi K3, Qwen3.7 Max |
| General-purpose, efficient | GPT-5.6 Luna, Claude Haiku 4.5, DeepSeek V4 Flash |
| Dedicated MT services | Google Cloud Translation NMT, Google Cloud Translation LLM, DeepL, Azure AI Translator, Amazon Translate, Alibaba Cloud MT |
| Open-weight MT | Opus-MT en-cel/cel-en, MADLAD-400 3B, NLLB-200 600M, TranslateGemma 4B, SalamandraTA 7B, Tiny Aya Water |
| Open-weight general | Qwen3.5 9B |

Coverage rules differ by role, and the difference is deliberate:

- **General-purpose models are run on all six languages.** No vendor
  publishes a Celtic support contract for them, so their coverage is exactly
  what the benchmark is for.
- **Dedicated MT services are run only on the languages in their own current
  published language list.** Anything else is refused before a request is
  made, and shows as `n/a` on the leaderboard rather than as a score of zero.
  A refusal is a coverage fact; a zero would be a fabricated measurement.
- **A registered system that has not run yet says so.** The leaderboard's
  coverage table separates `ok` (scored) from `.` (runnable, not yet run) from
  `n/a` (not offered). Registration is a commitment to run a system, not a
  claim to have run it. Two reasons a cell currently sits at `.`: no
  credentials, or local weights whose throughput makes a full pass
  impractical on the machine at hand (a 7B causal model takes about 90 seconds
  per line here, so one FLORES direction pair is roughly 50 hours). Neither is
  ever resolved by publishing a partial row.

Current dedicated-MT coverage: Alibaba ga/cy/br/gv/kw (the only commercial
API whose list contains Manx and Cornish), Google NMT ga/cy/gd/br, DeepL
ga/cy/br, Google Translation LLM cy with ga/gd experimental, Azure ga/cy,
Amazon ga/cy. Alibaba's `sco` is Scots, a different language, and is never
substituted for Scottish Gaelic.

Google appears three times on purpose, as three different products: the
Gemini API (a general model prompted to translate), Cloud Translation NMT (a
dedicated MT model), and Cloud Translation LLM (a translation-tuned model
inside Cloud Translation Advanced). They are not interchangeable and are
never merged into one row.

**Grok is absent by exclusion, not oversight.** xAI's terms prohibit using
the service to benchmark it. It will be added if xAI permits it in writing.

## Systems and pins

The registry (`celticbench/registry.py`) is fail-closed: unregistered
systems and unsupported language/direction combinations refuse to run.
Hosted model IDs carry a pin status (`verified` / `provisional` / `alias`);
`bench.py doctor` validates every pin against the provider's live model list
and suggests corrections. Moving aliases (`*-latest`) are not accepted as
pins: an alias that changes mid-edition produces two incomparable rows with
the same name. Receipts record the model ID the API *reported*, not just the
one requested, and a run where the vendor reported more than one model ID is
flagged rather than averaged.

Local systems are pinned to immutable Hugging Face commit SHAs and never
change; they anchor the table across years while hosted models churn. Gated
weights (TranslateGemma, Tiny Aya) are downloaded only with a token whose
holder has accepted the licence; the harness never fetches them anonymously.

A system flagged `benchmark-only` (NLLB-200 and Tiny Aya, both CC-BY-NC) is
scored as a reference point and is not usable in any product; the flag is
printed in every table it appears in.

## Receipts

Every hypothesis file has a sibling receipt binding: system, provider,
requested and reported model ID (and every variant reported during the run),
revision, pin status, licence, corpus + language + direction, n, slice limit,
failed-line count, prompt template and its SHA-256, the declared decoding and
its SHA-256, the decoding actually sent and its SHA-256, any forced
deviations, the corpus manifest contract hash, the SHA-256 of both eval files
as they were at run time, the hypothesis filename and SHA-256, token and
request usage, the timestamp, the harness git version, and the Python and
library versions.

`bench.py score` re-verifies all of it and excludes - with a printed reason -
any run whose hypothesis bytes, filename, line count, eval files, corpus
manifest, prompt, or decoding contract no longer match what the receipt
claims, or whose combination the registry no longer offers. The leaderboard
lists excluded runs. There is no path to a published number without a
verifiable receipt.

Hypotheses are committed alongside their receipts. A receipt proves which
bytes were scored; without those bytes, a clean checkout can only take our
word for it.

## Failure handling

A line that still fails after 4 attempts (with backoff) becomes an empty
hypothesis line, counted in `blank_rate` and in the receipt's `fails` -
never dropped, never retried into a different run. Runs are resumable via a
line-level cache keyed on (model, prompt hash, decoding, direction,
language, text); the cache never crosses systems or prompts, and it stores
the reported model, decoding and usage next to the text so a fully cached
rerun still writes a truthful receipt.

Partial slices (`--limit N`) are written to their own filename and marked
`partial slice`. They are engineering smoke tests; a partial row is never a
comparable leaderboard row.

## Reproducing

Track A is fully reproducible from a clean checkout: `prepare` downloads the
same upstream archives, rebuilds byte-identical eval files (hash-checked
against committed manifests), and `run`/`score`/`leaderboard` regenerate
every published number for any system you hold credentials for. Because the
hypotheses are committed, `prepare` plus `score` re-derives every published
Track A metric from committed bytes without holding a single credential.

**Track B is the honest exception.** Its text is quoted from publishers under
their own terms and is not redistributed here, and a re-harvest cannot
recover it: public bodies edit and withdraw pages, so the same command a year
later returns a different slice. What is committed is the contract - per-row
source URL and publication date, file hashes, cutoff, harvest date, extractor
version - which lets anyone re-fetch what is still online, verify any copy
they hold against the hashes, and see exactly which pages a number came from.
A slice that cannot be rebuilt is retired rather than quietly re-scored.
