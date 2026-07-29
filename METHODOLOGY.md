# Methodology

This document is the public contract for every number the benchmark
publishes. Anything not written here is not part of the method.

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

### Track B: rolling fresh harvest (design; activates with its first slice)

Quarterly slices of bilingual text published *after* the newest benchmarked
model's training cutoff, harvested from public bodies that publish in a
Celtic language and English. Per slice: per-row source URL, bytes, hash,
licence, and publication date are recorded in a manifest; the manifest hash
is committed and published **before** any model sees the slice (a public
commit is a third-party-verifiable timestamp); after use the slice is
retired and published for audit, and is never reused. Register caveat:
institutional text measures formal-register translation only.

### Track C: reference-free integrity metrics

Computed on every hypothesis, immune to contamination, available even where
references barely exist:

| Metric | Definition |
| --- | --- |
| `off_target_rate` | share of non-blank lines confidently identified as a language other than the requested target (fastText lid.176, confidence >= 0.5) |
| `blank_rate` | share of empty output lines |
| `copy_rate` | share of non-blank lines identical to the source after whitespace/case normalization |
| `repetition_rate` | share of lines with a repeated 4-gram loop or near-total token collapse (unique-token ratio < 0.3 over >= 8 tokens) |

**Off-target reliability is measured, not assumed.** `prepare` runs the
detector over the gold references and commits per-language false-positive
rates to `manifests/lid-validation.json`. Measured on 2026-07-29:

| Lang | FLORES refs fp | Tatoeba refs fp | Off-target status |
| --- | ---: | ---: | --- |
| ga | 0.0% | 8.5% | authoritative on FLORES, advisory on Tatoeba |
| cy | 0.0% | 9.4% | authoritative on FLORES, advisory on Tatoeba |
| gd | 0.4% | 6.6% | authoritative on FLORES, advisory on Tatoeba |
| br | - | 6.6% | advisory |
| gv | - | 27.8% | advisory (lid.176 barely recognizes Manx) |
| kw | - | 5.6% | advisory |

Rows above the 5% false-positive threshold are labelled `(advisory)` on the
leaderboard automatically. Short sentences are intrinsically hard to
identify; FLORES-length text is reliable.

## Reference metrics

- **chrF++** (sacrebleu, `word_order=2`), 0-100: the headline metric, robust
  for morphologically rich low-resource targets.
- **BLEU** (sacrebleu defaults): secondary, for comparability with older
  literature.
- **No COMET**: its encoder does not cover Cornish or Manx; a metric that
  silently drops half the matrix is worse than chrF++ everywhere.
- **No human preference panel** in the current version; nothing here claims
  human-judged quality.

## The fixed prompt

Every chat system gets exactly this prompt, no system message, temperature
0, top_p 1, max_tokens 2048 (generous so reasoning-style models are not
truncated; receipts record any provider-forced deviation):

```
Translate the following {source language} sentence into {target language}. Output only the {target language} translation as plain text: no quotes, no notes, no explanation, no alternatives.

{text}
```

No per-model prompt tuning, ever: the benchmark measures the model, not our
prompting. Dedicated MT APIs (Google Cloud Translation) are called through
their native interface with no prompt. Local anchor systems use their
published inference templates (NLLB forced-BOS target code; Opus-MT
`>>xxx<<` target token; MADLAD `<2xx>` prefix) at their pinned revisions.

## Systems and pins

The registry (`celticbench/registry.py`) is fail-closed: unregistered
systems and unsupported language/direction combinations refuse to run.
Hosted model IDs carry a pin status (`verified` / `provisional` / `alias`);
`bench.py doctor` validates every pin against the provider's live model list
and suggests corrections. Receipts record the model ID the API *reported*,
not just the one requested. Local anchors are pinned to immutable Hugging
Face revisions and never change; they anchor the table across years while
hosted models churn.

A system flagged `benchmark-only` (currently NLLB-200, CC-BY-NC) is scored
as a reference point and is not usable in any product; the flag is printed
in every table it appears in.

## Receipts

Every hypothesis file has a sibling receipt binding: system, provider,
requested and reported model ID, revision (local anchors), pin status,
licence, corpus + language + direction, n, failed-line count, prompt
template and its SHA-256, decoding config and its SHA-256, the corpus
manifest contract hash, the hypothesis file SHA-256, timestamp, and the
harness git version. `bench.py score` re-verifies all of it and excludes -
with a printed reason - anything that fails. The leaderboard lists excluded
runs. There is no path to a published number without a verifiable receipt.

## Failure handling

A line that still fails after 4 attempts (with backoff) becomes an empty
hypothesis line, counted in `blank_rate` and in the receipt's `fails` -
never dropped, never retried into a different run. Runs are resumable via a
line-level cache keyed on (model, prompt hash, decoding, direction,
language, text); the cache never crosses systems or prompts.

## Reproducing

Track A is fully reproducible from a clean checkout: `prepare` downloads the
same upstream archives, rebuilds byte-identical eval files (hash-checked
against committed manifests), and `run`/`score`/`leaderboard` regenerate
every published number for any system you hold credentials for. Receipts for
our published runs live beside the scores.
