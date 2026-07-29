# Celtic Translation Benchmark - leaderboard

Method **v2**, generated from `scores/scores.json` (2026-07-29T20:11:39+00:00). Every row is backed by a verified receipt in `out/`; excluded runs are listed at the bottom with reasons. Rows from another method version are refused, not ranked: `METHODOLOGY.md`, `CHANGELOG.md`.

- Metrics: `chrF2++|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`, `BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- Off-target detector: `lid.176.ftz` sha256 `8f3472cfe873`, confidence >= 0.5
- Runtime: python 3.14.3, sacrebleu 2.6.0, torch 2.13.0, transformers 5.14.1

## English -> Celtic

### Irish (ga)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | flores | 1012 | 50.1 | 23.7 | 0.0% | 0.0% | 0.0% |  |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 2339 | 57.0 | 35.2 | 9.1% (advisory) | 0.0% | 0.5% |  |

### Welsh (cy)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | flores | 1012 | 50.3 | 25.6 | 0.2% | 0.0% | 0.0% |  |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 1631 | 54.9 | 32.7 | 9.6% (advisory) | 0.0% | 0.9% |  |

### Scottish Gaelic (gd)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | flores | 1012 | 29.1 | 5.7 | 1.7% | 0.0% | 0.0% |  |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 961 | 30.0 | 7.9 | 9.8% (advisory) | 0.0% | 0.1% |  |

### Breton (br)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 394 | 35.5 | 14.7 | 6.3% (advisory) | 0.0% | 0.2% |  |

### Manx (gv)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 18 | 40.9 | 14.1 | 33.3% (advisory) | 0.0% | 0.0% | directional (n=18) |

### Cornish (kw)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 3402 | 9.9 | 0.3 | 81.4% (advisory) | 0.0% | 0.1% |  |

## Celtic -> English

### Irish (ga)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | flores | 1012 | 55.2 | 28.6 | 0.0% | 0.0% | 0.0% |  |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 2339 | 66.3 | 49.6 | 0.8% | 0.0% | 0.0% |  |

### Welsh (cy)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | flores | 1012 | 54.5 | 30.8 | 0.0% | 0.0% | 0.0% |  |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 1631 | 59.6 | 42.1 | 1.2% | 0.0% | 0.7% |  |

### Scottish Gaelic (gd)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | flores | 1012 | 30.8 | 7.9 | 0.0% | 0.0% | 0.0% |  |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 961 | 33.0 | 15.8 | 0.9% | 0.0% | 0.3% |  |

### Breton (br)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 394 | 40.6 | 20.5 | 0.5% | 0.0% | 0.2% |  |

### Manx (gv)

| System | Tier | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | Open-weight anchors | tatoeba | 18 | 37.5 | 23.3 | 0.0% | 0.0% | 0.0% | directional (n=18) |

## Coverage

### English -> Celtic

| System | ga | cy | gd | br | gv | kw |
| --- | --- | --- | --- | --- | --- | --- |
| alibaba-mt | . | . | n/a | . | . | . |
| aws-translate | . | . | n/a | n/a | n/a | n/a |
| azure-translator | . | . | n/a | n/a | n/a | n/a |
| claude-haiku-4-5 | . | . | . | . | . | . |
| claude-opus-5 | . | . | . | . | . | . |
| deepl | . | . | n/a | . | n/a | n/a |
| deepseek-v4-flash | . | . | . | . | . | . |
| deepseek-v4-pro | . | . | . | . | . | . |
| gemini-3.6-flash | . | . | . | . | . | . |
| google-translate-v2 | . | . | . | . | n/a | n/a |
| google-translation-llm | . | . | . | n/a | n/a | n/a |
| gpt-5.6-luna | . | . | . | . | . | . |
| gpt-5.6-sol | . | . | . | . | . | . |
| kimi-k3 | . | . | . | . | . | . |
| madlad400-3b | . | . | . | . | . | . |
| nllb-600m | . | . | . | . | n/a | n/a |
| opus-mt-cel | ok | ok | ok | ok | ok | ok |
| qwen3.5-9b | . | . | . | . | . | . |
| qwen3.7-max | . | . | . | . | . | . |
| salamandrata-7b | . | . | n/a | n/a | n/a | n/a |
| tiny-aya-water | . | . | n/a | n/a | n/a | n/a |
| translategemma-4b | . | n/a | . | . | . | n/a |

### Celtic -> English

| System | ga | cy | gd | br | gv | kw |
| --- | --- | --- | --- | --- | --- | --- |
| alibaba-mt | . | . | n/a | . | . | . |
| aws-translate | . | . | n/a | n/a | n/a | n/a |
| azure-translator | . | . | n/a | n/a | n/a | n/a |
| claude-haiku-4-5 | . | . | . | . | . | . |
| claude-opus-5 | . | . | . | . | . | . |
| deepl | . | . | n/a | . | n/a | n/a |
| deepseek-v4-flash | . | . | . | . | . | . |
| deepseek-v4-pro | . | . | . | . | . | . |
| gemini-3.6-flash | . | . | . | . | . | . |
| google-translate-v2 | . | . | . | . | n/a | n/a |
| google-translation-llm | . | . | . | n/a | n/a | n/a |
| gpt-5.6-luna | . | . | . | . | . | . |
| gpt-5.6-sol | . | . | . | . | . | . |
| kimi-k3 | . | . | . | . | . | . |
| madlad400-3b | . | . | . | . | . | . |
| nllb-600m | . | . | . | . | n/a | n/a |
| opus-mt-cel | ok | ok | ok | ok | ok | . |
| qwen3.5-9b | . | . | . | . | . | . |
| qwen3.7-max | . | . | . | . | . | . |
| salamandrata-7b | . | . | n/a | n/a | n/a | n/a |
| tiny-aya-water | . | . | n/a | n/a | n/a | n/a |
| translategemma-4b | n/a | n/a | n/a | . | . | n/a |

`ok` scored, `.` runnable but not run yet, `n/a` the vendor's own language list does not offer it. An `n/a` is a coverage fact, not a score of zero.

## Reading the numbers

- chrF++/BLEU: 0-100, higher is better; chrF++ is the headline metric.
- Off-target: share of *non-blank* lines confidently detected in the wrong language. Blank output is counted as blank, never as off-target.
- FLORES rows are the comparable public benchmark; Tatoeba rows are coverage-only and contamination-inflated for systems trained on it. `trackb-*` rows are the fresh, post-cutoff harvest.
- `directional (n<50)` rows (Manx) are reported for honesty, not ranking.
- A `decoding deviation` note means the vendor refused part of the common decoding contract; the receipt records exactly what was sent.
