# Celtic Translation Benchmark — leaderboard

Generated from `scores/scores.json` (2026-07-29T16:56:53+00:00). Every row is backed by a verified receipt in `out/`; excluded runs are listed at the bottom with reasons. Method: `METHODOLOGY.md`.

## English -> Celtic

### Cornish (kw)

| System | Corpus | n | chrF++ | BLEU | Off-target | Blank | Copy | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Opus-MT en-cel / cel-en (anchor) | tatoeba | 8 | 9.8 | 1.6 | 75.0% (advisory) | 0.0% | 0.0% | partial slice; directional (n=8) |

## Reading the numbers

- chrF++/BLEU: 0-100, higher is better; chrF++ is the headline metric.
- Off-target: share of lines confidently detected in the wrong language.
- FLORES rows are the comparable public benchmark; Tatoeba rows are coverage-only and contamination-inflated for systems trained on it.
- `directional (n<50)` rows (Manx) are reported for honesty, not ranking.
