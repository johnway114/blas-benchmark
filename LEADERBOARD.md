# Celtic Translation Benchmark - leaderboard

Method **v3**, generated from `scores/scores.json` (2026-07-29T22:43:37+00:00). Every row is backed by a verified receipt in `out/`; excluded runs are listed at the bottom with reasons. Rows from another method version are refused, not ranked: `METHODOLOGY.md`, `CHANGELOG.md`.

- Off-target detector: `lid.176.ftz` sha256 `8f3472cfe873`, confidence >= 0.5
- Runtime: python 3.14.3, sacrebleu 2.6.0

## Coverage

### English -> Celtic

| System | ga | cy | gd | br | gv | kw |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5 | . | . | . | . | . | . |
| deepseek-v4-pro | . | . | . | . | . | . |
| gemini-3.6-flash | . | . | . | . | . | . |
| gpt-5.6-sol | . | . | . | . | . | . |
| kimi-k3 | . | . | . | . | . | . |
| qwen3.7-max | . | . | . | . | . | . |

### Celtic -> English

| System | ga | cy | gd | br | gv | kw |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5 | . | . | . | . | . | . |
| deepseek-v4-pro | . | . | . | . | . | . |
| gemini-3.6-flash | . | . | . | . | . | . |
| gpt-5.6-sol | . | . | . | . | . | . |
| kimi-k3 | . | . | . | . | . | . |
| qwen3.7-max | . | . | . | . | . | . |

`ok` scored, `.` runnable but not run yet, `n/a` the vendor's own language list does not offer it. An `n/a` is a coverage fact, not a score of zero.

## Reading the numbers

- chrF++/BLEU: 0-100, higher is better; chrF++ is the headline metric.
- Off-target: share of *non-blank* lines confidently detected in the wrong language. Blank output is counted as blank, never as off-target.
- FLORES rows are the comparable public benchmark; Tatoeba rows are coverage-only and contamination-inflated for systems trained on it. `trackb-*` rows are the fresh, post-cutoff harvest.
- `directional (n<50)` rows (Manx) are reported for honesty, not ranking.
- A `decoding deviation` note means the vendor refused part of the common decoding contract; the receipt records exactly what was sent.
