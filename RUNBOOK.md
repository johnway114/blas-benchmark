# Runbook

Operational procedures. The method itself lives in METHODOLOGY.md; if the
two ever disagree, METHODOLOGY.md wins and this file has a bug.

## One-time setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-local.txt   # optional: local models
cp .env.example .env                              # add whichever keys exist
.venv/bin/python bench.py prepare                 # ~55 MB of downloads, ~1 min
.venv/bin/python -m pytest -q                     # 201 tests, all offline
```

`prepare` is idempotent and hash-checked: re-running it verifies the built
eval sets against `manifests/` and refuses silently-changed upstream data
(`--force` accepts a deliberate, to-be-published corpus change). It ends by
rebuilding `manifests/corpus-qa.json` and `manifests/lid-validation.json`,
the two measurement artifacts the leaderboard's caveats come from.

`requirements.txt` is the install contract; `requirements.lock` is the
reproduction contract, a `pip freeze` of the environment behind the current
LEADERBOARD.md. Install the lock when you need to reproduce a published
number exactly rather than merely run the harness. Every receipt records the
subset of it that can move a number (Python, sacrebleu, torch, transformers).

Several systems need more than one variable (`aws-translate` needs three).
`plan` and `doctor` both list every variable that is still missing, not just
the first.

## Launch bake-off

1. `bench.py doctor` - every hosted system should read `OK`. For any
   `MODEL_NOT_FOUND` on a `provisional`/`alias` pin, update the ID in
   `celticbench/registry.py` from the doctor's suggestions and re-run
   doctor. Commit the pin change.
   - `LANGS_MISSING` means the vendor's live language list no longer matches
     the coverage the registry claims. Fix the `LANGS` table in
     `celticbench/lib.py`, do not widen the run.
   - `NO_LANGUAGE_PROBE` is expected for `alibaba-mt`: that API publishes no
     list-languages operation, so its coverage cannot be checked live.
   - `NEEDS_HF_TOKEN` means a gated local model whose licence you have not
     accepted yet (`translategemma-4b`, `tiny-aya-water`).
2. `bench.py plan` - sanity-check the matrix and request volume before
   spending.
3. `bench.py run --all-ready` - runs every supported combination for every
   system with credentials, plus the local models. Resumable at any point
   (line-level cache); re-invoking skips finished lines.
   - `--api-only` skips local models (useful on a machine without torch).
     With no keys set, `--all-ready` runs just the local ones.
   - One system at a time: `bench.py run <system-id> --all-ready`.
   - `--workers N` runs N hosted requests concurrently; line order in the
     hypothesis file is unaffected. Start at 4 and watch for 429s.
   - Download sizes, once: MADLAD-400 3B ~12 GiB, Qwen3.5 9B ~18 GiB,
     SalamandraTA 7B ~15 GiB, NLLB ~2.5 GiB, Opus-MT ~0.6 GiB.
   - **Local throughput decides what is feasible where.** Measured on an M4
     Pro (24 GiB, MPS): Opus-MT runs ~33 lines/second, so its entire matrix
     across three corpora finishes in about 20 minutes. SalamandraTA 7B in
     bfloat16 runs ~90 seconds *per line*, which is ~50 hours for one FLORES
     direction pair; Qwen3.5 9B is worse. Run the billion-parameter local
     models on a rented GPU, not here. A system that cannot be run within an
     edition's compute budget is reported as not run - never as a partial row,
     which would not be comparable anyway.
4. `bench.py score && bench.py leaderboard`.
5. Review `LEADERBOARD.md` and `scores/scores.json`, then commit them
   together with `out/*.hyp` and `out/*.receipt.json`. The hypotheses are
   part of the published record: without them nobody can re-derive a number.

### Cost expectations

`plan` currently estimates ~313k hosted-API requests for the complete matrix
(15 hosted systems, both directions, full Tatoeba + FLORES + the sealed Track
B slice). Requests are single sentences; typical spend is a few dollars for
efficient tiers and tens of dollars for flagship tiers. The dedicated MT
services bill by character rather than token, and their matrices are smaller
because they only run the languages they actually offer. Set a billing alert
before the first full run. To trial cheaply first, `--limit 50` produces
partial-slice rows in their own `.limit50` files, clearly flagged and never
comparable.

## Sealing a Track B slice

Quarterly. The slice must be committed before any model sees it: a public
commit is the third-party-verifiable timestamp behind the freshness claim.

```bash
.venv/bin/python bench.py harvest 2026q4 --cutoff <newest documented training cutoff> \
    --limit-per-lang 300 --harvest-date 2026-10-01
.venv/bin/python bench.py qa           # measure the new slice: defects + detector reliability
.venv/bin/python bench.py slice 2026q4 # confirm it is runnable
git add manifests/trackb-2026q4.*      # commit BEFORE running any system
```

- `--harvest-date` is explicit so the same command rebuilds byte-identical
  eval files later. Let it default to today only for the sealing run itself.
- Do not pass `--lang`: a full-language harvest is what records the
  languages with no viable publisher, and those refusals are published
  findings.
- `bench.py qa` after every harvest. Without it the new corpus has no
  measured detector reliability and its off-target column renders
  `(unmeasured)`.
- The eval text itself stays local (`eval/` is gitignored); the manifests,
  per-row source URLs and dates are what get committed.

## When a new model releases

Target: published row within 72 hours of API availability.

1. Add a registry entry in `celticbench/registry.py`: ID, vendor, provider
   (`openai_compat` covers any OpenAI-compatible endpoint), `key_env`,
   tier, `pin_status: "provisional"`, the shared `CHAT_DECODING`. Never pin
   a `-latest` alias. If the vendor forces a decoding departure, declare it
   in `decoding_deviation` with the reason rather than quietly complying.
2. `bench.py doctor` - confirm the pin against the live model list.
3. Run every combination for just that system:
   ```bash
   .venv/bin/python bench.py run <system-id> --all-ready
   ```
   (A bare `run --all-ready` also works: finished systems are line-level
   cached, so only the new system actually spends API calls.)
4. `bench.py score && bench.py leaderboard`.
5. Commit: registry pin, hypotheses, receipts, scores.json, LEADERBOARD.md.
   One commit per system keeps the history auditable.
6. Write the release note from the row: headline chrF++ vs the standing
   leaders, off-target anecdotes quoted verbatim from `out/*.hyp` (they are
   the most legible evidence), and any pin/decoding deviations the receipt
   recorded.

## Refresh cadence

- **Quarterly**: seal a new Track B slice as above, retire and publish the
  previous one.
- **Annually, or on any harness/metric change**: bump the leaderboard
  version, re-run every listed system on the current slices, and
  `bench.py leaderboard --archive` to freeze the outgoing board and scores
  under `archive/`. Add a CHANGELOG.md entry saying what the change
  invalidates. Never mix conditions inside one leaderboard version.

## Troubleshooting

- **429 / rate limits**: lower `--workers`, raise `--sleep` (default 0.15s,
  applied per worker). The cache makes interrupted runs free to resume.
- **A provider rejects temperature/top_p** (some reasoning tiers): the
  openai_compat adapter retries without them and the receipt records the
  deviation string and the exact parameters sent. The row stays valid and
  the leaderboard flags it.
- **`manifest ... does not match`**: upstream data changed or a local file
  was edited. Never force through it silently; either restore from git or
  make the corpus change deliberately with `prepare --force` and publish it
  as a new leaderboard version.
- **`MODEL_NOT_FOUND` from doctor**: the vendor renamed or retired the
  pinned ID; update the registry from the suggestion list. Provisional pins
  exist precisely so this is a 2-minute fix, not a silent wrong-model run
  (receipts record the reported model ID, so a silent alias swap by the
  vendor is also detectable after the fact).
- **`score` excludes a run you expected to see**: the printed reason is
  exact. `prompt changed` or `declared decoding changed` means the run
  belongs to an older leaderboard version and must be re-run; `eval input
  file changed` means the corpus moved under it.
- **fasttext wheel issues**: the harness uses `fasttext-predict` (pure
  wheel). If you swap in full `fasttext`, nothing changes: `langid.py`
  tries both import names.
