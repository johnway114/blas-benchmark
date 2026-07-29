# Runbook

Operational procedures. The method itself lives in METHODOLOGY.md; if the
two ever disagree, METHODOLOGY.md wins and this file has a bug.

## One-time setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-local.txt   # optional: local anchors
cp .env.example .env                              # add whichever keys exist
.venv/bin/python bench.py prepare                 # ~55 MB of downloads, ~1 min
.venv/bin/python -m pytest -q                     # 31 tests, all offline
```

`prepare` is idempotent and hash-checked: re-running it verifies the built
eval sets against `manifests/` and refuses silently-changed upstream data
(`--force` accepts a deliberate, to-be-published corpus change).

## Launch bake-off

1. `bench.py doctor` - every hosted system should read `OK`. For any
   `MODEL_NOT_FOUND` on a `provisional`/`alias` pin, update the ID in
   `celticbench/registry.py` from the doctor's suggestions and re-run
   doctor. Commit the pin change.
2. `bench.py plan` - sanity-check the matrix and request volume before
   spending.
3. `bench.py run --all-ready` - runs every supported combination for every
   system with credentials, plus the local anchors. Resumable at any point
   (line-level cache); re-invoking skips finished lines.
   - `--api-only` skips the local anchors (useful on a machine without
     torch). With no keys set, `--all-ready` runs just the anchors.
   - One system at a time: `bench.py run <system-id> --all-ready`.
   - MADLAD-400 3B downloads ~12 GiB once; NLLB ~2.5 GiB; Opus-MT ~0.6 GiB.
4. `bench.py score && bench.py leaderboard`.
5. Review `LEADERBOARD.md` and `scores/scores.json`, commit both together
   with the `out/*.receipt.json` files for the published runs.

### Cost expectations

`plan` currently estimates ~205k hosted-API requests for the complete
matrix (9 hosted systems, both directions, full Tatoeba + FLORES). Requests
are single sentences; typical spend is a few dollars for efficient tiers and
tens of dollars for flagship tiers. Google Cloud Translation bills by
character (~1.1M chars for its 16 combinations). Set a billing alert before
the first full run. To trial cheaply first, `--limit 50` produces
partial-slice rows that are clearly flagged and never comparable.

## When a new model releases

Target: published row within 72 hours of API availability.

1. Add a registry entry in `celticbench/registry.py`: ID, vendor, provider
   (`openai_compat` covers any OpenAI-compatible endpoint), `key_env`,
   tier, `pin_status: "provisional"`, the shared `CHAT_DECODING`.
2. `bench.py doctor` - confirm the pin against the live model list.
3. Run every combination for just that system:
   ```bash
   .venv/bin/python bench.py run <system-id> --all-ready
   ```
   (A bare `run --all-ready` also works: finished systems are line-level
   cached, so only the new system actually spends API calls.)
4. `bench.py score && bench.py leaderboard`.
5. Commit: registry pin, receipts, scores.json, LEADERBOARD.md. One commit
   per system keeps the history auditable.
6. Write the release note from the row: headline chrF++ vs the standing
   leaders, off-target anecdotes quoted verbatim from `out/*.hyp` (they are
   the most legible evidence), and any pin/decoding deviations the receipt
   recorded.

## Refresh cadence

- **Quarterly**: seal a new Track B slice (harvest manifest hash committed
  before any model run), retire and publish the previous one.
- **Annually, or on any harness/metric change**: bump the leaderboard
  version, re-run every listed system on the current slices, and archive
  the previous `scores/scores.json` under `scores/archive/`. Never mix
  conditions inside one leaderboard version.

## Troubleshooting

- **429 / rate limits**: raise `--sleep` (default 0.15s). The cache makes
  interrupted runs free to resume.
- **A provider rejects temperature/top_p** (some reasoning tiers): the
  openai_compat adapter retries without them and the receipt records
  `adjusted: true` with the exact parameters used. The row remains valid;
  the deviation is visible.
- **`manifest ... does not match`**: upstream data changed or a local file
  was edited. Never force through it silently; either restore from git or
  make the corpus change deliberately with `prepare --force` and publish it
  as a new leaderboard version.
- **`MODEL_NOT_FOUND` from doctor**: the vendor renamed or retired the
  pinned ID; update the registry from the suggestion list. Provisional pins
  exist precisely so this is a 2-minute fix, not a silent wrong-model run
  (receipts record the reported model ID, so a silent alias swap by the
  vendor is also detectable after the fact).
- **fasttext wheel issues**: the harness uses `fasttext-predict` (pure
  wheel). If you swap in full `fasttext`, nothing changes: `langid.py`
  tries both import names.
