# Runbook

Operational procedures. The method itself lives in METHODOLOGY.md; if the
two ever disagree, METHODOLOGY.md wins and this file has a bug.

## One-time setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                              # add all six provider keys
.venv/bin/python bench.py prepare                 # ~55 MB of downloads, ~1 min
.venv/bin/python -m pytest -q
```

`prepare` is idempotent and hash-checked: re-running it verifies the built
eval sets against `manifests/` and refuses silently changed upstream data
(`--force` accepts a deliberate, to-be-published corpus change). It ends by
rebuilding `manifests/corpus-qa.json` and `manifests/lid-validation.json`,
the two measurement artifacts the leaderboard's caveats come from.

`requirements.txt` is the install contract; `requirements.lock` is the
reproduction contract, a `pip freeze` of the environment behind the current
LEADERBOARD.md. Install the lock when reproducing a published number exactly.
Every receipt records the runtime versions that can move that number.

Each v3 system needs one provider key. `plan` and `doctor` list every missing
key, so an incomplete environment cannot silently narrow the edition.

The edition roster is `gpt-5.6-sol`, `claude-opus-5`,
`gemini-3.6-flash`, `deepseek-v4-pro`, `kimi-k3`, and `qwen3.7-max`.

## Launch bake-off

1. `bench.py doctor` - all six systems must read `OK`. If a provisional pin
   reports `MODEL_NOT_FOUND`, do not edit the live edition in place. Archive
   its outgoing results, bump the method version and document the invalidation;
   only then update the ID in `celticbench/registry.py` from the doctor's
   suggestions, re-run doctor and rerun the complete panel.
2. `bench.py plan` - confirm exactly 144 runs and 150,036 requests before
   spending. Any smaller matrix means a credential or registry problem; do
   not proceed with a narrowed edition.
3. `bench.py run --all-ready` - runs every combination for the six systems.
   It is resumable through the line-level cache; re-invoking skips finished
   lines.
   - One system at a time: `bench.py run <system-id> --all-ready`.
   - `--workers N` runs N requests concurrently; line order in the hypothesis
     file is unaffected. Start at 4 and watch for 429s.
4. `bench.py score && bench.py leaderboard`.
5. Review `LEADERBOARD.md` and `scores/scores.json`, then commit them
   together with `out/*.hyp` and `out/*.receipt.json`. The hypotheses are
   part of the published record: without them nobody can re-derive a number.

### Cost expectations

`plan` must report 24 runs and 25,006 sentence requests for each system, or
144 runs and 150,036 requests for the six-system edition. Requests are single
sentences. Set provider billing alerts before the first full run. To exercise
the pipeline cheaply, `--limit 50` writes partial-slice rows to distinct
`.limit50` files; these are smoke artifacts and are never comparable results.

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

## Changing the model roster

The roster never changes inside a live edition. To add, remove or repin a
model:

1. Archive the outgoing leaderboard and scores with
   `bench.py leaderboard --archive`, and preserve its hypotheses and receipts.
2. Bump `method_version` and add a CHANGELOG.md entry stating what the change
   invalidates.
3. Update `celticbench/registry.py` with the complete new roster. Use an exact
   hosted model ID, the appropriate provider key, `pin_status:
   "provisional"`, and the shared `CHAT_DECODING`; never pin a moving
   `*-latest` alias.
4. Run `bench.py doctor` and resolve every credential or pin failure before
   spending on the matrix.
5. Run every system across the complete edition, then run `bench.py score`
   and `bench.py leaderboard`.
6. Publish the registry, hypotheses, receipts, scores and leaderboard
   together. Do not carry a receipt or score across the version boundary.

## Refresh cadence

- **Quarterly**: seal a new Track B slice as above, retire and publish the
  previous one.
- **At every model-roster, prompt, harness or metric change**: archive the
  outgoing leaderboard and scores, bump the method version, document the
  invalidation, and rerun the complete panel. Never mix conditions or change
  the roster inside one leaderboard edition.

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
