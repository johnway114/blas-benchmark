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

## Publishing an edition to the website

The board at `blasapp.com/research/benchmark/translation` renders one committed
file. It never fetches this repo at build time: a site that can resolve a
different edition than the one its build was checked against would publish
numbers nobody verified together.

`bench.py export` is the only supported way to produce that file. It carries
every reading rule with the data -- whether an off-target rate is authoritative,
whether a row may be ordered, how complete the edition is -- so the renderer
never re-derives a method decision and cannot drift from LEADERBOARD.md.

```bash
.venv/bin/python bench.py score
.venv/bin/python bench.py leaderboard
.venv/bin/python bench.py export                      # scores/web.json, the public contract
.venv/bin/python bench.py export --out ../blas/webApp/content/benchmark/celtic-translation.json
```

Commit `scores/web.json` here (it is the stable JSON endpoint third parties
cite) and the copy in the web repo in the same session, so the two can never
describe different editions.

Order matters, and each step gates the next:

1. **`score` first, always.** `export` reads `scores/scores.json` and does not
   recompute anything. Exporting before scoring publishes the previous edition
   under the new one's date.
2. **Read the export's own summary line.** It prints
   `edition <version> <status>: <n>/<expected> runs scored`. If that says
   `in-progress` and you intended a launch, the matrix is short and the site
   will correctly refuse to present it as final. Fix the runs, not the banner.
3. **Check the excluded list.** `export` carries exclusions through to the site
   verbatim. A run excluded for `prompt changed` or `eval input file changed`
   belongs to an older method version and means the edition is not coherent.
4. **Only then update the site.** The page derives its status banner, its
   coverage grid and its social card from the export, so a stale file quietly
   backdates all three.

### Release order on launch day

Publish in this order so that no link ever points at something that does not
exist yet, and so the canonical numbers are live before anyone is invited to
argue with them:

1. Commit and push this repo: hypotheses, receipts, `scores/scores.json`,
   `LEADERBOARD.md`, `scores/web.json`.
2. Deploy the website with the refreshed export. Confirm the board's status
   banner matches the edition you meant to ship, and open
   `/research/benchmark/translation/opengraph-image` to confirm the social card
   renders the same numbers as the page.
3. Publish the write-up under `content/research/`. A results post is a snapshot
   and says so: the board is canonical and the post is never edited to match a
   later edition.
4. Only then post externally. Every external link goes to the board, not to the
   post, so the first thing a reader sees is the live edition with its caveats
   attached.

A results post carries, in this order: the finding box before any methodology,
including at least one bullet that cuts against our own headline; the table
with its incompleteness stated; method in brief with a link to the full method;
findings by language and by direction; the integrity metrics named per system;
the negative results and the ties; what this does not show; how to reproduce it;
and the conflict-of-interest statement. Lead with the gap or the negative
result, never with a superlative -- the qualified claim goes in the title.

### Corrections after publication

Never edit a number in place. Add a dated `CHANGELOG.md` entry saying what
changed and what it invalidates, rerun `score`, `leaderboard` and `export`, and
redeploy. If the correction changes a ranking, say so in the entry. A board that
silently rewrites its own history cannot be cited, and being citable is the
entire point of publishing one.

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
