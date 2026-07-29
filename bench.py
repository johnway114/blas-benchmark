#!/usr/bin/env python3
"""Celtic Translation Benchmark CLI.

Typical lifecycle:

    python bench.py prepare              # download + build eval sets, pin manifests
    python bench.py doctor               # check keys, confirm pinned model IDs
    python bench.py plan                 # show the full run matrix
    python bench.py run --all-ready      # run every system whose key is present
    python bench.py run gpt-5.6-sol flores ga en-xx   # or one combination
    python bench.py score                # verify receipts, compute all metrics
    python bench.py leaderboard          # render LEADERBOARD.md

Running with no network access and no keys still supports prepare (after its
one network build), plan, score, leaderboard, qa, and doctor (which reports).
"""
from __future__ import annotations

import argparse
import os
import sys

from celticbench.lib import ALL_LANGS, DIRECTIONS, all_corpora, load_env, manifest_path
from celticbench.registry import SYSTEMS, credential_envs, matrix


def cmd_prepare(args: argparse.Namespace) -> int:
    from celticbench.prepare import prepare
    prepare(force=args.force)
    return 0


def _missing_credentials(system_id: str) -> list[str]:
    return [env for env in credential_envs(SYSTEMS[system_id]) if not os.environ.get(env)]


def cmd_plan(_args: argparse.Namespace) -> int:
    rows = matrix()
    runnable = [r for r in rows if r["supported"]]
    print(f"{len(runnable)} runnable combinations\n")
    by_system: dict[str, int] = {}
    for row in runnable:
        by_system[row["system"]] = by_system.get(row["system"], 0) + 1
    width = max(len(s) for s in by_system)
    print("flagship-chat:")
    for system_id in sorted(by_system):
        entry = SYSTEMS[system_id]
        missing = _missing_credentials(system_id)
        state = "needs " + ", ".join(missing) if missing else "credentials OK"
        print(f"  {system_id:<{width}}  {by_system[system_id]:>3} combos  "
              f"[{entry['pin_status']:>11}]  {state}")
    total_requests = sum(
        _combo_n(row["corpus"], row["lang"]) for row in runnable
    )
    ready_requests = sum(
        _combo_n(row["corpus"], row["lang"]) for row in runnable
        if not _missing_credentials(row["system"])
    )
    print(f"\nhosted requests: ~{total_requests:,} for the full bake-off, "
          f"~{ready_requests:,} with currently available credentials")
    return 0


def _combo_n(corpus: str, lang: str) -> int:
    import json
    path = manifest_path(corpus, lang)
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        return int(json.load(handle)["n"])


def cmd_run(args: argparse.Namespace) -> int:
    from celticbench.runner import run_system

    if args.all_ready:
        ran = skipped = 0
        for row in matrix():
            if not row["supported"]:
                continue
            if args.system and row["system"] != args.system:
                continue
            if _missing_credentials(row["system"]):
                skipped += 1
                continue
            run_system(row["system"], row["corpus"], row["lang"], row["direction"],
                       limit=args.limit, sleep=args.sleep, workers=args.workers)
            ran += 1
        print(f"run --all-ready: {ran} combinations run, {skipped} skipped (missing credentials)")
        return 0

    if not (args.system and args.corpus and args.lang and args.direction):
        raise SystemExit("run needs SYSTEM CORPUS LANG DIRECTION, or --all-ready")
    run_system(args.system, args.corpus, args.lang, args.direction,
               limit=args.limit, sleep=args.sleep, workers=args.workers)
    return 0


def cmd_score(_args: argparse.Namespace) -> int:
    from celticbench.scoring import score_all
    score_all()
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from celticbench.leaderboard import write
    write(archive=args.archive)
    return 0


def cmd_qa(_args: argparse.Namespace) -> int:
    """Re-measure the built eval sets: corpus defects and detector reliability.

    Both are measurements over eval/, so both go stale the moment a Track B
    slice is sealed. Refreshing them together is what keeps a new slice from
    publishing off-target rates with no measured false-positive floor.
    """
    from celticbench.corpus_qa import build_corpus_qa
    from celticbench.prepare import build_lid_validation

    build_corpus_qa()
    build_lid_validation()
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    from celticbench.trackb import harvest_slice

    record = harvest_slice(args.slice, args.cutoff, langs=tuple(args.lang) or None,
                           limit_per_lang=args.limit_per_lang, force=args.force,
                           harvest_date=args.harvest_date)
    for lang, summary in sorted(record["languages"].items()):
        print(f"  {lang}: n={summary['n']} from {summary['documents']} documents")
    for item in record.get("unavailable", []):
        print(f"  {item['lang']}: unavailable - {item['reason'][:120]}")
    print(f"sealed {record['corpus']} -> manifests/{record['corpus']}.slice.json")
    return 0


def cmd_slice(args: argparse.Namespace) -> int:
    import json

    from celticbench.trackb import slice_summary

    print(json.dumps(slice_summary(args.slice), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    from celticbench.providers import doctor_check

    print("doctor: checking registry pins and credentials\n")
    worst = 0
    width = max(len(s) for s in SYSTEMS)
    for system_id, entry in SYSTEMS.items():
        result = doctor_check(system_id, entry)
        status = result["status"]
        marker = {"OK": " ", "MISSING_KEY": "-"}.get(status, "!")
        print(f"[{marker}] {system_id:<{width}} {status:<22} {result['detail']}")
        if status in ("AUTH_OR_NETWORK_FAIL", "MODEL_NOT_FOUND"):
            worst = 1
    print("\nlegend: [ ] ready, [-] waiting on credentials, [!] needs attention")
    print("MODEL_NOT_FOUND on a provisional pin: update registry.py with a suggested ID.")
    return worst


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(prog="bench.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="download corpora, build eval sets, pin manifests")
    p_prepare.add_argument("--force", action="store_true",
                           help="accept a deliberate corpus change (rewrites manifests)")
    p_prepare.set_defaults(func=cmd_prepare)

    p_plan = sub.add_parser("plan", help="show the run matrix and key readiness")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="run one combination or everything ready")
    p_run.add_argument("system", nargs="?", choices=sorted(SYSTEMS))
    p_run.add_argument("corpus", nargs="?", choices=all_corpora())
    p_run.add_argument("lang", nargs="?", choices=ALL_LANGS)
    p_run.add_argument("direction", nargs="?", choices=DIRECTIONS)
    p_run.add_argument("--all-ready", action="store_true",
                       help="run every supported combination whose credentials exist; "
                            "combine with SYSTEM to scope to one system")
    p_run.add_argument("--limit", type=int, default=None,
                       help="only the first N lines; written to its own .limitN file "
                            "and marked partial in scores")
    p_run.add_argument("--sleep", type=float, default=0.15,
                       help="pause between hosted-API requests, per worker (seconds)")
    p_run.add_argument("--workers", type=int, default=1,
                       help="concurrent hosted-API requests; order is preserved")
    p_run.set_defaults(func=cmd_run)

    p_score = sub.add_parser("score", help="verify receipts and compute all metrics")
    p_score.set_defaults(func=cmd_score)

    p_board = sub.add_parser("leaderboard", help="render LEADERBOARD.md from scores.json")
    p_board.add_argument("--archive", action="store_true",
                         help="also freeze a dated copy of the board and scores under archive/")
    p_board.set_defaults(func=cmd_leaderboard)

    p_qa = sub.add_parser("qa", help="rebuild manifests/corpus-qa.json from the eval sets")
    p_qa.set_defaults(func=cmd_qa)

    p_harvest = sub.add_parser("harvest", help="harvest and seal a Track B slice")
    p_harvest.add_argument("slice", help="slice id, lowercase alphanumeric, e.g. 2026q3")
    p_harvest.add_argument("--cutoff", required=True,
                           help="ISO date; only text published strictly after it is eligible. "
                                "Use the newest benchmarked model's training cutoff")
    p_harvest.add_argument("--lang", action="append", default=[], choices=ALL_LANGS,
                           help="restrict to these languages (repeatable)")
    p_harvest.add_argument("--limit-per-lang", type=int, default=500,
                           dest="limit_per_lang", help="cap segments per language")
    p_harvest.add_argument("--harvest-date", default=None, dest="harvest_date",
                           help="ISO date bounding the window; defaults to today (UTC). "
                                "Passing the original date rebuilds a slice byte-identically")
    p_harvest.add_argument("--force", action="store_true",
                           help="re-seal a slice that already exists")
    p_harvest.set_defaults(func=cmd_harvest)

    p_slice = sub.add_parser("slice", help="show a sealed Track B slice record")
    p_slice.add_argument("slice", help="slice id, e.g. 2026q3")
    p_slice.set_defaults(func=cmd_slice)

    p_doctor = sub.add_parser("doctor", help="validate keys and pinned model IDs")
    p_doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
