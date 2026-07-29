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

Running with no network access, no keys, and no torch still supports:
prepare (needs network once), plan, score, leaderboard, doctor (reports).
"""
from __future__ import annotations

import argparse
import os
import sys

from celticbench.lib import (
    ALL_LANGS, CORPORA, DIRECTIONS, load_env, manifest_path,
)
from celticbench.registry import SYSTEMS, matrix


def cmd_prepare(args: argparse.Namespace) -> int:
    from celticbench.prepare import prepare
    prepare(force=args.force)
    return 0


def cmd_plan(_args: argparse.Namespace) -> int:
    rows = matrix()
    runnable = [r for r in rows if r["supported"]]
    print(f"{len(runnable)} runnable combinations "
          f"({len(rows) - len(runnable)} unsupported skipped)\n")
    by_system: dict[str, int] = {}
    for row in runnable:
        by_system[row["system"]] = by_system.get(row["system"], 0) + 1
    width = max(len(s) for s in by_system)
    for system_id, count in sorted(by_system.items()):
        entry = SYSTEMS[system_id]
        key_env = entry.get("key_env")
        key_state = "local" if key_env is None else (
            "key OK" if os.environ.get(key_env) else f"needs {key_env}"
        )
        print(f"  {system_id:<{width}}  {count:>3} combos  [{entry['pin_status']:>11}]  {key_state}")
    total_requests = sum(
        _combo_n(row["corpus"], row["lang"]) for row in runnable
        if SYSTEMS[row["system"]].get("key_env") is not None
    )
    print(f"\nestimated hosted-API requests for a full bake-off: ~{total_requests:,}")
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
            entry = SYSTEMS[row["system"]]
            key_env = entry.get("key_env")
            if key_env is not None and not os.environ.get(key_env):
                skipped += 1
                continue
            if entry["provider"] == "hf_local" and args.api_only:
                skipped += 1
                continue
            run_system(row["system"], row["corpus"], row["lang"], row["direction"],
                       limit=args.limit, sleep=args.sleep)
            ran += 1
        print(f"run --all-ready: {ran} combinations run, {skipped} skipped (missing key/filtered)")
        return 0

    if not (args.system and args.corpus and args.lang and args.direction):
        raise SystemExit("run needs SYSTEM CORPUS LANG DIRECTION, or --all-ready")
    run_system(args.system, args.corpus, args.lang, args.direction,
               limit=args.limit, sleep=args.sleep)
    return 0


def cmd_score(_args: argparse.Namespace) -> int:
    from celticbench.scoring import score_all
    score_all()
    return 0


def cmd_leaderboard(_args: argparse.Namespace) -> int:
    from celticbench.leaderboard import write
    write()
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    from celticbench.providers import doctor_check

    print("doctor: checking registry pins and credentials\n")
    worst = 0
    for system_id, entry in SYSTEMS.items():
        result = doctor_check(system_id, entry)
        status = result["status"]
        marker = {"OK": " ", "MISSING_KEY": "-", "NEEDS_DEPS": "-"}.get(status, "!")
        print(f"[{marker}] {system_id:<22} {status:<22} {result['detail']}")
        if status in ("AUTH_OR_NETWORK_FAIL", "MODEL_NOT_FOUND", "LANGS_MISSING"):
            worst = 1
    print("\nlegend: [ ] ready, [-] waiting on key/deps, [!] needs attention")
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
    p_run.add_argument("corpus", nargs="?", choices=CORPORA)
    p_run.add_argument("lang", nargs="?", choices=ALL_LANGS)
    p_run.add_argument("direction", nargs="?", choices=DIRECTIONS)
    p_run.add_argument("--all-ready", action="store_true",
                       help="run every supported combination whose credentials exist; "
                            "combine with SYSTEM to scope to one system")
    p_run.add_argument("--api-only", action="store_true",
                       help="with --all-ready: skip local anchor systems")
    p_run.add_argument("--limit", type=int, default=None,
                       help="only the first N lines (smoke runs; marked partial in scores)")
    p_run.add_argument("--sleep", type=float, default=0.15,
                       help="pause between hosted-API requests (seconds)")
    p_run.set_defaults(func=cmd_run)

    p_score = sub.add_parser("score", help="verify receipts and compute all metrics")
    p_score.set_defaults(func=cmd_score)

    p_board = sub.add_parser("leaderboard", help="render LEADERBOARD.md from scores.json")
    p_board.set_defaults(func=cmd_leaderboard)

    p_doctor = sub.add_parser("doctor", help="validate keys and pinned model IDs")
    p_doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
