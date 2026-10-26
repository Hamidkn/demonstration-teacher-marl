"""Extend the main warm-start experiment to more training seeds.

Three reviews in a row made the same point: the teacher-source comparison is
consistent in sign across all three seeds but gives p = 0.12 with the seed as
the unit of analysis, and no amount of rewriting fixes that. Only more
independent training runs can.

This reproduces the Colab configuration of Section 5.2 exactly — the full
demonstration sets with no cap, 500,000 steps — for additional seeds, so the
new runs pool with seeds 1-3 rather than forming a separate experiment.

Resumable: a run whose .pt exists is skipped, so this can be stopped and
restarted. Output goes to local disk, never to the synced project folder.

Usage:
    py -3.10 run_extra_seeds.py                   # seeds 4-8, all three arms
    py -3.10 run_extra_seeds.py --seeds 4 5       # a subset
    py -3.10 run_extra_seeds.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

STEPS = 500_000
DEFAULT_OUT = "results/main_runs"

# Exactly the three configurations of Section 5.2; no --demo-cap.
ARMS = [
    ("cold",       "none"),
    ("warm_astar", "demos/demos_astar.npz"),
    ("warm_bco",   "demos/demos_bco.npz"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for _tag, demo in ARMS:
        if demo != "none" and not os.path.exists(demo):
            sys.exit(f"missing demonstration set: {demo}")

    jobs = [(s, tag, demo) for s in args.seeds for tag, demo in ARMS]
    todo = [(s, t, d) for s, t, d in jobs
            if not os.path.exists(os.path.join(args.out_dir, f"{t}_s{s}.pt"))]

    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run")
    if not todo:
        print("nothing left to do")
        return
    print(f"at ~45 min per run this is roughly {len(todo) * 0.75:.1f} hours")
    print(f"output: {args.out_dir}\n")

    if args.dry_run:
        for s, tag, demo in todo:
            print(f"  seed {s}  {tag:<11} <- {demo}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    for i, (seed, tag, demo) in enumerate(todo, 1):
        print(f"\n{'=' * 68}\n[{i}/{len(todo)}] seed {seed}  {tag}\n{'=' * 68}")
        cmd = [sys.executable, "train_marl.py",
               "--tag", f"{tag}_s{seed}",
               "--warmstart", demo,
               "--total-steps", str(args.steps),
               "--seed", str(seed),
               "--out-dir", args.out_dir]
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"run failed: {tag}_s{seed}")
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] done. elapsed {el/60:.0f} min, "
              f"est. remaining {el/i*(len(todo)-i)/60:.0f} min")

    print(f"\nall done in {(time.time() - t0)/60:.0f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
