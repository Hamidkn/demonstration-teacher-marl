"""Run the whole matched-budget ablation locally, without Colab.

This workload does not benefit from a GPU: the policy is a ~48k-parameter MLP,
and the wall-clock cost is dominated by the numpy environment simulation, which
is CPU-bound either way. Running locally therefore costs little and removes the
Colab quota and disconnection problems entirely.

The script is resumable. A run whose CSV already exists is skipped, so you can
stop it (Ctrl+C), close the laptop, and start it again later without losing
completed work.

Setup (once):
    py -3.10 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Usage:
    py -3.10 run_ablation_local.py              # all seeds, all arms
    py -3.10 run_ablation_local.py --seeds 1    # one seed only
    py -3.10 run_ablation_local.py --dry-run    # show the plan, run nothing
"""

import argparse
import os
import subprocess
import sys
import time

CAP = 11173          # common demonstration budget = size of the smallest set
STEPS = 500_000

# Write results to local disk, not to the synced project folder. Google Drive's
# virtual filesystem fails long-lived write handles with OSError 22, which
# killed one run after 13 minutes of training.
DEFAULT_OUT = "results/ablation_runs"

ARMS = [
    ("warm_astar_cap",  "demos/demos_astar.npz"),
    ("warm_bco_k1_cap", "demos/demos_bco_k1.npz"),
    ("warm_bco_k3_cap", "demos/demos_bco.npz"),
    ("warm_bco_k5_cap", "demos/demos_bco_k5.npz"),
]


def check_inputs():
    missing = [f for _t, f in ARMS if not os.path.exists(f)]
    missing += [f for f in ("marl_env.py", "train_marl.py") if not os.path.exists(f)]
    if missing:
        sys.exit("missing required files: " + ", ".join(missing))
    try:
        import numpy as np
    except ImportError:
        sys.exit("numpy is required")
    for _t, f in ARMS:
        n = len(np.load(f)["actions"])
        if n < CAP:
            sys.exit(f"{f} has only {n} pairs, below the {CAP} budget")
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("PyTorch is not installed. Run:\n"
                 "  py -3.10 -m pip install torch "
                 "--index-url https://download.pytorch.org/whl/cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="where runs are written. Defaults to a local-disk "
                         "path, because writing 977 log flushes to a synced "
                         "drive is what killed an earlier run.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        check_inputs()

    jobs = [(s, tag, demos) for s in args.seeds for tag, demos in ARMS]
    # Completion is marked by the .pt file, which train_marl.py writes only
    # after the final step. Testing for the CSV instead would treat a run that
    # died midway as finished and silently leave a truncated result in the
    # analysis -- which is exactly what happened to warm_bco_k5_cap_s2.
    todo = [(s, t, d) for s, t, d in jobs
            if not os.path.exists(os.path.join(args.out_dir, f"{t}_s{s}.pt"))]

    print(f"{len(jobs)} runs total, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run")
    if not todo:
        print("nothing left to do")
        return
    print(f"at ~45 min per run this is roughly {len(todo) * 0.75:.1f} hours\n")

    if args.dry_run:
        for s, tag, demos in todo:
            print(f"  seed {s}  {tag:<17} <- {demos}")
        return

    t_start = time.time()
    for i, (seed, tag, demos) in enumerate(todo, 1):
        run_tag = f"{tag}_s{seed}"
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] seed {seed}  {tag}\n{'=' * 70}")
        cmd = [sys.executable, "train_marl.py",
               "--tag", run_tag,
               "--warmstart", demos,
               "--demo-cap", str(CAP),
               "--total-steps", str(args.steps),
               "--seed", str(seed),
               "--out-dir", args.out_dir]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"run failed: {run_tag} (exit {r.returncode})")
        el = time.time() - t_start
        print(f"[{i}/{len(todo)}] done. elapsed {el/60:.0f} min, "
              f"est. remaining {el/i*(len(todo)-i)/60:.0f} min")

    print(f"\nall done in {(time.time() - t_start)/60:.0f} min. "
          f"Results are in runs/ as <arm>_s<seed>.csv")


if __name__ == "__main__":
    main()
