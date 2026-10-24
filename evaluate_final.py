"""Re-evaluate every saved policy on a large held-out scenario set.

Reviewer point R1.7: the per-checkpoint evaluation during training used five
scenarios, which cannot resolve the differences the paper claims and is a large
part of the reported variance. The trained policies are saved, so a proper
evaluation costs no retraining at all.

Held-out seeds are 20000..20000+N, disjoint from everything used earlier:
demonstrations 5000-5029, in-training evaluation 9000-9004, and training
rollouts seed*100000 upward.

Usage:
    py -3.10 evaluate_final.py                 # 50 scenarios, all checkpoints
    py -3.10 evaluate_final.py --episodes 30
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

from train_marl import PolicyNet, DEVICE
from marl_env import MultiRobotEnv

HELD_OUT_SEED0 = 20_000
ABL_DIR = "results/ablation_runs"
ORIG_DIR = "results/training_runs"

# (configuration label, checkpoint path template over seed)
MODELS = [
    ("cold",            lambda s: f"{ORIG_DIR}/seed{s}/cold.pt"),
    ("warm_astar",      lambda s: f"{ORIG_DIR}/seed{s}/warm_astar.pt"),
    ("warm_bco",        lambda s: f"{ORIG_DIR}/seed{s}/warm_bco.pt"),
    ("warm_astar_cap",  lambda s: f"{ABL_DIR}/warm_astar_cap_s{s}.pt"),
    ("warm_bco_k1_cap", lambda s: f"{ABL_DIR}/warm_bco_k1_cap_s{s}.pt"),
    ("warm_bco_k3_cap", lambda s: f"{ABL_DIR}/warm_bco_k3_cap_s{s}.pt"),
    ("warm_bco_k5_cap", lambda s: f"{ABL_DIR}/warm_bco_k5_cap_s{s}.pt"),
]
SEEDS = (1, 2, 3)


def evaluate(net, episodes, seed0=HELD_OUT_SEED0):
    env = MultiRobotEnv(num_robots=5, num_moving=3, num_static=3)
    succ = reached = 0
    for e in range(episodes):
        obs = env.reset(seed=seed0 + e)
        done, info = False, {}
        while not done:
            actions, _lp, _v = net.act(obs, greedy=True)
            obs, _r, done, info = env.step(actions)
        succ += int(info["success"])
        reached += info["reached"]
    return succ / episodes * 100, reached / (episodes * env.num_robots) * 100


def wilson(k, n, z=1.96):
    """Wilson score interval, appropriate for a proportion from n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, (c - h)) * 100, min(1.0, (c + h)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="results/eval_final.csv")
    args = ap.parse_args()

    print(f"held-out evaluation: {args.episodes} scenarios, "
          f"seeds {HELD_OUT_SEED0}..{HELD_OUT_SEED0 + args.episodes - 1}")
    print(f"device: {DEVICE}\n")

    rows = []
    for label, path_of in MODELS:
        per_seed = []
        for s in SEEDS:
            path = path_of(s)
            if not os.path.exists(path):
                print(f"  [skip] missing {path}")
                continue
            net = PolicyNet().to(DEVICE)
            net.load_state_dict(torch.load(path, map_location=DEVICE))
            net.eval()
            with torch.no_grad():
                succ, reach = evaluate(net, args.episodes)
            per_seed.append((s, succ, reach))
            rows.append({"config": label, "seed": s,
                         "strict_success": round(succ, 2),
                         "reached": round(reach, 2),
                         "episodes": args.episodes})
            print(f"  {label:<17} seed {s}: reached {reach:5.1f}%  "
                  f"strict {succ:5.1f}%")
        if per_seed:
            r = [x[2] for x in per_seed]
            sc = [x[1] for x in per_seed]
            print(f"  {label:<17} MEAN    : reached {np.mean(r):5.1f}% "
                  f"+/-{np.std(r, ddof=1):4.1f}   "
                  f"strict {np.mean(sc):5.1f}% +/-{np.std(sc, ddof=1):4.1f}\n")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")

    # Pooled view: every seed's scenarios treated as one sample, with a Wilson
    # interval -- this is what the five-scenario protocol could never give.
    print(f"\npooled over 3 seeds ({3 * args.episodes} scenarios per config)")
    print(f"{'config':<18}{'reached':>10}{'strict':>9}"
          f"{'strict 95% CI':>20}")
    print("-" * 58)
    for label, _p in MODELS:
        sub = [r for r in rows if r["config"] == label]
        if not sub:
            continue
        n = sum(r["episodes"] for r in sub)
        k = sum(r["strict_success"] / 100 * r["episodes"] for r in sub)
        lo, hi = wilson(k, n)
        print(f"{label:<18}{np.mean([r['reached'] for r in sub]):>9.1f}%"
              f"{k / n * 100:>8.1f}%{f'[{lo:.1f}, {hi:.1f}]':>20}")


if __name__ == "__main__":
    main()
