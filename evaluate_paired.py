"""Per-scenario held-out evaluation, so comparisons can be analysed correctly.

Reviewer objection: the paper pooled 3 seeds x 50 scenarios into 150 and applied
an unpaired two-proportion z-test. That is wrong twice over. The same 50
scenario IDs are replayed for every policy, so outcomes are PAIRED by scenario;
and the three policies per configuration come from three training seeds, so
outcomes are CLUSTERED by seed. Treating 150 outcomes as independent inflates
significance.

This script stores the outcome of every (config, seed, scenario) so the right
analyses become possible: McNemar within a seed (paired by scenario), and
seed-level effects treated as the independent replicates they are.

Usage:
    py -3.10 evaluate_paired.py --episodes 50
"""

import argparse
import csv
import os

import torch

from train_marl import PolicyNet, DEVICE
from marl_env import MultiRobotEnv

HELD_OUT_SEED0 = 20_000
ABL_DIR = "results/ablation_runs"
ORIG_DIR = "results/training_runs"

MAIN_DIR = "results/main_runs"   # seeds 4+ , run locally


def _main(tag):
    """Seeds 1-3 came from Colab and live under ORIG_DIR; later seeds are local."""
    def path(s):
        if s <= 3:
            return f"{ORIG_DIR}/seed{s}/{tag}.pt"
        return os.path.join(MAIN_DIR, f"{tag}_s{s}.pt")
    return path


MODELS = [
    ("cold",            _main("cold")),
    ("warm_astar",      _main("warm_astar")),
    ("warm_bco",        _main("warm_bco")),
    ("warm_astar_cap",  lambda s: f"{ABL_DIR}/warm_astar_cap_s{s}.pt"),
    ("warm_bco_k1_cap", lambda s: f"{ABL_DIR}/warm_bco_k1_cap_s{s}.pt"),
    ("warm_bco_k3_cap", lambda s: f"{ABL_DIR}/warm_bco_k3_cap_s{s}.pt"),
    ("warm_bco_k5_cap", lambda s: f"{ABL_DIR}/warm_bco_k5_cap_s{s}.pt"),
]
SEEDS = tuple(range(1, 9))


def evaluate_per_scenario(net, episodes):
    env = MultiRobotEnv(num_robots=5, num_moving=3, num_static=3)
    out = []
    for e in range(episodes):
        obs = env.reset(seed=HELD_OUT_SEED0 + e)
        done, info = False, {}
        while not done:
            with torch.no_grad():
                actions, _lp, _v = net.act(obs, greedy=True)
            obs, _r, done, info = env.step(actions)
        out.append((HELD_OUT_SEED0 + e, int(info["success"]),
                    int(info["reached"]), env.num_robots))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="results/eval_per_scenario.csv")
    args = ap.parse_args()

    # Append after every model and skip what is already stored, so an
    # interrupted run loses at most one model rather than the whole evaluation.
    fields = ["config", "train_seed", "scenario", "success", "reached", "robots"]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["config"], int(r["train_seed"])))
        print(f"resuming: {len(done)} (config, seed) pairs already stored")
    else:
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    total = 0
    for label, path_of in MODELS:
        for s in SEEDS:
            if (label, s) in done:
                print(f"  {label:<17} seed {s}: already done")
                continue
            path = path_of(s)
            if not os.path.exists(path):
                print(f"  [skip] missing {path}")
                continue
            net = PolicyNet().to(DEVICE)
            net.load_state_dict(torch.load(path, map_location=DEVICE))
            net.eval()
            batch = [{"config": label, "train_seed": s, "scenario": scen,
                      "success": succ, "reached": reached, "robots": n_rob}
                     for scen, succ, reached, n_rob
                     in evaluate_per_scenario(net, args.episodes)]
            with open(args.out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerows(batch)
            total += len(batch)
            print(f"  {label:<17} seed {s}: done ({len(batch)} scenarios)")

    print(f"\n{args.out}: {total} new rows this run")


if __name__ == "__main__":
    main()
