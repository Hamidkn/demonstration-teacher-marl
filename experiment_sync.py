"""Robustness to imperfect twin synchronization (reviewer point R2.1).

A digital twin is synchronized from a plant, so whatever reasons about it acts
on a slightly stale and slightly noisy picture. This experiment degrades that
picture and asks which decision-maker survives it: the reactive A* replanner,
which needs current obstacle positions to replan around, or the learned policy,
which was trained with predictive features and can anticipate.

Both are run in the same environment, on the same scenarios, under the same
degradation, and are scored on the same strict criterion. Only perception is
degraded; collisions are always resolved against the true state.

Usage:
    py -3.10 experiment_sync.py                     # full grid
    py -3.10 experiment_sync.py --episodes 20       # quicker
"""

import argparse
import csv
import itertools
import os

import numpy as np
import torch

from marl_env import MultiRobotEnv
from train_marl import PolicyNet, DEVICE
from generate_demos import run_expert_episode, make_astar_planner

HELD_OUT_SEED0 = 20_000
ORIG_DIR = "results/training_runs"
POLICY = [f"{ORIG_DIR}/seed{s}/warm_bco.pt" for s in (1, 2, 3)]

LAGS = (0, 2, 4, 6)       # steps of staleness (DT = 0.5 s per step)
NOISES = (0.0, 3.0, 6.0)  # std of position error, workspace units


def run_policy(net, lag, noise, episodes):
    env = MultiRobotEnv(num_robots=5, num_moving=3, num_static=3,
                        sync_lag=lag, sync_noise=noise)
    succ = reached = 0
    for e in range(episodes):
        obs = env.reset(seed=HELD_OUT_SEED0 + e)
        done, info = False, {}
        while not done:
            with torch.no_grad():
                actions, _lp, _v = net.act(obs, greedy=True)
            obs, _r, done, info = env.step(actions)
        succ += int(info["success"])
        reached += info["reached"]
    return succ / episodes * 100, reached / (episodes * env.num_robots) * 100


def run_planner(lag, noise, episodes):
    env = MultiRobotEnv(num_robots=5, num_moving=3, num_static=3,
                        sync_lag=lag, sync_noise=noise)
    planner = make_astar_planner()
    succ = reached = 0
    for e in range(episodes):
        _o, _a, info = run_expert_episode(env, planner, HELD_OUT_SEED0 + e)
        succ += int(info.get("success", False))
        reached += info.get("reached", 0)
    return succ / episodes * 100, reached / (episodes * env.num_robots) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="results/sync_robustness.csv")
    args = ap.parse_args()

    nets = []
    for p in POLICY:
        n = PolicyNet().to(DEVICE)
        n.load_state_dict(torch.load(p, map_location=DEVICE))
        n.eval()
        nets.append(n)

    print(f"{args.episodes} held-out scenarios per cell, "
          f"policy averaged over {len(nets)} seeds\n")
    print(f"{'lag':>4}{'noise':>7} | {'A* replanner':>22} | {'learned policy':>24}")
    print(f"{'':>4}{'':>7} | {'reached':>10}{'strict':>12} | "
          f"{'reached':>12}{'strict':>12}")
    print("-" * 74)

    rows = []
    for lag, noise in itertools.product(LAGS, NOISES):
        p_s, p_r = run_planner(lag, noise, args.episodes)
        res = [run_policy(n, lag, noise, args.episodes) for n in nets]
        l_s, l_r = float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))
        l_sd = float(np.std([r[1] for r in res], ddof=1))
        rows.append({"lag": lag, "noise": noise,
                     "planner_reached": round(p_r, 1), "planner_strict": round(p_s, 1),
                     "policy_reached": round(l_r, 1), "policy_strict": round(l_s, 1),
                     "policy_reached_sd": round(l_sd, 1)})
        print(f"{lag:>4}{noise:>7.1f} | {p_r:>9.1f}%{p_s:>11.1f}% | "
              f"{l_r:>11.1f}%{l_s:>11.1f}%")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")

    base = rows[0]
    print("\ndegradation from the perfectly synchronized cell (lag 0, noise 0):")
    print(f"{'lag':>4}{'noise':>7}{'A* reached':>14}{'policy reached':>17}")
    print("-" * 44)
    for r in rows:
        dp = r["planner_reached"] - base["planner_reached"]
        dl = r["policy_reached"] - base["policy_reached"]
        print(f"{r['lag']:>4}{r['noise']:>7.1f}{dp:>+13.1f}{dl:>+16.1f}")


if __name__ == "__main__":
    main()
