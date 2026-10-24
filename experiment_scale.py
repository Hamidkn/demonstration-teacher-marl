"""Scaling in the number of robots (reviewer point R2.5).

All results in the paper use five robots, which is small next to the instances
the multi-agent pathfinding community reports. Two things can be tested cheaply
without retraining anything:

  * the reactive A* replanner, which has no fixed robot count at all;
  * the learned policy, which can be evaluated zero-shot at any robot count,
    because the observation is egocentric over the K nearest obstacles and
    robots and so has a fixed width (44) independent of how many robots exist.

That second property makes this a generalization test as well as a scaling one:
the policy was trained with five robots and has never seen a denser workspace.

Usage:
    py -3.10 experiment_scale.py                   # 5, 8, 10, 12 robots
    py -3.10 experiment_scale.py --episodes 20
"""

import argparse
import csv

import numpy as np
import torch

from marl_env import MultiRobotEnv
from train_marl import PolicyNet, DEVICE
from generate_demos import run_expert_episode, make_astar_planner

HELD_OUT_SEED0 = 20_000
ORIG_DIR = "results/training_runs"
POLICY = [f"{ORIG_DIR}/seed{s}/warm_bco.pt" for s in (1, 2, 3)]
COUNTS = (5, 8, 10, 12)


def run_policy(net, n_robots, episodes):
    env = MultiRobotEnv(num_robots=n_robots, num_moving=3, num_static=3)
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
    return succ / episodes * 100, reached / (episodes * n_robots) * 100


def run_planner(n_robots, episodes):
    env = MultiRobotEnv(num_robots=n_robots, num_moving=3, num_static=3)
    planner = make_astar_planner()
    succ = reached = 0
    for e in range(episodes):
        _o, _a, info = run_expert_episode(env, planner, HELD_OUT_SEED0 + e)
        succ += int(info.get("success", False))
        reached += info.get("reached", 0)
    return succ / episodes * 100, reached / (episodes * n_robots) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="results/scale_results.csv")
    args = ap.parse_args()

    nets = []
    for p in POLICY:
        n = PolicyNet().to(DEVICE)
        n.load_state_dict(torch.load(p, map_location=DEVICE))
        n.eval()
        nets.append(n)

    print(f"{args.episodes} held-out scenarios per cell; policies trained with "
          f"5 robots, evaluated zero-shot\n")
    print(f"{'robots':>7} | {'A* replanner':>22} | {'learned policy (zero-shot)':>30}")
    print(f"{'':>7} | {'reached':>10}{'strict':>12} | {'reached':>14}{'strict':>12}")
    print("-" * 78)

    rows = []
    for n_rob in COUNTS:
        p_s, p_r = run_planner(n_rob, args.episodes)
        res = [run_policy(net, n_rob, args.episodes) for net in nets]
        l_s = float(np.mean([r[0] for r in res]))
        l_r = float(np.mean([r[1] for r in res]))
        l_sd = float(np.std([r[1] for r in res], ddof=1))
        rows.append({"robots": n_rob,
                     "planner_reached": round(p_r, 1), "planner_strict": round(p_s, 1),
                     "policy_reached": round(l_r, 1), "policy_strict": round(l_s, 1),
                     "policy_reached_sd": round(l_sd, 1)})
        print(f"{n_rob:>7} | {p_r:>9.1f}%{p_s:>11.1f}% | "
              f"{l_r:>13.1f}%{l_s:>11.1f}%")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")

    b = rows[0]
    print("\nrelative to 5 robots:")
    print(f"{'robots':>7}{'A* reached':>14}{'policy reached':>17}")
    print("-" * 40)
    for r in rows:
        print(f"{r['robots']:>7}"
              f"{r['planner_reached'] - b['planner_reached']:>+13.1f}"
              f"{r['policy_reached'] - b['policy_reached']:>+16.1f}")


if __name__ == "__main__":
    main()
