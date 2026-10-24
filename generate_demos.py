"""
Generate expert demonstrations for MARL warm-start.

An expert = a planner (BCO-GA or A*) driving robots in MultiRobotEnv via
waypoint following with replanning — the same logic as the Digital Twin
hybrid, but recorded as (observation, discrete action) pairs.

Usage (local machine):
    py -3.10 generate_demos.py --planner bco   --episodes 30 --out demos_bco.npz
    py -3.10 generate_demos.py --planner astar --episodes 30 --out demos_astar.npz

The .npz files are then uploaded to Colab together with marl_env.py
and train_marl.py.
"""

import argparse
import numpy as np

from marl_env import (MultiRobotEnv, ACTIONS, N_ACTIONS,
                      GRID, SAFETY, GOAL_TOL, STEP_PX)

from bco_ga_planner import bco_ga_plan, bco_ga_plan_diverse
from digital_twin_env import astar          # grid A* (same as benchmark)


# ─────────────────────────────────────────
# Waypoint-following expert
# ─────────────────────────────────────────
def _path_blocked(waypoints, wp_index, obstacles) -> bool:
    for wp in waypoints[wp_index:]:
        for o in obstacles:
            if np.hypot(wp[0] - o.x, wp[1] - o.y) < o.r + SAFETY:
                return True
    return False


def _quantize(direction: np.ndarray) -> int:
    """Map a desired continuous direction to the best discrete action."""
    n = np.linalg.norm(direction)
    if n < 1e-6:
        return 0
    scores = ACTIONS[1:] @ (direction / n)
    return 1 + int(np.argmax(scores))


def run_expert_episode(env: MultiRobotEnv, planner, seed: int,
                       replan_cooldown: int = 4):
    """Roll one episode with the planner as expert; record (obs, action)."""
    obs = env.reset(seed=seed)

    n = env.num_robots
    waypoints = [None] * n
    wp_index = [0] * n
    cooldown = [0] * n

    def plan(i):
        wps = planner(env.pos[i].copy(), env.goal[i].copy(), env.perceived_obstacles())
        waypoints[i] = wps
        wp_index[i] = 0

    for i in range(n):
        plan(i)

    demo_obs, demo_act = [], []
    ep_info = {}

    for _t in range(env.max_steps):
        actions = np.zeros(n, dtype=int)
        for i in range(n):
            if env.reached[i]:
                continue

            # replan if needed (with cooldown, like the DT hybrid)
            need = waypoints[i] is None or _path_blocked(
                waypoints[i], wp_index[i], env.perceived_obstacles())
            if need and cooldown[i] == 0:
                plan(i)
                cooldown[i] = replan_cooldown
            cooldown[i] = max(0, cooldown[i] - 1)

            if not waypoints[i]:
                actions[i] = 0
                continue

            # advance waypoint index if close
            while (wp_index[i] < len(waypoints[i]) - 1
                   and np.linalg.norm(waypoints[i][wp_index[i]] - env.pos[i])
                       < STEP_PX):
                wp_index[i] += 1

            target = waypoints[i][min(wp_index[i], len(waypoints[i]) - 1)]
            actions[i] = _quantize(np.asarray(target) - env.pos[i])

            demo_obs.append(obs[i].copy())
            demo_act.append(actions[i])

        obs, _r, done, ep_info = env.step(actions)
        if done:
            break

    return demo_obs, demo_act, ep_info


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
# A* is deterministic (one path per query). BCO-GA is population-based, so we
# exploit it: `--variants K` harvests K diverse paths per query and rolls one
# episode per variant — a structural advantage A* cannot match.
def make_bco_planner(variant: int, k: int):
    def plan(s, g, obs):
        paths = bco_ga_plan_diverse(s, g, obs, k=k)
        return paths[variant % len(paths)]
    return plan


def make_astar_planner():
    def plan(s, g, obs):
        return astar(s, g, obs)
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["bco", "astar"], default="bco")
    ap.add_argument("--episodes", type=int, default=30, help="number of scenarios")
    ap.add_argument("--variants", type=int, default=1,
                    help="diverse rollouts per scenario (BCO only; A* ignores)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--robots", type=int, default=5)
    args = ap.parse_args()

    out = args.out or f"demos_{args.planner}.npz"
    variants = args.variants if args.planner == "bco" else 1

    env = MultiRobotEnv(num_robots=args.robots, num_moving=3, num_static=3)

    all_obs, all_act = [], []
    successes, reached_total, n_episodes = 0, 0, 0
    print(f"Generating {args.episodes} scenarios x {variants} variant(s) "
          f"with '{args.planner}'...")
    for ep in range(args.episodes):
        seed = 5000 + ep
        for v in range(variants):
            planner = (make_bco_planner(v, variants) if args.planner == "bco"
                       else make_astar_planner())
            o, a, info = run_expert_episode(env, planner, seed)
            all_obs.extend(o)
            all_act.extend(a)
            successes += int(info.get("success", False))
            reached_total += info.get("reached", 0)
            n_episodes += 1
        print(f"  scenario {ep+1:3d}/{args.episodes} done "
              f"(x{variants} variants)")

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_act, dtype=np.int64)
    np.savez_compressed(out, obs=obs_arr, actions=act_arr,
                        planner=args.planner)
    print(f"\nSaved {len(act_arr)} (obs, action) pairs from {n_episodes} "
          f"episodes -> {out}")
    print(f"Expert quality: success {successes}/{n_episodes} episodes, "
          f"avg reached {reached_total/max(n_episodes,1):.1f}/{args.robots}")
