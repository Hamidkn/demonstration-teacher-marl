"""
MultiRobotEnv — Gym-style multi-robot path-planning environment.

Self-contained (numpy only) so it can be uploaded to Google Colab together
with train_marl.py and a demonstration .npz file — no other project files needed.

World model matches the Digital Twin benchmark:
  200x200 arena, random static obstacles + moving (bouncing) obstacles,
  varied random start/goal pairs, N robots with inter-robot collision.

API (gym-like, multi-agent with shared policy in mind):
    env = MultiRobotEnv(num_robots=5, num_moving=3, num_static=3)
    obs = env.reset(seed=123)          # (N, OBS_DIM) float32
    obs, rewards, done, info = env.step(actions)   # actions: (N,) ints in [0..8]
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field, replace as _dc_replace
from typing import List, Optional, Tuple

GRID    = 200.0
STEP_PX = 4.0        # robot movement per step
ROBOT_R = 7.0        # robot body radius (collision + mutual avoidance)
SAFETY  = 6.0        # planner clearance (kept for demo generation parity)
DT      = 0.5        # seconds per step (obstacle motion)

# Discrete action set: stay + 8 compass directions
_DIRS = [(0.0, 0.0)]
for k in range(8):
    a = 2.0 * np.pi * k / 8.0
    _DIRS.append((np.cos(a), np.sin(a)))
ACTIONS = np.array(_DIRS)          # (9, 2)
N_ACTIONS = len(ACTIONS)           # 9

# Observation layout (per robot)
K_OBS = 4       # nearest obstacles observed
K_ROB = 3       # nearest other robots observed
PRED_H = 6      # horizon (steps) for obstacle-motion prediction features
# per obstacle: [dx, dy, r, vx, vy, pred_dx@H, pred_dy@H] = 7 features
OBS_DIM = 2 + 1 + K_OBS * 7 + K_ROB * 3 + 4   # = 44

GOAL_TOL = 8.0       # distance considered "reached"


def _reflect(v: float, lo: float, hi: float) -> float:
    """Reflect a value into [lo, hi] once (approx. single wall bounce)."""
    if v < lo:
        return lo + (lo - v)
    if v > hi:
        return hi - (v - hi)
    return v


@dataclass
class Obstacle:
    x: float
    y: float
    r: float
    moving: bool = False
    vx: float = 0.0
    vy: float = 0.0


class MultiRobotEnv:
    def __init__(self, num_robots: int = 5, num_moving: int = 3,
                 num_static: int = 3, max_steps: int = 300,
                 sync_lag: int = 0, sync_noise: float = 0.0):
        """sync_lag / sync_noise model an imperfect digital twin.

        A twin is synchronized from a plant, so its picture of the world is
        slightly stale and slightly wrong. `sync_lag` is how many steps old the
        obstacle state available to a decision-maker is; `sync_noise` is the
        standard deviation, in workspace units, of the error on each obstacle
        coordinate. Both affect PERCEPTION only: collisions, motion and the
        success criterion are always evaluated against the true state, exactly
        as a physical plant would resolve them regardless of what the twin
        believed. With the defaults (0, 0.0) the environment is unchanged.
        """
        self.num_robots = num_robots
        self.num_moving = num_moving
        self.num_static = num_static
        self.max_steps = max_steps
        self.sync_lag = sync_lag
        self.sync_noise = sync_noise
        self._sync_buf: deque = deque(maxlen=max(1, sync_lag + 1))
        self._perceived: List[Obstacle] = []
        self.rng = np.random.default_rng()
        # populated by reset()
        self.obstacles: List[Obstacle] = []
        self.pos = np.zeros((num_robots, 2))
        self.goal = np.zeros((num_robots, 2))
        self.reached = np.zeros(num_robots, dtype=bool)
        self.t = 0
        self.collision_frames = 0
        self.robot_collision_frames = 0
        self._prev_dist = np.zeros(num_robots)

    # ── Scenario generation (same style as the DT benchmark) ──────────
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng

        # Varied random start/goal pairs, well separated
        placed: List[np.ndarray] = []
        starts, goals = [], []
        for _ in range(self.num_robots):
            start = rng.uniform(10, GRID - 10, 2)
            goal = rng.uniform(10, GRID - 10, 2)
            for _attempt in range(200):
                start = rng.uniform(10, GRID - 10, 2)
                goal = rng.uniform(10, GRID - 10, 2)
                if np.linalg.norm(goal - start) < GRID * 0.45:
                    continue
                if any(np.linalg.norm(start - p) < 18 for p in placed):
                    continue
                if any(np.linalg.norm(goal - p) < 18 for p in placed):
                    continue
                break
            starts.append(start)
            goals.append(goal)
            placed.extend([start, goal])

        self.pos = np.array(starts)
        self.goal = np.array(goals)

        # Static obstacles
        self.obstacles = []
        margin = 22.0
        forbidden = placed[:]
        for _ in range(self.num_static):
            for _attempt in range(100):
                x = float(rng.uniform(40, 160))
                y = float(rng.uniform(40, 160))
                r = float(rng.uniform(14, 22))
                if any(np.hypot(x - p[0], y - p[1]) < r + margin for p in forbidden):
                    continue
                if any(np.hypot(x - o.x, y - o.y) < r + o.r + 10 for o in self.obstacles):
                    continue
                self.obstacles.append(Obstacle(x, y, r))
                forbidden.append(np.array([x, y]))
                break

        # Moving obstacles
        for _ in range(self.num_moving):
            for _attempt in range(100):
                x = float(rng.uniform(20, 180))
                y = float(rng.uniform(20, 180))
                r = float(rng.uniform(10, 16))
                if any(np.hypot(x - p[0], y - p[1]) < r + margin for p in forbidden):
                    continue
                if any(np.hypot(x - o.x, y - o.y) < r + o.r + 10 for o in self.obstacles):
                    continue
                angle = rng.uniform(0, 2 * np.pi)
                speed = rng.uniform(1.5, 3.5)
                self.obstacles.append(Obstacle(
                    x, y, r, moving=True,
                    vx=float(np.cos(angle) * speed),
                    vy=float(np.sin(angle) * speed)))
                forbidden.append(np.array([x, y]))
                break

        self.reached = np.zeros(self.num_robots, dtype=bool)
        self.t = 0
        self.collision_frames = 0
        self.robot_collision_frames = 0
        self._prev_dist = np.linalg.norm(self.goal - self.pos, axis=1)
        self._sync_buf.clear()
        self._sync()
        return self._observe()

    # ── Obstacle physics (identical to the Digital Twin) ──────────────
    def _move_obstacles(self):
        for obs in self.obstacles:
            if not obs.moving:
                continue
            obs.x += obs.vx * DT
            obs.y += obs.vy * DT
            if obs.x < obs.r or obs.x > GRID - obs.r:
                obs.vx *= -1
                obs.x = float(np.clip(obs.x, obs.r, GRID - obs.r))
            if obs.y < obs.r or obs.y > GRID - obs.r:
                obs.vy *= -1
                obs.y = float(np.clip(obs.y, obs.r, GRID - obs.r))
            for stat in self.obstacles:
                if stat.moving or stat is obs:
                    continue
                dx, dy = obs.x - stat.x, obs.y - stat.y
                d = np.hypot(dx, dy)
                min_d = obs.r + stat.r
                if 1e-6 < d < min_d:
                    nx, ny = dx / d, dy / d
                    obs.x += nx * (min_d - d)
                    obs.y += ny * (min_d - d)
                    vdotn = obs.vx * nx + obs.vy * ny
                    obs.vx -= 2 * vdotn * nx
                    obs.vy -= 2 * vdotn * ny

    # ── Collision helpers ──────────────────────────────────────────────
    def _in_obstacle(self, p: np.ndarray, margin: float = 0.0) -> bool:
        for o in self.obstacles:
            if np.hypot(p[0] - o.x, p[1] - o.y) < o.r + margin:
                return True
        return False

    def _robot_overlap(self, i: int, p: np.ndarray) -> bool:
        for j in range(self.num_robots):
            if j == i or self.reached[j]:
                continue
            if np.linalg.norm(p - self.pos[j]) < ROBOT_R:
                return True
        return False

    # ── Step ───────────────────────────────────────────────────────────
    def step(self, actions: np.ndarray):
        self.t += 1
        self._move_obstacles()

        rewards = np.zeros(self.num_robots, dtype=np.float32)

        for i in range(self.num_robots):
            if self.reached[i]:
                continue

            rewards[i] -= 0.02                       # step cost

            a = int(actions[i])
            direction = ACTIONS[a]
            proposed = self.pos[i] + direction * STEP_PX
            proposed = np.clip(proposed, 2.0, GRID - 2.0)

            # movement gate: never step into an obstacle or another robot
            blocked_obs = self._in_obstacle(proposed, margin=1.0)
            blocked_rob = self._robot_overlap(i, proposed)
            if not blocked_obs and not blocked_rob:
                self.pos[i] = proposed
            else:
                rewards[i] -= 0.1                    # bumped into something

            # progress reward (potential-based shaping)
            new_dist = float(np.linalg.norm(self.goal[i] - self.pos[i]))
            rewards[i] += (self._prev_dist[i] - new_dist) * 0.25
            self._prev_dist[i] = new_dist

            # hard collision states (moving obstacle ran us over)
            if self._in_obstacle(self.pos[i], margin=0.0):
                rewards[i] -= 2.0
                self.collision_frames += 1

            # caution shaping: mild penalty for hugging an obstacle, to
            # encourage a safety margin and lift the strict-success rate
            near = min((np.hypot(self.pos[i][0] - o.x, self.pos[i][1] - o.y)
                        - o.r for o in self.obstacles), default=99.0)
            if near < 12.0:
                rewards[i] -= 0.05 * (12.0 - near) / 12.0

            if new_dist < GOAL_TOL:
                self.reached[i] = True
                rewards[i] += 20.0

        # robot-robot body overlap accounting (pairs)
        for i in range(self.num_robots):
            if self.reached[i]:
                continue
            for j in range(i + 1, self.num_robots):
                if self.reached[j]:
                    continue
                if np.linalg.norm(self.pos[i] - self.pos[j]) < ROBOT_R:
                    self.robot_collision_frames += 1
                    rewards[i] -= 1.0
                    rewards[j] -= 1.0

        done = bool(self.reached.all() or self.t >= self.max_steps)
        info = {
            "reached": int(self.reached.sum()),
            "all_reached": bool(self.reached.all()),
            "collisions": self.collision_frames + self.robot_collision_frames,
            "success": bool(self.reached.all()
                            and self.collision_frames == 0
                            and self.robot_collision_frames == 0),
            "t": self.t,
        }
        self._sync()
        return self._observe(), rewards, done, info

    # ── Observation ────────────────────────────────────────────────────
    # ── Twin synchronization ─────────────────────────────────────────
    def _sync(self):
        """Refresh the twin's picture of the obstacles.

        Called once per step so that every robot — and any planner querying
        the twin — reasons about one shared state estimate, as they would if
        the estimate came from a single synchronization channel. The estimate
        is `sync_lag` steps old and carries Gaussian position error.
        """
        self._sync_buf.append([(o.x, o.y, o.vx, o.vy) for o in self.obstacles])
        snap = self._sync_buf[0]          # oldest retained == lag steps back
        if self.sync_noise <= 0.0:
            self._perceived = [_dc_replace(o, x=s[0], y=s[1], vx=s[2], vy=s[3])
                               for o, s in zip(self.obstacles, snap)]
            return
        noise = self.rng.normal(0.0, self.sync_noise, size=(len(snap), 2))
        self._perceived = [
            _dc_replace(o, x=s[0] + n[0], y=s[1] + n[1], vx=s[2], vy=s[3])
            for o, s, n in zip(self.obstacles, snap, noise)
        ]

    def perceived_obstacles(self) -> List[Obstacle]:
        """The obstacle state a decision-maker is entitled to use.

        Planners should call this rather than reading `obstacles` directly, so
        that a planner and a learned policy are compared on the same
        information. Physics and collision scoring still use `obstacles`.
        """
        return self._perceived if self._perceived else self.obstacles

    def _observe(self) -> np.ndarray:
        obs = np.zeros((self.num_robots, OBS_DIM), dtype=np.float32)
        for i in range(self.num_robots):
            p = self.pos[i]
            g = self.goal[i]
            k = 0
            # goal vector + distance
            obs[i, 0:2] = (g - p) / GRID
            obs[i, 2] = np.linalg.norm(g - p) / GRID
            k = 3
            # K nearest obstacles:
            # [dx, dy, r, vx, vy, predicted_dx@H, predicted_dy@H] (normalized).
            # The predicted terms tell the policy where each obstacle WILL be in
            # PRED_H steps (linear extrapolation, reflected at walls) — the
            # anticipation signal a reactive planner does not use.
            dists = [(np.hypot(o.x - p[0], o.y - p[1]) - o.r, o)
                     for o in self.perceived_obstacles()]
            dists.sort(key=lambda z: z[0])
            for _d, o in dists[:K_OBS]:
                obs[i, k:k+2] = [(o.x - p[0]) / GRID, (o.y - p[1]) / GRID]
                obs[i, k+2] = o.r / GRID
                obs[i, k+3] = o.vx / 5.0
                obs[i, k+4] = o.vy / 5.0
                fx = o.x + o.vx * DT * PRED_H
                fy = o.y + o.vy * DT * PRED_H
                fx = _reflect(fx, o.r, GRID - o.r)      # approx wall bounce
                fy = _reflect(fy, o.r, GRID - o.r)
                obs[i, k+5] = (fx - p[0]) / GRID
                obs[i, k+6] = (fy - p[1]) / GRID
                k += 7
            k = 3 + K_OBS * 7
            # K nearest other robots: [dx, dy, active]
            others = [(np.linalg.norm(self.pos[j] - p), j)
                      for j in range(self.num_robots) if j != i]
            others.sort(key=lambda z: z[0])
            for _d, j in others[:K_ROB]:
                obs[i, k:k+2] = (self.pos[j] - p) / GRID
                obs[i, k+2] = 0.0 if self.reached[j] else 1.0
                k += 3
            k = 3 + K_OBS * 7 + K_ROB * 3
            # wall distances
            obs[i, k:k+4] = [p[0] / GRID, p[1] / GRID,
                             (GRID - p[0]) / GRID, (GRID - p[1]) / GRID]
        return obs


# ─────────────────────────────────────────
# Self-test: random policy rollout
# ─────────────────────────────────────────
if __name__ == "__main__":
    env = MultiRobotEnv(num_robots=5, num_moving=3, num_static=3)
    obs = env.reset(seed=42)
    print(f"OBS_DIM = {OBS_DIM}, obs shape = {obs.shape}, actions = {N_ACTIONS}")

    rng = np.random.default_rng(0)
    total = np.zeros(env.num_robots)
    for t in range(300):
        actions = rng.integers(0, N_ACTIONS, env.num_robots)
        obs, r, done, info = env.step(actions)
        total += r
        if done:
            break
    print(f"Random policy: t={info['t']}, reached={info['reached']}/5, "
          f"collisions={info['collisions']}, success={info['success']}")
    print(f"Mean episode return per robot: {total.mean():.1f}")

    # Greedy-to-goal policy sanity check (should reach more than random)
    obs = env.reset(seed=42)
    total = np.zeros(env.num_robots)
    for t in range(300):
        acts = []
        for i in range(env.num_robots):
            d = env.goal[i] - env.pos[i]
            n = np.linalg.norm(d)
            if n < 1e-6:
                acts.append(0)
                continue
            scores = ACTIONS[1:] @ (d / n)
            acts.append(1 + int(np.argmax(scores)))
        obs, r, done, info = env.step(np.array(acts))
        total += r
        if done:
            break
    print(f"Greedy policy: t={info['t']}, reached={info['reached']}/5, "
          f"collisions={info['collisions']}, success={info['success']}")
    print(f"Mean episode return per robot: {total.mean():.1f}")
