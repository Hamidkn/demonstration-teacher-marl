"""
Digital Twin Environment for Multi-Robot Path Planning
Phase 1 (current) : A* path planning with real-time replanning
Phase 2 (next)    : A* replaced by BCO-GA planner
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING
import heapq

from bco_ga_planner import bco_ga_plan   # Phase 2 primary planner
from metaheuristic_planners import pso_plan, aco_plan, gwo_plan  # baselines

import time as _time
rng = np.random.default_rng(int(_time.time() * 1000) % (2**32))

GRID      = 200
CELL_SIZE = 5        # each grid cell = 5 units
COLS      = GRID // CELL_SIZE
ROWS      = GRID // CELL_SIZE
SAFETY    = 6        # extra clearance around obstacles (units)
STEP_PX   = 4.0     # robot movement speed per frame
ROBOT_R   = 7.0     # clearance radius used when robots avoid each other


# ─────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────
@dataclass
class Obstacle:
    x: float
    y: float
    r: float
    moving: bool = False
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Robot:
    id: int
    start: np.ndarray
    goal: np.ndarray
    color: str
    position: np.ndarray = field(init=False)
    path: List[np.ndarray] = field(default_factory=list)       # travelled history
    waypoints: List[np.ndarray] = field(default_factory=list)  # planned waypoints
    wp_index: int = 0
    reached: bool = False
    replan_count: int = 0
    wait_count: int = 0          # frames spent waiting for a moving obstacle
    cooldown: int = 0            # frames until next replan is allowed

    def __post_init__(self):
        self.position = self.start.copy()


# ─────────────────────────────────────────
# PHASE 1 — A* planner (will be replaced by BCO-GA in Phase 2)
# ─────────────────────────────────────────
def _world_to_grid(pt: np.ndarray) -> Tuple[int, int]:
    col = int(np.clip(pt[0] / CELL_SIZE, 0, COLS - 1))
    row = int(np.clip(pt[1] / CELL_SIZE, 0, ROWS - 1))
    return (row, col)


def _grid_to_world(row: int, col: int) -> np.ndarray:
    return np.array([col * CELL_SIZE + CELL_SIZE / 2,
                     row * CELL_SIZE + CELL_SIZE / 2])


def _build_occupancy(obstacles: List[Obstacle]) -> np.ndarray:
    """Mark grid cells that are inside any obstacle (+ safety margin)."""
    grid = np.zeros((ROWS, COLS), dtype=bool)
    for obs in obstacles:
        r_cells = int((obs.r + SAFETY) / CELL_SIZE) + 1
        oc, or_ = int(obs.x / CELL_SIZE), int(obs.y / CELL_SIZE)
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                nr, nc = or_ + dr, oc + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    cx = nc * CELL_SIZE + CELL_SIZE / 2
                    cy = nr * CELL_SIZE + CELL_SIZE / 2
                    if np.hypot(cx - obs.x, cy - obs.y) < obs.r + SAFETY:
                        grid[nr, nc] = True
    return grid


def astar(start: np.ndarray,
          goal: np.ndarray,
          obstacles: List[Obstacle]) -> Optional[List[np.ndarray]]:
    """
    PHASE 1 planner — shortest path on a discrete grid.
    Phase 2: this function will be replaced by bco_ga_plan().
    """
    occupancy = _build_occupancy(obstacles)
    s = _world_to_grid(start)
    g = _world_to_grid(goal)

    # If start or goal cell is blocked, find nearest free cell
    for target, name in [(s, "start"), (g, "goal")]:
        if occupancy[target]:
            found = False
            for dist in range(1, 20):
                for dr in range(-dist, dist + 1):
                    for dc in range(-dist, dist + 1):
                        nr, nc = target[0] + dr, target[1] + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS and not occupancy[nr, nc]:
                            if name == "start":
                                s = (nr, nc)
                            else:
                                g = (nr, nc)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    def h(a, b):
        # Octile distance — admissible for 8-connected grids with diagonal
        # cost sqrt(2); Manhattan would overestimate and break optimality.
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

    open_heap = [(h(s, g), 0, s, [s])]
    visited = set()
    neighbors_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while open_heap:
        _, cost, cur, path_so_far = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == g:
            world = [_grid_to_world(r, c) for r, c in path_so_far]
            return world
        for dr, dc in neighbors_8:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and not occupancy[nr, nc]:
                step_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                new_cost = cost + step_cost
                heapq.heappush(open_heap,
                               (new_cost + h((nr, nc), g),
                                new_cost, (nr, nc),
                                path_so_far + [(nr, nc)]))
    return None


# ─────────────────────────────────────────
# PHASE 2 — BCO-GA planner (replaces astar above)
# ─────────────────────────────────────────
# def bco_ga_plan(start: np.ndarray,
#                 goal: np.ndarray,
#                 obstacles: List[Obstacle]) -> Optional[List[np.ndarray]]:
#     """
#     BCO-GA planner imported from improved_bco_ga.py.
#     Returns smooth, optimised waypoints instead of grid-snapped A* path.
#     Called exactly where astar() is called now.
#     """
#     from improved_bco_ga import bco_phase, ga_phase, Config
#     # TODO: wire up in Phase 2
#     pass


# ─────────────────────────────────────────
# Path-blocked helper (shared by both phases)
# ─────────────────────────────────────────
def path_blocked(waypoints: List[np.ndarray],
                 wp_index: int,
                 obstacles: List[Obstacle]) -> bool:
    """Check if any upcoming waypoint is inside an obstacle."""
    for wp in waypoints[wp_index:]:
        for obs in obstacles:
            if np.hypot(wp[0] - obs.x, wp[1] - obs.y) < obs.r + SAFETY:
                return True
    return False


def _segment_clear(p0: np.ndarray, p1: np.ndarray,
                   obstacles: List[Obstacle], margin: float = SAFETY) -> bool:
    """True if the straight segment p0->p1 stays clear of all obstacles."""
    d = p1 - p0
    L = np.linalg.norm(d)
    steps = max(2, int(L / 2))
    for t in np.linspace(0, 1, steps):
        pt = p0 + t * d
        for obs in obstacles:
            if np.hypot(pt[0] - obs.x, pt[1] - obs.y) < obs.r + margin:
                return False
    return True


def smooth_path(waypoints: List[np.ndarray],
                obstacles: List[Obstacle]) -> List[np.ndarray]:
    """
    Line-of-sight shortcutting (string pulling):
    replace jagged A* waypoints with straight shortcuts where clear.
    """
    if len(waypoints) <= 2:
        return waypoints
    result = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        j = len(waypoints) - 1
        # find the farthest waypoint reachable in a straight clear line
        while j > i + 1:
            if _segment_clear(waypoints[i], waypoints[j], obstacles):
                break
            j -= 1
        result.append(waypoints[j])
        i = j
    return result


def predicted_obstacles(obstacles: List[Obstacle],
                        lookahead_steps: int = 5) -> List[Obstacle]:
    """
    Add phantom obstacles at the future positions of moving obstacles so the
    planner routes around where they WILL be, not just where they are now.
    """
    augmented = list(obstacles)
    for o in obstacles:
        if not o.moving:
            continue
        for k in range(1, lookahead_steps + 1):
            fx = o.x + o.vx * DigitalTwinEnv.DT * k * 2.0
            fy = o.y + o.vy * DigitalTwinEnv.DT * k * 2.0
            # bounce prediction off walls
            fx = np.clip(fx, o.r, GRID - o.r)
            fy = np.clip(fy, o.r, GRID - o.r)
            augmented.append(Obstacle(fx, fy, o.r * 0.85, moving=False))
    return augmented


# ─────────────────────────────────────────
# Digital Twin Environment
# ─────────────────────────────────────────
class DigitalTwinEnv:
    DT = 0.5   # time step (seconds)

    def __init__(self, num_robots: int = 5, num_moving_obs: int = 2,
                 num_static_obs: int = 3, mode: str = "hybrid",
                 seed: Optional[int] = None, verbose: bool = True):
        """
        mode:
          "hybrid" -> BCO-GA initial + predictive avoidance + A*/smoothing replan (proposed)
          "astar"  -> plain A* only (baseline 1)
          "bco_ga" -> BCO-GA only, no A* fallback, no prediction (baseline 2)
        seed: reseed the module RNG for a reproducible scenario across methods.
        """
        global rng
        if seed is not None:
            rng = np.random.default_rng(seed)
            # Also reseed the planners' RNGs so every method sees an
            # identical, reproducible stochastic search per scenario.
            import bco_ga_planner, metaheuristic_planners
            bco_ga_planner._rng = np.random.default_rng(seed + 1)
            metaheuristic_planners._rng = np.random.default_rng(seed + 2)
        self.mode = mode
        self.num_robots = num_robots
        self.robots: List[Robot] = []
        self.obstacles: List[Obstacle] = []
        self.time = 0.0
        self.collision_frames = 0        # frames a robot was inside an obstacle
        self.robot_collision_frames = 0  # frames two robots overlapped
        self.verbose = verbose           # set False for silent benchmark runs
        self._build(num_moving_obs, num_static_obs)
        self._plan_all()

    # ── Setup ─────────────────────────────
    def _build(self, num_moving: int, num_static: int = 3):
        palette = ["red", "blue", "green", "purple", "orange",
                   "cyan", "magenta", "yellow", "#f97316", "#84cc16"]

        # Varied random start/goal pairs (realistic — not all through centre).
        # Each pair is well-separated and spaced from other robots' endpoints.
        starts, goals = [], []
        placed = []   # every start & goal placed so far
        for i in range(self.num_robots):
            for attempt in range(200):
                start = rng.uniform(10, GRID - 10, 2)
                goal  = rng.uniform(10, GRID - 10, 2)
                # require a decent journey length, and spacing from others
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
            self.robots.append(Robot(i, start, goal, palette[i % len(palette)]))

        # Static obstacles — random positions each run, away from starts/goals
        margin = 22.0
        forbidden = [s for s in starts] + [g for g in goals]
        for _ in range(num_static):
            for attempt in range(100):
                x = float(rng.uniform(40, 160))
                y = float(rng.uniform(40, 160))
                r = float(rng.uniform(14, 22))
                # reject if too close to any start, goal, or existing obstacle
                too_close = any(
                    np.hypot(x - p[0], y - p[1]) < r + margin
                    for p in forbidden
                ) or any(
                    np.hypot(x - o.x, y - o.y) < r + o.r + 10
                    for o in self.obstacles
                )
                if not too_close:
                    self.obstacles.append(Obstacle(x, y, r))
                    forbidden.append(np.array([x, y]))
                    break

        # Moving obstacles — random position, speed, and direction each run
        for _ in range(num_moving):
            for attempt in range(100):
                x  = float(rng.uniform(20, 180))
                y  = float(rng.uniform(20, 180))
                r  = float(rng.uniform(10, 16))
                too_close = any(
                    np.hypot(x - p[0], y - p[1]) < r + margin
                    for p in forbidden
                ) or any(
                    np.hypot(x - o.x, y - o.y) < r + o.r + 10
                    for o in self.obstacles
                )
                if not too_close:
                    angle = rng.uniform(0, 2 * np.pi)
                    speed = rng.uniform(1.5, 3.5)
                    vx = float(np.cos(angle) * speed)
                    vy = float(np.sin(angle) * speed)
                    self.obstacles.append(
                        Obstacle(x, y, r, moving=True, vx=vx, vy=vy))
                    forbidden.append(np.array([x, y]))
                    break

    # ── Planning ──────────────────────────
    def _plan_all(self):
        # Initial planning uses BCO-GA (high quality, smooth paths)
        for rob in self.robots:
            self._plan_robot(rob, fast=False)

    def _escape_obstacle(self, rob: Robot):
        """If robot is inside an obstacle, push it to nearest free position."""
        if not self._too_close(rob.position, margin=0.0):
            return
        for radius in range(1, 40):
            for angle in np.linspace(0, 2 * np.pi, 16 * radius, endpoint=False):
                candidate = rob.position + np.array([
                    np.cos(angle) * radius * CELL_SIZE * 0.5,
                    np.sin(angle) * radius * CELL_SIZE * 0.5,
                ])
                candidate = np.clip(candidate, 5, GRID - 5)
                if not self._too_close(candidate, margin=SAFETY):
                    if self.verbose:
                        print(f"  [ESCAPE] Robot {rob.id+1} pushed out of obstacle")
                    rob.position = candidate
                    rob.path.append(rob.position.copy())
                    return

    def _plan_robot(self, rob: Robot, fast: bool = False):
        """
        Planner dispatch by self.mode:
          "hybrid" -> BCO-GA initial + predictive obstacles + A*/smoothing replan
          "astar"  -> plain A* only (no prediction, no smoothing)
          "bco_ga" -> BCO-GA only (no A* fallback, no prediction)
        `fast=True` means a real-time replan (used by step()).
        """
        self._escape_obstacle(rob)
        wps = None

        # Other active robots are treated as obstacles (inter-robot avoidance).
        # All modes get this, so the comparison is purely about the planner.
        others = [Obstacle(o.position[0], o.position[1], ROBOT_R)
                  for o in self.robots if o.id != rob.id and not o.reached]
        base_obs = self.obstacles + others

        if self.mode == "astar":
            wps = astar(rob.position, rob.goal, base_obs)

        elif self.mode == "bco_ga":
            # best-effort path (never freeze) — pure BCO-GA baseline behaviour
            wps = bco_ga_plan(rob.position, rob.goal, base_obs, strict=False)

        elif self.mode in ("pso", "aco", "gwo"):
            # metaheuristic baselines — same representation/fitness/budget,
            # best-effort like the pure BCO-GA baseline for a fair comparison
            planner = {"pso": pso_plan, "aco": aco_plan, "gwo": gwo_plan}[self.mode]
            wps = planner(rob.position, rob.goal, base_obs, strict=False)

        else:  # "hybrid" (proposed)
            # Predictive avoidance removed — it over-constrained the space and
            # mispredicted bouncing obstacles, hurting more than it helped.
            if not fast:
                wps = bco_ga_plan(rob.position, rob.goal, base_obs)
            if wps is None:                  # replan OR BCO-GA fallback
                wps = astar(rob.position, rob.goal, base_obs)
                if wps:
                    wps = smooth_path(wps, base_obs)

        if wps:
            rob.waypoints = wps
            rob.wp_index  = 0
        elif self.verbose:
            print(f"  [WARNING] Robot {rob.id+1}: no path found, will retry")

    # ── Collision helpers ─────────────────
    def _too_close(self, pos: np.ndarray, margin: float = 0.0) -> bool:
        for obs in self.obstacles:
            if np.hypot(pos[0] - obs.x, pos[1] - obs.y) < obs.r + margin:
                return True
        return False

    def _robot_ahead(self, rob: Robot) -> bool:
        """True if another active robot sits on our next step position."""
        nxt = self._next_step_pos(rob)
        if nxt is None:
            return False
        for other in self.robots:
            if other.id == rob.id or other.reached:
                continue
            if np.hypot(nxt[0] - other.position[0],
                        nxt[1] - other.position[1]) < 2 * ROBOT_R:
                return True
        return False

    def _robot_overlap(self, rob: Robot, pos: np.ndarray) -> bool:
        """True if pos would overlap another active robot's body."""
        for other in self.robots:
            if other.id == rob.id or other.reached:
                continue
            if np.hypot(pos[0] - other.position[0],
                        pos[1] - other.position[1]) < ROBOT_R:
                return True
        return False

    def _next_step_pos(self, rob: Robot) -> Optional[np.ndarray]:
        if rob.wp_index >= len(rob.waypoints):
            return None
        target = rob.waypoints[rob.wp_index]
        diff   = target - rob.position
        dist   = np.linalg.norm(diff)
        if dist < STEP_PX:
            return target.copy()
        return rob.position + (diff / dist) * STEP_PX

    # ── Simulation step ───────────────────
    def step(self):
        self.time += self.DT

        # 1. Move obstacles
        for obs in self.obstacles:
            if not obs.moving:
                continue
            obs.x += obs.vx * self.DT
            obs.y += obs.vy * self.DT
            # Bounce off walls
            if obs.x < obs.r or obs.x > GRID - obs.r:
                obs.vx *= -1
                obs.x = np.clip(obs.x, obs.r, GRID - obs.r)
            if obs.y < obs.r or obs.y > GRID - obs.r:
                obs.vy *= -1
                obs.y = np.clip(obs.y, obs.r, GRID - obs.r)
            # Bounce off static obstacles (reflect velocity, push out)
            for stat in self.obstacles:
                if stat.moving or stat is obs:
                    continue
                dx, dy = obs.x - stat.x, obs.y - stat.y
                d = np.hypot(dx, dy)
                min_d = obs.r + stat.r
                if 1e-6 < d < min_d:
                    nx, ny = dx / d, dy / d
                    obs.x += nx * (min_d - d)          # push out of overlap
                    obs.y += ny * (min_d - d)
                    vdotn = obs.vx * nx + obs.vy * ny  # reflect velocity
                    obs.vx -= 2 * vdotn * nx
                    obs.vy -= 2 * vdotn * ny

        # 2. Move robots with per-step collision check
        for rob in self.robots:
            if rob.reached or not rob.waypoints:
                continue

            needs_replan = False

            # a) Current position hit by moving obstacle
            if self._too_close(rob.position, margin=2.0):
                needs_replan = True

            # b) Next step would enter an obstacle
            if not needs_replan:
                next_pos = self._next_step_pos(rob)
                if next_pos is not None and self._too_close(next_pos, margin=SAFETY):
                    needs_replan = True

            # c) Upcoming waypoints blocked
            if not needs_replan and path_blocked(
                    rob.waypoints, rob.wp_index, self.obstacles):
                needs_replan = True

            # d) Another robot is right in front of us
            if not needs_replan and self._robot_ahead(rob):
                needs_replan = True

            if needs_replan:
                # Tick down cooldown; while active, WAIT in place instead of
                # replanning every frame (lets a moving obstacle pass by).
                if rob.cooldown > 0:
                    rob.cooldown -= 1
                    rob.wait_count += 1
                    rob.path.append(rob.position.copy())
                    continue

                rob.replan_count += 1
                rob.cooldown = 4   # don't replan again for a few frames
                if self.verbose:
                    print(f"  t={self.time:.1f}s | Robot {rob.id+1} replanning "
                          f"(#{rob.replan_count})")
                self._plan_robot(rob, fast=True)   # A* for low-latency replan

                # If the fresh path is STILL blocked (obstacle sitting on us),
                # just wait this frame rather than moving into it.
                if path_blocked(rob.waypoints, rob.wp_index, self.obstacles):
                    rob.wait_count += 1
                    rob.path.append(rob.position.copy())
                    continue
            else:
                rob.cooldown = 0   # path is clear, reset cooldown

            # Move one step
            if rob.wp_index < len(rob.waypoints):
                target = rob.waypoints[rob.wp_index]
                diff   = target - rob.position
                dist   = np.linalg.norm(diff)
                if dist < STEP_PX:
                    rob.position = target.copy()
                    rob.wp_index += 1
                else:
                    new_pos = rob.position + (diff / dist) * STEP_PX
                    # Gate: don't move into an obstacle OR onto another robot
                    if (not self._too_close(new_pos, margin=1.0)
                            and not self._robot_overlap(rob, new_pos)):
                        rob.position = new_pos
                    else:
                        rob.wait_count += 1

            rob.path.append(rob.position.copy())

            if np.linalg.norm(rob.position - rob.goal) < 8:
                rob.reached = True
                if self.verbose:
                    print(f"  t={self.time:.1f}s | Robot {rob.id+1} reached goal!")

        # 3. Count real collisions
        for rob in self.robots:
            if rob.reached:
                continue
            # robot vs obstacle
            for obs in self.obstacles:
                if np.hypot(rob.position[0] - obs.x,
                            rob.position[1] - obs.y) < obs.r:
                    self.collision_frames += 1
                    break
            # robot vs robot (count each overlapping pair once)
            for other in self.robots:
                if other.id <= rob.id or other.reached:
                    continue
                if np.hypot(rob.position[0] - other.position[0],
                            rob.position[1] - other.position[1]) < ROBOT_R:
                    self.robot_collision_frames += 1

    def all_reached(self) -> bool:
        return all(r.reached for r in self.robots)

    def reset(self):
        for r in self.robots:
            r.position     = r.start.copy()
            r.path         = [r.start.copy()]
            r.waypoints    = []
            r.wp_index     = 0
            r.reached      = False
            r.replan_count = 0
            r.wait_count   = 0
            r.cooldown     = 0
        self.time = 0.0
        self.collision_frames = 0
        self.robot_collision_frames = 0
        self._plan_all()

    # ── Headless benchmark run ─────────────
    def run_headless(self, max_steps: int = 1000) -> dict:
        """
        Run the simulation with no rendering until all robots reach the goal
        or max_steps is hit. Returns a metrics dict for benchmarking.
        """
        self.verbose = False
        steps = 0
        while not self.all_reached() and steps < max_steps:
            self.step()
            steps += 1

        # Total travelled distance + total turning angle (smoothness)
        total_len = 0.0
        total_turn = 0.0
        for r in self.robots:
            if len(r.path) > 2:
                arr = np.array(r.path)
                seg = np.diff(arr, axis=0)
                total_len += float(np.sum(np.linalg.norm(seg, axis=1)))
                # sum of heading changes between consecutive segments (radians)
                for i in range(len(seg) - 1):
                    n1, n2 = np.linalg.norm(seg[i]), np.linalg.norm(seg[i + 1])
                    if n1 < 1e-6 or n2 < 1e-6:
                        continue
                    cos_a = np.clip(np.dot(seg[i], seg[i + 1]) / (n1 * n2), -1, 1)
                    total_turn += np.arccos(cos_a)

        reached = sum(1 for r in self.robots if r.reached)
        total_collisions = self.collision_frames + self.robot_collision_frames
        all_reached = reached == self.num_robots
        # TRUE success = every robot reached the goal AND never collided
        success = all_reached and total_collisions == 0
        return {
            "mode":            self.mode,
            "success":         success,
            "all_reached":     all_reached,
            "reached":         reached,
            "num_robots":      self.num_robots,
            "steps":           steps,
            "sim_time":        round(self.time, 1),
            "path_length":     round(total_len, 1),
            "turn_angle":      round(np.degrees(total_turn), 1),  # smoothness (lower=smoother)
            "replans":         sum(r.replan_count for r in self.robots),
            "waits":           sum(r.wait_count for r in self.robots),
            "collision_frames": self.collision_frames,
            "robot_collisions": self.robot_collision_frames,
        }

    # ── Animation ─────────────────────────
    def animate(self, steps: int = 200, interval: int = 80):
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        ax.set_xlim(0, GRID)
        ax.set_ylim(0, GRID)
        ax.set_aspect("equal")
        ax.grid(True, color="#1e293b", linewidth=0.5)
        ax.set_title("Digital Twin — BCO-GA Planning + A* Replanning",
                     color="white", fontsize=11)
        ax.tick_params(colors="gray")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")

        path_lines = []
        for rob in self.robots:
            if rob.waypoints:
                wps = np.array(rob.waypoints)
                ln, = ax.plot(wps[:, 0], wps[:, 1],
                              "--", color=rob.color, linewidth=1, alpha=0.35)
            else:
                ln, = ax.plot([], [], "--", color=rob.color, linewidth=1, alpha=0.35)
            path_lines.append(ln)

        obs_patches = []
        for obs in self.obstacles:
            fc = "#ef444488" if obs.moving else "#64748b88"
            ec = "#ef4444"   if obs.moving else "#64748b"
            c = plt.Circle((obs.x, obs.y), obs.r, facecolor=fc,
                            edgecolor=ec, linewidth=1.5)
            ax.add_patch(c)
            obs_patches.append(c)

        robot_dots, trail_lines = [], []
        for rob in self.robots:
            dot, = ax.plot(*rob.position, "o", color=rob.color,
                           markersize=11, zorder=5)
            ax.plot(*rob.start, "s", color=rob.color, markersize=8, alpha=0.4)
            ax.plot(*rob.goal,  "*", color=rob.color, markersize=16)
            ax.text(rob.goal[0]+3, rob.goal[1]+3, f"G{rob.id+1}",
                    color=rob.color, fontsize=8)
            trail, = ax.plot([], [], "-", color=rob.color,
                             linewidth=1.5, alpha=0.6)
            robot_dots.append(dot)
            trail_lines.append(trail)

        info = ax.text(4, GRID - 12, "", color="#94a3b8", fontsize=8)

        def update(frame):
            if not self.all_reached():
                self.step()

            for i, obs in enumerate(self.obstacles):
                obs_patches[i].center = (obs.x, obs.y)

            for i, rob in enumerate(self.robots):
                robot_dots[i].set_data([rob.position[0]], [rob.position[1]])
                if len(rob.path) > 1:
                    arr = np.array(rob.path)
                    trail_lines[i].set_data(arr[:, 0], arr[:, 1])
                if rob.waypoints and rob.wp_index < len(rob.waypoints):
                    remaining = np.array(rob.waypoints[rob.wp_index:])
                    full = np.vstack([rob.position, remaining])
                    path_lines[i].set_data(full[:, 0], full[:, 1])

            reached = sum(1 for r in self.robots if r.reached)
            replans = sum(r.replan_count for r in self.robots)
            waits   = sum(r.wait_count for r in self.robots)
            info.set_text(
                f"t={self.time:.1f}s  |  reached: {reached}/{len(self.robots)}"
                f"  |  replans: {replans}  |  waits: {waits}"
            )
            return obs_patches + robot_dots + trail_lines + path_lines + [info]

        ani = FuncAnimation(fig, update, frames=steps,
                            interval=interval, blit=True)
        plt.tight_layout()
        plt.show()
        return ani


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # Hard scenario by default: 8 robots crossing the centre + 4 moving obstacles
    hard = "--easy" not in sys.argv
    if hard:
        print("Building HARD Digital Twin scenario...")
        env = DigitalTwinEnv(num_robots=8, num_moving_obs=4, num_static_obs=4)
    else:
        print("Building easy Digital Twin scenario...")
        env = DigitalTwinEnv(num_robots=5, num_moving_obs=2, num_static_obs=3)
    print(f"Robots   : {env.num_robots}")
    print(f"Obstacles: {len(env.obstacles)} "
          f"({sum(1 for o in env.obstacles if o.moving)} moving)")
    print("\nStarting animation (close window to stop)...")
    env.animate(steps=800, interval=40)
    print("Done.")
