"""
PSO / ACO / GWO single-robot planners for the Digital Twin.

Fair-comparison design (important for the paper):
  - SAME path representation as BCO-GA (perpendicular offsets from the
    straight start->goal line, NUM_WAYPOINTS control points)
  - SAME multi-objective fitness (length + smoothness + safety)
  - SAME evaluation budget (~800 fitness evaluations per plan)
  Only the search strategy differs -> the comparison isolates the optimizer.

Public API (drop-in, same signature as bco_ga_plan):
    pso_plan(start, goal, obstacles, strict=True)
    aco_plan(start, goal, obstacles, strict=True)   # ACO-R, continuous domain
    gwo_plan(start, goal, obstacles, strict=True)
"""

import numpy as np
from typing import List, Optional

from bco_ga_planner import (
    GRID, NUM_WAYPOINTS, OFFSET_MAX,
    _geometry, _genome_to_path, _fitness,
    _obstacle_positions, _obstacle_penalty_vec,
)

_rng = np.random.default_rng()

# Equal evaluation budget for all planners (~BCO-GA's total evals)
POP  = 20      # particles / ants / wolves
ITER = 40      # iterations  ->  ~POP*(ITER+1) = 820 evaluations


# ─────────────────────────────────────────
# PSO — standard constriction-coefficient particle swarm
# ─────────────────────────────────────────
def _pso_optimize(evaluate) -> np.ndarray:
    w, c1, c2 = 0.72, 1.49, 1.49            # Clerc constriction values
    vmax = OFFSET_MAX * 0.5

    X = _rng.uniform(-OFFSET_MAX, OFFSET_MAX, (POP, NUM_WAYPOINTS))
    V = np.zeros_like(X)
    fit = np.array([evaluate(x) for x in X])

    pbest, pbest_f = X.copy(), fit.copy()
    g = int(np.argmin(fit))
    gbest, gbest_f = X[g].copy(), fit[g]

    for _ in range(ITER):
        r1 = _rng.random((POP, NUM_WAYPOINTS))
        r2 = _rng.random((POP, NUM_WAYPOINTS))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, -OFFSET_MAX, OFFSET_MAX)

        fit = np.array([evaluate(x) for x in X])
        improved = fit < pbest_f
        pbest[improved] = X[improved]
        pbest_f[improved] = fit[improved]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), pbest_f[g]

    return gbest


# ─────────────────────────────────────────
# ACO-R — ant colony optimization for continuous domains
# (Socha & Dorigo 2008: solution archive + Gaussian sampling)
# ─────────────────────────────────────────
def _aco_optimize(evaluate) -> np.ndarray:
    k  = POP        # archive size
    q  = 0.5        # locality of search (rank-weight spread)
    xi = 0.85       # convergence speed (sigma scale)

    archive = _rng.uniform(-OFFSET_MAX, OFFSET_MAX, (k, NUM_WAYPOINTS))
    fit = np.array([evaluate(x) for x in archive])
    order = np.argsort(fit)
    archive, fit = archive[order], fit[order]

    ranks = np.arange(1, k + 1)
    w = np.exp(-((ranks - 1) ** 2) / (2 * (q * k) ** 2)) / (q * k * np.sqrt(2 * np.pi))
    p = w / w.sum()

    for _ in range(ITER):
        new = np.zeros((POP, NUM_WAYPOINTS))
        for m in range(POP):
            j = _rng.choice(k, p=p)                       # pick a guide by rank
            mean = archive[j]
            # per-dimension sigma: mean distance of archive to the guide
            sigma = xi * np.mean(np.abs(archive - mean), axis=0) + 1e-9
            new[m] = np.clip(_rng.normal(mean, sigma),
                             -OFFSET_MAX, OFFSET_MAX)
        new_fit = np.array([evaluate(x) for x in new])

        # merge, keep best k
        all_x = np.vstack([archive, new])
        all_f = np.concatenate([fit, new_fit])
        order = np.argsort(all_f)[:k]
        archive, fit = all_x[order], all_f[order]

    return archive[0]


# ─────────────────────────────────────────
# GWO — grey wolf optimizer (Mirjalili 2014)
# ─────────────────────────────────────────
def _gwo_optimize(evaluate) -> np.ndarray:
    X = _rng.uniform(-OFFSET_MAX, OFFSET_MAX, (POP, NUM_WAYPOINTS))
    fit = np.array([evaluate(x) for x in X])
    best_x = X[int(np.argmin(fit))].copy()      # running global best (alpha)
    best_f = float(fit.min())

    for t in range(ITER):
        order = np.argsort(fit)
        alpha = X[order[0]].copy()
        beta  = X[order[1]].copy()
        delta = X[order[2]].copy()

        a = 2.0 - 2.0 * t / ITER       # linearly decreasing 2 -> 0
        for i in range(POP):
            x_new = np.zeros(NUM_WAYPOINTS)
            for leader in (alpha, beta, delta):
                r1 = _rng.random(NUM_WAYPOINTS)
                r2 = _rng.random(NUM_WAYPOINTS)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - X[i])
                x_new += leader - A * D
            X[i] = np.clip(x_new / 3.0, -OFFSET_MAX, OFFSET_MAX)

        fit = np.array([evaluate(x) for x in X])
        if fit.min() < best_f:                   # carry the best across iters
            best_f = float(fit.min())
            best_x = X[int(np.argmin(fit))].copy()

    return best_x


# ─────────────────────────────────────────
# Shared plan wrapper — identical to bco_ga_plan's contract
# ─────────────────────────────────────────
def _plan_with(optimizer, start, goal, obstacles,
               strict: bool = True) -> Optional[List[np.ndarray]]:
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    base, perp = _geometry(start, goal)
    obs_pos, obs_rad = _obstacle_positions(obstacles)

    def evaluate(genome):
        return _fitness(_genome_to_path(genome, start, goal, base, perp),
                        obs_pos, obs_rad)

    best = optimizer(evaluate)
    path = _genome_to_path(best, start, goal, base, perp)

    if strict and _obstacle_penalty_vec(path, obs_pos, obs_rad) > 1e-6:
        return None
    return [path[i].copy() for i in range(len(path))]


def pso_plan(start, goal, obstacles, strict: bool = True):
    """Particle Swarm Optimization planner (single robot)."""
    return _plan_with(_pso_optimize, start, goal, obstacles, strict)


def aco_plan(start, goal, obstacles, strict: bool = True):
    """Continuous Ant Colony Optimization (ACO-R) planner (single robot)."""
    return _plan_with(_aco_optimize, start, goal, obstacles, strict)


def gwo_plan(start, goal, obstacles, strict: bool = True):
    """Grey Wolf Optimizer planner (single robot)."""
    return _plan_with(_gwo_optimize, start, goal, obstacles, strict)


# ─────────────────────────────────────────
# Self-test: all planners on one scenario, with timing
# ─────────────────────────────────────────
if __name__ == "__main__":
    from dataclasses import dataclass
    import time

    @dataclass
    class Obs:
        x: float; y: float; r: float

    obstacles = [Obs(60, 60, 18), Obs(120, 120, 18), Obs(100, 60, 15)]
    start = np.array([10., 10.])
    goal = np.array([190., 190.])

    from bco_ga_planner import bco_ga_plan
    planners = {
        "BCO-GA": bco_ga_plan,
        "PSO":    pso_plan,
        "ACO":    aco_plan,
        "GWO":    gwo_plan,
    }

    print(f"{'Planner':<8}{'time(ms)':>10}{'found':>7}{'length':>9}")
    print("-" * 36)
    for name, fn in planners.items():
        t0 = time.time()
        path = fn(start, goal, obstacles)
        dt = (time.time() - t0) * 1000
        if path is None:
            print(f"{name:<8}{dt:>10.0f}{'no':>7}{'-':>9}")
        else:
            arr = np.array(path)
            length = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
            print(f"{name:<8}{dt:>10.0f}{'yes':>7}{length:>9.1f}")
