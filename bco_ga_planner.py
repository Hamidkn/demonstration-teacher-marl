"""
Single-robot BCO-GA planner for the Digital Twin.
Same call signature as astar(start, goal, obstacles).

Key design (Phase 2):
  - Path = perpendicular offsets from the straight start->goal line.
    Guarantees forward progress (no zigzag) and halves the search space.
  - Vectorised obstacle penalty (numpy) for speed.
  - BCO seeds the population, GA refines it, Levy-flight scouts explore.

The multi-robot version stays in improved_bco_ga.py for the paper benchmark.
"""

import numpy as np
from typing import List, Optional

# ─────────────────────────────────────────
# Planner parameters (tuned for real-time, one robot at a time)
# ─────────────────────────────────────────
GRID          = 200
NUM_WAYPOINTS = 6      # intermediate control points along the route
NUM_BEES      = 16
MAX_CYCLES    = 15
LIMIT         = 5      # scout trigger threshold
POP_SIZE      = 16
MAX_GEN       = 15
MUT_INIT      = 0.30
MUT_FINAL     = 0.03
LEVY_BETA     = 1.5
SAFETY        = 6.0    # must match digital_twin_env.SAFETY
OFFSET_MAX    = 90.0   # max perpendicular deviation from the straight line
SAMPLES       = 6      # collision samples per segment

_rng = np.random.default_rng()


# ─────────────────────────────────────────
# Representation
#   genome = NUM_WAYPOINTS perpendicular offsets (scalars).
#   waypoint_i = S + t_i*(G-S) + offset_i * perpendicular_unit
# ─────────────────────────────────────────
def _geometry(start, goal):
    """Return (t values, base points on the line, perpendicular unit vector)."""
    direction = goal - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        direction = np.array([1.0, 0.0])
        length = 1.0
    d_unit = direction / length
    perp = np.array([-d_unit[1], d_unit[0]])          # rotate 90 deg
    t = np.linspace(0, 1, NUM_WAYPOINTS + 2)[1:-1]     # interior fractions
    base = start[None, :] + t[:, None] * (goal - start)[None, :]
    return base, perp


def _genome_to_path(genome, start, goal, base, perp) -> np.ndarray:
    """Convert perpendicular offsets to a full path incl. start & goal."""
    interior = base + genome[:, None] * perp[None, :]
    interior = np.clip(interior, 5, GRID - 5)
    return np.vstack([start, interior, goal])


# ─────────────────────────────────────────
# Fitness (vectorised) — lower is better
# ─────────────────────────────────────────
def _obstacle_positions(obstacles):
    pos = np.array([[o.x, o.y] for o in obstacles], dtype=float)
    rad = np.array([o.r for o in obstacles], dtype=float)
    return pos, rad


def _path_length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _path_smoothness(path):
    v = np.diff(path, axis=0)
    n = np.linalg.norm(v, axis=1)
    angles = 0.0
    for i in range(len(v) - 1):
        if n[i] < 1e-6 or n[i + 1] < 1e-6:
            continue
        cos_a = np.clip(np.dot(v[i], v[i + 1]) / (n[i] * n[i + 1]), -1, 1)
        angles += np.arccos(cos_a)          # turning angle (0=straight, pi=U-turn)
    return angles


def _obstacle_penalty_vec(path, obs_pos, obs_rad):
    """Sample along every segment, compute distance to all obstacles at once."""
    if len(obs_pos) == 0:
        return 0.0
    # Build sample points: (num_segments * SAMPLES, 2)
    p0 = path[:-1]                      # (S, 2)
    p1 = path[1:]                       # (S, 2)
    ts = np.linspace(0, 1, SAMPLES)     # (SAMPLES,)
    # samples[s, k] = p0[s] + ts[k]*(p1[s]-p0[s])
    samples = (p0[:, None, :]
               + ts[None, :, None] * (p1 - p0)[:, None, :]).reshape(-1, 2)
    # distances (M, K)
    diff = samples[:, None, :] - obs_pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    thresh = obs_rad[None, :] + SAFETY
    violation = np.clip(thresh - dist, 0, None)
    return float(violation.sum() * 100.0)


def _fitness(path, obs_pos, obs_rad):
    length = _path_length(path)
    smooth = _path_smoothness(path)
    obs_pen = _obstacle_penalty_vec(path, obs_pos, obs_rad)
    return (
        0.5 * length / (GRID * NUM_WAYPOINTS)
        + 0.2 * smooth / (NUM_WAYPOINTS * np.pi)
        + 0.3 * obs_pen / 1000.0
    )


# ─────────────────────────────────────────
# Lévy flight (scout bees)
# ─────────────────────────────────────────
def _levy_step(beta=LEVY_BETA):
    from math import gamma, sin, pi
    sigma = (
        gamma(1 + beta) * sin(pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = _rng.normal(0, sigma)
    v = _rng.normal(0, 1)
    return u / (abs(v) ** (1 / beta))


def _random_genome():
    return _rng.uniform(-OFFSET_MAX, OFFSET_MAX, NUM_WAYPOINTS)


def _scout_genome():
    g = _random_genome()
    g += _levy_step() * 15
    return np.clip(g, -OFFSET_MAX, OFFSET_MAX)


# ─────────────────────────────────────────
# BCO phase → GA phase (operate on genomes, not full paths)
# ─────────────────────────────────────────
def _bco_phase(start, goal, base, perp, obs_pos, obs_rad):
    pop = [_random_genome() for _ in range(NUM_BEES)]
    fit = [_fitness(_genome_to_path(g, start, goal, base, perp), obs_pos, obs_rad)
           for g in pop]
    trial = [0] * NUM_BEES

    def evaluate(g):
        return _fitness(_genome_to_path(g, start, goal, base, perp), obs_pos, obs_rad)

    for _ in range(MAX_CYCLES):
        # Employed bees
        for i in range(NUM_BEES):
            j = _rng.integers(0, NUM_BEES)
            while j == i:
                j = _rng.integers(0, NUM_BEES)
            phi = _rng.uniform(-1, 1, NUM_WAYPOINTS)
            cand = np.clip(pop[i] + phi * (pop[i] - pop[j]), -OFFSET_MAX, OFFSET_MAX)
            fc = evaluate(cand)
            if fc < fit[i]:
                pop[i], fit[i], trial[i] = cand, fc, 0
            else:
                trial[i] += 1

        # Onlooker bees
        inv = np.array([1 / (f + 1e-9) for f in fit])
        probs = inv / inv.sum()
        for _ in range(NUM_BEES):
            i = _rng.choice(NUM_BEES, p=probs)
            j = _rng.integers(0, NUM_BEES)
            while j == i:
                j = _rng.integers(0, NUM_BEES)
            phi = _rng.uniform(-1, 1, NUM_WAYPOINTS)
            cand = np.clip(pop[i] + phi * (pop[i] - pop[j]), -OFFSET_MAX, OFFSET_MAX)
            fc = evaluate(cand)
            if fc < fit[i]:
                pop[i], fit[i], trial[i] = cand, fc, 0

        # Scout bees (Lévy)
        for i in range(NUM_BEES):
            if trial[i] > LIMIT:
                pop[i] = _scout_genome()
                fit[i] = evaluate(pop[i])
                trial[i] = 0

    ranked = sorted(zip(fit, range(NUM_BEES)))
    return [pop[idx] for _, idx in ranked[:POP_SIZE]]


def _crossover(g1, g2):
    pt = _rng.integers(1, NUM_WAYPOINTS)
    c1 = np.concatenate([g1[:pt], g2[pt:]])
    c2 = np.concatenate([g2[:pt], g1[pt:]])
    return c1, c2


def _mutate(g, rate):
    res = g.copy()
    mask = _rng.random(NUM_WAYPOINTS) < rate
    res[mask] += _rng.uniform(-25, 25, mask.sum())
    return np.clip(res, -OFFSET_MAX, OFFSET_MAX)


def _ga_phase(seed_pop, start, goal, base, perp, obs_pos, obs_rad):
    pop = seed_pop[:]

    def evaluate(g):
        return _fitness(_genome_to_path(g, start, goal, base, perp), obs_pos, obs_rad)

    fit = [evaluate(g) for g in pop]

    for gen in range(MAX_GEN):
        rate = MUT_INIT * (1 - gen / MAX_GEN) + MUT_FINAL * (gen / MAX_GEN)
        new_pop = []
        while len(new_pop) < POP_SIZE:
            idxs = _rng.choice(len(pop), 4, replace=False)
            p1 = pop[idxs[np.argmin([fit[i] for i in idxs[:2]])]]
            p2 = pop[idxs[2 + np.argmin([fit[i] for i in idxs[2:]])]]
            c1, c2 = _crossover(p1, p2)
            new_pop.append(_mutate(c1, rate))
            new_pop.append(_mutate(c2, rate))

        combined = pop + new_pop
        cfit = fit + [evaluate(g) for g in new_pop]
        ranked = sorted(zip(cfit, range(len(combined))))
        pop = [combined[idx] for _, idx in ranked[:POP_SIZE]]
        fit = [f for f, _ in ranked[:POP_SIZE]]

    # return the whole final population, sorted best-first (genomes)
    return pop


def _select_diverse(genomes, start, goal, base, perp, obs_pos, obs_rad, k):
    """
    Greedy max-min selection of k diverse genomes from a fitness-sorted list.
    Prefers collision-free genomes; the best one is always included first.
    Returns a list of world-space paths.
    """
    # rank collision-free first, keeping fitness order within each group
    clean, dirty = [], []
    for g in genomes:
        p = _genome_to_path(g, start, goal, base, perp)
        (clean if _obstacle_penalty_vec(p, obs_pos, obs_rad) <= 1e-6
         else dirty).append(g)
    pool = clean if clean else dirty          # fall back to best-effort
    selected = [pool[0]]
    while len(selected) < k and len(selected) < len(pool):
        best_cand, best_dist = None, -1.0
        for g in pool:
            if any(np.array_equal(g, s) for s in selected):
                continue
            d = min(float(np.linalg.norm(g - s)) for s in selected)
            if d > best_dist:
                best_dist, best_cand = d, g
        if best_cand is None:
            break
        selected.append(best_cand)
    return [[p.copy() for p in _genome_to_path(g, start, goal, base, perp)]
            for g in selected]


# ─────────────────────────────────────────
# Public API — drop-in replacement for astar()
# ─────────────────────────────────────────
def bco_ga_plan(start, goal, obstacles, strict: bool = True) -> Optional[List[np.ndarray]]:
    """
    Plan a path for ONE robot using BCO-GA.
    Returns a list of world-space waypoints (incl. start & goal).

    strict=True  -> return None if the best path still collides
                    (used by the hybrid planner so it can fall back to A*).
    strict=False -> always return the best-effort path even if it collides
                    (used by the pure BCO-GA baseline so robots keep moving).
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    base, perp = _geometry(start, goal)
    obs_pos, obs_rad = _obstacle_positions(obstacles)

    seed = _bco_phase(start, goal, base, perp, obs_pos, obs_rad)
    pop = _ga_phase(seed, start, goal, base, perp, obs_pos, obs_rad)
    best = pop[0]

    path = _genome_to_path(best, start, goal, base, perp)

    if strict and _obstacle_penalty_vec(path, obs_pos, obs_rad) > 1e-6:
        return None

    return [path[i].copy() for i in range(len(path))]


def bco_ga_plan_diverse(start, goal, obstacles, k: int = 3) -> List[List[np.ndarray]]:
    """
    Exploit the population: return up to k DIVERSE high-quality paths for one
    robot (a deterministic planner such as A* can only return one). Best-effort
    (paths may collide); the caller decides how to use them.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    base, perp = _geometry(start, goal)
    obs_pos, obs_rad = _obstacle_positions(obstacles)

    seed = _bco_phase(start, goal, base, perp, obs_pos, obs_rad)
    pop = _ga_phase(seed, start, goal, base, perp, obs_pos, obs_rad)
    # Pool = refined GA population (best-first) + the more exploratory BCO seed
    # population; the GA best stays first so the top path is always high quality.
    pool = list(pop) + list(seed)
    return _select_diverse(pool, start, goal, base, perp, obs_pos, obs_rad, k)


# ─────────────────────────────────────────
# Quick self-test
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

    t0 = time.time()
    path = bco_ga_plan(start, goal, obstacles)
    dt = time.time() - t0

    if path is None:
        print(f"No collision-free path found ({dt*1000:.0f} ms)")
    else:
        print(f"Path found with {len(path)} waypoints in {dt*1000:.0f} ms")
        for i, p in enumerate(path):
            print(f"  wp{i}: ({p[0]:.1f}, {p[1]:.1f})")
