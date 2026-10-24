"""Matched-demonstration-budget diversity analysis (reviewer point R1.1).

The uncapped comparison in Section 4.2 confounds two things: the population
planner rolls K variants per scenario, so it produces ~4x more (obs, action)
pairs than the deterministic planner AND each pair is somewhat more diverse.
This script separates them by subsampling every set to a common budget and
reporting unique-state counts at the same quantization.

Usage:
    py -3.10 analyze_demo_budget.py                       # default pairing
    py -3.10 analyze_demo_budget.py demos_astar.npz demos_bco.npz ...
"""

import sys
import numpy as np

QUANT = 0.05
N_DRAWS = 5          # subsample repetitions, to report a mean +- std
DEFAULT = ["demos/demos_astar.npz", "demos/demos_bco.npz"]


def unique_states(obs, quant=QUANT):
    return len(np.unique(np.round(obs.astype(np.float64) / quant), axis=0))


def action_entropy(act, n_actions=9):
    counts = np.bincount(act, minlength=n_actions).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def load(path):
    d = np.load(path)
    return d["obs"], d["actions"], str(d["planner"])


def main(paths):
    sets = {p: load(p) for p in paths}
    budget = min(len(a) for _o, a, _n in sets.values())

    print(f"quantization {QUANT} | matched budget = {budget} pairs "
          f"| {N_DRAWS} subsample draws\n")
    print(f"{'demo set':<22}{'pairs':>8}{'uniq(full)':>12}{'cov':>7}"
          f"{'uniq@budget':>16}{'cov':>8}{'H(a)':>8}")
    print("-" * 81)

    rows = {}
    for path, (obs, act, name) in sets.items():
        n_full = len(act)
        u_full = unique_states(obs)
        if n_full == budget:
            u_mean, u_std, c_mean = float(u_full), 0.0, u_full / n_full
        else:
            us = []
            for draw in range(N_DRAWS):
                rng = np.random.default_rng(draw)
                sel = rng.choice(n_full, size=budget, replace=False)
                us.append(unique_states(obs[sel]))
            u_mean, u_std = float(np.mean(us)), float(np.std(us, ddof=1))
            c_mean = u_mean / budget
        rows[path] = u_mean
        std_txt = f"+/-{u_std:.0f}" if u_std else "  exact"
        print(f"{path:<22}{n_full:>8}{u_full:>12}{u_full/n_full:>7.3f}"
              f"{u_mean:>11.0f}{std_txt:>5}{c_mean:>8.3f}"
              f"{action_entropy(act):>8.3f}")

    base = min(rows, key=lambda p: len(sets[p][1]))
    print(f"\nratios against '{base}' (the smallest set, i.e. the budget):")
    for path, u in rows.items():
        if path == base:
            continue
        n_ratio = len(sets[path][1]) / len(sets[base][1])
        u_ratio_full = unique_states(sets[path][0]) / unique_states(sets[base][0])
        u_ratio_cap = u / rows[base]
        print(f"  {path}")
        print(f"    data-volume ratio           : {n_ratio:5.2f}x")
        print(f"    unique states, UNCAPPED     : {u_ratio_full:5.2f}x  "
              f"<- confounds volume + diversity")
        print(f"    unique states, MATCHED budget: {u_ratio_cap:5.2f}x  "
              f"<- the true diversity effect")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
