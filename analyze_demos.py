"""
Quantify demonstration DIVERSITY: BCO-GA expert vs A* expert.

Why this matters for the paper: the reviewer will ask "if Hybrid only ties A*,
why keep the metaheuristic at all?" One defensible answer is that a
population-based planner produces more DIVERSE demonstrations, which is known
to help imitation-learning generalization. This script turns that claim into
measured numbers:

  1. Action entropy (bits)            — variety of expert behavior
  2. Unique quantized observations    — state-space coverage
  3. Mean pairwise obs distance       — spread of visited states
"""

import numpy as np

FILES = {
    "BCO-GA expert": "demos/demos_bco.npz",
    "A* expert":     "demos/demos_astar.npz",
}
rng = np.random.default_rng(0)

print(f"{'metric':<38}{'BCO-GA':>12}{'A*':>12}")
print("-" * 62)

results = {}
for name, path in FILES.items():
    d = np.load(path)
    obs, act = d["obs"], d["actions"]

    # 1) action entropy
    counts = np.bincount(act, minlength=9).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    # 2) unique quantized observations (coverage)
    q = np.round(obs / 0.05).astype(np.int16)
    unique = len(np.unique(q, axis=0))
    coverage = unique / len(obs)

    # 3) mean pairwise distance on a 2000-point subsample
    idx = rng.choice(len(obs), min(2000, len(obs)), replace=False)
    sub = obs[idx]
    diffs = sub[:, None, :] - sub[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(-1))
    mean_dist = float(dists[np.triu_indices(len(sub), 1)].mean())

    results[name] = dict(n=len(act), entropy=entropy, unique=unique,
                         coverage=coverage, spread=mean_dist)

b, a = results["BCO-GA expert"], results["A* expert"]
rows = [
    ("samples",                    f"{b['n']}",              f"{a['n']}"),
    ("action entropy (bits)",      f"{b['entropy']:.3f}",    f"{a['entropy']:.3f}"),
    ("unique quantized obs",       f"{b['unique']}",         f"{a['unique']}"),
    ("coverage ratio",             f"{b['coverage']:.3f}",   f"{a['coverage']:.3f}"),
    ("mean pairwise obs distance", f"{b['spread']:.3f}",     f"{a['spread']:.3f}"),
]
for label, vb, va in rows:
    print(f"{label:<38}{vb:>12}{va:>12}")

print()
for metric, key in [("entropy", "entropy"), ("coverage", "unique"),
                    ("spread", "spread")]:
    rel = (b[key] - a[key]) / a[key] * 100
    print(f"BCO-GA vs A*  {metric:>8}: {rel:+.1f}%")
