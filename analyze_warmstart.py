"""
Extract the paper's headline numbers from the Colab learning curves:
  - steps needed to reach return thresholds (sample-efficiency speedup)
  - early-phase (first 50k steps) mean return per run
  - final-phase (last 50k steps) mean return per run
Also renders the final paper-grade figure via plot_curves.py styling.
"""

import csv
import sys
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "results/runs"
RUNS = {
    "cold":       f"{BASE}/cold.csv",
    "warm_astar": f"{BASE}/warm_astar.csv",
    "warm_bco":   f"{BASE}/warm_bco.csv",
}
print(f"[data: {BASE}]")


def load(path):
    steps, ret, succ, reached = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["steps"]))
            ret.append(float(row["mean_ep_return"]))
            succ.append(float(row["eval_success"]))
            reached.append(float(row["eval_reached"]))
    return (np.array(steps), np.array(ret),
            np.array(succ), np.array(reached))


def smooth(y, k=5):
    """Centered moving average to de-noise before threshold detection."""
    if len(y) < k:
        return y
    pad = k // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(ypad, np.ones(k) / k, mode="valid")


data = {tag: load(p) for tag, p in RUNS.items()}
max_steps = int(max(data[tag][0].max() for tag in RUNS))
PHASES = [("first 50k steps", 0, 50_000),
          ("last 50k steps", max_steps - 50_000, max_steps + 1)]

print("=" * 74)
print(f"WARM-START SAMPLE-EFFICIENCY ANALYSIS  (total ~{max_steps:,} steps)")
print("=" * 74)

# 1) steps to reach return thresholds (on smoothed curves)
thresholds = [100, 150, 175, 200]
print(f"\n{'threshold':<12}" + "".join(f"{t:>14}" for t in RUNS))
for th in thresholds:
    row = f"return>={th:<4}  "
    for tag in RUNS:
        steps, ret, _s, _r = data[tag]
        sm = smooth(ret)
        idx = np.argmax(sm >= th)
        if sm[idx] >= th:
            row += f"{steps[idx]:>13,}"
        else:
            row += f"{'never':>13}"
    print(row)

# 2) early/final phase means
print(f"\n{'phase':<22}" + "".join(f"{t:>14}" for t in RUNS))
for label, lo, hi in PHASES:
    row = f"{label:<22}"
    for tag in RUNS:
        steps, ret, _s, _r = data[tag]
        mask = (steps >= lo) & (steps <= hi)
        row += f"{ret[mask].mean():>14.1f}"
    print(row)

# 3) eval reached fraction, same phases
print(f"\n{'eval reached':<22}" + "".join(f"{t:>14}" for t in RUNS))
for label, lo, hi in PHASES:
    row = f"{label:<22}"
    for tag in RUNS:
        steps, _ret, _s, reached = data[tag]
        mask = (steps >= lo) & (steps <= hi)
        row += f"{reached[mask].mean()*100:>13.1f}%"
    print(row)

# 3b) eval strict success, same phases
print(f"\n{'eval success':<22}" + "".join(f"{t:>14}" for t in RUNS))
for label, lo, hi in PHASES:
    row = f"{label:<22}"
    for tag in RUNS:
        steps, _ret, succ, _r = data[tag]
        mask = (steps >= lo) & (steps <= hi)
        row += f"{succ[mask].mean()*100:>13.1f}%"
    print(row)

# 4) headline speedup: cold steps-to-threshold / warm steps-to-threshold
print("\nHEADLINE:")
for th in thresholds:
    vals = {}
    for tag in RUNS:
        steps, ret, _s, _r = data[tag]
        sm = smooth(ret)
        idx = np.argmax(sm >= th)
        vals[tag] = steps[idx] if sm[idx] >= th else None
    if vals["cold"] and vals["warm_bco"]:
        base = max(vals["warm_bco"], 1)
        print(f"  return>={th}: cold needs {vals['cold']:,} steps, "
              f"warm_bco needs {vals['warm_bco']:,} "
              f"-> {vals['cold']/base:.1f}x fewer samples")
