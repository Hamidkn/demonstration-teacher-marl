"""Figure 3 — per-seed learning curves as a 3x2 grid (6 panels):
rows = seed 1,2,3 ; columns = training return, evaluation robots-reached.
Each panel shows the three initializations for that individual seed."""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = {1: "ColabResults-Path 2/results_seed1/runs",
         2: "ColabResults-Path 2/results_seed2/runs",
         3: "ColabResults-Path 2/results_seed3/runs"}
METHODS = [
    ("cold",       "Cold start",          "#1565c0", "-"),
    ("warm_astar", "Warm (A* demos)",     "#e65100", "--"),
    ("warm_bco",   "Warm (BCO-GA demos)", "#2e7d32", "-"),
]


def load(path):
    s, r, sc, re = [], [], [], []
    for row in csv.DictReader(open(path)):
        s.append(int(row["steps"])); r.append(float(row["mean_ep_return"]))
        sc.append(float(row["eval_success"])); re.append(float(row["eval_reached"]))
    return np.array(s), np.array(r), np.array(sc), np.array(re)


def smooth(y, k=9):
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


fig, axes = plt.subplots(3, 2, figsize=(11, 11))

for row, (seed, d) in enumerate(SEEDS.items()):
    cols = {tag: load(f"{d}/{tag}.csv") for tag, *_ in METHODS}
    for col, (ci, ylab) in enumerate([(1, "mean episode return"),
                                      (3, "fraction of robots reached")]):
        ax = axes[row, col]
        for tag, label, color, ls in METHODS:
            s, r, sc, re = cols[tag]
            y = {1: r, 3: re}[ci]
            ax.plot(s, smooth(y), label=label, color=color, ls=ls, lw=2.0)
        title = ("Training episode return" if col == 0
                 else "Evaluation: robots reaching goals")
        ax.set_title(f"Seed {seed} — {title}", fontsize=11, fontweight="bold")
        ax.set_xlabel("environment steps", fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=8.5, width=1.2)
        for sp in ax.spines.values():
            sp.set_linewidth(1.4)
        ax.grid(True, color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        if row == 0 and col == 1:
            leg = ax.legend(fontsize=8.5, framealpha=0.95, loc="lower right")
            for t in leg.get_texts():
                t.set_fontweight("bold")

plt.tight_layout()
plt.savefig("figure_curves_grid.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("saved figure_curves_grid.png (3 seeds x 2 metrics = 6 panels)")
