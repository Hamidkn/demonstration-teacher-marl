"""Figure 4 — matched-demonstration-budget ablation.

Left: in-training goal-reaching over the run for the four teachers (five
scenarios per point, as logged during training), mean over three seeds with a
+/-1 sd band; every arm was capped to 11,173 demonstration pairs.

Right: the point of the figure. The SAME twelve runs are scored two ways
against the diversity of their demonstration set — by the five-scenario
protocol logged during training, and by the 50-scenario final evaluation of
Table 6. The first shows a strong inverse trend, the second shows none. An
earlier version of this analysis reported the first and drew a confident,
spurious conclusion from it.

Previously this panel plotted only the five-scenario scores while the caption
described them as final performance, which contradicted Table 6.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = "results/ablation_runs"
OUT = "figure_ablation.png"
SEEDS = (1, 2, 3)
PHASE = 50_000

ARMS = [
    ("warm_astar_cap",  "A* (deterministic)", 8599,  "#1565c0", "-"),
    ("warm_bco_k1_cap", "BCO-GA, K = 1",      9621,  "#2e7d32", "--"),
    ("warm_bco_k3_cap", "BCO-GA, K = 3",     10641,  "#ef6c00", "-."),
    ("warm_bco_k5_cap", "BCO-GA, K = 5",     10848,  "#6a1b9a", ":"),
]


def load(arm, seed):
    rows = list(csv.DictReader(open(os.path.join(RUN_DIR, f"{arm}_s{seed}.csv"))))
    return (np.array([int(r["steps"]) for r in rows]),
            np.array([float(r["eval_reached"]) for r in rows]) * 100)


def smooth(y, k=15):
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.4),
                               gridspec_kw={"width_ratios": [1.55, 1]})

# ── Left: learning curves at matched budget ────────────────────────
finals = {}
for arm, label, uniq, color, ls in ARMS:
    curves = [load(arm, s) for s in SEEDS]
    steps = curves[0][0]
    stack = np.vstack([smooth(c[1]) for c in curves])
    m, sd = stack.mean(0), stack.std(0, ddof=1)
    axL.plot(steps, m, color=color, ls=ls, lw=2.3, label=label)
    axL.fill_between(steps, m - sd, m + sd, color=color, alpha=0.12, lw=0)
    last = [c[1][c[0] >= c[0].max() - PHASE].mean() for c in curves]
    finals[arm] = (uniq, float(np.mean(last)), float(np.std(last, ddof=1)), label, color)

axL.set_title("(a) In-training goal-reaching, matched demonstration budget\n"
              "(11,173 pairs per teacher; 5 scenarios per logged point)",
              fontsize=12, fontweight="bold", pad=10)
axL.set_xlabel("environment steps", fontsize=10.5, fontweight="bold")
axL.set_ylabel("robots reaching goals (%)", fontsize=10.5, fontweight="bold")
axL.legend(fontsize=9.2, loc="lower right", framealpha=0.95)
axL.grid(True, color="#e4e4e4", lw=0.7)
axL.set_axisbelow(True)
for s in axL.spines.values():
    s.set_linewidth(1.4)
axL.tick_params(labelsize=9)

# ── Right: the same runs scored two ways against diversity ────────
# 50-scenario final evaluation (Table 6), read from the saved evaluation.
FINAL50 = {"warm_astar_cap": (57.3, 3.9), "warm_bco_k1_cap": (49.3, 8.5),
           "warm_bco_k3_cap": (59.3, 5.4), "warm_bco_k5_cap": (52.5, 2.7)}

xs = [finals[a][0] for a, *_ in ARMS]
y5 = [finals[a][1] for a, *_ in ARMS]           # five-scenario, in-training
e5 = [finals[a][2] for a, *_ in ARMS]
y50 = [FINAL50[a][0] for a, *_ in ARMS]
e50 = [FINAL50[a][1] for a, *_ in ARMS]

r5 = np.corrcoef(xs, y5)[0, 1]
r50 = np.corrcoef(xs, y50)[0, 1]

gx = np.linspace(min(xs) - 400, max(xs) + 400, 50)
axR.plot(gx, np.polyval(np.polyfit(xs, y5, 1), gx), color="#b71c1c",
         lw=1.8, ls="--", zorder=2,
         label=f"5-scenario protocol  (r = {r5:+.2f})")
axR.plot(gx, np.polyval(np.polyfit(xs, y50, 1), gx), color="#1b5e20",
         lw=1.8, ls="-", zorder=2,
         label=f"50-scenario evaluation  (r = {r50:+.2f})")

axR.errorbar(xs, y5, yerr=e5, fmt="o", color="#b71c1c", markersize=9,
             markeredgecolor="black", markeredgewidth=1.0, capsize=4,
             elinewidth=1.4, linestyle="none", zorder=4)
axR.errorbar(xs, y50, yerr=e50, fmt="s", color="#1b5e20", markersize=9,
             markeredgecolor="black", markeredgewidth=1.0, capsize=4,
             elinewidth=1.4, linestyle="none", zorder=4)

for (arm, label, uniq, color, _ls) in ARMS:
    axR.annotate(label.replace("BCO-GA, ", ""), (uniq, FINAL50[arm][0]),
                 textcoords="offset points", xytext=(0, 17), ha="center",
                 fontsize=8.6, fontweight="bold", color="#333333", zorder=5)

axR.set_title("(b) The apparent diversity effect is an artefact\n"
              "of the thin in-training evaluation",
              fontsize=12, fontweight="bold", pad=10)
axR.set_xlabel("unique states in the demonstration set\n(at the matched budget)",
               fontsize=10.5, fontweight="bold")
axR.set_ylabel("goal-reaching (%)", fontsize=10.5, fontweight="bold")
axR.grid(True, color="#e4e4e4", lw=0.7)
axR.set_axisbelow(True)
lo = min(min(y5) - max(e5), min(y50) - max(e50)) - 5
hi = max(max(y5) + max(e5), max(y50) + max(e50)) + 10
axR.set_ylim(lo, hi)
for s in axR.spines.values():
    s.set_linewidth(1.4)
axR.tick_params(labelsize=9)
axR.legend(fontsize=8.6, loc="lower left", framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT, f"| r(5-scenario) = {r5:+.3f}  r(50-scenario) = {r50:+.3f}")
