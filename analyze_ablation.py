"""Analyse the matched-budget ablation (reviewer points R1.1 and R1.2).

Every arm saw the same number of demonstration pairs (CAP = 11,173), so data
volume is held constant and the only thing that varies is how diverse those
pairs are. If the demonstration-diversity claim is real, held-out performance
should track the diversity ladder measured in analyze_demo_budget.py:

    A* 8,599  <  K=1 9,621  <  K=3 10,641  <  K=5 10,848   unique states

Usage:
    py -3.10 analyze_ablation.py [run_dir]
"""

import csv
import os
import sys
from statistics import mean, stdev

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/ablation_runs"
SEEDS = (1, 2, 3)
ARMS = [
    ("warm_astar_cap",  "A* (deterministic)", 8599),
    ("warm_bco_k1_cap", "BCO K=1",            9621),
    ("warm_bco_k3_cap", "BCO K=3",           10641),
    ("warm_bco_k5_cap", "BCO K=5",           10848),
]
PHASE = 50_000


def load(arm, seed):
    path = os.path.join(RUN_DIR, f"{arm}_s{seed}.csv")
    rows = list(csv.DictReader(open(path)))
    steps = [int(r["steps"]) for r in rows]
    return {
        "steps": steps,
        "reached": [float(r["eval_reached"]) for r in rows],
        "success": [float(r["eval_success"]) for r in rows],
        "ret": [float(r["mean_ep_return"]) for r in rows],
        "last": steps[-1],
    }


def phase_mean(d, key, first):
    lo, hi = (0, PHASE) if first else (d["last"] - PHASE, d["last"])
    vals = [v for s, v in zip(d["steps"], d[key]) if lo <= s <= hi]
    return mean(vals) * 100


def ms(xs):
    return mean(xs), (stdev(xs) if len(xs) > 1 else 0.0)


def main():
    print(f"run dir: {RUN_DIR}")
    print(f"matched demonstration budget: 11,173 pairs for every arm\n")

    metrics = {}
    for arm, label, uniq in ARMS:
        per_seed = {s: load(arm, s) for s in SEEDS}
        metrics[arm] = {
            "label": label, "uniq": uniq, "per_seed": per_seed,
            "reach_first": [phase_mean(per_seed[s], "reached", True) for s in SEEDS],
            "reach_last": [phase_mean(per_seed[s], "reached", False) for s in SEEDS],
            "succ_last": [phase_mean(per_seed[s], "success", False) for s in SEEDS],
        }

    print("=" * 78)
    print("HELD-OUT EVALUATION, mean +/- sd over 3 seeds")
    print("=" * 78)
    print(f"{'arm':<22}{'uniq states':>12}{'reached 1st50k':>16}"
          f"{'reached last50k':>17}{'strict last50k':>16}")
    print("-" * 78)
    for arm, label, uniq in ARMS:
        m = metrics[arm]
        a, asd = ms(m["reach_first"])
        b, bsd = ms(m["reach_last"])
        c, csd = ms(m["succ_last"])
        print(f"{label:<22}{uniq:>12,}{a:>11.1f}+/-{asd:<4.1f}"
              f"{b:>12.1f}+/-{bsd:<4.1f}{c:>11.1f}+/-{csd:<4.1f}")

    base = metrics["warm_astar_cap"]
    print(f"\n{'arm':<22}{'vs A* (reached last50k)':>26}{'vs A* (1st50k)':>18}")
    print("-" * 66)
    for arm, label, _u in ARMS:
        m = metrics[arm]
        d_last = mean(m["reach_last"]) - mean(base["reach_last"])
        d_first = mean(m["reach_first"]) - mean(base["reach_first"])
        print(f"{label:<22}{d_last:>+22.1f} pts{d_first:>+14.1f} pts")

    print("\n" + "=" * 78)
    print("PER-SEED, reached in the last 50k (the metric with the clean signal)")
    print("=" * 78)
    print(f"{'arm':<22}" + "".join(f"{'seed '+str(s):>12}" for s in SEEDS)
          + f"{'beats A*?':>14}")
    print("-" * 78)
    for arm, label, _u in ARMS:
        m = metrics[arm]
        cells = "".join(f"{v:>11.1f}%" for v in m["reach_last"])
        if arm == "warm_astar_cap":
            verdict = "-"
        else:
            wins = sum(1 for i in range(3)
                       if m["reach_last"][i] > base["reach_last"][i])
            verdict = f"{wins}/3 seeds"
        print(f"{label:<22}{cells}{verdict:>14}")

    print("\n" + "=" * 78)
    print("DOES PERFORMANCE TRACK DIVERSITY?")
    print("=" * 78)
    order = [metrics[a]["label"] for a, _l, _u in ARMS]
    reach = [mean(metrics[a]["reach_last"]) for a, _l, _u in ARMS]
    uniq = [u for _a, _l, u in ARMS]
    print("diversity order :", " < ".join(order))
    print("unique states   :", "  ".join(f"{u:,}" for u in uniq))
    print("reached last50k :", "  ".join(f"{r:.1f}%" for r in reach))
    mono = all(reach[i] <= reach[i + 1] for i in range(len(reach) - 1))
    print(f"\nmonotone in K   : {'YES' if mono else 'NO'}")
    pop_vs_det = all(r > reach[0] for r in reach[1:])
    print(f"every population arm beats the deterministic teacher: "
          f"{'YES' if pop_vs_det else 'NO'}")

    # Pearson correlation between demonstration diversity and final performance
    n = len(uniq)
    mu, mr = mean(uniq), mean(reach)
    cov = sum((uniq[i] - mu) * (reach[i] - mr) for i in range(n))
    vu = sum((uniq[i] - mu) ** 2 for i in range(n)) ** 0.5
    vr = sum((reach[i] - mr) ** 2 for i in range(n)) ** 0.5
    if vu and vr:
        print(f"correlation(unique states, final reached) r = {cov/(vu*vr):+.3f}")


if __name__ == "__main__":
    main()
