"""Paired significance testing for the planning benchmark (reviewer point R1.3).

Scenarios are seeded and identical across planners, so per-scenario success is
PAIRED data. An unpaired two-proportion z-test is the wrong test for it: it
ignores the pairing and therefore throws away power. This script runs McNemar's
exact test on the discordant pairs and prints both results side by side so the
difference is visible.

Usage:
    py -3.10 analyze_significance.py
"""

import csv
import math
from collections import defaultdict
from itertools import combinations

SRC = "results/benchmark_results.csv"
BASE = "hybrid"


def binom_two_sided(b, c):
    """Exact McNemar: two-sided binomial test of b successes in n=b+c at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    def pmf(k):
        return math.comb(n, k) * 0.5 ** n
    obs = pmf(b)
    # sum probabilities of outcomes no more likely than the observed one
    return min(1.0, sum(pmf(k) for k in range(n + 1) if pmf(k) <= obs + 1e-12))


def z_unpaired(p1, p2, n1, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def main():
    by_mode = defaultdict(dict)
    for r in csv.DictReader(open(SRC)):
        by_mode[r["mode"]][int(r["seed_index"])] = r["success"].strip().lower() == "true"

    modes = list(by_mode)
    n = len(by_mode[modes[0]])
    rate = {m: sum(by_mode[m].values()) / n for m in modes}

    print(f"{SRC}: {len(modes)} methods x {n} paired scenarios\n")
    print("strict success rate")
    for m in sorted(modes, key=lambda x: -rate[x]):
        print(f"  {m:<9}{rate[m]*100:5.0f}%")

    print(f"\npaired McNemar (exact) vs unpaired two-proportion z, baseline = {BASE}")
    print(f"{'comparison':<22}{'b':>4}{'c':>4}{'McNemar p':>12}"
          f"{'z-test p':>11}   verdict")
    print("-" * 70)
    others = [m for m in modes if m != BASE]
    for m in sorted(others, key=lambda x: -rate[x]):
        b = sum(1 for s in range(n) if by_mode[BASE][s] and not by_mode[m][s])
        c = sum(1 for s in range(n) if not by_mode[BASE][s] and by_mode[m][s])
        p_mc = binom_two_sided(b, c)
        _z, p_z = z_unpaired(rate[BASE], rate[m], n, n)
        flag = "significant" if p_mc < 0.05 else "n.s."
        print(f"{BASE+' vs '+m:<22}{b:>4}{c:>4}{p_mc:>12.4f}{p_z:>11.4f}   {flag}")

    print("\nall pairs, McNemar exact p:")
    for a, b_ in combinations(sorted(modes, key=lambda x: -rate[x]), 2):
        bb = sum(1 for s in range(n) if by_mode[a][s] and not by_mode[b_][s])
        cc = sum(1 for s in range(n) if not by_mode[a][s] and by_mode[b_][s])
        p = binom_two_sided(bb, cc)
        mark = " *" if p < 0.05 else ""
        print(f"  {a:<9} vs {b_:<9} b={bb:<3} c={cc:<3} p={p:.4f}{mark}")


if __name__ == "__main__":
    main()
