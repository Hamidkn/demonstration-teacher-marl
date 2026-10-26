"""Paired, seed-aware analysis of the held-out evaluation (reviewer objection 2).

The submitted version pooled 3 seeds x 50 scenarios into 150 outcomes and ran
an unpaired two-proportion z-test. That is wrong twice: the same 50 scenario
IDs are replayed for every policy, so outcomes are paired by scenario; and the
three policies per configuration come from three training seeds, so outcomes
are clustered by seed. Pooling treats 150 dependent outcomes as independent and
overstates significance.

This script reports three analyses of increasing conservatism:

  (1) the original pooled unpaired z-test, for comparison only;
  (2) McNemar's exact test within each training seed, where scenarios really
      are paired and independent of one another;
  (3) the seed as the unit of analysis (n = 3), which is the honest treatment
      if we want a statement about the configuration rather than about three
      particular trained networks.

Usage:
    py -3.10 analyze_paired.py
"""

import csv
import math
from collections import defaultdict
from statistics import mean, stdev

SRC = "results/eval_per_scenario.csv"
SEEDS = ()          # discovered from the file; see load()


def load():
    """Return the per-scenario outcomes and the seeds actually present.

    The seed set is read from the file rather than hard-coded, so adding
    training runs changes the analysis without changing this script.
    """
    data = defaultdict(dict)        # (config, seed) -> {scenario: (success, reached)}
    seeds = set()
    for r in csv.DictReader(open(SRC)):
        s = int(r["train_seed"])
        seeds.add(s)
        data[(r["config"], s)][int(r["scenario"])] = (
            int(r["success"]), int(r["reached"]) / int(r["robots"]))
    return data, tuple(sorted(seeds))


def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    pmf = lambda k: math.comb(n, k) * 0.5 ** n
    obs = pmf(b)
    return min(1.0, sum(pmf(k) for k in range(n + 1) if pmf(k) <= obs + 1e-12))


def z_unpaired(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    m, s = mean(diffs), stdev(diffs)
    if s == 0:
        return float("inf") if m else 0.0, 0.0 if m else 1.0
    t = m / (s / math.sqrt(n))
    # two-sided p from Student's t with n-1 df, computed by numeric integration
    df = n - 1
    from math import gamma
    c = gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * gamma(df / 2))
    steps, hi = 20000, abs(t)
    h = hi / steps
    area = sum(c * (1 + (h * i) ** 2 / df) ** (-(df + 1) / 2) * h
               for i in range(steps))
    return t, max(0.0, min(1.0, 2 * (0.5 - area)))


def compare(data, a, b, metric_idx, label, SEEDS):
    print(f"\n{'=' * 74}\n{label}:  {a}  vs  {b}\n{'=' * 74}")

    # (1) the pooled unpaired test, as originally reported
    ka = sum(v[metric_idx] for s in SEEDS for v in data[(a, s)].values())
    kb = sum(v[metric_idx] for s in SEEDS for v in data[(b, s)].values())
    na = sum(len(data[(a, s)]) for s in SEEDS)
    nb = sum(len(data[(b, s)]) for s in SEEDS)
    z, p = z_unpaired(ka, na, kb, nb)
    print(f"(1) pooled unpaired z-test  [AS ORIGINALLY REPORTED - INVALID]")
    print(f"    {a} {ka}/{na} = {ka/na*100:.1f}%   {b} {kb}/{nb} = {kb/nb*100:.1f}%"
          f"   z = {z:+.2f}  p = {p:.4f}")

    # (2) McNemar within each seed
    print(f"(2) McNemar exact, within each training seed (scenarios paired)")
    ps = []
    for s in SEEDS:
        da, db = data[(a, s)], data[(b, s)]
        scen = sorted(set(da) & set(db))
        bb = sum(1 for x in scen if da[x][metric_idx] and not db[x][metric_idx])
        cc = sum(1 for x in scen if not da[x][metric_idx] and db[x][metric_idx])
        pv = binom_two_sided(bb, cc)
        ps.append(pv)
        print(f"    seed {s}: {a} wins {bb}, {b} wins {cc}  ->  p = {pv:.4f}"
              f"{'  *' if pv < 0.05 else ''}")
    print(f"    Bonferroni across {len(SEEDS)} seeds: smallest p x {len(SEEDS)} = {min(ps)*len(SEEDS):.4f}"
          f"{'  *' if min(ps)*len(SEEDS) < 0.05 else '  (n.s.)'}")

    # (3) seed as the unit of analysis
    da_m = [mean(v[metric_idx] for v in data[(a, s)].values()) * 100 for s in SEEDS]
    db_m = [mean(v[metric_idx] for v in data[(b, s)].values()) * 100 for s in SEEDS]
    diffs = [x - y for x, y in zip(da_m, db_m)]
    t, pt = paired_t(diffs)
    print(f"(3) seed as unit (n = {len(SEEDS)}), paired t on seed means")
    print(f"    {a}: {[f'{v:.1f}' for v in da_m]}")
    print(f"    {b}: {[f'{v:.1f}' for v in db_m]}")
    print(f"    diffs: {[f'{d:+.1f}' for d in diffs]}   mean {mean(diffs):+.1f}"
          f"   t = {t:+.2f}  p = {pt:.4f}"
          f"{'  *' if pt < 0.05 else '  (n.s.)'}")


def main():
    data, SEEDS = load()
    print(f'seeds present: {SEEDS}  (n = {len(SEEDS)})')
    have = {c for c, _s in data}
    import sys
    pairs = [("warm_bco", "warm_astar"), ("warm_bco", "cold"),
             ("warm_astar", "cold")]
    if "--ablation" in sys.argv:
        pairs = [("warm_bco_k3_cap", "warm_astar_cap"),
                 ("warm_bco_k5_cap", "warm_astar_cap"),
                 ("warm_bco_k1_cap", "warm_astar_cap")]
    only = 1 if "--reached-only" in sys.argv else None
    for a, b in pairs:
        if a in have and b in have:
            if only != 1:
                compare(data, a, b, 0, "STRICT SUCCESS", SEEDS)
            compare(data, a, b, 1, "GOAL-REACHING", SEEDS)


if __name__ == "__main__":
    main()
