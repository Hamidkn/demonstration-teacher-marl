"""Paired analysis of the stress studies (reviewer point M4).

The lag and crowding studies were previously reported descriptively because
their tables aggregate away the scenario pairing. The experiments now record
every (arm, condition, seed, scenario) outcome, so the comparisons can be
tested the way Section 5.2 tests the main one:

  * planner vs policy within a condition is paired by scenario, with the
    policy's three training seeds treated as the independent replicates;
  * the question of interest is not whether the planner leads — it plainly
    does — but whether its lead *changes* with the stressor, which is a
    contrast of differences across conditions.

Usage:
    py -3.10 analyze_stress.py sync_robustness_per_scenario.csv
    py -3.10 analyze_stress.py scale_results_per_scenario.csv
"""

import csv
import math
import sys
from collections import defaultdict
from statistics import mean, stdev

# Two-sided 0.975 Student-t quantiles, keyed by DEGREES OF FREEDOM (n - 1),
# not by n.  Keying this table by n is how an earlier version of the paper
# came to report intervals a quarter too narrow, so the lookup below passes
# n - 1 explicitly.
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}

# The contrast each experiment's table reports, so that running this script
# reproduces the paper rather than whatever pair sorts first.
PAPER_CONTRAST = {
    "sync":  ("lag0_noise0.0", "lag6_noise0.0"),   # lag axis at sigma = 0
    "scale": ("robots5", "robots12"),
}


def load(path):
    d = defaultdict(lambda: defaultdict(dict))   # arm -> cond -> (seed, scen) -> (succ, reach)
    for r in csv.DictReader(open(path)):
        d[r["arm"]][r["condition"]][(int(r["train_seed"]), int(r["scenario"]))] = (
            int(r["success"]), float(r["reached"]))
    return d


def t_test(x, label, baseline=0.0):
    n = len(x)
    if n < 2:
        return
    m, s = mean(x), stdev(x)
    if s == 0:
        print(f"    {label}: mean {m:+.1f}, zero variance"); return
    se = s / math.sqrt(n)
    df = n - 1
    t = (m - baseline) / se
    crit = T_CRIT.get(df, 1.96)
    lo, hi = m - crit * se, m + crit * se
    sig = "significant" if abs(t) > crit else "n.s."
    print(f"    {label}: mean {m:+.1f}, 95% CI [{lo:+.1f}, {hi:+.1f}], "
          f"t = {t:+.2f}, df = {df}, p {'<' if abs(t) > crit else '>'} 0.05  -> {sig}")
    # The interval and the statistic come from one standard error, so this
    # identity must hold; print it so a reader can see that it does.
    print(f"      s = {s:.2f}, se = {se:.2f}, t_crit(df={df}) = {crit:.3f}, "
          f"t recovered from CI = {m / ((hi - lo) / 2 / crit):+.2f}")


def main(path, mild=None, harsh=None):
    d = load(path)
    conds = sorted(d["policy"].keys())
    seeds = sorted({s for s, _ in d["policy"][conds[0]]})
    print(f"{path}\n  {len(conds)} conditions, policy seeds {seeds}\n")

    # Per-condition planner-minus-policy gap, with the seed as the unit.
    gaps = {}
    for c in conds:
        per_seed = []
        for sd in seeds:
            scen = sorted({sc for s, sc in d["policy"][c] if s == sd})
            pol = mean(d["policy"][c][(sd, sc)][0] for sc in scen) * 100
            pln = mean(d["planner"][c][(0, sc)][0] for sc in scen) * 100
            per_seed.append(pln - pol)
        gaps[c] = per_seed
        print(f"  {c}: planner-minus-policy strict-success gap per seed "
              + " ".join(f"{g:+.1f}" for g in per_seed)
              + f"   mean {mean(per_seed):+.1f}")

    # Which pair to contrast. Sorting the condition labels does NOT give the
    # pair the paper reports: for the sync grid it ends at lag6_noise6.0,
    # mixing in the noise axis that Section 5.4 explicitly excludes, and for
    # the scaling sweep "robots12" sorts before "robots5" and "robots8".
    key = "scale" if "scale" in path else "sync"
    a, b = PAPER_CONTRAST[key]
    if mild:
        a = mild
    if harsh:
        b = harsh
    missing = [c for c in (a, b) if c not in gaps]
    if missing:
        sys.exit(f"condition(s) not in {path}: {missing}\navailable: {conds}")

    print(f"\n  Does the gap change between {a} and {b}?"
          f"   (the contrast reported in the paper for this experiment)")
    diff = [x - y for x, y in zip(gaps[a], gaps[b])]
    print("    per-seed reduction: " + " ".join(f"{d:+.1f}" for d in diff))
    t_test(diff, f"gap({a}) - gap({b})")
    print("    (positive means the planner's lead shrinks under the stressor)")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "sync_robustness_per_scenario.csv",
         args[1] if len(args) > 1 else None,
         args[2] if len(args) > 2 else None)
