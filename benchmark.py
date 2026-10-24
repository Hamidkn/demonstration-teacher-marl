"""
Benchmark: compare A* / BCO-GA / Hybrid (proposed) across many random scenarios.
Produces a metrics table + comparison plots for the paper.

Usage:
    py -3.10 benchmark.py            # default 15 trials
    py -3.10 benchmark.py 30         # 30 trials
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from digital_twin_env import DigitalTwinEnv

MODES = {
    "astar":  "A*",
    "pso":    "PSO",
    "aco":    "ACO",
    "gwo":    "GWO",
    "bco_ga": "BCO-GA",
    "hybrid": "Hybrid (proposed)",
}
MODE_COLORS = {
    "astar":  "#3b82f6",
    "pso":    "#f59e0b",
    "aco":    "#ef4444",
    "gwo":    "#64748b",
    "bco_ga": "#a78bfa",
    "hybrid": "#22c55e",
}


def run_benchmark(num_trials: int = 15, num_robots: int = 8,
                  num_moving: int = 4, num_static: int = 4,
                  max_steps: int = 500):
    """Run every mode on the SAME seeded scenarios for a fair comparison."""
    results = {m: [] for m in MODES}

    print(f"Running benchmark: {num_trials} scenarios x {len(MODES)} methods")
    print(f"Config: {num_robots} robots, {num_moving} moving + "
          f"{num_static} static obstacles")
    print("=" * 62)

    for trial in range(num_trials):
        seed = 1000 + trial   # same seed -> same scenario for all methods
        for mode in MODES:
            env = DigitalTwinEnv(num_robots=num_robots,
                                 num_moving_obs=num_moving,
                                 num_static_obs=num_static,
                                 mode=mode, seed=seed, verbose=False)
            metrics = env.run_headless(max_steps=max_steps)
            results[mode].append(metrics)
        print(f"  scenario {trial + 1:2d}/{num_trials}  (seed {seed}) done")

    return results


def summarize(results: dict) -> dict:
    """Aggregate mean/std per mode. Path length averaged over SUCCESSFUL runs."""
    summary = {}
    for mode, runs in results.items():
        success = [r for r in runs if r["success"]]
        success_rate = 100.0 * len(success) / len(runs)
        reached_rate = 100.0 * sum(1 for r in runs if r.get("all_reached")) / len(runs)

        def mean_std(key, subset):
            vals = [r[key] for r in subset]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

        # length/time/smoothness only meaningful on runs that reached
        reached_runs = [r for r in runs if r.get("all_reached")] or runs
        len_m, len_s = mean_std("path_length", reached_runs)
        time_m, time_s = mean_std("sim_time", reached_runs)
        turn_m, _ = mean_std("turn_angle", reached_runs)
        rep_m, _ = mean_std("replans", runs)
        wait_m, _ = mean_std("waits", runs)
        col_m, _ = mean_std("collision_frames", runs)
        rcol_m, _ = mean_std("robot_collisions", runs)

        summary[mode] = {
            "success_rate":   success_rate,
            "reached_rate":   reached_rate,
            "path_length":    (len_m, len_s),
            "sim_time":       (time_m, time_s),
            "turn_angle":     turn_m,
            "replans":        rep_m,
            "waits":          wait_m,
            "collisions":     col_m,
            "robot_collisions": rcol_m,
        }
    return summary


def print_table(summary: dict):
    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS")
    print("=" * 78)
    header = (f"{'Method':<20}{'TrueSucc%':>11}{'PathLen':>10}{'Smooth°':>9}"
              f"{'SimTime':>9}{'Replans':>9}{'ObsCol':>8}")
    print(header)
    print("  (TrueSucc = reached AND zero collisions | Smooth = total turn angle, lower=smoother)")
    print("-" * 82)
    for mode, label in MODES.items():
        s = summary[mode]
        print(f"{label:<20}"
              f"{s['success_rate']:>10.0f}%"
              f"{s['path_length'][0]:>9.0f}"
              f"{s['turn_angle']:>9.0f}"
              f"{s['sim_time'][0]:>9.1f}"
              f"{s['replans']:>9.1f}"
              f"{s['collisions']:>8.1f}")
    print("=" * 82)


def plot_results(summary: dict, save_path: str):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Multi-Robot Path Planning — 6-Method Benchmark",
                 color="white", fontsize=14, fontweight="bold")

    labels = [MODES[m] for m in MODES]
    colors = [MODE_COLORS[m] for m in MODES]

    metrics = [
        ("success_rate", "Success Rate (%)", lambda s: s["success_rate"], None),
        ("path_length",  "Avg Path Length", lambda s: s["path_length"][0],
         lambda s: s["path_length"][1]),
        ("sim_time",     "Avg Completion Time (s)", lambda s: s["sim_time"][0],
         lambda s: s["sim_time"][1]),
        ("replans",      "Avg Replans", lambda s: s["replans"], None),
    ]

    for ax, (key, title, getter, err) in zip(axes, metrics):
        vals = [getter(summary[m]) for m in MODES]
        errs = [err(summary[m]) for m in MODES] if err else None
        bars = ax.bar(labels, vals, color=colors, yerr=errs,
                      capsize=4, edgecolor="white", linewidth=0.5)
        ax.set_title(title, color="white", fontsize=11)
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(True, axis="y", color="#1e293b", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.0f}" if val > 5 else f"{val:.1f}",
                    ha="center", va="bottom", color="white", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"\nPlot saved: {save_path}")
    plt.show()


def save_csv(results: dict, path: str):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "seed_index", "success", "reached",
                         "steps", "sim_time", "path_length",
                         "replans", "waits", "collision_frames",
                         "robot_collisions"])
        for mode, runs in results.items():
            for i, r in enumerate(runs):
                writer.writerow([mode, i, r["success"], r["reached"],
                                 r["steps"], r["sim_time"], r["path_length"],
                                 r["replans"], r["waits"], r["collision_frames"],
                                 r["robot_collisions"]])
    print(f"Raw data saved: {path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    # MEDIUM config: 5 robots, varied random start/goal + 3 moving + 3 static
    results = run_benchmark(num_trials=n, num_robots=5,
                            num_moving=3, num_static=3, max_steps=600)
    summary = summarize(results)
    print_table(summary)
    save_csv(results, "results/benchmark_results.csv")
    plot_results(summary, "benchmark_plot.png")
