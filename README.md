# What Makes a Good Demonstration Teacher?

Code, demonstration sets and raw run logs for the paper *"What Makes a Good
Demonstration Teacher? Metaheuristic Warm-Starting of Multi-Agent Reinforcement
Learning for Multi-Robot Path Planning"*.

Hamid Keshmiri Neghab · Independent Researcher, Haibach, Germany ·
[ORCID 0000-0001-9896-9749](https://orcid.org/0000-0001-9896-9749)

Everything needed to reproduce every number in the paper is here, including the
per-scenario evaluation outcomes and the trained policy checkpoints, so the
statistical analyses can be re-run without retraining anything.

## What the paper finds

Five robots navigate a 2-D workspace containing static and moving obstacles. A
hybrid planner (a bee-colony/genetic metaheuristic for global search, grid A\*
for reactive replanning) is benchmarked against A\*, PSO, ACO-R and GWO under a
matched evaluation budget. The planners are then used as *demonstration
generators* to warm-start a shared-policy PPO agent, and the paper asks which
property of a demonstration set makes it a good teacher.

The headline result is partly negative, and the repository is arranged so that
this can be checked:

- Warm-starting from the population planner's demonstrations produces a better
  policy than warm-starting from A\*'s, consistently in sign across all three
  training seeds — but **not at conventional significance with three seeds**
  (`analyze_paired.py`).
- The obvious explanation, that the population planner's demonstrations are
  more diverse, **does not survive a matched-budget ablation**
  (`analyze_ablation.py`).
- An earlier version of this analysis reported a strong effect (r = −0.97)
  that came entirely from scoring on five evaluation scenarios. The same
  policies on fifty scenarios give r = −0.04. Both numbers are reproducible
  from the shipped logs; `fig_ablation.py` plots them side by side.

## Layout

```
*.py                      environment, planners, training, analysis, figures
notebooks/                Colab notebooks for the training runs
demos/                    demonstration sets (A*, and BCO-GA at K = 1, 3, 5)
results/
  benchmark_results.csv   per-scenario planner benchmark (Table 2)
  eval_final.csv          per-(config, seed) held-out evaluation
  eval_per_scenario.csv   per-scenario outcomes, needed for the paired tests
  sync_robustness.csv     observation-degradation grid (Tables 7, 8)
  scale_results.csv       robot-count sweep (Table 9)
  training_runs/seed{1,2,3}/    cold / warm_astar / warm_bco logs + checkpoints
  ablation_runs/                matched-budget arms, logs + checkpoints
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10. PyTorch CPU is sufficient and is what the paper used: the policy
is a ~48k-parameter MLP and the wall-clock cost is dominated by the NumPy
environment, so a GPU buys essentially nothing.

## Reproducing the paper

Run from the repository root. Scripts that only re-analyse shipped logs take
seconds; the ones that retrain are marked.

| Paper item | Command |
|---|---|
| Table 1, demonstration diversity | `python analyze_demo_budget.py demos/demos_astar.npz demos/demos_bco_k1.npz demos/demos_bco.npz demos/demos_bco_k5.npz` |
| Table 2, planner benchmark + McNemar | `python analyze_significance.py` |
| Table 2, regenerate from scratch (~hours) | `python benchmark.py` |
| Tables 4–5, held-out evaluation | `python evaluate_final.py --episodes 50` |
| Section 5.2 statistics | `python analyze_paired.py` |
| Table 6, ablation | `python analyze_ablation.py` |
| Tables 7–8, observation degradation (~25 min) | `python experiment_sync.py --episodes 30` |
| Table 9, robot-count sweep (~40 min) | `python experiment_scale.py --episodes 30` |
| Figures 1–4 | `python fig_architecture_v4.py` · `fig_scenario_v3.py` · `fig_curves_grid.py` · `fig_ablation.py` |

### Retraining

The twelve ablation runs (4 teachers × 3 seeds × 500k steps, roughly 45 minutes
each on CPU):

```bash
python run_ablation_local.py          # resumable; skips runs whose .pt exists
```

Regenerating the demonstration sets, which is the slow part because every
replan invokes the metaheuristic:

```bash
python generate_demos.py --planner astar --episodes 30 --out demos/demos_astar.npz
python generate_demos.py --planner bco   --episodes 30 --variants 3 --out demos/demos_bco.npz
```

A single training run:

```bash
python train_marl.py --tag mytag --warmstart demos/demos_bco.npz \
                     --total-steps 500000 --seed 1 --out-dir results/my_runs
```

Note `--out-dir`. Do not write run logs to a synced folder (Google Drive,
OneDrive): their virtual filesystems invalidate long-lived write handles and
will kill a run partway with `OSError 22`.

## Reproducibility notes

- **Seed ranges are disjoint by construction.** Demonstrations use scenario
  seeds 5000–5029, in-training evaluation 9000–9004, held-out evaluation
  20000–20049, and training rollouts `train_seed × 100000` upward.
- **A run is complete only if its `.pt` exists.** The checkpoint is written
  after the final step, so a `.csv` without one is a run that died midway.
- **Three training seeds.** This is the binding limitation on every comparison
  in the paper and the reason most results are reported as consistent in
  direction rather than significant.
- The planner benchmark and the demonstration sets are stochastic; re-running
  `benchmark.py` or `generate_demos.py` will not reproduce the shipped files
  bit-for-bit, though the shipped files are what the paper's tables use.

## License

MIT, see `LICENSE`.
