"""Figure 1 — framework architecture.

Redesign goals over v3:
  * every arrow leaves the box that actually produces the artefact
    (the benchmark is fed by the twin's scenario stream, not by the demo set);
  * the evaluation arrow closes the loop back into the twin instead of
    dangling near the benchmark;
  * numbered stage badges so the pipeline can be read in order;
  * the three compared initializations (cold / warm-A* / warm-BCO-GA) are
    shown explicitly, since that ablation is the paper's core claim;
  * the workspace thumbnail is placed with imshow + explicit extent so it is
    large enough to read.

Geometry note: the canvas is 100 units wide drawn at 13 in, so one unit is
about 9.4 pt. Every box below is sized so its longest string fits inside it
at the chosen font size — text overflow was the main defect of the previous
revision.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = "figure_architecture.png"
THUMB = "dt_thumbnail.png"

NAVY, GREEN, ORANGE, GREY = "#0d47a1", "#1b5e20", "#b45309", "#8a8a8a"

fig, ax = plt.subplots(figsize=(13.0, 8.19))
ax.set_xlim(0, 100)
ax.set_ylim(0, 63)
ax.set_aspect("equal")
ax.axis("off")


def box(x0, y0, x1, y1, title, body="", fc="#eef2f8", ec=NAVY,
        tfs=11.5, bfs=8.6, lw=2.2, t_dy=0.66, b_dy=0.28):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, facecolor=fc,
                                edgecolor=ec, boxstyle="round,pad=0.30",
                                linewidth=lw, zorder=4))
    cx, h = (x0 + x1) / 2, y1 - y0
    if body:
        ax.text(cx, y0 + h * t_dy, title, ha="center", va="center",
                fontsize=tfs, color=ec, fontweight="bold", zorder=5)
        ax.text(cx, y0 + h * b_dy, body, ha="center", va="center",
                fontsize=bfs, color="#222222", zorder=5, linespacing=1.5)
    else:
        ax.text(cx, y0 + h / 2, title, ha="center", va="center",
                fontsize=tfs, color=ec, fontweight="bold", zorder=5)


def badge(x0, y1, n, color=NAVY):
    ax.add_patch(Circle((x0 + 1.0, y1 - 1.1), 1.5, facecolor=color,
                        edgecolor="white", linewidth=1.4, zorder=7))
    ax.text(x0 + 1.0, y1 - 1.1, str(n), ha="center", va="center", fontsize=9.5,
            color="white", fontweight="bold", zorder=8)


def arrow(p0, p1, ls="-", color="#111111", lw=2.4, rad=0.0, head=True):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>" if head else "-", mutation_scale=22,
        linewidth=lw, linestyle=ls, color=color, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}", zorder=6))


def flag(x, y, text, fs=8.4):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            style="italic", fontweight="bold", color="#111111", zorder=9,
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                      edgecolor="none"))


# ── Physical system (dashed: not yet deployed) ─────────────────────
ax.add_patch(FancyBboxPatch((1.0, 38.0), 13.0, 14.0, facecolor="#f6f6f6",
                            edgecolor=GREY, linestyle="--",
                            boxstyle="round,pad=0.35", linewidth=2.2, zorder=4))
ax.text(7.5, 47.3, "Physical\nmulti-robot\nsystem", ha="center", va="center",
        fontsize=10.5, color="#4a4a4a", fontweight="bold", zorder=5,
        linespacing=1.4)
ax.text(7.5, 41.0, "(not yet deployed)", ha="center", va="center",
        fontsize=7.8, color=GREY, style="italic", zorder=5)

# ── Container: DIGITAL TWIN ────────────────────────────────────────
ax.add_patch(FancyBboxPatch((16, 13), 45, 45, facecolor="#fbfcff",
                            edgecolor=NAVY, boxstyle="round,pad=0.5",
                            linewidth=3.0, zorder=2))
ax.text(38.5, 55.3, "DIGITAL TWIN", ha="center", va="center", fontsize=13.5,
        color=NAVY, fontweight="bold", zorder=5)

# workspace thumbnail, placed by explicit extent so it stays legible
TX0, TX1, TY0, TY1 = 19.0, 37.0, 33.0, 51.0
try:
    ax.imshow(mpimg.imread(THUMB), extent=[TX0, TX1, TY0, TY1],
              aspect="auto", zorder=3, interpolation="antialiased")
except FileNotFoundError:
    ax.add_patch(Rectangle((TX0, TY0), TX1 - TX0, TY1 - TY0,
                           facecolor="#ffffff", edgecolor=NAVY, zorder=3))
ax.add_patch(Rectangle((TX0, TY0), TX1 - TX0, TY1 - TY0, facecolor="none",
                       edgecolor=NAVY, linewidth=2.2, zorder=5))
ax.text(28.0, 31.4, "synchronized workspace", ha="center", va="center",
        fontsize=8.2, color=NAVY, fontweight="bold", zorder=5)

box(40, 39.5, 59, 50.5, "Hybrid planner",
    "BCO-GA global search\n+ A* reactive replanning", fc="#e3edf9", tfs=10.5)
badge(40, 50.5, 1)

box(40, 23.5, 59, 34.5, "Rollouts → demonstrations",
    "A*: 1 path per scenario\nBCO-GA: K = 3 diverse paths",
    fc="#eaf5ea", ec=GREEN, tfs=8.2)
badge(40, 34.5, 2, GREEN)

box(19, 15.0, 37, 26.0, "Planner benchmark",
    "A*, PSO, ACO-R, GWO,\nBCO-GA, hybrid\n100 seeded scenarios",
    fc="#ffffff", tfs=10.0, bfs=7.6, t_dy=0.76, b_dy=0.33)

# ── Container: LEARNING LAYER ──────────────────────────────────────
ax.add_patch(FancyBboxPatch((68, 13), 31, 45, facecolor="#fcfdfb",
                            edgecolor=GREEN, boxstyle="round,pad=0.5",
                            linewidth=3.0, zorder=2))
ax.text(83.5, 55.3, "LEARNING LAYER", ha="center", va="center", fontsize=13.5,
        color=GREEN, fontweight="bold", zorder=5)

ax.add_patch(FancyBboxPatch((69, 46.5), 29, 5.5, facecolor="#f4f7f2",
                            edgecolor="#7a9a7a", linestyle="--",
                            boxstyle="round,pad=0.28", linewidth=1.6, zorder=4))
ax.text(83.5, 50.4, "three configurations compared", ha="center", va="center",
        fontsize=7.8, color="#33502f", fontweight="bold", zorder=5)
ax.text(83.5, 48.2, "cold (no BC)  ·  warm-start from A*  ·  warm-start from BCO-GA",
        ha="center", va="center", fontsize=6.4, color="#33502f", zorder=5)

box(70, 36.5, 97, 45.5, "Behavior cloning",
    "initialize the policy head from D\n15 epochs · cross-entropy",
    fc="#eaf5ea", ec=GREEN, tfs=11.5)
badge(70, 45.5, 3, GREEN)

box(70, 24.5, 97, 33.5, "Shared-policy PPO",
    "multi-agent RL · 500 000 steps\n3 seeds per configuration",
    fc="#eaf5ea", ec=GREEN, tfs=11.5)
badge(70, 33.5, 4, GREEN)

box(70, 14.0, 97, 21.5, "Learned policy",
    "anticipatory execution among moving obstacles",
    fc="#fdf0e2", ec=ORANGE, tfs=11.5, bfs=8.0)
badge(70, 21.5, 5, ORANGE)

# ── Arrows ─────────────────────────────────────────────────────────
arrow((14.2, 45.0), (15.6, 45.0), ls="--", color="#555555")   # physical → twin
flag(15.0, 47.2, "sync", fs=8.0)

arrow((37.2, 45.0), (39.5, 45.0))                             # workspace → planner
arrow((49.5, 39.3), (49.5, 34.9))                             # planner → rollouts
arrow((28.0, 30.2), (28.0, 26.4))                             # scenarios → benchmark

arrow((59.3, 29.0), (69.7, 41.0), rad=-0.18)                  # demos → BC
flag(64.5, 32.8, "demonstrations")

# evaluation loop: policy → below both containers → back into the twin
arrow((83.5, 13.8), (83.5, 8.0), head=False, color=ORANGE)
arrow((83.5, 8.0), (28.0, 8.0), head=False, color=ORANGE)
arrow((28.0, 8.0), (28.0, 14.6), color=ORANGE)
flag(56.0, 9.5, "evaluation on held-out twin scenarios")

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT)
