"""Figure 2 — two-panel scenario figure.

Redesign goals over v2:
  (a) start/goal pairing is made explicit with a dashed task line and both
      endpoints labelled, and moving obstacles carry their speed;
  (b) the direction of time is unambiguous: each moving obstacle gets a dotted
      centre track with an arrowhead, robot trajectories carry mid-path
      arrowheads, and every reactive replan is marked on the trajectory;
  * both panels get a real legend and the run's measured outcome is annotated.

Also re-emits dt_thumbnail.png (used inside Figure 1) with trajectories drawn,
so it stays legible when scaled down.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from digital_twin_env import DigitalTwinEnv, GRID

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1012
# Deliberately red-free: red is reserved for obstacles, so a robot trajectory
# can never be confused with a hazard marking.
ROBOT_COLORS = ["#1565c0", "#2e7d32", "#6a1b9a", "#ef6c00", "#0097a7"]
STAT_FC, STAT_EC = "#bdbdbd", "#424242"
MOV_FC, MOV_EC = "#ef9a9a", "#b71c1c"

# ── run the scenario, recording robots, obstacles and replan events ──
env = DigitalTwinEnv(num_robots=5, num_moving_obs=3, num_static_obs=3,
                     mode="hybrid", seed=SEED, verbose=False)

mov = [o for o in env.obstacles if o.moving]
stat = [o for o in env.obstacles if not o.moving]
mov_init = [(o.x, o.y, o.r) for o in mov]
mov_vel = [(o.vx, o.vy) for o in mov]
mov_hist = [[(o.x, o.y)] for o in mov]

replan_pts = [[] for _ in env.robots]
prev_replans = [r.replan_count for r in env.robots]

steps = 0
while not env.all_reached() and steps < 600:
    env.step()
    for k, o in enumerate(mov):
        mov_hist[k].append((o.x, o.y))
    for i, r in enumerate(env.robots):
        if r.replan_count > prev_replans[i]:
            replan_pts[i].append(tuple(r.position))
            prev_replans[i] = r.replan_count
    steps += 1

n_reached = sum(r.reached for r in env.robots)
n_replans = sum(r.replan_count for r in env.robots)
print(f"seed {SEED}: reached {n_reached}/5, collisions {env.collision_frames}, "
      f"steps {steps}, replans {n_replans}")


def draw_frame(ax, title):
    ax.set_xlim(0, GRID)
    ax.set_ylim(0, GRID)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=22)
    ax.set_xlabel("x (units)", fontsize=11, fontweight="bold")
    ax.set_ylabel("y (units)", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9.5, width=1.3)
    for s in ax.spines.values():
        s.set_linewidth(1.6)
    ax.grid(True, color="#e4e4e4", linewidth=0.7)
    ax.set_axisbelow(True)


def draw_static(ax):
    for o in stat:
        ax.add_patch(Circle((o.x, o.y), o.r, facecolor=STAT_FC,
                            edgecolor=STAT_EC, linewidth=1.8, zorder=2))


def path_arrow(ax, pts, color, frac=0.55, size=17):
    """Arrowhead along a polyline, marking the direction of travel."""
    p = np.asarray(pts, dtype=float)
    if len(p) < 4:
        return
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return
    i = int(np.searchsorted(cum, frac * cum[-1]))
    i = max(1, min(i, len(p) - 1))
    ax.annotate("", xy=p[i], xytext=p[i - 1],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0,
                                mutation_scale=size, shrinkA=0, shrinkB=0),
                zorder=7)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 6.6))

# ══ Panel (a): initial configuration ═══════════════════════════════
draw_frame(axA, "(a) Initial configuration")
draw_static(axA)

for (x, y, r), (vx, vy) in zip(mov_init, mov_vel):
    axA.add_patch(Circle((x, y), r, facecolor=MOV_FC, edgecolor=MOV_EC,
                         linewidth=2.0, zorder=3))
    sp = float(np.hypot(vx, vy))
    ux, uy = vx / (sp + 1e-9), vy / (sp + 1e-9)
    # Arrow starts on the rim, not the centre, so the speed label can sit
    # inside the disc where it can never collide with a goal label.
    axA.annotate("", xy=(x + ux * (r + 17), y + uy * (r + 17)),
                 xytext=(x + ux * r * 0.55, y + uy * r * 0.55),
                 arrowprops=dict(arrowstyle="-|>", color=MOV_EC, lw=2.4),
                 zorder=4)
    axA.text(x - ux * r * 0.34, y - uy * r * 0.34, f"{sp:.1f}",
             ha="center", va="center", fontsize=8.4, fontweight="bold",
             color="#7f1010", zorder=5)

for i, rob in enumerate(env.robots):
    c = ROBOT_COLORS[i % 5]
    sx, sy = rob.start
    gx, gy = rob.goal
    axA.plot([sx, gx], [sy, gy], color=c, linewidth=1.2, linestyle=(0, (5, 4)),
             alpha=0.45, zorder=4)
    axA.plot(sx, sy, marker="s", color=c, markersize=11, markeredgecolor="black",
             markeredgewidth=1.0, zorder=6)
    axA.plot(gx, gy, marker="*", color=c, markersize=18, markeredgecolor="black",
             markeredgewidth=1.0, zorder=6)
    axA.annotate(f"R{i+1}", (sx, sy), textcoords="offset points", xytext=(8, 6),
                 fontsize=9.5, fontweight="bold", color=c, zorder=7)
    axA.annotate(f"G{i+1}", (gx, gy), textcoords="offset points", xytext=(8, 6),
                 fontsize=9.5, fontweight="bold", color=c, zorder=7)

# Placed above the axes, never on top of the scene: an in-panel box hid the
# G5 label and part of a trajectory in the previous revision.
axA.text(0.5, 1.012,
         f"5 robots · 3 static · 3 moving obstacles · "
         f"{GRID} × {GRID} units · Δt = 0.5 s",
         transform=axA.transAxes, ha="center", va="bottom", fontsize=8.6,
         color="#444444")

axA.legend(handles=[
    Line2D([], [], marker="s", color="none", markerfacecolor="#777777",
           markeredgecolor="black", markersize=10, label="robot start"),
    Line2D([], [], marker="*", color="none", markerfacecolor="#777777",
           markeredgecolor="black", markersize=16, label="robot goal"),
    Line2D([], [], color="#777777", linestyle=(0, (5, 4)), linewidth=1.4,
           label="start–goal pair"),
    Patch(facecolor=STAT_FC, edgecolor=STAT_EC, linewidth=1.6,
          label="static obstacle"),
    Line2D([], [], marker="o", color="none", markerfacecolor=MOV_FC,
           markeredgecolor=MOV_EC, markersize=11,
           label="moving obstacle (arrow = heading, value = speed)"),
], loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.8,
    framealpha=0.96, borderpad=0.7, columnspacing=1.4)

# ══ Panel (b): executed trajectories ═══════════════════════════════
draw_frame(axB, "(b) Executed trajectories (hybrid planner)")
draw_static(axB)

N_GHOST = 5
for (x0, y0, r), hist in zip(mov_init, mov_hist):
    h = np.array(hist)
    idxs = np.linspace(0, len(h) - 1, N_GHOST).astype(int)
    # Connect only the sampled snapshots: the full bouncing track over ~180
    # steps is unreadable spaghetti and competes with the robot trajectories.
    axB.plot(h[idxs, 0], h[idxs, 1], color=MOV_EC, linewidth=1.3,
             linestyle=(0, (1.5, 2.5)), alpha=0.75, zorder=3)
    for gi, idx in enumerate(idxs):
        frac = gi / (N_GHOST - 1)
        gx, gy = h[idx]
        if gi == N_GHOST - 1:
            axB.add_patch(Circle((gx, gy), r, facecolor=MOV_FC,
                                 edgecolor=MOV_EC, linewidth=2.0, zorder=4))
        else:
            axB.add_patch(Circle((gx, gy), r,
                                 facecolor=(0.72, 0.11, 0.11, 0.07 + 0.16 * frac),
                                 edgecolor=(0.72, 0.11, 0.11, 0.30 + 0.35 * frac),
                                 linewidth=1.2, zorder=3))
    path_arrow(axB, h[idxs], MOV_EC, frac=0.85, size=14)

for i, rob in enumerate(env.robots):
    c = ROBOT_COLORS[i % 5]
    arr = np.array(rob.path)
    axB.plot(arr[:, 0], arr[:, 1], color=c, linewidth=2.5, alpha=0.95,
             zorder=5, label=f"robot {i+1}", solid_capstyle="round")
    path_arrow(axB, arr, c, frac=0.55)
    if replan_pts[i]:
        rp = np.array(replan_pts[i])
        axB.plot(rp[:, 0], rp[:, 1], linestyle="none", marker="o",
                 markersize=4.2, markerfacecolor="white", markeredgecolor=c,
                 markeredgewidth=1.3, zorder=6)
    axB.plot(*rob.start, marker="s", color=c, markersize=10,
             markeredgecolor="black", markeredgewidth=1.0, zorder=7)
    axB.plot(*rob.goal, marker="*", color=c, markersize=17,
             markeredgecolor="black", markeredgewidth=1.0, zorder=7)

axB.text(0.5, 1.012,
         f"{n_reached}/5 reached · {env.collision_frames} collision frames · "
         f"{steps} steps · {n_replans} reactive replans",
         transform=axB.transAxes, ha="center", va="bottom", fontsize=8.6,
         color="#444444")

handles = [Line2D([], [], color=ROBOT_COLORS[i], linewidth=2.5,
                  label=f"robot {i+1}") for i in range(5)]
handles += [
    Line2D([], [], marker="o", color="none", markerfacecolor="white",
           markeredgecolor="#555555", markeredgewidth=1.3, markersize=7,
           label="reactive replan"),
    Line2D([], [], color=MOV_EC, linestyle=(0, (1.5, 2.5)), linewidth=1.5,
           label="moving-obstacle track"),
    Line2D([], [], marker="o", color="none", markerfacecolor=(0.72, 0.11, 0.11, 0.18),
           markeredgecolor=(0.72, 0.11, 0.11, 0.5), markersize=11,
           label="past obstacle position"),
]
axB.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
           ncol=3, fontsize=8.8, framealpha=0.96, borderpad=0.7,
           columnspacing=1.4)

plt.tight_layout()
plt.subplots_adjust(bottom=0.24)
plt.savefig("figure_scenario.png",
            dpi=300, bbox_inches="tight", facecolor="white")
print("saved figure_scenario.png")

# ══ Thumbnail for Figure 1 — bold and legible when scaled down ══════
figt, axt = plt.subplots(figsize=(3.2, 3.2))
axt.set_xlim(0, GRID)
axt.set_ylim(0, GRID)
axt.set_aspect("equal")
axt.set_xticks([]); axt.set_yticks([])
axt.set_facecolor("white")
for o in stat:
    axt.add_patch(Circle((o.x, o.y), o.r, facecolor=STAT_FC,
                         edgecolor=STAT_EC, linewidth=2.2, zorder=2))
for (x, y, r), (vx, vy) in zip(mov_init, mov_vel):
    axt.add_patch(Circle((x, y), r, facecolor=MOV_FC, edgecolor=MOV_EC,
                         linewidth=2.2, zorder=3))
    sp = np.hypot(vx, vy) + 1e-9
    axt.annotate("", xy=(x + vx / sp * (r + 16), y + vy / sp * (r + 16)),
                 xytext=(x, y),
                 arrowprops=dict(arrowstyle="-|>", color=MOV_EC, lw=2.6),
                 zorder=4)
for i, rob in enumerate(env.robots):
    c = ROBOT_COLORS[i % 5]
    arr = np.array(rob.path)
    axt.plot(arr[:, 0], arr[:, 1], color=c, linewidth=2.6, alpha=0.95, zorder=5)
    axt.plot(*rob.start, marker="s", color=c, markersize=10,
             markeredgecolor="black", markeredgewidth=1.0, zorder=6)
    axt.plot(*rob.goal, marker="*", color=c, markersize=15,
             markeredgecolor="black", markeredgewidth=1.0, zorder=6)
# No spines: Figure 1 draws its own frame around the image extent, and a baked
# border here would show up as a second, doubled frame.
for s in axt.spines.values():
    s.set_visible(False)
plt.tight_layout(pad=0.05)
plt.savefig("dt_thumbnail.png",
            dpi=220, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved dt_thumbnail.png")
