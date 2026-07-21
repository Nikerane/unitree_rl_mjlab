"""Clean per-panel x-z trajectory grid (the `trajectories_grid.png` style — viridis=time, green=start,
red=in-contact, brown ▽=nail), applied to the decomposition (dc) and longer-training (lg) campaigns.
Row = arm, col = seed.  Run: python plot_grid.py dc   |   python plot_grid.py lg"""
from __future__ import annotations
import sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

SP = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/2cd822e7-752b-4070-969b-f93a7bd62f87/scratchpad"
OUTDIR = "docs/results/assets/2026-07-20_policy_videos"
which = sys.argv[1] if len(sys.argv) > 1 else "dc"
if which == "dc":
    IN, OUT = f"{SP}/traj_dc.json", f"{OUTDIR}/dc_trajectory_grid.png"
    ROWS = [("maxoff", "maxoff (0/0)"), ("imponly", "imponly (24/0 · speed reward)"),
            ("delonly", "delonly (0/4 · impulse reward)"), ("maxmax", "maxmax (24/4)")]
    SUB = "Decomposition — path by reward (row) × seed (col): speed→swing, impulse→straight-ish press"
else:
    IN, OUT = f"{SP}/traj_lg.json", f"{OUTDIR}/lg_trajectory_grid.png"
    ROWS = [("maxoff500", "maxoff · 500 it"), ("maxoff1500", "maxoff · 1500 it"),
            ("maxmax500", "maxmax · 500 it"), ("maxmax1500", "maxmax · 1500 it")]
    SUB = "500 vs 1500 iterations — path by arm (row) × seed (col): more training drifts, not improves"

d = json.load(open(IN)); A = d["arms"]; nx, nz = d["nail_top"][0], d["nail_top"][2]
def pre_swing(a):
    s = a["substep"]; fc = next((i for i, p in enumerate(s) if p[3]), len(s) - 1)
    return (max(p[0] for p in s[:fc + 1]) - nx) * 100
allx = [p[0] for a in A.values() for p in a["substep"]]; allz = [p[2] for a in A.values() for p in a["substep"]]
xlim = (min(allx + [nx]) - 0.01, max(allx + [nx]) + 0.01); zlim = (min(allz + [nz]) - 0.01, max(allz + [nz]) + 0.02)

fig, axes = plt.subplots(4, 3, figsize=(12, 13), sharex=True, sharey=True)
for r, (arm, lab) in enumerate(ROWS):
    for c, s in enumerate((0, 1, 2)):
        ax = axes[r, c]; key = f"{arm}_s{s}"
        if key not in A: ax.axis("off"); continue
        a = A[key]; sub = np.array([[p[0], p[2]] for p in a["substep"]]); ct = np.array([p[3] for p in a["substep"]])
        pts = sub.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(segs, cmap="viridis", array=np.linspace(0, 1, len(segs)), linewidth=2.2))
        ax.scatter(sub[ct, 0], sub[ct, 1], s=16, c="#d62728", zorder=5)
        ax.scatter(sub[0, 0], sub[0, 1], s=42, c="#2ca02c", zorder=6, edgecolor="k", lw=0.4)
        ax.plot([nx, nx], [nz, nz - 0.06], color="#8c564b", lw=2.6); ax.scatter([nx], [nz], marker="v", s=70, c="#8c564b", zorder=4)
        ax.axvline(nx, color="#bbb", ls=":", lw=0.8)
        ok = a.get("success", a["max_nail_mm"] >= 30.0)
        ax.set_title(f"{lab}  ·  s{s}\nswing {pre_swing(a):+.1f}cm · {a['max_nail_mm']:.0f}mm{'' if ok else ' · FAIL'}",
                     fontsize=9.5, color="#111" if ok else "#b3402a")
        ax.set_xlim(xlim); ax.set_ylim(zlim); ax.set_aspect("equal", "box"); ax.tick_params(labelsize=7)
fig.suptitle(f"Hammer-head trajectory (x–z side view) — {SUB}\n"
             "green=start · viridis=normalized time · red=in-contact · brown ▽=nail", fontsize=12)
fig.text(0.5, 0.02, "forward  x (m) →", ha="center", fontsize=11)
fig.text(0.04, 0.5, "height  z (m) →", va="center", rotation=90, fontsize=11)
fig.tight_layout(rect=[0.04, 0.03, 1, 0.95]); fig.savefig(OUT, dpi=125)
print(f"-> {OUT}")
