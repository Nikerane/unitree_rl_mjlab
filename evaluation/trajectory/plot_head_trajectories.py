"""Plot hammer-head trajectories (x-z side view) for all July policies, small-multiples + a
swing-excursion summary. The swing shows as the path bulging past the nail x before coming back."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

IN = "evaluation/data/traj_all.json"
OUTDIR = "docs/results/assets/2026-07-20_policy_videos"
d = json.load(open(IN)); arms = d["arms"]
nail_x, nail_z = d["nail_top"][0], d["nail_top"][2]

order = ["af1_fixed","a05_none","a05_track","b1_noterm","g2_delivoff","dg0_nogate",
         "nf1_none","nf1_track","g1_none","g1_track","dg1_gate","ip24_ip"]
order = [a for a in order if a in arms]

def pre_swing(a):
    """Forward excursion past nail-x measured UP TO first contact only (matches swing_vs_speed.py) —
    so a parker's post-tap fly-off doesn't inflate the 'before striking' number the caption claims."""
    sub = a["substep"]
    fc = next((i for i, p in enumerate(sub) if p[3]), len(sub) - 1)
    return max(p[0] for p in sub[:fc + 1]) - nail_x

# shared limits
allx = [p[0] for a in arms.values() for p in a["substep"]]
allz = [p[2] for a in arms.values() for p in a["substep"]]
xlim = (min(allx+[nail_x])-0.01, max(allx+[nail_x])+0.01)
zlim = (min(allz+[nail_z])-0.01, max(allz+[nail_z])+0.02)

fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
for ax, label in zip(axes.flat, order):
    a = arms[label]; sub = np.array([[p[0], p[2]] for p in a["substep"]])
    ct = np.array([p[3] for p in a["substep"]])
    # path colored by time
    pts = sub.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", array=np.linspace(0, 1, len(segs)), linewidth=2.2)
    ax.add_collection(lc)
    ax.scatter(sub[ct, 0], sub[ct, 1], s=18, c="#d62728", zorder=5, label="contact")  # contact substeps
    ax.scatter(sub[0, 0], sub[0, 1], s=45, c="#2ca02c", marker="o", zorder=6, edgecolor="k", lw=0.5)  # start
    # nail: target marker + shaft
    ax.plot([nail_x, nail_x], [nail_z, nail_z-0.06], color="#8c564b", lw=3, solid_capstyle="butt")
    ax.scatter([nail_x], [nail_z], marker="v", s=70, c="#8c564b", zorder=4)
    ax.axvline(nail_x, color="#bbbbbb", ls=":", lw=0.8, zorder=0)
    # swing excursion (pre-contact only)
    fwd = pre_swing(a)
    ok = a.get("success", a["max_nail_mm"] >= 30.0)  # ground-truth done flag
    tag = "★ " if label == "af1_fixed" else ""
    ax.set_title(f"{tag}{label}   depth {a['max_nail_mm']:.0f}mm"
                 f"\nfwd-swing {fwd*100:+.1f}cm", fontsize=10.5,
                 color="#111" if ok else "#444")
    ax.set_xlim(xlim); ax.set_ylim(zlim); ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
for ax in axes.flat[len(order):]:
    ax.axis("off")
fig.suptitle("Hammer-head trajectory (x–z side view) across July policies — identical fixed reset (seed 12345)\n"
             "green=start · viridis=normalized time (per panel) · red=in-contact · brown ▽=nail top   "
             "(‘fwd-swing’ = how far past the nail-x the head arcs before striking)",
             fontsize=12.5)
fig.text(0.5, 0.02, "forward  x (m) →", ha="center", fontsize=11)
fig.text(0.04, 0.5, "height  z (m) →", va="center", rotation=90, fontsize=11)
fig.tight_layout(rect=[0.05, 0.03, 1, 0.94])
fig.savefig(f"{OUTDIR}/trajectories_grid.png", dpi=125)
print(f"-> {OUTDIR}/trajectories_grid.png")

# summary: forward-swing excursion per policy
fig2, ax = plt.subplots(figsize=(9, 5))
labs = sorted(order, key=lambda l: pre_swing(arms[l]))
fwd = [pre_swing(arms[l]) * 100 for l in labs]
cols = ["#1f77b4" if l != "af1_fixed" else "#ff7f0e" for l in labs]
ax.barh(labs, fwd, color=cols)
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("forward swing past nail-x  (cm)")
ax.set_title("How far past the nail does each policy's head arc before striking?\n"
             "(0 = straight-down descent; large + = pronounced forward swing)")
for i, v in enumerate(fwd):
    ax.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:+.1f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8)
fig2.tight_layout(); fig2.savefig(f"{OUTDIR}/swing_excursion.png", dpi=125)
print(f"-> {OUTDIR}/swing_excursion.png")
