"""Side-by-side: PREVIOUS (July policies, correlational) vs CURRENT (mx ablation, controlled).
Both traced from the identical fixed reset (seed 12345), so the hammer-head paths are directly
comparable. Top row: 3 July policies — all swing ~4-7cm, all with impact_progress ON, so we could NOT
tell if the swing was reward-caused or kinematic. Bottom row: the mx ablation dials the impact-max
reward 0->8->24 and the swing follows 0.1->4.8->7.4cm — proving it's the reward."""
from __future__ import annotations
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

SP = "evaluation/data"
OUTDIR = "evaluation/results/2026-07-20_fic_maximization"
J = json.load(open(f"{SP}/traj_all.json")); M = json.load(open(f"{SP}/traj_mx.json"))
nail_x, nail_z = J["nail_top"][0], J["nail_top"][2]

def pre_swing(a):
    sub = a["substep"]; fc = next((i for i, p in enumerate(sub) if p[3]), len(sub) - 1)
    return (max(p[0] for p in sub[:fc + 1]) - nail_x) * 100

# (dataset, key, short-title) — top row PREVIOUS, bottom row CURRENT
PANELS = [
    (J, "af1_fixed",  "af1  (impact ON)"),
    (J, "a05_none",   "a05  (impact ON)"),
    (J, "b1_noterm",  "b1  (impact ON)"),
    (M, "maxoff_s0",  "maxoff  (impact 0)"),
    (M, "maxon_s1",   "maxon  (impact 8)"),
    (M, "maxmax_s1",  "maxmax  (impact 24)"),
]
allx = [p[0] for D in (J, M) for a in D["arms"].values() for p in a["substep"]]
allz = [p[2] for D in (J, M) for a in D["arms"].values() for p in a["substep"]]
xlim = (min(allx + [nail_x]) - 0.01, max(allx + [nail_x]) + 0.01)
zlim = (min(allz + [nail_z]) - 0.01, max(allz + [nail_z]) + 0.02)

fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharex=True, sharey=True)
fig.subplots_adjust(top=0.80, bottom=0.09, left=0.08, right=0.97, hspace=0.42, wspace=0.12)
for ax, (D, key, head) in zip(axes.flat, PANELS):
    a = D["arms"][key]; sub = np.array([[p[0], p[2]] for p in a["substep"]]); ct = np.array([p[3] for p in a["substep"]])
    pts = sub.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, cmap="viridis", array=np.linspace(0, 1, len(segs)), linewidth=2.3))
    ax.scatter(sub[ct, 0], sub[ct, 1], s=16, c="#d62728", zorder=5)
    ax.scatter(sub[0, 0], sub[0, 1], s=40, c="#2ca02c", zorder=6, edgecolor="k", lw=0.4)
    ax.plot([nail_x, nail_x], [nail_z, nail_z - 0.06], color="#8c564b", lw=2.6)
    ax.axvline(nail_x, color="#bbb", ls=":", lw=0.8)
    dv = a.get("delivered_x_iref"); dvs = f" · deliv {dv:.2f}×" if dv is not None else ""
    ax.set_title(f"{head}\nswing {pre_swing(a):+.1f}cm{dvs}", fontsize=10)
    ax.set_xlim(xlim); ax.set_ylim(zlim); ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=7)
fig.suptitle("Hammer-head trajectory — PREVIOUS (July, uncontrolled) vs CURRENT (mx ablation)  ·  same fixed reset\n"
             "green=start · viridis=time · red=in-contact · brown ▽=nail", fontsize=12.5, y=0.975)
# row-band labels (centered above each row, no overlap)
fig.text(0.5, 0.855, "PREVIOUS — every competent policy swings ~4–7cm (all have impact_progress ON → couldn't tell if reward-caused or kinematic)",
         ha="center", fontsize=11, color="#555", fontweight="bold")
fig.text(0.5, 0.44, "CURRENT (ablation) — dial the impact-max reward 0→8→24, and the swing follows 0.1→4.8→7.4cm  ·  it's the reward",
         ha="center", fontsize=11, color="#b3402a", fontweight="bold")
fig.text(0.5, 0.03, "forward  x (m) →", ha="center", fontsize=11)
fig.text(0.03, 0.5, "height  z (m) →", va="center", rotation=90, fontsize=11)
fig.savefig(f"{OUTDIR}/before_after_trajectories.png", dpi=125)
print(f"-> {OUTDIR}/before_after_trajectories.png")
