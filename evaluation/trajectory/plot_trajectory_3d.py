"""Full 3D hammer-head trajectory — what path does the hammer actually take? The x-z side views hid the
lateral (y) motion; this shows the true 3D path + all three orthographic projections for the four
decomposition strategies (maxoff straight / imponly speed-swing / delonly pressstraight / maxmax both)."""
from __future__ import annotations
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

SP = "evaluation/data"
OUTDIR = "docs/results/assets/2026-07-20_policy_videos"
d = json.load(open(f"{SP}/traj_dc.json")); A = d["arms"]; nx, ny, nz = d["nail_top"]
ARMS = [("maxoff_s0", "maxoff (straight)", "#2ca02c"),
        ("imponly_s0", "imponly (speed→swing)", "#1f77b4"),
        ("delonly_s0", "delonly (impulse→straight press)", "#d62728"),
        ("maxmax_s0", "maxmax (both)", "#ff7f0e")]
def path(key):
    s = np.array([[p[0], p[1], p[2]] for p in A[key]["substep"]])
    ct = np.array([p[3] for p in A[key]["substep"]])
    return s, ct

fig = plt.figure(figsize=(17, 12))
ax3d = fig.add_subplot(2, 2, 1, projection="3d")
axxz = fig.add_subplot(2, 2, 2)   # side (forward x vs height z)
axxy = fig.add_subplot(2, 2, 3)   # top  (forward x vs lateral y)  <- the hidden view
axyz = fig.add_subplot(2, 2, 4)   # front (lateral y vs height z)

for key, lab, c in ARMS:
    if key not in A: continue
    s, ct = path(key)
    ax3d.plot(s[:, 0], s[:, 1], s[:, 2], color=c, lw=2, label=lab)
    ax3d.scatter(*s[0], color=c, marker="o", s=40, edgecolor="k", lw=0.4)      # start
    if ct.any(): ax3d.scatter(s[ct, 0], s[ct, 1], s[ct, 2], color=c, marker="x", s=30)  # contact
    for ax, (i, j) in [(axxz, (0, 2)), (axxy, (0, 1)), (axyz, (1, 2))]:
        ax.plot(s[:, i], s[:, j], color=c, lw=2, label=lab)
        ax.scatter(s[0, i], s[0, j], color=c, marker="o", s=35, edgecolor="k", lw=0.4)
        if ct.any(): ax.scatter(s[ct, i], s[ct, j], color=c, marker="x", s=22)

# nail marker in every panel
ax3d.scatter([nx], [ny], [nz], marker="v", s=90, c="#8c564b"); ax3d.plot([nx, nx], [ny, ny], [nz, nz-0.06], c="#8c564b", lw=3)
for ax, (i, j) in [(axxz, (0, 2)), (axxy, (0, 1)), (axyz, (1, 2))]:
    ax.scatter([[nx, ny, nz][i]], [[nx, ny, nz][j]], marker="v", s=80, c="#8c564b", zorder=5)
ax3d.set_xlabel("x forward (m)"); ax3d.set_ylabel("y lateral (m)"); ax3d.set_zlabel("z height (m)")
ax3d.set_title("3D head path"); ax3d.view_init(elev=18, azim=-60); ax3d.legend(fontsize=8, loc="upper left")
axxz.set(xlabel="x forward (m)", ylabel="z height (m)", title="SIDE (x–z)"); axxz.set_aspect("equal", "box")
axxy.set(xlabel="x forward (m)", ylabel="y lateral (m)", title="TOP (x–y) — the hidden lateral sweep"); axxy.set_aspect("equal", "box")
axyz.set(xlabel="y lateral (m)", ylabel="z height (m)", title="FRONT (y–z)"); axyz.set_aspect("equal", "box")
for ax in (axxz, axxy, axyz): ax.grid(alpha=0.3); ax.axhline if False else None
axxy.axhline(ny, color="#bbb", ls=":", lw=0.8); axxy.axvline(nx, color="#bbb", ls=":", lw=0.8)
fig.suptitle("Hammer-head trajectory — full 3D path + orthographic projections (○=start · ✕=in-contact · ▽=nail)\n"
             "the strategies take genuinely different paths; note the real lateral (y) sweep the x–z views hid",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"{OUTDIR}/trajectory_3d.png", dpi=120)
print(f"-> {OUTDIR}/trajectory_3d.png")

# per-arm path length + net displacement (how indirect is each path?)
print(f"\n{'arm':30s} {'path_len_cm':>11} {'net_disp_cm':>11} {'indirectness':>12}")
for key, lab, c in ARMS:
    if key not in A: continue
    s, _ = path(key)
    seglen = np.linalg.norm(np.diff(s, axis=0), axis=1).sum() * 100
    net = np.linalg.norm(s[-1] - s[0]) * 100
    print(f"{lab:30s} {seglen:>11.1f} {net:>11.1f} {seglen/max(net,1e-6):>12.2f}")
