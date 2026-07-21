"""Hero figure: the best policy's (af1) hammer-head path, clean x-z style, phases annotated —
the canonical 'this is the trajectory the hammer takes' figure for the deck."""
from __future__ import annotations
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

IN = "evaluation/data/traj_all.json"
OUT = "evaluation/results/2026-07-20_fic_maximization/af1_hero_trajectory.png"
d = json.load(open(IN)); nx, nz = d["nail_top"][0], d["nail_top"][2]
a = d["arms"]["af1_fixed"]; sub = np.array([[p[0], p[2]] for p in a["substep"]]); ct = np.array([p[3] for p in a["substep"]])

fig, ax = plt.subplots(figsize=(7, 8))
pts = sub.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap="viridis", array=np.linspace(0, 1, len(segs)), linewidth=3.2)
ax.add_collection(lc)
ax.scatter(sub[ct, 0], sub[ct, 1], s=45, c="#d62728", zorder=5, label="in contact")
ax.scatter(sub[0, 0], sub[0, 1], s=90, c="#2ca02c", marker="o", zorder=6, edgecolor="k", lw=0.6, label="start")
ax.plot([nx, nx], [nz, nz - 0.06], color="#8c564b", lw=4, solid_capstyle="butt")
ax.scatter([nx], [nz], marker="v", s=140, c="#8c564b", zorder=4, label="nail")
ax.axvline(nx, color="#bbb", ls=":", lw=1)
# phase annotations along the path
apex = int(np.argmax(sub[:, 0]))              # forward-most point (wind-up apex)
fc = next((i for i, c in enumerate(ct) if c), len(sub) - 1)
ax.annotate("① start", (sub[0, 0], sub[0, 1]), fontsize=11, xytext=(12, 0), textcoords="offset points", color="#2ca02c")
ax.annotate("② wind-up\n(forward arc)", (sub[apex, 0], sub[apex, 1]), fontsize=10.5, xytext=(10, 0),
            textcoords="offset points", color="#333")
ax.annotate("③ descend", (sub[(apex + fc) // 2, 0], sub[(apex + fc) // 2, 1]), fontsize=10.5, xytext=(10, 0),
            textcoords="offset points", color="#333")
ax.annotate("④ strike", (sub[fc, 0], sub[fc, 1]), fontsize=11, xytext=(14, -6), textcoords="offset points", color="#d62728")
ax.set_xlabel("forward  x (m) →", fontsize=12); ax.set_ylabel("height  z (m) →", fontsize=12)
ax.set_aspect("equal", "box"); ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=10)
ax.set_title(f"af1 — best policy: hammer-head path (side view)\n"
             f"forward-swing +{(max(p[0] for p in a['substep'][:fc+1])-nx)*100:.1f}cm · "
             f"drives nail to {a['max_nail_mm']:.0f}mm · viridis = time",
             fontsize=12)
fig.tight_layout(); fig.savefig(OUT, dpi=140)
print(f"-> {OUT}")
