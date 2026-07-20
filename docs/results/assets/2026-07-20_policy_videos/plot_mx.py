"""mx ablation figures: (1) trajectory grid (4 arms × 3 seeds), (2) the dose-response — swing, contact
speed, and delivered impulse vs impact-maximization reward dose. The result: the forward-swing is CAUSED
by the impact-max reward (not kinematic, not the imitation prior), and it buys DELIVERED IMPULSE (press
integral) — NOT contact speed (velocity-ceiling-limited)."""
from __future__ import annotations
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

IN = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/2cd822e7-752b-4070-969b-f93a7bd62f87/scratchpad/traj_mx.json"
OUTDIR = "docs/results/assets/2026-07-20_policy_videos"
d = json.load(open(IN)); arms = d["arms"]; nail_x, nail_z = d["nail_top"][0], d["nail_top"][2]
for a in arms.values():  # contact dwell (ms) = #contact substeps × physics_dt — the mechanism core
    a["dwell_ms"] = sum(p[3] for p in a["substep"]) * a.get("physics_dt", 0.002) * 1000.0
ARM_ORDER = ["maxoff", "maxon", "maxmax", "maxofftrk"]
DOSE = {"maxoff": 0, "maxon": 10, "maxmax": 28, "maxofftrk": 0}  # impact_progress + delivered weight
TITLE = {"maxoff": "maxoff (0/0)", "maxon": "maxon (8/2)", "maxmax": "maxmax (24/4)",
         "maxofftrk": "maxofftrk (0/0 + prior)"}

def rows(arm): return [arms[f"{arm}_s{s}"] for s in (0, 1, 2) if f"{arm}_s{s}" in arms]

# ---------- Figure 1: trajectory grid (rows = arm, cols = seed) ----------
allx = [p[0] for a in arms.values() for p in a["substep"]]; allz = [p[2] for a in arms.values() for p in a["substep"]]
xlim = (min(allx + [nail_x]) - 0.01, max(allx + [nail_x]) + 0.01); zlim = (min(allz + [nail_z]) - 0.01, max(allz + [nail_z]) + 0.02)
fig, axes = plt.subplots(4, 3, figsize=(11, 13), sharex=True, sharey=True)
for r, arm in enumerate(ARM_ORDER):
    for c, s in enumerate((0, 1, 2)):
        ax = axes[r, c]; key = f"{arm}_s{s}"
        if key not in arms: ax.axis("off"); continue
        a = arms[key]; sub = np.array([[p[0], p[2]] for p in a["substep"]]); ct = np.array([p[3] for p in a["substep"]])
        pts = sub.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(segs, cmap="viridis", array=np.linspace(0, 1, len(segs)), linewidth=2))
        ax.scatter(sub[ct, 0], sub[ct, 1], s=14, c="#d62728", zorder=5)
        ax.scatter(sub[0, 0], sub[0, 1], s=35, c="#2ca02c", zorder=6, edgecolor="k", lw=0.4)
        ax.plot([nail_x, nail_x], [nail_z, nail_z - 0.06], color="#8c564b", lw=2.5)
        ax.axvline(nail_x, color="#bbb", ls=":", lw=0.7)
        ax.set_title(f"{arm} s{s}  swing {a['swing_pre_cm']:+.1f}cm  deliv {a['delivered_x_iref']:.2f}×", fontsize=9)
        ax.set_xlim(xlim); ax.set_ylim(zlim); ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=6)
fig.suptitle("mx ablation — head trajectory by arm (row) × seed (col), identical fixed reset\n"
             "maxoff/maxofftrk strike STRAIGHT · maxmax SWINGS — the swing scales with impact-max reward dose",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(f"{OUTDIR}/mx_trajectory_grid.png", dpi=120)
print(f"-> {OUTDIR}/mx_trajectory_grid.png")

# ---------- Figure 2: dose-response (swing / v_touch / delivered / dwell) ----------
fig2, axs = plt.subplots(1, 4, figsize=(19, 5))
metrics = [("swing_pre_cm", "pre-contact forward swing (cm)", "swing RISES with dose (ON/OFF solid;\ngraded 8→24 underpowered, n=3)"),
           ("v_touch", "contact speed v_touch (m/s)", "speed FLAT ~1.4 (dose-insensitive;\nsoft effort clamp, NOT a hard wall)"),
           ("delivered_x_iref", "delivered impulse (× i_ref)", "impulse RISES with dose\n(~1.9×)"),
           ("dwell_ms", "contact dwell (ms)", "DWELL rises — the mechanism\n(longer contact at FLAT force)")]
dose3 = ["maxoff", "maxon", "maxmax"]
for ax, (key, ylab, sub) in zip(axs, metrics):
    xs = [DOSE[a] for a in dose3]; means = []
    for a in dose3:
        vals = [r[key] for r in rows(a)]; means.append(np.mean(vals))
        ax.scatter([DOSE[a]] * len(vals), vals, s=55, c="#1f77b4", zorder=3, edgecolor="k", lw=0.4)
    ax.plot(xs, means, "-o", c="#ff7f0e", zorder=4, lw=2, label="mean (none arm)")
    # prior control at dose 0
    tv = [r[key] for r in rows("maxofftrk")]
    ax.scatter([0] * len(tv), tv, s=55, c="#888", marker="s", zorder=3, label="maxofftrk (0/0 + prior)")
    ax.set_xticks([0, 10, 28]); ax.set_xticklabels(["maxoff\n0/0", "maxon\n8/2", "maxmax\n24/4"])
    ax.set_xlabel("impact-maximization reward dose (impact_progress + delivered)")
    ax.set_ylabel(ylab); ax.set_title(sub); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig2.suptitle("mx ablation: the forward-swing is a LEARNED impact-maximization maneuver (not kinematic, not the prior).\n"
              "Paid up to 24× for SPEED, the policy went no faster — it bought impulse via longer DWELL at flat force "
              "(press, not momentum). All 12 seat the nail to 32.0mm ⇒ impulse without extra work.", fontsize=11.5)
fig2.tight_layout(rect=[0, 0, 1, 0.93]); fig2.savefig(f"{OUTDIR}/mx_dose_response.png", dpi=120)
print(f"-> {OUTDIR}/mx_dose_response.png")

# summary table
print(f"\n{'arm':12s} {'swing_cm':>18} {'v_touch':>16} {'deliv×iref':>18}")
for a in ARM_ORDER:
    r = rows(a)
    def ms(k): v = [x[k] for x in r]; return f"{np.mean(v):.2f}±{np.std(v):.2f}"
    print(f"{a:12s} {ms('swing_pre_cm'):>18} {ms('v_touch'):>16} {ms('delivered_x_iref'):>18}")
