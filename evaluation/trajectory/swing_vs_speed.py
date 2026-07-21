"""Partial test of 'the swing is for impact maximization': on the already-traced July policies,
does a bigger PRE-CONTACT forward swing buy a higher contact speed? Reads traj_all.json (substep
head path at 2ms). Fixes the earlier metric: swing measured only UP TO first contact (so nf1's
post-tap fly-off no longer contaminates it)."""
from __future__ import annotations
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = "evaluation/data/traj_all.json"
OUTDIR = "docs/results/assets/2026-07-20_policy_videos"
DT = 0.002
d = json.load(open(IN)); arms = d["arms"]; nail_x = d["nail_top"][0]

rows = []
for label, a in arms.items():
    sub = a["substep"]  # [x,y,z,contact,clamped_mm]
    ok = a["success"]  # ground-truth done flag (survives mjlab auto-reset)
    fc = next((i for i, p in enumerate(sub) if p[3]), None)
    if fc is None or fc < 1:
        rows.append((label, a["max_nail_mm"], float("nan"), float("nan"), None, ok)); continue
    v_touch = (sub[fc - 1][2] - sub[fc][2]) / DT          # downward speed at contact onset
    swing_pre = max(p[0] for p in sub[:fc + 1]) - nail_x    # forward excursion BEFORE contact
    rows.append((label, a["max_nail_mm"], swing_pre * 100, v_touch, fc, ok))

rows.sort(key=lambda r: (-(r[3] if r[3] == r[3] else -1)))
print(f"{'policy':13s} {'peak_mm':>7} {'success':>7} {'swing_pre_cm':>12} {'v_touch_m/s':>11} {'contact@substep':>15}")
for label, depth, sw, vt, fc, ok in rows:
    print(f"{label:13s} {depth:7.1f} {str(ok):>7} {sw:12.1f} {vt:11.3f} {str(fc):>15}")

# scatter: pre-contact swing vs contact speed, split by ground-truth success (done flag)
good = [(l, sw, vt, depth) for (l, depth, sw, vt, fc, ok) in rows if vt == vt and ok]
weak = [(l, sw, vt, depth) for (l, depth, sw, vt, fc, ok) in rows if vt == vt and not ok]
fig, ax = plt.subplots(figsize=(8, 6))
for (l, sw, vt, depth) in good:
    ax.scatter(sw, vt, s=90, c="#1f77b4" if l != "af1_fixed" else "#ff7f0e", zorder=3,
               edgecolor="k", lw=0.6)
    ax.annotate(l, (sw, vt), fontsize=8, xytext=(4, 4), textcoords="offset points")
for (l, sw, vt, depth) in weak:
    ax.scatter(sw, vt, s=60, c="#cccccc", zorder=2, edgecolor="k", lw=0.4)
    ax.annotate(f"{l} (park {depth:.0f}mm)", (sw, vt), fontsize=7, color="#888",
                xytext=(4, -8), textcoords="offset points")
ax.set_xlabel("pre-contact forward swing past nail-x  (cm)")
ax.set_ylabel("contact speed  v_touch  (m/s, downward)")
ax.set_title("Does the swing buy impact speed?\n"
             "blue/orange = successful strike · grey = parker (nf1)   "
             "(flat cloud ⇒ swing does NOT buy speed)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/swing_vs_speed.png", dpi=125)
print(f"\n-> {OUTDIR}/swing_vs_speed.png")
