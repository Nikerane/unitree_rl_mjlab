"""Decomposition figure (30 randomized resets/policy, play=False): the two impact-max rewards drive
DISSOCIABLE strategies. impact_progress(speed) -> SWING (dwell+taps at flat force, no speed gain);
delivered_impulse(impulse) -> straight HARD PRESS (higher peak+mean force). maxmax = blend."""
from __future__ import annotations
import json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = "evaluation/data/dc_multireset.json"
OUTDIR = "docs/results/assets/2026-07-20_policy_videos"
A = json.load(open(IN))["arms"]
ARMS = ["maxoff", "imponly", "delonly", "maxmax"]
LAB = {"maxoff": "maxoff\n(neither)", "imponly": "imponly\n(SPEED 24/0)",
       "delonly": "delonly\n(IMPULSE 0/4)", "maxmax": "maxmax\n(both)"}
COL = {"maxoff": "#888888", "imponly": "#1f77b4", "delonly": "#d62728", "maxmax": "#ff7f0e"}
def pool(a, k): return [r[k] for s in (0, 1, 2) if f"{a}_s{s}" in A for r in A[f"{a}_s{s}"] if r[k] == r[k]]

METRICS = [("swing", "forward swing (cm)", "SWING = the speed reward\n(imponly), NOT impulse"),
           ("v_touch", "contact speed (m/s)", "speed FLAT ~1.4 for ALL\n(no reward buys speed)"),
           ("delivered", "delivered (× i_ref)", "both rewards ↑ delivered\n(different mechanisms)"),
           ("dwell_ms", "contact dwell (ms)", "dwell ↑ (speed-reward swing\n+ taps)"),
           ("peak_force", "PEAK axial force (N)", "FORCE = the impulse reward\n(delonly), NOT speed"),
           ("events", "# contact events", "taps ↑ with the swing")]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (k, ylab, title) in zip(axes.flat, METRICS):
    bp = ax.boxplot([pool(a, k) for a in ARMS], tick_labels=[LAB[a] for a in ARMS],
                    showmeans=True, patch_artist=True, widths=0.6)
    for patch, a in zip(bp["boxes"], ARMS): patch.set_facecolor(COL[a]); patch.set_alpha(0.5)
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=8)
fig.suptitle("Decomposition — the two impact-max rewards drive DISSOCIABLE strategies (30 resets/policy)\n"
             "impact_progress(speed) → wind-up SWING (dwell+taps, no speed gain) · "
             "delivered_impulse(impulse) → straight HARD PRESS (peak force 43N) · neither breaks the ~1.4 m/s ceiling",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{OUTDIR}/dc_decomposition_box.png", dpi=120)
print(f"-> {OUTDIR}/dc_decomposition_box.png")
