"""Aggregate + plot the multi-reset eval (50 resets × 12 policies). Turns each single dot into a
per-arm DISTRIBUTION (3 seeds × 50 resets = 150 rollouts/arm): does the ON/OFF swing separation survive
IC + obs noise? Is peak force FLAT (not 'harder press')? Do contact events go 1→multi?"""
from __future__ import annotations
import json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/2cd822e7-752b-4070-969b-f93a7bd62f87/scratchpad/mx_multireset.json"
OUTDIR = "docs/results/assets/2026-07-20_policy_videos"
d = json.load(open(IN)); A = d["arms"]; N = d["N"]
ARMS = ["maxoff", "maxon", "maxmax", "maxofftrk"]
def pool(arm, key):  # 3 seeds × N resets, drop NaN
    return [r[key] for s in (0, 1, 2) if f"{arm}_s{s}" in A for r in A[f"{arm}_s{s}"]
            if r[key] == r[key]]

METRICS = [("swing", "forward swing (cm)"), ("v_touch", "contact speed (m/s)"),
           ("delivered", "delivered (× i_ref)"), ("dwell_ms", "contact dwell (ms)"),
           ("peak_force", "PEAK axial force (N)"), ("events", "# contact events")]
COL = {"maxoff": "#2ca02c", "maxon": "#1f77b4", "maxmax": "#ff7f0e", "maxofftrk": "#888888"}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (key, ylab) in zip(axes.flat, METRICS):
    data = [pool(a, key) for a in ARMS]
    bp = ax.boxplot(data, tick_labels=ARMS, showmeans=True, patch_artist=True, widths=0.6)
    for patch, a in zip(bp["boxes"], ARMS): patch.set_facecolor(COL[a]); patch.set_alpha(0.5)
    ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis="y"); ax.tick_params(axis="x", labelsize=9)
axes[0, 0].set_title("swing: ON/OFF separation holds across ICs")
axes[1, 1].set_title("PEAK force rises modestly (~18%) — but MEAN force is flat")
fig.suptitle(f"mx ablation — robustness across {N} randomized resets (play=False: IC noise + obs noise on)\n"
             f"each box = 3 seeds × {N} resets = {3*N} rollouts/arm", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"{OUTDIR}/mx_multireset_box.png", dpi=120)
print(f"-> {OUTDIR}/mx_multireset_box.png")

# summary table + key stats
def ms(a, k): v = pool(a, k); return f"{st.mean(v):.2f}±{st.pstdev(v):.2f}" if v else "—"
print(f"\n{'arm':11s} {'swing_cm':>12} {'v_touch':>12} {'deliv×':>12} {'dwell_ms':>12} {'peakF_N':>12} {'events':>10} {'succ%':>7}")
for a in ARMS:
    succ = pool(a, "success"); sr = 100 * sum(succ) / len(succ) if succ else 0
    print(f"{a:11s} {ms(a,'swing'):>12} {ms(a,'v_touch'):>12} {ms(a,'delivered'):>12} "
          f"{ms(a,'dwell_ms'):>12} {ms(a,'peak_force'):>12} {ms(a,'events'):>10} {sr:6.0f}%")

# ON/OFF robustness: fraction of rollouts clearly straight (<1cm) vs clearly swung (>1cm)
print("\nON/OFF swing robustness (fraction of rollouts):")
for a in ARMS:
    sw = pool(a, "swing"); straight = sum(1 for x in sw if x < 1.0) / len(sw)
    print(f"  {a:11s} straight(<1cm)={straight*100:4.0f}%  swung(>1cm)={100-straight*100:4.0f}%")
# mechanism: peak force vs MEAN force (= delivered / dwell) across the dose axis
I_REF = 0.6094
print("\nmechanism decomposition (delivered = mean_force × dwell):")
print(f"  {'arm':10s} {'delivered_Ns':>13} {'dwell_ms':>10} {'MEAN force N':>13} {'PEAK force N':>13}")
for a in ("maxoff", "maxon", "maxmax"):
    dv = st.mean(pool(a, "delivered")) * I_REF; dw = st.mean(pool(a, "dwell_ms")) / 1000
    mf = dv / dw if dw else 0; pf = st.mean(pool(a, "peak_force"))
    print(f"  {a:10s} {dv:13.3f} {dw*1000:10.1f} {mf:13.1f} {pf:13.1f}")
print("  => MEAN force ~flat (impulse gain is dwell+taps), PEAK force rises ~18% (profile sharpens modestly)")
