"""Analysis figures for the impulse-CaT vacuity + enforcement findings (thesis-grade, CVD-safe).
Palette: Okabe-Ito subset, validated PASS (dataviz skill). No dual-axis. matplotlib Agg -> PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/c51273df-43b1-4386-842f-086eab5e35db/scratchpad/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"    # reachable / safe / enforcement-ON
VERM = "#D55E00"    # over-cap / artifact / critical
GREEN = "#009E73"   # secondary series
MUTE = "#9aa0ab"    # unreachable ceiling / recessive
INK = "#22262e"; INK2 = "#5b616b"; GRID = "#e6e8ec"; CAPLINE = "#6b7280"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": "#c7ccd3", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
})
def clean(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

# ---------- FIG 1: Λ/cap ceiling across scenarios ----------
labels = ["Natural strike\n(gentle case)", "Realistic strike\n@4.65 rad/s",
          "Best coordinated whip\n@4.65 rad/s (beyond DiffIK)", "Analytic ceiling\n(all-joints, full stop)",
          "7 kg-nail press\n(1000× mass)", "Whip @8 rad/s\n(90% sustained press)"]
vals = [0.11, 0.36, 0.39, 0.90, 1.09, 1.63]
# "reach" = genuinely policy-reachable; "ceil" = kinematically unreachable by DiffIK (shown for
# context only — includes both the coordinated-whip kinematic ceiling and the analytic full-stop
# ceiling); "artifact" = measurement artifact (sustained press masquerading as an impulse).
kind = ["reach", "reach", "ceil", "ceil", "artifact", "artifact"]
cmap = {"reach": BLUE, "ceil": MUTE, "artifact": VERM}
hatchmap = {"reach": "", "ceil": "///", "artifact": "xx"}
colors = [cmap[k] for k in kind]
fig, ax = plt.subplots(figsize=(9.0, 4.7))
y = np.arange(len(labels))[::-1]
bars = ax.barh(y, vals, color=colors, height=0.62, zorder=3,
               hatch=[hatchmap[k] for k in kind], edgecolor="white", linewidth=0.6)
ax.axvline(1.0, color=CAPLINE, ls="--", lw=1.4, zorder=4)
ax.text(1.0, len(labels)-0.35, "  cap (Λ = J_limit)", color=CAPLINE, fontsize=10, va="center", fontweight="bold")
for yi, v in zip(y, vals):
    ax.text(v + 0.02, yi, f"{v:.2f}", va="center", ha="left", fontsize=10, color=INK, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlim(0, 1.85); ax.set_xlabel("worst-joint  Λ / cap")
ax.set_title("Reachable impact impulse never reaches the cap; only artifacts cross",
             fontsize=11.5, fontweight="bold", pad=12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=BLUE, label="reachable by a real policy"),
                   Patch(fc=MUTE, hatch="///", label="kinematic ceiling (unreachable)"),
                   Patch(fc=VERM, hatch="xx", label="measurement artifact (press, >URDF)")],
          loc="upper right", frameon=True, framealpha=0.95, edgecolor="#e0e2e6", fontsize=8.5)
clean(ax, "x")
fig.tight_layout(); fig.savefig(OUT / "fig1_ceiling.png", dpi=150); plt.close(fig)

# ---------- FIG 2: speed sweep — two panels (no dual axis) ----------
v = np.array([0.44, 1.18, 1.28, 1.33, 1.38])
lam = np.array([0.12, 0.15, 0.27, 0.11, 0.11])
pkmn = np.array([1.73, 3.25, 1.90, 2.31, 1.57])
o = np.argsort(v); v, lam, pkmn = v[o], lam[o], pkmn[o]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.8, 6.0), sharex=True)
a1.plot(v, lam, "-o", color=BLUE, lw=2, ms=8, zorder=3, markeredgecolor="white", markeredgewidth=1.2)
a1.axhline(1.0, color=CAPLINE, ls="--", lw=1.4); a1.text(0.46, 1.02, "cap", color=CAPLINE, fontsize=10, fontweight="bold")
a1.set_ylim(0, 1.15); a1.set_ylabel("worst-joint Λ / cap")
a1.set_title("Even the hardest strike in this sweep stays ≤0.27 of the cap", fontsize=12, fontweight="bold", pad=8)
for xi, yi in zip(v, lam): a1.text(xi, yi+0.05, f"{yi:.2f}", ha="center", fontsize=9, color=INK2)
clean(a1)
a2.plot(v, pkmn, "-s", color=GREEN, lw=2, ms=8, zorder=3, markeredgecolor="white", markeredgewidth=1.2)
a2.axhline(1.0, color=MUTE, ls=":", lw=1.2); a2.text(0.46, 1.06, "flat (press)", color=MUTE, fontsize=9)
a2.set_ylabel("peak / mean contact force"); a2.set_xlabel("approach velocity at contact  (m/s)")
a2.set_title("Force profile: flat press when gentle → spike when firm (never ballistic)", fontsize=11, color=INK2, pad=8)
for xi, yi in zip(v, pkmn): a2.text(xi, yi+0.12, f"{yi:.1f}", ha="center", fontsize=9, color=INK2)
clean(a2)
fig.tight_layout(); fig.savefig(OUT / "fig2_speed.png", dpi=150); plt.close(fig)

# ---------- FIG 3: nail-mass bindable sweep ----------
mass = np.array([1, 10, 50, 100, 300, 1000])
lam3 = np.array([0.11, 0.23, 0.35, 0.34, 0.49, 1.09])
fig, ax = plt.subplots(figsize=(8.6, 4.5))
ax.plot(mass, lam3, "-o", color=BLUE, lw=2, ms=8, zorder=3, markeredgecolor="white", markeredgewidth=1.2)
ax.plot(mass[-1], lam3[-1], "o", color=VERM, ms=11, zorder=4, markeredgecolor="white", markeredgewidth=1.4)
ax.axhline(1.0, color=CAPLINE, ls="--", lw=1.4); ax.text(1.1, 1.03, "cap", color=CAPLINE, fontsize=10, fontweight="bold")
ax.set_xscale("log"); ax.set_xlabel("nail mass  (× the real 7 g nail)"); ax.set_ylabel("worst-joint Λ / cap")
ax.set_ylim(0, 1.2)
ax.set_title("An absurd 7 kg nail crosses the cap — but via a press, not an impact",
             fontsize=11, fontweight="bold", pad=10)
ax.annotate("press artifact\n(7 kg nail; removed by\nthe Phase-2 press-gap fix)",
            xy=(1000, 1.09), xytext=(90, 0.62), fontsize=9, color=VERM, ha="center",
            arrowprops=dict(arrowstyle="->", color=VERM, lw=1.3))
ax.annotate("realistic nail masses:\nΛ stays well under cap", xy=(50, 0.35), xytext=(3, 0.75),
            fontsize=9, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
clean(ax); fig.tight_layout(); fig.savefig(OUT / "fig3_nailmass.png", dpi=150); plt.close(fig)

# ---------- FIG 4: enforcement plumbing — δ vs Λ/cap ----------
fig, ax = plt.subplots(figsize=(7.6, 4.6))
# enforcement ON: compliant (0.04,0) -> over-cap (1.29,0.5). δ is 0 for ALL Λ<=cap and only rises
# past the cap, so the connector is a flat-then-step HINGE at Λ/cap=1.0, not a ramp (a straight
# interpolating line would wrongly imply δ grows gradually with Λ across the whole range).
hinge_x = [0.04, 1.0, 1.0, 1.29]
hinge_y = [0.0, 0.0, 0.5, 0.5]
ax.plot(hinge_x, hinge_y, color=BLUE, lw=1.6, ls=(0, (5, 3)), zorder=3)
ax.scatter([0.04], [0.0], s=120, color=BLUE, zorder=4, edgecolor="white", linewidth=1.4, label="enforcement ON (imp_max_p=0.5)")
ax.scatter([1.29], [0.5], s=140, color=BLUE, zorder=4, edgecolor="white", linewidth=1.4)
# log-only control: (0.04,0),(1.29,0)
ax.scatter([0.04, 1.29], [0.0, 0.0], s=110, color=VERM, marker="s", zorder=3, edgecolor="white", linewidth=1.2, label="log-only control (imp_max_p=0)")
ax.plot([0.04, 1.29], [0.0, 0.0], color=VERM, lw=1.4, ls=":")
ax.axvline(1.0, color=CAPLINE, ls="--", lw=1.4); ax.text(1.01, 0.44, "cap", color=CAPLINE, fontsize=10, fontweight="bold")
ax.axhline(0.5, color=MUTE, ls=":", lw=1); ax.text(1.12, 0.512, "max_p = 0.5", color=MUTE, fontsize=9)
ax.annotate("δ = 0\n(compliant)", xy=(0.04, 0.0), xytext=(0.12, 0.14), fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.annotate("δ fires to max_p\nwhen Λ crosses cap", xy=(1.29, 0.5), xytext=(0.55, 0.42), fontsize=9.5, color=BLUE,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
ax.annotate("δ stays 0 despite Λ>cap\n(flag gates enforcement)", xy=(1.29, 0.0), xytext=(0.62, 0.10), fontsize=9, color=VERM,
            arrowprops=dict(arrowstyle="->", color=VERM, lw=1.2))
ax.set_xlim(0, 1.55); ax.set_ylim(-0.04, 0.58)
ax.set_xlabel("worst-joint Λ / cap"); ax.set_ylabel("termination probability  δ")
ax.set_title("The constraint enforces: δ fires when Λ crosses the cap", fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper left", frameon=False, fontsize=9)
clean(ax); fig.tight_layout(); fig.savefig(OUT / "fig4_plumbing.png", dpi=150); plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
