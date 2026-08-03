"""18-policy trajectory grids (x-z, x-y) and the per-arm summary figure.

Grid layout is 3 rows (arms) x 6 columns (seeds 2,3 | 4,5,6,7) on ONE shared
square scale, so straightness is comparable across every panel. The Wave-1/Wave-2
boundary is drawn, and every panel is labelled with its preregistered outcome --
including the policies that fail, which are never omitted.
"""

import csv
import hashlib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

WAVE1_ROOT = Path("evaluation/results/2026-08-02_wave1_waypoint")
WAVE2_ROOT = Path("evaluation/results/2026-08-03_wave2_waypoint")
ARMS = ("C0", "G", "P")
SEEDS = (2, 3, 4, 5, 6, 7)
SHORT = {"C0": "c0", "G": "g", "P": "p"}
GATE_R = 0.015
PERP_MM = 15.0
RATIO = 1.15


def campaign_of(seed):
    return "wave1" if seed in (2, 3) else "wave2"


def leaf(arm, seed):
    root = WAVE1_ROOT if campaign_of(seed) == "wave1" else WAVE2_ROOT
    return root / "videos" / f"{campaign_of(seed)}_{SHORT[arm]}_seed{seed}"


rows = {}
with (WAVE2_ROOT / "tables" / "wave1_wave2_eighteen_policy_comparison.csv").open() as fh:
    for row in csv.DictReader(fh):
        rows[(row["arm"], int(row["training_seed"]))] = row

traces = {(a, s): dict(np.load(leaf(a, s) / "trace.npz")) for a in ARMS for s in SEEDS}

stacked = np.vstack(
    [t["substep_head_position_m"] for t in traces.values()]
    + [t["guideline_entry_m"][None, :] for t in traces.values()]
    + [t["guideline_nail_m"][None, :] for t in traces.values()]
)
half = float(np.max(np.ptp(stacked, axis=0))) * 0.6
centre = {a: float(0.5 * (stacked[:, a].max() + stacked[:, a].min())) for a in (0, 1, 2)}


def label(arm, seed):
    r = rows[(arm, seed)]
    gates = int(r["gates_at_contact_onset"])
    perp = float(r["precontact_segment_perp_max_mm"])
    ratio = float(r["precontact_path_ratio"])
    straight = gates >= 6 and perp <= PERP_MM and ratio <= RATIO
    mark = "STRAIGHT" if straight else ("gates but wide" if gates >= 6 else "no gates")
    return f"{arm} seed {seed} [{campaign_of(seed)}]\n{gates}/6 gates  {perp:.1f} mm  r={ratio:.2f}  {mark}", straight


def grid(ordinate, axis_label, path, title):
    fig, axes = plt.subplots(3, 6, figsize=(26, 14))
    for i, arm in enumerate(ARMS):
        for j, seed in enumerate(SEEDS):
            ax = axes[i][j]
            t = traces[(arm, seed)]
            pos = t["substep_head_position_m"]
            contact = t["substep_contact"].astype(bool)
            boundary = t["substep_is_control_boundary"].astype(bool)
            entry, nail = t["guideline_entry_m"], t["guideline_nail_m"]
            ax.plot([entry[0], nail[0]], [entry[ordinate], nail[ordinate]],
                    "k--", lw=1.0, zorder=1)
            for c in t["guideline_gate_centers_m"]:
                ax.add_patch(plt.Circle((c[0], c[ordinate]), GATE_R, facecolor="none",
                                        edgecolor="#1f77b4", lw=0.8, alpha=0.85, zorder=2))
            if len(pos) > 1:
                pts = pos[:, [0, ordinate]].reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                ax.add_collection(LineCollection(segs, cmap="viridis",
                                                 array=np.linspace(0, 1, len(segs)),
                                                 linewidth=1.3, zorder=3))
            ax.scatter(pos[boundary, 0], pos[boundary, ordinate], facecolors="none",
                       edgecolors="#444", s=16, lw=0.5, zorder=4)
            ax.scatter(pos[0, 0], pos[0, ordinate], c="#2ca02c", s=32, zorder=6)
            if contact.any():
                ax.scatter(pos[contact, 0], pos[contact, ordinate], c="#d62728",
                           s=12, zorder=7)
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[ordinate] - half, centre[ordinate] + half)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            text, straight = label(arm, seed)
            ax.set_title(text, fontsize=8,
                         color="#0a7d18" if straight else "#444")
            for side in ax.spines.values():
                side.set_edgecolor("#0a7d18" if straight else "#bbbbbb")
                side.set_linewidth(1.8 if straight else 0.8)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f"{arm}\n{axis_label}", fontsize=9)
            if i == 2:
                ax.set_xlabel("x (m)", fontsize=7)
            # Wave-1 | Wave-2 boundary
            if j == 2:
                ax.annotate("", xy=(-0.06, 0), xycoords="axes fraction",
                            xytext=(-0.06, 1), textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="-", lw=2.0, color="#c05000"))
    fig.suptitle(
        title + "\nCPU fixed-reset · seeds 2,3 = Wave 1 | 4-7 = Wave 2 (orange rule)"
        " · dashed = tracker entry->nail · blue = 15 mm gate disks · viridis = time"
        " · red = contact · green border = meets all three predeclared straight clauses",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    # two-line panel titles need real room, or each row's title lands on the
    # x-axis of the row above and the numbers become unreadable
    fig.subplots_adjust(hspace=0.42)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    colors = {"C0": "#7f7f7f", "G": "#1f77b4", "P": "#2ca02c"}

    ax = axes[0]
    for k, arm in enumerate(ARMS):
        n = sum(1 for s in SEEDS if int(rows[(arm, s)]["gates_at_contact_onset"]) >= 6)
        straight = sum(
            1 for s in SEEDS
            if int(rows[(arm, s)]["gates_at_contact_onset"]) >= 6
            and float(rows[(arm, s)]["precontact_segment_perp_max_mm"]) <= PERP_MM
            and float(rows[(arm, s)]["precontact_path_ratio"]) <= RATIO)
        ax.bar(k - 0.18, n, 0.36, color=colors[arm], label="6 gates" if k == 0 else None)
        ax.bar(k + 0.18, straight, 0.36, color=colors[arm], alpha=0.45, hatch="//",
               label="straight (all 3)" if k == 0 else None)
        ax.text(k - 0.18, n + 0.12, f"{n}/6", ha="center", fontsize=10)
        ax.text(k + 0.18, straight + 0.12, f"{straight}/6", ha="center", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(ARMS); ax.set_ylim(0, 6.9)
    ax.set_ylabel("policies out of 6 seeds")
    ax.set_title("Primary endpoint and predeclared straight label", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for k, arm in enumerate(ARMS):
        vals = [float(rows[(arm, s)]["precontact_segment_perp_max_mm"]) for s in SEEDS]
        ax.scatter([k] * 6, vals, c=colors[arm], s=54, zorder=3)
        for s, v in zip(SEEDS, vals):
            ax.annotate(str(s), (k, v), fontsize=7, xytext=(7, -3),
                        textcoords="offset points")
    ax.axhline(PERP_MM, color="#c05000", ls="--", lw=1.3)
    # left of the C0 column: the right side is where the boundary-straddling
    # G/4, P/4 and P/5 points sit, and the label must not cover them
    ax.text(-0.42, PERP_MM * 1.08, "predeclared 15 mm", color="#c05000", fontsize=8,
            ha="left")
    ax.set_xlim(-0.5, 2.6)
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(ARMS)
    ax.set_ylabel("pre-contact max segment error (mm, log)")
    ax.set_title("Approach straightness, all 18 (labels = seed)", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[2]
    for k, arm in enumerate(ARMS):
        vals = [float(rows[(arm, s)]["traj_peak_qvel_rad_s"]) for s in SEEDS]
        ax.scatter([k] * 6, vals, c=colors[arm], s=54, zorder=3)
    ax.axhline(3.1415, color="#d62728", ls="--", lw=1.4)
    ax.text(2.42, 3.22, "hardware limit 3.1415 rad/s", color="#d62728", fontsize=8,
            ha="right")
    ax.set_ylim(0, 5.8); ax.set_xticks(range(3)); ax.set_xticklabels(ARMS)
    ax.set_ylabel("peak |qvel| (rad/s)")
    ax.set_title("Velocity legality: 0/18 legal, measured not enforced", fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


figs = WAVE2_ROOT / "figures"
figs.mkdir(parents=True, exist_ok=True)
h1 = grid(2, "z (m)", figs / "wave2_eighteen_panel_xz.png",
          "18-policy hammer-head trajectories (x-z side view)")
h2 = grid(1, "y (m)", figs / "wave2_eighteen_panel_xy.png",
          "18-policy hammer-head trajectories (x-y top view)")
h3 = summary(figs / "wave2_per_arm_summary.png")
print("x-z grid   sha256:", h1[:16])
print("x-y grid   sha256:", h2[:16])
print("summary    sha256:", h3[:16])
print("shared half-span (m):", round(half, 4))
