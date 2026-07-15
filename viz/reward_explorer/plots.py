"""Matplotlib figure builders for the Reward Explorer (no Gradio dependency -> headless-testable).

Term-explorer curves + the strike-overlay view. Pure numpy/matplotlib on top of reward_formulas.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (Space / tests)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from reward_formulas import (  # noqa: E402
    WEIGHTS,
    approach,
    completion,
    impact_progress,
    nail_depth_delta,
    nail_driven,
    r_imit,
)

_HERE = Path(__file__).parent

# LaTeX (KaTeX) for each term -- shown in the app and the README.
LATEX = {
    "approach": r"r_{\text{app}} = \exp\!\left(-\lVert p_{\text{head}}-p_{\text{nail}}\rVert^2/\sigma^2\right),\ \sigma=0.08",
    "nail_driven": r"r_{\text{nd}} = \exp\!\left(-(d_{\text{goal}}-d)^2/\sigma^2\right),\ d_{\text{goal}}=0.032,\ \sigma=0.013",
    "nail_depth_delta": r"r_{\Delta} = \max\!\left(0,\ d - \max_{t'\le t} d_{t'}\right)",
    "impact_progress": r"r_{\text{ip}} = \frac{\max(0,\,v_{\text{axial}})}{v_{\text{exp}}}\cdot \mathbb{1}[\text{first contact}]\cdot \mathbb{1}[\Delta d>\epsilon]",
    "completion": r"r_{\text{c}} = \mathbb{1}[\,d \ge d_{\text{succ}}\,],\ d_{\text{succ}}=0.030",
    "r_imit": r"r_{\text{imit}} = \exp\!\left(-\lVert p_{\text{head}}-p^{*}(\phi)\rVert^2/\sigma^2\right)\cdot \mathbb{1}[\text{pre-contact}],\ \sigma=0.05",
    "action_rate": r"r_{\text{ar}} = \lVert a_t - a_{t-1}\rVert^2\quad(\text{weight}<0)",
    "joint_pos_limits": r"r_{\text{jl}} = \sum_j \big[\max(0,\,q_j-q_j^{\max}) + \max(0,\,q_j^{\min}-q_j)\big]\quad(\text{weight}<0)",
}

# Plain-language definition of each term (shown on the "All terms" reference page).
DEFINITIONS = {
    "approach": "Gaussian that pulls the hammer head toward the nail top. Low weight so hovering near the nail is never a stable optimum. Always on.",
    "nail_driven": "Gaussian centred on the full-drive depth (32 mm): a dense pull once the nail starts moving; near-zero at 0 mm.",
    "nail_depth_delta": "Progress reward — pays only NEW maximum depth (ratchets on max-so-far), so re-driving already-covered ground earns nothing. 4 mm dead zone at the start.",
    "impact_progress": "Downward impact speed at contact, paid once per fresh contact that actually advances the nail (double-gated). The controllable 'hit hard' lever on a position-only action space.",
    "completion": "Sparse +1 the step the nail crosses the 27 mm success threshold; the episode then terminates.",
    "r_imit": "Weak ante-impact tracking prior (A-TRACK arm only): rewards following the scripted strike reference BEFORE contact, then gates off. Annealed to 0 by iter ~250 so the policy ends up free to deviate.",
    "action_rate": "Smoothness penalty on the change in action between consecutive steps (negative weight).",
    "joint_pos_limits": "Penalty for driving joints past their soft position limits (negative weight); ~0 in normal operation, a guard against limit-slamming.",
}

# Per-term tunable scalar slider: (param_label, min, max, default). None = no param.
PARAM = {
    "nail_driven": ("std", 0.005, 0.05, 0.013),
    "approach": ("std", 0.02, 0.20, 0.08),
    "r_imit": ("sigma", 0.01, 0.15, 0.05),
    "completion": ("threshold", 0.015, 0.032, 0.030),
    "impact_progress": ("v_expected", 0.3, 2.0, 1.0),
    "nail_depth_delta": ("settle", 0.0, 0.01, 0.004),
    "action_rate": (None, 0.0, 0.0, 0.0),
}
# Terms with a standalone curve in the explorer (joint_pos_limits is reference-only).
TERMS = ["nail_driven", "approach", "r_imit", "completion", "impact_progress", "nail_depth_delta", "action_rate"]


def overview_markdown() -> str:
    """A single reference page: every reward term, its equation, definition, and live weight."""
    lines = [
        "## All reward terms",
        "The Z1 hammer reward — **A-TRACK** arm (the A-BASE 7-term reward + the annealed `r_imit` "
        "prior). Weights are the live training values; equations are pinned to the code by the parity test.",
    ]
    for name in LATEX:
        lines.append(f"### `{name}` · weight **{WEIGHTS.get(name, 0):+g}**")
        lines.append(f"$$ {LATEX[name]} $$")
        lines.append(DEFINITIONS[name])
    return "\n\n".join(lines)


def term_curve_figure(term: str, param_value: float):
    """Plot one term's UNWEIGHTED value over its natural input domain."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if term == "nail_driven":
        x = np.linspace(0.0, 0.035, 300)
        ax.plot(x * 1000, nail_driven(x, std=param_value))
        ax.axvline(27, ls="--", c="gray", lw=1, label="success 27 mm")
        ax.set_xlabel("nail depth (mm)"); ax.legend()
    elif term in ("approach", "r_imit"):
        x = np.linspace(0.0, 0.30, 300)
        y = approach(x, std=param_value) if term == "approach" else r_imit(x, sigma=param_value)
        ax.plot(x * 1000, y); ax.set_xlabel("distance head→target (mm)")
    elif term == "completion":
        x = np.linspace(0.0, 0.035, 300)
        ax.plot(x * 1000, completion(x, threshold=param_value)); ax.set_xlabel("nail depth (mm)")
    elif term == "impact_progress":
        x = np.linspace(0.0, 1.5, 300)
        ax.plot(x, impact_progress(x, 1.0, 1.0, v_expected=param_value))
        ax.set_xlabel("axial speed at contact (m/s)  [both gates on]")
    elif term == "nail_depth_delta":
        x = np.linspace(0.0, 0.032, 80)  # a monotonic depth ramp
        ax.plot(x * 1000, nail_depth_delta(x, settle=param_value))
        ax.set_xlabel("nail depth along a monotonic drive (mm)")
    elif term == "action_rate":
        x = np.linspace(0.0, 2.0, 300)
        ax.plot(x, x**2); ax.set_xlabel("‖aₜ − aₜ₋₁‖")
    ax.set_ylabel("unweighted reward"); ax.set_title(term); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def load_trace(path: str | None = None) -> dict:
    return json.loads((Path(path) if path else _HERE / "rollout_trace.json").read_text())


def strike_overlay_figure(weight_overrides: dict, trace: dict | None = None):
    """Per-step WEIGHTED contribution of the depth/contact-driven terms over a real strike."""
    tr = trace or load_trace()
    steps = tr["steps"]
    k = np.array([s["k"] for s in steps])
    depth = np.array([s["nail_mm"] for s in steps]) / 1000.0
    v_ax = np.array([s.get("head_axial_speed", 0.0) for s in steps])
    fc = np.array([s.get("first_contact", False) for s in steps], float)
    adv = np.concatenate([[0.0], (np.diff(depth) > 5e-4).astype(float)])  # advanced past max
    w = {**WEIGHTS, **(weight_overrides or {})}

    contrib = {
        "nail_driven": w["nail_driven"] * nail_driven(depth),
        "nail_depth_delta": w["nail_depth_delta"] * nail_depth_delta(depth),
        "impact_progress": w["impact_progress"] * impact_progress(v_ax, fc, adv),
        "completion": w["completion"] * completion(depth),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, y in contrib.items():
        ax.plot(k, y, marker="o", ms=3, label=name)
    ax.set_xlabel("control step"); ax.set_ylabel("weighted reward contribution")
    ax.set_title("Per-term contribution over a real strike (a_track)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(k, depth * 1000, c="k", ls=":", lw=1.2, label="nail depth (mm)")
    ax2.axhline(27, c="gray", ls="--", lw=0.8)
    ax2.set_ylabel("nail depth (mm)")
    fig.tight_layout()
    return fig
