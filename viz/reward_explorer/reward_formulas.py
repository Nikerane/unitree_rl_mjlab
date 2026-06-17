"""Pure-numpy re-implementation of the Z1 hammer reward-term math (for the Reward Explorer).

NO mjlab / torch dependency -- so the Gradio Space stays light. These formulas mirror
`src/tasks/hammer/mdp/rewards.py`; `tests/test_reward_viz_parity.py` pins both the formulas
and the default params below to the live env config, so the viz cannot silently drift.

Each function returns the UNWEIGHTED term value; the app multiplies by the weight.
"""

from __future__ import annotations

import numpy as np

# --- Live defaults (mirror hammer_env_cfg.py + nail_block.py; pinned by the parity test) ---
GOAL_DEPTH = 0.032         # nail_driven Gaussian centre (NAIL_GOAL_DEPTH)
SUCCESS_THRESHOLD = 0.027  # completion threshold (NAIL_SUCCESS_THRESHOLD)
NAIL_DRIVEN_STD = 0.013
APPROACH_STD = 0.08
R_IMIT_SIGMA = 0.05
IMPACT_V_EXPECTED = 1.0
SETTLE_OFFSET = 0.004      # NailDepthDeltaTerm._SETTLE_OFFSET (4 mm dead zone)

# Live reward weights (a_track / imitation arm). The parity test asserts these match.
WEIGHTS = {
    "approach": 0.1,
    "nail_driven": 2.0,
    "nail_depth_delta": 600.0,
    "impact_progress": 8.0,
    "completion": 100.0,
    "action_rate": -0.01,
    "joint_pos_limits": -10.0,
    "r_imit": 0.1,
}


def gaussian_distance(dist, std):
    """exp(-dist^2 / std^2) -- shared by approach (head->nail) and r_imit (head->ref)."""
    return np.exp(-(np.asarray(dist, float) ** 2) / std**2)


def nail_driven(depth, goal=GOAL_DEPTH, std=NAIL_DRIVEN_STD):
    """Gaussian on driven depth: exp(-(goal - depth)^2 / std^2)."""
    return np.exp(-((goal - np.asarray(depth, float)) ** 2) / std**2)


def approach(dist, std=APPROACH_STD):
    return gaussian_distance(dist, std)


def r_imit(dist, sigma=R_IMIT_SIGMA, pre_contact=True):
    """Ante-impact tracking prior: exp(-dist^2/sigma^2), zeroed once contact has occurred."""
    return gaussian_distance(dist, sigma) * np.asarray(pre_contact, float)


def completion(depth, threshold=SUCCESS_THRESHOLD):
    """Sparse +1 once depth >= threshold."""
    return (np.asarray(depth, float) >= threshold).astype(float)


def impact_progress(v_axial, first_contact, advanced, v_expected=IMPACT_V_EXPECTED):
    """Double-gated momentum: max(0, v_axial)/v_expected * 1[first_contact] * 1[advanced]."""
    v = np.clip(np.asarray(v_axial, float), 0.0, None)
    return (v / v_expected) * np.asarray(first_contact, float) * np.asarray(advanced, float)


def nail_depth_delta(depth_series, settle=SETTLE_OFFSET):
    """Ratchet on max-depth-so-far: per-step max(0, depth - running_max); running_max starts at `settle`.

    Stateful over an episode -> takes a per-step depth series, returns the per-step term value.
    """
    depth_series = np.asarray(depth_series, float)
    out = np.zeros_like(depth_series)
    running = float(settle)
    for i, d in enumerate(depth_series):
        out[i] = max(0.0, float(d) - running)
        running = max(running, float(d))
    return out


def action_rate(delta_action):
    """L2 penalty on the per-step action change: ||a_t - a_{t-1}||^2 (returns the magnitude; weight is negative)."""
    return np.sum(np.asarray(delta_action, float) ** 2, axis=-1)
