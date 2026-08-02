"""Pure, episode-first analysis for the fixed-reset Cartesian guideline pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


FAILURE_ERROR_M = 0.050
CORRIDOR_RADIUS_M = 0.005
NUM_GATES = 6
EXPECTED_EPISODE_COUNT = 512
PILOT_IDENTITIES = (("C0", 0), ("C0", 1), ("C-Gate", 0), ("C-Gate", 1))
_PILOT_SHARED_FIELDS = (
    "reset_digest",
    "guideline_geometry_digest",
    "reset_rng_seed",
    "observation_rng_seed",
    "action_rng_seed",
    "git_revision",
    "asset_git_revision",
)


def _array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    require_finite: bool = True,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if require_finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _aligned_vector(value: Any, *, name: str, count: int) -> np.ndarray:
    array = _array(value, name=name)
    if array.shape != (count,):
        raise ValueError(f"{name} must have one value per physical substep")
    return array


def _first_gate_one_index(next_gate: np.ndarray) -> int | None:
    previous = np.concatenate((np.array([0.0]), next_gate[:-1]))
    advances = np.flatnonzero((previous == 0.0) & (next_gate > 0.0))
    return int(advances[0]) if advances.size else None


def summarize_guideline_episode(trace: Mapping) -> dict:
    """Reduce one stored guideline trace without conditioning on success.

    The primary value is one q90 per episode, deliberately preventing long
    trajectories from receiving more influence than short trajectories.
    """

    if not isinstance(trace, Mapping):
        raise ValueError("episode trace must be a mapping")
    physical = trace.get("physical")
    guideline = trace.get("guideline")
    first_strike = trace.get("first_strike")
    if not all(isinstance(value, Mapping) for value in (physical, guideline, first_strike)):
        raise ValueError("trace requires physical, guideline, and first_strike mappings")

    contact = _array(physical.get("contact"), name="physical.contact")
    if contact.ndim != 1 or contact.size == 0:
        raise ValueError("physical.contact must be a non-empty one-dimensional stream")
    count = int(contact.size)
    head = _array(physical.get("head_position_m"), name="physical.head_position_m")
    if head.shape != (count, 3):
        raise ValueError("physical.head_position_m must align as [substep, xyz]")
    qvel_pre = _array(
        physical.get("joint_speed_rad_s"),
        name="physical.joint_speed_rad_s",
        require_finite=False,
    )
    qvel_post = _array(
        physical.get("post_step_joint_speed_rad_s"),
        name="physical.post_step_joint_speed_rad_s",
        require_finite=False,
    )
    if qvel_pre.shape != (count, 6) or qvel_post.shape != (count, 6):
        raise ValueError("physical qvel streams must align as [substep, six joints]")

    event_trace = trace.get("event_trace", {})
    if not isinstance(event_trace, Mapping):
        raise ValueError("event_trace must be a mapping")
    if "tracker_started" in event_trace:
        _aligned_vector(
            event_trace["tracker_started"],
            name="event_trace.tracker_started",
            count=count,
        )

    entry = _array(guideline.get("entry_m"), name="guideline.entry_m", shape=(3,))
    nail = _array(guideline.get("nail_m"), name="guideline.nail_m", shape=(3,))
    axis = nail - entry
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("guideline entry and nail must define a non-zero segment")
    next_gate = _aligned_vector(
        guideline.get("next_gate"), name="guideline.next_gate", count=count
    )
    if not np.equal(next_gate, np.floor(next_gate)).all() or np.any(
        (next_gate < 0) | (next_gate > NUM_GATES)
    ):
        raise ValueError("guideline.next_gate must contain gate indices 0 through 6")
    errors = _aligned_vector(
        guideline.get("perpendicular_error_m"),
        name="guideline.perpendicular_error_m",
        count=count,
    )
    if np.any(errors < 0.0):
        raise ValueError("guideline.perpendicular_error_m must be non-negative")
    disarmed = _aligned_vector(
        guideline.get("disarmed"), name="guideline.disarmed", count=count
    )
    if not np.isin(disarmed, (0.0, 1.0)).all():
        raise ValueError("guideline.disarmed must be boolean-valued")
    payouts = _array(guideline.get("gate_payout"), name="guideline.gate_payout")
    if payouts.ndim != 1 or np.any(payouts < 0.0):
        raise ValueError("guideline.gate_payout must be a non-negative control-rate stream")
    gate_reward_present = guideline.get("gate_reward_present")
    if not isinstance(gate_reward_present, bool):
        raise ValueError("guideline.gate_reward_present must be a bool")
    if not gate_reward_present and np.any(payouts != 0.0):
        raise ValueError("C0-style trace cannot contain a gate payout")

    accepted_raw = first_strike.get("accepted_onset_index")
    if accepted_raw is None:
        accepted_contact = None
    else:
        if isinstance(accepted_raw, bool):
            raise ValueError("first_strike.accepted_onset_index must be an index")
        try:
            accepted_contact = int(accepted_raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("first_strike.accepted_onset_index must be an index") from error
        if accepted_contact < 0 or accepted_contact >= count:
            raise ValueError("first_strike.accepted_onset_index is out of range")

    gate_one = _first_gate_one_index(next_gate)
    if gate_one is None or (accepted_contact is not None and accepted_contact < gate_one):
        endpoint = FAILURE_ERROR_M
        window_start = None
        window_end = None
        window_errors = np.empty(0, dtype=np.float64)
    else:
        window_start = gate_one
        window_end = count - 1 if accepted_contact is None else accepted_contact
        window_errors = np.minimum(errors[window_start : window_end + 1], FAILURE_ERROR_M)
        endpoint = float(np.quantile(window_errors, 0.90))

    if window_start is None:
        corridor_occupancy = 0.0
        backward_progress = 0
    else:
        corridor_occupancy = float(np.mean(window_errors <= CORRIDOR_RADIUS_M))
        longitudinal = (head[window_start : window_end + 1] - entry) @ (axis / axis_norm)
        backward_progress = int(np.count_nonzero(np.diff(longitudinal) < 0.0))

    peak_lambda = _array(trace.get("episode_peak_lambda"), name="episode_peak_lambda")
    caps = _array(trace.get("impulse_limits_n_m_s"), name="impulse_limits_n_m_s")
    if peak_lambda.shape != (6,) or caps.shape != (6,) or np.any(peak_lambda < 0.0) or np.any(caps <= 0.0):
        raise ValueError("terminal Lambda and impulse caps must be six non-negative/positive values")
    speed = float(first_strike.get("v_precontact_m_s", 0.0))
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("first_strike.v_precontact_m_s must be finite and non-negative")
    depth = float(trace.get("episode_depth_m", 0.0))
    if not math.isfinite(depth):
        raise ValueError("episode_depth_m must be finite")

    qvel = np.concatenate((qvel_pre, qvel_post))
    finite_qvel = np.isfinite(qvel)
    qvel_peak = float(np.max(np.abs(qvel[finite_qvel]))) if np.any(finite_qvel) else 0.0
    return {
        "q90_terminal_descent_perpendicular_error_m": endpoint,
        "all_six_gates": bool(next_gate[-1] >= NUM_GATES),
        "corridor_occupancy": corridor_occupancy,
        "backward_progress_count": backward_progress,
        "actual_gate_return": float(np.sum(payouts)),
        "gate_reward_present": gate_reward_present,
        "success": bool(trace.get("overall_success", False)),
        "useful_precontact_speed_m_s": speed,
        "episode_depth_m": depth,
        "lambda_cap_ratios": (peak_lambda / caps).tolist(),
        "qvel_finite": bool(finite_qvel.all()),
        "qvel_peak_abs_rad_s": qvel_peak,
    }


def _episode_row(episode: Mapping) -> dict:
    if "q90_terminal_descent_perpendicular_error_m" in episode:
        return dict(episode)
    return summarize_guideline_episode(episode)


def _finite_mean(rows: Sequence[Mapping], field: str) -> float:
    values = _array([row[field] for row in rows], name=field)
    return float(np.mean(values))


def aggregate_guideline_seed(
    episodes: Sequence[Mapping], *, expected_count: int = EXPECTED_EPISODE_COUNT
) -> dict:
    """Aggregate one already-fixed episode population, never pooled substeps."""

    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    if len(episodes) != expected_count:
        raise ValueError(f"expected exactly {expected_count} episodes, found {len(episodes)}")
    rows = [_episode_row(episode) for episode in episodes]
    endpoints = _array(
        [row["q90_terminal_descent_perpendicular_error_m"] for row in rows],
        name="episode endpoints",
    )
    ratios = _array([row["lambda_cap_ratios"] for row in rows], name="lambda_cap_ratios")
    if ratios.shape != (len(rows), 6):
        raise ValueError("each episode must provide six Lambda/cap ratios")
    return {
        "n_episodes_sampled": len(rows),
        "q90_terminal_descent_perpendicular_error_m_sampled": float(np.quantile(endpoints, 0.90)),
        "all_six_gates_rate_sampled": _finite_mean(rows, "all_six_gates"),
        "corridor_occupancy_mean_sampled": _finite_mean(rows, "corridor_occupancy"),
        "backward_progress_count_mean_sampled": _finite_mean(rows, "backward_progress_count"),
        "actual_gate_return_total_sampled": float(sum(float(row["actual_gate_return"]) for row in rows)),
        "success_rate_sampled": _finite_mean(rows, "success"),
        "useful_precontact_speed_mean_sampled": _finite_mean(rows, "useful_precontact_speed_m_s"),
        "episode_depth_mean_m_sampled": _finite_mean(rows, "episode_depth_m"),
        "lambda_cap_ratio_max_sampled": float(np.max(ratios)),
        "qvel_finite": bool(all(bool(row["qvel_finite"]) for row in rows)),
        "qvel_peak_abs_rad_s_sampled": float(max(float(row["qvel_peak_abs_rad_s"]) for row in rows)),
    }


def validate_guideline_pilot_rows(rows: Sequence[Mapping]) -> dict:
    """Fail closed on the exact, excluded four-row C0/C-Gate pilot contract."""

    rows = [dict(row) for row in rows]
    if any(
        isinstance(row.get("training_seed"), bool)
        or not isinstance(row.get("training_seed"), int)
        for row in rows
    ):
        raise ValueError("pilot training seeds must be literal integers")
    identities = [(row.get("treatment"), row.get("training_seed")) for row in rows]
    if identities != list(PILOT_IDENTITIES):
        raise ValueError("pilot rows must be exactly C0/C-Gate seeds 0/1 in frozen order")
    for row in rows:
        identity = f"{row['treatment']}/seed{row['training_seed']}"
        if row.get("checkpoint_filename") != "model_499.pt":
            raise ValueError(f"{identity}: final model_499.pt checkpoint is required")
        if row.get("n_episodes_sampled") != EXPECTED_EPISODE_COUNT:
            raise ValueError(f"{identity}: exactly 512 sampled episodes are required")
        if bool(row.get("git_dirty")) or bool(row.get("asset_git_dirty")):
            raise ValueError(f"{identity}: clean code and asset provenance are required")
        for sentinel in ("impossible_success_n", "lambda_dead_n"):
            if row.get(sentinel) != 0:
                raise ValueError(f"{identity}: {sentinel} must be zero")
        try:
            nonfinite_qvel = float(row["qvel_nonfinite_rate_sampled"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{identity}: qvel nonfinite rate is required") from error
        if not math.isfinite(nonfinite_qvel) or nonfinite_qvel != 0.0:
            raise ValueError(f"{identity}: nonfinite qvel invalidates the pilot row")
        try:
            payout = float(row["actual_gate_return_total_sampled"])
            success_rate = float(row["success_rate_sampled"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{identity}: gate payout and success rate are required") from error
        if not math.isfinite(payout) or not math.isfinite(success_rate):
            raise ValueError(f"{identity}: gate payout and success rate must be finite")
        if row["treatment"] == "C0":
            if row.get("gate_reward_present") is not False or payout != 0.0:
                raise ValueError(f"{identity}: C0 needs absent gate reward and zero payout")
        elif row.get("gate_reward_present") is not True or payout <= 0.0:
            raise ValueError(f"{identity}: C-Gate needs positive actual gate payout")

    for field in _PILOT_SHARED_FIELDS:
        values = [row.get(field) for row in rows]
        if values[0] in (None, "") or len(set(values)) != 1:
            raise ValueError(f"pilot rows must share {field}")

    continuation_by_arm = {
        arm: any(float(row["success_rate_sampled"]) >= 0.25 for row in rows if row["treatment"] == arm)
        for arm in ("C0", "C-Gate")
    }
    return {
        "valid": True,
        "row_identities": list(PILOT_IDENTITIES),
        "continuation_by_arm": continuation_by_arm,
        "continuation_allowed": all(continuation_by_arm.values()),
    }
