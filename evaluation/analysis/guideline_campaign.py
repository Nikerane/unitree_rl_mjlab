"""Pure, episode-first analysis for the fixed-reset Cartesian guideline pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


FAILURE_ERROR_M = 0.050
CORRIDOR_RADIUS_M = 0.005
NUM_GATES = 6
EXPECTED_EPISODE_COUNT = 512
_PILOT_EXACT_FIELDS = {
    "checkpoint_filename": "model_499.pt",
    "num_envs": 256,
    "episode_len_s": 4.0,
    "n_episodes_sampled": EXPECTED_EPISODE_COUNT,
    "episodes_per_env_sampled": 2,
    "reset_rng_seed": 2036072919,
    "observation_rng_seed": 2046072933,
    "action_rng_seed": 2056072941,
}
_PILOT_BOUNDED_RESULTS = {
    "q90_terminal_descent_perpendicular_error_m_sampled": (
        0.0, FAILURE_ERROR_M, "primary q90"
    ),
    "success_rate_sampled": (0.0, 1.0, "success rate"),
    "all_six_gates_rate_sampled": (0.0, 1.0, "all-six-gates rate"),
    "corridor_occupancy_mean_sampled": (0.0, 1.0, "corridor occupancy"),
    "backward_progress_count_mean_sampled": (0.0, math.inf, "backward progress"),
}
PILOT_TASKS = {
    "C0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
    "C-Gate": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
}
FROZEN_MANUFACTURER_IMPULSE_CAPS_N_M_S = (
    1.640,
    3.280,
    1.640,
    1.640,
    1.640,
    1.640,
)
PILOT_IDENTITIES = (("C0", 0), ("C0", 1), ("C-Gate", 0), ("C-Gate", 1))
_PILOT_SHARED_FIELDS = (
    "reset_digest",
    "guideline_geometry_digest",
    "reset_rng_seed",
    "observation_rng_seed",
    "action_rng_seed",
    "git_revision",
    "asset_git_revision",
    "treatment_base_identity",
    "accepted_manifest_sha256",
    "training_code_revision",
    "training_asset_revision",
    "campaign_config_sha256",
)
_PILOT_SHA256_FIELDS = (
    "checkpoint_sha256",
    "accepted_checkpoint_sha256",
    "accepted_manifest_sha256",
    "campaign_config_sha256",
    "treatment_config_sha256",
    "sampled_trace_digest",
    "sampled_trace_artifact_sha256",
    "reset_digest",
    "guideline_geometry_digest",
    "treatment_base_identity",
)
_PILOT_REVISION_FIELDS = (
    "training_code_revision",
    "training_asset_revision",
    "git_revision",
    "asset_git_revision",
)


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value.lower()
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
        if isinstance(accepted_raw, bool) or not isinstance(
            accepted_raw, (int, np.integer)
        ):
            raise ValueError(
                "first_strike.accepted_onset_index must be a literal integral index"
            )
        accepted_contact = int(accepted_raw)
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
    if not np.array_equal(caps, np.asarray(FROZEN_MANUFACTURER_IMPULSE_CAPS_N_M_S)):
        raise ValueError("trace impulse caps drift from frozen manufacturer impulse caps")
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
    if len(identities) != len(PILOT_IDENTITIES) or set(identities) != set(PILOT_IDENTITIES):
        raise ValueError("pilot rows must contain exactly C0/C-Gate seeds 0/1")
    for row in rows:
        identity = f"{row['treatment']}/seed{row['training_seed']}"
        if row.get("task") != PILOT_TASKS[row["treatment"]]:
            raise ValueError(f"{identity}: exact registered task is required")
        for field, expected in _PILOT_EXACT_FIELDS.items():
            value = row.get(field)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"{identity}: {field} must be literal {expected!r}")
        reset_range = _array(
            row.get("reset_position_range_rad"),
            name=f"{identity}: reset_position_range_rad",
            shape=(2,),
        )
        if not np.array_equal(reset_range, np.zeros(2)):
            raise ValueError(f"{identity}: reset range must be literally (0.0, 0.0)")
        if row.get("windup_enabled") is not False:
            raise ValueError(f"{identity}: wind-up must be disabled")
        if row.get("impedance_mode") != "fixed":
            raise ValueError(f"{identity}: fixed impedance is required")
        imp_max_p = row.get("imp_max_p")
        if isinstance(imp_max_p, bool):
            raise ValueError(f"{identity}: imp_max_p must be literal 0.0")
        try:
            imp_max_p = float(imp_max_p)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{identity}: imp_max_p must be literal 0.0") from error
        if not math.isfinite(imp_max_p) or imp_max_p != 0.0:
            raise ValueError(f"{identity}: imp_max_p must be literal 0.0")
        for field in ("git_dirty", "asset_git_dirty"):
            if row.get(field) is not False:
                raise ValueError(f"{identity}: {field} must be literal False")
        for field in _PILOT_SHA256_FIELDS:
            if not _is_hex(row.get(field), 64):
                raise ValueError(f"{identity}: {field} must be 64 hexadecimal characters")
        for field in _PILOT_REVISION_FIELDS:
            if not _is_hex(row.get(field), 40):
                raise ValueError(f"{identity}: {field} must be a 40-hex revision")
        if row["checkpoint_sha256"] != row["accepted_checkpoint_sha256"]:
            raise ValueError(f"{identity}: accepted checkpoint SHA-256 mismatch")
        if row["training_asset_revision"] != row["asset_git_revision"]:
            raise ValueError(f"{identity}: training/evaluation asset revision mismatch")
        for sentinel in ("impossible_success_n", "lambda_dead_n"):
            if row.get(sentinel) != 0:
                raise ValueError(f"{identity}: {sentinel} must be zero")
        try:
            nonfinite_qvel = float(row["qvel_nonfinite_rate_sampled"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{identity}: qvel nonfinite rate is required") from error
        if not math.isfinite(nonfinite_qvel) or nonfinite_qvel != 0.0:
            raise ValueError(f"{identity}: nonfinite qvel invalidates the pilot row")
        results = {}
        for field, (lower, upper, label) in _PILOT_BOUNDED_RESULTS.items():
            try:
                raw_value = row[field]
                value = float(raw_value)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{identity}: {label} is required") from error
            if isinstance(raw_value, bool) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{identity}: {label} is outside its valid range")
            results[field] = value
        success_rate = results["success_rate_sampled"]
        try:
            payout = float(row["actual_gate_return_total_sampled"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{identity}: gate payout is required") from error
        if not math.isfinite(payout):
            raise ValueError(f"{identity}: gate payout must be finite")
        if row["treatment"] == "C0":
            if (
                row.get("r_gate_present") is not False
                or row.get("r_gate_weight") is not None
                or row.get("gate_reward_present") is not False
                or payout != 0.0
            ):
                raise ValueError(f"{identity}: C0 needs absent gate reward and zero payout")
        else:
            r_gate_weight = row.get("r_gate_weight")
            if isinstance(r_gate_weight, bool):
                raise ValueError(f"{identity}: C-Gate needs r_gate at weight 8.0")
            try:
                r_gate_weight = float(r_gate_weight)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{identity}: C-Gate needs r_gate at weight 8.0") from error
            if (
                row.get("r_gate_present") is not True
                or not math.isfinite(r_gate_weight)
                or r_gate_weight != 8.0
                or row.get("gate_reward_present") is not True
                or payout <= 0.0
            ):
                raise ValueError(f"{identity}: C-Gate needs r_gate at weight 8.0 and positive actual payout")

    for field in _PILOT_SHARED_FIELDS:
        values = [row.get(field) for row in rows]
        if values[0] in (None, "") or len(set(values)) != 1:
            raise ValueError(f"pilot rows must share {field}")

    treatment_configs = {
        arm: {row["treatment_config_sha256"] for row in rows if row["treatment"] == arm}
        for arm in PILOT_TASKS
    }
    if any(len(values) != 1 for values in treatment_configs.values()):
        raise ValueError("pilot rows must share treatment config within each arm")
    if len({next(iter(values)) for values in treatment_configs.values()}) != 2:
        raise ValueError("C0 and C-Gate require distinct treatment config identities")
    for field in ("sampled_trace_digest", "sampled_trace_artifact_sha256"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"pilot rows require unique {field}")

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


_PILOT_CSV_SCHEMA = {
    **dict.fromkeys(("training_seed", "num_envs", "n_episodes_sampled",
                     "episodes_per_env_sampled", "reset_rng_seed",
                     "observation_rng_seed", "action_rng_seed",
                     "impossible_success_n", "lambda_dead_n"), "int"),
    **dict.fromkeys(("episode_len_s", "imp_max_p",
                     "qvel_nonfinite_rate_sampled",
                     "q90_terminal_descent_perpendicular_error_m_sampled",
                     "all_six_gates_rate_sampled",
                     "corridor_occupancy_mean_sampled",
                     "backward_progress_count_mean_sampled",
                     "actual_gate_return_total_sampled",
                     "success_rate_sampled"), "float"),
    **dict.fromkeys(("windup_enabled", "r_gate_present", "gate_reward_present",
                     "git_dirty", "asset_git_dirty"), "bool"),
    **dict.fromkeys(("task", "treatment", "checkpoint_filename",
                     "checkpoint_sha256", "accepted_checkpoint_sha256",
                     "accepted_manifest_sha256", "training_code_revision",
                     "training_asset_revision", "campaign_config_sha256",
                     "treatment_config_sha256", "sampled_trace_digest",
                     "sampled_trace_artifact_sha256", "reset_digest",
                     "guideline_geometry_digest", "impedance_mode",
                     "treatment_base_identity", "git_revision",
                     "asset_git_revision"), "text"),
    "reset_position_range_rad": "reset_range",
    "r_gate_weight": "optional_float",
}


def _decode_pilot_csv_field(field: str, value: str, kind: str):
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"CSV field {field} has malformed text {value!r}")
    if kind == "optional_float" and value == "":
        return None
    if value == "":
        raise ValueError(f"CSV field {field} must not be empty")
    if kind == "text":
        return value
    if kind == "bool":
        if value not in ("True", "False"):
            raise ValueError(f"CSV field {field} must be literal True or False")
        return value == "True"
    if kind == "int":
        if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value) is None:
            raise ValueError(f"CSV field {field} must be a literal integer")
        return int(value)
    if kind == "reset_range":
        if value != "(0.0, 0.0)":
            raise ValueError(f"CSV field {field} must be literal (0.0, 0.0)")
        return (0.0, 0.0)
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"CSV field {field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"CSV field {field} must be finite")
    return number


def load_and_validate_guideline_pilot_csv(path: str | Path) -> dict:
    """Decode evaluator DictWriter rows and run the exact four-row pilot gate."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("guideline pilot CSV is missing a header")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("guideline pilot CSV contains duplicate columns")
        missing = sorted(set(_PILOT_CSV_SCHEMA) - set(fieldnames))
        if missing:
            raise ValueError(f"guideline pilot CSV missing columns: {missing}")
        raw_rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in raw_rows):
        raise ValueError("guideline pilot CSV contains malformed row widths")
    typed_rows = [
        {
            field: _decode_pilot_csv_field(field, row[field], kind)
            for field, kind in _PILOT_CSV_SCHEMA.items()
        }
        for row in raw_rows
    ]
    return validate_guideline_pilot_rows(typed_rows)
