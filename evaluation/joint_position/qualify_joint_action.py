"""Pure helpers for causal joint-target tape qualification."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import copy
import dataclasses
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


POST_REFERENCE_HOLD_CONTROL_STEPS = 10
SOURCE_TASK_ID = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    "CProgress-Vel-Delivered4"
)
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
REQUIRED_SEEDS = tuple(range(1000, 1016))
QVEL_LIMIT_RAD_S = 3.1415
CORRIDOR_RADIUS_M = 0.005
GEOMETRY_ATOL_M = 1e-6


def _finite_array(value: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _copied_row(value: object, joint_ids: object) -> np.ndarray:
    """Copy one environment's resolved joint row from NumPy or Torch storage."""
    row = value[0, joint_ids]
    if hasattr(row, "detach"):
        row = row.detach().cpu().numpy()
    return np.asarray(row, dtype=np.float64).copy()


def capture_control_step(
    env: Any,
    action: object,
    *,
    action_term_name: str,
    robot_data: Any,
    joint_ids: object,
    terminal_reader: Any | None = None,
) -> dict[str, Any]:
    """Step once while copying the first publicly applied joint target.

    DiffIK recomputes its joint target at every physics substep.  The command
    causal to a policy interval is therefore the public target written by its
    first ``apply_actions`` call, not the final recomputation and not realized
    joint position.
    """
    if bool(env.cfg.auto_reset):
        raise ValueError("causal target capture requires cfg.auto_reset=False")
    action_term = env.action_manager.get_term(action_term_name)
    original_apply = action_term.apply_actions
    first_applied_target: np.ndarray | None = None
    applied_substeps = 0

    def apply_and_capture() -> None:
        nonlocal applied_substeps, first_applied_target
        original_apply()
        applied_substeps += 1
        if applied_substeps == 1:
            first_applied_target = _copied_row(
                robot_data.joint_pos_target, joint_ids
            )

    action_term.apply_actions = apply_and_capture
    try:
        env.step(action)
    finally:
        action_term.apply_actions = original_apply
    if first_applied_target is None:
        raise RuntimeError("action term was not applied during env.step")
    terminal = terminal_reader() if terminal_reader is not None else None
    return {
        "first_applied_target": first_applied_target,
        "applied_substeps": applied_substeps,
        "terminal": terminal,
    }


def squared_tracking_error(applied_target: object, q_next: object) -> float:
    """Return the unweighted joint target-to-next-state squared error."""
    target = _finite_array(applied_target, name="applied_target", ndim=1)
    next_state = _finite_array(q_next, name="q_next", ndim=1)
    if target.shape != next_state.shape:
        raise ValueError("applied_target and q_next must have the same shape")
    return float(np.sum(np.square(target - next_state)))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using the artifact's canonical encoding."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite_json(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite_json(item)
            for key, item in value.items()
        )
    return False


def _geometry_max_error_m(actual: object, expected: object) -> float:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return math.inf
    maximum = 0.0
    for name, shape in (
        ("entry_m", (3,)),
        ("nail_m", (3,)),
        ("gate_centers_m", (6, 3)),
    ):
        try:
            actual_array = np.asarray(actual[name], dtype=np.float64)
            expected_array = np.asarray(expected[name], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return math.inf
        if (
            actual_array.shape != shape
            or expected_array.shape != shape
            or not np.isfinite(actual_array).all()
            or not np.isfinite(expected_array).all()
        ):
            return math.inf
        maximum = max(maximum, float(np.max(np.abs(actual_array - expected_array))))
    return maximum


def rollout_gate_failures(
    record: dict[str, Any], *, expected_geometry: object | None = None
) -> list[str]:
    """Apply every frozen source/replay G1 gate to one diagnostic record."""
    if record.get("available") is False:
        mode = record.get("mode", "rollout")
        reason = record.get("unavailable_reason")
        if not isinstance(reason, str) or not reason:
            reason = "no finite unavailability reason was recorded"
        failures = [f"{mode} unavailable: {reason}"]
        if not _finite_json(record):
            failures.append("unavailable evidence contains a non-finite value")
        return failures
    failures: list[str] = []
    if tuple(record.get("joint_names", ())) != JOINT_NAMES:
        failures.append("joint names must be exactly joint1 through joint6")
    if tuple(record.get("actuator_names", ())) != JOINT_NAMES:
        failures.append("actuator names must be exactly joint1 through joint6")
    if record.get("finite") is not True or not _finite_json(record):
        failures.append("non-finite action, target, state, reward, or metric")
    target_count = record.get("target_count")
    playback_count = record.get("playback_length")
    executed_hold = record.get("executed_post_reference_hold_control_steps")
    scheduled_hold = record.get("scheduled_post_reference_hold_control_steps")
    if scheduled_hold != POST_REFERENCE_HOLD_CONTROL_STEPS:
        failures.append("scheduled post-reference hold must be exactly ten steps")
    if (
        isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or isinstance(playback_count, bool)
        or not isinstance(playback_count, int)
        or isinstance(executed_hold, bool)
        or not isinstance(executed_hold, int)
        or executed_hold != max(0, target_count - playback_count)
    ):
        failures.append("executed hold count does not match executed targets")
    if (
        isinstance(executed_hold, int)
        and executed_hold < POST_REFERENCE_HOLD_CONTROL_STEPS
        and (
            record.get("terminal_transition_captured") is not True
            or not isinstance(record.get("terminal_reason"), str)
            or not record.get("terminal_reason")
        )
    ):
        failures.append("shortened hold lacks captured terminal evidence")
    substeps = record.get("applied_substeps_per_target")
    if (
        not isinstance(substeps, list)
        or len(substeps) != target_count
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value != 10
            for value in substeps
        )
    ):
        failures.append("every target must receive exactly ten applied substeps")
    if record.get("normalized_action_saturation_count") != 0:
        failures.append("normalized-action saturation occurred")
    if record.get("physical_target_clipping_count") != 0:
        failures.append("physical target clipping occurred")
    if record.get("joint_limit_violation_count") != 0:
        failures.append("joint-limit violation occurred")
    try:
        qvel_peak = float(record["qvel_peak_rad_s"])
    except (KeyError, TypeError, ValueError, OverflowError):
        qvel_peak = math.inf
    if not math.isfinite(qvel_peak) or qvel_peak > QVEL_LIMIT_RAD_S:
        failures.append("full-rate qvel exceeds 3.1415 rad/s")
    if record.get("production_contact_seen") is not True:
        failures.append("production-accepted contact was not observed")
    if record.get("first_strike_finalized") is not True:
        failures.append("first strike did not finalize")
    if record.get("first_strike_productive") is not True:
        failures.append("first strike was not productive")
    onset = record.get("accepted_onset_control_step")
    playback_length = record.get("playback_length")
    if (
        isinstance(onset, bool)
        or not isinstance(onset, int)
        or isinstance(playback_length, bool)
        or not isinstance(playback_length, int)
        or onset > playback_length + 2
    ):
        failures.append("accepted onset occurred later than playback_length + 2")
    try:
        speed = float(record["productive_precontact_axial_speed_m_s"])
    except (KeyError, TypeError, ValueError, OverflowError):
        speed = -math.inf
    if not math.isfinite(speed) or speed < 0.5:
        failures.append("productive precontact axial speed is below 0.5 m/s")
    try:
        impulse = float(record["first_event_nail_axis_impulse_n_s"])
    except (KeyError, TypeError, ValueError, OverflowError):
        impulse = -math.inf
    if not math.isfinite(impulse) or impulse <= 0.0:
        failures.append("finalized first-event nail-axis impulse is not positive")
    if record.get("success_within_first_event") is not True:
        failures.append("nail threshold was not reached within the first event")
    if record.get("success") is not True:
        failures.append("success threshold not reached")
    if record.get("ordered_waypoints_before_contact") != 6:
        failures.append("six ordered waypoints were not crossed before contact")
    try:
        corridor = float(record["post_gate_1_corridor_max_m"])
    except (KeyError, TypeError, ValueError, OverflowError):
        corridor = math.inf
    if not math.isfinite(corridor) or corridor > CORRIDOR_RADIUS_M:
        failures.append("post-gate-1 corridor error exceeds 0.005 m")
    if record.get("multi_gate_crossings") != 0:
        failures.append("multi-gate-crossing anomaly occurred")
    if record.get("waypoint_credit_after_contact") is not False:
        failures.append("waypoint credit accrued after contact")
    geometry = record.get("geometry")
    if not math.isfinite(_geometry_max_error_m(geometry, geometry)):
        failures.append("geometry is missing, malformed, or non-finite")
    if (
        expected_geometry is not None
        and _geometry_max_error_m(geometry, expected_geometry) > GEOMETRY_ATOL_M
    ):
        failures.append("source/replay geometry differs by more than 1e-6 m")
    return failures


def build_contract_payload(
    *,
    source_code_revision: str,
    source_asset_revision: str,
    source_task_config_projection: dict[str, Any],
    default_joint_pos_rad: object,
    physical_clip_rad: object,
    source_target_tape_rad: object,
    source_rows: Sequence[dict[str, Any]],
    replay_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic PASS/FAIL artifact from paired source/replay rows."""
    tape = _finite_array(source_target_tape_rad, name="source_target_tape_rad", ndim=2)
    default = _finite_array(default_joint_pos_rad, name="default_joint_pos_rad", ndim=1)
    clips = _finite_array(physical_clip_rad, name="physical_clip_rad", ndim=2)
    tape_json = tape.tolist()
    tape_sha256 = canonical_sha256(tape_json)
    projection = copy.deepcopy(source_task_config_projection)
    projection_sha256 = canonical_sha256(projection)

    paired_rows: list[dict[str, Any]] = []
    pair_count = max(len(source_rows), len(replay_rows))
    for index in range(pair_count):
        source = copy.deepcopy(source_rows[index]) if index < len(source_rows) else {}
        replay = copy.deepcopy(replay_rows[index]) if index < len(replay_rows) else {}
        source_failures = rollout_gate_failures(source)
        replay_failures = rollout_gate_failures(
            replay, expected_geometry=source.get("geometry")
        )
        source["failures"] = source_failures
        source["passed"] = not source_failures
        replay["failures"] = replay_failures
        replay["passed"] = not replay_failures
        seed = source.get("seed", replay.get("seed"))
        pair_failures = list(source_failures)
        pair_failures.extend(replay_failures)
        if source.get("seed") != replay.get("seed"):
            pair_failures.append("source/replay seed mismatch")
        if source.get("source_target_tape_sha256") != tape_sha256:
            pair_failures.append("source row target-tape hash mismatch")
        if replay.get("source_target_tape_sha256") != tape_sha256:
            pair_failures.append("replay row target-tape hash mismatch")
        paired_rows.append(
            {
                "seed": seed,
                "passed": not pair_failures,
                "failures": pair_failures,
                "source_target_tape_sha256": tape_sha256,
                "source": source,
                "replay": replay,
            }
        )

    decision = "PASS" if qualification_passes(paired_rows, REQUIRED_SEEDS) else "FAIL"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "action_semantics": "absolute_default_offset_joint_position",
        "source_task_id": SOURCE_TASK_ID,
        "source_code_revision": source_code_revision,
        "source_asset_revision": source_asset_revision,
        "source_task_config_projection": projection,
        "source_task_config_sha256": projection_sha256,
        "joint_names": list(JOINT_NAMES),
        "actuator_names": list(JOINT_NAMES),
        "default_joint_pos_rad": default.tolist(),
        "physical_clip_rad": clips.tolist(),
        "scale_rad": derive_scale_by_joint(tape, default).tolist(),
        "physics_dt_s": 0.002,
        "control_decimation": 10,
        "post_reference_hold_control_steps": POST_REFERENCE_HOLD_CONTROL_STEPS,
        "seeds": list(REQUIRED_SEEDS),
        "source_target_tape_rad": tape_json,
        "source_target_tape_sha256": tape_sha256,
        "per_seed_replay_rows": paired_rows,
        "decision": decision,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def unavailable_rollout_record(
    *, seed: int, mode: str, reason: str, tape_sha256: str
) -> dict[str, Any]:
    """Represent an unexecuted rollout without inventing scientific evidence."""
    if mode not in {"source", "replay"}:
        raise ValueError("unavailable rollout mode must be source or replay")
    return {
        "seed": int(seed),
        "mode": mode,
        "available": False,
        "unavailable_reason": str(reason),
        "source_target_tape_sha256": tape_sha256,
        "finite": True,
        "passed": False,
    }


def build_unavailable_diagnostic_payload(
    *, seeds: tuple[int, ...], reason: str
) -> dict[str, Any]:
    """Build finite CLI evidence when qualification fails before source capture."""
    tape: list[Any] = []
    tape_sha256 = canonical_sha256(tape)
    rows = []
    for seed in seeds:
        source = unavailable_rollout_record(
            seed=seed, mode="source", reason=reason, tape_sha256=tape_sha256
        )
        replay = unavailable_rollout_record(
            seed=seed, mode="replay", reason=reason, tape_sha256=tape_sha256
        )
        rows.append(
            {
                "seed": seed,
                "passed": False,
                "failures": [reason],
                "source_target_tape_sha256": tape_sha256,
                "source": source,
                "replay": replay,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "action_semantics": "absolute_default_offset_joint_position",
        "source_task_id": SOURCE_TASK_ID,
        "source_code_revision": None,
        "source_asset_revision": None,
        "source_task_config_projection": None,
        "source_task_config_sha256": None,
        "joint_names": list(JOINT_NAMES),
        "actuator_names": list(JOINT_NAMES),
        "default_joint_pos_rad": None,
        "physical_clip_rad": None,
        "scale_rad": None,
        "physics_dt_s": 0.002,
        "control_decimation": 10,
        "post_reference_hold_control_steps": POST_REFERENCE_HOLD_CONTROL_STEPS,
        "seeds": list(seeds),
        "source_target_tape_rad": tape,
        "source_target_tape_sha256": tape_sha256,
        "per_seed_replay_rows": rows,
        "diagnostic_failures": [reason],
        "decision": "FAIL",
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def write_canonical_json(path: str | Path, payload: object) -> None:
    """Serialize deterministically while rejecting NaN and infinities."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_seed_spec(value: str) -> tuple[int, ...]:
    """Parse the CLI's explicit half-open ``start:stop`` seed range."""
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("--seeds must use half-open start:stop syntax")
    try:
        start, stop = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--seeds bounds must be integers") from exc
    if start < 0 or stop <= start:
        raise ValueError("--seeds must be a nonempty increasing range")
    return tuple(range(start, stop))


def _json_projection(value: object) -> Any:
    """Project config values to stable, explicit JSON-compatible identities."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("source configuration contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_projection(value.item())
    if isinstance(value, Enum):
        return _json_projection(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {
            "start": _json_projection(value.start),
            "stop": _json_projection(value.stop),
            "step": _json_projection(value.step),
        }
    if isinstance(value, dict):
        return {
            str(key): _json_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_projection(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_projection(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{qualname}"
    raise TypeError(f"cannot project config value of type {type(value).__name__}")


def _manager_section_projection(
    value: object, *, observation_groups: bool = False
) -> dict[str, Any]:
    """Project manager config while preserving behaviorally significant order."""
    if not isinstance(value, dict):
        raise TypeError("manager configuration section must be a mapping")
    projection = {
        "term_order": [str(name) for name in value],
        "terms": _json_projection(value),
    }
    if observation_groups:
        projection["group_term_order"] = {
            str(name): [str(term_name) for term_name in group.terms]
            for name, group in value.items()
        }
    return projection


def source_task_config_projection(cfg: Any) -> dict[str, Any]:
    """Bind every task-defining section needed to reproduce source capture."""
    robot_cfg = cfg.scene.entities["robot"]
    articulation = robot_cfg.articulation
    if articulation is None:
        raise ValueError("source robot configuration has no articulation")
    projection = {
        "action": _json_projection(cfg.actions),
        "observations": _manager_section_projection(
            cfg.observations, observation_groups=True
        ),
        "rewards": _manager_section_projection(cfg.rewards),
        "metrics": _manager_section_projection(cfg.metrics),
        "events": _manager_section_projection(cfg.events),
        "actuators": {
            "robot": _json_projection(articulation.actuators),
        },
        "timing": {
            "physics_dt_s": float(cfg.sim.mujoco.timestep),
            "control_decimation": int(cfg.decimation),
        },
        "reset": {
            "auto_reset": bool(cfg.auto_reset),
            "reset_robot_joints": _json_projection(
                cfg.events["reset_robot_joints"]
            ),
            "reset_nail": _json_projection(cfg.events["reset_nail"]),
        },
    }
    # This is both a validation pass and the exact canonical encoding used for
    # the digest.  It rejects unsupported or non-finite config content early.
    _canonical_json_bytes(projection)
    return projection


def validate_live_physical_limits(limits: object) -> np.ndarray:
    """Require compiled limits to equal the frozen Task-1 physical contract."""
    from src.tasks.hammer.config.z1.joint_position_contract import (
        PHYSICAL_CLIP_RAD,
    )

    live = _finite_array(limits, name="compiled physical joint limits", ndim=2)
    frozen = np.asarray(PHYSICAL_CLIP_RAD, dtype=np.float64)
    if live.shape != frozen.shape or not np.allclose(
        live, frozen, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            "compiled physical joint limits do not match the frozen Task-1 contract"
        )
    return live.copy()


def read_compiled_physical_limits(
    env: Any, robot: Any, local_joint_ids: object
) -> np.ndarray:
    """Resolve entity-local arm joints into MuJoCo's raw float64 range table."""
    global_joint_ids = robot.indexing.joint_ids[local_joint_ids]
    if hasattr(global_joint_ids, "detach"):
        global_joint_ids = global_joint_ids.detach().cpu().numpy()
    global_joint_ids = np.asarray(global_joint_ids, dtype=np.int64)
    raw_ranges = np.asarray(env.sim.mj_model.jnt_range, dtype=np.float64)
    return validate_live_physical_limits(raw_ranges[global_joint_ids])


def configure_replay_cfg(
    source_cfg: Any, *, scale_rad: object, physical_clip_rad: object
) -> Any:
    """Replace the Cartesian source action with a temporary six-joint replay."""
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    scale = _finite_array(scale_rad, name="scale_rad", ndim=1)
    if scale.shape != (len(JOINT_NAMES),) or np.any(scale <= 0.0):
        raise ValueError("scale_rad must contain six positive values")
    clips = validate_live_physical_limits(physical_clip_rad)
    cfg = copy.deepcopy(source_cfg)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=JOINT_NAMES,
            scale=dict(zip(JOINT_NAMES, scale.tolist(), strict=True)),
            clip={
                name: tuple(float(bound) for bound in bounds)
                for name, bounds in zip(JOINT_NAMES, clips, strict=True)
            },
            use_default_offset=True,
            preserve_order=True,
        )
    }
    return cfg


def _numpy_copy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).copy()


def numeric_tree_is_finite(value: object) -> bool:
    """Return whether every numeric leaf in a nested state tree is finite."""
    if isinstance(value, dict):
        return all(numeric_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_tree_is_finite(item) for item in value)
    try:
        array = _numpy_copy(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(array).all())


def manager_term_snapshot(
    manager: Any, *, env_index: int
) -> tuple[dict[str, Any], bool]:
    """Capture every active manager term without serializing NaN or infinity."""
    snapshot: dict[str, Any] = {}
    all_finite = True
    for name, values in manager.get_active_iterable_terms(env_index):
        numeric = [float(value) for value in values]
        finite = all(math.isfinite(value) for value in numeric)
        all_finite &= finite
        if not finite:
            snapshot[name] = None
        elif len(numeric) == 1:
            snapshot[name] = numeric[0]
        else:
            snapshot[name] = numeric
    return snapshot, all_finite


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _external_asset_revision(xml_path: Path) -> str:
    root = Path(
        subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=xml_path.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return _git_revision(root)


def _terminal_reason(env: Any, first_tracker: Any) -> str | None:
    if not bool(env.reset_buf[0]):
        return None
    if bool(env.reset_time_outs[0]):
        return "time_out"
    if bool(first_tracker.finalized[0]):
        reason = int(first_tracker.reason[0])
        if reason == 1:
            return "success"
        if reason == 2:
            return "first_event_window"
    return "non_timeout_termination"


def scheduled_replay_targets(
    observed_tape_rad: object,
    *,
    playback_length: int,
    hold_control_steps: int = POST_REFERENCE_HOLD_CONTROL_STEPS,
) -> np.ndarray:
    """Extend an observed tape only by zero-order holding its last target."""
    tape = _finite_array(observed_tape_rad, name="observed_tape_rad", ndim=2)
    if tape.shape[0] == 0 or tape.shape[1] != len(JOINT_NAMES):
        raise ValueError("observed_tape_rad must be nonempty and six joints wide")
    if (
        isinstance(playback_length, bool)
        or not isinstance(playback_length, int)
        or playback_length <= 0
        or isinstance(hold_control_steps, bool)
        or not isinstance(hold_control_steps, int)
        or hold_control_steps < 0
    ):
        raise ValueError("playback_length and hold_control_steps must be valid integers")
    scheduled_count = playback_length + hold_control_steps
    if len(tape) < playback_length:
        raise ValueError("source terminated before its playback targets were observed")
    if len(tape) > scheduled_count:
        raise ValueError("observed tape exceeds the preregistered playback/hold schedule")
    if len(tape) == scheduled_count:
        return tape.copy()
    extension = np.repeat(tape[-1:], scheduled_count - len(tape), axis=0)
    return np.concatenate((tape, extension), axis=0)


def _rollout_mode(
    cfg: Any,
    *,
    seeds: tuple[int, ...],
    mode: str,
    source_tape_rad: np.ndarray | None = None,
    playback_lengths: dict[int, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Execute a rollout while owning cleanup from the moment env construction ends."""
    from mjlab.envs import ManagerBasedRlEnv

    if mode not in {"source", "replay"}:
        raise ValueError(f"unsupported rollout mode {mode!r}")
    if mode == "replay" and (source_tape_rad is None or playback_lengths is None):
        raise ValueError("replay requires a source tape and playback lengths")

    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cpu")
    hook_state: dict[str, Any] = {"installed": False}
    try:
        return _rollout_mode_open_env(
            env,
            cfg,
            seeds=seeds,
            mode=mode,
            source_tape_rad=source_tape_rad,
            playback_lengths=playback_lengths,
            hook_state=hook_state,
        )
    finally:
        if hook_state["installed"]:
            hook_state["manager"].compute_substep = hook_state["original"]
        env.close()


def _rollout_mode_open_env(
    env: Any,
    cfg: Any,
    *,
    seeds: tuple[int, ...],
    mode: str,
    source_tape_rad: np.ndarray | None,
    playback_lengths: dict[int, int] | None,
    hook_state: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Execute against an owned open env; the caller restores hooks and closes it."""
    import torch
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from src.assets.robots.unitree_z1.z1_constants import (
        ARM_ACTUATOR_NAMES,
        ARM_JOINT_NAMES,
        HAMMER_HEAD_SITE_NAME,
        Z1_HAMMER_DELTA_POS_SCALE,
    )
    from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
    from src.tasks.hammer.mdp.guideline import (
        _ENV_GUIDELINE_ATTR,
        GUIDELINE_NUM_GATES,
        project_to_reference,
    )
    from src.tasks.hammer.mdp.references import SingleStrikeReference
    from src.tasks.hammer.mdp.rewards import clamped_nail_depth
    from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD

    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    contact_sensor = env.scene["hammer_nail_contact"]
    first_tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR)
    guideline_tracker = getattr(env, _ENV_GUIDELINE_ATTR)

    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    head_cfg.resolve(env.scene)
    nail_top_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    nail_top_cfg.resolve(env.scene)
    nail_joint_cfg = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
    nail_joint_cfg.resolve(env.scene)
    arm_ids = arm_cfg.joint_ids

    default_joint_pos = _numpy_copy(
        robot.data.default_joint_pos[0, arm_ids]
    )
    physical_limits = read_compiled_physical_limits(env, robot, arm_ids)
    public_limit_mirror = _numpy_copy(robot.data.joint_pos_limits[0, arm_ids])
    expected_public_mirror = physical_limits.astype(np.float32).astype(np.float64)
    if not np.array_equal(public_limit_mirror, expected_public_mirror):
        raise RuntimeError(
            "public float32 joint-limit mirror disagrees with raw compiled MuJoCo limits"
        )
    if tuple(ARM_JOINT_NAMES) != JOINT_NAMES or tuple(ARM_ACTUATOR_NAMES) != JOINT_NAMES:
        raise RuntimeError("live Z1 arm names do not match the six-joint contract")
    local_arm_ids = np.asarray(arm_ids, dtype=np.int64)
    resolved_joint_names = tuple(
        robot.joint_names[int(joint_id)] for joint_id in local_arm_ids
    )
    action_term_name = "ik_hammer_head" if mode == "source" else "joint_pos"
    live_action_term = env.action_manager.get_term(action_term_name)
    live_action_ids = (
        live_action_term._joint_ids
        if mode == "source"
        else live_action_term.target_ids
    )
    live_action_ids = np.asarray(
        _numpy_copy(live_action_ids), dtype=np.int64
    )
    live_actuator_names = (
        tuple(live_action_term.cfg.actuator_names)
        if mode == "source"
        else tuple(live_action_term.target_names)
    )
    if (
        resolved_joint_names != JOINT_NAMES
        or live_actuator_names != JOINT_NAMES
        or not np.array_equal(live_action_ids, local_arm_ids)
    ):
        raise RuntimeError(
            "resolved action target order is not exactly joint1 through joint6"
        )

    def head_position() -> Any:
        return robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)

    def nail_top_position() -> Any:
        return nail.data.site_pos_w[:, nail_top_cfg.site_ids].squeeze(1)

    rows: list[dict[str, Any]] = []
    captured_tapes: dict[int, np.ndarray] = {}
    active: dict[str, Any] | None = None
    control_index = 0
    original_substep = env.metrics_manager.compute_substep

    def record_substep() -> None:
        nonlocal active
        original_substep()
        if active is None:
            return
        qpos = _numpy_copy(robot.data.joint_pos[0, arm_ids])
        qvel = _numpy_copy(robot.data.joint_vel[0, arm_ids])
        target = _numpy_copy(robot.data.joint_pos_target[0, arm_ids])
        head = _numpy_copy(head_position()[0])
        nail_top = _numpy_copy(nail_top_position()[0])
        active["qvel_peak_rad_s"] = max(
            active["qvel_peak_rad_s"], float(np.max(np.abs(qvel)))
        )
        outside = (qpos < physical_limits[:, 0] - 1e-9) | (
            qpos > physical_limits[:, 1] + 1e-9
        )
        active["joint_limit_violation_count"] += int(bool(np.any(outside)))
        numeric = np.concatenate((qpos, qvel, target, head, nail_top))
        active["finite"] &= bool(np.isfinite(numeric).all())

        if bool(guideline_tracker.initialized[0]):
            if active["entry"] is None:
                active["entry"] = _numpy_copy(guideline_tracker.entry[0])
                active["nail"] = _numpy_copy(guideline_tracker.nail[0])
            if int(guideline_tracker.next_gate[0]) >= 1 and not bool(
                first_tracker.started[0]
            ):
                _, error = project_to_reference(
                    head_position(), guideline_tracker.entry, guideline_tracker.nail
                )
                active["corridor_errors"].append(float(error[0]))

        started = bool(first_tracker.started[0])
        if started and active["accepted_onset_control_step"] is None:
            active["accepted_onset_control_step"] = control_index
            active["ordered_waypoints_before_contact"] = int(
                guideline_tracker.next_gate[0]
            )
            active["credit_at_contact"] = float(
                guideline_tracker.episode_credit[0]
            )
            if bool(guideline_tracker.initialized[0]):
                _, error = project_to_reference(
                    head_position(), guideline_tracker.entry, guideline_tracker.nail
                )
                active["corridor_errors"].append(float(error[0]))
        elif started and active["credit_at_contact"] is not None:
            if float(guideline_tracker.episode_credit[0]) > (
                active["credit_at_contact"] + 1e-12
            ):
                active["waypoint_credit_after_contact"] = True

        active["production_contact_seen"] |= started
        active["raw_contact_seen"] |= bool(
            (contact_sensor.data.found[0] > 0).any()
        )
        metric_values = np.asarray(
            [
                float(first_tracker.v_precontact[0]),
                float(first_tracker.delivered[0]),
                float(first_tracker.peak_depth[0]),
                float(guideline_tracker.episode_credit[0]),
            ],
            dtype=np.float64,
        )
        active["finite"] &= bool(np.isfinite(metric_values).all())

    env.metrics_manager.compute_substep = record_substep
    hook_state.update(
        installed=True,
        manager=env.metrics_manager,
        original=original_substep,
    )

    try:
        for seed in seeds:
            active = None
            env.reset(seed=seed)
            reference = SingleStrikeReference(1, env.device)
            reference.update(
                head_position(), nail_top_position(), env.episode_length_buf
            )
            playback_length = (
                reference.playback_length()
                if mode == "source"
                else int(playback_lengths[seed])
            )
            scheduled_control_steps = (
                playback_length + POST_REFERENCE_HOLD_CONTROL_STEPS
                if mode == "source"
                else int(len(source_tape_rad))
            )
            trace: dict[str, Any] = {
                "finite": True,
                "qvel_peak_rad_s": float(
                    np.max(np.abs(_numpy_copy(robot.data.joint_vel[0, arm_ids])))
                ),
                "joint_limit_violation_count": 0,
                "production_contact_seen": False,
                "raw_contact_seen": False,
                "accepted_onset_control_step": None,
                "ordered_waypoints_before_contact": 0,
                "credit_at_contact": None,
                "waypoint_credit_after_contact": False,
                "corridor_errors": [],
                "entry": None,
                "nail": None,
            }
            active = trace
            applied_substeps: list[int] = []
            captured_targets: list[np.ndarray] = []
            normalized_saturation_count = 0
            normalized_action_peak_abs = 0.0
            cartesian_saturation_count = 0
            cartesian_action_peak_abs: float | None = None
            target_clipping_count = 0
            last_transition: dict[str, Any] | None = None
            control_index = 0

            for control_index in range(1, scheduled_control_steps + 1):
                if mode == "source":
                    target = reference.playback_target(
                        min(control_index, playback_length)
                    )
                    raw_action = (
                        target - head_position()
                    ) / Z1_HAMMER_DELTA_POS_SCALE
                    raw_peak = float(raw_action.abs().max())
                    cartesian_action_peak_abs = max(
                        cartesian_action_peak_abs or 0.0, raw_peak
                    )
                    cartesian_saturation_count += int(
                        bool((raw_action.abs() > 1.0).any())
                    )
                    action = raw_action.clamp(-1.0, 1.0)
                    intended_target: np.ndarray | None = None
                else:
                    intended_target = np.asarray(
                        source_tape_rad[control_index - 1], dtype=np.float64
                    )
                    normalized = normalized_action_for_target(
                        intended_target[None, :],
                        default_joint_pos,
                        derive_scale_by_joint(source_tape_rad, default_joint_pos),
                    )[0]
                    normalized_action_peak_abs = max(
                        normalized_action_peak_abs,
                        float(np.max(np.abs(normalized))),
                    )
                    normalized_saturation_count += int(
                        bool(np.any(np.abs(normalized) > 1.0))
                    )
                    action = torch.tensor(
                        normalized[None, :], dtype=torch.float32, device=env.device
                    )

                trace["finite"] &= bool(torch.isfinite(action).all())

                def read_transition() -> dict[str, Any]:
                    q_next = _numpy_copy(robot.data.joint_pos[0, arm_ids])
                    applied = _numpy_copy(
                        robot.data.joint_pos_target[0, arm_ids]
                    )
                    depth = float(clamped_nail_depth(env, nail_joint_cfg)[0])
                    reward_terms, reward_terms_finite = manager_term_snapshot(
                        env.reward_manager, env_index=0
                    )
                    metric_terms, metric_terms_finite = manager_term_snapshot(
                        env.metrics_manager, env_index=0
                    )
                    observation_finite = numeric_tree_is_finite(env.obs_buf)
                    trace["finite"] &= (
                        reward_terms_finite
                        and metric_terms_finite
                        and observation_finite
                    )
                    return {
                        "q_next_rad": q_next.tolist(),
                        "nail_depth_m": depth,
                        "first_strike_finalized": bool(first_tracker.finalized[0]),
                        "first_strike_productive": bool(first_tracker.productive[0]),
                        "first_strike_reason": int(first_tracker.reason[0]),
                        "applied_target_rad": applied.tolist(),
                        "tracking_error_sq_unweighted": squared_tracking_error(
                            applied, q_next
                        ),
                        "reward": float(env.reward_buf[0]),
                        "reward_terms": reward_terms,
                        "metric_terms": metric_terms,
                        "observations_finite": observation_finite,
                        "terminated": bool(env.reset_terminated[0]),
                        "time_out": bool(env.reset_time_outs[0]),
                    }

                captured = capture_control_step(
                    env,
                    action,
                    action_term_name=action_term_name,
                    robot_data=robot.data,
                    joint_ids=arm_ids,
                    terminal_reader=read_transition,
                )
                first_target = captured["first_applied_target"]
                captured_targets.append(first_target)
                applied_substeps.append(int(captured["applied_substeps"]))
                last_transition = captured["terminal"]
                trace["finite"] &= bool(
                    math.isfinite(float(last_transition["reward"]))
                    and _finite_json(last_transition)
                )
                if mode == "replay" and intended_target is not None:
                    target_clipping_count += int(
                        bool(
                            np.any(
                                np.abs(first_target - intended_target) > 1e-6
                            )
                        )
                    )
                target_clipping_count += int(
                    bool(
                        np.any(first_target < physical_limits[:, 0] - 1e-9)
                        or np.any(first_target > physical_limits[:, 1] + 1e-9)
                    )
                )
                if bool(env.reset_buf[0]):
                    break

            active = None
            tape = np.asarray(captured_targets, dtype=np.float64)
            captured_tapes[seed] = tape
            tape_hash = canonical_sha256(tape.tolist())
            entry = trace["entry"]
            frozen_nail = trace["nail"]
            if entry is None or frozen_nail is None:
                geometry: dict[str, Any] = {
                    "entry_m": None,
                    "nail_m": None,
                    "gate_centers_m": None,
                }
            else:
                centers = [
                    entry
                    + (index + 1) / (GUIDELINE_NUM_GATES + 1)
                    * (frozen_nail - entry)
                    for index in range(GUIDELINE_NUM_GATES)
                ]
                geometry = {
                    "entry_m": entry.tolist(),
                    "nail_m": frozen_nail.tolist(),
                    "gate_centers_m": [center.tolist() for center in centers],
                }
            terminal_captured = bool(env.reset_buf[0]) and last_transition is not None
            terminal_reason = _terminal_reason(env, first_tracker)
            depth = (
                float(last_transition["nail_depth_m"])
                if last_transition is not None
                else 0.0
            )
            executed_hold = max(0, len(captured_targets) - playback_length)
            corridor_max = (
                max(trace["corridor_errors"])
                if trace["corridor_errors"]
                else None
            )
            row: dict[str, Any] = {
                "seed": int(seed),
                "mode": mode,
                "joint_names": list(resolved_joint_names),
                "actuator_names": list(live_actuator_names),
                "source_target_tape_sha256": tape_hash,
                "finite": bool(trace["finite"]),
                "playback_length": int(playback_length),
                "target_count": len(captured_targets),
                "scheduled_post_reference_hold_control_steps": (
                    POST_REFERENCE_HOLD_CONTROL_STEPS
                ),
                "executed_post_reference_hold_control_steps": executed_hold,
                "terminal_transition_captured": terminal_captured,
                "terminal_reason": terminal_reason,
                "applied_substeps_per_target": applied_substeps,
                "normalized_action_saturation_count": normalized_saturation_count,
                "normalized_action_peak_abs": normalized_action_peak_abs,
                "cartesian_action_saturation_count": cartesian_saturation_count,
                "cartesian_action_peak_abs": cartesian_action_peak_abs,
                "physical_target_clipping_count": target_clipping_count,
                "joint_limit_violation_count": int(
                    trace["joint_limit_violation_count"]
                ),
                "qvel_peak_rad_s": float(trace["qvel_peak_rad_s"]),
                "production_contact_seen": bool(
                    trace["production_contact_seen"]
                ),
                "raw_contact_seen": bool(trace["raw_contact_seen"]),
                "first_strike_finalized": bool(first_tracker.finalized[0]),
                "first_strike_productive": bool(first_tracker.productive[0]),
                "accepted_onset_control_step": trace[
                    "accepted_onset_control_step"
                ],
                "productive_precontact_axial_speed_m_s": float(
                    first_tracker.v_precontact[0]
                ),
                "first_event_nail_axis_impulse_n_s": float(
                    first_tracker.delivered[0]
                ),
                "success_within_first_event": bool(
                    first_tracker.finalized[0]
                    and int(first_tracker.reason[0]) == 1
                ),
                "success": bool(depth >= NAIL_SUCCESS_THRESHOLD),
                "ordered_waypoints_before_contact": int(
                    trace["ordered_waypoints_before_contact"]
                ),
                "post_gate_1_corridor_max_m": corridor_max,
                "multi_gate_crossings": int(
                    guideline_tracker.multi_gate_crossings[0]
                ),
                "waypoint_credit_after_contact": bool(
                    trace["waypoint_credit_after_contact"]
                ),
                "geometry": geometry,
                "terminal_transition": last_transition,
            }
            rows.append(row)
    finally:
        active = None
    return rows, captured_tapes, default_joint_pos, physical_limits


def run_qualification(seeds: tuple[int, ...]) -> dict[str, Any]:
    """Capture source targets, replay them directly, and build the artifact."""
    from mjlab.tasks.registry import load_env_cfg

    import src.tasks  # noqa: F401 - populate registry
    from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

    root = Path(__file__).resolve().parents[2]
    source_code_revision = _git_revision(root)
    source_asset_revision = _external_asset_revision(Path(Z1_HAMMER_XML))
    source_cfg = load_env_cfg(SOURCE_TASK_ID, play=True)
    source_cfg.scene.num_envs = 1
    source_cfg.auto_reset = False
    projection = source_task_config_projection(source_cfg)
    source_rows, tapes, default, physical = _rollout_mode(
        source_cfg, seeds=seeds, mode="source"
    )
    if not tapes:
        raise RuntimeError("source capture produced no target tape")
    canonical_tape = tapes[seeds[0]]
    canonical_hash = canonical_sha256(canonical_tape.tolist())
    playback_lengths = {
        int(row["seed"]): int(row["playback_length"]) for row in source_rows
    }
    for row in source_rows:
        # Keep each independently measured hash; the paired decision will fail
        # if any source seed differs from the first seed's canonical tape.
        row["source_target_tape_sha256"] = canonical_sha256(
            tapes[int(row["seed"])].tolist()
        )

    scale = derive_scale_by_joint(canonical_tape, default)
    for row in source_rows:
        seed_tape = tapes[int(row["seed"])]
        normalized = normalized_action_for_target(seed_tape, default, scale)
        row["normalized_action_saturation_count"] = int(
            np.sum(np.any(np.abs(normalized) > 1.0, axis=1))
        )
        row["normalized_action_peak_abs"] = float(np.max(np.abs(normalized)))
    try:
        scheduled_tape = scheduled_replay_targets(
            canonical_tape,
            playback_length=playback_lengths[seeds[0]],
            hold_control_steps=POST_REFERENCE_HOLD_CONTROL_STEPS,
        )
    except ValueError as exc:
        reason = str(exc)
        unavailable_replay = [
            unavailable_rollout_record(
                seed=seed,
                mode="replay",
                reason=reason,
                tape_sha256=canonical_hash,
            )
            for seed in seeds
        ]
        return build_contract_payload(
            source_code_revision=source_code_revision,
            source_asset_revision=source_asset_revision,
            source_task_config_projection=projection,
            default_joint_pos_rad=default,
            physical_clip_rad=physical,
            source_target_tape_rad=canonical_tape,
            source_rows=source_rows,
            replay_rows=unavailable_replay,
        )
    replay_parent = load_env_cfg(SOURCE_TASK_ID, play=True)
    replay_cfg = configure_replay_cfg(
        replay_parent, scale_rad=scale, physical_clip_rad=physical
    )
    replay_rows, _, replay_default, replay_physical = _rollout_mode(
        replay_cfg,
        seeds=seeds,
        mode="replay",
        source_tape_rad=scheduled_tape,
        playback_lengths=playback_lengths,
    )
    if not np.allclose(replay_default, default, rtol=0.0, atol=1e-12):
        raise RuntimeError("source/replay default joint positions differ")
    if not np.allclose(replay_physical, physical, rtol=0.0, atol=1e-12):
        raise RuntimeError("source/replay physical limits differ")
    for row in replay_rows:
        row["source_target_tape_sha256"] = canonical_hash

    return build_contract_payload(
        source_code_revision=source_code_revision,
        source_asset_revision=source_asset_revision,
        source_task_config_projection=projection,
        default_joint_pos_rad=default,
        physical_clip_rad=physical,
        source_target_tape_rad=canonical_tape,
        source_rows=source_rows,
        replay_rows=replay_rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True, type=parse_seed_spec)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = run_qualification(args.seeds)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        payload = build_unavailable_diagnostic_payload(
            seeds=args.seeds, reason=reason
        )
    write_canonical_json(args.out, payload)
    print("seed source replay failures")
    for row in payload["per_seed_replay_rows"]:
        print(
            f"{row['seed']:4d} "
            f"{str(row['source']['passed']):>6s} "
            f"{str(row['replay']['passed']):>6s} "
            f"{len(row['failures']):8d}"
        )
    print(f"joint-position causal qualification: {payload['decision']}")
    return 0 if payload["decision"] == "PASS" else 1


def reduce_first_target_per_interval(
    targets_500hz: object, decimation: int = 10
) -> np.ndarray:
    """Keep the target applied at the first substep of each control interval."""
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation <= 0:
        raise ValueError("decimation must be a positive integer")
    targets = _finite_array(targets_500hz, name="targets_500hz", ndim=2)
    if targets.shape[0] == 0:
        raise ValueError("targets_500hz must contain at least one target")
    if targets.shape[0] % decimation != 0:
        raise ValueError("targets_500hz must contain complete control intervals")
    return targets[::decimation].copy()


def derive_scale_by_joint(
    tape: object,
    q_default: object,
    exploration_floor: float = 0.05,
    margin: float = 1.10,
) -> np.ndarray:
    """Derive per-joint action authority from target deviation plus exploration room."""
    target_tape = _finite_array(tape, name="tape", ndim=2)
    default = _finite_array(q_default, name="q_default", ndim=1)
    if target_tape.shape[0] == 0 or target_tape.shape[1] != default.shape[0]:
        raise ValueError("tape must be nonempty and match q_default width")
    if not math.isfinite(exploration_floor) or exploration_floor < 0.0:
        raise ValueError("exploration_floor must be finite and nonnegative")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    return margin * (np.max(np.abs(target_tape - default), axis=0) + exploration_floor)


def normalized_action_for_target(
    tape: object, q_default: object, scale: object
) -> np.ndarray:
    """Return absolute default-offset normalized actions for a target tape."""
    target_tape = _finite_array(tape, name="tape", ndim=2)
    default = _finite_array(q_default, name="q_default", ndim=1)
    scale_array = _finite_array(scale, name="scale", ndim=1)
    if (
        target_tape.shape[0] == 0
        or target_tape.shape[1] != default.shape[0]
        or scale_array.shape != default.shape
    ):
        raise ValueError("tape, q_default, and scale must have compatible widths")
    if np.any(scale_array <= 0.0):
        raise ValueError("scale must be positive")
    return (target_tape - default) / scale_array


def qualification_passes(
    rows: Sequence[dict[str, Any]], expected_seeds: tuple[int, ...] = tuple(range(1000, 1016))
) -> bool:
    """Return whether the replay rows are complete, ordered, and all explicitly passed."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    if len(rows) != len(expected_seeds):
        return False
    for row, expected_seed in zip(rows, expected_seeds, strict=True):
        if not isinstance(row, dict):
            return False
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed != expected_seed:
            return False
        if row.get("passed") is not True:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
