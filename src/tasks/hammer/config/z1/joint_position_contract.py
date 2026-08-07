"""Fail-closed loader for the banked joint-position qualification artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import numpy as np

from evaluation.joint_position.qualify_joint_action import (
    derive_scale_by_joint,
    qualification_passes,
)


JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
PHYSICAL_CLIP_RAD = (
    (-2.61799, 2.61799),
    (0.0, 2.96706),
    (-2.87979, 0.0),
    (-1.51844, 1.51844),
    (-1.34390, 1.34390),
    (-2.79253, 2.79253),
)
REQUIRED_SEEDS = tuple(range(1000, 1016))
ACTION_SEMANTICS = "absolute_default_offset_joint_position"
SOURCE_TASK_ID = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4"
)
SOURCE_TASK_CONFIG_PROJECTION_FIELDS = frozenset(
    {
        "action",
        "observations",
        "rewards",
        "metrics",
        "events",
        "actuators",
        "timing",
        "reset",
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "action_semantics",
        "source_task_id",
        "source_code_revision",
        "source_asset_revision",
        "source_task_config_projection",
        "source_task_config_sha256",
        "joint_names",
        "actuator_names",
        "default_joint_pos_rad",
        "physical_clip_rad",
        "scale_rad",
        "physics_dt_s",
        "control_decimation",
        "post_reference_hold_control_steps",
        "seeds",
        "source_target_tape_rad",
        "source_target_tape_sha256",
        "per_seed_replay_rows",
        "decision",
        "payload_sha256",
    }
)

_TRACKABILITY_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "joint_names",
        "physics_dt_s",
        "control_decimation",
        "reward_manager_dt_s",
        "target_q90_cost",
        "q90_squared_error_rad2",
        "k_tt",
        "canonical_seed",
        "canonical_sample_count",
        "canonical_pre_dt_cumulative_cost",
        "canonical_returned_dose",
        "canonical_applied_target_tape_sha256",
        "canonical_squared_errors_rad2",
        "canonical_squared_errors_sha256",
        "repeatability_rows",
        "source_qualification_payload_sha256",
        "source_qualification_code_revision",
        "source_asset_revision",
        "calibration_code_revision",
        "decision",
        "payload_sha256",
    }
)
_TRACKABILITY_ROW_FIELDS = frozenset(
    {
        "seed",
        "sample_count",
        "terminal_transition_captured",
        "applied_target_tape_sha256",
        "squared_errors_sha256",
    }
)


@dataclass(frozen=True)
class JointPositionContract:
    """Validated, scientific-identity-bound joint-position qualification data."""

    source_task_id: str
    source_code_revision: str
    source_asset_revision: str
    source_task_config_projection: Mapping[str, Any]
    source_task_config_sha256: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    default_joint_pos_rad: np.ndarray
    physical_clip_rad: np.ndarray
    scale_rad: np.ndarray
    physics_dt_s: float
    control_decimation: int
    post_reference_hold_control_steps: int
    seeds: tuple[int, ...]
    source_target_tape_rad: np.ndarray
    source_target_tape_sha256: str
    per_seed_replay_rows: tuple[Mapping[str, Any], ...]
    payload_sha256: str


@dataclass(frozen=True)
class JointTrackabilityContract:
    """Validated calibration bound to one immutable joint-position contract."""

    joint_names: tuple[str, ...]
    physics_dt_s: float
    control_decimation: int
    reward_manager_dt_s: float
    target_q90_cost: float
    q90_squared_error_rad2: float
    k_tt: float
    canonical_seed: int
    canonical_sample_count: int
    canonical_pre_dt_cumulative_cost: float
    canonical_returned_dose: float
    canonical_applied_target_tape_sha256: str
    canonical_squared_errors_rad2: np.ndarray
    canonical_squared_errors_sha256: str
    repeatability_rows: tuple[Mapping[str, Any], ...]
    source_qualification_payload_sha256: str
    source_qualification_code_revision: str
    source_asset_revision: str
    calibration_code_revision: str
    payload_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_finite_json(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_is_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_finite_json(item) for key, item in value.items())
    return False


def _require_hash(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return value


def _require_revision(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be a 40-character lowercase Git commit")
    return value


def _contains_json_bool(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, list):
        return any(_contains_json_bool(item) for item in value)
    return False


def _finite_vector(value: object, *, name: str) -> np.ndarray:
    if _contains_json_bool(value):
        raise ValueError(f"{name} must not contain JSON booleans")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if vector.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain six finite values")
    return vector


def _finite_matrix(value: object, *, name: str) -> np.ndarray:
    if _contains_json_bool(value):
        raise ValueError(f"{name} must not contain JSON booleans")
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"{name} must be a nonempty [target, six-joint] matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _finite_clip_matrix(value: object) -> np.ndarray:
    if _contains_json_bool(value):
        raise ValueError("physical_clip_rad must not contain JSON booleans")
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("physical_clip_rad must be numeric") from exc
    if matrix.shape != (len(JOINT_NAMES), 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("physical_clip_rad must contain one finite [min, max] pair per joint")
    return matrix


def _require_exact_joint_order(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or tuple(value) != JOINT_NAMES:
        raise ValueError(f"{name} must be exactly {JOINT_NAMES}")
    return tuple(value)


def _require_task_config_projection(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("source_task_config_projection must be a JSON object")
    missing = sorted(SOURCE_TASK_CONFIG_PROJECTION_FIELDS - set(value))
    unexpected = sorted(set(value) - SOURCE_TASK_CONFIG_PROJECTION_FIELDS)
    if missing or unexpected:
        raise ValueError(
            "source_task_config_projection must contain exactly action, observations, "
            "rewards, metrics, events, actuators, timing, and reset; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if any(not isinstance(value[name], dict) for name in SOURCE_TASK_CONFIG_PROJECTION_FIELDS):
        raise ValueError("each source_task_config_projection section must be a JSON object")
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def load_joint_position_contract(path: str | Path) -> JointPositionContract:
    """Load only a complete PASS artifact whose content and causal tape are intact."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load joint-position contract: {exc}") from exc
    if not isinstance(payload, dict) or not _is_finite_json(payload):
        raise ValueError("contract must be a finite JSON object")
    missing = sorted(_REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError(f"contract is missing required fields: {missing}")
    unexpected = sorted(set(payload) - _REQUIRED_FIELDS)
    if unexpected:
        raise ValueError(f"contract contains unexpected fields: {unexpected}")

    supplied_payload_sha256 = _require_hash(payload["payload_sha256"], name="payload_sha256")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("payload_sha256")
    if _sha256(unsigned_payload) != supplied_payload_sha256:
        raise ValueError("payload_sha256 does not match canonical payload")
    if isinstance(payload["schema_version"], bool) or payload["schema_version"] != 1:
        raise ValueError("unsupported joint-position contract schema_version")
    if payload["action_semantics"] != ACTION_SEMANTICS:
        raise ValueError("unexpected action_semantics")
    if payload["decision"] != "PASS":
        raise ValueError("joint-position contract decision must be PASS")
    if payload["source_task_id"] != SOURCE_TASK_ID:
        raise ValueError(f"source_task_id must be exactly {SOURCE_TASK_ID}")
    source_code_revision = _require_revision(
        payload["source_code_revision"], name="source_code_revision"
    )
    source_asset_revision = _require_revision(
        payload["source_asset_revision"], name="source_asset_revision"
    )
    source_task_config_projection = _require_task_config_projection(
        payload["source_task_config_projection"]
    )
    source_task_config_sha256 = _require_hash(
        payload["source_task_config_sha256"], name="source_task_config_sha256"
    )
    if _sha256(source_task_config_projection) != source_task_config_sha256:
        raise ValueError(
            "source_task_config_sha256 does not match source_task_config_projection"
        )
    joint_names = _require_exact_joint_order(payload["joint_names"], name="joint_names")
    actuator_names = _require_exact_joint_order(payload["actuator_names"], name="actuator_names")

    default_joint_pos_rad = _finite_vector(payload["default_joint_pos_rad"], name="default_joint_pos_rad")
    physical_clip_rad = _finite_clip_matrix(payload["physical_clip_rad"])
    if np.any(physical_clip_rad[:, 0] >= physical_clip_rad[:, 1]):
        raise ValueError("physical_clip_rad must contain increasing physical intervals")
    if not np.allclose(physical_clip_rad, np.asarray(PHYSICAL_CLIP_RAD), rtol=0.0, atol=1e-12):
        raise ValueError("physical_clip_rad does not match the frozen Z1 XML limits")
    if np.any(default_joint_pos_rad < physical_clip_rad[:, 0]) or np.any(
        default_joint_pos_rad > physical_clip_rad[:, 1]
    ):
        raise ValueError("default_joint_pos_rad must lie inside physical_clip_rad")
    scale_rad = _finite_vector(payload["scale_rad"], name="scale_rad")
    if np.any(scale_rad <= 0.0):
        raise ValueError("scale_rad must be positive")

    physics_dt_s = payload["physics_dt_s"]
    if isinstance(physics_dt_s, bool) or not isinstance(physics_dt_s, (int, float)) or physics_dt_s != 0.002:
        raise ValueError("physics_dt_s must be exactly 0.002")
    if payload["control_decimation"] != 10:
        raise ValueError("control_decimation must be exactly 10")
    if payload["post_reference_hold_control_steps"] != 10:
        raise ValueError("post_reference_hold_control_steps must be exactly 10")
    if payload["seeds"] != list(REQUIRED_SEEDS):
        raise ValueError("seeds must be exactly 1000 through 1015")

    source_target_tape_rad = _finite_matrix(
        payload["source_target_tape_rad"], name="source_target_tape_rad"
    )
    if np.any(source_target_tape_rad < physical_clip_rad[:, 0]) or np.any(
        source_target_tape_rad > physical_clip_rad[:, 1]
    ):
        raise ValueError("source_target_tape_rad must lie inside physical_clip_rad")
    expected_scale_rad = derive_scale_by_joint(source_target_tape_rad, default_joint_pos_rad)
    if not np.allclose(scale_rad, expected_scale_rad, rtol=0.0, atol=1e-12):
        raise ValueError("scale_rad does not match the frozen derivation")
    source_target_tape_sha256 = _require_hash(
        payload["source_target_tape_sha256"], name="source_target_tape_sha256"
    )
    if _sha256(payload["source_target_tape_rad"]) != source_target_tape_sha256:
        raise ValueError("source_target_tape_sha256 does not match source_target_tape_rad")

    rows = payload["per_seed_replay_rows"]
    if not isinstance(rows, list) or not qualification_passes(rows, REQUIRED_SEEDS):
        raise ValueError("per_seed_replay_rows must contain one passing row per required seed")
    replay_applied_reference_sha256: str | None = None
    for index, row in enumerate(rows):
        if row.get("source_target_tape_sha256") != source_target_tape_sha256:
            raise ValueError("per-seed replay rows must bind the same source target tape")
        replay = row.get("replay")
        if not isinstance(replay, Mapping):
            raise ValueError("per-seed replay row must contain replay evidence")
        replay_applied_sha256 = _require_hash(
            replay.get("replay_applied_target_tape_sha256"),
            name="replay applied target-tape SHA-256",
        )
        if index == 0:
            replay_applied_reference_sha256 = replay_applied_sha256
        elif replay_applied_sha256 != replay_applied_reference_sha256:
            raise ValueError(
                "replay applied target-tape SHA-256 must be identical across seeds"
            )

    default_joint_pos_rad = _immutable_array(default_joint_pos_rad)
    physical_clip_rad = _immutable_array(physical_clip_rad)
    scale_rad = _immutable_array(scale_rad)
    source_target_tape_rad = _immutable_array(source_target_tape_rad)
    frozen_source_task_config_projection = _freeze_json(source_task_config_projection)
    frozen_rows = tuple(_freeze_json(row) for row in rows)

    return JointPositionContract(
        source_task_id=payload["source_task_id"],
        source_code_revision=source_code_revision,
        source_asset_revision=source_asset_revision,
        source_task_config_projection=frozen_source_task_config_projection,
        source_task_config_sha256=source_task_config_sha256,
        joint_names=joint_names,
        actuator_names=actuator_names,
        default_joint_pos_rad=default_joint_pos_rad,
        physical_clip_rad=physical_clip_rad,
        scale_rad=scale_rad,
        physics_dt_s=float(physics_dt_s),
        control_decimation=payload["control_decimation"],
        post_reference_hold_control_steps=payload["post_reference_hold_control_steps"],
        seeds=REQUIRED_SEEDS,
        source_target_tape_rad=source_target_tape_rad,
        source_target_tape_sha256=source_target_tape_sha256,
        per_seed_replay_rows=frozen_rows,
        payload_sha256=supplied_payload_sha256,
    )


def _finite_number(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and strictly positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return number


def load_joint_trackability_contract(
    path: str | Path, *, source_contract: JointPositionContract
) -> JointTrackabilityContract:
    """Load a complete PASS calibration bound to ``source_contract``.

    The calibration revision is evidence provenance, not a demand that later
    training use the same code revision.  The immutable source qualification
    digest, task revision, asset revision, timing, and joint order are the
    scientific-identity binding.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load joint-trackability contract: {exc}") from exc
    if not isinstance(payload, dict) or not _is_finite_json(payload):
        raise ValueError("trackability contract must be a finite JSON object")
    missing = sorted(_TRACKABILITY_REQUIRED_FIELDS - set(payload))
    unexpected = sorted(set(payload) - _TRACKABILITY_REQUIRED_FIELDS)
    if missing or unexpected:
        raise ValueError(
            "trackability contract fields are not exact; "
            f"missing={missing}, unexpected={unexpected}"
        )

    supplied_payload_sha256 = _require_hash(
        payload["payload_sha256"], name="payload_sha256"
    )
    unsigned_payload = dict(payload)
    unsigned_payload.pop("payload_sha256")
    if _sha256(unsigned_payload) != supplied_payload_sha256:
        raise ValueError("payload_sha256 does not match canonical trackability payload")
    if isinstance(payload["schema_version"], bool) or payload["schema_version"] != 1:
        raise ValueError("unsupported joint-trackability schema_version")
    if payload["decision"] != "PASS":
        raise ValueError("joint-trackability contract decision must be PASS")
    joint_names = _require_exact_joint_order(payload["joint_names"], name="joint_names")
    if joint_names != tuple(source_contract.joint_names):
        raise ValueError("trackability joint_names do not match source qualification")

    physics_dt_s = _finite_number(payload["physics_dt_s"], name="physics_dt_s", positive=True)
    if physics_dt_s != 0.002 or physics_dt_s != float(source_contract.physics_dt_s):
        raise ValueError("physics_dt_s does not match the source 500 Hz contract")
    control_decimation = payload["control_decimation"]
    if (
        isinstance(control_decimation, bool)
        or not isinstance(control_decimation, int)
        or control_decimation != 10
        or control_decimation != source_contract.control_decimation
    ):
        raise ValueError("control_decimation does not match the source contract")
    reward_manager_dt_s = _finite_number(
        payload["reward_manager_dt_s"], name="reward_manager_dt_s", positive=True
    )
    if reward_manager_dt_s != physics_dt_s * control_decimation or reward_manager_dt_s != 0.02:
        raise ValueError("reward_manager_dt_s must be exactly physics_dt_s * decimation = 0.02")

    target_q90_cost = _finite_number(
        payload["target_q90_cost"], name="target_q90_cost", positive=True
    )
    if target_q90_cost != 0.1:
        raise ValueError("target_q90_cost must be exactly 0.1")
    q90 = _finite_number(
        payload["q90_squared_error_rad2"],
        name="q90_squared_error_rad2",
        positive=True,
    )
    if q90 < 1e-8:
        raise ValueError("q90_squared_error_rad2 is below 1e-8")
    k_tt = _finite_number(payload["k_tt"], name="k_tt", positive=True)
    if not math.isclose(k_tt, target_q90_cost / q90, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("k_tt does not equal target_q90_cost / q90")

    canonical_seed = payload["canonical_seed"]
    canonical_sample_count = payload["canonical_sample_count"]
    if canonical_seed != REQUIRED_SEEDS[0] or isinstance(canonical_seed, bool):
        raise ValueError("canonical_seed must be 1000")
    if (
        isinstance(canonical_sample_count, bool)
        or not isinstance(canonical_sample_count, int)
        or canonical_sample_count <= 0
    ):
        raise ValueError("canonical_sample_count must be a positive integer")
    for qualified_row in source_contract.per_seed_replay_rows:
        qualified_replay = qualified_row.get("replay")
        qualified_count = (
            qualified_replay.get("target_count")
            if isinstance(qualified_replay, Mapping)
            else None
        )
        if (
            isinstance(qualified_count, bool)
            or not isinstance(qualified_count, int)
            or qualified_count != canonical_sample_count
        ):
            raise ValueError(
                "canonical_sample_count must equal every qualified replay target_count"
            )
    pre_dt_cost = _finite_number(
        payload["canonical_pre_dt_cumulative_cost"],
        name="canonical_pre_dt_cumulative_cost",
    )
    if pre_dt_cost < 0.0 or pre_dt_cost > 5.0:
        raise ValueError("canonical pre-dt cumulative cost must lie in [0, 5]")
    returned_dose = _finite_number(
        payload["canonical_returned_dose"], name="canonical_returned_dose"
    )
    if not math.isclose(
        returned_dose, pre_dt_cost * reward_manager_dt_s, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError("canonical_returned_dose does not equal pre-dt cost times 0.02")
    target_hash = _require_hash(
        payload["canonical_applied_target_tape_sha256"],
        name="canonical_applied_target_tape_sha256",
    )
    error_hash = _require_hash(
        payload["canonical_squared_errors_sha256"],
        name="canonical_squared_errors_sha256",
    )
    if _contains_json_bool(payload["canonical_squared_errors_rad2"]):
        raise ValueError("canonical_squared_errors_rad2 must not contain JSON booleans")
    try:
        canonical_errors = np.asarray(
            payload["canonical_squared_errors_rad2"], dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical_squared_errors_rad2 must be numeric") from exc
    if (
        canonical_errors.shape != (canonical_sample_count,)
        or not np.all(np.isfinite(canonical_errors))
        or np.any(canonical_errors < 0.0)
    ):
        raise ValueError("canonical squared errors must be finite, nonnegative, and complete")
    if _sha256(payload["canonical_squared_errors_rad2"]) != error_hash:
        raise ValueError("canonical squared-error hash does not match its samples")
    derived_q90 = float(np.quantile(canonical_errors, 0.90))
    if not math.isclose(q90, derived_q90, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("q90 does not match the canonical squared-error samples")
    derived_pre_dt_cost = k_tt * float(np.sum(canonical_errors, dtype=np.float64))
    if not math.isclose(pre_dt_cost, derived_pre_dt_cost, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("pre-dt cumulative cost does not match canonical samples")

    rows = payload["repeatability_rows"]
    if not isinstance(rows, list) or len(rows) != len(REQUIRED_SEEDS):
        raise ValueError("repeatability_rows must contain exactly seeds 1000 through 1015")
    for row, expected_seed in zip(rows, REQUIRED_SEEDS, strict=True):
        if not isinstance(row, dict) or set(row) != _TRACKABILITY_ROW_FIELDS:
            raise ValueError("each repeatability row must have the exact frozen schema")
        if (
            isinstance(row["seed"], bool)
            or row["seed"] != expected_seed
            or row["sample_count"] != canonical_sample_count
            or isinstance(row["sample_count"], bool)
            or row["terminal_transition_captured"] is not True
        ):
            raise ValueError("repeatability row seed, count, or terminal evidence is invalid")
        if _require_hash(
            row["applied_target_tape_sha256"], name="applied target tape hash"
        ) != target_hash or _require_hash(
            row["squared_errors_sha256"], name="squared errors hash"
        ) != error_hash:
            raise ValueError("repeatability rows must match the unique canonical trajectory")

    qualification_hash = _require_hash(
        payload["source_qualification_payload_sha256"],
        name="source_qualification_payload_sha256",
    )
    if qualification_hash != source_contract.payload_sha256:
        raise ValueError("trackability calibration is bound to another qualification payload")
    qualified_replay_hash = source_contract.per_seed_replay_rows[0]["replay"][
        "replay_applied_target_tape_sha256"
    ]
    if target_hash != qualified_replay_hash:
        raise ValueError("calibration target tape does not match the qualified applied replay")
    source_code_revision = _require_revision(
        payload["source_qualification_code_revision"],
        name="source_qualification_code_revision",
    )
    if source_code_revision != source_contract.source_code_revision:
        raise ValueError("source qualification code revision does not match")
    source_asset_revision = _require_revision(
        payload["source_asset_revision"], name="source_asset_revision"
    )
    if source_asset_revision != source_contract.source_asset_revision:
        raise ValueError("source asset revision does not match")
    calibration_code_revision = _require_revision(
        payload["calibration_code_revision"], name="calibration_code_revision"
    )

    return JointTrackabilityContract(
        joint_names=joint_names,
        physics_dt_s=physics_dt_s,
        control_decimation=control_decimation,
        reward_manager_dt_s=reward_manager_dt_s,
        target_q90_cost=target_q90_cost,
        q90_squared_error_rad2=q90,
        k_tt=k_tt,
        canonical_seed=canonical_seed,
        canonical_sample_count=canonical_sample_count,
        canonical_pre_dt_cumulative_cost=pre_dt_cost,
        canonical_returned_dose=returned_dose,
        canonical_applied_target_tape_sha256=target_hash,
        canonical_squared_errors_rad2=_immutable_array(canonical_errors),
        canonical_squared_errors_sha256=error_hash,
        repeatability_rows=tuple(_freeze_json(row) for row in rows),
        source_qualification_payload_sha256=qualification_hash,
        source_qualification_code_revision=source_code_revision,
        source_asset_revision=source_asset_revision,
        calibration_code_revision=calibration_code_revision,
        payload_sha256=supplied_payload_sha256,
    )
