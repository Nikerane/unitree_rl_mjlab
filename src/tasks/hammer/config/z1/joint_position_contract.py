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
