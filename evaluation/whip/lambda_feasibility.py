#!/usr/bin/env python
"""CPU-first fixed-impedance Lambda feasibility evaluator.

The experiment contract lives at
``docs/superpowers/specs/2026-07-29-lambda-feasibility-design.md``.

This module deliberately keeps the offline trace classifier and saved-tape
inventory importable without constructing mjlab. The live Stage-0 runner is
added below those pure functions and must use only the ordinary DiffIK
``env.step(action)`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import argparse
import queue
import subprocess
import time
import traceback
from typing import Any

import numpy as np


LEGACY_WHIP_FILES = (
    "whip_delta015.json",
    "whip_delta015_cma.json",
    "whip_delta030.json",
    "whip_delta045.json",
)
ALLOWED_DELTAS = (0.15, 0.30, 0.45)
NONFACE_FORCE_TOL = 1e-6
POST_TAPE_ZERO_ACTIONS = 10
RELEASE_CONFIRM_SAMPLES = (2, 5, 10)
RESET_BANK_SEED = 2026072901
RESET_BANK_SIZE = 16
HARDWARE_QVEL_LIMIT_RAD_S = 3.1415
QVEL_MARGIN_FRACTION = 0.95

_RESULT_DEFAULTS: dict[str, object] = {
    "source_kind": None,
    "source_id": None,
    "source_path": None,
    "source_field": None,
    "source_aliases": [],
    "delta": None,
    "action_digest": None,
    "reset_digest": None,
    "reset_role": None,
    "solref_scale": None,
    "status": None,
    "error": None,
    "worker_exitcode": None,
    "worker_pid": None,
    "wall_time_s": None,
    "contact_seen": False,
    "accepted_onset_index": None,
    "release_index": None,
    "release_index_by_confirm": {},
    "lobe_end_index": None,
    "release_by_28ms": False,
    "lobe_completed": False,
    "no_recontact_50ms": False,
    "recontact_count_50ms": None,
    "face_only": False,
    "nonface_force_tolerance": NONFACE_FORCE_TOL,
    "nonface_max_efc_force": None,
    "productive_30mm_50ms": False,
    "qvel_legal": None,
    "qvel_legal_95pct": None,
    "qvel_peak_rad_s": None,
    "qvel_trace_rad_s": [],
    "qvel_pre_trace_rad_s": [],
    "qvel_post_trace_rad_s": [],
    "qpos_legal": None,
    "qpos_trace_rad": [],
    "qpos_pre_trace_rad": [],
    "qpos_post_trace_rad": [],
    "actuator_saturation_fraction": None,
    "actuator_saturation_perjoint": [],
    "head_pose_m_quat": [],
    "head_position_m": [],
    "head_twist_linear_angular": [],
    "realized_action_tape": [],
    "realized_action_digest": None,
    "nail_depth_m": [],
    "raw_face_contact": [],
    "nonface_efc_force_trace": [],
    "exact_contrib_perjoint_trace": [],
    "shipped_rolling_perjoint_trace": [],
    "shipped_contrib_perjoint_trace": [],
    "object_axial_force_n_trace": [],
    "object_axial_contrib_n_s_trace": [],
    "object_total_contrib_n_s_trace": [],
    "tracker_started_trace": [],
    "quality": None,
    "quality_valid": None,
    "quality_overflow": None,
    "quality_contact_point_w": [],
    "quality_error_m": None,
    "quality_first_contact_time_s": None,
    "quality_normal_axiality": None,
    "eligible": False,
    "binding_joint_index": None,
    "lobe_fraction": None,
    "lobe_fraction_60": None,
    "lobe_fraction_80": None,
    "rho_ship50": None,
    "rho_row26": None,
    "rho_row28": None,
    "rho_row50": None,
    "rho_row50_onset": None,
    "rho_row50_sliding_peak": None,
    "row26_perjoint": [],
    "row28_perjoint": [],
    "row50_perjoint": [],
    "row50_onset_perjoint": [],
    "row50_sliding_peak_perjoint": [],
    "row50_sliding_start_perjoint": [],
    "full_lobe_perjoint": [],
    "ship50_perjoint": [],
    "binding_class": "ineligible",
    "contact_duration_substeps": None,
    "contact_duration_ms": None,
    "first_event_axial_force_peak_n": None,
    "first_event_axial_force_mean_n": None,
    "object_axial_impulse_total": None,
    "object_total_impulse_total": None,
    "exact_contrib_max_n_m_s": None,
    "exact_total_perjoint": [],
    "production_peak_perjoint": [],
    "max_exact_off_raw_contact": None,
    "exact_live_contact_substeps": None,
    "object_live_contact_substeps": None,
    "aligned_live_contact_substeps": None,
    "object_axial_live_contact_substeps": None,
    "aligned_axial_live_contact_substeps": None,
    "non_axial_contact": False,
    "contact_signal_alignment_failure": False,
    "n_substeps": None,
    "n_state_samples": None,
    "post_tape_zero_actions": POST_TAPE_ZERO_ACTIONS,
    "impossible_success": False,
    "lambda_dead": False,
    "deterministic_trace_digest": None,
    "qualification": None,
    "parity_passed": None,
    "solref_comparison": None,
    "compiled_solref": {},
}

_DETERMINISTIC_TRACE_FIELDS = (
    "status",
    "contact_seen",
    "accepted_onset_index",
    "release_index",
    "lobe_end_index",
    "qvel_trace_rad_s",
    "qpos_trace_rad",
    "head_pose_m_quat",
    "head_position_m",
    "head_twist_linear_angular",
    "nail_depth_m",
    "raw_face_contact",
    "nonface_efc_force_trace",
    "exact_contrib_perjoint_trace",
    "shipped_rolling_perjoint_trace",
    "shipped_contrib_perjoint_trace",
    "object_axial_force_n_trace",
    "object_axial_contrib_n_s_trace",
    "object_total_contrib_n_s_trace",
    "tracker_started_trace",
    "quality",
    "quality_valid",
    "quality_overflow",
    "quality_contact_point_w",
    "quality_error_m",
    "quality_first_contact_time_s",
    "quality_normal_axiality",
    "row26_perjoint",
    "row28_perjoint",
    "row50_perjoint",
    "row50_onset_perjoint",
    "row50_sliding_peak_perjoint",
    "full_lobe_perjoint",
    "ship50_perjoint",
    "production_peak_perjoint",
    "realized_action_tape",
    "compiled_solref",
)

AMENDMENT_ID = "A1"
SUPERSEDES_FAILED_OUTPUT_SHA256 = (
    "6755cdefb74aebfb704f2271af4b020dcfc1d83bba045c5b1a997922d7b21414"
)
SUPERSEDES_FAILED_INPUT_SHA256 = (
    "86deebf186ef8360f515574b388cd52a834369f4d25f6246eaed4abfd848d588"
)
_RAW_PHYSICS_IDENTITY_FIELDS = (
    "source_kind",
    "source_id",
    "source_path",
    "source_field",
    "source_aliases",
    "delta",
    "action_digest",
    "reset_digest",
    "reset_role",
    "solref_scale",
)
_RAW_PHYSICS_TRACE_FIELDS = (
    "qvel_trace_rad_s",
    "qvel_pre_trace_rad_s",
    "qvel_post_trace_rad_s",
    "qpos_trace_rad",
    "qpos_pre_trace_rad",
    "qpos_post_trace_rad",
    "head_pose_m_quat",
    "head_position_m",
    "head_twist_linear_angular",
    "nail_depth_m",
    "raw_face_contact",
    "nonface_efc_force_trace",
    "exact_contrib_perjoint_trace",
    "shipped_rolling_perjoint_trace",
    "shipped_contrib_perjoint_trace",
    "object_axial_force_n_trace",
    "object_axial_contrib_n_s_trace",
    "object_total_contrib_n_s_trace",
    "tracker_started_trace",
    "quality",
    "quality_valid",
    "quality_overflow",
    "quality_contact_point_w",
    "quality_error_m",
    "quality_first_contact_time_s",
    "quality_normal_axiality",
    "realized_action_tape",
    "realized_action_digest",
    "compiled_solref",
    "n_substeps",
    "n_state_samples",
)


class BindingClass(str, Enum):
    """Mutually exclusive interpretation of the shipped and exact ratios."""

    INELIGIBLE = "ineligible"
    NO_BIND = "no_bind"
    SHIPPED_ONLY = "shipped_only"
    EXACT_ONLY = "exact_only"
    DUAL_BIND = "dual_bind"
    TIME_BOUNDARY = "time_discretization_boundary"
    SHIPPED_TIME_BOUNDARY = "shipped_time_discretization_boundary"


@dataclass(frozen=True)
class SavedTape:
    source_path: Path
    source_field: str
    delta: float
    actions: np.ndarray
    digest: str
    aliases: tuple[tuple[str, str], ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def reset_bank_digest(value: object) -> str:
    """Canonical SHA-256 used for reset states and reset-bank envelopes."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _new_result(
    identity: dict[str, object],
    *,
    status: str,
    error: str | None = None,
) -> dict[str, object]:
    """Return the uniform record-all schema for every replay outcome."""
    if status not in {"ok", "no_contact", "error", "timeout", "crash"}:
        raise ValueError(f"unknown replay status {status!r}")
    result = json.loads(json.dumps(_RESULT_DEFAULTS))
    unknown = sorted(set(identity) - set(result))
    if unknown:
        raise ValueError(f"unknown result identity fields: {unknown}")
    result.update(identity)
    result["status"] = status
    result["error"] = error
    result["binding_class"] = BindingClass.INELIGIBLE.value
    return result


def deterministic_trace_digest(result: dict[str, object]) -> str:
    """Digest only deterministic physics/classifier fields, not worker metadata."""
    payload = {key: result.get(key) for key in _DETERMINISTIC_TRACE_FIELDS}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _raw_physics_digest(result: dict[str, object]) -> str:
    payload = {key: result.get(key) for key in _RAW_PHYSICS_TRACE_FIELDS}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def raw_physics_equivalence_report(
    previous: list[dict[str, object]],
    current: list[dict[str, object]],
) -> dict[str, object]:
    """Compare only identities and raw physics, excluding amended classifications."""

    def indexed(
        rows: list[dict[str, object]],
    ) -> dict[str, tuple[dict[str, object], str]]:
        result: dict[str, tuple[dict[str, object], str]] = {}
        for row in rows:
            identity = {
                key: row.get(key) for key in _RAW_PHYSICS_IDENTITY_FIELDS
            }
            key = _canonical_json_bytes(identity).decode("utf-8")
            if key in result:
                raise ValueError(f"duplicate raw-physics identity: {identity}")
            result[key] = (identity, _raw_physics_digest(row))
        return result

    before = indexed(previous)
    after = indexed(current)
    identity_match = set(before) == set(after)
    mismatches: list[dict[str, object]] = []
    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        if old is None or new is None or old[1] != new[1]:
            identity = old[0] if old is not None else new[0]  # type: ignore[index]
            mismatches.append(
                {
                    "identity": identity,
                    "previous_raw_sha256": None if old is None else old[1],
                    "current_raw_sha256": None if new is None else new[1],
                }
            )
    return {
        "schema": "lambda-feasibility-raw-equivalence-v1",
        "amendment_id": AMENDMENT_ID,
        "supersedes_failed_output_sha256": (
            SUPERSEDES_FAILED_OUTPUT_SHA256
        ),
        "previous_result_n": len(previous),
        "current_result_n": len(current),
        "identity_match": identity_match,
        "mismatch_n": len(mismatches),
        "mismatches": mismatches,
        "raw_physics_fields": list(_RAW_PHYSICS_TRACE_FIELDS),
    }


def validate_reset_bank(
    bank: list[dict[str, object]],
    *,
    fixed_reset: dict[str, object],
    expected_seed: int = RESET_BANK_SEED,
    expected_size: int = RESET_BANK_SIZE,
    forbidden_digests: set[str] | None = None,
) -> list[dict[str, object]]:
    """Validate the frozen training-distribution reset bank fail-closed."""
    if len(bank) != expected_size:
        raise ValueError(f"reset bank must contain exactly {expected_size} states")
    forbidden = forbidden_digests or set()
    try:
        fixed_realized = fixed_reset["reset_state"]["realized"]  # type: ignore[index]
        fixed_robot_pos = np.asarray(
            fixed_realized["robot_joint_pos"], dtype=np.float64  # type: ignore[index]
        )
        fixed_nail_pos = np.asarray(
            fixed_realized["nail_joint_pos"], dtype=np.float64  # type: ignore[index]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed fixed reset contract") from exc
    if (
        fixed_robot_pos.ndim != 1
        or fixed_nail_pos.ndim != 1
        or not np.isfinite(fixed_robot_pos).all()
        or not np.isfinite(fixed_nail_pos).all()
    ):
        raise ValueError("fixed reset realized state must be finite and one-dimensional")
    seen: set[str] = set()
    validated: list[dict[str, object]] = []
    for index, row in enumerate(bank):
        try:
            version = int(row["reset_contract_version"])
            state = row["reset_state"]
            inputs = state["regeneration_inputs"]  # type: ignore[index]
            realized = state["realized"]  # type: ignore[index]
            seed = int(inputs["reset_rng_seed"])  # type: ignore[index]
            call = int(inputs["reset_rng_call_index"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed reset state at index {index}") from exc
        if version != 1:
            raise ValueError(f"reset contract version drift at index {index}")
        if seed != expected_seed:
            raise ValueError(
                f"reset bank seed mismatch at index {index}: {seed} != {expected_seed}"
            )
        if call != index:
            raise ValueError(
                f"reset call index mismatch at index {index}: {call} != {index}"
            )
        if not isinstance(realized, dict):
            raise ValueError(f"malformed realized reset state at index {index}")
        expected_keys = {
            "robot_joint_pos",
            "robot_joint_vel",
            "nail_joint_pos",
            "nail_joint_vel",
        }
        if set(realized) != expected_keys:
            raise ValueError(
                f"reset realized channels mismatch at index {index}: "
                f"{sorted(realized)}"
            )
        flat: list[float] = []
        for key in sorted(expected_keys):
            values = realized[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"reset {index} channel {key} must be non-empty")
            flat.extend(float(value) for value in values)
        if not np.isfinite(np.asarray(flat, dtype=np.float64)).all():
            raise ValueError(f"reset {index} contains non-finite state")
        robot_pos = np.asarray(realized["robot_joint_pos"], dtype=np.float64)
        robot_vel = np.asarray(realized["robot_joint_vel"], dtype=np.float64)
        nail_pos = np.asarray(realized["nail_joint_pos"], dtype=np.float64)
        nail_vel = np.asarray(realized["nail_joint_vel"], dtype=np.float64)
        if robot_pos.shape != fixed_robot_pos.shape:
            raise ValueError(f"reset {index} robot joint shape drift")
        if np.any(np.abs(robot_pos - fixed_robot_pos) > 0.05 + 1e-7):
            raise ValueError(
                f"reset {index} exceeds the frozen +/-0.05 rad position range"
            )
        if np.any(np.abs(robot_vel) > 1e-9) or np.any(np.abs(nail_vel) > 1e-9):
            raise ValueError(f"reset {index} velocity must remain zero")
        if (
            nail_pos.shape != fixed_nail_pos.shape
            or not np.allclose(nail_pos, fixed_nail_pos, rtol=0.0, atol=1e-9)
        ):
            raise ValueError(f"reset {index} nail state changed")
        digest = reset_bank_digest(realized)
        if row.get("reset_state_digest") != digest:
            raise ValueError(f"reset digest mismatch at index {index}")
        if digest in seen:
            raise ValueError(f"duplicate reset digest at index {index}: {digest}")
        if digest in forbidden:
            raise ValueError(f"reset digest collides with forbidden bank: {digest}")
        seen.add(digest)
        validated.append(row)
    return validated


def write_frozen_json(path: Path, payload: object) -> str:
    """Write canonical JSON plus SHA sidecar exactly once, then chmod both 0444."""
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
        with sidecar.open("x", encoding="utf-8") as handle:
            handle.write(f"{digest}  {target.name}\n")
    except Exception:
        # Preserve the first failure as evidence rather than silently replacing
        # it. A partial target is intentionally not unlinked here.
        raise
    os.chmod(target, 0o444)
    os.chmod(sidecar, 0o444)
    return digest


def _assert_authoritative_outside_repositories(
    out: Path, *, code_repo: Path, asset_repo: Path
) -> None:
    """Prevent evidence creation from dirtying either provenance repository."""
    target = Path(out).resolve()
    repositories = (Path(code_repo).resolve(), Path(asset_repo).resolve())
    if any(target == repo or target.is_relative_to(repo) for repo in repositories):
        raise ValueError(
            "authoritative --out must be outside the code and asset repositories"
        )


def first_release(
    contact: np.ndarray, *, onset: int, confirm_off: int = 5
) -> int | None:
    """Return the first off sample of the first confirmed post-onset release."""
    mask = np.asarray(contact, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"contact must be one-dimensional, got {mask.shape}")
    if not 0 <= onset < len(mask):
        raise ValueError(f"onset {onset} outside contact trace of length {len(mask)}")
    if confirm_off < 1:
        raise ValueError(f"confirm_off must be >=1, got {confirm_off}")
    if not mask[onset]:
        raise ValueError(f"accepted onset {onset} is not in contact")
    for start in range(onset + 1, len(mask) - confirm_off + 1):
        if not mask[start : start + confirm_off].any():
            return start
    return None


def first_impact_lobe_end(
    contact: np.ndarray,
    downward_speed: np.ndarray,
    *,
    onset: int,
    stop_exclusive: int | None = None,
) -> int | None:
    """First in-contact sample at/after onset with non-positive downward speed."""
    mask = np.asarray(contact, dtype=bool)
    speed = np.asarray(downward_speed, dtype=np.float64)
    if mask.ndim != 1 or speed.ndim != 1 or mask.shape != speed.shape:
        raise ValueError(
            f"contact and downward_speed must be matching 1-D arrays, got "
            f"{mask.shape} and {speed.shape}"
        )
    if not 0 <= onset < len(mask):
        raise ValueError(f"onset {onset} outside trace of length {len(mask)}")
    if not mask[onset]:
        raise ValueError(f"accepted onset {onset} is not in contact")
    if not np.isfinite(speed).all():
        raise ValueError("downward_speed contains non-finite values")
    stop = len(mask) if stop_exclusive is None else int(stop_exclusive)
    if not onset < stop <= len(mask):
        raise ValueError(
            f"stop_exclusive must be in ({onset}, {len(mask)}], got {stop}"
        )
    for idx in range(onset, stop):
        if mask[idx] and speed[idx] <= 0.0:
            return idx
    return None


def onset_window_impulses(
    contributions: np.ndarray,
    *,
    onset: int,
    lengths: tuple[int, ...] = (13, 14, 25),
) -> dict[int, np.ndarray]:
    """Sum per-substep per-joint contributions in frozen onset-anchored windows."""
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"contributions must have shape [substep,joint], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("contributions contain non-finite values")
    if np.any(values < 0.0):
        raise ValueError("contributions must be non-negative")
    if not 0 <= onset < values.shape[0]:
        raise ValueError(f"onset {onset} outside trace of length {values.shape[0]}")
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError(f"window lengths must all be positive, got {lengths}")
    longest = max(lengths)
    if onset + longest > values.shape[0]:
        raise ValueError(
            f"trace too short for {longest}-substep window: onset={onset}, "
            f"length={values.shape[0]}"
        )
    return {
        length: values[onset : onset + length].sum(axis=0)
        for length in lengths
    }


def sliding_window_impulses(
    contributions: np.ndarray, *, length: int = 25
) -> dict[str, np.ndarray]:
    """Return per-joint sliding-window peaks with earliest-start tie breaking."""
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"contributions must have shape [substep,joint], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("contributions contain non-finite values")
    if np.any(values < 0.0):
        raise ValueError("contributions must be non-negative")
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    if values.shape[0] < length:
        raise ValueError(
            f"trace too short for {length}-substep sliding window: "
            f"length={values.shape[0]}"
        )
    cumulative = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0)],
        axis=0,
    )
    windows = cumulative[length:] - cumulative[:-length]
    starts = np.argmax(windows, axis=0).astype(np.int64)
    peaks = windows[starts, np.arange(values.shape[1])]
    return {
        "peak_perjoint": peaks,
        "start_perjoint": starts,
    }


def assert_exact_zero_off_raw_contact(
    raw_face_contact: np.ndarray,
    exact_contrib: np.ndarray,
    *,
    tol: float = 1e-8,
) -> float:
    """Require the exact row contribution to be zero without raw face contact."""
    face = np.asarray(raw_face_contact, dtype=bool)
    exact = np.asarray(exact_contrib, dtype=np.float64)
    if face.ndim != 1 or exact.ndim != 2 or exact.shape[0] != len(face):
        raise ValueError("raw face mask and exact contribution shape mismatch")
    if not np.isfinite(exact).all() or np.any(exact < 0.0):
        raise ValueError("exact contribution must be finite and non-negative")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and non-negative, got {tol}")
    off = exact[~face]
    peak = float(np.abs(off).max()) if off.size else 0.0
    if peak > tol:
        raise RuntimeError(
            "contact-row impulse is nonzero without raw face contact: "
            f"peak={peak} > tolerance={tol}"
        )
    return peak


def strict_qvel_legal(qvel: np.ndarray, *, rail: float) -> tuple[bool, float]:
    """Check every sample/joint; a non-finite trace is automatically illegal."""
    values = np.asarray(qvel, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"qvel must have shape [substep,joint], got {values.shape}")
    if rail <= 0.0 or not np.isfinite(rail):
        raise ValueError(f"rail must be finite and positive, got {rail}")
    if not np.isfinite(values).all():
        return False, float("nan")
    peak = float(np.abs(values).max(initial=0.0))
    return peak <= rail, peak


def strict_qpos_legal(
    qpos: np.ndarray, *, limits: np.ndarray, tol: float = 1e-7
) -> bool:
    """Check every sample and every named arm joint against its hard limits."""
    values = np.asarray(qpos, dtype=np.float64)
    bounds = np.asarray(limits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"qpos must have shape [substep,joint], got {values.shape}")
    if bounds.shape != (values.shape[1], 2):
        raise ValueError(
            f"limits must have shape {(values.shape[1], 2)}, got {bounds.shape}"
        )
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("joint-position limits must be finite and ordered")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and non-negative, got {tol}")
    return bool(
        np.isfinite(values).all()
        and np.all(values >= bounds[:, 0] - tol)
        and np.all(values <= bounds[:, 1] + tol)
    )


def actuator_saturation_summary(
    effort: np.ndarray,
    *,
    limits: np.ndarray,
    atol: float = 1e-5,
) -> dict[str, object]:
    """Full-horizon occupancy of the native joint actuator-force clamp."""
    values = np.asarray(effort, dtype=np.float64)
    bounds = np.asarray(limits, dtype=np.float64)
    if values.ndim != 2 or bounds.shape != (values.shape[1],):
        raise ValueError("effort must be [sample,joint] and limits one per joint")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(bounds).all()
        or np.any(bounds <= 0.0)
    ):
        raise ValueError("effort and limits must be finite; limits positive")
    if atol < 0.0 or not np.isfinite(atol):
        raise ValueError("atol must be finite and non-negative")
    saturated = np.abs(values) >= (bounds[None, :] - atol)
    return {
        "overall": float(saturated.mean()) if saturated.size else 0.0,
        "perjoint": saturated.mean(axis=0).astype(np.float64).tolist(),
    }


def assert_deterministic_replay(
    first: dict[str, object], second: dict[str, object]
) -> str:
    """Require same action/reset/config identity and identical CPU trace digest."""
    identity_fields = (
        "source_kind",
        "source_id",
        "delta",
        "action_digest",
        "reset_digest",
        "reset_role",
        "solref_scale",
    )
    if any(first.get(key) != second.get(key) for key in identity_fields):
        raise RuntimeError("deterministic replay identity mismatch")
    digest_a = deterministic_trace_digest(first)
    digest_b = deterministic_trace_digest(second)
    if digest_a != digest_b:
        raise RuntimeError(
            f"deterministic replay mismatch: {digest_a} != {digest_b}"
        )
    return digest_a


def _determinism_qualification(
    first: dict[str, object], second: dict[str, object]
) -> dict[str, object]:
    """Turn every fixed-baseline replay pair into a fail-closed qualification."""
    identity = {
        key: first.get(key)
        for key in (
            "source_path",
            "source_field",
            "source_aliases",
            "delta",
            "action_digest",
            "reset_digest",
            "reset_role",
            "solref_scale",
        )
    }
    identity.update(
        {
            "source_kind": "qualification",
            "source_id": f"determinism:{first.get('source_id')}",
        }
    )
    qualification = _new_result(identity, status="error")
    qualification["qualification"] = "deterministic_replay"
    complete_statuses = {"ok", "no_contact"}
    first_status = str(first.get("status"))
    second_status = str(second.get("status"))
    digests = (
        first.get("realized_action_digest"),
        second.get("realized_action_digest"),
    )
    tapes = (
        first.get("realized_action_tape"),
        second.get("realized_action_tape"),
    )
    if (
        first_status not in complete_statuses
        or second_status not in complete_statuses
        or not all(isinstance(value, str) and len(value) == 64 for value in digests)
        or not all(isinstance(value, list) and len(value) == 30 for value in tapes)
    ):
        qualification["error"] = (
            "deterministic replay pair incomplete: "
            f"first={first_status}, repeat={second_status}, "
            f"first_actions={len(tapes[0]) if isinstance(tapes[0], list) else None}, "
            f"repeat_actions={len(tapes[1]) if isinstance(tapes[1], list) else None}"
        )
        return qualification
    try:
        digest = assert_deterministic_replay(first, second)
    except Exception as exc:
        qualification["error"] = str(exc)
        return qualification
    qualification["status"] = "ok"
    qualification["error"] = None
    qualification["deterministic_trace_digest"] = digest
    return qualification


def _reference_calibration_qualification(
    reference: dict[str, object],
) -> dict[str, object]:
    """Qualify the known reference without outcome-derived numeric bands."""
    identity = {
        key: reference.get(key)
        for key in (
            "source_path",
            "source_field",
            "source_aliases",
            "delta",
            "action_digest",
            "reset_digest",
            "reset_role",
            "solref_scale",
        )
    }
    identity.update(
        {
            "source_kind": "qualification",
            "source_id": "reference-calibration",
        }
    )
    qualification = _new_result(identity, status="error")
    qualification["qualification"] = "reference_calibration"
    qualification["observed_reference_class"] = reference.get("binding_class")
    failures: list[str] = []
    if reference.get("status") != "ok":
        failures.append(f"status={reference.get('status')}")
    if not bool(reference.get("contact_seen")):
        failures.append("no raw face contact")
    if reference.get("accepted_onset_index") is None:
        failures.append("no accepted onset")
    if not bool(reference.get("productive_30mm_50ms")):
        failures.append("not productive")
    if not bool(reference.get("qvel_legal")):
        failures.append("qvel illegal")
    if not bool(reference.get("qpos_legal")):
        failures.append("qpos illegal")
    if int(reference.get("aligned_live_contact_substeps") or 0) < 1:
        failures.append("contact instruments not aligned/live")
    if int(reference.get("aligned_axial_live_contact_substeps") or 0) < 1:
        failures.append("reference has no aligned axial contact impulse")
    if not isinstance(reference.get("solref_comparison"), dict):
        failures.append("fixed reference solref*2 calibration missing")
    if not bool(reference.get("quality_valid")):
        failures.append("first-contact quality invalid")
    if bool(reference.get("quality_overflow")):
        failures.append("first-contact quality overflow")
    if bool(reference.get("release_by_28ms")):
        failures.append(
            "reference no longer exhibits the required broad-contact calibration"
        )
    if bool(reference.get("eligible")):
        failures.append(
            "reference unexpectedly classified as a completed short impact; "
            "manual review required"
        )
    if failures:
        qualification["error"] = "; ".join(failures)
        return qualification
    qualification["status"] = "ok"
    qualification["error"] = None
    return qualification


def assert_ordinary_shadow_prefix_parity(
    ordinary: dict[str, object],
    shadow: dict[str, object],
    *,
    atol: float = 1e-9,
) -> None:
    """Require action/state parity through the ordinary task's success step."""
    success = ordinary.get("success_control_index")
    if success is None:
        raise RuntimeError("ordinary rollout never reached the original success")
    success_i = int(success)
    if success_i < 0:
        raise RuntimeError(f"invalid ordinary success index {success_i}")
    for key in ("actions",):
        a = np.asarray(ordinary[key], dtype=np.float64)[: success_i + 1]
        b = np.asarray(shadow[key], dtype=np.float64)[: success_i + 1]
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=atol):
            raise RuntimeError(f"ordinary-shadow prefix parity failed for {key}")
    state_stop = int(
        ordinary.get("success_state_index", success_i + 1)
    ) + 1
    for key in ("qpos", "qvel", "head_position", "nail_depth"):
        a = np.asarray(ordinary[key], dtype=np.float64)[:state_stop]
        b = np.asarray(shadow[key], dtype=np.float64)[:state_stop]
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=atol):
            raise RuntimeError(f"ordinary-shadow prefix parity failed for {key}")


def _replay_worker(result_queue: Any, request: dict[str, object]) -> None:
    try:
        result_queue.put(_execute_replay_request(request))
    except Exception as exc:
        identity = dict(request.get("identity", {}))
        result_queue.put(
            _new_result(
                identity,
                status="error",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )


def run_isolated_replay(
    request: dict[str, object],
    *,
    timeout_s: float,
    context: Any | None = None,
) -> dict[str, object]:
    """Run one replay in a fresh spawn child and record every exit mode."""
    if timeout_s <= 0.0 or not np.isfinite(timeout_s):
        raise ValueError(f"timeout_s must be finite and positive, got {timeout_s}")
    identity = dict(request.get("identity", {}))
    ctx = context if context is not None else mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_replay_worker, args=(result_queue, request))
    started = time.perf_counter()
    process.start()
    payload: dict[str, object] | None = None
    try:
        payload = result_queue.get(timeout=timeout_s)
    except queue.Empty:
        pass
    if payload is None:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            payload = _new_result(
                identity,
                status="timeout",
                error=f"replay exceeded timeout_s={timeout_s}",
            )
        else:
            process.join(timeout=5)
            payload = _new_result(
                identity,
                status="crash",
                error=f"worker exited without payload: exitcode={process.exitcode}",
            )
    else:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            payload = _new_result(
                identity,
                status="timeout",
                error="worker returned payload but did not exit",
            )
    payload["worker_exitcode"] = process.exitcode
    payload["worker_pid"] = process.pid
    payload["wall_time_s"] = time.perf_counter() - started
    return payload


def stage0_action_sequence(actions: np.ndarray) -> np.ndarray:
    """Return the frozen 30-command tape followed by ten zero commands."""
    tape = np.asarray(actions, dtype=np.float32)
    if tape.shape != (30, 3):
        raise ValueError(f"Stage-0 tape must have shape (30, 3), got {tape.shape}")
    if not np.isfinite(tape).all() or np.any(np.abs(tape) > 1.0):
        raise ValueError("Stage-0 actions must be finite and inside [-1,1]")
    return np.concatenate(
        [tape, np.zeros((POST_TAPE_ZERO_ACTIONS, 3), dtype=np.float32)],
        axis=0,
    )


def contact_signal_liveness(
    *,
    contact: np.ndarray,
    exact_contrib: np.ndarray,
    object_total_contrib: np.ndarray,
    object_axial_contrib: np.ndarray,
    onset: int,
    stop_exclusive: int | None = None,
    tol: float = 1e-12,
) -> dict[str, int | bool]:
    """Diagnose instrument alignment separately from physical force direction."""
    face = np.asarray(contact, dtype=bool)
    exact = np.asarray(exact_contrib, dtype=np.float64)
    object_total = np.asarray(object_total_contrib, dtype=np.float64)
    object_axial = np.asarray(object_axial_contrib, dtype=np.float64)
    if face.ndim != 1:
        raise ValueError(f"contact must be one-dimensional, got {face.shape}")
    if exact.ndim != 2 or exact.shape[0] != len(face):
        raise ValueError("exact_contrib must have shape [substep,joint]")
    if object_total.shape != face.shape or object_axial.shape != face.shape:
        raise ValueError("object-side contributions must match contact trace")
    if not 0 <= onset < len(face) or not face[onset]:
        raise ValueError(f"accepted onset {onset} is outside contact")
    stop = len(face) if stop_exclusive is None else int(stop_exclusive)
    if not onset < stop <= len(face):
        raise ValueError(
            f"stop_exclusive must be in ({onset}, {len(face)}], got {stop}"
        )
    if (
        not np.isfinite(exact).all()
        or np.any(exact < 0.0)
        or not np.isfinite(object_total).all()
        or np.any(object_total < 0.0)
        or not np.isfinite(object_axial).all()
        or np.any(object_axial < 0.0)
    ):
        raise ValueError("contact impulse contributions must be finite and non-negative")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and non-negative, got {tol}")
    if np.any(object_axial > object_total + tol):
        raise ValueError(
            "object axial projection cannot exceed total force magnitude"
        )

    in_event = face.copy()
    in_event[:onset] = False
    in_event[stop:] = False
    exact_live = exact.sum(axis=1) > tol
    object_live = object_total > tol
    object_axial_live = object_axial > tol
    exact_n = int(np.count_nonzero(in_event & exact_live))
    object_n = int(np.count_nonzero(in_event & object_live))
    aligned_n = int(np.count_nonzero(in_event & exact_live & object_live))
    object_axial_n = int(np.count_nonzero(in_event & object_axial_live))
    aligned_axial_n = int(
        np.count_nonzero(in_event & exact_live & object_axial_live)
    )
    return {
        "exact_live_contact_substeps": exact_n,
        "object_live_contact_substeps": object_n,
        "aligned_live_contact_substeps": aligned_n,
        "object_axial_live_contact_substeps": object_axial_n,
        "aligned_axial_live_contact_substeps": aligned_axial_n,
        "non_axial_contact": object_axial_n == 0,
        "contact_signal_alignment_failure": (
            exact_n == 0 or object_n == 0 or aligned_n == 0
        ),
        # Established campaign direction: axial delivery with no robot-side
        # Lambda is impossible.  The reverse is a legitimate lateral graze.
        "lambda_dead": object_axial_n > 0 and exact_n == 0,
    }


def _apply_contact_signal_diagnostics(
    result: dict[str, object],
    diagnostics: dict[str, int | bool],
) -> dict[str, object]:
    """Apply liveness diagnostics without turning a lateral graze into an error."""
    result.update(diagnostics)
    if bool(diagnostics["contact_signal_alignment_failure"]):
        result["status"] = "error"
        result["error"] = (
            "exact contact-row and object total-force instruments are not "
            "live on an overlapping first-event face-contact sample"
        )
        result["eligible"] = False
        result["binding_class"] = BindingClass.INELIGIBLE.value
    elif int(diagnostics["aligned_axial_live_contact_substeps"]) < 1:
        result["eligible"] = False
        result["binding_class"] = BindingClass.INELIGIBLE.value
    return result


def classify_binding(
    *,
    rho_ship50: float,
    rho_row26: float,
    rho_row28: float,
    eligible: bool,
) -> BindingClass:
    """Apply the frozen two-channel binding interpretation matrix."""
    ratios = np.asarray([rho_ship50, rho_row26, rho_row28], dtype=np.float64)
    if not np.isfinite(ratios).all() or np.any(ratios < 0.0):
        raise ValueError(f"binding ratios must be finite and non-negative, got {ratios}")
    if not eligible:
        return BindingClass.INELIGIBLE
    if rho_row26 < 1.0 <= rho_row28:
        return (
            BindingClass.SHIPPED_TIME_BOUNDARY
            if rho_ship50 >= 1.0
            else BindingClass.TIME_BOUNDARY
        )
    shipped = rho_ship50 >= 1.0
    exact = rho_row26 >= 1.0
    if shipped and exact:
        return BindingClass.DUAL_BIND
    if shipped:
        return BindingClass.SHIPPED_ONLY
    if exact:
        return BindingClass.EXACT_ONLY
    return BindingClass.NO_BIND


def aggregate_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate sentinels without treating shared-reset replays as samples."""
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    delta_counts: dict[str, dict[str, int]] = {}
    stratified_counts: dict[
        str, dict[str, dict[str, dict[str, int]]]
    ] = {}
    impossible_success_n = 0
    lambda_dead_n = 0
    contact_signal_alignment_failure_n = 0
    operational_failure_n = 0
    fixed_baseline_failure_n = 0
    for row in results:
        status = str(row.get("status"))
        source = str(row.get("source_kind"))
        status_counts[status] = status_counts.get(status, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        operational_failure_n += int(
            status in {"error", "timeout", "crash"}
        )
        fixed_baseline_failure_n += int(
            source == "legacy"
            and row.get("reset_role") == "fixed"
            and float(row.get("solref_scale") or 0.0) == 1.0
            and status in {"error", "timeout", "crash"}
        )
        if source == "legacy":
            delta = row.get("delta")
            delta_key = "none" if delta is None else f"{float(delta):.2f}"
            bucket = delta_counts.setdefault(
                delta_key,
                {
                    "total_n": 0,
                    "ok_n": 0,
                    "eligible_n": 0,
                    "dual_bind_n": 0,
                },
            )
            bucket["total_n"] += 1
            bucket["ok_n"] += int(status == "ok")
            bucket["eligible_n"] += int(bool(row.get("eligible")))
            bucket["dual_bind_n"] += int(
                row.get("binding_class") == BindingClass.DUAL_BIND.value
            )
            reset_key = str(row.get("reset_role"))
            solref_key = f"{float(row.get('solref_scale') or 0.0):.1f}"
            stratum = (
                stratified_counts.setdefault(delta_key, {})
                .setdefault(reset_key, {})
                .setdefault(
                    solref_key,
                    {
                        "total_n": 0,
                        "ok_n": 0,
                        "eligible_n": 0,
                        "dual_bind_n": 0,
                        "failure_n": 0,
                    },
                )
            )
            stratum["total_n"] += 1
            stratum["ok_n"] += int(status == "ok")
            stratum["eligible_n"] += int(bool(row.get("eligible")))
            stratum["dual_bind_n"] += int(
                row.get("binding_class") == BindingClass.DUAL_BIND.value
            )
            stratum["failure_n"] += int(
                status in {"error", "timeout", "crash"}
            )
        impossible_success_n += int(bool(row.get("impossible_success")))
        lambda_dead_n += int(bool(row.get("lambda_dead")))
        contact_signal_alignment_failure_n += int(
            bool(row.get("contact_signal_alignment_failure"))
        )

    fixed = [
        row
        for row in results
        if row.get("source_kind") != "reference"
        and row.get("reset_role") == "fixed"
        and float(row.get("solref_scale") or 0.0) == 1.0
    ]
    return {
        "result_n": len(results),
        "status_counts": status_counts,
        "source_counts": source_counts,
        "delta_counts_descriptive_only": delta_counts,
        "stratified_counts": stratified_counts,
        "operational_failure_n": operational_failure_n,
        "fixed_baseline_failure_n": fixed_baseline_failure_n,
        "impossible_success_n": impossible_success_n,
        "lambda_dead_n": lambda_dead_n,
        "contact_signal_alignment_failure_n": (
            contact_signal_alignment_failure_n
        ),
        "fixed_reset_existence": {
            "denominator_n": len(fixed),
            "eligible_n": sum(bool(row.get("eligible")) for row in fixed),
            "dual_bind_n": sum(
                row.get("binding_class") == BindingClass.DUAL_BIND.value
                for row in fixed
            ),
        },
        "reference_control_n": sum(
            row.get("source_kind") == "reference" for row in results
        ),
        "robustness_replay_n": sum(
            row.get("reset_role") == "robustness" for row in results
        ),
        "interpretation": {
            "fixed_reset_is_existence_test": True,
            "robustness_resets_are_independent_samples": False,
            "reference_in_binding_rates": False,
            "solref_low_change_proves_hardware_fidelity": False,
        },
    }


def compare_solref_results(
    baseline: dict[str, object], softened: dict[str, object]
) -> dict[str, object]:
    """Compare identical-tape solref*2 replay, preserving joint-switch evidence."""
    for key in ("action_digest", "reset_digest", "delta"):
        if baseline.get(key) != softened.get(key):
            raise ValueError(f"solref pair identity mismatch in {key}")
    if float(baseline.get("solref_scale") or 0.0) != 1.0:
        raise ValueError("baseline solref scale must be 1")
    if float(softened.get("solref_scale") or 0.0) != 2.0:
        raise ValueError("softened solref scale must be 2")
    channels = {
        "row26": "row26_perjoint",
        "row28": "row28_perjoint",
        "full_lobe": "full_lobe_perjoint",
    }
    relative: dict[str, list[float | None]] = {}
    for label, key in channels.items():
        before = np.asarray(baseline[key], dtype=np.float64)
        after = np.asarray(softened[key], dtype=np.float64)
        if before.shape != (6,) or after.shape != (6,):
            raise ValueError(f"solref pair {key} must contain six joints")
        relative[label] = [
            (float((after[j] - before[j]) / before[j]) if before[j] > 0.0 else None)
            for j in range(6)
        ]
    binding_joint = int(baseline["binding_joint_index"])
    if not 0 <= binding_joint < 6:
        raise ValueError(f"invalid baseline binding joint {binding_joint}")
    baseline_joint_change = {
        label: relative[label][binding_joint] for label in channels
    }
    soft_joint = int(softened["binding_joint_index"])
    robust = all(
        value is not None and abs(float(value)) < 0.20
        for value in baseline_joint_change.values()
    )
    return {
        "necessary_model_robustness_only": True,
        "proves_ballistic_or_hardware_fidelity": False,
        "relative_change_perjoint": relative,
        "baseline_binding_joint_index": binding_joint,
        "softened_binding_joint_index": soft_joint,
        "binding_joint_switched": soft_joint != binding_joint,
        "baseline_binding_joint_relative_change": baseline_joint_change,
        "within_20pct_at_baseline_binding_joint": robust,
    }


def _requires_solref_replay(result: dict[str, object]) -> bool:
    """Schedule candidate sensitivity plus the fixed reference calibration."""
    if result.get("source_kind") == "legacy":
        return bool(result.get("eligible"))
    return bool(
        result.get("source_kind") == "reference"
        and result.get("reset_role") == "fixed"
        and float(result.get("solref_scale") or 0.0) == 1.0
    )


def summarize_trace(
    *,
    contact: np.ndarray,
    downward_speed: np.ndarray,
    depth: np.ndarray,
    exact_contrib: np.ndarray,
    shipped_peak: np.ndarray,
    qvel: np.ndarray,
    qpos_legal: bool,
    quality_valid: bool,
    quality_overflow: bool,
    nonface_contact: np.ndarray,
    caps: np.ndarray,
    rail: float,
    accepted_onset: int,
    release_confirm_off: int = 5,
) -> dict[str, object]:
    """Summarize one no-auto-reset rollout under the frozen Stage-0 rules."""
    face = np.asarray(contact, dtype=bool)
    speed = np.asarray(downward_speed, dtype=np.float64)
    nail_depth = np.asarray(depth, dtype=np.float64)
    exact = np.asarray(exact_contrib, dtype=np.float64)
    ship = np.asarray(shipped_peak, dtype=np.float64)
    other = np.asarray(nonface_contact, dtype=np.float64)
    limits = np.asarray(caps, dtype=np.float64)
    if face.ndim != 1 or speed.shape != face.shape or nail_depth.shape != face.shape:
        raise ValueError("contact, downward_speed, and depth must be matching 1-D traces")
    if other.shape != face.shape:
        raise ValueError("nonface_contact must match contact trace")
    if not np.isfinite(other).all() or np.any(other < 0.0):
        raise ValueError("nonface contact force must be finite and non-negative")
    if exact.ndim != 2 or exact.shape[0] != len(face):
        raise ValueError("exact_contrib must have shape [substep,joint]")
    if ship.shape != (exact.shape[1],) or limits.shape != ship.shape:
        raise ValueError("shipped_peak and caps must match exact_contrib joint count")
    if not np.isfinite(nail_depth).all() or not np.isfinite(speed).all():
        raise ValueError("physical trace contains non-finite values")
    if (
        not np.isfinite(ship).all()
        or np.any(ship < 0.0)
        or not np.isfinite(limits).all()
        or np.any(limits <= 0)
    ):
        raise ValueError(
            "shipped peaks must be finite and non-negative; caps must be finite and positive"
        )

    windows = onset_window_impulses(exact, onset=accepted_onset)
    sliding50 = sliding_window_impulses(exact, length=25)
    rho_ship_joint = ship / limits
    rho26_joint = windows[13] / limits
    rho28_joint = windows[14] / limits
    rho50_onset_joint = windows[25] / limits
    rho50_sliding_joint = sliding50["peak_perjoint"] / limits
    binding_joint = int(np.argmax(rho26_joint))
    release_by_confirm = {
        str(confirm): first_release(face, onset=accepted_onset, confirm_off=confirm)
        for confirm in RELEASE_CONFIRM_SAMPLES
    }
    release = release_by_confirm[str(release_confirm_off)]
    lobe_end = first_impact_lobe_end(
        face,
        speed,
        onset=accepted_onset,
        stop_exclusive=release if release is not None else len(face),
    )
    full_lobe = (
        exact[accepted_onset : lobe_end + 1].sum(axis=0)
        if lobe_end is not None
        else np.zeros(exact.shape[1], dtype=np.float64)
    )
    lobe_share_end = (
        min(lobe_end + 1, accepted_onset + 25)
        if lobe_end is not None
        else accepted_onset
    )
    lobe_in_onset50 = exact[accepted_onset:lobe_share_end].sum(axis=0)
    denom = float(windows[25][binding_joint])
    lobe_fraction = (
        float(lobe_in_onset50[binding_joint] / denom)
        if denom > 0.0
        else 0.0
    )

    qvel_legal, qvel_peak = strict_qvel_legal(qvel, rail=rail)
    horizon_end = min(len(face), accepted_onset + 25)
    productive = bool(
        np.max(nail_depth[accepted_onset:horizon_end], initial=0.0) >= 0.030
    )
    release_by_28ms = release is not None and release <= accepted_onset + 14
    lobe_completed = (
        lobe_end is not None
        and release is not None
        and accepted_onset <= lobe_end < release
    )
    no_recontact = False
    recontact_count = 0
    if release is not None:
        confirmed_end = min(len(face), release + release_confirm_off)
        no_recontact = not bool(face[confirmed_end:horizon_end].any())
        tail = face[confirmed_end:horizon_end]
        if tail.size:
            prior = np.concatenate([np.array([False]), tail[:-1]])
            recontact_count = int(np.count_nonzero(tail & ~prior))
    face_only = not bool(
        (
            other[
                accepted_onset : (
                    release + 1 if release is not None else horizon_end
                )
            ]
            > NONFACE_FORCE_TOL
        ).any()
    )
    eligible = bool(
        qvel_legal
        and qpos_legal
        and quality_valid
        and not quality_overflow
        and productive
        and release_by_28ms
        and lobe_completed
        and no_recontact
        and face_only
        and lobe_fraction >= 0.70
    )
    rho_ship50 = float(rho_ship_joint.max(initial=0.0))
    rho_row26 = float(rho26_joint.max(initial=0.0))
    rho_row28 = float(rho28_joint.max(initial=0.0))
    binding_class = classify_binding(
        rho_ship50=rho_ship50,
        rho_row26=rho_row26,
        rho_row28=rho_row28,
        eligible=eligible,
    )
    return {
        "accepted_onset_index": int(accepted_onset),
        "release_index": release,
        "release_index_by_confirm": release_by_confirm,
        "lobe_end_index": lobe_end,
        "release_by_28ms": release_by_28ms,
        "lobe_completed": lobe_completed,
        "no_recontact_50ms": no_recontact,
        "recontact_count_50ms": recontact_count,
        "face_only": face_only,
        "nonface_force_tolerance": NONFACE_FORCE_TOL,
        "nonface_max_efc_force": float(other.max(initial=0.0)),
        "productive_30mm_50ms": productive,
        "qvel_legal": qvel_legal,
        "qvel_legal_95pct": bool(
            np.isfinite(qvel_peak)
            and qvel_peak <= QVEL_MARGIN_FRACTION * rail
        ),
        "qvel_peak_rad_s": qvel_peak,
        "qpos_legal": bool(qpos_legal),
        "quality_valid": bool(quality_valid),
        "quality_overflow": bool(quality_overflow),
        "eligible": eligible,
        "binding_joint_index": binding_joint,
        "lobe_fraction": lobe_fraction,
        "lobe_fraction_60": lobe_fraction >= 0.60,
        "lobe_fraction_80": lobe_fraction >= 0.80,
        "rho_ship50": rho_ship50,
        "rho_row26": rho_row26,
        "rho_row28": rho_row28,
        "rho_row50": float(rho50_onset_joint.max(initial=0.0)),
        "rho_row50_onset": float(rho50_onset_joint.max(initial=0.0)),
        "rho_row50_sliding_peak": float(
            rho50_sliding_joint.max(initial=0.0)
        ),
        "row26_perjoint": windows[13].tolist(),
        "row28_perjoint": windows[14].tolist(),
        "row50_perjoint": windows[25].tolist(),
        "row50_onset_perjoint": windows[25].tolist(),
        "row50_sliding_peak_perjoint": sliding50["peak_perjoint"].tolist(),
        "row50_sliding_start_perjoint": sliding50["start_perjoint"].tolist(),
        "full_lobe_perjoint": full_lobe.tolist(),
        "ship50_perjoint": ship.tolist(),
        "contact_duration_substeps": (
            int(release - accepted_onset) if release is not None else None
        ),
        "contact_duration_ms": (
            float((release - accepted_onset) * 2.0)
            if release is not None
            else None
        ),
        "binding_class": binding_class.value,
    }


def _action_digest(actions: np.ndarray) -> str:
    canonical = np.ascontiguousarray(actions, dtype="<f4")
    h = hashlib.sha256()
    h.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    h.update(canonical.tobytes())
    return h.hexdigest()


def assert_legacy_realized_digest(
    *, action_mode: str, expected: object, realized: object
) -> None:
    """A legal replay may not silently substitute a different legacy tape."""
    if action_mode != "tape":
        return
    if (
        not isinstance(expected, str)
        or not isinstance(realized, str)
        or len(expected) != 64
        or len(realized) != 64
        or expected != realized
    ):
        raise RuntimeError(
            f"legacy realized action digest mismatch: {realized} != {expected}"
        )


def load_saved_tapes(
    data_dir: Path,
    *,
    filenames: tuple[str, ...] = LEGACY_WHIP_FILES,
) -> list[SavedTape]:
    """Load the explicit legacy inventory and deduplicate by scale/action bytes."""
    tapes: list[SavedTape] = []
    identity_to_index: dict[tuple[float, str], int] = {}
    paths = [Path(data_dir) / filename for filename in filenames]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing required whip JSON files: {missing}")
    for path in paths:
        payload = json.loads(path.read_text())
        try:
            delta = float(payload["delta"])
            expected_t = int(payload["T"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid delta/T") from exc
        if delta not in ALLOWED_DELTAS:
            raise ValueError(
                f"{path}: delta {delta} is not in allowed set {ALLOWED_DELTAS}"
            )
        for field in ("best_actions", "best_actions_hw"):
            if field not in payload:
                raise ValueError(f"{path}: missing {field}")
            actions = np.asarray(payload[field], dtype=np.float32)
            if actions.shape != (expected_t, 3):
                raise ValueError(
                    f"{path}:{field}: expected shape {(expected_t, 3)}, got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise ValueError(f"{path}:{field}: non-finite action")
            if np.any(np.abs(actions) > 1.0):
                raise ValueError(f"{path}:{field}: action outside [-1,1]")
            digest = _action_digest(actions)
            identity = (delta, digest)
            alias = (path.name, field)
            if identity in identity_to_index:
                tape_index = identity_to_index[identity]
                existing = tapes[tape_index]
                tapes[tape_index] = SavedTape(
                    source_path=existing.source_path,
                    source_field=existing.source_field,
                    delta=existing.delta,
                    actions=existing.actions,
                    digest=existing.digest,
                    aliases=existing.aliases + (alias,),
                )
                continue
            actions.setflags(write=False)
            identity_to_index[identity] = len(tapes)
            tapes.append(
                SavedTape(
                    source_path=path,
                    source_field=field,
                    delta=delta,
                    actions=actions,
                    digest=digest,
                    aliases=(alias,),
                )
            )
    return tapes


def _max_nonface_efc_force(
    *,
    geom: np.ndarray,
    contact_type: np.ndarray,
    world_id: np.ndarray,
    efc_address: np.ndarray,
    efc_force: np.ndarray,
    robot_geom_ids: set[int],
    hammer_geom_ids: set[int],
    nail_geom_ids: set[int],
    constraint_bit: int,
) -> float:
    """Maximum active EFC-row force for disallowed robot contact pairs."""
    geom_a = np.asarray(geom, dtype=np.int64)
    type_a = np.asarray(contact_type, dtype=np.int64)
    world_a = np.asarray(world_id, dtype=np.int64)
    address_a = np.asarray(efc_address, dtype=np.int64)
    force_a = np.asarray(efc_force, dtype=np.float64)
    ncontact = len(type_a)
    if geom_a.shape != (ncontact, 2):
        raise ValueError(f"geom must have shape ({ncontact},2), got {geom_a.shape}")
    if world_a.shape != (ncontact,) or address_a.ndim != 2 or address_a.shape[0] != ncontact:
        raise ValueError("world_id/efc_address must match contact count")
    if force_a.ndim != 2 or not np.isfinite(force_a).all():
        raise ValueError("efc_force must be a finite [world,row] array")

    peak = 0.0
    for index in range(ncontact):
        if not (int(type_a[index]) & int(constraint_bit)):
            continue
        g0, g1 = int(geom_a[index, 0]), int(geom_a[index, 1])
        if g0 not in robot_geom_ids and g1 not in robot_geom_ids:
            continue
        allowed = (g0 in hammer_geom_ids and g1 in nail_geom_ids) or (
            g1 in hammer_geom_ids and g0 in nail_geom_ids
        )
        if allowed:
            continue
        world = int(world_a[index])
        if not 0 <= world < force_a.shape[0]:
            raise RuntimeError(f"contact {index} has invalid world id {world}")
        for row in address_a[index]:
            row_i = int(row)
            if row_i < 0:
                continue
            if row_i >= force_a.shape[1]:
                raise RuntimeError(f"contact {index} has invalid EFC row {row_i}")
            peak = max(peak, abs(float(force_a[world, row_i])))
    return peak


def _max_nonface_robot_contact_force(
    env: Any, robot_geom_ids: set[int]
) -> float:
    """Maximum active constraint force involving the robot outside hammer-nail."""
    import warp as wp

    from src.tasks.hammer.mdp.contact_row_impulse import (
        _CONTACT_TYPE_CONSTRAINT_BIT,
        _resolve_contact_geom_ids,
    )

    data = env.sim.wp_data
    nacon = int(wp.to_torch(data.nacon)[0])
    if nacon > data.naconmax:
        raise RuntimeError(
            f"contact overflow: nacon={nacon} > naconmax={data.naconmax}"
        )
    if nacon == 0:
        return 0.0
    geom = wp.to_torch(data.contact.geom)[:nacon]
    ctype = wp.to_torch(data.contact.type)[:nacon]
    world_id = wp.to_torch(data.contact.worldid)[:nacon]
    address = wp.to_torch(data.contact.efc_address)[:nacon]
    efc_force = wp.to_torch(data.efc.force)
    hammer_ids, nail_ids = _resolve_contact_geom_ids(env)
    return _max_nonface_efc_force(
        geom=geom.detach().cpu().numpy(),
        contact_type=ctype.detach().cpu().numpy(),
        world_id=world_id.detach().cpu().numpy(),
        efc_address=address.detach().cpu().numpy(),
        efc_force=efc_force.detach().cpu().numpy(),
        robot_geom_ids=robot_geom_ids,
        hammer_geom_ids=hammer_ids,
        nail_geom_ids=nail_ids,
        constraint_bit=_CONTACT_TYPE_CONSTRAINT_BIT,
    )


_SOLREF_GEOM_SUFFIXES = ("nail_head", "hammer_head_0", "hammer_head_1")
_BASELINE_SOLREF = {
    "nail_head": (0.008, 1.0),
    "hammer_head_0": (0.02, 1.0),
    "hammer_head_1": (0.02, 1.0),
}


def _apply_solref_scale(cfg: Any, scale: float) -> dict[str, dict[str, list[float]]]:
    """Patch only the three hammer/nail contact geoms before compilation."""
    if scale not in (1.0, 2.0):
        raise ValueError(f"solref scale must be 1 or 2, got {scale}")
    edits: dict[str, dict[str, list[float]]] = {}
    if scale == 1.0:
        return edits

    def wrap_spec(entity_cfg: Any) -> None:
        original = entity_cfg.spec_fn

        def patched_spec():
            spec = original()
            for geom in spec.geoms:
                if geom.name not in _SOLREF_GEOM_SUFFIXES:
                    continue
                before = [float(value) for value in geom.solref]
                if before[0] <= 0.0:
                    raise RuntimeError(
                        f"direct-stiffness solref is unsupported: "
                        f"{geom.name}={before}"
                    )
                after = [before[0] * scale, before[1]]
                geom.solref = after
                edits[geom.name] = {"before": before, "after": after}
            return spec

        entity_cfg.spec_fn = patched_spec

    wrap_spec(cfg.scene.entities["robot"])
    wrap_spec(cfg.scene.entities["nail_block"])
    return edits


def _stage0_env_cfg(
    delta: float,
    *,
    play: bool = True,
    shadow: bool = True,
    solref_scale: float = 1.0,
):
    """Build the ordinary task, removing only success termination for observation."""
    from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT, z1_hammer_env_cfg

    if float(delta) not in ALLOWED_DELTAS:
        raise ValueError(f"delta must be one of {ALLOWED_DELTAS}, got {delta}")
    cfg = z1_hammer_env_cfg(
        play=play,
        cat_impulse=True,
        event_correct=True,
        quality_instrumentation=True,
    )
    if "nail_driven" not in cfg.terminations:
        raise RuntimeError("ordinary task unexpectedly lacks nail_driven termination")
    if shadow:
        cfg.terminations.pop("nail_driven")
    cfg.scene.num_envs = 1
    cfg.auto_reset = not shadow
    cfg.actions["ik_hammer_head"].delta_pos_scale = float(delta)
    cat = cfg.metrics["cat_soft"].params
    if float(cat["imp_max_p"]) != 0.0:
        raise RuntimeError(f"imp_max_p must stay log-only 0, got {cat['imp_max_p']}")
    if [float(value) for value in cat["imp_limit"]] != list(IMP_J_LIMIT):
        raise RuntimeError("runtime impulse caps drifted from IMP_J_LIMIT")
    if [float(value) for value in IMP_J_LIMIT] != [
        1.640,
        3.280,
        1.640,
        1.640,
        1.640,
        1.640,
    ]:
        raise RuntimeError("manufacturer-derived IMP_J_LIMIT literal drift")
    action = cfg.actions["ik_hammer_head"]
    expected_action = {
        "entity_name": "robot",
        "actuator_names": (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ),
        "frame_type": "site",
        "frame_name": "hammer_head_site",
        "use_relative_mode": True,
        "delta_pos_scale": float(delta),
        "delta_ori_scale": 1.0,
        "damping": 0.05,
        "max_dq": 0.5,
        "position_weight": 1.0,
        "orientation_weight": 0.0,
        "joint_limit_weight": 0.0,
        "posture_weight": 0.0,
        "posture_target": None,
    }
    for name, expected in expected_action.items():
        actual = getattr(action, name)
        if isinstance(expected, tuple):
            actual = tuple(actual)
        if actual != expected:
            raise RuntimeError(
                f"fixed action contract drift: {name}={actual!r}, "
                f"expected {expected!r}"
            )
    actuator_signature = tuple(
        (
            type(actuator).__name__,
            tuple(actuator.target_names_expr),
            float(actuator.stiffness),
            float(actuator.damping),
            float(actuator.effort_limit),
            float(actuator.armature),
        )
        for actuator in cfg.scene.entities["robot"].articulation.actuators
    )
    expected_actuators = (
        (
            "BuiltinPositionActuatorCfg",
            ("joint1", "joint3", "joint4", "joint5", "joint6"),
            1000.0,
            100.0,
            30.0,
            0.01,
        ),
        (
            "BuiltinPositionActuatorCfg",
            ("joint2",),
            1500.0,
            150.0,
            60.0,
            0.02,
        ),
        (
            "BuiltinPositionActuatorCfg",
            ("jointGripper",),
            100.0,
            20.0,
            30.0,
            0.005,
        ),
    )
    if actuator_signature != expected_actuators:
        raise RuntimeError(
            f"fixed actuator contract drift: {actuator_signature!r}"
        )
    if abs(float(cfg.sim.mujoco.timestep) - 0.002) > 1e-15:
        raise RuntimeError(f"physics timestep drifted: {cfg.sim.mujoco.timestep}")
    if int(cfg.decimation) != 10:
        raise RuntimeError(f"control decimation drifted: {cfg.decimation}")
    cfg._lambda_solref_edits = _apply_solref_scale(cfg, float(solref_scale))
    return cfg


def _verify_compiled_solref(
    env: Any,
    *,
    scale: float,
    edits: dict[str, dict[str, list[float]]],
) -> dict[str, list[float]]:
    """Read back compiled solref values; a sensitivity arm cannot be assumed."""
    import mujoco

    model = getattr(env.sim, "mj_model", None)
    if model is None:
        model = getattr(env.sim, "model", None)
    if model is None or not hasattr(model, "geom_solref"):
        raise RuntimeError("compiled MuJoCo model does not expose geom_solref")
    compiled: dict[str, list[float]] = {}
    for geom_id in range(model.ngeom):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        )
        suffix = next(
            (item for item in _SOLREF_GEOM_SUFFIXES if name.endswith(item)),
            None,
        )
        if suffix is None:
            continue
        values = [float(value) for value in model.geom_solref[geom_id]]
        compiled[name] = values
        expected = [
            _BASELINE_SOLREF[suffix][0] * scale,
            _BASELINE_SOLREF[suffix][1],
        ]
        if not np.allclose(values, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"compiled solref mismatch for {name}: {values} != {expected}"
            )
    if len(compiled) != 3:
        raise RuntimeError(
            f"expected exactly 3 hammer/nail solref geoms, got {sorted(compiled)}"
        )
    if scale == 2.0 and set(edits) != set(_SOLREF_GEOM_SUFFIXES):
        raise RuntimeError(f"solref*2 did not edit all target geoms: {edits}")
    return compiled


def _build_stage0_env(
    delta: float,
    *,
    play: bool = True,
    shadow: bool = True,
    solref_scale: float = 1.0,
):
    """Construct the no-success-termination, no-auto-reset CPU shadow env."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = _stage0_env_cfg(
        delta,
        play=play,
        shadow=shadow,
        solref_scale=solref_scale,
    )
    env = ManagerBasedRlEnv(cfg, device="cpu")
    if abs(float(env.physics_dt) - 0.002) > 1e-12:
        raise RuntimeError(f"physics_dt drifted: {env.physics_dt}")
    if int(cfg.decimation) != 10:
        raise RuntimeError(f"decimation drifted: {cfg.decimation}")
    env._lambda_compiled_solref = _verify_compiled_solref(
        env,
        scale=float(solref_scale),
        edits=cfg._lambda_solref_edits,
    )
    return env


def _capture_reset_record(
    env: Any,
    *,
    reset_seed: int,
    reset_call_index: int,
) -> dict[str, object]:
    """Capture only the canonical reset contract's complete realized state."""
    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    realized = {
        "robot_joint_pos": (
            robot.data.joint_pos[0].detach().cpu().tolist()
        ),
        "robot_joint_vel": robot.data.joint_vel[0].detach().cpu().tolist(),
        "nail_joint_pos": nail.data.joint_pos[0].detach().cpu().tolist(),
        "nail_joint_vel": nail.data.joint_vel[0].detach().cpu().tolist(),
    }
    # Round-trip through plain floats to avoid dtype-specific JSON encoders.
    realized = {
        key: [float(value) for value in values]
        for key, values in realized.items()
    }
    return {
        "reset_contract_version": 1,
        "reset_state": {
            "regeneration_inputs": {
                "reset_rng_seed": int(reset_seed),
                "reset_rng_call_index": int(reset_call_index),
            },
            "realized": realized,
        },
        "reset_state_digest": reset_bank_digest(realized),
    }


def materialize_fixed_reset() -> dict[str, object]:
    """Materialize the original deterministic play reset."""
    fixed_env = _build_stage0_env(0.15, play=True, shadow=True)
    try:
        fixed_env.reset()
        return _capture_reset_record(
            fixed_env, reset_seed=0, reset_call_index=0
        )
    finally:
        fixed_env.close()


def materialize_stage0_resets() -> tuple[dict[str, object], list[dict[str, object]]]:
    """Materialize the fixed reset and 16 frozen training-distribution resets."""
    from scripts.eval_impulse import _install_evaluator_rng_streams

    fixed = materialize_fixed_reset()
    random_env = _build_stage0_env(0.15, play=False, shadow=True)
    bank: list[dict[str, object]] = []
    try:
        _install_evaluator_rng_streams(
            random_env,
            reset_seed=RESET_BANK_SEED,
            observation_seed=RESET_BANK_SEED + 1,
        )
        for call_index in range(RESET_BANK_SIZE):
            random_env.reset()
            bank.append(
                _capture_reset_record(
                    random_env,
                    reset_seed=RESET_BANK_SEED,
                    reset_call_index=call_index,
                )
            )
    finally:
        random_env.close()
    validate_reset_bank(
        bank,
        fixed_reset=fixed,
        forbidden_digests={str(fixed["reset_state_digest"])},
    )
    return fixed, bank


def _restore_canonical_reset(env: Any, reset_record: dict[str, object]) -> str:
    """Use the already-reviewed D2 reset replay contract; no other state writes."""
    from scripts.eval_impulse import restore_reset_state

    return restore_reset_state(env, reset_record)


def _git_provenance(
    repo_root: Path, *, scope: tuple[str, ...] = ()
) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        repository_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        if scope:
            status = subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    *scope,
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        else:
            status = repository_status
    except Exception as exc:
        return {
            "revision": "unknown",
            "dirty": True,
            "status": f"git provenance unavailable: {exc}",
            "scope": list(scope),
            "asset_scope_clean": False if scope else None,
            "repository_dirty": True,
            "repository_status": f"git provenance unavailable: {exc}",
        }
    return {
        "revision": revision,
        "dirty": bool(status),
        "status": status,
        "scope": list(scope),
        "asset_scope_clean": not bool(status) if scope else None,
        "repository_dirty": bool(repository_status),
        "repository_status": repository_status,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(
    *,
    repo_root: Path,
    data_dir: Path,
    require_clean: bool,
) -> dict[str, object]:
    """Collect code/asset/tape identities and refuse dirty authoritative work."""
    from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

    code = _git_provenance(repo_root)
    asset_root = Path(Z1_HAMMER_XML).resolve().parents[2]
    asset = _git_provenance(
        asset_root, scope=("hammer_z1_env/assets",)
    )
    if require_clean and (
        bool(code["dirty"])
        or bool(asset["dirty"])
        or code["revision"] == "unknown"
        or asset["revision"] == "unknown"
    ):
        raise RuntimeError(
            "authoritative Stage-0 requires clean code and asset provenance: "
            f"code={code!r}, asset={asset!r}"
        )
    tape_files = {
        filename: _file_sha256(Path(data_dir) / filename)
        for filename in LEGACY_WHIP_FILES
    }
    config = {
        "caps_n_m_s": [1.640, 3.280, 1.640, 1.640, 1.640, 1.640],
        "imp_max_p": 0.0,
        "allowed_delta_pos_scale": list(ALLOWED_DELTAS),
        "physics_dt_s": 0.002,
        "decimation": 10,
        "qvel_rail_rad_s": HARDWARE_QVEL_LIMIT_RAD_S,
        "qvel_margin_fraction": QVEL_MARGIN_FRACTION,
        "release_confirm_samples": list(RELEASE_CONFIRM_SAMPLES),
        "nonface_force_tolerance": NONFACE_FORCE_TOL,
        "reset_bank_seed": RESET_BANK_SEED,
        "reset_bank_size": RESET_BANK_SIZE,
        "contact_liveness_semantics": "exact-plus-object-total-v2",
    }
    return {
        "code": code,
        "asset": asset,
        "asset_scope_clean": bool(asset["asset_scope_clean"]),
        "tape_files_sha256": tape_files,
        "config": config,
        "config_sha256": reset_bank_digest(config),
    }


def _reference_action(
    *,
    env: Any,
    reference: Any,
    head_position: Any,
    control_index: int,
    playback_length: int,
    delta: float,
):
    """Canonical playback_reference action, evaluated from current live state."""
    target = reference.playback_target(min(control_index, playback_length))
    return ((target - head_position()) / float(delta)).clamp(-1.0, 1.0)


def _replay_action_request(request: dict[str, object]) -> dict[str, object]:
    """Legal DiffIK replay with complete 500 Hz Stage-0 instrumentation."""
    import torch
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from src.assets.robots.unitree_z1.z1_constants import (
        ARM_JOINT_NAMES,
        HAMMER_HEAD_SITE_NAME,
    )
    from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
    from src.tasks.hammer.mdp.contact_row_impulse import (
        arm_dof_cols,
        contact_row_qfrc,
    )
    from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
    from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR
    from src.tasks.hammer.mdp.references import SingleStrikeReference

    identity = dict(request["identity"])
    delta = float(identity["delta"])
    solref_scale = float(identity["solref_scale"])
    result = _new_result(identity, status="error")
    env = _build_stage0_env(
        delta,
        play=True,
        shadow=True,
        solref_scale=solref_scale,
    )
    result["compiled_solref"] = env._lambda_compiled_solref
    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    face_sensor = env.scene["hammer_nail_contact"]
    impulse_sensor = env.scene["hammer_nail_impulse"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    head_cfg.resolve(env.scene)
    arm_ids = arm_cfg.joint_ids
    head_id = head_cfg.site_ids[0]
    row_cols = arm_dof_cols(env)
    robot_geom_ids = {int(value) for value in robot.indexing.geom_ids.tolist()}
    acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)
    tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR)
    dt = float(env.physics_dt)
    axis = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float32, device=env.device)
    effort_limits = np.asarray(
        [30.0, 60.0, 30.0, 30.0, 30.0, 30.0], dtype=np.float64
    )

    qvel_pre: list[np.ndarray] = []
    qvel_post: list[np.ndarray] = []
    qpos_pre: list[np.ndarray] = []
    qpos_post: list[np.ndarray] = []
    actuator_effort: list[np.ndarray] = []
    head_pose: list[np.ndarray] = []
    head_twist: list[np.ndarray] = []
    depth: list[float] = []
    contact: list[bool] = []
    nonface_force: list[float] = []
    exact_contrib: list[np.ndarray] = []
    object_axial_force: list[float] = []
    object_axial_contrib: list[float] = []
    object_total_contrib: list[float] = []
    shipped_rolling: list[np.ndarray] = []
    shipped_contrib: list[np.ndarray] = []
    tracker_started: list[bool] = []
    realized_actions: list[np.ndarray] = []

    original_sim_step = env.sim.step
    original_substep = env.metrics_manager.compute_substep

    def preintegration_then_step() -> None:
        qvel_pre.append(
            robot.data.joint_vel[0, arm_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        qpos_pre.append(
            robot.data.joint_pos[0, arm_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        original_sim_step()

    def record_substep() -> None:
        original_substep()
        qvel_post.append(
            robot.data.joint_vel[0, arm_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        qpos_post.append(
            robot.data.joint_pos[0, arm_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        actuator_effort.append(
            robot.data.qfrc_actuator[0, arm_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        head_pose.append(
            robot.data.site_pose_w[0, head_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        head_twist.append(
            robot.data.site_vel_w[0, head_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        depth.append(float(nail.data.joint_pos[0, 0].detach().cpu()))
        contact.append(bool((face_sensor.data.found[0] > 0).any()))
        nonface_force.append(
            _max_nonface_robot_contact_force(env, robot_geom_ids)
        )
        row = contact_row_qfrc(env)[0, row_cols].detach().cpu().numpy()
        exact_contrib.append(np.abs(row.astype(np.float64)) * dt)
        object_force = impulse_sensor.data.force[0].sum(dim=-2)
        axial_force = float(
            torch.dot(object_force, axis).clamp_min(0.0).detach().cpu()
        )
        object_axial_force.append(axial_force)
        object_axial_contrib.append(axial_force * dt)
        object_total_contrib.append(
            float(torch.linalg.vector_norm(object_force).detach().cpu()) * dt
        )
        shipped_rolling.append(
            acc._rolling[0].detach().cpu().numpy().astype(np.float64)
        )
        last_slot = (int(acc._buf_i) - 1) % int(acc._window)
        shipped_contrib.append(
            acc._buf[0, :, last_slot].detach().cpu().numpy().astype(np.float64)
        )
        tracker_started.append(bool(tracker.started[0]))

    env.sim.step = preintegration_then_step
    env.metrics_manager.compute_substep = record_substep
    initial_qvel: np.ndarray | None = None
    initial_qpos: np.ndarray | None = None
    hard_limits: np.ndarray | None = None
    rollout_error: Exception | None = None
    try:
        env.reset()
        reset_record = request.get("reset_record")
        if not isinstance(reset_record, dict):
            raise ValueError("replay request missing canonical reset_record")
        restored_digest = _restore_canonical_reset(env, reset_record)
        if restored_digest != identity["reset_digest"]:
            raise RuntimeError(
                f"restored reset digest mismatch: {restored_digest} != "
                f"{identity['reset_digest']}"
            )
        initial_qvel = (
            robot.data.joint_vel[0, arm_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        initial_qpos = (
            robot.data.joint_pos[0, arm_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        hard_limits = (
            robot.data.joint_pos_limits[0, arm_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        action_mode = str(request.get("action_mode", "tape"))
        reference = None
        playback_length = 0
        if action_mode == "reference":
            reference = SingleStrikeReference(1, env.device)

            def current_head():
                return robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)

            nail_top_cfg = SceneEntityCfg(
                "nail_block", site_names=("nail_top",)
            )
            nail_top_cfg.resolve(env.scene)
            nail_top = nail.data.site_pos_w[
                :, nail_top_cfg.site_ids
            ].squeeze(1)
            reference.update(
                current_head(),
                nail_top,
                torch.zeros(1, dtype=torch.long, device=env.device),
            )
            playback_length = reference.playback_length()
            base_actions = None
        elif action_mode == "tape":
            base_actions = np.asarray(request["actions"], dtype=np.float32)
            if base_actions.shape != (30, 3):
                raise ValueError(
                    f"replay action tape must have shape (30,3), got "
                    f"{base_actions.shape}"
                )
        else:
            raise ValueError(f"unknown action_mode {action_mode!r}")

        for control_index in range(1, 31):
            if reference is not None:
                action_tensor = _reference_action(
                    env=env,
                    reference=reference,
                    head_position=current_head,
                    control_index=control_index,
                    playback_length=playback_length,
                    delta=delta,
                )
            else:
                action_tensor = torch.tensor(
                    np.array(base_actions[control_index - 1], copy=True),
                    dtype=torch.float32,
                    device=env.device,
                )[None]
            realized_actions.append(
                action_tensor[0].detach().cpu().numpy().astype(np.float32)
            )
            env.step(action_tensor)
        zero = torch.zeros((1, 3), dtype=torch.float32, device=env.device)
        for _ in range(POST_TAPE_ZERO_ACTIONS):
            env.step(zero)
        if int(env.episode_length_buf[0]) != 30 + POST_TAPE_ZERO_ACTIONS:
            raise RuntimeError(
                "shadow evaluator auto-reset, terminated, or skipped a control step: "
                f"episode_length={int(env.episode_length_buf[0])}"
            )
    except Exception as exc:
        rollout_error = exc
    finally:
        env.sim.step = original_sim_step
        env.metrics_manager.compute_substep = original_substep

    if rollout_error is not None:
        qvel_partial = (
            np.asarray(qvel_post, dtype=np.float64)
            if qvel_post
            else np.empty((0, 6), dtype=np.float64)
        )
        qpos_partial = (
            np.asarray(qpos_post, dtype=np.float64)
            if qpos_post
            else np.empty((0, 6), dtype=np.float64)
        )
        qvel_state = (
            np.concatenate([initial_qvel[None, :], qvel_partial], axis=0)
            if initial_qvel is not None
            else qvel_partial
        )
        qpos_state = (
            np.concatenate([initial_qpos[None, :], qpos_partial], axis=0)
            if initial_qpos is not None
            else qpos_partial
        )
        result.update(
            {
                "status": "error",
                "error": f"{type(rollout_error).__name__}: {rollout_error}",
                "qvel_trace_rad_s": qvel_state.tolist(),
                "qvel_pre_trace_rad_s": np.asarray(
                    qvel_pre, dtype=np.float64
                ).tolist(),
                "qvel_post_trace_rad_s": qvel_partial.tolist(),
                "qpos_trace_rad": qpos_state.tolist(),
                "qpos_pre_trace_rad": np.asarray(
                    qpos_pre, dtype=np.float64
                ).tolist(),
                "qpos_post_trace_rad": qpos_partial.tolist(),
                "head_pose_m_quat": np.asarray(
                    head_pose, dtype=np.float64
                ).tolist(),
                "head_position_m": (
                    np.asarray(head_pose, dtype=np.float64)[:, :3].tolist()
                    if head_pose
                    else []
                ),
                "head_twist_linear_angular": np.asarray(
                    head_twist, dtype=np.float64
                ).tolist(),
                "nail_depth_m": [float(value) for value in depth],
                "raw_face_contact": [bool(value) for value in contact],
                "nonface_efc_force_trace": [
                    float(value) for value in nonface_force
                ],
                "exact_contrib_perjoint_trace": np.asarray(
                    exact_contrib, dtype=np.float64
                ).tolist(),
                "shipped_rolling_perjoint_trace": np.asarray(
                    shipped_rolling, dtype=np.float64
                ).tolist(),
                "shipped_contrib_perjoint_trace": np.asarray(
                    shipped_contrib, dtype=np.float64
                ).tolist(),
                "object_axial_force_n_trace": [
                    float(value) for value in object_axial_force
                ],
                "object_axial_contrib_n_s_trace": [
                    float(value) for value in object_axial_contrib
                ],
                "object_total_contrib_n_s_trace": [
                    float(value) for value in object_total_contrib
                ],
                "tracker_started_trace": [
                    bool(value) for value in tracker_started
                ],
                "quality": float(tracker._contact_quality[0].detach().cpu()),
                "quality_valid": bool(tracker._contact_quality_valid[0]),
                "quality_overflow": bool(
                    tracker._contact_quality_overflow[0]
                ),
                "quality_contact_point_w": tracker._contact_point_w[0]
                .detach()
                .cpu()
                .tolist(),
                "quality_error_m": float(
                    tracker._contact_error_m[0].detach().cpu()
                ),
                "quality_first_contact_time_s": float(
                    tracker._first_contact_time_s[0].detach().cpu()
                ),
                "quality_normal_axiality": float(
                    tracker._contact_normal_axiality[0].detach().cpu()
                ),
                "realized_action_tape": np.asarray(
                    realized_actions, dtype=np.float32
                ).tolist(),
                "n_substeps": len(qvel_post),
                "n_state_samples": len(qvel_state),
            }
        )
        result["deterministic_trace_digest"] = deterministic_trace_digest(result)
        env.close()
        return result

    try:
        expected_substeps = (30 + POST_TAPE_ZERO_ACTIONS) * int(
            env.cfg.decimation
        )
        arrays = {
            "qvel_pre": np.asarray(qvel_pre, dtype=np.float64),
            "qvel_post": np.asarray(qvel_post, dtype=np.float64),
            "qpos_pre": np.asarray(qpos_pre, dtype=np.float64),
            "qpos_post": np.asarray(qpos_post, dtype=np.float64),
            "effort": np.asarray(actuator_effort, dtype=np.float64),
            "head_pose": np.asarray(head_pose, dtype=np.float64),
            "head_twist": np.asarray(head_twist, dtype=np.float64),
            "depth": np.asarray(depth, dtype=np.float64),
            "contact": np.asarray(contact, dtype=bool),
            "nonface": np.asarray(nonface_force, dtype=np.float64),
            "exact": np.asarray(exact_contrib, dtype=np.float64),
            "object_axial_force": np.asarray(
                object_axial_force, dtype=np.float64
            ),
            "object_axial": np.asarray(
                object_axial_contrib, dtype=np.float64
            ),
            "object_total": np.asarray(
                object_total_contrib, dtype=np.float64
            ),
            "rolling": np.asarray(shipped_rolling, dtype=np.float64),
            "shipped": np.asarray(shipped_contrib, dtype=np.float64),
            "started": np.asarray(tracker_started, dtype=bool),
        }
        for name, values in arrays.items():
            if len(values) != expected_substeps:
                raise RuntimeError(
                    f"{name}: expected {expected_substeps} substeps, got "
                    f"{len(values)}"
                )
            if name != "contact" and name != "started":
                if not np.isfinite(values).all():
                    raise RuntimeError(f"{name} contains non-finite values")
        if initial_qvel is None or initial_qpos is None or hard_limits is None:
            raise RuntimeError("initial state was not captured")
        qvel_state = np.concatenate(
            [initial_qvel[None, :], arrays["qvel_post"]], axis=0
        )
        qpos_state = np.concatenate(
            [initial_qpos[None, :], arrays["qpos_post"]], axis=0
        )
        qvel_legal, qvel_peak = strict_qvel_legal(
            qvel_state, rail=HARDWARE_QVEL_LIMIT_RAD_S
        )
        qpos_legal = strict_qpos_legal(qpos_state, limits=hard_limits)
        saturation = actuator_saturation_summary(
            arrays["effort"], limits=effort_limits
        )
        actions_a = np.asarray(realized_actions, dtype=np.float32)
        realized_digest = _action_digest(actions_a)
        assert_legacy_realized_digest(
            action_mode=action_mode,
            expected=identity["action_digest"],
            realized=realized_digest,
        )
        result.update(
            {
                "status": "no_contact",
                "contact_seen": bool(arrays["contact"].any()),
                "qvel_legal": qvel_legal,
                "qvel_legal_95pct": bool(
                    np.isfinite(qvel_peak)
                    and qvel_peak
                    <= QVEL_MARGIN_FRACTION * HARDWARE_QVEL_LIMIT_RAD_S
                ),
                "qvel_peak_rad_s": qvel_peak,
                "qvel_trace_rad_s": qvel_state.tolist(),
                "qvel_pre_trace_rad_s": arrays["qvel_pre"].tolist(),
                "qvel_post_trace_rad_s": arrays["qvel_post"].tolist(),
                "qpos_legal": qpos_legal,
                "qpos_trace_rad": qpos_state.tolist(),
                "qpos_pre_trace_rad": arrays["qpos_pre"].tolist(),
                "qpos_post_trace_rad": arrays["qpos_post"].tolist(),
                "actuator_saturation_fraction": saturation["overall"],
                "actuator_saturation_perjoint": saturation["perjoint"],
                "head_pose_m_quat": arrays["head_pose"].tolist(),
                "head_position_m": arrays["head_pose"][:, :3].tolist(),
                "head_twist_linear_angular": arrays["head_twist"].tolist(),
                "nail_depth_m": arrays["depth"].tolist(),
                "raw_face_contact": arrays["contact"].tolist(),
                "nonface_efc_force_trace": arrays["nonface"].tolist(),
                "exact_contrib_perjoint_trace": arrays["exact"].tolist(),
                "shipped_rolling_perjoint_trace": arrays["rolling"].tolist(),
                "shipped_contrib_perjoint_trace": arrays["shipped"].tolist(),
                "object_axial_force_n_trace": arrays[
                    "object_axial_force"
                ].tolist(),
                "object_axial_contrib_n_s_trace": arrays[
                    "object_axial"
                ].tolist(),
                "object_total_contrib_n_s_trace": arrays[
                    "object_total"
                ].tolist(),
                "tracker_started_trace": arrays["started"].tolist(),
                "n_substeps": expected_substeps,
                "n_state_samples": expected_substeps + 1,
                "realized_action_tape": actions_a.tolist(),
                "realized_action_digest": realized_digest,
                "action_digest": realized_digest,
                "object_axial_impulse_total": float(
                    arrays["object_axial"].sum()
                ),
                "object_total_impulse_total": float(
                    arrays["object_total"].sum()
                ),
                "exact_contrib_max_n_m_s": float(
                    arrays["exact"].max(initial=0.0)
                ),
                "exact_total_perjoint": arrays["exact"].sum(axis=0).tolist(),
                "nonface_max_efc_force": float(
                    arrays["nonface"].max(initial=0.0)
                ),
                "quality": float(
                    tracker._contact_quality[0].detach().cpu()
                ),
                "quality_valid": bool(
                    tracker._contact_quality_valid[0]
                ),
                "quality_overflow": bool(
                    tracker._contact_quality_overflow[0]
                ),
                "quality_contact_point_w": tracker._contact_point_w[0]
                .detach()
                .cpu()
                .tolist(),
                "quality_error_m": float(
                    tracker._contact_error_m[0].detach().cpu()
                ),
                "quality_first_contact_time_s": float(
                    tracker._first_contact_time_s[0].detach().cpu()
                ),
                "quality_normal_axiality": float(
                    tracker._contact_normal_axiality[0].detach().cpu()
                ),
            }
        )
        max_exact_off = assert_exact_zero_off_raw_contact(
            arrays["contact"], arrays["exact"]
        )
        result["max_exact_off_raw_contact"] = max_exact_off
        live_peak = arrays["rolling"].max(axis=0)
        episode_peak = (
            acc._episode_peak_perjoint[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        result["production_peak_perjoint"] = episode_peak.tolist()
        if not np.allclose(live_peak, episode_peak, rtol=1e-5, atol=1e-7):
            raise RuntimeError(
                f"production rolling/episode peak mismatch: {live_peak} vs "
                f"{episode_peak}"
            )

        if not arrays["started"].any():
            result["impossible_success"] = bool(
                arrays["depth"].max(initial=0.0) >= 0.030
            )
            result["deterministic_trace_digest"] = deterministic_trace_digest(
                result
            )
            return result

        onset = int(np.flatnonzero(arrays["started"])[0])
        downward_speed = -arrays["head_twist"][:, 2]
        summary = summarize_trace(
            contact=arrays["contact"],
            downward_speed=downward_speed,
            depth=arrays["depth"],
            exact_contrib=arrays["exact"],
            shipped_peak=episode_peak,
            qvel=qvel_state,
            qpos_legal=qpos_legal,
            quality_valid=bool(tracker._contact_quality_valid[0]),
            quality_overflow=bool(tracker._contact_quality_overflow[0]),
            nonface_contact=arrays["nonface"],
            caps=np.asarray(IMP_J_LIMIT, dtype=np.float64),
            rail=HARDWARE_QVEL_LIMIT_RAD_S,
            accepted_onset=onset,
        )
        result.update(summary)
        result["status"] = "ok"
        release = result["release_index"]
        stop = int(release) if release is not None else len(arrays["contact"])
        diagnostics = contact_signal_liveness(
            contact=arrays["contact"],
            exact_contrib=arrays["exact"],
            object_total_contrib=arrays["object_total"],
            object_axial_contrib=arrays["object_axial"],
            onset=onset,
            stop_exclusive=stop,
        )
        _apply_contact_signal_diagnostics(
            result,
            diagnostics,
        )
        event_end = (
            int(release)
            if release is not None
            else min(len(arrays["contact"]), onset + 25)
        )
        event_force = arrays["object_axial_force"][onset:event_end]
        result["first_event_axial_force_peak_n"] = float(
            event_force.max(initial=0.0)
        )
        result["first_event_axial_force_mean_n"] = (
            float(event_force.mean()) if event_force.size else 0.0
        )
        result["impossible_success"] = bool(
            result["productive_30mm_50ms"] and not result["contact_seen"]
        )
        result["deterministic_trace_digest"] = deterministic_trace_digest(
            result
        )
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["eligible"] = False
        result["binding_class"] = BindingClass.INELIGIBLE.value
        result["deterministic_trace_digest"] = deterministic_trace_digest(
            result
        )
        return result
    finally:
        env.close()


def _collect_parity_prefix(
    *,
    actions: np.ndarray,
    reset_record: dict[str, object],
    delta: float,
    shadow: bool,
    control_steps: int | None = None,
) -> dict[str, object]:
    """Collect state history for the ordinary/shadow qualification only."""
    import torch
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from src.assets.robots.unitree_z1.z1_constants import (
        ARM_JOINT_NAMES,
        HAMMER_HEAD_SITE_NAME,
    )

    env = _build_stage0_env(
        delta, play=True, shadow=shadow, solref_scale=1.0
    )
    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    head_cfg.resolve(env.scene)
    arm_ids = arm_cfg.joint_ids
    head_id = head_cfg.site_ids[0]
    qpos: list[list[float]] = []
    qvel: list[list[float]] = []
    head: list[list[float]] = []
    depth: list[float] = []
    applied: list[list[float]] = []
    success_control_index: int | None = None
    original_substep = env.metrics_manager.compute_substep

    def snapshot() -> None:
        qpos.append(
            robot.data.joint_pos[0, arm_ids].detach().cpu().tolist()
        )
        qvel.append(
            robot.data.joint_vel[0, arm_ids].detach().cpu().tolist()
        )
        head.append(
            robot.data.site_pos_w[0, head_id].detach().cpu().tolist()
        )
        depth.append(float(nail.data.joint_pos[0, 0].detach().cpu()))

    def record_substep() -> None:
        original_substep()
        snapshot()

    env.metrics_manager.compute_substep = record_substep
    try:
        env.reset()
        _restore_canonical_reset(env, reset_record)
        snapshot()
        limit = len(actions) if control_steps is None else int(control_steps)
        for control_index, action in enumerate(actions[:limit]):
            tensor = torch.tensor(
                np.array(action, copy=True),
                dtype=torch.float32,
                device=env.device,
            )[None]
            applied.append(tensor[0].detach().cpu().tolist())
            env.step(tensor)
            if not shadow and int(env.episode_length_buf[0]) == 0:
                success_control_index = control_index
                break
    finally:
        env.metrics_manager.compute_substep = original_substep
        env.close()
    return {
        "actions": applied,
        "qpos": qpos,
        "qvel": qvel,
        "head_position": head,
        "nail_depth": depth,
        "success_control_index": success_control_index,
        "success_state_index": len(qpos) - 1,
    }


def _ordinary_shadow_parity_request(
    request: dict[str, object]
) -> dict[str, object]:
    identity = dict(request["identity"])
    result = _new_result(identity, status="error")
    actions = np.asarray(request["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"parity actions must be [T,3], got {actions.shape}")
    reset_record = request.get("reset_record")
    if not isinstance(reset_record, dict):
        raise ValueError("parity request missing reset_record")
    ordinary = _collect_parity_prefix(
        actions=actions,
        reset_record=reset_record,
        delta=float(identity["delta"]),
        shadow=False,
    )
    success = ordinary["success_control_index"]
    if success is None:
        raise RuntimeError("ordinary parity rollout did not reach success")
    shadow = _collect_parity_prefix(
        actions=actions,
        reset_record=reset_record,
        delta=float(identity["delta"]),
        shadow=True,
        control_steps=int(success) + 1,
    )
    shadow["success_control_index"] = success
    ordinary["success_state_index"] = len(ordinary["qpos"]) - 1
    shadow["success_state_index"] = ordinary["success_state_index"]
    assert_ordinary_shadow_prefix_parity(ordinary, shadow)
    result.update(
        {
            "status": "ok",
            "qualification": "ordinary_shadow_prefix_parity",
            "parity_passed": True,
        }
    )
    return result


def _execute_replay_request(request: dict[str, object]) -> dict[str, object]:
    kind = str(request.get("request_kind", "replay"))
    if kind == "replay":
        return _replay_action_request(request)
    if kind == "ordinary_shadow_parity":
        return _ordinary_shadow_parity_request(request)
    raise ValueError(f"unknown replay request kind {kind!r}")


def replay_saved_tape(tape: SavedTape) -> dict[str, object]:
    """Backwards-compatible direct diagnostic for one tape on the fixed reset."""
    reset = materialize_fixed_reset()
    identity = {
        "source_kind": "legacy",
        "source_id": f"{tape.source_path.name}:{tape.source_field}",
        "source_path": str(tape.source_path),
        "source_field": tape.source_field,
        "source_aliases": [list(alias) for alias in tape.aliases],
        "delta": tape.delta,
        "action_digest": tape.digest,
        "reset_digest": reset["reset_state_digest"],
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    return _replay_action_request(
        {
            "identity": identity,
            "action_mode": "tape",
            "actions": tape.actions.tolist(),
            "reset_record": reset,
        }
    )


def _reference_controller_spec() -> dict[str, object]:
    """Inspectable provenance for the active direct-reference controller."""
    from src.tasks.hammer.mdp.references import SingleStrikeReference

    constructor = inspect.signature(SingleStrikeReference)
    parameter_names = ("overshoot", "descent_speed", "axis_tol")
    actual_names = tuple(constructor.parameters)[2:]
    if actual_names != parameter_names:
        raise RuntimeError(
            "SingleStrikeReference constructor parameters drifted: "
            f"expected {parameter_names}, got {actual_names}"
        )
    parameters: dict[str, object] = {}
    for name in parameter_names:
        default = constructor.parameters[name].default
        if default is inspect.Parameter.empty:
            raise RuntimeError(
                f"SingleStrikeReference parameter {name!r} has no default"
            )
        parameters[name] = default

    return {
        "class": "src.tasks.hammer.mdp.references.SingleStrikeReference",
        "parameters": parameters,
        "geometry": {
            "start": "frozen reset hammer-head position",
            "target_xy": "frozen nail x/y",
            "target_z": (
                "min(frozen nail z - overshoot, frozen reset hammer-head z)"
            ),
            "path": "single direct segment with distance-paced linear interpolation",
        },
        "playback": {
            "target": "playback_target(min(control_index, playback_length))",
            "action": "clip((target - live_head) / delta, -1, 1)",
            "recorded_commands": 30,
            "post_zero_commands": POST_TAPE_ZERO_ACTIONS,
        },
    }


def _reference_controller_digest() -> str:
    return reset_bank_digest(_reference_controller_spec())


def _source_specs(tapes: list[SavedTape]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "source_kind": "reference",
            "source_id": "SingleStrikeReference-default",
            "source_path": "src/tasks/hammer/mdp/references.py",
            "source_field": "canonical_playback",
            "source_aliases": [],
            "delta": 0.15,
            "action_mode": "reference",
            "action_digest": _reference_controller_digest(),
            "actions": None,
        }
    ]
    for tape in tapes:
        specs.append(
            {
                "source_kind": "legacy",
                "source_id": f"{tape.source_path.name}:{tape.source_field}",
                "source_path": str(tape.source_path),
                "source_field": tape.source_field,
                "source_aliases": [list(alias) for alias in tape.aliases],
                "delta": tape.delta,
                "action_mode": "tape",
                "action_digest": tape.digest,
                "actions": tape.actions.tolist(),
            }
        )
    return specs


def _make_replay_request(
    source: dict[str, object],
    reset_record: dict[str, object],
    *,
    reset_role: str,
    solref_scale: float,
    exact_actions: list[list[float]] | None = None,
    exact_action_digest: str | None = None,
) -> dict[str, object]:
    identity = {
        key: source[key]
        for key in (
            "source_kind",
            "source_id",
            "source_path",
            "source_field",
            "source_aliases",
            "delta",
        )
    }
    identity.update(
        {
            "action_digest": (
                exact_action_digest
                if exact_action_digest is not None
                else source["action_digest"]
            ),
            "reset_digest": reset_record["reset_state_digest"],
            "reset_role": reset_role,
            "solref_scale": float(solref_scale),
        }
    )
    action_mode = (
        "tape" if exact_actions is not None else str(source["action_mode"])
    )
    actions = exact_actions if exact_actions is not None else source["actions"]
    return {
        "request_kind": "replay",
        "identity": identity,
        "action_mode": action_mode,
        "actions": actions,
        "reset_record": reset_record,
    }


def _mandatory_solref_failures(
    results: list[dict[str, object]],
) -> list[str]:
    """Return missing/failed required solref*2 pair identities."""
    failures: list[str] = []
    eligible_baselines = [
        row
        for row in results
        if row.get("source_kind") == "legacy"
        and float(row.get("solref_scale") or 0.0) == 1.0
        and bool(row.get("eligible"))
    ]
    for baseline in eligible_baselines:
        identity_keys = (
            "source_id",
            "delta",
            "action_digest",
            "reset_digest",
            "reset_role",
        )
        softened = [
            row
            for row in results
            if float(row.get("solref_scale") or 0.0) == 2.0
            and all(row.get(key) == baseline.get(key) for key in identity_keys)
        ]
        if (
            len(softened) != 1
            or softened[0].get("status") != "ok"
            or not isinstance(softened[0].get("solref_comparison"), dict)
            or not isinstance(baseline.get("solref_comparison"), dict)
        ):
            failures.append(
                f"{baseline.get('source_id')}@{baseline.get('reset_digest')}"
            )
    return failures


def _continuation_label(
    results: list[dict[str, object]],
    *,
    qualifications: list[dict[str, object]],
) -> dict[str, object]:
    fixed = [
        row
        for row in results
        if row.get("source_kind") == "legacy"
        and row.get("reset_role") == "fixed"
        and float(row.get("solref_scale") or 0.0) == 1.0
    ]
    incomplete_fixed = [
        str(row.get("source_id"))
        for row in fixed
        if row.get("status") not in {"ok", "no_contact"}
    ]
    if incomplete_fixed:
        return {
            "label": "ambiguous",
            "stage1": "requires_review",
            "basis": (
                "fixed baseline replay failure: "
                + ", ".join(sorted(incomplete_fixed))
            ),
        }

    failed_qualifications = [
        str(row.get("source_id"))
        for row in qualifications
        if row.get("status") != "ok"
    ]
    if failed_qualifications:
        return {
            "label": "ambiguous",
            "stage1": "requires_review",
            "basis": (
                "failed or incomplete deterministic/parity qualification: "
                + ", ".join(sorted(failed_qualifications))
            ),
        }

    expected_determinism = {
        f"determinism:{row.get('source_id')}" for row in fixed
    }
    observed_determinism = [
        str(row.get("source_id"))
        for row in qualifications
        if row.get("qualification") == "deterministic_replay"
        and str(row.get("source_id")) in expected_determinism
    ]
    if (
        set(observed_determinism) != expected_determinism
        or len(observed_determinism) != len(expected_determinism)
    ):
        return {
            "label": "ambiguous",
            "stage1": "requires_review",
            "basis": "fixed baseline determinism qualification missing or duplicated",
        }

    solref_failures = _mandatory_solref_failures(results)
    if solref_failures:
        return {
            "label": "ambiguous",
            "stage1": "requires_review",
            "basis": (
                "mandatory solref*2 replay/comparison missing or failed for "
                + ", ".join(solref_failures)
            ),
        }

    strict = [
        row
        for row in fixed
        if row.get("binding_class") == BindingClass.DUAL_BIND.value
    ]
    if strict:
        return {
            "label": "simulator_existence_proven",
            "stage1": "not_required_for_existence",
            "basis": "strict fixed-reset eligible dual bind",
        }
    near = [
        row
        for row in fixed
        if row.get("status") == "ok"
        and bool(row.get("eligible"))
        and float(row.get("rho_row28") or 0.0) >= 0.70
    ]
    if near:
        return {
            "label": "near_lead",
            "stage1": "one_scale_bounded_pilot_justified",
            "basis": "eligible fixed-reset rho_row28 >= 0.70",
        }
    stable_legal = [
        row
        for row in fixed
        if row.get("status") in {"ok", "no_contact"}
        and bool(row.get("qvel_legal"))
        and bool(row.get("qpos_legal"))
    ]
    if stable_legal and all(
        float(row.get("rho_row28") or 0.0) < 0.70 for row in stable_legal
    ):
        return {
            "label": "no_stage1_lead",
            "stage1": "do_not_run_from_stage0_evidence",
            "basis": "all stable legal fixed-reset candidates below 0.70",
            "not_a_physical_impossibility_claim": True,
        }
    return {
        "label": "ambiguous",
        "stage1": "requires_review",
        "basis": "failures or non-impact cases prevent a simple continuation rule",
    }


def _json_default(value: object):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("evaluation/whip/data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--supersedes-failed-artifact",
        type=Path,
        help="required authoritative A1 input: immutable failed attempt-1 JSON",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    previous_payload: dict[str, object] | None = None
    if not args.smoke:
        if args.supersedes_failed_artifact is None:
            raise RuntimeError(
                "authoritative A1 requires --supersedes-failed-artifact"
            )
        previous_path = args.supersedes_failed_artifact.resolve()
        previous_sha = _file_sha256(previous_path)
        if previous_sha != SUPERSEDES_FAILED_OUTPUT_SHA256:
            raise RuntimeError(
                "failed-attempt artifact SHA mismatch: "
                f"{previous_sha} != {SUPERSEDES_FAILED_OUTPUT_SHA256}"
            )
        previous_payload = json.loads(previous_path.read_text())
        prior_input = previous_payload.get("inputs_artifact")
        if (
            not isinstance(prior_input, dict)
            or prior_input.get("sha256") != SUPERSEDES_FAILED_INPUT_SHA256
        ):
            raise RuntimeError("failed-attempt input SHA is not the frozen A1 parent")
        from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

        asset_root = Path(Z1_HAMMER_XML).resolve().parents[2]
        _assert_authoritative_outside_repositories(
            args.out, code_repo=repo_root, asset_repo=asset_root
        )
    provenance_before = collect_provenance(
        repo_root=repo_root,
        data_dir=args.data_dir,
        require_clean=not args.smoke,
    )
    tapes = load_saved_tapes(args.data_dir)
    sources = _source_specs(tapes)
    fixed_reset, robustness_bank = materialize_stage0_resets()
    reset_envelope = {
        "schema": "lambda-feasibility-reset-bank-v1",
        "fixed_reset": fixed_reset,
        "robustness_bank_seed": RESET_BANK_SEED,
        "robustness_bank": robustness_bank,
        "interpretation": (
            "shared-tape robustness diagnostics only; not independent rows "
            "and not a veto on fixed-reset existence"
        ),
    }
    reset_envelope["reset_envelope_sha256"] = reset_bank_digest(reset_envelope)
    source_manifest = [
        {
            key: source[key]
            for key in (
                "source_kind",
                "source_id",
                "source_path",
                "source_field",
                "source_aliases",
                "delta",
                "action_mode",
                "action_digest",
            )
        }
        for source in sources
    ]
    input_envelope = {
        "schema": "lambda-feasibility-stage0-inputs-v2",
        "amendment": {
            "amendment_id": AMENDMENT_ID,
            "supersedes_failed_output_sha256": (
                SUPERSEDES_FAILED_OUTPUT_SHA256
            ),
            "supersedes_failed_input_sha256": SUPERSEDES_FAILED_INPUT_SHA256,
            "no_change_to": [
                "plant",
                "actions",
                "resets",
                "caps",
                "rewards",
                "contact_model",
                "eligibility_numeric_thresholds",
            ],
        },
        "provenance": provenance_before,
        "sources": source_manifest,
        "resets": reset_envelope,
    }
    input_envelope["inputs_sha256"] = reset_bank_digest(input_envelope)
    inputs_path = args.out.with_name(f"{args.out.stem}_inputs.json")
    if args.smoke:
        reset_cases = [("fixed", fixed_reset)]
        delta015 = next(
            source
            for source in sources
            if source["source_kind"] == "legacy"
            and float(source["delta"]) == 0.15
        )
        run_sources = [sources[0], delta015]
        input_artifact_sha = None
    else:
        input_artifact_sha = write_frozen_json(inputs_path, input_envelope)
        reset_cases = [("fixed", fixed_reset)] + [
            ("robustness", reset) for reset in robustness_bank
        ]
        run_sources = sources

    results: list[dict[str, object]] = []
    qualifications: list[dict[str, object]] = []
    fixed_reference: dict[str, object] | None = None
    total_baseline = len(reset_cases) * len(run_sources)
    completed = 0
    for reset_role, reset_record in reset_cases:
        for source in run_sources:
            completed += 1
            print(
                f"[{completed}/{total_baseline}] {source['source_id']} "
                f"delta={source['delta']} reset={reset_role}"
            )
            request = _make_replay_request(
                source,
                reset_record,
                reset_role=reset_role,
                solref_scale=1.0,
            )
            result = run_isolated_replay(request, timeout_s=args.timeout)
            results.append(result)
            print(
                "  "
                f"status={result['status']} "
                f"qvel={result.get('qvel_peak_rad_s')} "
                f"ship={result.get('rho_ship50')} "
                f"row26={result.get('rho_row26')} "
                f"row28={result.get('rho_row28')} "
                f"class={result['binding_class']}"
            )

            realized_actions = result.get("realized_action_tape")
            realized_digest = result.get("realized_action_digest")
            replay_completed = (
                result["status"] in {"ok", "no_contact"}
                and isinstance(realized_actions, list)
                and len(realized_actions) == 30
                and isinstance(realized_digest, str)
            )
            if reset_role == "fixed":
                if replay_completed:
                    deterministic_request = _make_replay_request(
                        source,
                        reset_record,
                        reset_role=reset_role,
                        solref_scale=1.0,
                        exact_actions=realized_actions,
                        exact_action_digest=realized_digest,
                    )
                else:
                    # A failed primary still gets the identical original
                    # request. It cannot pass determinism, but the repeat
                    # distinguishes a stable failure from a one-off worker
                    # failure and ensures the qualification row exists.
                    deterministic_request = request
                repeat = run_isolated_replay(
                    deterministic_request, timeout_s=args.timeout
                )
                qualifications.append(
                    _determinism_qualification(result, repeat)
                )

            if (
                reset_role == "fixed"
                and source["source_kind"] == "reference"
                and replay_completed
            ):
                fixed_reference = result

            if _requires_solref_replay(result) and replay_completed:
                soft_request = _make_replay_request(
                    source,
                    reset_record,
                    reset_role=reset_role,
                    solref_scale=2.0,
                    exact_actions=realized_actions,
                    exact_action_digest=realized_digest,
                )
                softened = run_isolated_replay(
                    soft_request, timeout_s=args.timeout
                )
                if softened["status"] == "ok":
                    try:
                        comparison = compare_solref_results(result, softened)
                        result["solref_comparison"] = comparison
                        softened["solref_comparison"] = comparison
                    except Exception as exc:
                        softened["status"] = "error"
                        softened["error"] = f"solref comparison failed: {exc}"
                results.append(softened)

    if fixed_reference is None:
        calibration = _new_result(
            {
                "source_kind": "qualification",
                "source_id": "reference-calibration",
                "source_path": "src/tasks/hammer/mdp/references.py",
                "source_field": "canonical_playback",
                "source_aliases": [],
                "delta": 0.15,
                "action_digest": _reference_controller_digest(),
                "reset_digest": fixed_reset["reset_state_digest"],
                "reset_role": "fixed",
                "solref_scale": 1.0,
            },
            status="error",
            error="fixed-reset reference did not produce a replayable tape",
        )
        calibration["qualification"] = "reference_calibration"
        qualifications.append(calibration)
        qualification = _new_result(
            {
                "source_kind": "qualification",
                "source_id": "ordinary-shadow-prefix-parity",
                "source_path": "src/tasks/hammer/config/z1/env_cfgs.py",
                "source_field": "nail_driven",
                "source_aliases": [],
                "delta": 0.15,
                "action_digest": _reference_controller_digest(),
                "reset_digest": fixed_reset["reset_state_digest"],
                "reset_role": "fixed",
                "solref_scale": 1.0,
            },
            status="error",
            error="fixed-reset reference did not produce a replayable tape",
        )
        qualification["qualification"] = "ordinary_shadow_prefix_parity"
        qualifications.append(qualification)
    else:
        qualifications.append(
            _reference_calibration_qualification(fixed_reference)
        )
        parity_identity = {
            "source_kind": "qualification",
            "source_id": "ordinary-shadow-prefix-parity",
            "source_path": "src/tasks/hammer/config/z1/env_cfgs.py",
            "source_field": "nail_driven",
            "source_aliases": [],
            "delta": 0.15,
            "action_digest": fixed_reference["realized_action_digest"],
            "reset_digest": fixed_reset["reset_state_digest"],
            "reset_role": "fixed",
            "solref_scale": 1.0,
        }
        parity_request = {
            "request_kind": "ordinary_shadow_parity",
            "identity": parity_identity,
            "actions": fixed_reference["realized_action_tape"],
            "reset_record": fixed_reset,
        }
        qualifications.append(
            run_isolated_replay(parity_request, timeout_s=args.timeout)
        )

    raw_equivalence_artifact: dict[str, object] | None = None
    if not args.smoke:
        assert previous_payload is not None
        previous_results = previous_payload.get("results")
        if not isinstance(previous_results, list):
            raise RuntimeError("failed-attempt artifact has no result list")
        raw_report = raw_physics_equivalence_report(previous_results, results)
        raw_report_path = args.out.with_name(
            f"{args.out.stem}_raw_equivalence.json"
        )
        raw_report_sha = write_frozen_json(raw_report_path, raw_report)
        raw_equivalence_artifact = {
            "path": str(raw_report_path),
            "sha256": raw_report_sha,
            "mismatch_n": raw_report["mismatch_n"],
            "identity_match": raw_report["identity_match"],
        }
        if (
            not bool(raw_report["identity_match"])
            or int(raw_report["mismatch_n"]) != 0
        ):
            raise RuntimeError(
                "attempt-2 raw physics differs from immutable attempt 1; "
                f"report={raw_report_path}"
            )

    aggregate = aggregate_results(results)
    aggregate["mandatory_solref_failure_n"] = len(
        _mandatory_solref_failures(results)
    )
    qualification_failure_n = sum(
        row.get("status") != "ok" for row in qualifications
    )
    provenance_after = collect_provenance(
        repo_root=repo_root,
        data_dir=args.data_dir,
        require_clean=not args.smoke,
    )
    if provenance_after != provenance_before:
        raise RuntimeError(
            "code/asset/tape provenance changed during Stage-0 replay"
        )
    payload = {
        "schema": "lambda-feasibility-stage0-v3",
        "amendment": input_envelope["amendment"],
        "raw_physics_equivalence_artifact": raw_equivalence_artifact,
        "authoritative": not args.smoke,
        "git_hash": provenance_before["code"]["revision"],
        "asset_hash": provenance_before["asset"]["revision"],
        "asset_scope_clean": provenance_before["asset_scope_clean"],
        "provenance": provenance_before,
        "inputs_artifact": {
            "path": str(inputs_path) if not args.smoke else None,
            "sha256": input_artifact_sha,
            "envelope_sha256": input_envelope["inputs_sha256"],
        },
        "caps_n_m_s": [1.640, 3.280, 1.640, 1.640, 1.640, 1.640],
        "imp_max_p": 0.0,
        "legacy_whip_allowlist": list(LEGACY_WHIP_FILES),
        "post_tape_zero_actions": POST_TAPE_ZERO_ACTIONS,
        "expected_control_steps": 30 + POST_TAPE_ZERO_ACTIONS,
        "expected_physics_substeps": (30 + POST_TAPE_ZERO_ACTIONS) * 10,
        "nonface_efc_force_tolerance": NONFACE_FORCE_TOL,
        "reset_bank_seed": RESET_BANK_SEED,
        "reset_bank_size": RESET_BANK_SIZE,
        "qualifications": qualifications,
        "qualification_failure_n": qualification_failure_n,
        "results": results,
        "aggregate": aggregate,
        "continuation": _continuation_label(
            results, qualifications=qualifications
        ),
        "interpretation_guardrails": {
            "robustness_resets_are_shared_tape_diagnostics": True,
            "robustness_resets_are_independent_samples": False,
            "robustness_failure_vetoes_fixed_reset_existence": False,
            "reference_is_calibration_control": True,
            "reference_in_binding_rates": False,
            "solref_low_change_is_necessary_not_sufficient": True,
            "solref_low_change_proves_hardware_fidelity": False,
        },
    }
    if args.smoke:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
        )
        print(f"[saved diagnostic, non-authoritative] {args.out}")
    else:
        output_sha = write_frozen_json(args.out, payload)
        print(f"[saved immutable evidence] {args.out} sha256={output_sha}")
        if (
            aggregate["impossible_success_n"] != 0
            or aggregate["lambda_dead_n"] != 0
            or aggregate["contact_signal_alignment_failure_n"] != 0
            or aggregate["operational_failure_n"] != 0
            or aggregate["mandatory_solref_failure_n"] != 0
            or qualification_failure_n != 0
        ):
            raise RuntimeError(
                "Stage-0 evidence was frozen but failed acceptance sentinels: "
                f"impossible_success_n={aggregate['impossible_success_n']}, "
                f"lambda_dead_n={aggregate['lambda_dead_n']}, "
                "contact_signal_alignment_failure_n="
                f"{aggregate['contact_signal_alignment_failure_n']}, "
                f"operational_failure_n={aggregate['operational_failure_n']}, "
                "mandatory_solref_failure_n="
                f"{aggregate['mandatory_solref_failure_n']}, "
                f"qualification_failure_n={qualification_failure_n}"
            )


if __name__ == "__main__":
    main()
