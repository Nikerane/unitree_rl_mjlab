"""Deterministic fixed-reset 500 Hz companion analysis.

This is a thin batch wrapper around :mod:`scripts.diag_impulse_trace`.  It
validates the frozen 56-checkpoint population, runs one isolated CPU process
per policy, and derives deterministic metrics from the recorder's native
500 Hz arrays.  It does not implement another environment driver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from evaluation.analysis.first_strike_campaign import (
    _contact_structure,
    _path_length_ratio,
    _qvel_diagnostics,
)
from evaluation.analysis.fixed_reset_video_library import (
    APPROVED_FIXED_RESET_DIGEST,
    EXPECTED,
    expected_task,
    load_fixed_reset,
    validate_inventory,
)
from evaluation.analysis.terminal_funnel import (
    WINDOW_SAMPLES_BEFORE,
    compute_episode_metrics,
)
from scripts.diag_impulse_trace import validate_trace_leaf
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
from src.tasks.hammer.mdp.first_strike import REASON_SUCCESS


SOURCE_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = SOURCE_ROOT / (
    "docs/results/assets/2026-07-29_56_policy_fixed_reset_library/"
    "checkpoint_inventory.tsv"
)
INVENTORY_SHA256 = (
    "e1417c677e5a86135a98ff2dc2ff47c5903f9fb236d45db0f8fef4f5914261f3"
)
FIXED_RESET_PATH = SOURCE_ROOT / (
    "docs/results/assets/2026-07-29_lambda_feasibility_stage0/"
    "lambda_feasibility_stage0_inputs.json"
)
POPULATION_COVARIATES_PATH = SOURCE_ROOT / (
    "docs/results/assets/2026-07-29_bounded_quality_3x8/"
    "fq3x8_confirmatory_analysis.json"
)
POPULATION_COVARIATES_SHA256 = (
    "b28a20328db21f8945bb0eb64f3efe2713dc6d936de958336a4b97d9682d5508"
)
OUTPUT_ROOT = SOURCE_ROOT / (
    "docs/results/assets/2026-07-29_56_policy_500hz_companion"
)
DEFAULT_CHECKPOINT_ROOTS = {
    "fq4x8": Path("/private/tmp/fq56_checkpoints/fq4x8"),
    "fq3x8": Path("/private/tmp/fq56_checkpoints/fq3x8"),
}

PHYSICS_DT_S = 0.002
CONTROL_DECIMATION = 10
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
REQUESTED_CONTROLS = 80
HARDWARE_QVEL_LIMIT_RAD_S = 3.1415
NAIL_RADIUS_M = 0.012
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
NO_HARDWARE_LEGAL_QUARTET = "NO_HARDWARE_LEGAL_QUARTET"
NO_DISTINCT_CURVATURE_QUARTET = "NO DISTINCT-CURVATURE QUARTET"
NO_MARGIN_MATCHED_QUARTET = "NO MARGIN-MATCHED QUARTET"
ARTIFACT_INVENTORY_NAME = "ARTIFACTS.sha256"

# These are the four frozen 50 Hz population margins.  Pair differences are
# divided by the corresponding margin and combined as Euclidean distance.
MATCHING_FIELDS = (
    (
        "first_contact_quality_sampled",
        "quality",
        0.10,
    ),
    (
        "first_window_useful_speed_mean_sampled",
        "first_window_useful_speed",
        0.20,
    ),
    (
        "event_window_depth_gain_mean_sampled",
        "depth",
        0.001,
    ),
    (
        "first_window_success_rate_sampled",
        "first_window_success",
        0.05,
    ),
)

_INVENTORY_REQUIRED = frozenset(
    (
        "campaign",
        "arm",
        "task",
        "training_seed",
        "checkpoint_cache_path",
        "checkpoint_sha256",
        "training_asset_revision",
    )
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lower_hex(value: object, length: int, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be {length}-character lowercase hex")
    return value


def _read_sidecar(path: str | Path, *, filename: str) -> str:
    try:
        tokens = Path(path).read_text(encoding="utf-8").strip().split()
    except OSError as error:
        raise ValueError(f"cannot read inventory SHA sidecar: {path}") from error
    if len(tokens) != 2 or tokens[1].lstrip("*") != filename:
        raise ValueError("inventory SHA sidecar has invalid format or filename")
    return _lower_hex(tokens[0], 64, label="inventory SHA sidecar")


def load_checkpoint_inventory(
    path: str | Path,
    *,
    checkpoint_roots: Mapping[str, str | Path],
    expected_sha256: str = INVENTORY_SHA256,
    sidecar_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Load and fail-close the exact 56-row checkpoint population.

    The historical absolute ``checkpoint_path`` column is deliberately
    ignored.  A local checkpoint is formed only as
    ``checkpoint_roots[campaign] / checkpoint_cache_path``.
    """

    inventory_path = Path(path)
    expected_digest = _lower_hex(
        expected_sha256, 64, label="expected inventory SHA-256"
    )
    try:
        actual_digest = _sha256(inventory_path)
    except OSError as error:
        raise ValueError(f"cannot read checkpoint inventory: {path}") from error
    if actual_digest != expected_digest:
        raise ValueError(
            "checkpoint inventory SHA-256 drift: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    if sidecar_path is not None:
        sidecar_digest = _read_sidecar(
            sidecar_path, filename=inventory_path.name
        )
        if sidecar_digest != expected_digest:
            raise ValueError("inventory SHA sidecar disagrees with frozen digest")

    try:
        with inventory_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("checkpoint inventory has no header")
            missing_columns = _INVENTORY_REQUIRED - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    "checkpoint inventory missing columns: "
                    f"{sorted(missing_columns)}"
                )
            raw_rows = [dict(row) for row in reader]
    except OSError as error:
        raise ValueError(f"cannot read checkpoint inventory: {path}") from error

    rows: list[dict[str, object]] = []
    for ordinal, raw in enumerate(raw_rows, start=1):
        try:
            campaign = str(raw["campaign"])
            arm = str(raw["arm"])
            seed_text = raw["training_seed"]
            if seed_text is None or not seed_text.isdecimal():
                raise ValueError("training_seed must be a nonnegative integer")
            seed = int(seed_text)
            task = str(raw["task"])
            cache_text = str(raw["checkpoint_cache_path"])
            checkpoint_sha = _lower_hex(
                raw["checkpoint_sha256"],
                64,
                label=f"row {ordinal} checkpoint SHA-256",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid checkpoint inventory row {ordinal}") from error

        if task != expected_task(campaign, arm):
            raise ValueError(
                f"checkpoint task drift for {campaign}/{arm}/seed{seed}: {task}"
            )
        if campaign not in checkpoint_roots:
            raise ValueError(f"no checkpoint root registered for {campaign}")
        relative = PurePath(cache_text)
        if (
            not cache_text
            or relative.is_absolute()
            or relative.drive
            or ".." in relative.parts
        ):
            reason = (
                "traversal is forbidden"
                if ".." in relative.parts
                else "path must be relative"
            )
            raise ValueError(
                f"checkpoint_cache_path {reason}: {cache_text!r}"
            )
        checkpoint = Path(checkpoint_roots[campaign]) / cache_text
        if not checkpoint.is_file():
            raise ValueError(f"missing checkpoint: {checkpoint}")
        actual_checkpoint_sha = _sha256(checkpoint)
        if actual_checkpoint_sha != checkpoint_sha:
            raise ValueError(
                "checkpoint SHA-256 drift for "
                f"{campaign}/{arm}/seed{seed}: expected {checkpoint_sha}, "
                f"got {actual_checkpoint_sha}"
            )

        row: dict[str, object] = dict(raw)
        row.update(
            {
                "campaign": campaign,
                "arm": arm,
                "training_seed": seed,
                "task": task,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_file": str(checkpoint),
            }
        )
        rows.append(row)

    # The shared validator checks both the exact registered identities and
    # unique checkpoint hashes.  Preserve its stable diagnostic wording.
    validate_inventory(rows)
    return rows


def _verified_caps() -> np.ndarray:
    caps = np.asarray(IMP_J_LIMIT, dtype=np.float64)
    frozen = np.asarray(
        [1.640, 3.280, 1.640, 1.640, 1.640, 1.640],
        dtype=np.float64,
    )
    if caps.shape != (6,) or not np.array_equal(caps, frozen):
        raise ValueError(
            "authoritative IMP_J_LIMIT order/value drifted from the frozen "
            "manufacturer caps"
        )
    return caps


def _identity_expectations(
    row: Mapping[str, object],
    *,
    fixed_reset_envelope: str | Path,
    code_revision: str,
    asset_revision: str,
) -> dict[str, object]:
    if row.get("training_asset_revision") != asset_revision:
        raise ValueError(
            "invocation asset revision differs from the frozen inventory row"
        )
    return {
        "mode": "checkpoint",
        "campaign": str(row["campaign"]),
        "arm": str(row["arm"]),
        "training_seed": int(row["training_seed"]),
        "task": str(row["task"]),
        "checkpoint_file": str(Path(str(row["checkpoint_file"])).resolve()),
        "checkpoint_sha256": str(row["checkpoint_sha256"]),
        "fixed_reset_envelope": str(fixed_reset_envelope),
        "reset_state_digest": APPROVED_FIXED_RESET_DIGEST,
        "code_revision": code_revision,
        "asset_revision": asset_revision,
        "j_limit_n_m_s": _verified_caps().tolist(),
    }


def validate_resume_leaf(
    leaf: str | Path,
    row: Mapping[str, object],
    *,
    fixed_reset_envelope: str | Path,
    code_revision: str,
    asset_revision: str,
    requested_controls: int = REQUESTED_CONTROLS,
) -> dict:
    """Validate Task 1's leaf and bind it to this exact invocation."""

    metadata = validate_trace_leaf(leaf)
    expected = _identity_expectations(
        row,
        fixed_reset_envelope=fixed_reset_envelope,
        code_revision=code_revision,
        asset_revision=asset_revision,
    )
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"resume leaf {field} mismatch: "
                f"expected {value!r}, got {metadata.get(field)!r}"
            )

    timing = metadata.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("resume leaf has no complete timing metadata")
    try:
        physics_dt = float(timing["physics_dt_s"])
        decimation = int(timing["control_decimation"])
        control_dt = float(timing["control_dt_s"])
        substeps = int(timing["substep_count"])
        controls = int(timing["control_step_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("resume leaf has incomplete step metadata") from error
    if (
        not math.isclose(
            physics_dt, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1e-12
        )
        or decimation != CONTROL_DECIMATION
        or not math.isclose(
            control_dt, CONTROL_DT_S, rel_tol=0.0, abs_tol=1e-12
        )
        or controls < 1
        or controls > requested_controls
        or substeps != controls * CONTROL_DECIMATION
    ):
        raise ValueError("resume leaf timing/control metadata is incomplete or drifted")
    terminal_reason = metadata.get("terminal_reason")
    if terminal_reason == "step_limit" and controls != requested_controls:
        raise ValueError(
            "step-limit leaf did not execute every requested control"
        )
    schema = metadata.get("schema")
    try:
        schema_substeps = int(schema["rates"]["substep"]["count"])
        schema_controls = int(schema["rates"]["control"]["count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("resume leaf schema lacks rate counts") from error
    if (schema_substeps, schema_controls) != (substeps, controls):
        raise ValueError("resume leaf schema/timing step counts disagree")
    return metadata


def load_trace_payload(leaf: str | Path) -> dict[str, np.ndarray]:
    """Load one already validated Task 1 NPZ without pickle support."""

    try:
        with np.load(Path(leaf) / "trace.npz", allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load trace payload: {leaf}") from error


def _stream(
    payload: Mapping[str, Any],
    key: str,
    sample_count: int,
    *,
    dtype: Any,
) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"trace missing {key}")
    values = np.asarray(payload[key], dtype=dtype)
    if values.shape != (sample_count,):
        raise ValueError(f"{key} must have shape ({sample_count},)")
    return values


def _rising_edge(values: np.ndarray, *, label: str) -> int | None:
    previous = np.concatenate((np.array([False]), values[:-1]))
    indices = np.flatnonzero(values & ~previous)
    if indices.size == 0:
        return None
    first = int(indices[0])
    if not np.all(values[first:]):
        raise ValueError(f"{label} stream must remain true after its first edge")
    return first


def _chord_distances(points: np.ndarray) -> np.ndarray:
    chord = points[-1] - points[0]
    length = float(np.linalg.norm(chord))
    if length <= 1e-12:
        return np.linalg.norm(points - points[0], axis=1)
    unit = chord / length
    offsets = points - points[0]
    orthogonal = offsets - np.outer(offsets @ unit, unit)
    return np.linalg.norm(orthogonal, axis=1)


def _descent_phenotype(points: np.ndarray) -> dict[str, object]:
    """Return the rubric's support-gated 3-D descent curvature phenotype."""

    invalid = {
        "descent_geometry_valid": False,
        "descent_geometry_invalid_reason": "",
        "descent_chord_length_m": None,
        "descent_c_rms": None,
        "descent_b_rms_m": None,
        "descent_c_max": None,
        "descent_tortuosity": None,
        "descent_progress_sign_changes": None,
    }
    if np.unique(points, axis=0).shape[0] < 3:
        invalid["descent_geometry_invalid_reason"] = (
            "fewer_than_three_distinct_samples"
        )
        return invalid
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    invalid["descent_chord_length_m"] = chord_length
    if chord_length < 0.050:
        invalid["descent_geometry_invalid_reason"] = "chord_below_0.050m"
        return invalid

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = np.empty(len(points), dtype=np.float64)
    weights[0] = segment_lengths[0] / 2.0
    weights[-1] = segment_lengths[-1] / 2.0
    weights[1:-1] = (
        segment_lengths[:-1] + segment_lengths[1:]
    ) / 2.0
    distances = _chord_distances(points)
    b_rms = float(
        np.sqrt(np.sum(weights * np.square(distances)) / np.sum(weights))
    )
    unit = chord / chord_length
    progress = np.diff((points - points[0]) @ unit)
    nonzero_signs = np.sign(progress[progress != 0.0])
    sign_changes = int(
        np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1])
    )
    return {
        "descent_geometry_valid": True,
        "descent_geometry_invalid_reason": "",
        "descent_chord_length_m": chord_length,
        "descent_c_rms": b_rms / chord_length,
        "descent_b_rms_m": b_rms,
        "descent_c_max": float(np.max(distances) / chord_length),
        "descent_tortuosity": float(
            np.sum(segment_lengths) / chord_length
        ),
        "descent_progress_sign_changes": sign_changes,
    }


def derive_episode_metrics(
    payload: Mapping[str, Any],
    *,
    dt_s: float = PHYSICS_DT_S,
) -> dict[str, object]:
    """Derive one policy's pure 500 Hz fixed-reset episode metrics."""

    if not math.isclose(
        float(dt_s), PHYSICS_DT_S, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("companion analysis requires exact physics_dt=0.002")
    contact = np.asarray(payload.get("contact"), dtype=bool)
    if contact.ndim != 1 or contact.size == 0:
        raise ValueError("contact must be a non-empty one-dimensional array")
    sample_count = int(contact.size)
    positions = np.asarray(payload.get("head_position_m"), dtype=np.float64)
    velocities = np.asarray(payload.get("head_velocity_m_s"), dtype=np.float64)
    nail_positions = np.asarray(payload.get("nail_position_m"), dtype=np.float64)
    force = np.asarray(payload.get("axial_force_n"), dtype=np.float64)
    pre_qvel = np.asarray(payload.get("joint_velocity_rad_s"), dtype=np.float64)
    post_qvel = np.asarray(
        payload.get("joint_velocity_post_integration_rad_s"),
        dtype=np.float64,
    )
    lambda_read = np.asarray(
        payload.get("lambda_windowed_constraint_read_n_m_s"),
        dtype=np.float64,
    )
    expected_shapes = {
        "head_position_m": (positions, (sample_count, 3)),
        "head_velocity_m_s": (velocities, (sample_count, 3)),
        "nail_position_m": (nail_positions, (sample_count, 3)),
        "axial_force_n": (force, (sample_count,)),
        "joint_velocity_rad_s": (pre_qvel, (sample_count, 6)),
        "joint_velocity_post_integration_rad_s": (
            post_qvel,
            (sample_count, 6),
        ),
        "lambda_windowed_constraint_read_n_m_s": (
            lambda_read,
            (sample_count, 6),
        ),
    }
    for name, (values, shape) in expected_shapes.items():
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name, values in (
        ("head_position_m", positions),
        ("head_velocity_m_s", velocities),
        ("nail_position_m", nail_positions),
        ("axial_force_n", force),
        ("lambda_windowed_constraint_read_n_m_s", lambda_read),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
    if np.any(force < 0.0) or np.any(lambda_read < 0.0):
        raise ValueError("force and Lambda channels must be nonnegative")

    first_strike_available = bool(
        np.asarray(payload.get("first_strike_available", False))
    )
    quality_available = bool(
        np.asarray(payload.get("quality_available", False))
    )
    if quality_available and not first_strike_available:
        raise ValueError("quality cannot be available without first-strike tracking")

    if first_strike_available:
        started = _stream(
            payload,
            "tracker_started",
            sample_count,
            dtype=bool,
        )
        finalized = _stream(
            payload,
            "tracker_finalized",
            sample_count,
            dtype=bool,
        )
        productive_stream = _stream(
            payload,
            "tracker_productive",
            sample_count,
            dtype=bool,
        )
        reason_stream = _stream(
            payload,
            "tracker_reason",
            sample_count,
            dtype=np.int64,
        )
        accepted_onset = _rising_edge(started, label="tracker_started")
        finalization_index = _rising_edge(
            finalized, label="tracker_finalized"
        )
    else:
        started = np.zeros(sample_count, dtype=bool)
        finalized = np.zeros(sample_count, dtype=bool)
        productive_stream = np.zeros(sample_count, dtype=bool)
        reason_stream = np.zeros(sample_count, dtype=np.int64)
        accepted_onset = None
        finalization_index = None

    if (
        finalization_index is not None
        and (
            accepted_onset is None
            or finalization_index < accepted_onset
        )
    ):
        raise ValueError("tracker finalization must follow accepted onset")
    onset, post_event_recontact, tail_fraction = _contact_structure(
        contact,
        force,
        dt_s,
        accepted_onset=accepted_onset,
        finalization_index=finalization_index,
    )
    qvel = _qvel_diagnostics(
        pre_qvel,
        post_qvel,
        dt_s=dt_s,
        limit_rad_s=HARDWARE_QVEL_LIMIT_RAD_S,
        accepted_onset=accepted_onset,
        finalization_index=finalization_index,
    )
    caps = _verified_caps()
    lambda_peaks = np.max(lambda_read, axis=0)
    lambda_ratios = lambda_peaks / caps
    worst_joint_index = int(np.argmax(lambda_ratios))

    apex_index: int | None = None
    path_ratio: float | None = None
    chord_rms: float | None = None
    chord_max: float | None = None
    onset_downward: float | None = None
    onset_lateral: float | None = None
    onset_angle: float | None = None
    terminal: dict[str, Any] | None = None
    descent = {
        "descent_geometry_valid": False,
        "descent_geometry_invalid_reason": "no_accepted_onset",
        "descent_chord_length_m": None,
        "descent_c_rms": None,
        "descent_b_rms_m": None,
        "descent_c_max": None,
        "descent_tortuosity": None,
        "descent_progress_sign_changes": None,
    }
    if onset is not None:
        precontact_z = positions[: onset + 1, 2]
        maximum_z = float(np.max(precontact_z))
        apex_index = int(np.flatnonzero(precontact_z == maximum_z)[-1])
        apex_window = positions[apex_index : onset + 1]
        path_ratio = float(_path_length_ratio(apex_window))
        distances = _chord_distances(apex_window)
        chord_rms = float(np.sqrt(np.mean(np.square(distances))))
        chord_max = float(np.max(distances))
        descent = _descent_phenotype(apex_window)

        onset_velocity = velocities[onset]
        onset_downward = float(-onset_velocity[2])
        onset_lateral = float(np.linalg.norm(onset_velocity[:2]))
        onset_angle = (
            float(
                np.degrees(
                    np.arctan2(onset_lateral, onset_downward)
                )
            )
            if onset_downward > 0.0
            else None
        )
        if onset >= WINDOW_SAMPLES_BEFORE:
            window = positions[
                onset - WINDOW_SAMPLES_BEFORE : onset + 1
            ]
            terminal = compute_episode_metrics(
                window,
                nail_xy=nail_positions[onset, :2],
                dt_s=dt_s,
                tolerance_m=NAIL_RADIUS_M,
            )

    stop = (
        sample_count - 1
        if onset is not None and finalization_index is None
        else finalization_index
    )
    release_index: int | None = None
    live_duration = 0.0
    duty = 0.0
    force_peak = 0.0
    force_concentration = 0.0
    if onset is not None and stop is not None:
        event_contact = contact[onset : stop + 1]
        live_duration = float(np.count_nonzero(event_contact) * dt_s)
        duty = float(np.mean(event_contact))
        event_force = force[onset : stop + 1]
        force_peak = float(np.max(event_force))
        force_sum = float(np.sum(event_force))
        force_concentration = force_peak / force_sum if force_sum > 0 else 0.0
        releases = np.flatnonzero(~contact[onset + 1 :])
        if releases.size:
            release_index = onset + 1 + int(releases[0])
    accepted_run_duration = (
        0.0
        if onset is None
        else (
            (release_index - onset) * dt_s
            if release_index is not None
            else (sample_count - onset) * dt_s
        )
    )

    productive = (
        bool(productive_stream[-1]) if first_strike_available else False
    )
    reason = int(reason_stream[-1]) if first_strike_available else 0
    quality: float | None = None
    quality_valid = False
    quality_overflow = False
    quality_error: float | None = None
    quality_axiality: float | None = None
    if quality_available:
        snapshot_index = sample_count - 1 if onset is None else onset
        quality_arrays = {
            key: _stream(payload, key, sample_count, dtype=dtype)
            for key, dtype in (
                ("tracker_contact_quality", np.float64),
                ("tracker_contact_quality_valid", bool),
                ("tracker_contact_quality_overflow", bool),
                ("tracker_contact_error_m", np.float64),
                ("tracker_contact_normal_axiality", np.float64),
            )
        }
        quality = float(
            quality_arrays["tracker_contact_quality"][snapshot_index]
        )
        quality_valid = bool(
            quality_arrays["tracker_contact_quality_valid"][snapshot_index]
        )
        quality_overflow = bool(
            quality_arrays["tracker_contact_quality_overflow"][snapshot_index]
        )
        quality_error = float(
            quality_arrays["tracker_contact_error_m"][snapshot_index]
        )
        quality_axiality = float(
            quality_arrays["tracker_contact_normal_axiality"][snapshot_index]
        )

    raw_edges = int(
        np.count_nonzero(
            contact
            & ~np.concatenate((np.array([False]), contact[:-1]))
        )
    )
    finite_trace = bool(
        all(
            np.isfinite(np.asarray(value)).all()
            for value in payload.values()
            if np.asarray(value).dtype.kind in "biufc"
        )
    )
    result: dict[str, object] = {
        "trace_sample_count": sample_count,
        "trace_duration_s": sample_count * dt_s,
        "finite_trace": finite_trace,
        "first_strike_available": first_strike_available,
        "accepted_onset_index": onset,
        "accepted_onset_time_s": (
            None if onset is None else onset * dt_s
        ),
        "finalization_index": finalization_index,
        "finalization_time_s": (
            None
            if finalization_index is None
            else finalization_index * dt_s
        ),
        "tracker_finalized": finalization_index is not None,
        "tracker_productive": productive,
        "tracker_reason": reason,
        "tracker_success": reason == REASON_SUCCESS,
        "raw_contact_rising_edge_count": raw_edges,
        "release_index": release_index,
        "release_after_accepted_onset": release_index is not None,
        "accepted_contact_run_duration_s": accepted_run_duration,
        "first_strike_live_contact_duration_s": live_duration,
        "post_event_recontact": post_event_recontact,
        "tail_force_time_fraction": tail_fraction,
        "realized_apex_index": apex_index,
        "apex_to_onset_path_ratio_3d": path_ratio,
        "apex_to_onset_chord_rms_m": chord_rms,
        "apex_to_onset_chord_max_m": chord_max,
        **descent,
        "onset_downward_axial_velocity_m_s": onset_downward,
        "onset_lateral_velocity_m_s": onset_lateral,
        "onset_approach_angle_deg": onset_angle,
        "terminal_funnel_available": terminal is not None,
        "terminal_contraction_ratio": (
            None if terminal is None else terminal["contraction_ratio"]
        ),
        "terminal_error_m": (
            None if terminal is None else terminal["terminal_error_m"]
        ),
        "terminal_late_max_error_m": (
            None if terminal is None else terminal["late_max_error_m"]
        ),
        "terminal_classification": (
            None if terminal is None else terminal["classification"]
        ),
        "terminal_late_reexpansion": (
            False
            if terminal is None
            else terminal["late_max_error_m"] > 1.25 * NAIL_RADIUS_M
        ),
        "qvel_canonical_sample_count": sample_count + 1,
        **qvel,
        "quality_available": quality_available,
        "quality": quality,
        "quality_valid": quality_valid,
        "quality_overflow": quality_overflow,
        "quality_contact_error_m": quality_error,
        "quality_contact_normal_axiality": quality_axiality,
        # These are deliberately descriptors, not a binary impact/press label.
        "descriptor_live_contact_duration_s": live_duration,
        "descriptor_release_after_accepted_onset": release_index is not None,
        "descriptor_force_peak_n": force_peak,
        "descriptor_force_time_concentration_peak_sample_fraction": (
            force_concentration
        ),
        "descriptor_contact_duty_first_strike_window": duty,
        "impact_press_fields_are_descriptors_only": True,
    }
    for index, (peak, ratio) in enumerate(
        zip(lambda_peaks, lambda_ratios, strict=True), start=1
    ):
        result[f"lambda_peak_joint{index}_n_m_s"] = float(peak)
        result[f"lambda_cap_ratio_joint{index}"] = float(ratio)
    result["lambda_cap_ratio_worst"] = float(lambda_ratios[worst_joint_index])
    result["lambda_cap_ratio_worst_joint"] = JOINT_NAMES[worst_joint_index]
    return result


def hardware_eligibility(
    row: Mapping[str, object],
) -> tuple[bool, list[str]]:
    """Return the formal hardware gate and its deterministic failure reasons."""

    checks = (
        ("accepted_onset", row.get("accepted_onset_index") is not None),
        ("tracker_finalized", bool(row.get("tracker_finalized"))),
        ("productive", bool(row.get("tracker_productive"))),
        (
            "reason_success",
            int(row.get("tracker_reason", 0)) == REASON_SUCCESS,
        ),
        ("quality_available", bool(row.get("quality_available"))),
        ("quality_valid", bool(row.get("quality_valid"))),
        ("quality_overflow", not bool(row.get("quality_overflow"))),
        ("finite_trace", bool(row.get("finite_trace"))),
        ("qvel_rail", not bool(row.get("qvel_violation"))),
    )
    failures = [name for name, passed in checks if not passed]
    return not failures, failures


_POPULATION_SEED_FIELDS = frozenset(
    (
        "event_window_depth_gain_mean_sampled",
        "first_contact_quality_sampled",
        "first_window_success_rate_sampled",
        "first_window_useful_speed_mean_sampled",
        "overall_success_rate_sampled",
    )
)


def _validate_population_covariates(payload: Mapping[str, Any]) -> None:
    expected_seeds = list(range(16, 24))
    if (
        payload.get("schema_version") != 1
        or payload.get("campaign") != "fq3x8"
        or payload.get("seeds") != expected_seeds
    ):
        raise ValueError("population covariates have invalid fq3x8 schema")
    try:
        records = payload["seed_metrics"]["FQ"]
    except (KeyError, TypeError) as error:
        raise ValueError("population covariates have no FQ seed metrics") from error
    if not isinstance(records, Mapping) or set(records) != {
        str(seed) for seed in expected_seeds
    }:
        raise ValueError("population covariates must contain exact FQ seeds 16-23")
    for seed in expected_seeds:
        record = records[str(seed)]
        if not isinstance(record, Mapping) or set(record) != _POPULATION_SEED_FIELDS:
            raise ValueError(
                f"population covariates have invalid FQ seed {seed} schema"
            )
        values = np.asarray(list(record.values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite population covariates for seed {seed}")


def load_population_covariates(
    path: str | Path,
    *,
    expected_sha256: str = POPULATION_COVARIATES_SHA256,
) -> dict[str, Any]:
    """Load the byte-frozen FQ population covariates and exact seed schema."""

    expected = _lower_hex(
        expected_sha256, 64, label="population covariate SHA-256"
    )
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            "population covariate SHA-256 drift: "
            f"expected {expected}, got {actual}"
        )
    payload = _load_json_object(path)
    _validate_population_covariates(payload)
    return payload


def _covariates_for_seeds(
    payload: Mapping[str, Any], seeds: Sequence[int]
) -> dict[int, dict[str, float]]:
    _validate_population_covariates(payload)
    try:
        records = payload["seed_metrics"]["FQ"]
    except (KeyError, TypeError) as error:
        raise ValueError("population covariates have no FQ seed metrics") from error
    result: dict[int, dict[str, float]] = {}
    for seed in seeds:
        try:
            raw = records[str(seed)]
            values = {
                key: float(raw[key])
                for key, _label, _margin in MATCHING_FIELDS
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"population covariates missing FQ seed {seed}"
            ) from error
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"nonfinite population covariates for seed {seed}")
        result[seed] = values
    return result


def _pair_distance(
    lower_seed: int,
    higher_seed: int,
    covariates: Mapping[int, Mapping[str, float]],
) -> tuple[float, dict[str, float]]:
    standardized: dict[str, float] = {}
    squared = 0.0
    for field, label, margin in MATCHING_FIELDS:
        difference = (
            float(covariates[lower_seed][field])
            - float(covariates[higher_seed][field])
        )
        value = difference / margin
        standardized[label] = value
        squared += value * value
    return math.sqrt(squared), standardized


def rank_and_pair_fq(
    rows: Sequence[Mapping[str, object]],
    population_covariates: Mapping[str, Any],
) -> dict[str, object]:
    """Rank fq3x8/FQ and select two deterministic cross-curvature pairs."""

    _validate_population_covariates(population_covariates)
    fq_rows = [
        dict(row)
        for row in rows
        if row.get("campaign") == "fq3x8" and row.get("arm") == "FQ"
    ]
    identities = [int(row["training_seed"]) for row in fq_rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate fq3x8/FQ training seed")

    def has_phenotype(row: Mapping[str, object]) -> bool:
        values = (
            row.get("descent_c_rms"),
            row.get("descent_b_rms_m"),
            row.get("descent_tortuosity"),
        )
        return bool(row.get("descent_geometry_valid")) and all(
            value is not None and np.isfinite(float(value))
            for value in values
        )

    def curvature_key(
        row: Mapping[str, object],
    ) -> tuple[bool, float, float, str]:
        valid = has_phenotype(row)
        return (
            not valid,
            float(row["descent_c_rms"]) if valid else math.inf,
            float(row["descent_tortuosity"]) if valid else math.inf,
            str(row["checkpoint_sha256"]),
        )

    ordered = sorted(fq_rows, key=curvature_key)
    simulation: list[dict[str, object]] = []
    for rank, row in enumerate(ordered, start=1):
        output = dict(row)
        output["simulation_rank"] = rank
        eligible, failures = hardware_eligibility(row)
        output["hardware_eligible"] = eligible
        output["hardware_ineligibility_reasons"] = ";".join(failures)
        simulation.append(output)
    hardware_source = [
        row for row in ordered if hardware_eligibility(row)[0]
    ]
    hardware: list[dict[str, object]] = []
    geometry: list[dict[str, object]] = []
    geometry_rank = 0
    for row in hardware_source:
        output = dict(row)
        output["hardware_eligible"] = True
        output["hardware_ineligibility_reasons"] = ""
        if has_phenotype(row):
            geometry_rank += 1
            output["hardware_rank"] = geometry_rank
            output["selection_exclusion_reason"] = ""
            geometry.append(output)
        else:
            output["hardware_rank"] = None
            invalid_reason = str(
                row.get("descent_geometry_invalid_reason")
                or "invalid_or_nonfinite_metrics"
            )
            output["selection_exclusion_reason"] = (
                f"unsupported_descent_phenotype:{invalid_reason}"
            )
        hardware.append(output)

    formal_hardware_count = len(hardware)

    def no_pairing(status: str) -> dict[str, object]:
        return {
            "status": status,
            "eligible_count": formal_hardware_count,
            "geometry_eligible_count": len(geometry),
            "curvature_metric": "descent_c_rms",
            "population_covariates_sha256": POPULATION_COVARIATES_SHA256,
            "matching_semantics": (
                "each absolute covariate difference standardized by its "
                "frozen 50 Hz margin must be <= 1"
            ),
            "pairs": [],
        }

    if formal_hardware_count < 4:
        pairing = no_pairing(NO_HARDWARE_LEGAL_QUARTET)
        return {
            "simulation_ranking": simulation,
            "hardware_ranking": hardware,
            "pairing": pairing,
        }

    m = len(geometry) // 2
    lower = geometry[:m]
    higher = geometry[-m:] if m else []
    middle = geometry[m : len(geometry) - m]
    base_pairing = {
        "eligible_count": formal_hardware_count,
        "geometry_eligible_count": len(geometry),
        "curvature_metric": "descent_c_rms",
        "population_covariates_sha256": POPULATION_COVARIATES_SHA256,
        "lower_half_seeds": [int(row["training_seed"]) for row in lower],
        "higher_half_seeds": [int(row["training_seed"]) for row in higher],
        "unassigned_middle_seeds": [
            int(row["training_seed"]) for row in middle
        ],
        "matching_semantics": (
            "each absolute covariate difference standardized by its frozen "
            "50 Hz margin must be <= 1"
        ),
        "matching_margins": {
            label: margin for _field, label, margin in MATCHING_FIELDS
        },
    }
    if len(lower) < 2 or len(higher) < 2:
        pairing = {
            **base_pairing,
            "status": NO_DISTINCT_CURVATURE_QUARTET,
            "pairs": [],
        }
        return {
            "simulation_ranking": simulation,
            "hardware_ranking": hardware,
            "pairing": pairing,
        }

    covariates = _covariates_for_seeds(
        population_covariates,
        [int(row["training_seed"]) for row in geometry],
    )
    distinct_count = 0
    candidates: list[tuple[tuple[Any, ...], list[dict[str, object]]]] = []
    for lower_pair in itertools.combinations(lower, 2):
        for higher_pair in itertools.combinations(higher, 2):
            for ordered_higher in itertools.permutations(higher_pair):
                pairs: list[dict[str, object]] = []
                standardized_values: list[float] = []
                separations: list[float] = []
                for low, high in zip(
                    lower_pair, ordered_higher, strict=True
                ):
                    low_seed = int(low["training_seed"])
                    high_seed = int(high["training_seed"])
                    distance, standardized = _pair_distance(
                        low_seed, high_seed, covariates
                    )
                    b_separation = abs(
                        float(high["descent_b_rms_m"])
                        - float(low["descent_b_rms_m"])
                    )
                    standardized_values.extend(standardized.values())
                    separations.append(
                        float(high["descent_c_rms"])
                        - float(low["descent_c_rms"])
                    )
                    pairs.append(
                        {
                            "lower_seed": low_seed,
                            "higher_seed": high_seed,
                            "lower_curvature": float(
                                low["descent_c_rms"]
                            ),
                            "higher_curvature": float(
                                high["descent_c_rms"]
                            ),
                            "lower_b_rms_m": float(
                                low["descent_b_rms_m"]
                            ),
                            "higher_b_rms_m": float(
                                high["descent_b_rms_m"]
                            ),
                            "b_rms_separation_m": b_separation,
                            "standardized_margin_distance": distance,
                            "standardized_differences": standardized,
                            "lower_checkpoint_sha256": str(
                                low["checkpoint_sha256"]
                            ),
                            "higher_checkpoint_sha256": str(
                                high["checkpoint_sha256"]
                            ),
                        }
                    )
                if not all(
                    float(pair["b_rms_separation_m"]) >= 0.006
                    for pair in pairs
                ):
                    continue
                distinct_count += 1
                if any(abs(value) > 1.0 for value in standardized_values):
                    continue
                pairs.sort(
                    key=lambda pair: (
                        int(pair["lower_seed"]),
                        int(pair["higher_seed"]),
                    )
                )
                seed_key = tuple(
                    (
                        int(pair["lower_seed"]),
                        int(pair["higher_seed"]),
                    )
                    for pair in pairs
                )
                objective = (
                    max(abs(value) for value in standardized_values),
                    sum(value * value for value in standardized_values),
                    -float(np.mean(separations)),
                    seed_key,
                )
                candidates.append((objective, pairs))
    if distinct_count == 0:
        pairing = {
            **base_pairing,
            "status": NO_DISTINCT_CURVATURE_QUARTET,
            "pairs": [],
        }
        return {
            "simulation_ranking": simulation,
            "hardware_ranking": hardware,
            "pairing": pairing,
        }
    if not candidates:
        pairing = {
            **base_pairing,
            "status": NO_MARGIN_MATCHED_QUARTET,
            "pairs": [],
        }
        return {
            "simulation_ranking": simulation,
            "hardware_ranking": hardware,
            "pairing": pairing,
        }

    objective, selected_pairs = min(candidates, key=lambda item: item[0])
    pairing = {
        **base_pairing,
        "status": "SELECTED",
        "selection_objective": list(objective),
        "pairs": selected_pairs,
    }
    return {
        "simulation_ranking": simulation,
        "hardware_ranking": hardware,
        "pairing": pairing,
    }


def build_recorder_command(
    row: Mapping[str, object],
    *,
    leaf: str | Path,
    python_executable: str | Path,
    recorder_script: str | Path,
    fixed_reset_envelope: str | Path,
    code_revision: str,
    asset_revision: str,
) -> list[str]:
    """Build the exact one-policy Task 1 CPU invocation."""

    _lower_hex(code_revision, 40, label="source revision")
    _lower_hex(asset_revision, 40, label="asset revision")
    if str(row["task"]) != expected_task(
        str(row["campaign"]), str(row["arm"])
    ):
        raise ValueError("cannot build recorder command for task drift")
    cap_csv = ",".join(str(float(value)) for value in _verified_caps())
    return [
        str(python_executable),
        str(recorder_script),
        "--ckpt",
        str(row["checkpoint_file"]),
        "--out",
        str(leaf),
        "--j-limit",
        cap_csv,
        "--task",
        str(row["task"]),
        "--num-envs",
        "1",
        "--env-idx",
        "0",
        "--device",
        "cpu",
        "--nsteps",
        str(REQUESTED_CONTROLS),
        "--play",
        "--fixed-reset-envelope",
        str(fixed_reset_envelope),
        "--campaign",
        str(row["campaign"]),
        "--arm",
        str(row["arm"]),
        "--training-seed",
        str(int(row["training_seed"])),
        "--checkpoint-sha256",
        str(row["checkpoint_sha256"]),
        "--code-revision",
        code_revision,
        "--asset-revision",
        asset_revision,
    ]


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> list[str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        output.write_text("", encoding="utf-8")
        return []
    if fieldnames is None:
        identity = [
            "campaign",
            "arm",
            "training_seed",
            "task",
            "checkpoint_sha256",
        ]
        fields = identity + sorted(
            set().union(*(row.keys() for row in rows)) - set(identity)
        )
    else:
        fields = list(fieldnames)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field)) for field in fields}
            )
    temporary.replace(output)
    return fields


class BatchFailure(RuntimeError):
    """The exact-56 status table contains one or more failed rows."""


def run_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    output_root: str | Path,
    python_executable: str | Path,
    recorder_script: str | Path,
    fixed_reset_envelope: str | Path,
    code_revision: str,
    asset_revision: str,
    runner: Callable[..., Any] = subprocess.run,
    progress: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Run or resume all rows, retaining every identity in the status CSV."""

    validate_inventory(rows)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, object]] = [
        {
            "campaign": str(row["campaign"]),
            "arm": str(row["arm"]),
            "training_seed": int(row["training_seed"]),
            "task": str(row["task"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "status": "pending",
            "error": "",
            "elapsed_s": "",
        }
        for row in rows
    ]
    status_path = root / "run_status.csv"
    _write_csv(status_path, status_rows)
    started = time.monotonic()

    for index, (row, status) in enumerate(
        zip(rows, status_rows, strict=True), start=1
    ):
        row_started = time.monotonic()
        leaf = (
            root
            / str(row["campaign"])
            / str(row["arm"])
            / str(int(row["training_seed"]))
        )
        resume_error = ""
        if (leaf / "trace.npz").is_file() and (
            leaf / "metadata.json"
        ).is_file():
            try:
                validate_resume_leaf(
                    leaf,
                    row,
                    fixed_reset_envelope=fixed_reset_envelope,
                    code_revision=code_revision,
                    asset_revision=asset_revision,
                )
                status["status"] = "reused"
            except (OSError, ValueError, KeyError, TypeError) as error:
                resume_error = str(error)

        if status["status"] != "reused":
            command = build_recorder_command(
                row,
                leaf=leaf,
                python_executable=python_executable,
                recorder_script=recorder_script,
                fixed_reset_envelope=fixed_reset_envelope,
                code_revision=code_revision,
                asset_revision=asset_revision,
            )
            try:
                completed = runner(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            except OSError as error:
                completed = subprocess.CompletedProcess(
                    command, -1, stdout=str(error)
                )
            if int(completed.returncode) != 0:
                status["status"] = "failed"
                output = str(getattr(completed, "stdout", "") or "").strip()
                detail = output[-2000:] if output else "no child output"
                status["error"] = (
                    f"recorder exit {completed.returncode}: {detail}"
                )
                if resume_error:
                    status["error"] = (
                        f"resume rejected ({resume_error}); "
                        f"{status['error']}"
                    )
            else:
                try:
                    validate_resume_leaf(
                        leaf,
                        row,
                        fixed_reset_envelope=fixed_reset_envelope,
                        code_revision=code_revision,
                        asset_revision=asset_revision,
                    )
                    status["status"] = "completed"
                    status["error"] = (
                        "" if not resume_error else f"replaced: {resume_error}"
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    status["status"] = "failed"
                    status["error"] = (
                        "recorder returned zero but leaf validation failed: "
                        f"{error}"
                    )

        status["elapsed_s"] = round(time.monotonic() - row_started, 6)
        _write_csv(status_path, status_rows)
        elapsed = time.monotonic() - started
        eta = (elapsed / index) * (len(rows) - index)
        progress(
            f"[{index}/{len(rows)}] {row['campaign']}/{row['arm']}/"
            f"seed{row['training_seed']} {status['status']} "
            f"elapsed={elapsed:.1f}s ETA={eta:.1f}s"
        )

    failures = [row for row in status_rows if row["status"] == "failed"]
    if failures:
        raise BatchFailure(
            f"{len(failures)} of {len(status_rows)} companion rows failed; "
            f"see {status_path}"
        )
    return status_rows


def analyze_completed_leaves(
    rows: Sequence[Mapping[str, object]],
    *,
    output_root: str | Path,
    fixed_reset_envelope: str | Path,
    code_revision: str,
    asset_revision: str,
) -> tuple[list[dict[str, object]], list[dict[str, np.ndarray]]]:
    """Validate all 56 leaves before deriving any selection output."""

    validate_inventory(rows)
    root = Path(output_root)
    status_path = root / "run_status.csv"
    try:
        with status_path.open(encoding="utf-8", newline="") as handle:
            status_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read all-56 status table: {status_path}") from error

    def identity(row: Mapping[str, object]) -> tuple[str, str, int]:
        return (
            str(row["campaign"]),
            str(row["arm"]),
            int(row["training_seed"]),
        )

    status_by_identity = {identity(row): row for row in status_rows}
    expected_identities = {identity(row) for row in rows}
    if (
        len(status_rows) != 56
        or set(status_by_identity) != expected_identities
    ):
        raise ValueError("run_status.csv does not preserve the exact 56 identities")

    summaries: list[dict[str, object]] = []
    traces: list[dict[str, np.ndarray]] = []
    analysis_failures = 0
    for row in rows:
        leaf = (
            root
            / str(row["campaign"])
            / str(row["arm"])
            / str(int(row["training_seed"]))
        )
        try:
            validate_resume_leaf(
                leaf,
                row,
                fixed_reset_envelope=fixed_reset_envelope,
                code_revision=code_revision,
                asset_revision=asset_revision,
            )
            payload = load_trace_payload(leaf)
            metrics = derive_episode_metrics(payload)
        except (OSError, ValueError, KeyError, TypeError) as error:
            analysis_failures += 1
            status = status_by_identity[identity(row)]
            status["status"] = "analysis_failed"
            status["error"] = f"scientific analysis failed: {error}"
            continue
        eligible, failures = hardware_eligibility(metrics)
        summary = {
            "campaign": str(row["campaign"]),
            "arm": str(row["arm"]),
            "training_seed": int(row["training_seed"]),
            "task": str(row["task"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            **metrics,
            "hardware_eligible": eligible,
            "hardware_ineligibility_reasons": ";".join(failures),
        }
        summaries.append(summary)
        traces.append(payload)
    if analysis_failures:
        _write_csv(status_path, status_rows)
        raise BatchFailure(
            f"{analysis_failures} of 56 companion rows failed scientific "
            f"analysis; see {status_path}"
        )
    return summaries, traces


def _reference_waypoints(trace: Mapping[str, np.ndarray]) -> np.ndarray | None:
    if "strike_ref_error_control_m" not in trace:
        return None
    errors = np.asarray(trace["strike_ref_error_control_m"], dtype=float)
    positions = np.asarray(trace["head_position_m"], dtype=float)
    starts = np.arange(errors.shape[0]) * CONTROL_DECIMATION
    if errors.ndim != 2 or errors.shape[1] != 3 or np.any(starts >= len(positions)):
        return None
    return positions[starts] + errors


def _write_diagnostic_grids(
    rows: Sequence[Mapping[str, object]],
    traces: Sequence[Mapping[str, np.ndarray]],
    root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    arms = [
        (campaign, arm)
        for campaign, campaign_arms in EXPECTED.items()
        for arm in campaign_arms
    ]
    lookup = {
        (
            str(row["campaign"]),
            str(row["arm"]),
            int(row["training_seed"]),
        ): trace
        for row, trace in zip(rows, traces, strict=True)
    }
    caps = _verified_caps()
    specifications = (
        ("xz", "trajectories_xz_grid.png", "z (m)"),
        ("xy", "trajectories_xy_grid.png", "y (m)"),
        ("qvel", "qvel_grid.png", "|qvel| max (rad/s)"),
        ("lambda", "lambda_cap_grid.png", "max Lambda/cap"),
    )
    for mode, filename, ylabel in specifications:
        figure, axes = plt.subplots(
            len(arms),
            8,
            figsize=(16, 13),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for row_index, (campaign, arm) in enumerate(arms):
            seeds = list(EXPECTED[campaign][arm])
            for column, seed in enumerate(seeds):
                axis = axes[row_index, column]
                trace = lookup[(campaign, arm, seed)]
                if mode in {"xz", "xy"}:
                    ordinate = 2 if mode == "xz" else 1
                    positions = np.asarray(
                        trace["head_position_m"], dtype=float
                    )
                    axis.plot(
                        positions[:, 0],
                        positions[:, ordinate],
                        color="C0",
                        linewidth=0.8,
                    )
                    reference = _reference_waypoints(trace)
                    if reference is not None:
                        axis.plot(
                            reference[:, 0],
                            reference[:, ordinate],
                            color="black",
                            linestyle="--",
                            linewidth=0.5,
                        )
                    contact = np.asarray(trace["contact"], dtype=bool)
                    if np.any(contact):
                        axis.scatter(
                            positions[contact, 0],
                            positions[contact, ordinate],
                            color="C3",
                            s=2,
                        )
                elif mode == "qvel":
                    pre = np.asarray(
                        trace["joint_velocity_rad_s"], dtype=float
                    )
                    post = np.asarray(
                        trace["joint_velocity_post_integration_rad_s"],
                        dtype=float,
                    )
                    values = np.max(
                        np.abs(np.concatenate((pre, post[-1:]), axis=0)),
                        axis=1,
                    )
                    axis.plot(
                        np.arange(len(values)) * PHYSICS_DT_S,
                        values,
                        linewidth=0.7,
                    )
                    axis.axhline(
                        HARDWARE_QVEL_LIMIT_RAD_S,
                        color="C3",
                        linestyle="--",
                        linewidth=0.5,
                    )
                else:
                    reads = np.asarray(
                        trace["lambda_windowed_constraint_read_n_m_s"],
                        dtype=float,
                    )
                    values = np.max(reads / caps, axis=1)
                    axis.plot(
                        np.arange(len(values)) * PHYSICS_DT_S,
                        values,
                        linewidth=0.7,
                    )
                    axis.axhline(
                        1.0,
                        color="C3",
                        linestyle="--",
                        linewidth=0.5,
                    )
                axis.set_title(f"{arm} s{seed}", fontsize=6)
                axis.tick_params(labelsize=5)
                axis.grid(alpha=0.15)
        figure.supxlabel("x (m)" if mode in {"xz", "xy"} else "time (s)")
        figure.supylabel(ylabel)
        if mode in {"xz", "xy"}:
            figure.suptitle(
                "Fixed-reset trajectories; black dashed reference is observation-only",
                fontsize=10,
            )
        figure.tight_layout()
        figure.savefig(root / filename, dpi=150)
        plt.close(figure)


def write_artifact_inventory(root: str | Path) -> Path:
    """Hash every published artifact except the inventory itself."""

    output = Path(root)
    inventory = output / ARTIFACT_INVENTORY_NAME
    files = [
        path
        for path in output.rglob("*")
        if path.is_file()
        and path != inventory
        and not any(part.startswith(".") for part in path.relative_to(output).parts)
    ]
    lines = [
        f"{_sha256(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(files)
    ]
    inventory.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inventory


def write_analysis_outputs(
    summaries: Sequence[Mapping[str, object]],
    traces: Sequence[Mapping[str, np.ndarray]],
    *,
    rows: Sequence[Mapping[str, object]],
    population_covariates: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, object]:
    """Write the required CSV/JSON/PNG outputs after the all-56 gate."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "summary.csv", summaries)
    ranked = rank_and_pair_fq(summaries, population_covariates)
    ranking_fields = _write_csv(
        root / "fq_simulation_ranking.csv",
        ranked["simulation_ranking"],
    )
    _write_csv(
        root / "fq_hardware_ranking.csv",
        ranked["hardware_ranking"],
        fieldnames=ranking_fields,
    )
    (root / "fq_pairing.json").write_text(
        json.dumps(
            ranked["pairing"],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_diagnostic_grids(rows, traces, root)
    write_artifact_inventory(root)
    return ranked


def _load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot resolve source revision: {completed.stderr}")
    return _lower_hex(
        completed.stdout.strip(), 40, label="source revision"
    )


def resolve_code_revision(supplied: str | None) -> str:
    """Use only the current repository HEAD as rollout source identity."""

    current = _git_revision()
    if supplied is not None and supplied != current:
        raise ValueError(
            "--code-revision must equal the current source revision "
            f"{current}, got {supplied}"
        )
    return current


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact-56 fixed-reset 500 Hz companion"
    )
    parser.add_argument("--inventory", default=str(INVENTORY_PATH))
    parser.add_argument(
        "--inventory-sidecar",
        default=str(INVENTORY_PATH) + ".sha256",
    )
    parser.add_argument(
        "--fq4-root", default=str(DEFAULT_CHECKPOINT_ROOTS["fq4x8"])
    )
    parser.add_argument(
        "--fq3-root", default=str(DEFAULT_CHECKPOINT_ROOTS["fq3x8"])
    )
    parser.add_argument("--fixed-reset-envelope", default=str(FIXED_RESET_PATH))
    parser.add_argument(
        "--population-covariates",
        default=str(POPULATION_COVARIATES_PATH),
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--code-revision", default=None)
    parser.add_argument("--asset-revision", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    reset = load_fixed_reset(args.fixed_reset_envelope)
    if reset.get("reset_state_digest") != APPROVED_FIXED_RESET_DIGEST:
        raise SystemExit("approved fixed reset digest mismatch")
    rows = load_checkpoint_inventory(
        args.inventory,
        checkpoint_roots={
            "fq4x8": args.fq4_root,
            "fq3x8": args.fq3_root,
        },
        expected_sha256=INVENTORY_SHA256,
        sidecar_path=args.inventory_sidecar,
    )
    inventory_assets = {
        str(row["training_asset_revision"]) for row in rows
    }
    if len(inventory_assets) != 1:
        raise SystemExit("inventory has multiple frozen asset revisions")
    frozen_asset_revision = next(iter(inventory_assets))
    asset_revision = args.asset_revision or frozen_asset_revision
    if asset_revision != frozen_asset_revision:
        raise SystemExit("requested asset revision differs from frozen inventory")
    try:
        code_revision = resolve_code_revision(args.code_revision)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    _lower_hex(asset_revision, 40, label="asset revision")

    run_batch(
        rows,
        output_root=args.output_root,
        python_executable=args.python,
        recorder_script=SOURCE_ROOT / "scripts/diag_impulse_trace.py",
        fixed_reset_envelope=args.fixed_reset_envelope,
        code_revision=code_revision,
        asset_revision=asset_revision,
    )
    summaries, traces = analyze_completed_leaves(
        rows,
        output_root=args.output_root,
        fixed_reset_envelope=args.fixed_reset_envelope,
        code_revision=code_revision,
        asset_revision=asset_revision,
    )
    write_analysis_outputs(
        summaries,
        traces,
        rows=rows,
        population_covariates=load_population_covariates(
            args.population_covariates
        ),
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
