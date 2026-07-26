"""Seed-aware analysis for the fixed-impedance first-strike campaign.

The live evaluator stores one physical 500 Hz trace per episode.  Contact,
tail, first-event, hardware-speed, and geometry diagnostics are all derived
from that shared trace; reward streams are labelled actual treatment payouts
and are never used to duplicate the physical trajectory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from evaluation.analysis.terminal_funnel import (
    WINDOW_SAMPLES_BEFORE,
    compute_episode_metrics,
)


ARM_TASKS = {
    "C": "Unitree-Z1-Hammer-CaT-Impulse",
    "D-prime": "Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy",
    "F": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    "E": "Unitree-Z1-Hammer-CaT-Impulse-Event",
}
ARM_ORDER = tuple(ARM_TASKS)
PRIMARY_METRIC = "first_strike_useful_speed_mean_sampled"
HARDWARE_QVEL_LIMIT_RAD_S = 3.1415
ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
EXPECTED_SEEDS = tuple(range(8))
EXPECTED_EPISODES_PER_SEED = 512
EXPECTED_NUM_ENVS = 256
EXPECTED_EPISODES_PER_ENV = 2
EXPECTED_EVALUATION_EPISODE_LEN_S = 4.0
EXPECTED_EVALUATION_MEAN_NSTEPS = 400
EXPECTED_PHYSICS_DT_S = 0.002
EXPECTED_CONTROL_DECIMATION = 10
CAMPAIGN_EVALUATOR_SEED = 2_026_072_900
BOOTSTRAP_SEED = 2026072504
EXPECTED_IMPULSE_LIMITS_N_M_S = (1.640, 3.280, 1.640, 1.640, 1.640, 1.640)
EXPECTED_FIXED_ACTUATOR_SIGNATURE = (
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
EXPECTED_FIXED_ACTION_SIGNATURE = (
    (
        "ik_hammer_head",
        "DifferentialIKActionCfg",
        (
            ("entity_name", "robot"),
            ("clip", None),
            (
                "actuator_names",
                ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
            ),
            ("frame_type", "site"),
            ("frame_name", "hammer_head_site"),
            ("use_relative_mode", True),
            ("delta_pos_scale", 0.15),
            ("delta_ori_scale", 1.0),
            ("damping", 0.05),
            ("max_dq", 0.5),
            ("position_weight", 1.0),
            ("orientation_weight", 0.0),
            ("joint_limit_weight", 0.0),
            ("posture_weight", 0.0),
            ("posture_target", None),
        ),
    ),
)
EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256 = hashlib.sha256(
    json.dumps(
        EXPECTED_FIXED_ACTUATOR_SIGNATURE,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
).hexdigest()
EXPECTED_FIXED_ACTION_SIGNATURE_SHA256 = hashlib.sha256(
    json.dumps(
        EXPECTED_FIXED_ACTION_SIGNATURE,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
).hexdigest()
CAMPAIGN_ARM_SEMANTICS = {
    "C": ("legacy_50hz_repeated_credit", "c"),
    "D-prime": (
        "legacy_50hz_first_event_censored_one_shot",
        "dprime",
    ),
    "F": ("event_500hz_first_event_linear_delivered", "f"),
    "E": ("event_500hz_first_event_saturated_delivered", "e"),
}
CAMPAIGN_EVALUATOR_RNG = {
    "reset": 2_036_072_919,
    "observation": 2_046_072_933,
    "action": 2_056_072_941,
}
ARM_EVENT_I_REF_N_S = {
    "C": 0.6094,
    "D-prime": 0.6094,
    "F": 0.3088,
    "E": 0.3088,
}
PAYOUT_SEMANTICS = {
    "C": "actual_legacy_repeated",
    "D-prime": "actual_legacy_first_event",
    "F": "actual_event_linear",
    "E": "actual_event_saturated",
}
LEGACY_SAMPLED_COLUMNS = frozenset(
    {
        "success_rate_sampled",
        "worst_ratio_max_sampled",
        "delivered_mean_sampled",
    }
)
COMPARISON_IDENTITY_CONTRACT = {
    "identical_across_rows": [
        "git_revision",
        "asset_git_revision",
        "nail_asset_sha256",
        "campaign_config_sha256",
        "accepted_manifest_sha256",
        "training_code_revision",
        "training_asset_revision",
        "action_rng_seed",
        "reset_rng_seed",
        "observation_rng_seed",
    ],
    "bound_per_row_but_expected_to_vary": [
        "checkpoint_sha256",
        "accepted_checkpoint_sha256",
        "sampled_trace_digest",
        "sampled_trace_artifact_sha256",
    ],
    "bound_per_arm": ["treatment_config_sha256", "event_i_ref_n_s"],
    "treatment_identity": "exact ARM_TASKS task id plus configured 8/2 weights",
}


def _frozen_campaign_row(arm: str, seed: int) -> dict:
    semantics, slug = CAMPAIGN_ARM_SEMANTICS[arm]
    run_name = f"fsr4x8_{slug}_seed{seed}"
    return {
        "arm": arm,
        "task": ARM_TASKS[arm],
        "arm_semantics": semantics,
        "training_seed": seed,
        "impact_weight": 8.0,
        "delivered_weight": 2.0,
        "imp_max_p": 0.0,
        "impedance_mode": "fixed",
        "fixed_action_signature": EXPECTED_FIXED_ACTION_SIGNATURE,
        "impulse_limits_n_m_s": EXPECTED_IMPULSE_LIMITS_N_M_S,
        "training_iterations": 500,
        "training_num_envs": 4096,
        "checkpoint_filename": "model_499.pt",
        "run_name": run_name,
        "expected_checkpoint_glob": (
            f"logs/rsl_rl/z1_hammer/*_{run_name}/model_499.pt"
        ),
        "eval_num_envs": EXPECTED_NUM_ENVS,
        "eval_episodes_per_env": EXPECTED_EPISODES_PER_ENV,
        "eval_episode_count": EXPECTED_EPISODES_PER_SEED,
        "eval_episode_len_s": EXPECTED_EVALUATION_EPISODE_LEN_S,
        "eval_mean_nsteps": EXPECTED_EVALUATION_MEAN_NSTEPS,
        "eval_completion_rule": "first_two_completions_per_environment",
        "eval_stochastic_actions": True,
        "eval_reset_noise_rad": (-0.05, 0.05),
        "eval_base_rng_seed": CAMPAIGN_EVALUATOR_SEED,
        "eval_reset_rng_seed": CAMPAIGN_EVALUATOR_RNG["reset"],
        "eval_observation_rng_seed": CAMPAIGN_EVALUATOR_RNG["observation"],
        "eval_action_rng_seed": CAMPAIGN_EVALUATOR_RNG["action"],
        "eval_actor_observation_corruption": True,
        "eval_critic_observation_corruption": False,
    }


FROZEN_CAMPAIGN_MATRIX = tuple(
    _frozen_campaign_row(arm, seed)
    for arm in ARM_ORDER
    for seed in EXPECTED_SEEDS
)

_CAMPAIGN_FIELD_LABELS = {
    "task": "task",
    "arm_semantics": "arm semantics",
    "impact_weight": "impact weight",
    "delivered_weight": "delivered weight",
    "imp_max_p": "imp_max_p",
    "impedance_mode": "fixed impedance",
    "fixed_action_signature": "fixed action",
    "impulse_limits_n_m_s": "impulse limits",
    "training_iterations": "500 iterations",
    "training_num_envs": "4096 environments",
    "checkpoint_filename": "model_499.pt",
    "run_name": "run name",
    "expected_checkpoint_glob": "checkpoint location",
    "eval_num_envs": "256 evaluation environments",
    "eval_episodes_per_env": "two completions",
    "eval_episode_count": "512 sampled episodes",
    "eval_episode_len_s": "4.0 second evaluation horizon",
    "eval_mean_nsteps": "400 mean-action steps",
    "eval_completion_rule": "completion rule",
    "eval_stochastic_actions": "stochastic actions",
    "eval_reset_noise_rad": "reset noise",
    "eval_base_rng_seed": "base evaluator RNG",
    "eval_reset_rng_seed": "reset RNG",
    "eval_observation_rng_seed": "observation RNG",
    "eval_action_rng_seed": "action RNG",
    "eval_actor_observation_corruption": "actor observation corruption",
    "eval_critic_observation_corruption": "critic observation corruption",
}


def validate_frozen_campaign_matrix(rows: Sequence[Mapping]) -> None:
    """Reject any drift from the preregistered local 4x8 job matrix."""

    rows = [dict(row) for row in rows]
    if len(rows) != 32:
        raise ValueError(f"campaign must contain exactly 32 jobs, found {len(rows)}")

    pairs = [(row.get("arm"), row.get("training_seed")) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("campaign arm/seed identities must be unique")
    expected_pairs = {
        (arm, seed) for arm in ARM_ORDER for seed in EXPECTED_SEEDS
    }
    if set(pairs) != expected_pairs:
        raise ValueError(
            "campaign arm/seed identities must be exactly C/D-prime/F/E x 0..7"
        )

    if len({row.get("run_name") for row in rows}) != len(rows):
        raise ValueError("campaign run names must be unique")

    for row in rows:
        arm = str(row["arm"])
        seed = int(row["training_seed"])
        expected = _frozen_campaign_row(arm, seed)
        if set(row) != set(expected):
            raise ValueError(
                f"{arm}/seed{seed}: campaign fields differ from the frozen contract"
            )
        for field, expected_value in expected.items():
            if row[field] != expected_value:
                label = _CAMPAIGN_FIELD_LABELS.get(field, field)
                raise ValueError(f"{arm}/seed{seed}: {label} drift")


ACCEPTED_ATTEMPT_MANIFEST_FIELDS = (
    "arm",
    "training_seed",
    "task",
    "run_name",
    "attempt_path",
    "checkpoint_path",
    "checkpoint_sha256",
    "training_code_revision",
    "training_asset_revision",
    "disposition",
    "reason",
)


def load_accepted_attempt_manifest(
    path: str | Path,
    *,
    expected_code_revision: str,
    expected_asset_revision: str,
    require_files: bool = True,
) -> list[dict]:
    """Validate the explicit 32-checkpoint selection ledger.

    The manifest is a tab-separated file so it is both human-auditable and
    consumable by the Slurm launcher without a second checkpoint-selection
    implementation.  It contains accepted attempts only; failed attempts and
    their directories remain retained, but are never selected implicitly.
    """

    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ACCEPTED_ATTEMPT_MANIFEST_FIELDS:
            raise ValueError("accepted-attempt manifest header does not match contract")
        raw_rows = list(reader)
    if len(raw_rows) != 32:
        raise ValueError(
            f"accepted-attempt manifest must contain exactly 32 rows, found "
            f"{len(raw_rows)}"
        )
    if not _is_hex_digest(expected_code_revision, 40):
        raise ValueError("expected code revision must be a full 40-hex revision")
    if not _is_hex_digest(expected_asset_revision, 40):
        raise ValueError("expected asset revision must be a full 40-hex revision")

    validated: list[dict] = []
    identities: set[tuple[str, int]] = set()
    checkpoint_paths: set[Path] = set()
    checkpoint_hashes: set[str] = set()
    for raw in raw_rows:
        arm = str(raw["arm"])
        try:
            seed = int(raw["training_seed"])
        except (TypeError, ValueError) as error:
            raise ValueError("manifest training_seed must be an integer") from error
        identity = (arm, seed)
        if identity in identities:
            raise ValueError(f"duplicate accepted arm/seed row: {arm}/seed{seed}")
        identities.add(identity)
        if arm not in ARM_TASKS or seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected accepted arm/seed row: {arm}/seed{seed}")

        expected = _frozen_campaign_row(arm, seed)
        if raw["task"] != expected["task"]:
            raise ValueError(f"{arm}/seed{seed}: exact task mismatch")
        if raw["run_name"] != expected["run_name"]:
            raise ValueError(f"{arm}/seed{seed}: deterministic run name mismatch")
        if raw["disposition"] != "accepted":
            raise ValueError(f"{arm}/seed{seed}: disposition must be accepted")
        reason = str(raw["reason"]).strip()
        if not reason or reason.upper().startswith("PENDING"):
            raise ValueError(f"{arm}/seed{seed}: accepted disposition needs a reason")
        if raw["training_code_revision"] != expected_code_revision:
            raise ValueError(f"{arm}/seed{seed}: training code revision mismatch")
        if raw["training_asset_revision"] != expected_asset_revision:
            raise ValueError(f"{arm}/seed{seed}: training asset revision mismatch")
        if not _is_hex_digest(raw["checkpoint_sha256"], 64):
            raise ValueError(f"{arm}/seed{seed}: invalid checkpoint SHA-256")
        checkpoint_hash = str(raw["checkpoint_sha256"]).lower()
        if checkpoint_hash in checkpoint_hashes:
            raise ValueError(
                "accepted checkpoint SHA-256 values must be unique"
            )
        checkpoint_hashes.add(checkpoint_hash)

        attempt_path = Path(raw["attempt_path"])
        checkpoint_path = Path(raw["checkpoint_path"])
        if not attempt_path.is_absolute() or not checkpoint_path.is_absolute():
            raise ValueError(f"{arm}/seed{seed}: attempt/checkpoint paths must be absolute")
        if checkpoint_path.name != "model_499.pt":
            raise ValueError(f"{arm}/seed{seed}: checkpoint must be model_499.pt")
        if not (
            attempt_path.name == expected["run_name"]
            or attempt_path.name.endswith(f"_{expected['run_name']}")
        ):
            raise ValueError(f"{arm}/seed{seed}: attempt path/run name mismatch")
        if checkpoint_path.parent != attempt_path:
            raise ValueError(
                f"{arm}/seed{seed}: checkpoint must be directly under attempt path"
            )
        if checkpoint_path in checkpoint_paths:
            raise ValueError("accepted checkpoint paths must be unique")
        checkpoint_paths.add(checkpoint_path)
        if require_files:
            if not attempt_path.is_dir() or not checkpoint_path.is_file():
                raise ValueError(f"{arm}/seed{seed}: accepted checkpoint is missing")
            if _sha256(checkpoint_path) != raw["checkpoint_sha256"]:
                raise ValueError(f"{arm}/seed{seed}: checkpoint SHA-256 mismatch")

        validated.append(
            {
                **raw,
                "training_seed": seed,
                "attempt_path": str(attempt_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "reason": reason,
            }
        )

    expected_identities = {
        (arm, seed) for arm in ARM_ORDER for seed in EXPECTED_SEEDS
    }
    if identities != expected_identities:
        raise ValueError(
            "accepted-attempt manifest identities must be exactly "
            "C/D-prime/F/E x seeds 0..7"
        )
    order = {arm: index for index, arm in enumerate(ARM_ORDER)}
    return sorted(
        validated,
        key=lambda row: (order[str(row["arm"])], int(row["training_seed"])),
    )


REQUIRED_SAMPLED_COLUMNS = frozenset(
    {
        "first_strike_useful_speed_mean_sampled",
        "successful_first_strike_v_precontact_mean_sampled",
        "first_strike_success_rate_sampled",
        "recontact_rate_sampled",
        "first_strike_v_precontact_mean_sampled",
        "first_strike_productive_rate_sampled",
        "first_strike_delivered_success_mean_sampled",
        "first_strike_delivered_window_mean_sampled",
        "first_strike_success_n_sampled",
        "first_strike_window_n_sampled",
        "first_strike_no_contact_n_sampled",
        "first_strike_unfinished_event_n_sampled",
        "tail_fraction_mean_sampled",
        "maximize_return_discounted_mean_sampled",
        "impact_return_discounted_mean_sampled",
        "delivered_return_discounted_mean_sampled",
        "first_strike_saturation_rate_sampled",
        "overall_success_rate_sampled",
        "impossible_success_n",
        "lambda_dead_n",
        "qvel_violation_rate_sampled",
        "qvel_finite_exceedance_rate_sampled",
        "qvel_nonfinite_rate_sampled",
        "qvel_max_abs_rad_s_sampled",
        "qvel_max_excess_rad_s_sampled",
        "qvel_samples_over_rail_sampled",
        "qvel_seconds_over_rail_sampled",
        "qvel_peak_joint_index_sampled",
        "qvel_precontact_exceedance_rate_sampled",
        "qvel_precontact_samples_over_rail_sampled",
        "qvel_precontact_seconds_over_rail_sampled",
        "qvel_first_event_exceedance_rate_sampled",
        "qvel_first_event_samples_over_rail_sampled",
        "qvel_first_event_seconds_over_rail_sampled",
        "qvel_post_event_exceedance_rate_sampled",
        "qvel_post_event_samples_over_rail_sampled",
        "qvel_post_event_seconds_over_rail_sampled",
        "precontact_path_length_ratio_mean_sampled",
        "precontact_lateral_excursion_mean_sampled",
        "contact_approach_angle_mean_sampled",
        "first_contact_transverse_error_mean_sampled",
        "first_contact_within_nail_radius_rate_sampled",
        "first_contact_geometry_n_sampled",
        "terminal_contraction_ratio_mean_sampled",
        "late_lateral_reexpansion_rate_sampled",
        "terminal_funnel_eligible_n_sampled",
    }
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(node: ET.Element, attribute: str) -> np.ndarray:
    values = np.fromstring(node.attrib.get(attribute, ""), sep=" ")
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{node.tag} {attribute!r} must contain three finite values")
    return values


def load_frozen_nail_geometry(
    asset_path: str | Path, *, expected_sha256: str | None = None
) -> dict:
    """Read the nail axis and head radius from the provenance-bound scene XML."""

    path = Path(asset_path)
    actual_sha256 = _sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"nail asset hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    root = ET.parse(path).getroot()
    body = root.find(".//body[@name='nail']")
    joint = root.find(".//joint[@name='nail_slide']")
    head = root.find(".//geom[@name='nail_head']")
    if body is None or joint is None or head is None:
        raise ValueError("scene XML must define nail body, nail_slide, and nail_head")
    body_pos = _vector(body, "pos")
    axis = _vector(joint, "axis")
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("nail_slide axis must be non-zero")
    size = np.fromstring(head.attrib.get("size", ""), sep=" ")
    if size.size < 1 or not np.isfinite(size).all() or size[0] <= 0:
        raise ValueError("nail_head size must contain a positive radius")
    return {
        "nail_axis": (axis / norm).astype(float).tolist(),
        "nail_xy_m": body_pos[:2].astype(float).tolist(),
        "nail_radius_m": float(size[0]),
        "source_path": str(path),
        "source_sha256": actual_sha256,
        "basis": "nail body position, nail_slide axis, and nail_head radius",
    }


def _physical_arrays(trace: Mapping) -> dict[str, np.ndarray]:
    try:
        physical = trace["physical"]
        contact = np.asarray(physical["contact"], dtype=bool)
        position = np.asarray(physical["head_position_m"], dtype=float)
        depth = np.asarray(physical["clamped_depth_m"], dtype=float)
        force = np.asarray(physical["net_axial_force_n"], dtype=float)
        qvel = np.asarray(physical["joint_speed_rad_s"], dtype=float)
        qvel_post = np.asarray(
            physical["post_step_joint_speed_rad_s"], dtype=float
        )
        tracker_depth_post = np.asarray(
            physical["tracker_depth_post_integration_m"], dtype=float
        )
    except KeyError as error:
        raise ValueError(f"missing physical trace field: {error.args[0]}") from error
    n = contact.size
    if contact.ndim != 1 or n == 0:
        raise ValueError("contact must be a non-empty one-dimensional vector")
    if position.shape != (n, 3):
        raise ValueError("head_position_m must have shape [len(contact), 3]")
    if depth.shape != (n,) or force.shape != (n,):
        raise ValueError("depth and force must match contact length")
    if qvel.ndim != 2 or qvel.shape[0] != n or qvel.shape[1] < 1:
        raise ValueError("joint_speed_rad_s must have shape [len(contact), joints]")
    if qvel_post.shape != qvel.shape:
        raise ValueError(
            "post_step_joint_speed_rad_s must match joint_speed_rad_s"
        )
    if tracker_depth_post.shape != depth.shape:
        raise ValueError(
            "tracker_depth_post_integration_m must match clamped_depth_m"
        )
    for name, values in {
        "head_position_m": position,
        "clamped_depth_m": depth,
        "net_axial_force_n": force,
        "tracker_depth_post_integration_m": tracker_depth_post,
    }.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
    if np.any(force < 0):
        raise ValueError("net_axial_force_n must be non-negative")
    return {
        "contact": contact,
        "position": position,
        "depth": depth,
        "force": force,
        "qvel": qvel,
        "qvel_post": qvel_post,
    }


def _contact_structure(
    contact: np.ndarray,
    force: np.ndarray,
    dt_s: float,
    *,
    accepted_onset: int | None,
    finalization_index: int | None,
) -> tuple[int | None, bool, float]:
    if accepted_onset is None:
        if np.any(contact):
            raise ValueError(
                "raw contact before tracker acceptance invalidates the episode"
            )
        return None, False, 0.0
    onset = int(accepted_onset)
    if onset < 0 or onset >= contact.size or not bool(contact[onset]):
        raise ValueError("tracker-accepted onset must index an in-contact substep")
    if np.any(contact[:onset]):
        raise ValueError("raw contact before tracker acceptance invalidates the episode")
    stop = contact.size - 1 if finalization_index is None else int(finalization_index)
    if stop < onset or stop >= contact.size:
        raise ValueError("tracker finalization must follow accepted onset")
    recontact = bool(np.any(contact[stop + 1 :]))
    total_impulse = float(np.sum(force[onset:][contact[onset:]]) * dt_s)
    first_impulse = float(np.sum(force[onset : stop + 1]) * dt_s)
    tail_fraction = (
        max(0.0, total_impulse - first_impulse) / total_impulse
        if total_impulse > 0
        else 0.0
    )
    return onset, recontact, tail_fraction


def _path_length_ratio(position_window: np.ndarray) -> float:
    increments = np.diff(position_window, axis=0)
    path_length = float(np.linalg.norm(increments, axis=1).sum())
    chord = position_window[-1] - position_window[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-12:
        path_ratio = 1.0 if path_length <= 1e-12 else float("inf")
    else:
        path_ratio = path_length / chord_length

    return path_ratio


def _event_arrays(trace: Mapping, sample_count: int) -> dict[str, np.ndarray]:
    try:
        event = trace["event_trace"]
        started = np.asarray(event["tracker_started"], dtype=bool)
        finalized = np.asarray(event["tracker_finalized"], dtype=bool)
        productive = np.asarray(event["tracker_productive"], dtype=bool)
        reason = np.asarray(event["tracker_reason"], dtype=int)
        impulse = np.asarray(
            event["event_cumulative_impulse_n_s"], dtype=float
        )
    except KeyError as error:
        raise ValueError(f"missing event trace field: {error.args[0]}") from error
    for name, values in {
        "tracker_started": started,
        "tracker_finalized": finalized,
        "tracker_productive": productive,
        "tracker_reason": reason,
        "event_cumulative_impulse_n_s": impulse,
    }.items():
        if values.shape != (sample_count,):
            raise ValueError(f"{name} must match physical trace length")
    if not np.isfinite(impulse).all() or np.any(impulse < 0):
        raise ValueError("event cumulative impulse must be finite/non-negative")
    if np.any(np.diff(impulse) < -1e-12):
        raise ValueError("event cumulative impulse must be nondecreasing")
    finalized_indices = np.flatnonzero(finalized)
    finalization_index = (
        int(finalized_indices[0]) if finalized_indices.size else None
    )
    if finalization_index is not None:
        if not np.all(finalized[finalization_index:]):
            raise ValueError("tracker finalized stream must remain true")
        if not np.allclose(
            impulse[finalization_index:],
            impulse[finalization_index],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "event cumulative impulse must freeze at tracker finalization"
            )
    return {
        "started": started,
        "finalized": finalized,
        "productive": productive,
        "reason": reason,
        "impulse": impulse,
        "finalization_index": finalization_index,
    }


def _discounted(values: Sequence[float], gamma: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("reward payout streams must be finite one-dimensional vectors")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("reward gamma must be finite and in [0, 1]")
    return float(np.sum(array * np.power(gamma, np.arange(array.size))))


def _qvel_diagnostics(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    dt_s: float,
    limit_rad_s: float,
    accepted_onset: int | None,
    finalization_index: int | None,
) -> dict:
    """Summarize one non-duplicated 500 Hz joint-velocity state sequence.

    Transition ``i`` stores its state immediately before integration in
    ``pre[i]`` and immediately after integration in ``post[i]``.  Interior
    states must therefore satisfy ``post[i] == pre[i + 1]``.  We count the
    canonical state sequence ``pre[:]`` plus only the terminal ``post[-1]``.

    If an event starts on transition ``onset`` and finalizes on transition
    ``stop``, state indices ``0..onset`` are pre-contact,
    ``onset+1..stop+1`` are first-event, and later states are post-event.
    An unfinished event owns every state after onset; a no-contact episode is
    entirely pre-contact.
    """

    if pre.shape != post.shape or pre.ndim != 2 or pre.shape[0] == 0:
        raise ValueError("pre/post qvel channels must be matching non-empty matrices")
    if pre.shape[0] > 1:
        prior_post = post[:-1]
        next_pre = pre[1:]
        comparable = np.isfinite(prior_post) & np.isfinite(next_pre)
        if not np.allclose(
            prior_post[comparable],
            next_pre[comparable],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "pre/post qvel channels are not temporally coherent "
                "(finite post[i] must equal finite pre[i+1])"
            )
    states = np.concatenate((pre, post[-1:]), axis=0)
    raw_nonfinite = bool(
        np.any(~np.isfinite(pre)) or np.any(~np.isfinite(post))
    )
    state_indices = np.arange(states.shape[0])
    if accepted_onset is None:
        phases = np.full(states.shape[0], "precontact", dtype=object)
    else:
        phases = np.full(states.shape[0], "first_event", dtype=object)
        phases[state_indices <= accepted_onset] = "precontact"
        if finalization_index is not None:
            phases[state_indices > finalization_index + 1] = "post_event"

    finite = np.isfinite(states)
    absolute = np.where(finite, np.abs(states), -np.inf)
    finite_exceedance = finite & (absolute > limit_rad_s)
    sample_exceedance = np.any(finite_exceedance, axis=1)
    finite_values = absolute[finite]
    max_abs = float(np.max(finite_values)) if finite_values.size else 0.0
    max_excess = max(0.0, max_abs - limit_rad_s)
    peak_flat = int(np.argmax(absolute)) if finite_values.size else 0
    peak_state, peak_joint = np.unravel_index(peak_flat, states.shape)
    has_exceedance = bool(np.any(sample_exceedance))
    phase_metrics: dict[str, float | int | bool] = {}
    for phase in ("precontact", "first_event", "post_event"):
        count = int(np.count_nonzero(sample_exceedance & (phases == phase)))
        phase_metrics[f"qvel_{phase}_violation"] = count > 0
        phase_metrics[f"qvel_{phase}_samples_over_rail"] = count
        phase_metrics[f"qvel_{phase}_seconds_over_rail"] = count * dt_s
    return {
        "qvel_nonfinite": raw_nonfinite,
        "qvel_finite_exceedance": has_exceedance,
        # Backward-compatible name: finite rail exceedance plus malformed
        # nonfinite samples. The latter is separately retained and fail-closed.
        "qvel_violation": bool(has_exceedance or raw_nonfinite),
        "qvel_max_abs_rad_s": max_abs,
        "qvel_max_excess_rad_s": max_excess,
        "qvel_samples_over_rail": int(np.count_nonzero(sample_exceedance)),
        "qvel_seconds_over_rail": (
            int(np.count_nonzero(sample_exceedance)) * dt_s
        ),
        "qvel_offending_joint": (
            ARM_JOINT_NAMES[peak_joint] if has_exceedance else None
        ),
        "qvel_offending_joint_index": peak_joint + 1 if has_exceedance else 0,
        "qvel_peak_phase": str(phases[peak_state]) if has_exceedance else None,
        **phase_metrics,
    }


def summarize_episode(
    trace: Mapping,
    *,
    nail_geometry: Mapping,
    qvel_limit_rad_s: float = HARDWARE_QVEL_LIMIT_RAD_S,
) -> dict:
    """Derive all sampled episode metrics from one shared physical trace."""

    if qvel_limit_rad_s <= 0 or not np.isfinite(qvel_limit_rad_s):
        raise ValueError("qvel_limit_rad_s must be finite and positive")
    arrays = _physical_arrays(trace)
    dt_s = float(trace.get("physics_dt_s", 0.0))
    if dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError("physics_dt_s must be finite and positive")
    phase_contract = trace.get("phase_contract", {})
    expected_phase_contract = {
        "physical_channels": "pre_integration",
        "tracker_derived_channels": "pre_integration",
        "tracker_nail_depth": "post_integration",
        "tracker_depth_lead_substeps": 1,
    }
    if phase_contract != expected_phase_contract:
        raise ValueError(
            "trace must declare the frozen pre-integration physical phase and "
            "one-substep tracker depth lead"
        )
    event_arrays = _event_arrays(trace, arrays["contact"].size)
    nail_xy = np.asarray(nail_geometry["nail_xy_m"], dtype=float)
    radius = float(nail_geometry["nail_radius_m"])
    if nail_xy.shape != (2,) or not np.isfinite(nail_xy).all() or radius <= 0:
        raise ValueError("invalid frozen nail geometry")

    first = trace.get("first_strike", {})
    reason = str(first.get("reason", "none"))
    if reason not in {"none", "success", "window"}:
        raise ValueError(f"invalid first-strike reason: {reason}")
    started = bool(first.get("started", False))
    finalized = bool(first.get("finalized", False))
    if reason != "none" and not (started and finalized):
        raise ValueError("success/window first strike must be started and finalized")
    if finalized and reason == "none":
        raise ValueError("a finalized first strike needs success/window reason")
    accepted_onset_raw = first.get("accepted_onset_index")
    accepted_onset = (
        None if accepted_onset_raw is None else int(accepted_onset_raw)
    )
    if started and accepted_onset is None:
        raise ValueError("a started first strike needs its tracker-accepted onset")
    if not started and accepted_onset is not None:
        raise ValueError("an unstarted first strike cannot have an accepted onset")
    event_started_indices = np.flatnonzero(event_arrays["started"])
    event_onset = (
        int(event_started_indices[0]) if event_started_indices.size else None
    )
    if event_onset != accepted_onset:
        raise ValueError(
            "first_strike accepted onset must match tracker_started stream"
        )
    finalization_index = event_arrays["finalization_index"]
    if finalized != (finalization_index is not None):
        raise ValueError(
            "first_strike finalized flag must match tracker_finalized stream"
        )
    onset, recontact, tail_fraction = _contact_structure(
        arrays["contact"],
        arrays["force"],
        dt_s,
        accepted_onset=accepted_onset,
        finalization_index=finalization_index,
    )
    qvel_diagnostics = _qvel_diagnostics(
        arrays["qvel"],
        arrays["qvel_post"],
        dt_s=dt_s,
        limit_rad_s=qvel_limit_rad_s,
        accepted_onset=accepted_onset,
        finalization_index=finalization_index,
    )

    first_contact_error: float | None = None
    within_radius = False
    funnel_eligible = False
    path_ratio: float | None = None
    lateral_excursion: float | None = None
    approach_angle: float | None = None
    contraction_ratio: float | None = None
    late_reexpansion = False
    if onset is not None:
        first_contact_error = float(
            np.linalg.norm(arrays["position"][onset, :2] - nail_xy)
        )
        within_radius = first_contact_error <= radius
        path_ratio = _path_length_ratio(arrays["position"][: onset + 1])
        lateral_excursion = float(
            np.max(
                np.linalg.norm(
                    arrays["position"][: onset + 1, :2] - nail_xy,
                    axis=1,
                )
            )
        )
        if onset >= 1:
            delta = arrays["position"][onset] - arrays["position"][onset - 1]
            downward_speed = float(-delta[2] / dt_s)
            if downward_speed > 0:
                transverse_speed = float(np.linalg.norm(delta[:2]) / dt_s)
                approach_angle = float(
                    np.degrees(np.arctan2(transverse_speed, downward_speed))
                )
        if onset >= WINDOW_SAMPLES_BEFORE:
            funnel_eligible = True
            window = arrays["position"][
                onset - WINDOW_SAMPLES_BEFORE : onset + 1
            ]
            terminal = compute_episode_metrics(
                window,
                nail_xy=nail_xy,
                dt_s=dt_s,
                tolerance_m=radius,
            )
            contraction_ratio = float(terminal["contraction_ratio"])
            terminal_error = np.asarray(
                terminal["transverse_error_m"], dtype=float
            )
            terminal_time = np.asarray(terminal["time_ms"], dtype=float)
            late_mask = (terminal_time >= -10.0 - 1e-9) & (
                terminal_time <= 1e-9
            )
            # Independent predicate: it remains visible even when the mutually
            # exclusive descriptive class is "off_target" or another label.
            late_reexpansion = bool(
                np.max(terminal_error[late_mask]) > 1.25 * radius
            )

    v_precontact = float(first.get("v_precontact_m_s", 0.0))
    delivered = float(first.get("delivered_n_s", 0.0))
    if (
        not np.isfinite(v_precontact)
        or v_precontact < 0
        or not np.isfinite(delivered)
        or delivered < 0
    ):
        raise ValueError("first-strike speed and delivered impulse must be finite/non-negative")
    if not np.isclose(
        event_arrays["impulse"][-1], delivered, atol=1e-9, rtol=1e-9
    ):
        raise ValueError(
            "first-strike delivered impulse must match the stored tracker stream"
        )

    reward = trace.get("reward", {})
    gamma = float(reward.get("gamma", 0.99))
    impact_return = _discounted(reward.get("impact_payout", ()), gamma)
    delivered_return = _discounted(reward.get("delivered_payout", ()), gamma)
    first_success = reason == "success"
    return {
        "episode_id": str(trace.get("episode_id", "")),
        "env_id": int(trace.get("env_id", -1)),
        "episode_ordinal": int(trace.get("episode_ordinal", -1)),
        "arm": str(trace.get("arm", "")),
        "task": str(trace.get("task", "")),
        "trace_digest": str(trace.get("trace_digest", "")),
        "physical_trace_phase": phase_contract["physical_channels"],
        "tracker_depth_lead_substeps": int(
            phase_contract["tracker_depth_lead_substeps"]
        ),
        "tracker_depth_lead_s": (
            int(phase_contract["tracker_depth_lead_substeps"]) * dt_s
        ),
        "first_contact_index": onset,
        "first_strike_started": started,
        "first_strike_finalized": finalized,
        "first_strike_reason": reason,
        "first_strike_success": first_success,
        "first_strike_productive": bool(first.get("productive", False)),
        "first_strike_v_precontact_m_s": v_precontact,
        "first_strike_useful_speed_m_s": v_precontact if first_success else 0.0,
        "first_strike_delivered_n_s": delivered,
        "first_strike_saturated": bool(first.get("saturated", False)),
        "overall_success": bool(trace.get("overall_success", False)),
        "recontact": recontact,
        "tail_fraction": tail_fraction,
        **qvel_diagnostics,
        "precontact_path_length_ratio": path_ratio,
        "precontact_lateral_excursion_m": lateral_excursion,
        "contact_approach_angle_deg": approach_angle,
        "first_contact_transverse_error_m": first_contact_error,
        "first_contact_within_nail_radius": within_radius,
        "terminal_funnel_eligible": funnel_eligible,
        "terminal_contraction_ratio": contraction_ratio,
        "late_lateral_reexpansion": late_reexpansion,
        "impact_return_discounted": impact_return,
        "delivered_return_discounted": delivered_return,
        "maximize_return_discounted": impact_return + delivered_return,
    }


def _mean(rows: Sequence[Mapping], key: str) -> float:
    if not rows:
        return 0.0
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {key}")
    return float(np.mean(values))


def _optional_mean(rows: Sequence[Mapping], key: str) -> float:
    eligible = [row for row in rows if row.get(key) is not None]
    return _mean(eligible, key) if eligible else 0.0


def aggregate_episode_metrics(
    rows: Sequence[Mapping], *, expected_episode_count: int | None = None
) -> dict:
    """Aggregate episode metrics with explicit, frozen denominators."""

    rows = list(rows)
    if expected_episode_count is not None and len(rows) != expected_episode_count:
        raise ValueError(
            f"expected exactly {expected_episode_count} episodes, found {len(rows)}"
        )
    if not rows:
        raise ValueError("cannot aggregate zero episodes")
    success = [row for row in rows if row["first_strike_reason"] == "success"]
    window = [row for row in rows if row["first_strike_reason"] == "window"]
    no_contact = [row for row in rows if not row["first_strike_started"]]
    unfinished = [
        row
        for row in rows
        if row["first_strike_started"] and not row["first_strike_finalized"]
    ]
    if len(success) + len(window) + len(no_contact) + len(unfinished) != len(rows):
        raise ValueError("first-strike statuses do not exhaust the episode population")
    contact_geometry = [
        row for row in rows if row["first_contact_transverse_error_m"] is not None
    ]
    funnel = [row for row in rows if row["terminal_funnel_eligible"]]
    successful_speed = [row for row in rows if row["first_strike_success"]]
    qvel_peak_episode = max(rows, key=lambda row: float(row["qvel_max_abs_rad_s"]))
    return {
        "n_episodes_sampled": len(rows),
        "first_strike_useful_speed_mean_sampled": _mean(
            rows, "first_strike_useful_speed_m_s"
        ),
        "successful_first_strike_v_precontact_mean_sampled": _mean(
            successful_speed, "first_strike_v_precontact_m_s"
        ),
        "first_strike_success_rate_sampled": _mean(rows, "first_strike_success"),
        "recontact_rate_sampled": _mean(rows, "recontact"),
        "first_strike_v_precontact_mean_sampled": _mean(
            [row for row in rows if row["first_strike_started"]],
            "first_strike_v_precontact_m_s",
        ),
        "first_strike_productive_rate_sampled": _mean(
            rows, "first_strike_productive"
        ),
        "first_strike_delivered_success_mean_sampled": _mean(
            success, "first_strike_delivered_n_s"
        ),
        "first_strike_delivered_window_mean_sampled": _mean(
            window, "first_strike_delivered_n_s"
        ),
        "first_strike_success_n_sampled": len(success),
        "first_strike_window_n_sampled": len(window),
        "first_strike_no_contact_n_sampled": len(no_contact),
        "first_strike_unfinished_event_n_sampled": len(unfinished),
        "tail_fraction_mean_sampled": _mean(rows, "tail_fraction"),
        "maximize_return_discounted_mean_sampled": _mean(
            rows, "maximize_return_discounted"
        ),
        "impact_return_discounted_mean_sampled": _mean(
            rows, "impact_return_discounted"
        ),
        "delivered_return_discounted_mean_sampled": _mean(
            rows, "delivered_return_discounted"
        ),
        "first_strike_saturation_rate_sampled": _mean(
            rows, "first_strike_saturated"
        ),
        "overall_success_rate_sampled": _mean(rows, "overall_success"),
        "qvel_violation_rate_sampled": _mean(rows, "qvel_violation"),
        "qvel_finite_exceedance_rate_sampled": _mean(
            rows, "qvel_finite_exceedance"
        ),
        "qvel_nonfinite_rate_sampled": _mean(rows, "qvel_nonfinite"),
        "qvel_max_abs_rad_s_sampled": float(
            qvel_peak_episode["qvel_max_abs_rad_s"]
        ),
        "qvel_max_excess_rad_s_sampled": float(
            qvel_peak_episode["qvel_max_excess_rad_s"]
        ),
        "qvel_samples_over_rail_sampled": sum(
            int(row["qvel_samples_over_rail"]) for row in rows
        ),
        "qvel_seconds_over_rail_sampled": sum(
            float(row["qvel_seconds_over_rail"]) for row in rows
        ),
        "qvel_peak_joint_index_sampled": int(
            qvel_peak_episode["qvel_offending_joint_index"]
        ),
        "qvel_precontact_exceedance_rate_sampled": _mean(
            rows, "qvel_precontact_violation"
        ),
        "qvel_precontact_samples_over_rail_sampled": sum(
            int(row["qvel_precontact_samples_over_rail"]) for row in rows
        ),
        "qvel_precontact_seconds_over_rail_sampled": sum(
            float(row["qvel_precontact_seconds_over_rail"]) for row in rows
        ),
        "qvel_first_event_exceedance_rate_sampled": _mean(
            rows, "qvel_first_event_violation"
        ),
        "qvel_first_event_samples_over_rail_sampled": sum(
            int(row["qvel_first_event_samples_over_rail"]) for row in rows
        ),
        "qvel_first_event_seconds_over_rail_sampled": sum(
            float(row["qvel_first_event_seconds_over_rail"]) for row in rows
        ),
        "qvel_post_event_exceedance_rate_sampled": _mean(
            rows, "qvel_post_event_violation"
        ),
        "qvel_post_event_samples_over_rail_sampled": sum(
            int(row["qvel_post_event_samples_over_rail"]) for row in rows
        ),
        "qvel_post_event_seconds_over_rail_sampled": sum(
            float(row["qvel_post_event_seconds_over_rail"]) for row in rows
        ),
        "precontact_path_length_ratio_mean_sampled": _optional_mean(
            rows, "precontact_path_length_ratio"
        ),
        "precontact_lateral_excursion_mean_sampled": _optional_mean(
            rows, "precontact_lateral_excursion_m"
        ),
        "contact_approach_angle_mean_sampled": _optional_mean(
            rows, "contact_approach_angle_deg"
        ),
        "first_contact_transverse_error_mean_sampled": _mean(
            contact_geometry, "first_contact_transverse_error_m"
        ),
        "first_contact_within_nail_radius_rate_sampled": _mean(
            rows, "first_contact_within_nail_radius"
        ),
        "first_contact_geometry_n_sampled": len(contact_geometry),
        "terminal_contraction_ratio_mean_sampled": _mean(
            funnel, "terminal_contraction_ratio"
        ),
        "late_lateral_reexpansion_rate_sampled": _mean(
            funnel, "late_lateral_reexpansion"
        ),
        "terminal_funnel_eligible_n_sampled": len(funnel),
    }


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def exact_seed_tests(
    treatment: Sequence[float], control: Sequence[float]
) -> dict:
    """Exhaustive two-sided seed-label MWU and mean-difference permutation tests."""

    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    if (
        treatment.ndim != 1
        or control.ndim != 1
        or not treatment.size
        or not control.size
        or not np.isfinite(treatment).all()
        or not np.isfinite(control).all()
    ):
        raise ValueError("exact tests require non-empty finite one-dimensional samples")
    pooled = np.concatenate([treatment, control])
    ranks = _midranks(pooled)
    n_treatment = treatment.size
    n_control = control.size
    u_center = n_treatment * n_control / 2.0
    observed_u = float(
        np.sum(ranks[:n_treatment]) - n_treatment * (n_treatment + 1) / 2.0
    )
    observed_mean_difference = float(np.mean(treatment) - np.mean(control))

    assignment_count = 0
    mwu_extreme = 0
    mean_extreme = 0
    index = np.arange(pooled.size)
    observed_u_distance = abs(observed_u - u_center)
    observed_mean_distance = abs(observed_mean_difference)
    for chosen_tuple in combinations(range(pooled.size), n_treatment):
        chosen = np.fromiter(chosen_tuple, dtype=int, count=n_treatment)
        mask = np.ones(pooled.size, dtype=bool)
        mask[chosen] = False
        permuted_u = float(
            np.sum(ranks[chosen]) - n_treatment * (n_treatment + 1) / 2.0
        )
        permuted_difference = float(
            np.mean(pooled[chosen]) - np.mean(pooled[index[mask]])
        )
        assignment_count += 1
        mwu_extreme += abs(permuted_u - u_center) >= observed_u_distance - 1e-12
        mean_extreme += (
            abs(permuted_difference) >= observed_mean_distance - 1e-12
        )
    return {
        "n_treatment": int(n_treatment),
        "n_control": int(n_control),
        "assignment_count": assignment_count,
        "u_treatment": observed_u,
        "a12_treatment_over_control": observed_u
        / float(n_treatment * n_control),
        "mwu_two_sided_exact_p": mwu_extreme / assignment_count,
        "mean_difference": observed_mean_difference,
        "mean_difference_two_sided_exact_p": mean_extreme / assignment_count,
    }


def _arm_summary(values: np.ndarray) -> dict:
    return {
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _bootstrap_difference(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    treatment_draws = rng.choice(
        treatment, size=(samples, treatment.size), replace=True
    )
    control_draws = rng.choice(control, size=(samples, control.size), replace=True)
    return treatment_draws.mean(axis=1) - control_draws.mean(axis=1)


def _success_guardrails(
    by_arm: Mapping[str, Sequence[Mapping]], treatment: str, control: str
) -> dict:
    result = {}
    for label, metric in (
        ("first_window", "first_strike_success_rate_sampled"),
        ("overall", "overall_success_rate_sampled"),
    ):
        treatment_mean = _mean(by_arm[treatment], metric)
        control_mean = _mean(by_arm[control], metric)
        difference = treatment_mean - control_mean
        result[label] = {
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "difference": difference,
            "absolute_90": treatment_mean >= 0.90 - 1e-12,
            "within_five_percentage_points": difference >= -0.05 - 1e-12,
            "passed": treatment_mean >= 0.90 - 1e-12
            and difference >= -0.05 - 1e-12,
        }
    return result


def _is_hex_digest(value: object, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"invalid boolean scalar: {value!r}")


def _episode_trace_digest(trace: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "physical": trace["physical"],
                "event_trace": trace["event_trace"],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


REPRESENTATIVE_SELECTION_RULE = (
    "lexicographically_smallest_training_seed_env_id_episode_ordinal"
)


def _seed_qualified_representative_trace(
    trace: Mapping, *, training_seed: int
) -> dict:
    """Copy one raw trace and add its seed-qualified analysis identity."""

    reserved = {"source_episode_id", "source_training_seed"} & trace.keys()
    if reserved:
        raise ValueError(
            "raw representative trace contains reserved analysis fields: "
            f"{sorted(reserved)}"
        )
    source_episode_id = trace.get("episode_id")
    if not isinstance(source_episode_id, str) or not source_episode_id:
        raise ValueError("representative source episode ID must be non-empty")
    try:
        source_training_seed = (
            trace["training_seed"] if "training_seed" in trace else None
        )
        trace_seed = (
            training_seed
            if source_training_seed is None
            else int(source_training_seed)
        )
        env_id = int(trace["env_id"])
        episode_ordinal = int(trace["episode_ordinal"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("representative coordinates must be integers") from error
    if trace_seed != training_seed:
        raise ValueError("representative trace training seed binding mismatch")
    if env_id < 0 or episode_ordinal < 0:
        raise ValueError("representative coordinates must be nonnegative")
    arm = str(trace.get("arm", ""))
    if arm not in ARM_TASKS:
        raise ValueError(f"representative trace has invalid arm: {arm!r}")

    qualified = dict(trace)
    qualified.update(
        {
            "episode_id": (
                f"{arm}-seed{training_seed}-env{env_id}-episode"
                f"{episode_ordinal}"
            ),
            "source_episode_id": source_episode_id,
            "source_training_seed": source_training_seed,
            "training_seed": training_seed,
        }
    )
    return qualified


def _select_representative_traces_from_payloads(
    rows: Sequence[Mapping],
    sampled_payloads: Iterable[Mapping],
) -> dict:
    """Select deterministically from caller-held immutable payload snapshots."""

    rows = list(rows)
    selected: dict[tuple[str, bool], tuple[tuple[int, int, int], dict]] = {}
    for row, payload in zip(rows, sampled_payloads, strict=True):
        arm = str(row.get("treatment", ""))
        if arm not in ARM_TASKS:
            raise ValueError(f"representative row has invalid treatment: {arm!r}")
        try:
            training_seed = int(row["training_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "representative row training seed must be an integer"
            ) from error
        if (
            str(payload.get("treatment", "")) != arm
            or int(payload.get("training_seed", -1)) != training_seed
        ):
            raise ValueError("representative artifact row binding mismatch")

        for raw_trace in payload.get("episodes", ()):
            trace = _seed_qualified_representative_trace(
                raw_trace,
                training_seed=training_seed,
            )
            if str(trace["arm"]) != arm:
                raise ValueError("representative trace treatment binding mismatch")
            coordinate = (
                training_seed,
                int(trace["env_id"]),
                int(trace["episode_ordinal"]),
            )
            stratum = (arm, bool(trace.get("overall_success", False)))
            previous = selected.get(stratum)
            if previous is None or coordinate < previous[0]:
                selected[stratum] = (coordinate, trace)
            elif coordinate == previous[0] and trace != previous[1]:
                raise ValueError(
                    "distinct representative traces share one seed/env/ordinal"
                )

    traces = []
    omitted = []
    for arm in ARM_ORDER:
        for outcome, overall_success in (
            ("success", True),
            ("failure", False),
        ):
            candidate = selected.get((arm, overall_success))
            if candidate is None:
                omitted.append(
                    {
                        "arm": arm,
                        "outcome": outcome,
                        "overall_success": overall_success,
                        "reason": "no sampled episodes in outcome stratum",
                    }
                )
            else:
                traces.append(candidate[1])
    return {
        "selection_rule": REPRESENTATIVE_SELECTION_RULE,
        "traces": traces,
        "omitted_strata": omitted,
    }


def select_representative_traces(rows: Sequence[Mapping]) -> dict:
    """Verify artifact bytes, then select each representative stratum."""

    rows = list(rows)
    return _select_representative_traces_from_payloads(
        rows,
        _verified_sampled_artifact_payloads(rows),
    )


def _treatment_config_digest(row: Mapping) -> str:
    treatment = str(row["treatment"])
    return hashlib.sha256(
        json.dumps(
            {
                "schema_version": 1,
                "task": str(row["task"]),
                "treatment": treatment,
                "payout_semantics": PAYOUT_SEMANTICS[treatment],
                "impact_weight": float(row["impact_weight"]),
                "delivered_weight": float(row["delivered_weight"]),
                "event_i_ref_n_s": float(row["event_i_ref_n_s"]),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _sampled_payload_from_bytes(artifact_bytes: bytes) -> dict:
    with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as saved:
        if saved.files != ["payload_json"]:
            raise ValueError("NPZ must contain exactly payload_json")
        scalar = saved["payload_json"]
        if scalar.shape != () or scalar.dtype.kind != "U":
            raise ValueError("payload_json must be a scalar Unicode value")
        return json.loads(str(scalar))


def _load_and_recompute_sampled_artifact(
    artifact_bytes: bytes, artifact_sha256: str
) -> tuple[
    dict,
    str,
    str,
    dict | None,
    dict | None,
    tuple[str, ...],
]:
    del artifact_sha256  # Caller verified the bytes before parsing.
    payload = _sampled_payload_from_bytes(artifact_bytes)
    embedded_digest = str(payload.get("payload_digest", ""))
    digest_payload = dict(payload)
    digest_payload.pop("payload_digest", None)
    actual_payload_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    semantic_reasons: list[str] = []
    if payload.get("schema_version") != 2:
        semantic_reasons.append("raw schema version is not 2")
    episodes = payload.get("episodes")
    expected_count = int(payload.get("expected_episode_count", -1))
    if expected_count != EXPECTED_EPISODES_PER_SEED:
        semantic_reasons.append("raw expected episode count is not 512")
    if not isinstance(episodes, list) or len(episodes) != expected_count:
        semantic_reasons.append("raw artifact does not contain 512 episodes")
        return (
            payload,
            embedded_digest,
            actual_payload_digest,
            None,
            None,
            tuple(semantic_reasons),
        )
    treatment = str(payload.get("treatment", ""))
    task = str(payload.get("task", ""))
    geometry = payload.get("nail_geometry", {})
    try:
        impulse_limits = np.asarray(
            payload["impulse_limits_n_m_s"], dtype=float
        )
        if (
            impulse_limits.shape != (6,)
            or not np.isfinite(impulse_limits).all()
            or np.any(impulse_limits <= 0.0)
        ):
            raise ValueError("limits must be six finite positive values")
        if not np.array_equal(
            impulse_limits,
            np.asarray(EXPECTED_IMPULSE_LIMITS_N_M_S, dtype=float),
        ):
            raise ValueError("limits differ from the frozen campaign values")
    except Exception as error:
        semantic_reasons.append(f"raw impulse limits are invalid: {error}")
        impulse_limits = None
    episode_metrics: list[dict] = []
    env_ordinals: dict[int, set[int]] = defaultdict(set)
    legacy_success: list[float] = []
    legacy_delivered: list[float] = []
    legacy_worst_ratio: list[float] = []
    sampled_impossible = 0
    sampled_lambda_dead = 0
    for index, trace in enumerate(episodes):
        episode_id = str(trace.get("episode_id", f"index-{index}"))
        try:
            actual_trace_digest = _episode_trace_digest(trace)
        except Exception as error:
            semantic_reasons.append(
                f"{episode_id}: cannot recompute episode trace digest: {error}"
            )
            continue
        if str(trace.get("trace_digest", "")) != actual_trace_digest:
            semantic_reasons.append(
                f"{episode_id}: episode trace digest mismatch"
            )
        if trace.get("arm") != treatment or trace.get("task") != task:
            semantic_reasons.append(
                f"{episode_id}: episode treatment/task binding mismatch"
            )
        if trace.get("nail_geometry") != geometry:
            semantic_reasons.append(
                f"{episode_id}: episode nail geometry binding mismatch"
            )
        try:
            metrics = summarize_episode(trace, nail_geometry=geometry)
        except Exception as error:
            semantic_reasons.append(
                f"{episode_id}: sampled episode is invalid: {error}"
            )
            continue
        episode_metrics.append(metrics)
        env_id = int(metrics["env_id"])
        ordinal = int(metrics["episode_ordinal"])
        if env_id in range(EXPECTED_NUM_ENVS):
            env_ordinals[env_id].add(ordinal)
        try:
            peak_lambda = np.asarray(
                trace["episode_peak_lambda"], dtype=float
            )
            delivered_accumulator = float(
                trace["episode_delivered_accumulator_n_s"]
            )
            if (
                impulse_limits is None
                or peak_lambda.shape != impulse_limits.shape
                or not np.isfinite(peak_lambda).all()
                or not np.isfinite(delivered_accumulator)
            ):
                raise ValueError("invalid peak-lambda/delivered episode values")
            lam_worst = float(np.max(peak_lambda))
            legacy_success.append(float(bool(trace["overall_success"])))
            legacy_delivered.append(delivered_accumulator)
            legacy_worst_ratio.append(
                float(np.max(peak_lambda / impulse_limits))
            )
            if bool(trace["overall_success"]) and (
                delivered_accumulator <= 0.0 or lam_worst <= 0.0
            ):
                sampled_impossible += 1
            if delivered_accumulator > 0.0 and lam_worst <= 0.0:
                sampled_lambda_dead += 1
        except Exception as error:
            semantic_reasons.append(
                f"{episode_id}: cannot recompute legacy sampled metrics: {error}"
            )
    if any(
        env_ordinals.get(env_id) != {0, 1}
        for env_id in range(EXPECTED_NUM_ENVS)
    ):
        semantic_reasons.append(
            "raw episodes are not exactly ordinals 0/1 for all 256 environments"
        )
    aggregate = None
    raw_bindings = None
    if len(episode_metrics) == EXPECTED_EPISODES_PER_SEED:
        try:
            aggregate = aggregate_episode_metrics(
                episode_metrics,
                expected_episode_count=EXPECTED_EPISODES_PER_SEED,
            )
        except Exception as error:
            semantic_reasons.append(
                f"cannot recompute sampled aggregate: {error}"
            )
    if len(legacy_success) == EXPECTED_EPISODES_PER_SEED:
        mean_invariants = payload.get("mean_rollout_invariants")
        try:
            mean_impossible = int(mean_invariants["impossible_success_n"])
            mean_lambda_dead = int(mean_invariants["lambda_dead_n"])
            if min(mean_impossible, mean_lambda_dead) < 0:
                raise ValueError("invariant counts must be nonnegative")
        except Exception as error:
            semantic_reasons.append(
                f"raw mean-rollout invariants are invalid: {error}"
            )
        else:
            raw_bindings = {
                "legacy_sampled": {
                    "success_rate_sampled": float(np.mean(legacy_success)),
                    "worst_ratio_max_sampled": float(
                        np.max(legacy_worst_ratio)
                    ),
                    "delivered_mean_sampled": float(
                        np.mean(legacy_delivered)
                    ),
                },
                "combined_invariants": {
                    "impossible_success_n": (
                        mean_impossible + sampled_impossible
                    ),
                    "lambda_dead_n": mean_lambda_dead + sampled_lambda_dead,
                },
            }
    return (
        payload,
        embedded_digest,
        actual_payload_digest,
        aggregate,
        raw_bindings,
        tuple(semantic_reasons),
    )


def _sampled_artifact_reasons(
    row: Mapping,
    row_name: str,
    *,
    artifact_bytes: bytes | None = None,
) -> list[str]:
    reasons: list[str] = []
    path = Path(str(row.get("sampled_trace_path", "")))
    if artifact_bytes is None:
        if not path.is_file():
            return [f"{row_name}: sampled trace artifact is missing"]
        try:
            artifact_bytes = path.read_bytes()
        except OSError as error:
            return [
                f"{row_name}: sampled trace artifact cannot be read: {error}"
            ]
    expected_artifact_sha = str(
        row.get("sampled_trace_artifact_sha256", "")
    )
    actual_artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_artifact_sha != expected_artifact_sha:
        reasons.append(f"{row_name}: sampled trace artifact SHA-256 mismatch")
    try:
        (
            payload,
            embedded_digest,
            actual_payload_digest,
            recomputed_aggregate,
            recomputed_raw_bindings,
            semantic_reasons,
        ) = _load_and_recompute_sampled_artifact(
            artifact_bytes, actual_artifact_sha
        )
    except Exception as error:
        reasons.append(f"{row_name}: sampled trace artifact is invalid: {error}")
        return reasons
    reasons.extend(f"{row_name}: {reason}" for reason in semantic_reasons)
    row_digest = str(row.get("sampled_trace_digest", ""))
    if (
        embedded_digest != actual_payload_digest
        or row_digest != actual_payload_digest
    ):
        reasons.append(f"{row_name}: sampled trace digest mismatch")
    provenance = payload.get("provenance", {})
    expected_bindings = {
        "checkpoint_sha256": row.get("checkpoint_sha256"),
        "accepted_checkpoint_sha256": row.get(
            "accepted_checkpoint_sha256"
        ),
        "campaign_config_sha256": row.get("campaign_config_sha256"),
        "treatment_config_sha256": row.get("treatment_config_sha256"),
        "nail_asset_sha256": row.get("nail_asset_sha256"),
        "accepted_manifest_sha256": row.get("accepted_manifest_sha256"),
        "training_code_revision": row.get("training_code_revision"),
        "training_asset_revision": row.get("training_asset_revision"),
    }
    for key, expected in expected_bindings.items():
        if provenance.get(key) != expected:
            reasons.append(f"{row_name}: raw artifact {key} binding mismatch")
    for prefix, payload_key in (
        ("git", "code_git"),
        ("asset_git", "asset_git"),
    ):
        binding = provenance.get(payload_key, {})
        try:
            row_dirty = _parse_bool(row.get(f"{prefix}_dirty", True))
        except ValueError:
            row_dirty = True
        if binding.get("revision") != row.get(
            f"{prefix}_revision"
        ) or bool(binding.get("dirty", True)) != row_dirty:
            reasons.append(
                f"{row_name}: raw artifact {payload_key} binding mismatch"
            )
    raw_bindings = {
        "treatment": row.get("treatment"),
        "task": row.get("task"),
        "training_seed": int(row.get("training_seed", -1)),
        "imp_max_p": float(row.get("imp_max_p", np.nan)),
        "event_i_ref_n_s": float(row.get("event_i_ref_n_s", np.nan)),
    }
    for key, expected in raw_bindings.items():
        if payload.get(key) != expected:
            reasons.append(f"{row_name}: raw artifact {key} binding mismatch")
    expected_weights = {
        "impact_progress": float(row.get("impact_weight", np.nan)),
        "delivered_impulse": float(row.get("delivered_weight", np.nan)),
    }
    if payload.get("weights") != expected_weights:
        reasons.append(f"{row_name}: raw artifact reward weights mismatch")
    expected_rng = {
        "reset": int(row.get("reset_rng_seed", -1)),
        "observation": int(row.get("observation_rng_seed", -1)),
        "action": int(row.get("action_rng_seed", -1)),
    }
    if payload.get("rng_streams") != expected_rng:
        reasons.append(f"{row_name}: raw artifact RNG binding mismatch")
    try:
        expected_evaluation_contract = {
            "base_rng_seed": int(row.get("seed", -1)),
            "num_envs": int(row.get("num_envs", -1)),
            "episodes_per_env": int(row.get("episodes_per_env_sampled", -1)),
            "episode_len_s": float(row.get("episode_len_s", np.nan)),
            "mean_nsteps": int(row.get("nsteps", -1)),
            "completion_rule": row.get("sampled_completion_rule"),
            "stochastic_actions": _parse_bool(
                row.get("sampled_actions_stochastic", False)
            ),
            "reset_position_noise_rad": [
                float(row.get("reset_position_noise_min_rad", np.nan)),
                float(row.get("reset_position_noise_max_rad", np.nan)),
            ],
            "actor_observation_corruption": _parse_bool(
                row.get("actor_observation_corruption", False)
            ),
            "critic_observation_corruption": _parse_bool(
                row.get("critic_observation_corruption", True)
            ),
            "physics_dt_s": float(row.get("physics_dt_s", np.nan)),
            "control_decimation": int(row.get("control_decimation", -1)),
            "fixed_impedance_signature_sha256": row.get(
                "fixed_impedance_signature_sha256"
            ),
            "fixed_action_signature_sha256": row.get(
                "fixed_action_signature_sha256"
            ),
        }
    except (TypeError, ValueError) as error:
        reasons.append(f"{row_name}: invalid evaluation contract scalars: {error}")
    else:
        if payload.get("evaluation_contract") != expected_evaluation_contract:
            reasons.append(
                f"{row_name}: raw evaluation contract binding mismatch"
            )
    if (
        payload.get("nail_geometry", {}).get("source_sha256")
        != row.get("nail_asset_sha256")
    ):
        reasons.append(f"{row_name}: raw nail geometry binding mismatch")
    if recomputed_aggregate is not None:
        for key, expected in recomputed_aggregate.items():
            try:
                actual = float(row[key])
            except (KeyError, TypeError, ValueError):
                reasons.append(
                    f"{row_name}: sampled aggregate mismatch for {key}"
                )
                continue
            if not np.isclose(
                actual, float(expected), rtol=1e-9, atol=1e-12
            ):
                reasons.append(
                    f"{row_name}: sampled aggregate mismatch for {key}"
                )
    if recomputed_raw_bindings is not None:
        for key, expected in recomputed_raw_bindings[
            "legacy_sampled"
        ].items():
            try:
                actual = float(row[key])
            except (KeyError, TypeError, ValueError):
                actual = np.nan
            if not np.isclose(
                actual, float(expected), rtol=1e-6, atol=1e-8
            ):
                reasons.append(
                    f"{row_name}: legacy sampled metric mismatch for {key}"
                )
        for key, expected in recomputed_raw_bindings[
            "combined_invariants"
        ].items():
            try:
                actual = int(row[key])
            except (KeyError, TypeError, ValueError):
                actual = -1
            if actual != int(expected):
                reasons.append(
                    f"{row_name}: combined invariant mismatch for {key}"
                )
    return reasons


def _verified_sampled_artifact_payloads(
    rows: Sequence[Mapping],
) -> Iterable[dict]:
    """Yield one payload at a time from exact bytes revalidated for reporting."""

    for row in rows:
        arm = str(row.get("treatment", ""))
        row_name = str(
            row.get("name", f"{arm}/seed{row.get('training_seed')}")
        )
        path = Path(str(row.get("sampled_trace_path", "")))
        try:
            artifact_bytes = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"{row_name}: sampled trace artifact cannot be read: {error}"
            ) from error
        reasons = _sampled_artifact_reasons(
            row,
            row_name,
            artifact_bytes=artifact_bytes,
        )
        if reasons:
            raise ValueError(
                "representative sampled artifact verification failed: "
                + "; ".join(reasons)
            )
        yield _sampled_payload_from_bytes(artifact_bytes)


def _qvel_sampled_metrics_reasons(row: Mapping, row_name: str) -> list[str]:
    """Validate qvel diagnostics without requiring finite rail exceedance = 0."""

    reasons: list[str] = []
    rate_fields = (
        "qvel_violation_rate_sampled",
        "qvel_finite_exceedance_rate_sampled",
        "qvel_nonfinite_rate_sampled",
        "qvel_precontact_exceedance_rate_sampled",
        "qvel_first_event_exceedance_rate_sampled",
        "qvel_post_event_exceedance_rate_sampled",
    )
    try:
        rates = {field: float(row[field]) for field in rate_fields}
        max_abs = float(row["qvel_max_abs_rad_s_sampled"])
        max_excess = float(row["qvel_max_excess_rad_s_sampled"])
        peak_joint_raw = float(row["qvel_peak_joint_index_sampled"])
        count_values = {
            phase: float(row[f"qvel_{phase}_samples_over_rail_sampled"])
            for phase in ("precontact", "first_event", "post_event")
        }
        total_count_raw = float(row["qvel_samples_over_rail_sampled"])
        seconds = {
            phase: float(row[f"qvel_{phase}_seconds_over_rail_sampled"])
            for phase in ("precontact", "first_event", "post_event")
        }
        total_seconds = float(row["qvel_seconds_over_rail_sampled"])
    except (KeyError, TypeError, ValueError) as error:
        return [f"{row_name}: invalid qvel diagnostic scalar: {error}"]
    values = np.asarray(
        [
            *rates.values(),
            max_abs,
            max_excess,
            *seconds.values(),
            total_seconds,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        reasons.append(f"{row_name}: qvel diagnostics must be finite")
        return reasons
    if (
        not peak_joint_raw.is_integer()
        or not total_count_raw.is_integer()
        or any(not value.is_integer() for value in count_values.values())
    ):
        reasons.append(f"{row_name}: qvel counts/joint index must be integers")
        return reasons
    peak_joint = int(peak_joint_raw)
    counts = {phase: int(value) for phase, value in count_values.items()}
    total_count = int(total_count_raw)
    rate_episode_counts: dict[str, int] = {}
    for field, rate in rates.items():
        episode_count = rate * EXPECTED_EPISODES_PER_SEED
        if not np.isclose(
            episode_count, round(episode_count), rtol=0.0, atol=1e-9
        ):
            reasons.append(
                f"{row_name}: qvel rates must resolve to episode counts"
            )
        rate_episode_counts[field] = int(round(episode_count))
    if any(not 0.0 <= rate <= 1.0 for rate in rates.values()):
        reasons.append(f"{row_name}: qvel rates must be in [0, 1]")
    if rates["qvel_nonfinite_rate_sampled"] != 0.0:
        reasons.append(f"{row_name}: nonfinite 500 Hz qvel sample")
    if max_abs < 0.0 or max_excess < 0.0:
        reasons.append(f"{row_name}: qvel maxima must be non-negative")
    expected_excess = max(0.0, max_abs - HARDWARE_QVEL_LIMIT_RAD_S)
    if not np.isclose(max_excess, expected_excess, rtol=0.0, atol=1e-12):
        reasons.append(f"{row_name}: qvel max excess is inconsistent with max abs")
    if (
        total_count < 0
        or any(count < 0 for count in counts.values())
        or total_count != sum(counts.values())
    ):
        reasons.append(f"{row_name}: qvel phase sample counts are inconsistent")
    max_states_per_episode = (
        int(
            round(
                EXPECTED_EVALUATION_EPISODE_LEN_S
                / EXPECTED_PHYSICS_DT_S
            )
        )
        + 1
    )
    if total_count > EXPECTED_EPISODES_PER_SEED * max_states_per_episode:
        reasons.append(
            f"{row_name}: qvel sample count exceeds sampled population support"
        )
    if not np.isclose(
        total_seconds,
        total_count * EXPECTED_PHYSICS_DT_S,
        rtol=0.0,
        atol=1e-12,
    ) or any(
        not np.isclose(
            seconds[phase],
            counts[phase] * EXPECTED_PHYSICS_DT_S,
            rtol=0.0,
            atol=1e-12,
        )
        for phase in counts
    ):
        reasons.append(f"{row_name}: qvel seconds do not match 500 Hz sample counts")
    if peak_joint not in range(7):
        reasons.append(f"{row_name}: qvel peak joint index must be in 0..6")
    if (total_count > 0) != (
        rates["qvel_finite_exceedance_rate_sampled"] > 0.0
    ):
        reasons.append(f"{row_name}: qvel episode rate/count presence mismatch")
    finite_episode_count = rate_episode_counts[
        "qvel_finite_exceedance_rate_sampled"
    ]
    if (
        total_count < finite_episode_count
        or total_count > finite_episode_count * max_states_per_episode
    ):
        reasons.append(
            f"{row_name}: qvel sample count/rate support is inconsistent"
        )
    phase_episode_counts = {
        phase: rate_episode_counts[
            f"qvel_{phase}_exceedance_rate_sampled"
        ]
        for phase in ("precontact", "first_event", "post_event")
    }
    for phase, episode_count in phase_episode_counts.items():
        if (
            (counts[phase] > 0) != (episode_count > 0)
            or counts[phase] < episode_count
            or counts[phase] > episode_count * max_states_per_episode
        ):
            reasons.append(
                f"{row_name}: {phase} qvel phase rate/count is inconsistent"
            )
    if not (
        max(phase_episode_counts.values(), default=0)
        <= finite_episode_count
        <= sum(phase_episode_counts.values())
    ):
        reasons.append(
            f"{row_name}: qvel finite episode rate is inconsistent with phases"
        )
    finite_rate = rates["qvel_finite_exceedance_rate_sampled"]
    nonfinite_rate = rates["qvel_nonfinite_rate_sampled"]
    legacy_rate = rates["qvel_violation_rate_sampled"]
    union_lower = max(finite_rate, nonfinite_rate)
    union_upper = min(1.0, finite_rate + nonfinite_rate)
    if not union_lower - 1e-12 <= legacy_rate <= union_upper + 1e-12:
        reasons.append(f"{row_name}: legacy qvel violation rate is inconsistent")
    if nonfinite_rate == 0.0 and not np.isclose(
        legacy_rate, finite_rate, rtol=0.0, atol=1e-12
    ):
        reasons.append(f"{row_name}: legacy qvel violation rate/source mismatch")
    if total_count == 0:
        if max_excess != 0.0 or peak_joint != 0 or max_abs > HARDWARE_QVEL_LIMIT_RAD_S:
            reasons.append(
                f"{row_name}: qvel zero-count maxima/peak joint are inconsistent"
            )
    elif (
        max_excess <= 0.0
        or max_abs <= HARDWARE_QVEL_LIMIT_RAD_S
        or peak_joint == 0
    ):
        reasons.append(
            f"{row_name}: qvel exceedance maxima/peak joint are inconsistent"
        )
    if total_count > 0 and peak_joint == 0:
        reasons.append(f"{row_name}: qvel exceedance needs an offending joint")
    return reasons


def _campaign_validity(rows: Sequence[Mapping]) -> list[str]:
    reasons: list[str] = []
    by_arm: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        arm = str(row.get("treatment", ""))
        by_arm[arm].append(row)
        row_name = str(row.get("name", f"{arm}/seed{row.get('training_seed')}"))
        if arm not in ARM_TASKS:
            reasons.append(f"{row_name}: unknown treatment {arm!r}")
            continue
        if row.get("task") != ARM_TASKS[arm]:
            reasons.append(f"{row_name}: task id mismatch")
        if float(row.get("impact_weight", np.nan)) != 8.0:
            reasons.append(f"{row_name}: impact weight is not 8")
        if float(row.get("delivered_weight", np.nan)) != 2.0:
            reasons.append(f"{row_name}: delivered weight is not 2")
        if not np.isclose(
            float(row.get("event_i_ref_n_s", np.nan)),
            ARM_EVENT_I_REF_N_S[arm],
            rtol=0.0,
            atol=1e-12,
        ):
            reasons.append(f"{row_name}: event impulse reference mismatch")
        if float(row.get("imp_max_p", np.nan)) != 0.0:
            reasons.append(f"{row_name}: imp_max_p is not zero")
        if int(row.get("n_episodes_sampled", -1)) != EXPECTED_EPISODES_PER_SEED:
            reasons.append(f"{row_name}: sampled episode count is not 512")
        if int(row.get("num_envs", -1)) != EXPECTED_NUM_ENVS:
            reasons.append(f"{row_name}: num_envs is not 256")
        if int(row.get("episodes_per_env_sampled", -1)) != EXPECTED_EPISODES_PER_ENV:
            reasons.append(f"{row_name}: sampled episodes are not two per environment")
        expected_row = _frozen_campaign_row(arm, int(row.get("training_seed", -1)))
        if row.get("name") != expected_row["run_name"]:
            reasons.append(f"{row_name}: deterministic run name mismatch")
        checkpoint_path = Path(str(row.get("checkpoint_path", "")))
        if checkpoint_path.name != "model_499.pt":
            reasons.append(f"{row_name}: checkpoint is not model_499.pt")
        if not (
            checkpoint_path.parent.name == expected_row["run_name"]
            or checkpoint_path.parent.name.endswith(
                f"_{expected_row['run_name']}"
            )
        ):
            reasons.append(f"{row_name}: checkpoint/run-name binding mismatch")
        if int(row.get("seed", -1)) != CAMPAIGN_EVALUATOR_SEED:
            reasons.append(f"{row_name}: base evaluator RNG is not frozen")
        expected_rng_by_field = {
            "reset_rng_seed": CAMPAIGN_EVALUATOR_RNG["reset"],
            "observation_rng_seed": CAMPAIGN_EVALUATOR_RNG["observation"],
            "action_rng_seed": CAMPAIGN_EVALUATOR_RNG["action"],
        }
        for rng_field, expected_seed in expected_rng_by_field.items():
            if int(row.get(rng_field, -1)) != expected_seed:
                label = rng_field.replace("_seed", "").replace("_", " ")
                reasons.append(f"{row_name}: {label} is not frozen")
        if not np.isclose(
            float(row.get("episode_len_s", np.nan)),
            EXPECTED_EVALUATION_EPISODE_LEN_S,
            rtol=0.0,
            atol=0.0,
        ):
            reasons.append(f"{row_name}: 4.0 s evaluation cutoff drift")
        if int(row.get("nsteps", -1)) != EXPECTED_EVALUATION_MEAN_NSTEPS:
            reasons.append(f"{row_name}: 400 diagnostic mean steps drift")
        reset_range = (
            float(row.get("reset_position_noise_min_rad", np.nan)),
            float(row.get("reset_position_noise_max_rad", np.nan)),
        )
        if reset_range != (-0.05, 0.05):
            reasons.append(f"{row_name}: reset noise drift")
        try:
            actor_corruption = _parse_bool(
                row.get("actor_observation_corruption", False)
            )
            critic_corruption = _parse_bool(
                row.get("critic_observation_corruption", True)
            )
            stochastic_actions = _parse_bool(
                row.get("sampled_actions_stochastic", False)
            )
        except ValueError:
            actor_corruption = False
            critic_corruption = True
            stochastic_actions = False
        if not actor_corruption:
            reasons.append(f"{row_name}: actor corruption is not enabled")
        if critic_corruption:
            reasons.append(f"{row_name}: critic corruption is not disabled")
        if not stochastic_actions:
            reasons.append(f"{row_name}: sampled actions are not stochastic")
        if (
            row.get("sampled_completion_rule")
            != "first_two_completions_per_environment"
        ):
            reasons.append(f"{row_name}: completion rule drift")
        if not np.isclose(
            float(row.get("physics_dt_s", np.nan)),
            EXPECTED_PHYSICS_DT_S,
            rtol=0.0,
            atol=1e-15,
        ):
            reasons.append(f"{row_name}: 500 Hz physics timestep drift")
        if int(row.get("control_decimation", -1)) != EXPECTED_CONTROL_DECIMATION:
            reasons.append(f"{row_name}: control decimation drift")
        if (
            row.get("fixed_impedance_signature_sha256")
            != EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256
        ):
            reasons.append(f"{row_name}: fixed impedance signature mismatch")
        if (
            row.get("fixed_action_signature_sha256")
            != EXPECTED_FIXED_ACTION_SIGNATURE_SHA256
        ):
            reasons.append(f"{row_name}: fixed action signature mismatch")
        if row.get("accepted_checkpoint_sha256") != row.get(
            "checkpoint_sha256"
        ):
            reasons.append(f"{row_name}: accepted checkpoint SHA-256 mismatch")
        if not _is_hex_digest(row.get("accepted_manifest_sha256"), 64):
            reasons.append(f"{row_name}: invalid accepted manifest SHA-256")
        if row.get("training_code_revision") != row.get("git_revision"):
            reasons.append(f"{row_name}: training/eval code revision mismatch")
        if row.get("training_asset_revision") != row.get("asset_git_revision"):
            reasons.append(f"{row_name}: training/eval asset revision mismatch")
        missing_columns = sorted(
            (REQUIRED_SAMPLED_COLUMNS | LEGACY_SAMPLED_COLUMNS) - row.keys()
        )
        if missing_columns:
            reasons.append(
                f"{row_name}: missing sampled columns {missing_columns}"
            )
        else:
            sampled_values = np.asarray(
                [
                    float(row[key])
                    for key in (
                        REQUIRED_SAMPLED_COLUMNS | LEGACY_SAMPLED_COLUMNS
                    )
                ],
                dtype=float,
            )
            if not np.isfinite(sampled_values).all():
                reasons.append(f"{row_name}: sampled columns must be finite")
        reasons.extend(_qvel_sampled_metrics_reasons(row, row_name))
        status_total = sum(
            int(row.get(key, -10_000))
            for key in (
                "first_strike_success_n_sampled",
                "first_strike_window_n_sampled",
                "first_strike_no_contact_n_sampled",
                "first_strike_unfinished_event_n_sampled",
            )
        )
        if status_total != EXPECTED_EPISODES_PER_SEED:
            reasons.append(
                f"{row_name}: first-strike statuses do not sum to 512"
            )
        for prefix in ("git", "asset_git"):
            revision = row.get(f"{prefix}_revision")
            try:
                dirty = _parse_bool(row.get(f"{prefix}_dirty", True))
            except ValueError:
                dirty = True
            if not _is_hex_digest(revision, 40) or dirty:
                reasons.append(
                    f"{row_name}: {prefix} revision is dirty/unknown"
                )
        for identity_key in (
            "checkpoint_sha256",
            "accepted_checkpoint_sha256",
            "campaign_config_sha256",
            "treatment_config_sha256",
            "nail_asset_sha256",
            "sampled_trace_digest",
            "sampled_trace_artifact_sha256",
        ):
            if not _is_hex_digest(row.get(identity_key), 64):
                reasons.append(f"{row_name}: invalid {identity_key}")
        try:
            actual_treatment_digest = _treatment_config_digest(row)
        except Exception as error:
            reasons.append(
                f"{row_name}: cannot derive treatment config identity: {error}"
            )
        else:
            if row.get("treatment_config_sha256") != actual_treatment_digest:
                reasons.append(
                    f"{row_name}: treatment config identity mismatch"
                )
        rng_seeds = [
            int(row.get(key, -1))
            for key in (
                "action_rng_seed",
                "reset_rng_seed",
                "observation_rng_seed",
            )
        ]
        if min(rng_seeds) < 0 or len(set(rng_seeds)) != 3:
            reasons.append(f"{row_name}: evaluator RNG identities are invalid")
        reasons.extend(_sampled_artifact_reasons(row, row_name))
        if int(row.get("impossible_success_n", -1)) != 0:
            reasons.append(f"{row_name}: impossible_success_n is nonzero")
        if int(row.get("lambda_dead_n", -1)) != 0:
            reasons.append(f"{row_name}: lambda_dead_n is nonzero")
    common_reason = {
        "git_revision": "code revision differs across rows",
        "asset_git_revision": "asset revision differs across rows",
        "nail_asset_sha256": "nail asset identity differs across rows",
        "campaign_config_sha256": "campaign config identity differs across rows",
        "accepted_manifest_sha256": "accepted manifest identity differs across rows",
        "training_code_revision": "training code revision differs across rows",
        "training_asset_revision": "training asset revision differs across rows",
        "action_rng_seed": "action RNG identity differs across rows",
        "reset_rng_seed": "reset RNG identity differs across rows",
        "observation_rng_seed": "observation RNG identity differs across rows",
    }
    for field, reason in common_reason.items():
        if len({str(row.get(field)) for row in rows}) != 1:
            reasons.append(f"campaign: {reason}")
    for field, label in (
        ("checkpoint_sha256", "checkpoint"),
        ("sampled_trace_digest", "sampled trace"),
        ("sampled_trace_artifact_sha256", "sampled trace artifact"),
    ):
        identities = [str(row.get(field)) for row in rows]
        if len(set(identities)) != len(identities):
            reasons.append(f"campaign: duplicate {label} identity")
    if set(by_arm) != set(ARM_TASKS):
        reasons.append("campaign must contain exactly C/D-prime/F/E")
    for arm in ARM_ORDER:
        seeds = [int(row.get("training_seed", -1)) for row in by_arm.get(arm, ())]
        if sorted(seeds) != list(EXPECTED_SEEDS):
            reasons.append(f"{arm}: training seeds must be exactly 0..7")
        treatment_identities = {
            str(row.get("treatment_config_sha256"))
            for row in by_arm.get(arm, ())
        }
        if len(treatment_identities) != 1:
            reasons.append(
                f"{arm}: treatment config identity differs across seeds"
            )
    if len(
        {
            str(row.get("treatment_config_sha256"))
            for row in rows
        }
    ) != len(ARM_ORDER):
        reasons.append(
            "campaign: treatment config identities must distinguish all four arms"
        )
    return reasons


def _contrast(
    by_arm: Mapping[str, Sequence[Mapping]],
    treatment: str,
    control: str,
    *,
    metric: str = PRIMARY_METRIC,
) -> dict:
    treatment_values = np.asarray(
        [float(row[metric]) for row in by_arm[treatment]], dtype=float
    )
    control_values = np.asarray(
        [float(row[metric]) for row in by_arm[control]], dtype=float
    )
    tests = exact_seed_tests(treatment_values, control_values)
    return {
        "contrast": [treatment, control],
        "metric": metric,
        "treatment": _arm_summary(treatment_values),
        "control": _arm_summary(control_values),
        "absolute_mean_effect": tests["mean_difference"],
        "relative_mean_effect": (
            tests["mean_difference"] / float(np.mean(control_values))
            if float(np.mean(control_values)) > 0
            else None
        ),
        **tests,
    }


def _hardware_speed_qualification(
    by_arm: Mapping[str, Sequence[Mapping]],
    treatment: str,
    control: str,
) -> dict:
    """Scope descriptive speed-rail evidence to one named contrast.

    Finite exceedances do not remove any episode, seed, or row from the
    simulation reward comparison. They do prevent a hardware-safe reading.
    Absence of observed exceedance is explicitly not hardware certification.
    """

    arms = [treatment, control]
    selected = [
        row for arm in arms for row in by_arm.get(arm, ())
    ]
    observed = any(
        float(row["qvel_finite_exceedance_rate_sampled"]) > 0.0
        for row in selected
    )
    return {
        "arms": arms,
        "finite_rail_exceedance_observed": observed,
        "violating_seed_rows": sum(
            float(row["qvel_finite_exceedance_rate_sampled"]) > 0.0
            for row in selected
        ),
        "max_abs_rad_s": max(
            (float(row["qvel_max_abs_rad_s_sampled"]) for row in selected),
            default=0.0,
        ),
        "max_excess_rad_s": max(
            (float(row["qvel_max_excess_rad_s_sampled"]) for row in selected),
            default=0.0,
        ),
        "samples_over_rail": sum(
            int(row["qvel_samples_over_rail_sampled"]) for row in selected
        ),
        "seconds_over_rail": sum(
            float(row["qvel_seconds_over_rail_sampled"]) for row in selected
        ),
        "simulation_reward_comparison_valid": True,
        "hardware_safe_claim_allowed": False,
        "interpretation": (
            "simulation reward comparison retained; finite speed-rail "
            "exceedance observed, so no hardware-safe claim"
            if observed
            else "no finite speed-rail exceedance observed in this contrast; "
            "this is not hardware certification"
        ),
    }


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    adjusted_sorted = np.empty(p_values.size, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = (p_values.size - rank) * p_values[original_index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(p_values.size, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def analyze_campaign(
    rows: Sequence[Mapping],
    *,
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Analyze the complete 4x8 sampled-evaluator matrix, failing closed."""

    rows = [dict(row) for row in rows]
    reasons = _campaign_validity(rows)
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("treatment") in ARM_TASKS:
            by_arm[str(row["treatment"])].append(row)
    for arm in by_arm:
        by_arm[arm].sort(key=lambda row: int(row["training_seed"]))

    complete = all(len(by_arm.get(arm, ())) == 8 for arm in ARM_ORDER)
    if not complete:
        return {
            "valid": False,
            "invalidation_reasons": reasons or ["incomplete 4x8 matrix"],
            "comparison_identity_contract": COMPARISON_IDENTITY_CONTRACT,
            "primary": {
                "contrast": ["E", "D-prime"],
                "passed": False,
                "gates": {"valid_provenance_and_sentinels": False},
                "hardware_speed_qualification": {
                    "arms": ["E", "D-prime"],
                    "finite_rail_exceedance_observed": False,
                    "simulation_reward_comparison_valid": False,
                    "hardware_safe_claim_allowed": False,
                },
            },
            "mechanism_family": {"evaluated": False, "contrasts": []},
            "descriptive": {"contrast": ["C", "D-prime"]},
        }

    primary = _contrast(by_arm, "E", "D-prime")
    primary["hardware_speed_qualification"] = (
        _hardware_speed_qualification(by_arm, "E", "D-prime")
    )
    success = _success_guardrails(by_arm, "E", "D-prime")
    dprime_mean = primary["control"]["mean"]
    relative_valid = dprime_mean > 0
    relative_pass = (
        relative_valid
        and primary["relative_mean_effect"] is not None
        and primary["relative_mean_effect"] >= 0.10 - 1e-12
    )
    valid = not reasons
    gates = {
        "valid_provenance_and_sentinels": valid,
        "mwu_p_lt_0_05": primary["mwu_two_sided_exact_p"] < 0.05,
        "permutation_p_lt_0_05": (
            primary["mean_difference_two_sided_exact_p"] < 0.05
        ),
        "relative_effect_at_least_10_percent": relative_pass,
        "first_window_success_guardrail": success["first_window"]["passed"],
        "overall_success_guardrail": success["overall"]["passed"],
    }
    rng = np.random.default_rng(bootstrap_seed)
    primary_bootstrap = _bootstrap_difference(
        np.asarray(
            [row[PRIMARY_METRIC] for row in by_arm["E"]], dtype=float
        ),
        np.asarray(
            [row[PRIMARY_METRIC] for row in by_arm["D-prime"]], dtype=float
        ),
        samples=bootstrap_samples,
        rng=rng,
    )
    noninferiority_bounds = {}
    for label, metric in (
        ("first_window", "first_strike_success_rate_sampled"),
        ("overall", "overall_success_rate_sampled"),
    ):
        distribution = _bootstrap_difference(
            np.asarray([row[metric] for row in by_arm["E"]], dtype=float),
            np.asarray([row[metric] for row in by_arm["D-prime"]], dtype=float),
            samples=bootstrap_samples,
            rng=rng,
        )
        noninferiority_bounds[label] = float(np.quantile(distribution, 0.05))
    primary.update(
        {
            "relative_effect_denominator_valid": relative_valid,
            "success_guardrails": success,
            "gates": gates,
            "passed": all(gates.values()),
            "seed_bootstrap": {
                "samples": bootstrap_samples,
                "rng_seed": bootstrap_seed,
                "absolute_effect_95_interval": [
                    float(np.quantile(primary_bootstrap, 0.025)),
                    float(np.quantile(primary_bootstrap, 0.975)),
                ],
                "success_difference_one_sided_95_lower": noninferiority_bounds,
                "success_noninferiority_intersection_union": all(
                    value > -0.05 for value in noninferiority_bounds.values()
                ),
                "inference_label": "seed-level within-arm bootstrap; not exact permutation inference",
            },
        }
    )

    mechanism = {"evaluated": bool(primary["passed"]), "contrasts": []}
    if primary["passed"]:
        contrasts = [
            _contrast(by_arm, "F", "D-prime"),
            _contrast(by_arm, "E", "F"),
        ]
        p_joint = [
            max(
                row["mwu_two_sided_exact_p"],
                row["mean_difference_two_sided_exact_p"],
            )
            for row in contrasts
        ]
        holm = _holm_adjust(p_joint)
        for row, joint, adjusted in zip(contrasts, p_joint, holm, strict=True):
            direction_passed = row["absolute_mean_effect"] > 0
            f_guard = (
                _success_guardrails(by_arm, "F", "D-prime")
                if row["contrast"] == ["F", "D-prime"]
                else None
            )
            success_passed = (
                all(item["passed"] for item in f_guard.values())
                if f_guard is not None
                else True
            )
            row.update(
                {
                    "p_joint": joint,
                    "holm_adjusted_p": adjusted,
                    "holm_rejected_at_0_05": adjusted < 0.05,
                    "hypothesized_direction_passed": direction_passed,
                    "success_guardrails": f_guard,
                    "success_guardrails_passed": success_passed,
                    "interpretable": (
                        adjusted < 0.05 and direction_passed and success_passed
                    ),
                    "hardware_speed_qualification": (
                        _hardware_speed_qualification(
                            by_arm, row["contrast"][0], row["contrast"][1]
                        )
                    ),
                }
            )
        mechanism["contrasts"] = contrasts

    descriptive_full = _contrast(by_arm, "C", "D-prime")
    descriptive = {
        key: value
        for key, value in descriptive_full.items()
        if key
        not in {
            "mwu_two_sided_exact_p",
            "mean_difference_two_sided_exact_p",
            "assignment_count",
            "u_treatment",
            "a12_treatment_over_control",
        }
    }
    descriptive["interpretation"] = (
        "descriptive repeated-credit/censoring/timing package diagnostic only"
    )
    descriptive["hardware_speed_qualification"] = (
        _hardware_speed_qualification(by_arm, "C", "D-prime")
    )
    return {
        "valid": valid,
        "invalidation_reasons": reasons,
        "comparison_identity_contract": COMPARISON_IDENTITY_CONTRACT,
        "inferential_unit": "independently trained policy seed",
        "primary": primary,
        "mechanism_family": mechanism,
        "descriptive": descriptive,
    }


def _command_line() -> int:
    parser = argparse.ArgumentParser(
        description="First-strike campaign analysis support"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser(
        "validate-accepted-manifest",
        help="validate and emit the explicit fsr4x8 checkpoint selection",
    )
    manifest.add_argument("--manifest", required=True)
    manifest.add_argument("--expected-code-revision", required=True)
    manifest.add_argument("--expected-asset-revision", required=True)
    args = parser.parse_args()
    if args.command != "validate-accepted-manifest":
        parser.error("unsupported command")
    try:
        rows = load_accepted_attempt_manifest(
            args.manifest,
            expected_code_revision=args.expected_code_revision,
            expected_asset_revision=args.expected_asset_revision,
        )
    except Exception as error:
        print(f"MANIFEST_FAIL: {error}", file=sys.stderr)
        return 2
    manifest_sha256 = _sha256(args.manifest)
    for row in rows:
        slug = CAMPAIGN_ARM_SEMANTICS[str(row["arm"])][1]
        print(
            "\t".join(
                (
                    manifest_sha256,
                    slug,
                    str(row["training_seed"]),
                    str(row["task"]),
                    str(row["run_name"]),
                    str(row["checkpoint_path"]),
                    str(row["checkpoint_sha256"]),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_command_line())
