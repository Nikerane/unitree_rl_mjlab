from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import evaluation.analysis.first_strike_campaign as legacy
import evaluation.analysis.first_strike_quality_campaign as quality_analysis
import evaluation.analysis.plot_first_strike_quality_campaign as quality_plot
from evaluation.analysis import fq4x8_manifests as campaign_manifests
from evaluation.analysis.first_strike_quality_campaign import (
    FROZEN_QUALITY_CAMPAIGN_MATRIX,
    _d0_mechanism,
    _fq_min_practical_acceptance,
    aggregate_quality_episode_metrics,
    analyze_quality_campaign,
    analyze_quality_episode,
    exact_paired_sign_flip,
    holm_adjust,
    paired_bootstrap_ratio,
    validate_quality_campaign_contract,
)
from evaluation.analysis.plot_first_strike_quality_campaign import (
    build_aggregate_nail_plane_contact_map,
    render_quality_figures,
)


GEOMETRY = {
    "nail_axis": [0.0, 0.0, -1.0],
    "nail_xy_m": [0.5, 0.0],
    "nail_radius_m": 0.012,
    "source_sha256": "c" * 64,
}
SENTINELS = (
    "quality_overflow_n",
    "quality_nonfinite_n",
    "liveness_failure_n",
    "impossible_success_n",
    "lambda_dead_n",
    "quota_failure_n",
)
EXPECTED_REWARD_HASHES = {
    "F8": "47d993698852dc939c753d978e41c9124e24c470e0595439f004d2b561d76fb9",
    "F0": "a8fdd61dc521a6a4294945d94e52dde8fde6560ee963976c08251e0996d42f9c",
    "D0": "c2c8f069f5f744eb063514b0c4a20e75ed9b271b382f2397281c04e7ac12f325",
    "FQ-min": "5bdd740a32cb730e1e63392422387dff5df2c8812d52587060249fe2b54a09a7",
}
WRONG_READER_HASHES = {
    "F8": "48fb66eba010c011a5c701a2e81d901a3ed4e201635c48ad18cede9d3de28deb",
    "F0": "4902ea7e7b56ea1dc1765a29939db4b08f27c5f8f6d07af155604a05c826a474",
    "D0": "35985aeb3420b7c58a38fcf850be655690564703478427b603645d50505d46ac",
    "FQ-min": "9a482d9729b59e99d85b6dccd8c3aa9d909a28c4bf19eec64e5138625b024b37",
}
FQ_WRONG_NORMALIZER_HASH = (
    "35985aeb3420b7c58a38fcf850be655690564703478427b603645d50505d46ac"
)
QUALITY_SENSOR_SLOT_COUNT = 8


def _load_contact_quality_reference():
    source = (
        Path(__file__).parents[1]
        / "src/tasks/hammer/mdp/contact_quality.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_contact_quality_reference", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.contact_point_quality


CONTACT_QUALITY_REFERENCE = _load_contact_quality_reference()


def _bank_producer_quality(trace: dict) -> None:
    """Replace fixture latches with the actual Torch float32 producer values."""

    onset = trace["first_strike"]["accepted_onset_index"]
    assert onset is not None
    physical = trace["physical"]
    found = torch.tensor(
        physical["quality_found_count"][onset], dtype=torch.float32
    ).unsqueeze(0)
    force = torch.zeros(
        1, QUALITY_SENSOR_SLOT_COUNT, 3, dtype=torch.float32
    )
    force[0, :, 0] = torch.tensor(
        physical["quality_normal_force_n"][onset], dtype=torch.float32
    )
    position = torch.tensor(
        physical["quality_contact_position_m"][onset], dtype=torch.float32
    ).unsqueeze(0)
    normal = torch.tensor(
        physical["quality_contact_normal"][onset], dtype=torch.float32
    ).unsqueeze(0)
    axis = torch.tensor(GEOMETRY["nail_axis"], dtype=torch.float32)
    nail_top = torch.tensor(
        [[*GEOMETRY["nail_xy_m"], 0.0]], dtype=torch.float32
    )
    point, error, contact_quality, valid, overflow = (
        CONTACT_QUALITY_REFERENCE(
            found=found,
            force_contact=force,
            position_w=position,
            nail_top_w=nail_top,
            nail_axis_w=axis,
            nail_radius_m=GEOMETRY["nail_radius_m"],
            num_slots=QUALITY_SENSOR_SLOT_COUNT,
        )
    )
    weights = torch.where(
        found > 0.0,
        force[..., 0].clamp_min(0.0),
        torch.zeros_like(found),
    )
    weighted_normal = (normal * weights.unsqueeze(-1)).sum(dim=1)
    normal_norm = torch.linalg.vector_norm(
        weighted_normal, dim=-1, keepdim=True
    )
    unit_normal = weighted_normal / torch.where(
        normal_norm > 0.0,
        normal_norm,
        torch.ones_like(normal_norm),
    )
    axiality = (unit_normal * axis).sum(dim=-1).clamp(0.0, 1.0)
    axiality = torch.where(
        valid & (normal_norm.squeeze(-1) > 0.0),
        axiality,
        torch.zeros_like(axiality),
    )
    snapshot = {
        "contact_point_w": point[0].tolist(),
        "contact_error_m": float(error[0]),
        "contact_quality": float(contact_quality[0]),
        "contact_quality_valid": bool(valid[0]),
        "contact_quality_overflow": bool(overflow[0]),
        "contact_normal_axiality": float(axiality[0]),
    }
    trace["first_strike"].update(snapshot)
    event = trace["event_trace"]
    first_time = float(
        np.float32(trace["first_strike"]["first_contact_time_s"])
    )
    trace["first_strike"]["first_contact_time_s"] = first_time
    event["tracker_first_contact_time_s"][onset:] = [
        first_time for _ in event["tracker_first_contact_time_s"][onset:]
    ]
    for event_key, snapshot_key in (
        ("tracker_contact_point_w", "contact_point_w"),
        ("tracker_contact_error_m", "contact_error_m"),
        ("tracker_contact_quality", "contact_quality"),
        ("tracker_contact_quality_valid", "contact_quality_valid"),
        ("tracker_contact_quality_overflow", "contact_quality_overflow"),
        ("tracker_contact_normal_axiality", "contact_normal_axiality"),
    ):
        event[event_key][onset:] = [
            snapshot[snapshot_key]
            for _ in event[event_key][onset:]
        ]


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _trace(
    *,
    label: str = "F8",
    raw_treatment: str = "F8",
    task: str = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    seed: int = 8,
    env_id: int = 0,
    ordinal: int = 0,
    quality: float = 0.55,
    useful_speed: float = 1.0,
    delivered: float = 0.10,
    dwell_two_substeps: bool = True,
) -> dict:
    del label, seed
    error = 0.012 * float(np.sqrt(1.0 - quality))
    point = [0.5 + error, 0.0, 0.1]
    contact = [False, True, dwell_two_substeps]
    force = [0.0, 10.0, 8.0 if dwell_two_substeps else 0.0]
    quality_normal = [float(np.sqrt(1.0 - 0.9**2)), 0.0, -0.9]
    zero_vector = [0.0, 0.0, 0.0]
    raw_position_first = [
        zero_vector,
        point,
        point if dwell_two_substeps else zero_vector,
    ]
    raw_normal_first = [
        zero_vector,
        quality_normal,
        quality_normal if dwell_two_substeps else zero_vector,
    ]
    start = [point[0] + 0.002, -0.001, 0.104]
    qvel = [[3.0] * 6 for _ in contact]
    trace = {
        "episode_id": f"{raw_treatment}-env{env_id}-episode{ordinal}",
        "env_id": env_id,
        "episode_ordinal": ordinal,
        "arm": raw_treatment,
        "task": task,
        "physics_dt_s": 0.002,
        "phase_contract": {
            "physical_channels": "pre_integration",
            "tracker_derived_channels": "pre_integration",
            "tracker_nail_depth": "post_integration",
            "tracker_depth_lead_substeps": 1,
        },
        "nail_geometry": copy.deepcopy(GEOMETRY),
        "physical": {
            "contact": contact,
            "head_position_m": [start, point, point],
            "clamped_depth_m": [0.0, 0.0, 0.005],
            "net_axial_force_n": force,
            "joint_speed_rad_s": qvel,
            "post_step_joint_speed_rad_s": copy.deepcopy(qvel),
            "tracker_depth_post_integration_m": [0.0, 0.0, 0.005],
            "quality_found_count": [
                [int(value)] + [0] * (QUALITY_SENSOR_SLOT_COUNT - 1)
                for value in contact
            ],
            "quality_normal_force_n": [
                [value] + [0.0] * (QUALITY_SENSOR_SLOT_COUNT - 1)
                for value in force
            ],
            "quality_contact_position_m": [
                [list(value)]
                + [
                    [0.0, 0.0, 0.0]
                    for _ in range(QUALITY_SENSOR_SLOT_COUNT - 1)
                ]
                for value in raw_position_first
            ],
            "quality_contact_normal": [
                [list(value)]
                + [
                    [0.0, 0.0, 0.0]
                    for _ in range(QUALITY_SENSOR_SLOT_COUNT - 1)
                ]
                for value in raw_normal_first
            ],
        },
        "event_trace": {
            "tracker_started": [False, True, True],
            "tracker_finalized": [False, False, True],
            "tracker_productive": [False, False, True],
            "tracker_reason": [0, 0, 1],
            "event_cumulative_impulse_n_s": [0.0, delivered / 2.0, delivered],
            "tracker_contact_point_w": [[0.0, 0.0, 0.0], point, point],
            "tracker_contact_error_m": [0.0, error, error],
            "tracker_contact_quality": [0.0, quality, quality],
            "tracker_contact_quality_valid": [False, True, True],
            "tracker_contact_quality_overflow": [False, False, False],
            "tracker_first_contact_time_s": [0.0, 0.004, 0.004],
            "tracker_contact_normal_axiality": [0.0, 0.9, 0.9],
            "event_cumulative_transverse_impulse_n_s": [0.0, 0.001, 0.002],
        },
        "payout_semantics": {
            "F": "actual_event_linear",
            "F8": "actual_event_linear",
            "F0": "actual_event_linear_speed_disabled",
            "D0": "actual_event_linear_delivered_disabled",
            "FQ": "actual_event_quality_bounded",
        }[raw_treatment],
        "payout_is_counterfactual": False,
        "first_strike": {
            "started": True,
            "finalized": True,
            "reason": "success",
            "accepted_onset_index": 1,
            "productive": True,
            "v_precontact_m_s": useful_speed,
            "delivered_n_s": delivered,
            "delivered_transverse_n_s": 0.002,
            "contact_point_w": point,
            "contact_error_m": error,
            "contact_quality": quality,
            "contact_quality_valid": True,
            "contact_quality_overflow": False,
            "first_contact_time_s": 0.004,
            "contact_normal_axiality": 0.9,
            "saturated": False,
        },
        "overall_success": True,
        "episode_peak_lambda": [0.10, 0.20, 0.10, 0.10, 0.10, 0.10],
        "episode_delivered_accumulator_n_s": delivered,
        "episode_depth_m": 0.005,
        "reward": {
            "gamma": 0.99,
            "impact_payout": [0.0],
            "delivered_payout": [0.0],
        },
    }
    _bank_producer_quality(trace)
    trace["trace_digest"] = legacy._episode_trace_digest(
        trace, schema_version=3
    )
    return trace


def _no_contact_trace() -> dict:
    trace = _trace()
    n = 3
    trace["physical"]["contact"] = [False] * n
    trace["physical"]["net_axial_force_n"] = [0.0] * n
    trace["physical"]["quality_found_count"] = [
        [0] * QUALITY_SENSOR_SLOT_COUNT for _ in range(n)
    ]
    trace["physical"]["quality_normal_force_n"] = [
        [0.0] * QUALITY_SENSOR_SLOT_COUNT for _ in range(n)
    ]
    trace["physical"]["quality_contact_position_m"] = [
        [[0.0, 0.0, 0.0] for _ in range(QUALITY_SENSOR_SLOT_COUNT)]
        for _ in range(n)
    ]
    trace["physical"]["quality_contact_normal"] = [
        [[0.0, 0.0, 0.0] for _ in range(QUALITY_SENSOR_SLOT_COUNT)]
        for _ in range(n)
    ]
    trace["event_trace"].update(
        {
            "tracker_started": [False] * n,
            "tracker_finalized": [False] * n,
            "tracker_productive": [False] * n,
            "tracker_reason": [0] * n,
            "event_cumulative_impulse_n_s": [0.0] * n,
            "tracker_contact_point_w": [[0.0, 0.0, 0.0]] * n,
            "tracker_contact_error_m": [0.0] * n,
            "tracker_contact_quality": [0.0] * n,
            "tracker_contact_quality_valid": [False] * n,
            "tracker_contact_quality_overflow": [False] * n,
            "tracker_first_contact_time_s": [0.0] * n,
            "tracker_contact_normal_axiality": [0.0] * n,
            "event_cumulative_transverse_impulse_n_s": [0.0] * n,
        }
    )
    trace["first_strike"].update(
        {
            "started": False,
            "finalized": False,
            "reason": "none",
            "accepted_onset_index": None,
            "productive": False,
            "v_precontact_m_s": 0.0,
            "delivered_n_s": 0.0,
            "delivered_transverse_n_s": 0.0,
            "contact_point_w": [0.0, 0.0, 0.0],
            "contact_error_m": 0.0,
            "contact_quality": 0.0,
            "contact_quality_valid": False,
            "contact_quality_overflow": False,
            "first_contact_time_s": 0.0,
            "contact_normal_axiality": 0.0,
        }
    )
    trace["overall_success"] = False
    trace["episode_peak_lambda"] = [0.0] * 6
    trace["episode_delivered_accumulator_n_s"] = 0.0
    return trace


def _resize_raw_quality_slots(trace: dict, width: int) -> None:
    """Keep the first raw slot while changing the instrument signature."""

    physical = trace["physical"]
    for key in ("quality_found_count", "quality_normal_force_n"):
        zero = 0 if key == "quality_found_count" else 0.0
        physical[key] = [
            [row[0]] + [zero] * (width - 1) for row in physical[key]
        ]
    for key in ("quality_contact_position_m", "quality_contact_normal"):
        physical[key] = [
            [list(row[0])]
            + [[0.0, 0.0, 0.0] for _ in range(width - 1)]
            for row in physical[key]
        ]


def _treatment_digest(row: dict, payout_semantics: str) -> str:
    return _digest(
        {
            "schema_version": 1,
            "task": row["task"],
            "treatment": row["raw_treatment"],
            "payout_semantics": payout_semantics,
            "impact_weight": row["impact_weight"],
            "delivered_weight": row["delivered_weight"],
            "event_i_ref_n_s": 0.3088,
        }
    )


def _campaign_rows(tmp_path: Path) -> tuple[list[dict], list[dict]]:
    rows, manifest = [], []
    for frozen in FROZEN_QUALITY_CAMPAIGN_MATRIX:
        frozen = copy.deepcopy(frozen)
        label = frozen["treatment"]
        raw = frozen["raw_treatment"]
        seed = frozen["training_seed"]
        identity = f"{frozen['short']}-seed{seed}"
        checkpoint_sha = hashlib.sha256(f"checkpoint-{identity}".encode()).hexdigest()
        treatment_sha = _treatment_digest(
            frozen,
            {
                "F": "actual_event_linear",
                "F0": "actual_event_linear_speed_disabled",
                "D0": "actual_event_linear_delivered_disabled",
                "FQ": "actual_event_quality_bounded",
            }[raw],
        )
        row = {
            "name": identity,
            "task": frozen["task"],
            "treatment": raw,
            "training_seed": seed,
            "checkpoint_path": f"/accepted/{identity}/model_499.pt",
            "checkpoint_sha256": checkpoint_sha,
            "accepted_checkpoint_sha256": checkpoint_sha,
            "num_envs": 256,
            "nsteps": 400,
            "episode_len_s": 4.0,
            "seed": 2_026_072_900,
            "imp_max_p": 0.0,
            "impact_weight": frozen["impact_weight"],
            "delivered_weight": frozen["delivered_weight"],
            "event_i_ref_n_s": 0.3088,
            "n_episodes_sampled": 512,
            "episodes_per_env_sampled": 2,
            "action_rng_seed": 2_056_072_941,
            "reset_rng_seed": 2_036_072_919,
            "observation_rng_seed": 2_046_072_933,
            "reset_position_noise_min_rad": -0.05,
            "reset_position_noise_max_rad": 0.05,
            "actor_observation_corruption": True,
            "critic_observation_corruption": False,
            "sampled_completion_rule": "first_two_completions_per_environment",
            "sampled_actions_stochastic": True,
            "physics_dt_s": 0.002,
            "control_decimation": 10,
            "fixed_impedance_signature_sha256": frozen[
                "fixed_impedance_signature_sha256"
            ],
            "fixed_action_signature_sha256": frozen[
                "fixed_action_signature_sha256"
            ],
            "campaign_config_sha256": hashlib.sha256(
                f"campaign-{label}".encode()
            ).hexdigest(),
            "treatment_config_sha256": treatment_sha,
            "nail_asset_sha256": "c" * 64,
            "accepted_manifest_sha256": "e" * 64,
            "training_code_revision": "a" * 40,
            "training_asset_revision": "b" * 40,
            "git_revision": "a" * 40,
            "git_dirty": False,
            "asset_git_revision": "b" * 40,
            "asset_git_dirty": False,
            "impossible_success_n": 0,
            "lambda_dead_n": 0,
            "qvel_nonfinite_rate_sampled": 0.0,
        }
        training_config = hashlib.sha256(f"train-{label}".encode()).hexdigest()
        evaluation_config = hashlib.sha256(f"eval-{label}".encode()).hexdigest()
        accepted = {
            "campaign": "fq4x8",
            "disposition": "accepted",
            "arm": label,
            "short": frozen["short"],
            "task": frozen["task"],
            "training_seed": seed,
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": checkpoint_sha,
            "code_revision": row["training_code_revision"],
            "asset_revision": row["training_asset_revision"],
            "campaign_config_sha256": hashlib.sha256(
                b"training-campaign"
            ).hexdigest(),
            "treatment_config_sha256": (
                campaign_manifests.EXPECTED_TREATMENT_CONFIG_SHA256[label]
            ),
            "fixed_action_signature_sha256": frozen[
                "fixed_action_signature_sha256"
            ],
            "fixed_impedance_signature_sha256": frozen[
                "fixed_impedance_signature_sha256"
            ],
            "cap_signature_sha256": (
                campaign_manifests.EXPECTED_CAP_SIGNATURE_SHA256
            ),
            "clean_state": True,
            "accepted_training_manifest_sha256": "e" * 64,
            "evaluation_attempt": "attempt2",
            "evaluation_retry_history": (
                "attempt1:infra_failure;attempt2:accepted"
            ),
            "training_config_sha256": training_config,
            "evaluation_config_sha256": evaluation_config,
            "training_policy_observation_sha256": "1" * 64,
            "evaluation_policy_observation_sha256": "1" * 64,
            "training_treatment_reward_sha256": EXPECTED_REWARD_HASHES[label],
            "evaluation_treatment_reward_sha256": EXPECTED_REWARD_HASHES[label],
            "action_rng_seed": row["action_rng_seed"],
            "reset_rng_seed": row["reset_rng_seed"],
            "observation_rng_seed": row["observation_rng_seed"],
            "sampled_trace_digest": "2" * 64,
            "sampled_trace_artifact_sha256": "3" * 64,
            "num_envs": 256,
            "episodes_per_env_sampled": 2,
            "n_episodes_sampled": 512,
            **{sentinel: 0 for sentinel in SENTINELS},
        }
        rows.append(row)
        manifest.append(accepted)
    return rows, manifest


def _write_artifact(path: Path, row: dict, accepted: dict) -> None:
    label = accepted["arm"]
    raw = row["treatment"]
    seed = accepted["training_seed"]
    offset = 0.005 * (seed - 8)
    quality = {"F8": 0.55, "F0": 0.70, "D0": 0.60, "FQ-min": 0.75}[label] + offset
    speed = {"F8": 1.0, "F0": 0.96, "D0": 1.0, "FQ-min": 0.98}[label]
    delivered = {"F8": 0.10, "F0": 0.095, "D0": 0.08, "FQ-min": 0.08}[label]
    episodes = [
        _trace(
            label=label,
            raw_treatment=raw,
            task=row["task"],
            seed=seed,
            env_id=env_id,
            ordinal=ordinal,
            quality=quality,
            useful_speed=speed,
            delivered=delivered,
            dwell_two_substeps=label != "D0",
        )
        for ordinal in range(2)
        for env_id in range(256)
    ]
    legacy_metrics = [
        legacy.summarize_episode(trace, nail_geometry=GEOMETRY)
        for trace in episodes
    ]
    row.update(
        legacy.aggregate_episode_metrics(
            legacy_metrics, expected_episode_count=512
        )
    )
    peaks = np.asarray([trace["episode_peak_lambda"] for trace in episodes])
    delivered_values = np.asarray(
        [trace["episode_delivered_accumulator_n_s"] for trace in episodes]
    )
    row.update(
        {
            "success_rate_sampled": 1.0,
            "worst_ratio_max_sampled": float(
                np.max(peaks / np.asarray(quality_analysis.IMPULSE_LIMITS))
            ),
            "delivered_mean_sampled": float(delivered_values.mean()),
        }
    )
    observation_sha = "1" * 64
    reward_sha = EXPECTED_REWARD_HASHES[label]
    payload = {
        "schema_version": 3,
        "selection": "first two completed episodes from each of 256 environments",
        "expected_episode_count": 512,
        "treatment": raw,
        "task": row["task"],
        "training_seed": seed,
        "weights": {
            "impact_progress": row["impact_weight"],
            "delivered_impulse": row["delivered_weight"],
        },
        "event_i_ref_n_s": 0.3088,
        "impulse_limits_n_m_s": list(quality_analysis.IMPULSE_LIMITS),
        "imp_max_p": 0.0,
        "rng_streams": {
            "reset": row["reset_rng_seed"],
            "observation": row["observation_rng_seed"],
            "action": row["action_rng_seed"],
        },
        "evaluation_contract": {
            "base_rng_seed": row["seed"],
            "num_envs": 256,
            "episodes_per_env": 2,
            "episode_len_s": 4.0,
            "mean_nsteps": 400,
            "completion_rule": "first_two_completions_per_environment",
            "stochastic_actions": True,
            "reset_position_noise_rad": [-0.05, 0.05],
            "actor_observation_corruption": True,
            "critic_observation_corruption": False,
            "physics_dt_s": 0.002,
            "control_decimation": 10,
            "fixed_impedance_signature_sha256": accepted[
                "fixed_impedance_signature_sha256"
            ],
            "fixed_action_signature_sha256": accepted[
                "fixed_action_signature_sha256"
            ],
            "strict_config_identities": {
                "training_config_sha256": accepted["training_config_sha256"],
                "evaluation_config_sha256": accepted["evaluation_config_sha256"],
                "training_policy_observation_sha256": observation_sha,
                "evaluation_policy_observation_sha256": observation_sha,
                "training_treatment_reward_sha256": reward_sha,
                "evaluation_treatment_reward_sha256": reward_sha,
            },
        },
        "nail_geometry": copy.deepcopy(GEOMETRY),
        "provenance": {
            "code_git": {"revision": "a" * 40, "dirty": False, "status": ""},
            "asset_git": {"revision": "b" * 40, "dirty": False, "status": ""},
            "checkpoint_sha256": row["checkpoint_sha256"],
            "accepted_checkpoint_sha256": row["accepted_checkpoint_sha256"],
            "campaign_config_sha256": row["campaign_config_sha256"],
            "treatment_config_sha256": row["treatment_config_sha256"],
            "nail_asset_sha256": row["nail_asset_sha256"],
            "accepted_manifest_sha256": row["accepted_manifest_sha256"],
            "training_code_revision": row["training_code_revision"],
            "training_asset_revision": row["training_asset_revision"],
        },
        "mean_rollout_invariants": {
            "impossible_success_n": 0,
            "lambda_dead_n": 0,
        },
        "episodes": episodes,
    }
    payload["payload_digest"] = _digest(payload)
    np.savez_compressed(
        path,
        payload_json=np.asarray(
            json.dumps(payload, sort_keys=True, allow_nan=False), dtype=np.str_
        ),
    )
    row["sampled_trace_path"] = str(path)
    row["sampled_trace_digest"] = payload["payload_digest"]
    row["sampled_trace_artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    accepted["sampled_trace_digest"] = row["sampled_trace_digest"]
    accepted["sampled_trace_artifact_sha256"] = row[
        "sampled_trace_artifact_sha256"
    ]


@pytest.fixture(scope="module")
def complete_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("lean-quality-campaign")
    rows, manifest = _campaign_rows(root)
    for row, accepted in zip(rows, manifest, strict=True):
        _write_artifact(root / f"{row['name']}.npz", row, accepted)
    return rows, manifest


@pytest.fixture(scope="module")
def analysis(complete_campaign):
    return analyze_quality_campaign(*complete_campaign)


def test_exact_sign_flip_sensitivity_and_ties() -> None:
    control = np.arange(8, dtype=float)
    result = exact_paired_sign_flip(control + np.arange(1, 9), control)
    assert result["two_sided_exact_p"] == pytest.approx(2 / 256)
    tied = exact_paired_sign_flip(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 1.0, 3.0])
    )
    assert tied["zero_difference_count"] == 2
    assert tied["two_sided_exact_p"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="matching"):
        exact_paired_sign_flip(np.ones(7), np.ones(8))


def test_holm_mapping_and_paired_pcg64_bootstrap() -> None:
    assert holm_adjust({"late": 0.04, "small": 0.01, "middle": 0.03}) == pytest.approx(
        {"late": 0.06, "small": 0.03, "middle": 0.06}
    )
    control = np.arange(1.0, 9.0)
    first = paired_bootstrap_ratio(1.1 * control, control)
    assert first == paired_bootstrap_ratio(1.1 * control, control)
    assert first["samples"] == 100_000
    assert first["one_sided_95_lower"] == pytest.approx(1.1)
    with pytest.raises(ValueError, match="positive denominator"):
        paired_bootstrap_ratio(np.ones(8), np.zeros(8), samples=100)
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_ratio(np.ones(8), np.full(8, np.nan), samples=100)


def test_ratio_bootstrap_accepts_one_zero_control_seed() -> None:
    treatment = np.asarray([0.4, 1.2, 1.6, 2.1, 2.8, 3.4, 4.0, 4.7])
    control = np.asarray([0.0, 1.0, 1.4, 1.9, 2.5, 3.0, 3.7, 4.2])
    samples = 4_096
    seed = 8_130_001

    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, treatment.size, size=(samples, treatment.size))
    control_means = control[indices].mean(axis=1)
    assert np.isfinite(control_means).all()
    assert np.all(control_means > 0.0)
    expected = {
        "estimate": 1.1412429378531073,
        "one_sided_95_lower": 1.1091703056768558,
        "two_sided_95_interval": [
            1.105827347159464,
            1.236170668953688,
        ],
        "samples": samples,
        "seed": seed,
    }

    assert paired_bootstrap_ratio(
        treatment, control, samples=samples, seed=seed
    ) == expected


def test_ratio_bootstrap_rejects_a_zero_resampled_control_mean() -> None:
    treatment = np.ones(4)
    control = np.asarray([0.0, 0.0, 0.0, 1.0])
    samples = 32
    seed = 7

    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, control.size, size=(samples, control.size))
    resampled_control_means = control[indices].mean(axis=1)
    assert control.mean() > 0.0
    assert np.any(resampled_control_means == 0.0)

    with pytest.raises(ValueError, match="positive denominator"):
        paired_bootstrap_ratio(
            treatment, control, samples=samples, seed=seed
        )


def test_corrected_task9_ratios_preserve_registered_decisions() -> None:
    vectors = {
        "F8": {
            "first_contact_quality_sampled": [
                0.5036742347292602,
                0.3831014957977459,
                0.5050828743260354,
                0.41221152362413704,
                0.0,
                0.5379423153353855,
                0.40998203528579324,
                0.4610586961498484,
            ],
            "first_window_useful_speed_mean_sampled": [
                1.6279039273504168,
                1.584909035358578,
                1.5962672247551382,
                1.5793949563521892,
                0.0,
                1.6237372122704983,
                1.5807533340994269,
                1.6060538250021636,
            ],
            "first_window_success_rate_sampled": [
                0.994140625,
                0.998046875,
                0.994140625,
                0.994140625,
                0.0,
                0.998046875,
                0.990234375,
                0.98828125,
            ],
            "overall_success_rate_sampled": [
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
            ],
            "event_window_depth_gain_mean_sampled": [
                0.029733949567344098,
                0.029844579181911968,
                0.029690979335413203,
                0.029862657856995156,
                0.0,
                0.029970114365028167,
                0.029726006140066374,
                0.02987526327103307,
            ],
            "contact_dwell_mean_ms_sampled": [
                14.48828125,
                13.79296875,
                13.15234375,
                13.67578125,
                0.0,
                14.8515625,
                13.78125,
                14.01171875,
            ],
            "recontact_count_mean_sampled": [
                0.140625,
                0.240234375,
                0.16015625,
                0.14453125,
                0.0,
                0.08203125,
                0.23828125,
                0.138671875,
            ],
            "raw_delivered_impulse_mean_n_s_sampled": [
                0.3025186459708493,
                0.3034256250830367,
                0.29835992321022786,
                0.30739123668172397,
                0.0,
                0.3163799072790425,
                0.30204335879534483,
                0.309845700598089,
            ],
        },
        "FQ-min": {
            "first_contact_quality_sampled": [
                0.954254379728809,
                0.8532140729948878,
                0.8855333261890337,
                0.7496621025493369,
                0.15213712793774903,
                0.9543937215348706,
                0.9223095137858763,
                0.8355440971208736,
            ],
            "first_window_useful_speed_mean_sampled": [
                1.4014848785009235,
                1.443736095679924,
                1.50254652579315,
                1.4121356598334387,
                1.2037739223451354,
                1.5175650981254876,
                1.4717686141375452,
                1.4753968361765146,
            ],
            "first_window_success_rate_sampled": [
                0.994140625,
                0.990234375,
                0.98828125,
                0.98828125,
                0.705078125,
                0.99609375,
                0.998046875,
                0.986328125,
            ],
            "overall_success_rate_sampled": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "event_window_depth_gain_mean_sampled": [
                0.029827237762788172,
                0.03020755206446779,
                0.029846469138760767,
                0.029987172412432983,
                0.02616081258917724,
                0.029956888594540487,
                0.029796309249718433,
                0.0301492365881586,
            ],
        },
        "D0": {
            "event_window_depth_gain_mean_sampled": [
                0.029605363045220656,
                0.02972545047902031,
                0.029484915292186997,
                0.029498320551056167,
                0.0,
                0.029590381906018592,
                0.029597308516201792,
                0.029419827200541704,
            ],
            "contact_dwell_mean_ms_sampled": [
                13.17578125,
                13.33984375,
                12.0625,
                12.8515625,
                0.0,
                12.12890625,
                13.36328125,
                13.1171875,
            ],
            "recontact_count_mean_sampled": [
                0.16015625,
                0.158203125,
                0.115234375,
                0.130859375,
                0.0,
                0.12109375,
                0.216796875,
                0.126953125,
            ],
            "raw_delivered_impulse_mean_n_s_sampled": [
                0.29467341967392713,
                0.30201914574718103,
                0.2890758275461849,
                0.28927960895816796,
                0.0,
                0.28880848304834217,
                0.2943382180383196,
                0.2978941454348387,
            ],
            "first_window_success_rate_sampled": [
                0.9921875,
                0.982421875,
                0.9921875,
                0.986328125,
                0.0,
                0.998046875,
                0.990234375,
                0.94921875,
            ],
            "overall_success_rate_sampled": [
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
            ],
        },
    }
    rows = {
        label: [
            {
                "training_seed": seed,
                **{key: values[index] for key, values in metrics.items()},
            }
            for index, seed in enumerate(range(8, 16))
        ]
        for label, metrics in vectors.items()
    }

    fq_speed = paired_bootstrap_ratio(
        vectors["FQ-min"]["first_window_useful_speed_mean_sampled"],
        vectors["F8"]["first_window_useful_speed_mean_sampled"],
    )
    assert fq_speed["estimate"] == pytest.approx(1.020483, abs=1e-6)
    assert fq_speed["one_sided_95_lower"] == pytest.approx(0.903143, abs=1e-6)
    assert fq_speed["one_sided_95_lower"] <= 0.95

    fq_depth = paired_bootstrap_ratio(
        vectors["FQ-min"]["event_window_depth_gain_mean_sampled"],
        vectors["F8"]["event_window_depth_gain_mean_sampled"],
    )
    assert fq_depth["estimate"] == pytest.approx(1.130463, abs=1e-6)
    assert fq_depth["one_sided_95_lower"] == pytest.approx(1.003629, abs=1e-6)
    assert fq_depth["one_sided_95_lower"] > 0.90

    d0_depth = paired_bootstrap_ratio(
        vectors["D0"]["event_window_depth_gain_mean_sampled"],
        vectors["F8"]["event_window_depth_gain_mean_sampled"],
    )
    assert d0_depth["estimate"] == pytest.approx(0.991462, abs=1e-6)
    assert d0_depth["one_sided_95_lower"] == pytest.approx(0.988661, abs=1e-6)
    assert d0_depth["one_sided_95_lower"] > 0.90

    assert _fq_min_practical_acceptance(
        rows, safety_gates_pass=True
    )["passed"] is False
    assert _d0_mechanism(rows)["claim_allowed"] is False


def test_episode_metrics_cover_quality_motion_mechanism_qvel_and_lambda() -> None:
    trace = _trace()
    trace["physical"]["joint_speed_rad_s"] = [[3.2] * 6] * 3
    trace["physical"]["post_step_joint_speed_rad_s"] = [[3.2] * 6] * 3
    metrics = analyze_quality_episode(trace, nail_geometry=GEOMETRY)
    error = 0.012 * np.sqrt(0.45)
    assert metrics["first_contact_quality_sampled"] == pytest.approx(0.55)
    assert metrics["first_contact_radial_error_m_sampled"] == pytest.approx(error)
    assert metrics["contact_plane_x_m_sampled"] == pytest.approx(error)
    assert metrics["contact_plane_y_m_sampled"] == pytest.approx(0.0)
    assert metrics["onset_axial_speed_m_s_sampled"] == pytest.approx(2.0)
    assert metrics["onset_lateral_speed_m_s_sampled"] == pytest.approx(
        np.sqrt(1.25)
    )
    assert metrics["contact_normal_axiality_sampled"] == pytest.approx(0.9)
    assert metrics["approach_angle_deg_sampled"] == pytest.approx(
        np.degrees(np.arctan2(np.sqrt(1.25), 2.0))
    )
    assert metrics["first_window_useful_speed_m_s_sampled"] == pytest.approx(1.0)
    assert metrics["event_window_depth_gain_m_sampled"] == pytest.approx(0.005)
    assert metrics["contact_dwell_ms_sampled"] == pytest.approx(4.0)
    assert metrics["recontact_count_sampled"] == 0
    assert metrics["raw_delivered_impulse_n_s_sampled"] == pytest.approx(0.1)
    assert metrics["peak_qvel_rad_s_sampled"] == pytest.approx(3.2)
    assert metrics["qvel_rail_exceeded_sampled"] is True
    assert metrics["lambda_per_joint_n_m_s_sampled"] == pytest.approx(
        [0.1, 0.2, 0.1, 0.1, 0.1, 0.1]
    )
    assert metrics["worst_lambda_cap_ratio_sampled"] == pytest.approx(0.2 / 3.28)


def test_event_depth_gain_uses_peak_inside_window_not_final_depth() -> None:
    trace = _trace()
    for key in (
        "contact",
        "head_position_m",
        "clamped_depth_m",
        "net_axial_force_n",
        "joint_speed_rad_s",
        "post_step_joint_speed_rad_s",
        "tracker_depth_post_integration_m",
        "quality_found_count",
        "quality_normal_force_n",
        "quality_contact_position_m",
        "quality_contact_normal",
    ):
        trace["physical"][key].append(copy.deepcopy(trace["physical"][key][-1]))
    trace["physical"]["tracker_depth_post_integration_m"] = [
        0.0,
        0.0,
        0.005,
        0.003,
    ]
    trace["physical"]["clamped_depth_m"] = [0.0, 0.0, 0.005, 0.003]
    for key in (
        "tracker_started",
        "tracker_finalized",
        "tracker_productive",
        "tracker_reason",
        "event_cumulative_impulse_n_s",
        "tracker_contact_point_w",
        "tracker_contact_error_m",
        "tracker_contact_quality",
        "tracker_contact_quality_valid",
        "tracker_contact_quality_overflow",
        "tracker_first_contact_time_s",
        "tracker_contact_normal_axiality",
        "event_cumulative_transverse_impulse_n_s",
    ):
        trace["event_trace"][key].append(
            copy.deepcopy(trace["event_trace"][key][-1])
        )
    trace["event_trace"]["tracker_finalized"] = [False, False, False, True]
    trace["event_trace"]["event_cumulative_impulse_n_s"] = [
        0.0,
        0.05,
        0.08,
        0.10,
    ]
    metrics = analyze_quality_episode(trace, nail_geometry=GEOMETRY)
    assert metrics["event_window_depth_gain_m_sampled"] == pytest.approx(0.005)


def test_no_contact_and_zero_force_are_zero_but_instrumentation_failures_reject() -> None:
    no_contact = analyze_quality_episode(_no_contact_trace(), nail_geometry=GEOMETRY)
    assert no_contact["first_contact_quality_sampled"] == 0.0
    assert no_contact["first_contact_radial_error_m_sampled"] == 0.0
    zero_force = _trace()
    zero_force["physical"]["quality_normal_force_n"][1] = [
        0.0
    ] * QUALITY_SENSOR_SLOT_COUNT
    zero_force["first_strike"]["contact_quality_valid"] = False
    zero_force["first_strike"]["contact_quality"] = 0.0
    zero_force["first_strike"]["contact_error_m"] = 0.0
    zero_force["first_strike"]["contact_point_w"] = [0.0, 0.0, 0.0]
    zero_force["first_strike"]["contact_normal_axiality"] = 0.0
    for key, value in {
        "tracker_contact_quality_valid": False,
        "tracker_contact_quality": 0.0,
        "tracker_contact_error_m": 0.0,
        "tracker_contact_point_w": [0.0, 0.0, 0.0],
        "tracker_contact_normal_axiality": 0.0,
    }.items():
        zero_force["event_trace"][key][1:] = [copy.deepcopy(value)] * 2
    assert (
        analyze_quality_episode(zero_force, nail_geometry=GEOMETRY)[
            "first_contact_quality_sampled"
        ]
        == 0.0
    )
    cases = [
        (lambda t: t["first_strike"].pop("contact_quality"), "missing"),
        (
            lambda t: t["first_strike"].__setitem__(
                "contact_quality_overflow", True
            ),
            "overflow",
        ),
        (
            lambda t: t["first_strike"]["contact_point_w"].__setitem__(0, np.nan),
            "nonfinite",
        ),
    ]
    for mutate, message in cases:
        invalid = _trace()
        mutate(invalid)
        with pytest.raises(ValueError, match=message):
            analyze_quality_episode(invalid, nail_geometry=GEOMETRY)


def test_quality_recomputes_from_positive_contact_normal_force_not_axial_force() -> None:
    trace = _trace()
    trace["physical"]["net_axial_force_n"][1] = 0.0
    metrics = analyze_quality_episode(trace, nail_geometry=GEOMETRY)
    assert metrics["first_contact_quality_sampled"] == pytest.approx(0.55)
    assert metrics["first_contact_quality_valid_sampled"] is True


def _float32_quality_trace() -> tuple[dict, float, float]:
    trace = _trace()
    zero = [0.0, 0.0, 0.0]
    found = [1, 1] + [0] * (QUALITY_SENSOR_SLOT_COUNT - 2)
    force = [
        12.345678329467773,
        8.765432357788086,
    ] + [0.0] * (QUALITY_SENSOR_SLOT_COUNT - 2)
    positions = [
        [0.5071234703063965, 0.0023456700146198273, 0.10000000149011612],
        [0.5065432190895081, -0.00123456004075706, 0.10000000149011612],
    ] + [zero] * (QUALITY_SENSOR_SLOT_COUNT - 2)
    normals = [
        [0.10000000149011612, 0.20000000298023224, -0.9700000286102295],
        [-0.20000000298023224, 0.10000000149011612, -0.9700000286102295],
    ] + [zero] * (QUALITY_SENSOR_SLOT_COUNT - 2)
    physical = trace["physical"]
    physical["quality_found_count"][1] = found
    physical["quality_normal_force_n"][1] = force
    physical["quality_contact_position_m"][1] = positions
    physical["quality_contact_normal"][1] = normals

    # Hand-banked outputs from the producer's float32 tensor arithmetic.  The
    # test deliberately does not derive its expected values with analysis code.
    point = [
        0.5068825483322144,
        0.000859141640830785,
        0.10000000149011612,
    ]
    error = 0.006935963872820139
    contact_quality = 0.6659195423126221
    axiality = 0.9866067171096802
    event = trace["event_trace"]
    event["tracker_contact_point_w"][1:] = [point, point]
    event["tracker_contact_error_m"][1:] = [error, error]
    event["tracker_contact_quality"][1:] = [
        contact_quality,
        contact_quality,
    ]
    event["tracker_contact_normal_axiality"][1:] = [axiality, axiality]
    trace["first_strike"].update(
        {
            "contact_point_w": point,
            "contact_error_m": error,
            "contact_quality": contact_quality,
            "contact_normal_axiality": axiality,
        }
    )
    return trace, contact_quality, axiality


def test_quality_validation_replays_producer_float32_arithmetic() -> None:
    """Serialized finite float32 evidence must survive offline recomputation."""

    trace, contact_quality, _ = _float32_quality_trace()
    metrics = analyze_quality_episode(trace, nail_geometry=GEOMETRY)
    assert metrics["first_contact_quality_sampled"] == contact_quality


def test_quality_validation_matches_actual_torch_producer_across_contacts() -> None:
    """Offline validation must follow the real producer, not NumPy lookalikes."""

    source = (
        Path(__file__).parents[1]
        / "src/tasks/hammer/mdp/contact_quality.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_contact_quality_reference", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    producer = module.contact_point_quality
    rng = np.random.default_rng(20260728)
    axis = torch.tensor(GEOMETRY["nail_axis"], dtype=torch.float32)
    nail_top = torch.tensor(
        [[*GEOMETRY["nail_xy_m"], 0.0]], dtype=torch.float32
    )

    for _ in range(32):
        trace = _trace()
        found = torch.zeros(1, QUALITY_SENSOR_SLOT_COUNT, dtype=torch.float32)
        found[:, :4] = 1.0
        force = torch.zeros(
            1, QUALITY_SENSOR_SLOT_COUNT, 3, dtype=torch.float32
        )
        force[0, :4, 0] = torch.tensor(
            rng.uniform(0.1, 50.0, 4), dtype=torch.float32
        )
        position = torch.zeros(
            1, QUALITY_SENSOR_SLOT_COUNT, 3, dtype=torch.float32
        )
        position[0, :4] = torch.tensor(
            np.column_stack(
                (
                    rng.uniform(0.49, 0.51, 4),
                    rng.uniform(-0.01, 0.01, 4),
                    rng.uniform(0.08, 0.12, 4),
                )
            ),
            dtype=torch.float32,
        )
        normal = torch.zeros(
            1, QUALITY_SENSOR_SLOT_COUNT, 3, dtype=torch.float32
        )
        normal[0, :4] = torch.tensor(
            rng.normal(size=(4, 3)), dtype=torch.float32
        )
        point, error, contact_quality, valid, overflow = producer(
            found=found,
            force_contact=force,
            position_w=position,
            nail_top_w=nail_top,
            nail_axis_w=axis,
            nail_radius_m=GEOMETRY["nail_radius_m"],
            num_slots=QUALITY_SENSOR_SLOT_COUNT,
        )
        weights = torch.where(
            found > 0.0,
            force[..., 0].clamp_min(0.0),
            torch.zeros_like(found),
        )
        weighted_normal = (normal * weights.unsqueeze(-1)).sum(dim=1)
        normal_norm = torch.linalg.vector_norm(
            weighted_normal, dim=-1, keepdim=True
        )
        unit_normal = weighted_normal / torch.where(
            normal_norm > 0.0,
            normal_norm,
            torch.ones_like(normal_norm),
        )
        axiality = (unit_normal * axis).sum(dim=-1).clamp(0.0, 1.0)
        axiality = torch.where(
            valid & (normal_norm.squeeze(-1) > 0.0),
            axiality,
            torch.zeros_like(axiality),
        )

        physical = trace["physical"]
        physical["quality_found_count"][1] = found[0].tolist()
        physical["quality_normal_force_n"][1] = force[0, :, 0].tolist()
        physical["quality_contact_position_m"][1] = position[0].tolist()
        physical["quality_contact_normal"][1] = normal[0].tolist()
        snapshot = {
            "contact_point_w": point[0].tolist(),
            "contact_error_m": float(error[0]),
            "contact_quality": float(contact_quality[0]),
            "contact_quality_valid": bool(valid[0]),
            "contact_quality_overflow": bool(overflow[0]),
            "contact_normal_axiality": float(axiality[0]),
        }
        trace["first_strike"].update(snapshot)
        event = trace["event_trace"]
        for event_key, snapshot_key in (
            ("tracker_contact_point_w", "contact_point_w"),
            ("tracker_contact_error_m", "contact_error_m"),
            ("tracker_contact_quality", "contact_quality"),
            ("tracker_contact_quality_valid", "contact_quality_valid"),
            ("tracker_contact_quality_overflow", "contact_quality_overflow"),
            ("tracker_contact_normal_axiality", "contact_normal_axiality"),
        ):
            event[event_key][1:] = [
                snapshot[snapshot_key],
                snapshot[snapshot_key],
            ]
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_quality_recomputation_allows_bounded_float32_roundoff() -> None:
    """Accept only the bounded CUDA roundoff observed for axiality."""

    trace, _, axiality = _float32_quality_trace()
    producer_value = np.float32(axiality)
    roundoff = float(producer_value + np.float32(np.finfo(np.float32).eps))
    trace["first_strike"]["contact_normal_axiality"] = roundoff
    trace["event_trace"]["tracker_contact_normal_axiality"][1:] = [
        roundoff,
        roundoff,
    ]
    analyze_quality_episode(trace, nail_geometry=GEOMETRY)


@pytest.mark.parametrize(
    ("field", "drift"),
    [
        ("contact_point_w", 1e-7),
        ("contact_error_m", 1e-8),
        ("contact_quality", 5e-7),
        ("contact_normal_axiality", 2e-7),
    ],
)
def test_quality_recomputation_rejects_field_specific_material_drift(
    field: str, drift: float
) -> None:
    """A single generic tolerance must not hide dimensionally different drift."""

    trace = _trace()
    value = trace["first_strike"][field]
    if isinstance(value, list):
        material_drift = list(value)
        material_drift[0] = float(
            np.float32(material_drift[0]) + np.float32(drift)
        )
    else:
        material_drift = float(np.float32(value) + np.float32(drift))
    event_key = {
        "contact_point_w": "tracker_contact_point_w",
        "contact_error_m": "tracker_contact_error_m",
        "contact_quality": "tracker_contact_quality",
        "contact_normal_axiality": "tracker_contact_normal_axiality",
    }[field]
    trace["first_strike"][field] = material_drift
    trace["event_trace"][event_key][1:] = [
        material_drift,
        material_drift,
    ]
    with pytest.raises(
        ValueError,
        match=rf"snapshot differs from recomputed {field}",
    ):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_no_onset_requires_zero_initialized_tracker_quality_streams() -> None:
    """A no-contact snapshot cannot coexist with a nonzero dense tracker latch."""

    trace = _no_contact_trace()
    trace["event_trace"]["tracker_contact_quality"][1] = 0.5

    with pytest.raises(ValueError, match=r"event/snapshot"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_no_onset_requires_false_initialized_tracker_overflow_stream() -> None:
    """Overflow cannot be latched when the tracker accepted no contact onset."""

    trace = _no_contact_trace()
    trace["event_trace"]["tracker_contact_quality_overflow"][1] = True

    with pytest.raises(ValueError, match=r"overflow"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_quality_validation_rejects_values_outside_float32_range() -> None:
    """Finite JSON numbers that overflow the producer dtype fail closed."""

    trace = _trace()
    trace["first_strike"]["contact_point_w"][0] = 1e300
    trace["event_trace"]["tracker_contact_point_w"][1:] = [
        list(trace["first_strike"]["contact_point_w"]),
        list(trace["first_strike"]["contact_point_w"]),
    ]

    with pytest.raises(ValueError, match=r"float32 range"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_event_snapshot_binding_remains_exact_across_float32_roundoff() -> None:
    """Tolerance applies only to raw recomputation, never to the stored latch."""

    trace = _trace()
    stored = np.float32(trace["first_strike"]["contact_quality"])
    trace["event_trace"]["tracker_contact_quality"][1] = float(
        np.nextafter(stored, np.float32(np.inf))
    )

    with pytest.raises(ValueError, match=r"event/snapshot"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


@pytest.mark.parametrize("sample", [0, 2])
def test_raw_quality_requires_float32_range_across_dense_trace(sample: int) -> None:
    """Every raw slot was produced as float32, not only the onset sample."""

    trace = _trace()
    trace["physical"]["quality_contact_position_m"][sample][0][0] = 1e300

    with pytest.raises(ValueError, match=r"float32 range"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_no_onset_raw_quality_requires_float32_range() -> None:
    """No-contact evidence cannot hide an impossible dense raw value."""

    trace = _no_contact_trace()
    trace["physical"]["quality_contact_position_m"][1][0][0] = 1e300

    with pytest.raises(ValueError, match=r"float32 range"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


@pytest.mark.parametrize(
    ("event_key", "sample", "value"),
    [
        ("tracker_contact_quality", 0, 0.5),
        ("tracker_contact_quality", 2, 0.5),
        ("tracker_contact_quality_overflow", 0, True),
        ("tracker_first_contact_time_s", 2, 0.5),
    ],
)
def test_accepted_onset_quality_latches_are_dense_and_immutable(
    event_key: str, sample: int, value: object
) -> None:
    """The dense event trace must be zero before onset and latched afterward."""

    trace = _trace()
    trace["event_trace"][event_key][sample] = value

    with pytest.raises(ValueError, match=r"event/snapshot|overflow"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_first_contact_time_binding_is_exact_and_float32_bounded() -> None:
    """The event and snapshot serialize one float32 tracker latch."""

    mismatch = _trace()
    stored_time = np.float32(
        mismatch["event_trace"]["tracker_first_contact_time_s"][1]
    )
    mismatch["event_trace"]["tracker_first_contact_time_s"][1] = float(
        np.nextafter(stored_time, np.float32(np.inf))
    )
    with pytest.raises(ValueError, match=r"event/snapshot"):
        analyze_quality_episode(mismatch, nail_geometry=GEOMETRY)

    out_of_range = _trace()
    out_of_range["first_strike"]["first_contact_time_s"] = 1e300
    out_of_range["event_trace"]["tracker_first_contact_time_s"][1:] = [
        1e300,
        1e300,
    ]
    with pytest.raises(ValueError, match=r"float32 range"):
        analyze_quality_episode(out_of_range, nail_geometry=GEOMETRY)


def test_no_onset_requires_zero_snapshot_time() -> None:
    """The unlatched first-contact clock remains exactly initialized."""

    trace = _no_contact_trace()
    trace["first_strike"]["first_contact_time_s"] = 5e-13

    with pytest.raises(ValueError, match=r"first_contact_time_s"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("first_contact_time_s", -0.004),
        ("contact_error_m", -5e-10),
        ("contact_quality", float(np.float32(1.0) + np.finfo(np.float32).eps)),
        ("contact_normal_axiality", -float(np.finfo(np.float32).eps)),
    ],
)
def test_quality_snapshot_enforces_physical_domains(
    field: str, value: float
) -> None:
    """Finite, mutually latched values still must respect their domains."""

    trace = _trace()
    event_key = {
        "first_contact_time_s": "tracker_first_contact_time_s",
        "contact_error_m": "tracker_contact_error_m",
        "contact_quality": "tracker_contact_quality",
        "contact_normal_axiality": "tracker_contact_normal_axiality",
    }[field]
    trace["first_strike"][field] = value
    trace["event_trace"][event_key][1:] = [value, value]

    with pytest.raises(ValueError, match=field):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_tracker_started_is_initialized_then_immutable() -> None:
    """The tracker-started latch cannot turn off after accepted onset."""

    trace = _trace()
    trace["event_trace"]["tracker_started"][2] = False

    with pytest.raises(ValueError, match=r"started"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_no_onset_still_validates_nail_geometry_float32_range() -> None:
    """A no-contact outcome cannot bypass malformed frozen geometry."""

    geometry = copy.deepcopy(GEOMETRY)
    geometry["nail_radius_m"] = 1e300

    with pytest.raises(ValueError, match=r"nail"):
        analyze_quality_episode(_no_contact_trace(), nail_geometry=geometry)


def test_numeric_quality_arrays_reject_mixed_json_booleans() -> None:
    """A bool nested among numbers cannot be silently coerced to 0/1."""

    trace = _trace()
    trace["physical"]["quality_normal_force_n"][1][0] = True

    with pytest.raises(ValueError, match=r"JSON numbers"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


@pytest.mark.parametrize(
    "width",
    [
        pytest.param(1, id="missing-sensor-one-slot-fallback"),
        pytest.param(7, id="undersized-seven-slot"),
        pytest.param(9, id="oversized-nine-slot"),
    ],
)
def test_analyze_quality_episode_requires_exactly_eight_raw_slots(width) -> None:
    """Task 9 must reject a raw sensor signature other than eight slots."""

    trace = _trace()
    _resize_raw_quality_slots(trace, width)

    with pytest.raises(ValueError, match=r"exactly 8 slots"):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda trace: trace["physical"].__setitem__(
                "quality_found_count", [0, 1, 1]
            ),
            "shape",
        ),
        (
            lambda trace: trace["physical"]["quality_found_count"][1].__setitem__(
                0, 9
            ),
            "overflow",
        ),
        (
            lambda trace: trace["first_strike"].__setitem__(
                "contact_quality", 0.25
            ),
            "event/snapshot|recomputed",
        ),
        (
            lambda trace: trace["event_trace"][
                "tracker_contact_quality"
            ].__setitem__(1, 0.25),
            "event/snapshot",
        ),
        (
            lambda trace: trace["event_trace"][
                "tracker_contact_quality_overflow"
            ].__setitem__(1, True),
            "overflow",
        ),
    ],
)
def test_raw_quality_slots_and_latches_fail_closed(mutation, message) -> None:
    trace = _trace()
    mutation(trace)
    with pytest.raises(ValueError, match=message):
        analyze_quality_episode(trace, nail_geometry=GEOMETRY)


def test_quality_overflow_scope_is_only_the_tracker_accepted_onset() -> None:
    """Later raw slot pressure cannot rewrite the immutable onset snapshot."""
    trace = _trace()
    trace["physical"]["quality_found_count"][2][0] = 9

    metrics = analyze_quality_episode(trace, nail_geometry=GEOMETRY)

    assert metrics["first_contact_quality_sampled"] == pytest.approx(0.55)


def test_mechanism_delivered_endpoint_excludes_post_window_tail_contact() -> None:
    trace = _trace(delivered=0.10)
    trace["episode_delivered_accumulator_n_s"] = 0.45
    metrics = analyze_quality_episode(trace, nail_geometry=GEOMETRY)
    assert metrics["raw_delivered_impulse_n_s_sampled"] == pytest.approx(0.10)


def test_seed_aggregate_keeps_episode_denominators_and_safety_channels() -> None:
    contact = analyze_quality_episode(_trace(), nail_geometry=GEOMETRY)
    no_contact = analyze_quality_episode(_no_contact_trace(), nail_geometry=GEOMETRY)
    result = aggregate_quality_episode_metrics(
        [contact, no_contact], expected_episode_count=2
    )
    assert result["first_contact_quality_sampled"] == pytest.approx(0.275)
    assert result["first_contact_radial_error_mean_sampled"] == pytest.approx(
        contact["first_contact_radial_error_m_sampled"] / 2
    )
    assert result["first_window_success_rate_sampled"] == pytest.approx(0.5)
    assert result["overall_success_rate_sampled"] == pytest.approx(0.5)
    assert result["lambda_joint2_max_n_m_s_sampled"] == pytest.approx(0.2)
    assert result["worst_lambda_cap_ratio_max_sampled"] == pytest.approx(
        0.2 / 3.28
    )
    assert len(result["valid_contact_coordinates_sampled"]) == 1
    coordinate = result["valid_contact_coordinates_sampled"][0]
    assert coordinate["episode_id"] == "F8-env0-episode0"
    assert coordinate["env_id"] == 0
    assert coordinate["episode_ordinal"] == 0
    assert coordinate["nail_asset_sha256"] == "c" * 64
    assert coordinate["x_m"] == pytest.approx(
        contact["contact_plane_x_m_sampled"]
    )


def test_frozen_contract_maps_fq_to_fq_min_and_accepts_complete_manifest(tmp_path) -> None:
    rows, manifest = _campaign_rows(tmp_path)
    for row, accepted in zip(rows, manifest, strict=True):
        path = tmp_path / f"{row['name']}.npz"
        path.write_bytes(row["name"].encode())
        row["sampled_trace_path"] = str(path)
        row["sampled_trace_digest"] = hashlib.sha256(
            f"payload-{row['name']}".encode()
        ).hexdigest()
        row["sampled_trace_artifact_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        accepted["sampled_trace_digest"] = row["sampled_trace_digest"]
        accepted["sampled_trace_artifact_sha256"] = row[
            "sampled_trace_artifact_sha256"
        ]
    assert len(campaign_manifests.EVALUATION_FIELDS) == 39
    serialized = campaign_manifests.serialize_evaluation_manifest(manifest)
    parsed_manifest = campaign_manifests.parse_evaluation_manifest(serialized)
    assert tuple(parsed_manifest[0]) == campaign_manifests.EVALUATION_FIELDS
    for row, accepted in zip(rows, parsed_manifest, strict=True):
        assert row["campaign_config_sha256"] != accepted["campaign_config_sha256"]
        assert row["treatment_config_sha256"] != accepted[
            "treatment_config_sha256"
        ]
    validate_quality_campaign_contract(rows, parsed_manifest)
    fq = next(item for item in manifest if item["short"] == "fq")
    assert fq["arm"] == "FQ-min"
    frozen_fq = next(
        item
        for item in FROZEN_QUALITY_CAMPAIGN_MATRIX
        if item["treatment"] == "FQ-min"
    )
    assert frozen_fq["raw_treatment"] == "FQ"
    assert frozen_fq["delivered_weight"] == 0.0
    assert frozen_fq["impact_reader"] == "FirstStrikeQualityImpactRewardTerm"
    assert frozen_fq["delivered_reader"] == "FirstStrikeDeliveredRewardTerm"
    assert frozen_fq["speed_normalizer_m_s"] == pytest.approx(1.4598331451416016)
    tasks = {
        item["treatment"]: item["task"]
        for item in FROZEN_QUALITY_CAMPAIGN_MATRIX
    }
    assert tasks["F0"].endswith("-Event-Linear-F0")
    assert tasks["D0"].endswith("-Event-Linear-D0")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows, manifest: manifest[0].__setitem__(
                "campaign", "wrong-campaign"
            ),
            "campaign",
        ),
        (
            lambda rows, manifest: rows[0].__setitem__(
                "fixed_impedance_signature_sha256", "f" * 64
            ),
            "gains",
        ),
        (
            lambda rows, manifest: rows[0].__setitem__(
                "fixed_action_signature_sha256", "f" * 64
            ),
            "action",
        ),
        (
            lambda rows, manifest: rows[0].__setitem__(
                "accepted_checkpoint_sha256", "f" * 64
            ),
            "checkpoint",
        ),
        (
            lambda rows, manifest: (
                rows[0].__setitem__("accepted_manifest_sha256", None),
                manifest[0].__setitem__(
                    "accepted_training_manifest_sha256", None
                ),
            ),
            "manifest",
        ),
        (
            lambda rows, manifest: rows[0].__setitem__(
                "git_revision", "unknown"
            ),
            "revision",
        ),
        (
            lambda rows, manifest: manifest[0].__setitem__(
                "asset_revision", "f" * 40
            ),
            "revision",
        ),
        (
            lambda rows, manifest: manifest[0].__setitem__(
                "code_revision", "9" * 40
            ),
            "training code revision",
        ),
        (
            lambda rows, manifest: (
                manifest[1].__setitem__(
                    "training_policy_observation_sha256", "f" * 64
                ),
                manifest[1].__setitem__(
                    "evaluation_policy_observation_sha256", "f" * 64
                ),
            ),
            "treatment-shared",
        ),
        (
            lambda rows, manifest: manifest[1].__setitem__(
                "training_config_sha256", "f" * 64
            ),
            "per-treatment",
        ),
    ],
)
def test_contract_rejects_mutant_campaign_identity_bindings(
    tmp_path, mutation, message
) -> None:
    rows, manifest = _campaign_rows(tmp_path)
    for row, accepted in zip(rows, manifest, strict=True):
        path = tmp_path / f"{row['name']}.npz"
        path.write_bytes(row["name"].encode())
        row["sampled_trace_path"] = str(path)
        row["sampled_trace_digest"] = hashlib.sha256(
            f"payload-{row['name']}".encode()
        ).hexdigest()
        row["sampled_trace_artifact_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        accepted["sampled_trace_digest"] = row["sampled_trace_digest"]
        accepted["sampled_trace_artifact_sha256"] = row[
            "sampled_trace_artifact_sha256"
        ]
    mutation(rows, manifest)
    with pytest.raises(ValueError, match=message):
        validate_quality_campaign_contract(rows, manifest)


def test_schema_v3_payload_identities_are_manifest_bound(complete_campaign) -> None:
    rows, manifest = complete_campaign
    row, accepted = rows[0], manifest[0]
    payload = legacy._sampled_payload_from_bytes(
        Path(row["sampled_trace_path"]).read_bytes()
    )
    assert quality_analysis._quality_payload_identity_reasons(
        row, accepted, payload
    ) == []

    missing = copy.deepcopy(payload)
    missing["evaluation_contract"]["strict_config_identities"].pop(
        "evaluation_config_sha256"
    )
    assert any(
        "six strict" in reason
        for reason in quality_analysis._quality_payload_identity_reasons(
            row, accepted, missing
        )
    )
    drifted = copy.deepcopy(payload)
    drifted["evaluation_contract"][
        "fixed_impedance_signature_sha256"
    ] = "f" * 64
    assert any(
        "gains" in reason
        for reason in quality_analysis._quality_payload_identity_reasons(
            row, accepted, drifted
        )
    )


@pytest.mark.parametrize("label", ("F8", "F0", "D0", "FQ-min"))
def test_self_consistent_wrong_reader_reward_hash_is_rejected(
    complete_campaign, label
) -> None:
    rows, manifest = complete_campaign
    accepted = copy.deepcopy(
        next(item for item in manifest if item["arm"] == label)
    )
    row = next(
        item
        for item in rows
        if item["treatment"] == quality_analysis.RAW_TREATMENT[accepted["arm"]]
        and item["training_seed"] == accepted["training_seed"]
    )
    payload = legacy._sampled_payload_from_bytes(
        Path(row["sampled_trace_path"]).read_bytes()
    )
    payload = copy.deepcopy(payload)
    mutant_hash = WRONG_READER_HASHES[label]
    accepted["training_treatment_reward_sha256"] = mutant_hash
    accepted["evaluation_treatment_reward_sha256"] = mutant_hash
    strict = payload["evaluation_contract"]["strict_config_identities"]
    strict["training_treatment_reward_sha256"] = mutant_hash
    strict["evaluation_treatment_reward_sha256"] = mutant_hash
    reasons = quality_analysis._quality_payload_identity_reasons(
        row, accepted, payload
    )
    assert any("frozen reward semantics" in reason for reason in reasons)


def test_self_consistent_fq_wrong_speed_normalizer_hash_is_rejected(
    complete_campaign,
) -> None:
    rows, manifest = complete_campaign
    accepted = copy.deepcopy(
        next(item for item in manifest if item["arm"] == "FQ-min")
    )
    row = next(
        item
        for item in rows
        if item["treatment"] == "FQ"
        and item["training_seed"] == accepted["training_seed"]
    )
    payload = legacy._sampled_payload_from_bytes(
        Path(row["sampled_trace_path"]).read_bytes()
    )
    payload = copy.deepcopy(payload)
    accepted["training_treatment_reward_sha256"] = FQ_WRONG_NORMALIZER_HASH
    accepted["evaluation_treatment_reward_sha256"] = FQ_WRONG_NORMALIZER_HASH
    strict = payload["evaluation_contract"]["strict_config_identities"]
    strict["training_treatment_reward_sha256"] = FQ_WRONG_NORMALIZER_HASH
    strict["evaluation_treatment_reward_sha256"] = FQ_WRONG_NORMALIZER_HASH
    reasons = quality_analysis._quality_payload_identity_reasons(
        row, accepted, payload
    )
    assert any("frozen reward semantics" in reason for reason in reasons)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r, m: r.pop(), "32"),
        (
            lambda r, m: r[1].__setitem__("training_seed", 8),
            "identities",
        ),
        (lambda r, m: m[-1].__setitem__("short", "fq-min"), "short"),
        (lambda r, m: r[-1].__setitem__("task", "wrong"), "task"),
        (lambda r, m: r[-1].__setitem__("delivered_weight", 2.0), "weights"),
        (
            lambda r, m: m[0].__setitem__(
                "fixed_impedance_signature_sha256", "f" * 64
            ),
            "gains",
        ),
        (
            lambda r, m: m[0].__setitem__("cap_signature_sha256", "f" * 64),
            "caps",
        ),
        (lambda r, m: r[0].__setitem__("imp_max_p", 0.1), "imp_max_p"),
        (
            lambda r, m: m[0].__setitem__(
                "checkpoint_path", "/accepted/f8-seed8/model_500.pt"
            ),
            "checkpoint",
        ),
        (lambda r, m: r[0].__setitem__("num_envs", 128), "quota"),
        (lambda r, m: m[0].__setitem__("num_envs", 128), "quota"),
        (
            lambda r, m: m[0].__setitem__("episodes_per_env_sampled", 1),
            "quota",
        ),
        (lambda r, m: m[0].__setitem__("n_episodes_sampled", 511), "quota"),
        (
            lambda r, m: m[0].update(
                {
                    "evaluation_attempt": "attempt99",
                    "evaluation_retry_history": "attempt99:accepted",
                }
            ),
            "evaluation attempt",
        ),
        (lambda r, m: r[0].__setitem__("git_dirty", True), "dirty"),
        (
            lambda r, m: m[0].__setitem__(
                "sampled_trace_digest", "f" * 64
            ),
            "binding",
        ),
        (lambda r, m: m[0].__setitem__("clean_state", False), "clean_state"),
        (
            lambda r, m: m[0].__setitem__("liveness_failure_n", 1),
            "sentinel",
        ),
        (
            lambda r, m: m[0].__setitem__("quality_overflow_n", 0.5),
            "sentinel",
        ),
        (
            lambda r, m: m[0].__setitem__("quality_nonfinite_n", -0.5),
            "sentinel",
        ),
        (
            lambda r, m: m[0].__setitem__("quota_failure_n", False),
            "sentinel",
        ),
        (lambda r, m: m.pop(), "accepted-evaluation"),
    ],
)
def test_contract_rejects_every_frozen_drift(
    tmp_path, mutation, message
) -> None:
    rows, manifest = _campaign_rows(tmp_path)
    for row, accepted in zip(rows, manifest, strict=True):
        path = tmp_path / f"{row['name']}.npz"
        path.write_bytes(row["name"].encode())
        row["sampled_trace_path"] = str(path)
        row["sampled_trace_digest"] = hashlib.sha256(row["name"].encode()).hexdigest()
        row["sampled_trace_artifact_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        accepted["sampled_trace_digest"] = row["sampled_trace_digest"]
        accepted["sampled_trace_artifact_sha256"] = row[
            "sampled_trace_artifact_sha256"
        ]
    mutation(rows, manifest)
    with pytest.raises(ValueError, match=message):
        validate_quality_campaign_contract(rows, manifest)


def test_contract_accepts_training_code_revision_that_differs_from_eval_git_revision(
    tmp_path,
) -> None:
    """Provenance over-constraint fix: training and evaluation code
    revisions are independently pinned, not required to be equal -- a
    persistence/provenance-only fix can land in the evaluation checkout
    after training froze. Canonical code_revision still binds the row's
    training_code_revision, and the asset revision equality (training vs.
    evaluation) is untouched and still required -- only the code side is
    decoupled."""
    rows, manifest = _campaign_rows(tmp_path)
    new_training_code_revision = "9" * 40
    for row, accepted in zip(rows, manifest, strict=True):
        path = tmp_path / f"{row['name']}.npz"
        path.write_bytes(row["name"].encode())
        row["sampled_trace_path"] = str(path)
        row["sampled_trace_digest"] = hashlib.sha256(
            f"payload-{row['name']}".encode()
        ).hexdigest()
        row["sampled_trace_artifact_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        accepted["sampled_trace_digest"] = row["sampled_trace_digest"]
        accepted["sampled_trace_artifact_sha256"] = row[
            "sampled_trace_artifact_sha256"
        ]
        row["training_code_revision"] = new_training_code_revision
        accepted["code_revision"] = new_training_code_revision
        # git_revision (the evaluation checkout) is deliberately left at the
        # fixture baseline ("a" * 40): canonical code_revision binds the
        # training checkout, not the later persistence-only evaluator fix.

    validate_quality_campaign_contract(rows, manifest)  # must not raise


def test_analysis_uses_one_mwu_holm_family_and_sign_flip_is_sensitivity(
    analysis,
) -> None:
    assert analysis["valid"] is True, analysis
    assert set(analysis["primary_contrasts"]) == {
        "F0_minus_F8",
        "D0_minus_F8",
        "FQ-min_minus_D0",
    }
    assert len(analysis["holm_mann_whitney_family"]) == 3
    for contrast in analysis["primary_contrasts"].values():
        assert len(contrast["paired_differences"]) == 8
        assert set(contrast["arm_means"]) == {"treatment", "control"}
        assert contrast["paired_bootstrap_difference"]["samples"] == 100_000
        assert contrast["decision_basis"] == "holm_mann_whitney_only"
        assert "paired_sign_flip" in contrast
        assert "paired_sign_flip" not in contrast["decision_gates"]
    assert "holm_paired_sign_flip_family" not in analysis


def test_fq_min_practical_gate_is_strict_and_excludes_delivered_and_sensitivity() -> None:
    rows = {
        "F8": [
            {
                "first_contact_quality_sampled": 0.5,
                "first_window_useful_speed_mean_sampled": 1.0,
                "first_window_success_rate_sampled": 0.95,
                "overall_success_rate_sampled": 0.95,
                "event_window_depth_gain_mean_sampled": 0.01,
            }
        ]
        * 8,
        "FQ-min": [
            {
                "first_contact_quality_sampled": 0.6,
                "first_window_useful_speed_mean_sampled": 0.95,
                "first_window_success_rate_sampled": 0.90,
                "overall_success_rate_sampled": 0.90,
                "event_window_depth_gain_mean_sampled": 0.009,
            }
        ]
        * 8,
    }
    result = _fq_min_practical_acceptance(rows, safety_gates_pass=True)
    assert result["gates"]["quality_gain_at_least_0_10"] is True
    assert result["gates"]["useful_speed_ratio_lower_gt_0_95"] is False
    assert result["gates"]["depth_gain_ratio_lower_gt_0_90"] is False
    assert "raw_delivered_impulse" not in json.dumps(result)
    assert "paired_sign_flip" not in json.dumps(result)
    assert result["passed"] is False
    for endpoint in ("first_window_success", "overall_success"):
        assert result[endpoint]["paired_bootstrap_difference"]["samples"] == 100_000
        assert set(result[endpoint]["arm_means"]) == {
            "treatment",
            "control",
        }


def test_practical_ratio_denominator_failure_returns_failed_gates() -> None:
    rows = {
        label: [
            {
                "training_seed": seed,
                "first_contact_quality_sampled": 0.7,
                "first_window_useful_speed_mean_sampled": (
                    0.9 if label == "FQ-min" else 0.0
                ),
                "first_window_success_rate_sampled": 1.0,
                "overall_success_rate_sampled": 1.0,
                "event_window_depth_gain_mean_sampled": (
                    0.01 if label == "FQ-min" else 0.0
                ),
            }
            for seed in range(8, 16)
        ]
        for label in ("F8", "FQ-min")
    }
    result = _fq_min_practical_acceptance(rows, safety_gates_pass=True)
    assert result["useful_speed_ratio"]["valid"] is False
    assert result["depth_gain_ratio"]["valid"] is False
    assert result["gates"]["useful_speed_ratio_lower_gt_0_95"] is False
    assert result["gates"]["depth_gain_ratio_lower_gt_0_90"] is False
    assert result["passed"] is False


def test_analysis_emits_d0_mechanism_without_enlarging_holm(analysis) -> None:
    mechanism = analysis["d0_mechanism"]
    assert mechanism["interpretation"] == "mainly selected dwell/recontact"
    assert mechanism["claim_allowed"] is True
    assert set(mechanism["endpoints"]) == {
        "event_window_depth_gain",
        "contact_dwell",
        "recontact_count",
        "raw_delivered_impulse",
        "first_window_success",
        "overall_success",
    }
    assert all(
        len(endpoint["paired_differences"]) == 8
        for endpoint in mechanism["endpoints"].values()
    )
    assert len(analysis["holm_mann_whitney_family"]) == 3


def test_analysis_retains_provenance_linked_valid_contact_coordinates(
    analysis,
) -> None:
    coordinates = analysis["valid_contact_coordinates"]
    assert len(coordinates) == 32 * 512
    assert {
        "treatment",
        "raw_treatment",
        "training_seed",
        "episode_id",
        "env_id",
        "episode_ordinal",
        "nail_asset_sha256",
        "x_m",
        "y_m",
    } <= set(coordinates[0])
    assert all(
        coordinate["nail_asset_sha256"] == "c" * 64
        for coordinate in coordinates
    )


def test_common_nail_geometry_requires_identical_provenance_frame_and_radius() -> None:
    assert quality_analysis._validate_common_nail_geometry(
        [GEOMETRY, copy.deepcopy(GEOMETRY)]
    ) == GEOMETRY
    radius_drift = copy.deepcopy(GEOMETRY)
    radius_drift["nail_radius_m"] = 0.02
    with pytest.raises(ValueError, match="common nail"):
        quality_analysis._validate_common_nail_geometry(
            [GEOMETRY, radius_drift]
        )
    frame_drift = copy.deepcopy(GEOMETRY)
    frame_drift["nail_axis"] = [0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="common nail"):
        quality_analysis._validate_common_nail_geometry(
            [GEOMETRY, frame_drift]
        )


def test_production_geometry_extras_use_canonical_required_projection() -> None:
    production_geometry = {
        **copy.deepcopy(GEOMETRY),
        "source_path": "/accepted/assets/hammer_scene.xml",
        "basis": {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
        },
    }
    assert quality_analysis._validate_common_nail_geometry(
        [production_geometry, copy.deepcopy(production_geometry)]
    ) == GEOMETRY
    drifted = copy.deepcopy(production_geometry)
    drifted["nail_axis"] = [0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="common nail"):
        quality_analysis._validate_common_nail_geometry(
            [production_geometry, drifted]
        )


def test_d0_mechanism_phrase_requires_all_four_registered_conditions(analysis) -> None:
    by_arm = {
        label: [
            copy.deepcopy(row)
            for row in analysis["seed_aggregates"]
            if row["treatment"] == label
        ]
        for label in ("F8", "D0")
    }
    for row in by_arm["D0"]:
        row["raw_delivered_impulse_mean_n_s_sampled"] = 0.10
    mechanism = _d0_mechanism(by_arm)
    assert mechanism["conditions"][
        "raw_delivered_upper_below_zero"
    ] is False
    assert mechanism["claim_allowed"] is False
    assert mechanism["interpretation"] != "mainly selected dwell/recontact"


def test_non_schema_v3_quality_artifact_invalidates_complete_analysis(
    tmp_path, complete_campaign
) -> None:
    rows, manifest = copy.deepcopy(complete_campaign)
    source = Path(rows[0]["sampled_trace_path"])
    with np.load(source, allow_pickle=False) as saved:
        payload = json.loads(str(saved["payload_json"]))
    payload["schema_version"] = 2
    payload_without_digest = dict(payload)
    payload_without_digest.pop("payload_digest")
    payload["payload_digest"] = _digest(payload_without_digest)
    target = tmp_path / "schema2.npz"
    np.savez_compressed(
        target,
        payload_json=np.asarray(
            json.dumps(payload, sort_keys=True, allow_nan=False), dtype=np.str_
        ),
    )
    rows[0]["sampled_trace_path"] = str(target)
    rows[0]["sampled_trace_digest"] = payload["payload_digest"]
    rows[0]["sampled_trace_artifact_sha256"] = hashlib.sha256(
        target.read_bytes()
    ).hexdigest()
    manifest[0]["sampled_trace_digest"] = rows[0]["sampled_trace_digest"]
    manifest[0]["sampled_trace_artifact_sha256"] = rows[0][
        "sampled_trace_artifact_sha256"
    ]
    invalid = analyze_quality_campaign(rows, manifest)
    assert invalid["valid"] is False
    assert any("schema v3" in reason for reason in invalid["invalidation_reasons"])


def test_contact_map_has_identical_nail_plane_axes() -> None:
    contacts = {
        arm: np.asarray([[0.001, 0.002], [-0.002, 0.001]])
        for arm in ("F8", "F0", "D0", "FQ-min")
    }
    figure = build_aggregate_nail_plane_contact_map(
        contacts, nail_radius_m=0.012
    )
    axes = figure.axes
    assert len(axes) == 4
    assert all(axis.get_aspect() == 1.0 for axis in axes)
    assert len({tuple(axis.get_xlim()) for axis in axes}) == 1
    assert len({tuple(axis.get_ylim()) for axis in axes}) == 1
    assert [axis.get_title() for axis in axes] == ["F8", "F0", "D0", "FQ-min"]


def test_renderer_writes_exactly_two_static_pngs(
    tmp_path, complete_campaign
) -> None:
    result = render_quality_figures(
        *complete_campaign, output_dir=tmp_path
    )
    paths = {Path(path).name: Path(path) for path in result["artifacts"]}
    assert set(paths) == {
        "paired_seed_effects.png",
        "aggregate_nail_plane_contact_map.png",
    }
    assert all(path.stat().st_size > 1_000 for path in paths.values())
    assert list(tmp_path.glob("*")) == list(paths.values())


def test_renderer_marks_unavailable_practical_ratios_and_still_writes_two_pngs(
    monkeypatch, tmp_path, analysis
) -> None:
    invalid_ratios = copy.deepcopy(analysis)
    practical = invalid_ratios["fq_min_practical_acceptance"]
    for key in ("useful_speed_ratio", "depth_gain_ratio"):
        practical[key].pop("estimate", None)
        practical[key]["valid"] = False
        practical[key]["one_sided_95_lower"] = None
    figure = quality_plot._build_paired_seed_effects(invalid_ratios)
    assert [text.get_text() for text in figure.axes[3].texts].count(
        "unavailable"
    ) == 2
    assert figure.axes[3].get_xlim() == pytest.approx((-0.5, 1.5))
    monkeypatch.setattr(
        quality_plot,
        "analyze_quality_campaign",
        lambda rows, manifest: invalid_ratios,
    )
    result = render_quality_figures([], [], output_dir=tmp_path)
    assert {Path(path).name for path in result["artifacts"]} == {
        "paired_seed_effects.png",
        "aggregate_nail_plane_contact_map.png",
    }
    assert len(list(tmp_path.glob("*.png"))) == 2
