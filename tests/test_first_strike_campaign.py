from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import evaluation.analysis.first_strike_campaign as first_strike_campaign
import evaluation.analysis.plot_first_strike_campaign as plot_first_strike_campaign
from evaluation.analysis.first_strike_campaign import (
    ARM_TASKS,
    CAMPAIGN_EVALUATOR_RNG,
    CAMPAIGN_EVALUATOR_SEED,
    EXPECTED_CONTROL_DECIMATION,
    EXPECTED_EVALUATION_EPISODE_LEN_S,
    EXPECTED_EVALUATION_MEAN_NSTEPS,
    EXPECTED_FIXED_ACTION_SIGNATURE,
    EXPECTED_FIXED_ACTION_SIGNATURE_SHA256,
    EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256,
    EXPECTED_PHYSICS_DT_S,
    FROZEN_CAMPAIGN_MATRIX,
    REQUIRED_SAMPLED_COLUMNS,
    aggregate_episode_metrics,
    analyze_campaign,
    exact_seed_tests,
    load_accepted_attempt_manifest,
    load_frozen_nail_geometry,
    summarize_episode,
    validate_frozen_campaign_matrix,
)
from evaluation.analysis.plot_first_strike_campaign import (
    build_campaign_figure,
    build_fixed_reset_xz_grid,
    build_trajectory_figure,
    generate_report,
    write_video_overlay_html,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET = (
    ROOT.parent
    / "safe_impact_manipulation"
    / "hammer_z1_env"
    / "assets"
    / "nail_block_scene.xml"
)

_SPEC = importlib.util.spec_from_file_location(
    "eval_impulse_script", ROOT / "scripts" / "eval_impulse.py"
)
assert _SPEC is not None and _SPEC.loader is not None
eval_impulse = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_impulse)


def _literal_digest(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compact_artifact_trace(
    *,
    arm: str,
    episode_index: int,
    useful_speed_target: float,
) -> dict:
    success = episode_index < 486
    window = 486 <= episode_index < 506
    no_contact = 506 <= episode_index < 511
    unfinished = episode_index == 511
    started = not no_contact
    finalized = success or window
    reason = "success" if success else ("window" if window else "none")
    reason_code = 1 if success else (2 if window else 0)
    delivered = 0.1 if success else (0.05 if window else (0.02 if unfinished else 0.0))
    success_speed = useful_speed_target * 512.0 / 486.0
    physical = {
        "contact": [False, started],
        "head_position_m": [[0.51, 0.0, 0.100], [0.51, 0.0, 0.098]],
        "clamped_depth_m": [0.0, 0.0],
        "net_axial_force_n": [0.0, 50.0 if started else 0.0],
        "joint_speed_rad_s": [[0.0] * 6, [0.0] * 6],
        "post_step_joint_speed_rad_s": [[0.0] * 6, [0.0] * 6],
        "tracker_depth_post_integration_m": [0.0, 0.0],
    }
    event_trace = {
        "tracker_started": [False, started],
        "tracker_finalized": [False, finalized],
        "tracker_productive": [False, finalized],
        "tracker_reason": [0, reason_code],
        "event_cumulative_impulse_n_s": [0.0, delivered],
    }
    trace_digest = _literal_digest(
        {"physical": physical, "event_trace": event_trace}
    )
    trace = {
        "episode_id": f"{arm}-episode-{episode_index}",
        "env_id": episode_index % 256,
        "episode_ordinal": episode_index // 256,
        "arm": arm,
        "task": ARM_TASKS[arm],
        "trace_digest": trace_digest,
        "physics_dt_s": 0.002,
        "phase_contract": {
            "physical_channels": "pre_integration",
            "tracker_derived_channels": "pre_integration",
            "tracker_nail_depth": "post_integration",
            "tracker_depth_lead_substeps": 1,
        },
        "nail_geometry": {
            "nail_axis": [0.0, 0.0, -1.0],
            "nail_xy_m": [0.5, 0.0],
            "nail_radius_m": 0.012,
            "source_sha256": "c" * 64,
        },
        "physical": physical,
        "event_trace": event_trace,
        "first_strike": {
            "started": started,
            "finalized": finalized,
            "reason": reason,
            "accepted_onset_index": 1 if started else None,
            "productive": finalized,
            "v_precontact_m_s": success_speed if success else 0.0,
            "delivered_n_s": delivered,
            "saturated": False,
        },
        "overall_success": episode_index < 492,
        "episode_peak_lambda": [0.1] * 6 if started else [0.0] * 6,
        "episode_delivered_accumulator_n_s": delivered,
        "episode_depth_m": 0.032 if episode_index < 492 else 0.0,
        "reward": {
            "gamma": 0.99,
            "impact_payout": [0.0],
            "delivered_payout": [0.0],
        },
    }
    return trace


def _literal_sampled_aggregate(useful_speed_target: float) -> dict:
    success_speed = useful_speed_target * 512.0 / 486.0
    return {
        "n_episodes_sampled": 512,
        "first_strike_useful_speed_mean_sampled": useful_speed_target,
        "successful_first_strike_v_precontact_mean_sampled": success_speed,
        "first_strike_success_rate_sampled": 486 / 512,
        "recontact_rate_sampled": 0.0,
        "first_strike_v_precontact_mean_sampled": 486 * success_speed / 507,
        "first_strike_productive_rate_sampled": 506 / 512,
        "first_strike_delivered_success_mean_sampled": 0.1,
        "first_strike_delivered_window_mean_sampled": 0.05,
        "first_strike_success_n_sampled": 486,
        "first_strike_window_n_sampled": 20,
        "first_strike_no_contact_n_sampled": 5,
        "first_strike_unfinished_event_n_sampled": 1,
        "tail_fraction_mean_sampled": 0.0,
        "maximize_return_discounted_mean_sampled": 0.0,
        "impact_return_discounted_mean_sampled": 0.0,
        "delivered_return_discounted_mean_sampled": 0.0,
        "first_strike_saturation_rate_sampled": 0.0,
        "overall_success_rate_sampled": 492 / 512,
        "qvel_violation_rate_sampled": 0.0,
        "qvel_finite_exceedance_rate_sampled": 0.0,
        "qvel_nonfinite_rate_sampled": 0.0,
        "qvel_max_abs_rad_s_sampled": 0.0,
        "qvel_max_excess_rad_s_sampled": 0.0,
        "qvel_samples_over_rail_sampled": 0,
        "qvel_seconds_over_rail_sampled": 0.0,
        "qvel_peak_joint_index_sampled": 0,
        "qvel_precontact_exceedance_rate_sampled": 0.0,
        "qvel_precontact_samples_over_rail_sampled": 0,
        "qvel_precontact_seconds_over_rail_sampled": 0.0,
        "qvel_first_event_exceedance_rate_sampled": 0.0,
        "qvel_first_event_samples_over_rail_sampled": 0,
        "qvel_first_event_seconds_over_rail_sampled": 0.0,
        "qvel_post_event_exceedance_rate_sampled": 0.0,
        "qvel_post_event_samples_over_rail_sampled": 0,
        "qvel_post_event_seconds_over_rail_sampled": 0.0,
        "precontact_path_length_ratio_mean_sampled": 1.0,
        "precontact_lateral_excursion_mean_sampled": 0.01,
        "contact_approach_angle_mean_sampled": 0.0,
        "first_contact_transverse_error_mean_sampled": 0.01,
        "first_contact_within_nail_radius_rate_sampled": 507 / 512,
        "first_contact_geometry_n_sampled": 507,
        "terminal_contraction_ratio_mean_sampled": 0.0,
        "late_lateral_reexpansion_rate_sampled": 0.0,
        "terminal_funnel_eligible_n_sampled": 0,
    }


def _write_literal_sampled_artifact(
    path: Path,
    *,
    arm: str,
    seed: int,
    useful_speed_target: float,
) -> dict:
    identity = f"{arm}-seed{seed}"
    checkpoint_digest = hashlib.sha256(identity.encode()).hexdigest()
    event_i_ref = 0.6094 if arm in {"C", "D-prime"} else 0.3088
    contract = {
        "treatment": arm,
        "impact_weight": 8.0,
        "delivered_weight": 2.0,
        "event_i_ref_n_s": event_i_ref,
    }
    treatment_config_sha256 = eval_impulse._treatment_config_digest(
        task=ARM_TASKS[arm], contract=contract
    )
    episodes = [
        _compact_artifact_trace(
            arm=arm,
            episode_index=index,
            useful_speed_target=useful_speed_target,
        )
        for index in range(512)
    ]
    payload = {
        "schema_version": 2,
        "selection": "first two completed episodes from each of 256 environments",
        "expected_episode_count": 512,
        "treatment": arm,
        "task": ARM_TASKS[arm],
        "training_seed": seed,
        "weights": {"impact_progress": 8.0, "delivered_impulse": 2.0},
        "event_i_ref_n_s": event_i_ref,
        "impulse_limits_n_m_s": [1.64, 3.28, 1.64, 1.64, 1.64, 1.64],
        "imp_max_p": 0.0,
        "rng_streams": {
            "reset": CAMPAIGN_EVALUATOR_RNG["reset"],
            "observation": CAMPAIGN_EVALUATOR_RNG["observation"],
            "action": CAMPAIGN_EVALUATOR_RNG["action"],
        },
        "evaluation_contract": {
            "base_rng_seed": CAMPAIGN_EVALUATOR_SEED,
            "num_envs": 256,
            "episodes_per_env": 2,
            "episode_len_s": EXPECTED_EVALUATION_EPISODE_LEN_S,
            "mean_nsteps": EXPECTED_EVALUATION_MEAN_NSTEPS,
            "completion_rule": "first_two_completions_per_environment",
            "stochastic_actions": True,
            "reset_position_noise_rad": [-0.05, 0.05],
            "actor_observation_corruption": True,
            "critic_observation_corruption": False,
            "physics_dt_s": EXPECTED_PHYSICS_DT_S,
            "control_decimation": EXPECTED_CONTROL_DECIMATION,
            "fixed_impedance_signature_sha256": (
                EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256
            ),
            "fixed_action_signature_sha256": (
                EXPECTED_FIXED_ACTION_SIGNATURE_SHA256
            ),
        },
        "nail_geometry": {
            "nail_axis": [0.0, 0.0, -1.0],
            "nail_xy_m": [0.5, 0.0],
            "nail_radius_m": 0.012,
            "source_sha256": "c" * 64,
        },
        "provenance": {
            "code_git": {"revision": "a" * 40, "dirty": False, "status": ""},
            "asset_git": {"revision": "b" * 40, "dirty": False, "status": ""},
            "checkpoint_sha256": checkpoint_digest,
            "accepted_checkpoint_sha256": checkpoint_digest,
            "campaign_config_sha256": "d" * 64,
            "treatment_config_sha256": treatment_config_sha256,
            "nail_asset_sha256": "c" * 64,
            "accepted_manifest_sha256": "e" * 64,
            "training_code_revision": "a" * 40,
            "training_asset_revision": "b" * 40,
        },
        "mean_rollout_invariants": {
            "impossible_success_n": 0,
            "lambda_dead_n": 0,
        },
        "episodes": episodes,
    }
    digest = _literal_digest(payload)
    payload["payload_digest"] = digest
    np.savez_compressed(
        path,
        payload_json=np.asarray(
            json.dumps(payload, sort_keys=True, allow_nan=False), dtype=np.str_
        ),
    )
    return {
        "sampled_trace_path": str(path),
        "sampled_trace_digest": digest,
        "sampled_trace_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_sha256": payload["provenance"]["checkpoint_sha256"],
        "accepted_checkpoint_sha256": checkpoint_digest,
        "treatment_config_sha256": treatment_config_sha256,
        "event_i_ref_n_s": event_i_ref,
        "legacy_sampled": {
            "success_rate_sampled": 492 / 512,
            "worst_ratio_max_sampled": 0.1 / 1.64,
            "delivered_mean_sampled": 49.62 / 512,
        },
        "sampled_aggregate": _literal_sampled_aggregate(useful_speed_target),
    }


def _literal_trace(
    *,
    arm: str = "E",
    reason: str = "success",
    overall_success: bool = True,
    no_contact: bool = False,
    unfinished: bool = False,
    reexpanded: bool = False,
) -> dict:
    # First contact is sample 30, giving exact complete [-60, 0] ms support.
    # The first 31 points form a straight x-z segment toward the frozen nail axis.
    errors = [0.030 - 0.0008 * i for i in range(31)]
    if reexpanded:
        errors[27] = 0.020  # -6 ms: above 1.25 * the 12 mm nail-head radius.
    positions = [
        [0.5 + errors[i], 0.0, 0.160 - 0.002 * i] for i in range(31)
    ]
    positions.extend(
        [
            [0.5055, 0.0, 0.098],
            [0.5050, 0.0, 0.097],
            [0.5045, 0.0, 0.096],
            [0.5040, 0.0, 0.095],
        ]
    )
    contact = [False] * 35
    force = [0.0] * 35
    if not no_contact:
        contact[30] = contact[31] = contact[34] = True
        force[30] = force[31] = 2.0
        force[34] = 6.0
    qvel = [[0.0] * 6 for _ in range(35)]
    qvel[10][1] = 3.2
    qvel_post = [qvel[index + 1][:] for index in range(34)]
    qvel_post.append(qvel[-1][:])
    depth = [0.0] * 35
    if not no_contact:
        depth[30] = 0.020
        depth[31:] = [0.031] * 4

    started = not no_contact
    final_reason = "none" if no_contact or unfinished else reason
    productive = started and not unfinished and reason in {"success", "window"}
    first_delivered = 0.0 if no_contact else (0.3088 if reason == "success" else 0.1)
    trace = {
        "episode_id": f"{arm}-{reason}-{'none' if no_contact else 'contact'}",
        "env_id": 0,
        "episode_ordinal": 0,
        "arm": arm,
        "task": ARM_TASKS[arm],
        "trace_digest": "",
        "physics_dt_s": 0.002,
        "phase_contract": {
            "physical_channels": "pre_integration",
            "tracker_derived_channels": "pre_integration",
            "tracker_nail_depth": "post_integration",
            "tracker_depth_lead_substeps": 1,
        },
        "nail_geometry": {
            "nail_axis": [0.0, 0.0, -1.0],
            "nail_xy_m": [0.5, 0.0],
            "nail_radius_m": 0.012,
            "source_sha256": "frozen-fixture-asset",
        },
        "physical": {
            "contact": contact,
            "head_position_m": positions,
            "clamped_depth_m": depth,
            "net_axial_force_n": force,
            "joint_speed_rad_s": qvel,
            "post_step_joint_speed_rad_s": qvel_post,
            "tracker_depth_post_integration_m": depth[:],
        },
        "event_trace": {
            "tracker_started": [
                False if no_contact or index < 30 else True for index in range(35)
            ],
            "tracker_finalized": [
                False
                if no_contact
                or unfinished
                or index < (31 if reason == "success" else 34)
                else True
                for index in range(35)
            ],
            "tracker_productive": [
                bool(productive and index >= (31 if reason == "success" else 34))
                for index in range(35)
            ],
            "tracker_reason": [
                0
                if no_contact
                or unfinished
                or index < (31 if reason == "success" else 34)
                else (1 if reason == "success" else 2)
                for index in range(35)
            ],
            "event_cumulative_impulse_n_s": [
                0.0
                if no_contact or index < 30
                else (
                    first_delivered
                    if index >= (31 if reason == "success" else 34)
                    else first_delivered / 2.0
                )
                for index in range(35)
            ],
        },
        "first_strike": {
            "started": started,
            "finalized": started and not unfinished,
            "reason": final_reason,
            "accepted_onset_index": None if no_contact else 30,
            "productive": productive,
            "v_precontact_m_s": 1.0 if started else 0.0,
            "delivered_n_s": first_delivered,
            "saturated": first_delivered >= 0.3088 if started else False,
        },
        "overall_success": overall_success,
        # Already weighted once by RewardManager; aggregation must not multiply by 8/2.
        "reward": {
            "gamma": 0.99,
            "impact_payout": [0.0, 0.8],
            "delivered_payout": [0.0, 0.2],
        },
    }
    trace["trace_digest"] = _literal_digest(
        {
            "physical": trace["physical"],
            "event_trace": trace["event_trace"],
        }
    )
    return trace


def _schema_v3_quality_trace() -> dict:
    trace = _literal_trace()
    trace["physical"].update(
        {
            "quality_found_count": [[0] * 8, [1, 7, 0, 0, 0, 0, 0, 0]],
            "quality_normal_force_n": [[0.0] * 8, [3.0] * 8],
            "quality_contact_position_m": [[[0.0] * 3] * 8, [[0.5, 0.0, 0.032]] * 8],
            "quality_contact_normal": [[[0.0] * 3] * 8, [[0.0, 0.0, -1.0]] * 8],
        }
    )
    trace["event_trace"].update(
        {
            "tracker_contact_point_w": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.032]],
            "tracker_contact_error_m": [0.0, 0.001],
            "tracker_contact_quality": [0.0, 0.95],
            "tracker_contact_quality_valid": [False, True],
            "tracker_contact_quality_overflow": [False, False],
            "tracker_first_contact_time_s": [0.0, 0.004],
            "tracker_contact_normal_axiality": [0.0, 1.0],
            "event_cumulative_transverse_impulse_n_s": [0.0, 0.002],
        }
    )
    trace["first_strike"].update(
        {
            "contact_point_w": [0.5, 0.0, 0.032],
            "contact_error_m": 0.001,
            "contact_quality": 0.95,
            "contact_quality_valid": True,
            "contact_quality_overflow": False,
            "first_contact_time_s": 0.004,
            "contact_normal_axiality": 1.0,
            "delivered_transverse_n_s": 0.002,
        }
    )
    trace.update(
        {
            "episode_peak_lambda": [0.1] * 6,
            "episode_delivered_accumulator_n_s": 0.1,
            "episode_depth_m": 0.032,
        }
    )
    trace["trace_digest"] = eval_impulse._physical_trace_digest(trace)
    return trace


def test_quality_found_slots_are_raw_and_digest_sensitive():
    trace = _schema_v3_quality_trace()
    assert trace["physical"]["quality_found_count"][1] == [1, 7, 0, 0, 0, 0, 0, 0]
    baseline = eval_impulse._physical_trace_digest(trace)
    trace["physical"]["quality_found_count"][1][1] = 1
    assert eval_impulse._physical_trace_digest(trace) != baseline


def test_persistence_rejects_nonfinite_quality_geometry(tmp_path):
    trace = _schema_v3_quality_trace()
    trace["physical"]["quality_contact_position_m"][1][0][0] = None
    with pytest.raises(ValueError, match="nonfinite physical trace value"):
        eval_impulse._persist_sampled_traces(
            out_dir=tmp_path, name="invalid", sampled_rec={"control_steps": 1, "episodes": [trace]},
            task=ARM_TASKS["E"], contract={"treatment": "E", "impact_weight": 8.0,
            "delivered_weight": 2.0, "event_i_ref_n_s": 0.3088,
            "impulse_limits_n_m_s": [1.64] * 6}, training_seed=0, reset_seed=1,
            observation_seed=2, action_seed=3, nail_geometry={}, provenance={"checkpoint_sha256": "e" * 64},
            mean_rollout_invariants={},
        )


def test_persistence_rejects_overflowed_quality_marked_valid(tmp_path):
    trace = _schema_v3_quality_trace()
    trace["event_trace"]["tracker_contact_quality_overflow"][-1] = True
    trace["first_strike"]["contact_quality_overflow"] = True
    with pytest.raises(ValueError, match="overflowed quality"):
        eval_impulse._persist_sampled_traces(
            out_dir=tmp_path, name="invalid", sampled_rec={"control_steps": 1, "episodes": [trace]},
            task=ARM_TASKS["E"], contract={"treatment": "E", "impact_weight": 8.0,
            "delivered_weight": 2.0, "event_i_ref_n_s": 0.3088,
            "impulse_limits_n_m_s": [1.64] * 6}, training_seed=0, reset_seed=1,
            observation_seed=2, action_seed=3, nail_geometry={}, provenance={"checkpoint_sha256": "e" * 64},
            mean_rollout_invariants={},
        )


def test_action_tape_physics_requires_all_quality_channels_and_rejects_drift():
    """A missing or changed passive quality sample must invalidate replay identity."""
    trace = _literal_trace()
    trace["physical"].update(
        {
            "quality_found_count": [0, 1],
            "quality_normal_force_n": [[0.0], [3.0]],
            "quality_contact_position_m": [
                [[0.0, 0.0, 0.0]],
                [[0.50, 0.0, 0.032]],
            ],
            "quality_contact_normal": [
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, -1.0]],
            ],
        }
    )
    trace["event_trace"].update(
        {
            "tracker_contact_point_w": [[0.0, 0.0, 0.0], [0.50, 0.0, 0.032]],
            "tracker_contact_error_m": [0.0, 0.001],
            "tracker_contact_quality": [0.0, 0.95],
            "tracker_contact_quality_valid": [False, True],
            "tracker_contact_quality_overflow": [False, False],
            "tracker_first_contact_time_s": [0.0, 0.004],
            "tracker_contact_normal_axiality": [0.0, 1.0],
            "event_cumulative_transverse_impulse_n_s": [0.0, 0.002],
        }
    )
    trace["first_strike"].update(
        {
            "contact_point_w": [0.50, 0.0, 0.032],
            "contact_error_m": 0.001,
            "contact_quality": 0.95,
            "contact_quality_valid": True,
            "contact_quality_overflow": False,
            "first_contact_time_s": 0.004,
            "contact_normal_axiality": 1.0,
            "delivered_transverse_n_s": 0.002,
        }
    )
    trace.update(
        {
            "episode_peak_lambda": [0.1] * 6,
            "episode_delivered_accumulator_n_s": 0.1,
            "episode_depth_m": 0.032,
        }
    )
    trace["trace_digest"] = eval_impulse._physical_trace_digest(trace)
    traces = {arm: copy.deepcopy(trace) for arm in ("F8", "F0", "D0", "FQ")}

    result = eval_impulse.compare_action_tape_physics(traces)

    assert result["physical_digest"] == trace["trace_digest"]
    assert result["arms"] == ("D0", "F0", "F8", "FQ")

    traces["FQ"]["event_trace"]["tracker_contact_quality"][1] = 0.50
    traces["FQ"]["first_strike"]["contact_quality"] = 0.50
    traces["FQ"]["trace_digest"] = eval_impulse._physical_trace_digest(traces["FQ"])
    with pytest.raises(ValueError, match="physical/event replay drift"):
        eval_impulse.compare_action_tape_physics(traces)


def test_action_tape_physics_rejects_overflowed_quality_as_valid():
    """Overflow is a failed quality measurement, never a valid onset snapshot."""
    trace = _literal_trace()
    trace["physical"].update(
        {
            "quality_found_count": [0, 9],
            "quality_normal_force_n": [[0.0], [3.0]],
            "quality_contact_position_m": [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.032]]],
            "quality_contact_normal": [[[0.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]]],
        }
    )
    trace["event_trace"].update(
        {
            "tracker_contact_point_w": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.032]],
            "tracker_contact_error_m": [0.0, 0.001],
            "tracker_contact_quality": [0.0, 0.95],
            "tracker_contact_quality_valid": [False, True],
            "tracker_contact_quality_overflow": [False, True],
            "tracker_first_contact_time_s": [0.0, 0.004],
            "tracker_contact_normal_axiality": [0.0, 1.0],
            "event_cumulative_transverse_impulse_n_s": [0.0, 0.002],
        }
    )
    trace["first_strike"].update(
        {
            "contact_point_w": [0.5, 0.0, 0.032],
            "contact_error_m": 0.001,
            "contact_quality": 0.95,
            "contact_quality_valid": True,
            "contact_quality_overflow": True,
            "first_contact_time_s": 0.004,
            "contact_normal_axiality": 1.0,
            "delivered_transverse_n_s": 0.002,
        }
    )
    trace.update(
        {
            "episode_peak_lambda": [0.1] * 6,
            "episode_delivered_accumulator_n_s": 0.1,
            "episode_depth_m": 0.032,
        }
    )
    trace["trace_digest"] = eval_impulse._physical_trace_digest(trace)

    with pytest.raises(ValueError, match="overflowed quality"):
        eval_impulse.compare_action_tape_physics({"F8": trace})


@pytest.mark.parametrize("arm", ("F8", "F0", "D0", "FQ"))
def test_strict_quality_evaluation_config_preserves_policy_and_treatment(arm):
    """The passive sensor may alter only evaluation instrumentation metadata."""
    task = eval_impulse.QUALITY_ARM_TASKS[arm]

    training_cfg, evaluation_cfg, identities = (
        eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
    )

    assert "hammer_nail_quality" not in {
        sensor.name for sensor in (training_cfg.scene.sensors or ())
    } if arm != "FQ" else True
    assert "hammer_nail_quality" in {
        sensor.name for sensor in (evaluation_cfg.scene.sensors or ())
    }
    assert identities["training_config_sha256"]
    assert identities["evaluation_config_sha256"]
    if arm != "FQ":
        assert identities["training_config_sha256"] != identities["evaluation_config_sha256"]
    assert (
        identities["training_policy_observation_sha256"]
        == identities["evaluation_policy_observation_sha256"]
    )
    assert (
        identities["training_treatment_reward_sha256"]
        == identities["evaluation_treatment_reward_sha256"]
    )


def _campaign_rows(tmp_path: Path) -> list[dict]:
    values = {
        "C": 0.9,
        "D-prime": 1.0,
        "F": 1.1,
        "E": 1.2,
    }
    rows = []
    for arm, value in values.items():
        for seed in range(8):
            identity = f"{arm}-seed{seed}"
            artifact = _write_literal_sampled_artifact(
                tmp_path / f"{identity}.npz",
                arm=arm,
                seed=seed,
                useful_speed_target=value,
            )
            sampled_defaults = {
                field: 0.0 for field in REQUIRED_SAMPLED_COLUMNS
            }
            _, _, slug = _TASK5_ARM_CONTRACT[arm]
            run_name = f"fsr4x8_{slug}_seed{seed}"
            rows.append(
                {
                    **sampled_defaults,
                    "name": run_name,
                    "treatment": arm,
                    "task": ARM_TASKS[arm],
                    "training_seed": seed,
                    "n_episodes_sampled": 512,
                    "num_envs": 256,
                    "episodes_per_env_sampled": 2,
                    "impact_weight": 8.0,
                    "delivered_weight": 2.0,
                    "event_i_ref_n_s": artifact["event_i_ref_n_s"],
                    "imp_max_p": 0.0,
                    "git_hash": "a" * 40,
                    "git_revision": "a" * 40,
                    "git_dirty": False,
                    "asset_git_hash": "b" * 40,
                    "asset_git_revision": "b" * 40,
                    "asset_git_dirty": False,
                    "nail_asset_sha256": "c" * 64,
                    "campaign_config_sha256": "d" * 64,
                    "accepted_manifest_sha256": "e" * 64,
                    "training_code_revision": "a" * 40,
                    "training_asset_revision": "b" * 40,
                    "checkpoint_path": (
                        "/checkpoints/"
                        f"2026-07-25_12-00-00_{run_name}/model_499.pt"
                    ),
                    **artifact,
                    "seed": CAMPAIGN_EVALUATOR_SEED,
                    "action_rng_seed": CAMPAIGN_EVALUATOR_RNG["action"],
                    "reset_rng_seed": CAMPAIGN_EVALUATOR_RNG["reset"],
                    "observation_rng_seed": CAMPAIGN_EVALUATOR_RNG[
                        "observation"
                    ],
                    "episode_len_s": EXPECTED_EVALUATION_EPISODE_LEN_S,
                    "nsteps": EXPECTED_EVALUATION_MEAN_NSTEPS,
                    "reset_position_noise_min_rad": -0.05,
                    "reset_position_noise_max_rad": 0.05,
                    "actor_observation_corruption": True,
                    "critic_observation_corruption": False,
                    "sampled_completion_rule": (
                        "first_two_completions_per_environment"
                    ),
                    "sampled_actions_stochastic": True,
                    "physics_dt_s": EXPECTED_PHYSICS_DT_S,
                    "control_decimation": EXPECTED_CONTROL_DECIMATION,
                    "fixed_impedance_signature_sha256": (
                        EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256
                    ),
                    "fixed_action_signature_sha256": (
                        EXPECTED_FIXED_ACTION_SIGNATURE_SHA256
                    ),
                    "impossible_success_n": 0,
                    "lambda_dead_n": 0,
                    **artifact["legacy_sampled"],
                    **artifact["sampled_aggregate"],
                }
            )
    return rows


@pytest.fixture(scope="module")
def campaign_rows_fixture(tmp_path_factory):
    return _campaign_rows(tmp_path_factory.mktemp("task4_campaign"))


def _artifact_representatives(rows: list[dict]) -> list[dict]:
    return first_strike_campaign.select_representative_traces(rows)["traces"]


def _rewrite_campaign_artifacts(
    rows: list[dict],
    *,
    treatment: str,
    output_dir: Path,
    transform,
) -> None:
    for row in rows:
        if row["treatment"] != treatment:
            continue
        with np.load(row["sampled_trace_path"], allow_pickle=False) as saved:
            payload = json.loads(str(saved["payload_json"]))
        for trace in payload["episodes"]:
            transform(trace)
        payload_without_digest = dict(payload)
        payload_without_digest.pop("payload_digest")
        payload["payload_digest"] = _literal_digest(payload_without_digest)
        path = output_dir / f"{row['name']}-rewritten.npz"
        np.savez_compressed(
            path,
            payload_json=np.asarray(
                json.dumps(payload, sort_keys=True, allow_nan=False),
                dtype=np.str_,
            ),
        )
        episode_metrics = [
            summarize_episode(trace, nail_geometry=payload["nail_geometry"])
            for trace in payload["episodes"]
        ]
        row.update(
            aggregate_episode_metrics(
                episode_metrics, expected_episode_count=512
            )
        )
        peaks = np.asarray(
            [
                trace["episode_peak_lambda"]
                for trace in payload["episodes"]
            ],
            dtype=float,
        )
        delivered = np.asarray(
            [
                trace["episode_delivered_accumulator_n_s"]
                for trace in payload["episodes"]
            ],
            dtype=float,
        )
        success = np.asarray(
            [trace["overall_success"] for trace in payload["episodes"]],
            dtype=bool,
        )
        lam_worst = peaks.max(axis=1)
        limits = np.asarray(payload["impulse_limits_n_m_s"], dtype=float)
        row.update(
            {
                "success_rate_sampled": float(success.mean()),
                "worst_ratio_max_sampled": float(
                    (peaks / limits).max()
                ),
                "delivered_mean_sampled": float(delivered.mean()),
                "impossible_success_n": int(
                    payload["mean_rollout_invariants"][
                        "impossible_success_n"
                    ]
                    + np.count_nonzero(
                        success
                        & ((delivered <= 0.0) | (lam_worst <= 0.0))
                    )
                ),
                "lambda_dead_n": int(
                    payload["mean_rollout_invariants"]["lambda_dead_n"]
                    + np.count_nonzero(
                        (delivered > 0.0) & (lam_worst <= 0.0)
                    )
                ),
            }
        )
        row["sampled_trace_path"] = str(path)
        row["sampled_trace_digest"] = payload["payload_digest"]
        row["sampled_trace_artifact_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_frozen_nail_geometry_comes_from_the_scene_asset():
    geometry = load_frozen_nail_geometry(ASSET)

    assert geometry["nail_axis"] == [0.0, 0.0, -1.0]
    assert geometry["nail_xy_m"] == [0.5, 0.0]
    assert geometry["nail_radius_m"] == pytest.approx(0.012)
    assert geometry["source_path"] == str(ASSET)
    assert len(geometry["source_sha256"]) == 64


def test_literal_episode_uses_one_trace_for_first_window_tail_and_geometry():
    metrics = summarize_episode(
        _literal_trace(), nail_geometry=load_frozen_nail_geometry(ASSET)
    )

    assert metrics["first_strike_success"] is True
    assert metrics["first_strike_productive"] is True
    assert metrics["recontact"] is True
    assert metrics["tail_fraction"] == pytest.approx(0.6)
    assert metrics["qvel_violation"] is True
    assert metrics["first_strike_useful_speed_m_s"] == pytest.approx(1.0)
    assert metrics["precontact_path_length_ratio"] == pytest.approx(1.0)
    assert metrics["precontact_lateral_excursion_m"] == pytest.approx(0.030)
    assert metrics["contact_approach_angle_deg"] == pytest.approx(
        np.degrees(np.arctan2(0.4, 1.0))
    )
    assert metrics["first_contact_transverse_error_m"] == pytest.approx(0.006)
    assert metrics["first_contact_within_nail_radius"] is True
    assert metrics["terminal_contraction_ratio"] == pytest.approx(0.006 / 0.022)
    assert metrics["late_lateral_reexpansion"] is False
    assert metrics["maximize_return_discounted"] == pytest.approx(0.99)
    assert metrics["impact_return_discounted"] == pytest.approx(0.792)
    assert metrics["delivered_return_discounted"] == pytest.approx(0.198)
    assert metrics["trace_digest"] == _literal_trace()["trace_digest"]


def test_contact_gap_inside_tracker_event_is_not_recontact_or_tail():
    metrics = summarize_episode(
        _literal_trace(reason="window", overall_success=False),
        nail_geometry=load_frozen_nail_geometry(ASSET),
    )

    # Raw contact 34 follows a gap after contact 31, but the tracker event does
    # not finalize until 34. It is still first-window mass, not recontact/tail.
    assert metrics["recontact"] is False
    assert metrics["tail_fraction"] == 0.0


def test_no_contact_and_terminal_geometry_denominators_are_explicit():
    geometry = load_frozen_nail_geometry(ASSET)
    rows = [
        summarize_episode(_literal_trace(), nail_geometry=geometry),
        summarize_episode(
            _literal_trace(no_contact=True, overall_success=False),
            nail_geometry=geometry,
        ),
        summarize_episode(
            _literal_trace(unfinished=True, overall_success=False),
            nail_geometry=geometry,
        ),
    ]

    summary = aggregate_episode_metrics(rows, expected_episode_count=3)

    assert summary["first_contact_geometry_n_sampled"] == 2
    assert summary["first_contact_transverse_error_mean_sampled"] == pytest.approx(
        0.006
    )
    # No contact is false in this all-episode rate: 2 within / 3 total.
    assert summary["first_contact_within_nail_radius_rate_sampled"] == pytest.approx(
        2 / 3
    )
    assert summary["terminal_funnel_eligible_n_sampled"] == 2
    assert summary["terminal_contraction_ratio_mean_sampled"] == pytest.approx(
        0.006 / 0.022
    )
    assert summary["late_lateral_reexpansion_rate_sampled"] == 0.0
    assert summary["first_strike_no_contact_n_sampled"] == 1
    assert summary["first_strike_unfinished_event_n_sampled"] == 1
    assert (
        summary["first_strike_success_n_sampled"]
        + summary["first_strike_window_n_sampled"]
        + summary["first_strike_no_contact_n_sampled"]
        + summary["first_strike_unfinished_event_n_sampled"]
        == 3
    )
    assert summary["first_strike_v_precontact_mean_sampled"] == pytest.approx(1.0)
    assert summary["first_strike_useful_speed_mean_sampled"] == pytest.approx(1 / 3)


def test_tracker_accepted_onset_rejects_prearming_raw_contact():
    trace = _literal_trace()
    trace["physical"]["contact"][0] = True
    trace["physical"]["net_axial_force_n"][0] = 100.0

    with pytest.raises(ValueError, match="raw contact before tracker acceptance"):
        summarize_episode(
            trace, nail_geometry=load_frozen_nail_geometry(ASSET)
        )


def test_raw_contact_without_tracker_accepted_onset_is_invalid():
    trace = _literal_trace(no_contact=True, overall_success=False)
    trace["physical"]["contact"][0:3] = [True, True, True]
    trace["physical"]["net_axial_force_n"][0:3] = [1.0, 1.0, 1.0]

    with pytest.raises(ValueError, match="raw contact before tracker acceptance"):
        summarize_episode(
            trace, nail_geometry=load_frozen_nail_geometry(ASSET)
        )


def test_tracker_phase_contract_exposes_the_one_substep_depth_lead():
    metrics = summarize_episode(
        _literal_trace(), nail_geometry=load_frozen_nail_geometry(ASSET)
    )

    assert metrics["physical_trace_phase"] == "pre_integration"
    assert metrics["tracker_depth_lead_substeps"] == 1
    assert metrics["tracker_depth_lead_s"] == pytest.approx(0.002)


def test_late_lateral_reexpansion_uses_the_frozen_terminal_window():
    metrics = summarize_episode(
        _literal_trace(reexpanded=True),
        nail_geometry=load_frozen_nail_geometry(ASSET),
    )

    assert metrics["terminal_funnel_eligible"] is True
    assert metrics["late_lateral_reexpansion"] is True


def test_late_reexpansion_is_independent_of_mutually_exclusive_funnel_label():
    trace = _literal_trace(reexpanded=True)
    # End off target; the frozen classifier calls this off_target before considering
    # its label branch, but the campaign's re-expansion diagnostic remains true.
    trace["physical"]["head_position_m"][30][0] = 0.520

    metrics = summarize_episode(
        trace, nail_geometry=load_frozen_nail_geometry(ASSET)
    )

    assert metrics["first_contact_within_nail_radius"] is False
    assert metrics["late_lateral_reexpansion"] is True


def test_full_precontact_path_metrics_include_motion_before_terminal_window():
    trace = _literal_trace()
    physical = trace["physical"]
    physical["contact"].insert(0, False)
    physical["net_axial_force_n"].insert(0, 0.0)
    physical["clamped_depth_m"].insert(0, 0.0)
    physical["joint_speed_rad_s"].insert(0, [0.0] * 6)
    physical["post_step_joint_speed_rad_s"].insert(0, [0.0] * 6)
    physical["tracker_depth_post_integration_m"].insert(0, 0.0)
    physical["head_position_m"].insert(0, [0.5, 0.04, 0.18])
    for key in (
        "tracker_started",
        "tracker_finalized",
        "tracker_productive",
    ):
        trace["event_trace"][key].insert(0, False)
    trace["event_trace"]["tracker_reason"].insert(0, 0)
    trace["event_trace"]["event_cumulative_impulse_n_s"].insert(0, 0.0)
    trace["first_strike"]["accepted_onset_index"] = 31

    metrics = summarize_episode(
        trace, nail_geometry=load_frozen_nail_geometry(ASSET)
    )

    first_segment = np.sqrt(0.03**2 + 0.04**2 + 0.02**2)
    terminal_segment = np.sqrt(0.024**2 + 0.06**2)
    endpoint_distance = np.sqrt(0.006**2 + 0.04**2 + 0.08**2)
    assert metrics["precontact_path_length_ratio"] == pytest.approx(
        (first_segment + terminal_segment) / endpoint_distance
    )
    assert metrics["precontact_lateral_excursion_m"] == pytest.approx(0.04)

    # These full-precontact metrics are contact-conditioned, not restricted to
    # episodes with complete terminal-funnel support.
    early = _literal_trace()
    early["physical"]["contact"] = [False] * 35
    early["physical"]["contact"][5:7] = [True, True]
    early["physical"]["net_axial_force_n"] = [0.0] * 35
    early["physical"]["net_axial_force_n"][5:7] = [2.0, 2.0]
    early["first_strike"]["accepted_onset_index"] = 5
    early["event_trace"]["tracker_started"] = [
        index >= 5 for index in range(35)
    ]
    early["event_trace"]["tracker_finalized"] = [
        index >= 6 for index in range(35)
    ]
    early["event_trace"]["tracker_productive"] = [
        index >= 6 for index in range(35)
    ]
    early["event_trace"]["tracker_reason"] = [
        1 if index >= 6 else 0 for index in range(35)
    ]
    early["event_trace"]["event_cumulative_impulse_n_s"] = [
        0.0 if index < 5 else (0.1544 if index == 5 else 0.3088)
        for index in range(35)
    ]

    early_metrics = summarize_episode(
        early, nail_geometry=load_frozen_nail_geometry(ASSET)
    )

    assert early_metrics["terminal_funnel_eligible"] is False
    assert early_metrics["precontact_path_length_ratio"] == pytest.approx(1.0)
    assert early_metrics["precontact_lateral_excursion_m"] == pytest.approx(0.03)
    assert early_metrics["contact_approach_angle_deg"] == pytest.approx(
        np.degrees(np.arctan2(0.4, 1.0))
    )


def test_nonfinite_500hz_joint_speed_is_a_fail_closed_violation():
    trace = _literal_trace()
    trace["physical"]["joint_speed_rad_s"][5][0] = float("nan")

    metrics = summarize_episode(
        trace, nail_geometry=load_frozen_nail_geometry(ASSET)
    )

    assert metrics["qvel_violation"] is True
    assert metrics["qvel_nonfinite"] is True


def test_interior_post_only_nonfinite_qvel_survives_dedup_and_exits_two():
    trace = _literal_trace()
    trace["physical"]["joint_speed_rad_s"] = [[0.0] * 6 for _ in range(35)]
    trace["physical"]["post_step_joint_speed_rad_s"] = [
        [0.0] * 6 for _ in range(35)
    ]
    trace["physical"]["post_step_joint_speed_rad_s"][5][3] = float("nan")

    metrics = summarize_episode(
        trace, nail_geometry=load_frozen_nail_geometry(ASSET)
    )
    aggregate = aggregate_episode_metrics([metrics], expected_episode_count=1)

    assert metrics["qvel_nonfinite"] is True
    assert metrics["qvel_finite_exceedance"] is False
    assert aggregate["qvel_nonfinite_rate_sampled"] == 1.0
    assert aggregate["qvel_finite_exceedance_rate_sampled"] == 0.0
    with pytest.raises(SystemExit) as error:
        eval_impulse._enforce_postwrite_invariants(
            name="post-only-nonfinite",
            row=aggregate,
            impossible_success_n=0,
            lambda_dead_n=0,
            provenance_invalid=False,
            repo_hash="a" * 40,
            asset_hash="b" * 40,
        )
    assert error.value.code == 2


def test_qvel_rail_diagnostics_use_unique_states_and_event_phase_boundaries():
    trace = _literal_trace()
    pre = [[0.0] * 6 for _ in range(35)]
    pre[5][0] = 3.2
    pre[31][1] = 4.0
    pre[33][2] = 5.0
    trace["physical"]["joint_speed_rad_s"] = pre
    trace["physical"]["post_step_joint_speed_rad_s"] = [
        pre[index + 1][:] for index in range(34)
    ] + [pre[-1][:]]

    metrics = summarize_episode(
        trace, nail_geometry=load_frozen_nail_geometry(ASSET)
    )

    assert metrics["qvel_violation"] is True
    assert metrics["qvel_nonfinite"] is False
    assert metrics["qvel_max_abs_rad_s"] == pytest.approx(5.0)
    assert metrics["qvel_max_excess_rad_s"] == pytest.approx(5.0 - 3.1415)
    assert metrics["qvel_samples_over_rail"] == 3
    assert metrics["qvel_seconds_over_rail"] == pytest.approx(0.006)
    assert metrics["qvel_offending_joint"] == "joint3"
    assert metrics["qvel_peak_phase"] == "post_event"
    assert metrics["qvel_precontact_samples_over_rail"] == 1
    assert metrics["qvel_first_event_samples_over_rail"] == 1
    assert metrics["qvel_post_event_samples_over_rail"] == 1


def test_qvel_pre_post_channels_must_form_one_coherent_state_sequence():
    trace = _literal_trace()
    trace["physical"]["post_step_joint_speed_rad_s"][5][0] = 1.0

    with pytest.raises(ValueError, match="temporally coherent"):
        summarize_episode(
            trace, nail_geometry=load_frozen_nail_geometry(ASSET)
        )


def test_terminal_post_step_joint_speed_is_included_in_hardware_gate():
    trace = _literal_trace()
    trace["physical"]["joint_speed_rad_s"] = [[0.0] * 6 for _ in range(35)]
    trace["physical"]["post_step_joint_speed_rad_s"] = [
        [0.0] * 6 for _ in range(35)
    ]
    trace["physical"]["post_step_joint_speed_rad_s"][-1][2] = 3.2

    metrics = summarize_episode(
        trace, nail_geometry=load_frozen_nail_geometry(ASSET)
    )

    assert metrics["qvel_violation"] is True
    assert metrics["qvel_samples_over_rail"] == 1
    assert metrics["qvel_peak_phase"] == "post_event"


def test_balanced_stopping_keeps_exactly_first_two_episodes_per_environment():
    completions = [
        {"env_id": 0, "episode_ordinal": 0},
        {"env_id": 0, "episode_ordinal": 1},
        {"env_id": 0, "episode_ordinal": 2},
        {"env_id": 1, "episode_ordinal": 0},
        {"env_id": 1, "episode_ordinal": 1},
    ]

    selected = []
    counts = [0, 0]
    accepted = [
        eval_impulse._retain_first_completed(
            record,
            counts=counts,
            selected=selected,
            episodes_per_env=2,
        )
        for record in completions
    ]

    assert [(row["env_id"], row["episode_ordinal"]) for row in selected] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert accepted == [True, True, False, True, True]
    assert counts == [2, 2]


def test_evaluator_csv_contains_the_complete_sampled_contract():
    assert REQUIRED_SAMPLED_COLUMNS <= set(eval_impulse.FIELDNAMES)
    assert {
        "first_contact_geometry_n_sampled",
        "terminal_funnel_eligible_n_sampled",
        "first_strike_unfinished_event_n_sampled",
        "action_rng_seed",
        "reset_rng_seed",
        "observation_rng_seed",
        "reset_position_noise_min_rad",
        "reset_position_noise_max_rad",
        "actor_observation_corruption",
        "critic_observation_corruption",
        "sampled_completion_rule",
        "sampled_actions_stochastic",
        "physics_dt_s",
        "control_decimation",
        "fixed_impedance_signature_sha256",
        "fixed_action_signature_sha256",
        "accepted_checkpoint_sha256",
        "accepted_manifest_sha256",
        "training_code_revision",
        "training_asset_revision",
        "qvel_nonfinite_rate_sampled",
        "qvel_finite_exceedance_rate_sampled",
        "qvel_max_abs_rad_s_sampled",
        "qvel_max_excess_rad_s_sampled",
        "qvel_samples_over_rail_sampled",
        "qvel_seconds_over_rail_sampled",
        "qvel_peak_joint_index_sampled",
        "qvel_precontact_exceedance_rate_sampled",
        "qvel_first_event_exceedance_rate_sampled",
        "qvel_post_event_exceedance_rate_sampled",
    } <= set(eval_impulse.FIELDNAMES)


@pytest.mark.parametrize(("arm", "task"), ARM_TASKS.items())
def test_live_task_configs_restore_training_rng_inputs_and_identical_tracker(arm, task):
    cfg = eval_impulse.load_env_cfg(task, play=False)
    cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
    eval_impulse._ensure_first_strike_instrumentation(cfg)

    contract = eval_impulse._validate_sampled_env_contract(cfg, task)

    assert contract["treatment"] == arm
    assert contract["impact_weight"] == 8.0
    assert contract["delivered_weight"] == 2.0
    assert (
        contract["reset_position_noise_min_rad"],
        contract["reset_position_noise_max_rad"],
    ) == (-0.05, 0.05)
    assert contract["actor_observation_corruption"] is True
    assert contract["critic_observation_corruption"] is False
    assert contract["physics_dt_s"] == pytest.approx(0.002)
    assert contract["control_decimation"] == 10
    assert (
        contract["fixed_impedance_signature_sha256"]
        == EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256
    )
    assert (
        contract["fixed_action_signature_sha256"]
        == EXPECTED_FIXED_ACTION_SIGNATURE_SHA256
    )
    assert cfg.metrics["first_strike"].per_substep is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta_pos_scale", 0.30),
        ("max_dq", 0.29),
        ("damping", 0.10),
    ],
)
def test_sampled_contract_rejects_action_solver_drift(field, value):
    drifted = eval_impulse.load_env_cfg(ARM_TASKS["E"], play=False)
    drifted.metrics["cat_soft"].params["imp_max_p"] = 0.0
    setattr(drifted.actions["ik_hammer_head"], field, value)
    with pytest.raises(ValueError, match="fixed action signature"):
        eval_impulse._validate_sampled_env_contract(
            drifted, ARM_TASKS["E"]
        )


def test_sampled_contract_rejects_action_set_drift():
    extra = eval_impulse.load_env_cfg(ARM_TASKS["E"], play=False)
    extra.metrics["cat_soft"].params["imp_max_p"] = 0.0
    extra.actions["set_gains"] = extra.actions["ik_hammer_head"]
    with pytest.raises(ValueError, match="fixed action signature"):
        eval_impulse._validate_sampled_env_contract(extra, ARM_TASKS["E"])


def test_live_campaign_timestep_uses_mujoco_config_not_missing_sim_dt():
    cfg = eval_impulse.load_env_cfg(ARM_TASKS["E"], play=False)

    assert not hasattr(cfg.sim, "dt")
    assert cfg.sim.mujoco.timestep == pytest.approx(0.002)
    assert eval_impulse._physics_timestep_s(cfg) == pytest.approx(0.002)
    source = inspect.getsource(eval_impulse.main)
    assert "physics_dt_s=_physics_timestep_s(env_cfg)" in source
    assert "env_cfg.sim.dt" not in source


def test_common_campaign_identity_excludes_live_arm_specific_impulse_reference():
    contracts = {}
    for arm, task in ARM_TASKS.items():
        cfg = eval_impulse.load_env_cfg(task, play=False)
        cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
        contracts[arm] = eval_impulse._validate_sampled_env_contract(cfg, task)

    assert {
        arm: contract["event_i_ref_n_s"]
        for arm, contract in contracts.items()
    } == {
        "C": pytest.approx(0.6094),
        "D-prime": pytest.approx(0.6094),
        "F": pytest.approx(0.3088),
        "E": pytest.approx(0.3088),
    }
    common = {
        eval_impulse._campaign_config_digest(
            contract=contract,
            num_envs=256,
            episode_len_s=3.0,
            physics_dt_s=0.002,
            decimation=10,
            reset_seed=11,
            observation_seed=12,
            action_seed=13,
        )
        for contract in contracts.values()
    }
    treatment = {
        eval_impulse._treatment_config_digest(
            task=ARM_TASKS[arm],
            contract=contract,
        )
        for arm, contract in contracts.items()
    }

    assert len(common) == 1
    assert len(treatment) == 4


def test_action_and_environment_rng_streams_are_interleaving_independent():
    action = eval_impulse._TorchRngStream(101, "cpu")
    reset = eval_impulse._TorchRngStream(202, "cpu")
    action_reference = eval_impulse._TorchRngStream(101, "cpu")

    action_first = action.run(lambda: torch.rand(4))
    reset.run(lambda: torch.rand(100))
    action_second = action.run(lambda: torch.rand(4))
    reference_first = action_reference.run(lambda: torch.rand(4))
    reference_second = action_reference.run(lambda: torch.rand(4))

    torch.testing.assert_close(action_first, reference_first)
    torch.testing.assert_close(action_second, reference_second)


def test_substep_phase_capture_lags_post_step_joint_and_depth_to_preintegration():
    phase = eval_impulse._PreIntegrationTracePhase()
    phase.cache_pre_step(
        joint_speed_rad_s=torch.tensor([[1.0, 2.0]]),
        clamped_depth_m=torch.tensor([0.010]),
    )

    first = phase.capture(
        head_position_m=torch.tensor([[10.0, 0.0, 0.0]]),
        contact=torch.tensor([True]),
        net_axial_force_n=torch.tensor([3.0]),
        post_step_joint_speed_rad_s=torch.tensor([[11.0, 12.0]]),
        post_step_clamped_depth_m=torch.tensor([0.020]),
    )
    phase.cache_pre_step(
        joint_speed_rad_s=torch.tensor([[11.0, 12.0]]),
        clamped_depth_m=torch.tensor([0.020]),
    )
    second = phase.capture(
        head_position_m=torch.tensor([[20.0, 0.0, 0.0]]),
        contact=torch.tensor([False]),
        net_axial_force_n=torch.tensor([4.0]),
        post_step_joint_speed_rad_s=torch.tensor([[21.0, 22.0]]),
        post_step_clamped_depth_m=torch.tensor([0.030]),
    )

    assert first["head_position_m"].tolist() == [[10.0, 0.0, 0.0]]
    assert first["joint_speed_rad_s"].tolist() == [[1.0, 2.0]]
    assert first["clamped_depth_m"].tolist() == pytest.approx([0.010])
    assert second["head_position_m"].tolist() == [[20.0, 0.0, 0.0]]
    assert second["joint_speed_rad_s"].tolist() == [[11.0, 12.0]]
    assert second["clamped_depth_m"].tolist() == pytest.approx([0.020])
    assert second["post_step_joint_speed_rad_s"].tolist() == [[21.0, 22.0]]

    phase.cache_pre_step(
        joint_speed_rad_s=torch.tensor([[31.0, 32.0]]),
        clamped_depth_m=torch.tensor([0.040]),
    )
    after_reset = phase.capture(
        head_position_m=torch.tensor([[30.0, 0.0, 0.0]]),
        contact=torch.tensor([False]),
        net_axial_force_n=torch.tensor([0.0]),
        post_step_joint_speed_rad_s=torch.tensor([[41.0, 42.0]]),
        post_step_clamped_depth_m=torch.tensor([0.050]),
    )
    assert after_reset["joint_speed_rad_s"].tolist() == [[31.0, 32.0]]
    assert after_reset["clamped_depth_m"].tolist() == pytest.approx([0.040])

    guard = eval_impulse._PreIntegrationTracePhase()
    guard.cache_pre_step(
        joint_speed_rad_s=torch.tensor([[1.0, 2.0]]),
        clamped_depth_m=torch.tensor([0.010]),
    )
    with pytest.raises(RuntimeError, match="unconsumed pre-integration cache"):
        guard.cache_pre_step(
            joint_speed_rad_s=torch.tensor([[3.0, 4.0]]),
            clamped_depth_m=torch.tensor([0.020]),
        )


def test_live_substep_trace_channels_cross_real_terminal_autoreset_in_phase():
    task = eval_impulse.QUALITY_ARM_TASKS["FQ"]
    _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = float(
        cfg.sim.mujoco.timestep * cfg.decimation
    )
    cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
    env = eval_impulse.ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    snapshot = eval_impulse._install_episode_hook(env)
    geometry = load_frozen_nail_geometry(ASSET)
    collector = eval_impulse._SampledTraceCollector(
        env,
        snapshot=snapshot,
        treatment="FQ",
        task=task,
        gamma=0.99,
        event_i_ref_n_s=0.3088,
        nail_geometry=geometry,
    )
    original_step = env.sim.step
    literal_frames = []

    def instrumented_step():
        pre_qvel = (
            collector._robot.data.joint_vel[:, collector._arm_ids].detach().clone()
        )
        pre_depth = (
            collector._nail.data.joint_pos[:, 0]
            .clamp(0.0, 0.032)
            .detach()
            .clone()
        )
        original_step()
        force = collector._net_force.data.force
        axis = collector._axis.to(dtype=force.dtype)
        literal_frames.append(
            {
                "joint_speed_rad_s": pre_qvel,
                "clamped_depth_m": pre_depth,
                "head_position_m": (
                    collector._robot.data.site_pos_w[:, collector._head_ids]
                    .squeeze(1)
                    .detach()
                    .clone()
                ),
                "contact": (
                    (collector._contact.data.found > 0)
                    .any(dim=-1)
                    .detach()
                    .clone()
                ),
                "net_axial_force_n": (
                    (force * axis)
                    .sum(dim=-1)
                    .sum(dim=-1)
                    .clamp_min(0.0)
                    .detach()
                    .clone()
                ),
            }
        )

    env.sim.step = instrumented_step
    try:
        env.reset()
        zeros = torch.zeros(
            (1, env.action_manager.total_action_dim), device=env.device
        )
        env.step(zeros)
        assert len(collector.completed) == 1
        assert collector.completed[0]["episode_ordinal"] == 0
        first_start = 0
        for index, expected in enumerate(literal_frames):
            for key, value in expected.items():
                torch.testing.assert_close(
                    collector._substeps[key][first_start + index], value
                )

        # ManagerBasedRlEnv.step already auto-reset the terminal environment.
        # No manual reset is allowed between these two control steps.
        reset_start = len(collector._substeps["contact"])
        literal_frames.clear()
        env.step(zeros)
        assert len(collector.completed) == 2
        assert [
            trace["episode_ordinal"] for trace in collector.completed
        ] == [0, 1]
        for index, expected in enumerate(literal_frames):
            for key, value in expected.items():
                torch.testing.assert_close(
                    collector._substeps[key][reset_start + index], value
                )
        assert (
            len(collector.completed[0]["physical"]["contact"])
            == reset_start
        )
        assert (
            len(collector.completed[1]["physical"]["contact"])
            == len(literal_frames)
        )
        trace = collector.completed[0]
        n_substeps = len(trace["physical"]["contact"])
        assert set(eval_impulse._TRACE_PHYSICAL_KEYS) <= set(trace["physical"])
        assert set(eval_impulse._TRACE_EVENT_KEYS) <= set(trace["event_trace"])
        assert len(trace["physical"]["quality_found_count"]) == n_substeps
        assert all(
            len(frame) == 8
            for frame in trace["physical"]["quality_found_count"]
        )
        assert all(
            len(frame) == 8
            for frame in trace["physical"]["quality_normal_force_n"]
        )
        assert all(
            len(frame) == 8 and all(len(point) == 3 for point in frame)
            for frame in trace["physical"]["quality_contact_position_m"]
        )
        assert all(
            len(frame) == 8 and all(len(normal) == 3 for normal in frame)
            for frame in trace["physical"]["quality_contact_normal"]
        )
        assert all(
            len(values) == n_substeps
            for values in trace["event_trace"].values()
        )
        assert eval_impulse._physical_trace_digest(trace) == trace["trace_digest"]
        eval_impulse.compare_action_tape_physics({"FQ": trace})
        assert collector._phase._joint_speed_rad_s is None
        assert collector._phase._clamped_depth_m is None

        wrapped = eval_impulse.RslRlVecEnvWrapper(env, clip_actions=1.0)
        raw_action = torch.tensor([[2.0, -3.0, 0.5]], device=env.device)
        wrapped.step(raw_action)
        assert collector._actions[-1].tolist() == [[1.0, -1.0, 0.5]]
    finally:
        env.close()


def test_git_provenance_marks_untracked_files_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    tracked = repo / "tracked.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task4 Test",
            "-c",
            "user.email=task4@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    clean = eval_impulse._git_provenance(repo)
    (repo / "untracked.py").write_text("EXECUTED = True\n")
    dirty = eval_impulse._git_provenance(repo)

    assert clean["dirty"] is False
    assert len(clean["revision"]) == 40
    assert dirty["dirty"] is True
    assert "?? untracked.py" in dirty["status"]


def test_git_provenance_is_captured_before_rollout_and_verified_after(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    tracked = repo / "tracked.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task4 Test",
            "-c",
            "user.email=task4@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    expected = eval_impulse._git_provenance(repo)
    tracked.write_text("VALUE = 2\n")

    with pytest.raises(RuntimeError, match="code provenance changed"):
        eval_impulse._verify_unchanged_git(
            repo,
            expected=expected,
            label="code",
        )

    source = inspect.getsource(eval_impulse.main)
    assert source.index("code_git = _git_provenance") < source.index(
        "mean_rec = _rollout"
    )
    assert source.index("_verify_unchanged_git(") > source.index(
        "sampled_rec = _rollout_balanced_sampled"
    )


def test_raw_artifact_verification_does_not_cache_full_episode_payloads():
    loader = first_strike_campaign._load_and_recompute_sampled_artifact

    assert not hasattr(loader, "cache_info")


def test_sampled_trace_artifacts_are_content_addressed_and_keep_provenance(tmp_path):
    contract = {
        "treatment": "E",
        "impact_weight": 8.0,
        "delivered_weight": 2.0,
        "event_i_ref_n_s": 0.3088,
        "impulse_limits_n_m_s": [1.64, 3.28, 1.64, 1.64, 1.64, 1.64],
    }
    provenance = {
        "code_git": {"revision": "a" * 40, "dirty": False, "status": ""},
        "asset_git": {"revision": "b" * 40, "dirty": False, "status": ""},
        "checkpoint_sha256": "e" * 64,
        "campaign_config_sha256": "d" * 64,
        "nail_asset_sha256": "c" * 64,
    }
    common = {
        "out_dir": tmp_path,
        "name": "rerun-label",
        "task": ARM_TASKS["E"],
        "contract": contract,
        "training_seed": 0,
        "reset_seed": 11,
        "observation_seed": 12,
        "action_seed": 13,
        "nail_geometry": {
            "nail_axis": [0.0, 0.0, -1.0],
            "nail_xy_m": [0.5, 0.0],
            "nail_radius_m": 0.012,
            "source_sha256": "c" * 64,
        },
        "provenance": provenance,
        "mean_rollout_invariants": {
            "impossible_success_n": 0,
            "lambda_dead_n": 0,
        },
    }
    first = eval_impulse._persist_sampled_traces(
        sampled_rec={
            "control_steps": 1,
            "episodes": [{"episode_id": "first", "value": 1}],
        },
        **common,
    )
    first_bytes = Path(first["path"]).read_bytes()
    second = eval_impulse._persist_sampled_traces(
        sampled_rec={
            "control_steps": 2,
            "episodes": [{"episode_id": "second", "value": 2}],
        },
        **common,
    )

    assert first["path"] != second["path"]
    assert Path(first["path"]).read_bytes() == first_bytes
    assert first["payload_digest"] in Path(first["path"]).name
    assert first["artifact_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    with np.load(first["path"], allow_pickle=False) as saved:
        payload = json.loads(str(saved["payload_json"]))
    assert payload["schema_version"] == 3
    assert payload["provenance"] == provenance


def test_exact_seed_tests_enumerate_all_assignments_with_midranks():
    result = exact_seed_tests([1.2] * 8, [1.0] * 8)

    assert result["assignment_count"] == 12870
    assert result["u_treatment"] == pytest.approx(64.0)
    assert result["a12_treatment_over_control"] == pytest.approx(1.0)
    assert result["mwu_two_sided_exact_p"] == pytest.approx(2 / 12870)
    assert result["mean_difference_two_sided_exact_p"] == pytest.approx(2 / 12870)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("git_revision", "f" * 40, "code revision differs"),
        ("nail_asset_sha256", "f" * 64, "nail asset identity differs"),
        ("campaign_config_sha256", "f" * 64, "campaign config identity differs"),
        ("sampled_trace_digest", "f" * 64, "sampled trace digest mismatch"),
    ],
)
def test_campaign_rejects_mismatched_clean_identity_or_trace_digest(
    field, value, reason, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)
    rows[0][field] = value

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["valid"] is False
    assert any(reason in item for item in result["invalidation_reasons"])
    assert result["primary"]["passed"] is False


def test_campaign_requires_complete_sampled_contract_and_exhaustive_statuses(
    campaign_rows_fixture,
):
    missing = copy.deepcopy(campaign_rows_fixture)
    del missing[0]["tail_fraction_mean_sampled"]
    missing_result = analyze_campaign(missing, bootstrap_samples=1000)
    assert missing_result["valid"] is False
    assert any(
        "missing sampled columns" in item
        for item in missing_result["invalidation_reasons"]
    )

    nonexhaustive = copy.deepcopy(campaign_rows_fixture)
    nonexhaustive[0]["first_strike_unfinished_event_n_sampled"] = 0
    count_result = analyze_campaign(nonexhaustive, bootstrap_samples=1000)
    assert count_result["valid"] is False
    assert any(
        "first-strike statuses do not sum to 512" in item
        for item in count_result["invalidation_reasons"]
    )


@pytest.mark.parametrize(
    ("field", "invalid_value", "reason"),
    [
        ("seed", 42, "base evaluator RNG"),
        ("reset_rng_seed", 1, "reset RNG"),
        ("observation_rng_seed", 2, "observation RNG"),
        ("action_rng_seed", 3, "action RNG"),
        ("episode_len_s", 20.0, "4.0 s evaluation cutoff"),
        ("nsteps", 399, "400 diagnostic mean steps"),
        ("reset_position_noise_min_rad", 0.0, "reset noise"),
        ("reset_position_noise_max_rad", 0.0, "reset noise"),
        ("actor_observation_corruption", False, "actor corruption"),
        ("critic_observation_corruption", True, "critic corruption"),
        ("physics_dt_s", 0.004, "500 Hz physics"),
        ("control_decimation", 5, "control decimation"),
        ("fixed_impedance_signature_sha256", "f" * 64, "fixed impedance"),
        ("fixed_action_signature_sha256", "f" * 64, "fixed action"),
        ("accepted_checkpoint_sha256", "f" * 64, "accepted checkpoint"),
        ("accepted_manifest_sha256", "not-a-digest", "accepted manifest"),
        ("training_code_revision", "f" * 40, "training/eval code"),
        ("training_asset_revision", "f" * 40, "training/eval asset"),
        ("checkpoint_path", "/wrong/model_498.pt", "model_499.pt"),
    ],
)
def test_campaign_rejects_any_nonfrozen_runtime_contract(
    field, invalid_value, reason, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)
    rows[0][field] = invalid_value

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["valid"] is False
    assert any(reason in item for item in result["invalidation_reasons"])


def test_campaign_reports_the_frozen_identity_comparison_contract(
    campaign_rows_fixture,
):
    result = analyze_campaign(campaign_rows_fixture, bootstrap_samples=1000)

    assert result["comparison_identity_contract"] == {
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
        "bound_per_arm": [
            "treatment_config_sha256",
            "event_i_ref_n_s",
        ],
        "treatment_identity": "exact ARM_TASKS task id plus configured 8/2 weights",
    }


def test_campaign_accepts_csv_string_scalars_without_treating_false_as_dirty(
    campaign_rows_fixture,
):
    rows = [
        {
            key: (
                str(value)
                if isinstance(value, (bool, int, float))
                else value
            )
            for key, value in row.items()
        }
        for row in campaign_rows_fixture
    ]

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["valid"] is True
    assert result["primary"]["passed"] is True


def test_campaign_recomputes_sampled_csv_statistics_from_raw_episodes(
    campaign_rows_fixture,
):
    rows = copy.deepcopy(campaign_rows_fixture)
    rows[0]["first_strike_useful_speed_mean_sampled"] += 0.25

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["valid"] is False
    assert any(
        "sampled aggregate mismatch" in item
        for item in result["invalidation_reasons"]
    )


def test_campaign_recomputes_each_episode_trace_digest(
    tmp_path, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)
    source = Path(rows[0]["sampled_trace_path"])
    with np.load(source, allow_pickle=False) as saved:
        payload = json.loads(str(saved["payload_json"]))
    payload["episodes"][0]["physical"]["head_position_m"][0][0] += 0.001
    payload_without_digest = dict(payload)
    payload_without_digest.pop("payload_digest")
    payload["payload_digest"] = _literal_digest(payload_without_digest)
    mutated = tmp_path / "mutated-trace.npz"
    np.savez_compressed(
        mutated,
        payload_json=np.asarray(
            json.dumps(payload, sort_keys=True, allow_nan=False), dtype=np.str_
        ),
    )
    rows[0]["sampled_trace_path"] = str(mutated)
    rows[0]["sampled_trace_digest"] = payload["payload_digest"]
    rows[0]["sampled_trace_artifact_sha256"] = hashlib.sha256(
        mutated.read_bytes()
    ).hexdigest()

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["valid"] is False
    assert any(
        "episode trace digest mismatch" in item
        for item in result["invalidation_reasons"]
    )


@pytest.mark.parametrize(
    "field",
    (
        "success_rate_sampled",
        "worst_ratio_max_sampled",
        "delivered_mean_sampled",
    ),
)
def test_campaign_rebinds_legacy_sampled_csv_scalars_to_raw_episodes(
    field, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)
    rows[0][field] += 0.01

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["valid"] is False
    assert any(
        f"legacy sampled metric mismatch for {field}" in item
        for item in result["invalidation_reasons"]
    )


@pytest.mark.parametrize("field", ("impossible_success_n", "lambda_dead_n"))
def test_campaign_rebinds_combined_invariants_to_raw_artifact(
    field, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)
    rows[0][field] = 1

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["valid"] is False
    assert any(
        f"combined invariant mismatch for {field}" in item
        for item in result["invalidation_reasons"]
    )


def test_checkpoint_identity_is_verified_again_after_rollout(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"loaded checkpoint bytes")
    expected = eval_impulse._sha256_file(checkpoint)
    checkpoint.write_bytes(b"mutated checkpoint bytes")

    with pytest.raises(RuntimeError, match="checkpoint changed during evaluation"):
        eval_impulse._verify_unchanged_file(
            checkpoint,
            expected_sha256=expected,
            label="checkpoint",
        )


def test_expected_manifest_checkpoint_hash_is_verified_before_rollout(tmp_path):
    checkpoint = tmp_path / "model_499.pt"
    checkpoint.write_bytes(b"mutated after manifest preflight")

    with pytest.raises(RuntimeError, match="accepted checkpoint SHA-256 mismatch"):
        eval_impulse._verify_expected_checkpoint_sha256(
            checkpoint,
            expected_sha256=hashlib.sha256(b"preflight bytes").hexdigest(),
        )


def test_both_rollouts_load_identical_frozen_bytes_if_original_changes(
    tmp_path,
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"original-checkpoint")
    snapshot_path = None

    with pytest.raises(RuntimeError, match="checkpoint changed"):
        with eval_impulse._frozen_checkpoint(
            checkpoint, snapshot_parent=tmp_path
        ) as (source_path, frozen_path, digest):
            snapshot_path = frozen_path
            assert source_path == checkpoint.resolve()
            assert digest == hashlib.sha256(b"original-checkpoint").hexdigest()
            assert frozen_path.read_bytes() == b"original-checkpoint"
            checkpoint.write_bytes(b"replacement-checkpoint")
            assert frozen_path.read_bytes() == b"original-checkpoint"

    assert snapshot_path is not None
    assert not snapshot_path.exists()
    source = inspect.getsource(eval_impulse.main)
    assert source.count("str(checkpoint_load_path)") == 2
    assert "_rollout(env_cfg, agent_cfg, runner_cls, args.ckpt" not in source
    assert "runner_cls,\n    args.ckpt," not in source


def test_primary_is_only_e_vs_dprime_and_requires_both_tests_effect_and_guardrails(
    campaign_rows_fixture,
):
    result = analyze_campaign(campaign_rows_fixture, bootstrap_samples=1000)

    assert result["primary"]["contrast"] == ["E", "D-prime"]
    assert result["primary"]["passed"] is True
    assert result["primary"]["gates"] == {
        "valid_provenance_and_sentinels": True,
        "mwu_p_lt_0_05": True,
        "permutation_p_lt_0_05": True,
        "relative_effect_at_least_10_percent": True,
        "first_window_success_guardrail": True,
        "overall_success_guardrail": True,
    }
    assert result["descriptive"]["contrast"] == ["C", "D-prime"]

    below_effect = copy.deepcopy(campaign_rows_fixture)
    for row in below_effect:
        if row["treatment"] == "E":
            row["first_strike_useful_speed_mean_sampled"] = 1.05
    failed = analyze_campaign(below_effect, bootstrap_samples=1000)
    assert failed["primary"]["gates"]["mwu_p_lt_0_05"] is True
    assert failed["primary"]["gates"]["permutation_p_lt_0_05"] is True
    assert failed["primary"]["gates"]["relative_effect_at_least_10_percent"] is False
    assert failed["primary"]["passed"] is False


@pytest.mark.parametrize(
    ("field", "value", "failed_gate"),
    [
        (
            "first_strike_success_rate_sampled",
            0.89,
            "first_window_success_guardrail",
        ),
        ("overall_success_rate_sampled", 0.89, "overall_success_guardrail"),
    ],
)
def test_primary_enforces_both_sampled_success_guardrails(
    field, value, failed_gate, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)
    for row in rows:
        if row["treatment"] == "E":
            row[field] = value

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["primary"]["gates"][failed_gate] is False
    assert result["primary"]["passed"] is False


def test_success_guardrail_also_rejects_more_than_five_points_below_dprime(
    campaign_rows_fixture,
):
    rows = copy.deepcopy(campaign_rows_fixture)
    for row in rows:
        if row["treatment"] == "D-prime":
            row["first_strike_success_rate_sampled"] = 0.98
        elif row["treatment"] == "E":
            row["first_strike_success_rate_sampled"] = 0.92

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["primary"]["success_guardrails"]["first_window"]["absolute_90"] is True
    assert result["primary"]["success_guardrails"]["first_window"]["difference"] == pytest.approx(
        -0.06
    )
    assert result["primary"]["gates"]["first_window_success_guardrail"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__("git_dirty", True),
        lambda row: row.__setitem__("impossible_success_n", 1),
        lambda row: row.__setitem__("lambda_dead_n", 1),
    ],
)
def test_dirty_and_sentinel_rows_fail_closed(
    mutate, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)
    mutate(rows[0])

    result = analyze_campaign(rows, bootstrap_samples=1000)

    assert result["valid"] is False
    assert result["primary"]["passed"] is False
    assert result["primary"]["gates"]["valid_provenance_and_sentinels"] is False
    assert result["invalidation_reasons"]


def test_finite_qvel_exceedance_metrics_are_valid_but_range_checked():
    finite_exceedance = {
        "qvel_violation_rate_sampled": 0.25,
        "qvel_finite_exceedance_rate_sampled": 0.25,
        "qvel_nonfinite_rate_sampled": 0.0,
        "qvel_max_abs_rad_s_sampled": 4.0,
        "qvel_max_excess_rad_s_sampled": 4.0 - 3.1415,
        "qvel_samples_over_rail_sampled": 160,
        "qvel_seconds_over_rail_sampled": 0.32,
        "qvel_peak_joint_index_sampled": 2,
        "qvel_precontact_exceedance_rate_sampled": 64 / 512,
        "qvel_precontact_samples_over_rail_sampled": 64,
        "qvel_precontact_seconds_over_rail_sampled": 0.128,
        "qvel_first_event_exceedance_rate_sampled": 32 / 512,
        "qvel_first_event_samples_over_rail_sampled": 32,
        "qvel_first_event_seconds_over_rail_sampled": 0.064,
        "qvel_post_event_exceedance_rate_sampled": 64 / 512,
        "qvel_post_event_samples_over_rail_sampled": 64,
        "qvel_post_event_seconds_over_rail_sampled": 0.128,
    }

    assert first_strike_campaign._qvel_sampled_metrics_reasons(
        finite_exceedance, "E/seed0"
    ) == []

    out_of_range = dict(finite_exceedance)
    out_of_range["qvel_violation_rate_sampled"] = 1.01
    assert first_strike_campaign._qvel_sampled_metrics_reasons(
        out_of_range, "E/seed0"
    )

    fractional_count = dict(finite_exceedance)
    fractional_count["qvel_samples_over_rail_sampled"] = 100.5
    assert first_strike_campaign._qvel_sampled_metrics_reasons(
        fractional_count, "E/seed0"
    )


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    (
        (
            "qvel_samples_over_rail_sampled",
            512 * 2001 + 1,
            "exceeds sampled population support",
        ),
        (
            "qvel_precontact_exceedance_rate_sampled",
            0.0,
            "precontact qvel phase rate/count",
        ),
        (
            "qvel_finite_exceedance_rate_sampled",
            127.5 / 512,
            "qvel rates must resolve to episode counts",
        ),
        (
            "qvel_violation_rate_sampled",
            0.5,
            "legacy qvel violation rate",
        ),
    ),
)
def test_qvel_aggregate_consistency_mutations_are_rejected(
    field, value, reason_fragment
):
    row = {
        "qvel_violation_rate_sampled": 128 / 512,
        "qvel_finite_exceedance_rate_sampled": 128 / 512,
        "qvel_nonfinite_rate_sampled": 0.0,
        "qvel_max_abs_rad_s_sampled": 4.0,
        "qvel_max_excess_rad_s_sampled": 4.0 - 3.1415,
        "qvel_samples_over_rail_sampled": 160,
        "qvel_seconds_over_rail_sampled": 0.32,
        "qvel_peak_joint_index_sampled": 2,
        "qvel_precontact_exceedance_rate_sampled": 64 / 512,
        "qvel_precontact_samples_over_rail_sampled": 64,
        "qvel_precontact_seconds_over_rail_sampled": 0.128,
        "qvel_first_event_exceedance_rate_sampled": 32 / 512,
        "qvel_first_event_samples_over_rail_sampled": 32,
        "qvel_first_event_seconds_over_rail_sampled": 0.064,
        "qvel_post_event_exceedance_rate_sampled": 64 / 512,
        "qvel_post_event_samples_over_rail_sampled": 64,
        "qvel_post_event_seconds_over_rail_sampled": 0.128,
    }
    row[field] = value
    if field == "qvel_samples_over_rail_sampled":
        row["qvel_seconds_over_rail_sampled"] = value * 0.002

    reasons = first_strike_campaign._qvel_sampled_metrics_reasons(
        row, "E/seed0"
    )

    assert any(reason_fragment in reason for reason in reasons), reasons


def test_hardware_speed_qualification_is_contrast_scoped():
    rows = {
        arm: [
            {
                "qvel_violation_rate_sampled": (
                    0.25 if arm in {"C", "F"} else 0.0
                ),
                "qvel_finite_exceedance_rate_sampled": (
                    0.25 if arm in {"C", "F"} else 0.0
                ),
                "qvel_max_abs_rad_s_sampled": (
                    4.0 if arm in {"C", "F"} else 3.0
                ),
                "qvel_max_excess_rad_s_sampled": (
                    4.0 - 3.1415 if arm in {"C", "F"} else 0.0
                ),
                "qvel_samples_over_rail_sampled": (
                    100 if arm in {"C", "F"} else 0
                ),
                "qvel_seconds_over_rail_sampled": (
                    0.2 if arm in {"C", "F"} else 0.0
                ),
            }
        ]
        for arm in ARM_TASKS
    }

    primary = first_strike_campaign._hardware_speed_qualification(
        rows, "E", "D-prime"
    )
    f_vs_dprime = first_strike_campaign._hardware_speed_qualification(
        rows, "F", "D-prime"
    )
    descriptive = first_strike_campaign._hardware_speed_qualification(
        rows, "C", "D-prime"
    )

    assert primary["arms"] == ["E", "D-prime"]
    assert primary["finite_rail_exceedance_observed"] is False
    assert f_vs_dprime["finite_rail_exceedance_observed"] is True
    assert f_vs_dprime["hardware_safe_claim_allowed"] is False
    assert descriptive["finite_rail_exceedance_observed"] is True


def test_clean_campaign_reports_separate_hardware_speed_qualifications(
    campaign_rows_fixture,
):
    result = analyze_campaign(campaign_rows_fixture, bootstrap_samples=1000)

    assert result["primary"]["hardware_speed_qualification"]["arms"] == [
        "E",
        "D-prime",
    ]
    assert {
        tuple(row["contrast"]): row["hardware_speed_qualification"]["arms"]
        for row in result["mechanism_family"]["contrasts"]
    } == {
        ("F", "D-prime"): ["F", "D-prime"],
        ("E", "F"): ["E", "F"],
    }
    assert result["descriptive"]["hardware_speed_qualification"]["arms"] == [
        "C",
        "D-prime",
    ]


def test_postwrite_qvel_policy_keeps_finite_exceedances_but_exits_two_on_nonfinite(
    capsys,
):
    status = eval_impulse._enforce_postwrite_invariants(
        name="finite",
        row={
            "qvel_violation_rate_sampled": 0.25,
            "qvel_finite_exceedance_rate_sampled": 0.25,
            "qvel_nonfinite_rate_sampled": 0.0,
        },
        impossible_success_n=0,
        lambda_dead_n=0,
        provenance_invalid=False,
        repo_hash="a" * 40,
        asset_hash="b" * 40,
    )

    assert status == "simulation_only_hardware_speed_unqualified"
    assert "not excluded" in capsys.readouterr().err

    with pytest.raises(SystemExit) as error:
        eval_impulse._enforce_postwrite_invariants(
            name="nonfinite",
            row={
                "qvel_violation_rate_sampled": 0.0,
                "qvel_finite_exceedance_rate_sampled": 0.0,
                "qvel_nonfinite_rate_sampled": 1 / 512,
            },
            impossible_success_n=0,
            lambda_dead_n=0,
            provenance_invalid=False,
            repo_hash="a" * 40,
            asset_hash="b" * 40,
        )
    assert error.value.code == 2


def test_holm_family_is_primary_gated_and_excludes_descriptive_c_contrast(
    campaign_rows_fixture,
):
    passed = analyze_campaign(campaign_rows_fixture, bootstrap_samples=1000)

    assert passed["mechanism_family"]["evaluated"] is True
    assert {
        tuple(row["contrast"]) for row in passed["mechanism_family"]["contrasts"]
    } == {("F", "D-prime"), ("E", "F")}
    assert all(
        "holm_adjusted_p" in row for row in passed["mechanism_family"]["contrasts"]
    )
    assert passed["descriptive"]["contrast"] == ["C", "D-prime"]
    assert "holm_adjusted_p" not in passed["descriptive"]

    failed_rows = copy.deepcopy(campaign_rows_fixture)
    for row in failed_rows:
        if row["treatment"] == "E":
            row["first_strike_useful_speed_mean_sampled"] = 1.0
    failed = analyze_campaign(failed_rows, bootstrap_samples=1000)
    assert failed["mechanism_family"]["evaluated"] is False
    assert failed["mechanism_family"]["contrasts"] == []
    assert failed["descriptive"]["contrast"] == ["C", "D-prime"]


def test_f_success_guardrails_gate_f_vs_dprime_interpretation(
    campaign_rows_fixture, tmp_path
):
    rows = copy.deepcopy(campaign_rows_fixture)

    def lower_success(trace):
        episode_index = int(trace["episode_id"].rsplit("-", 1)[1])
        trace["overall_success"] = episode_index < 410

    _rewrite_campaign_artifacts(
        rows,
        treatment="F",
        output_dir=tmp_path,
        transform=lower_success,
    )

    result = analyze_campaign(rows, bootstrap_samples=1000)
    by_contrast = {
        tuple(row["contrast"]): row
        for row in result["mechanism_family"]["contrasts"]
    }

    assert by_contrast[("F", "D-prime")]["success_guardrails_passed"] is False
    assert by_contrast[("F", "D-prime")]["interpretable"] is False


def test_mechanism_contrasts_require_the_frozen_hypothesized_directions(
    campaign_rows_fixture, tmp_path
):
    rows = copy.deepcopy(campaign_rows_fixture)

    def reverse_direction(trace):
        if trace["first_strike"]["reason"] == "success":
            trace["first_strike"]["v_precontact_m_s"] = 0.8 * 512 / 486

    _rewrite_campaign_artifacts(
        rows,
        treatment="F",
        output_dir=tmp_path,
        transform=reverse_direction,
    )

    result = analyze_campaign(rows, bootstrap_samples=1000)
    by_contrast = {
        tuple(row["contrast"]): row
        for row in result["mechanism_family"]["contrasts"]
    }

    assert by_contrast[("F", "D-prime")]["hypothesized_direction_passed"] is False
    assert by_contrast[("F", "D-prime")]["interpretable"] is False
    assert by_contrast[("E", "F")]["hypothesized_direction_passed"] is True


def test_representative_selection_is_seed_qualified_and_coordinate_only(tmp_path):
    def candidate(
        *,
        env_id: int,
        episode_ordinal: int,
        overall_success: bool,
        useful_speed: float,
        source_episode_id: str | None = None,
    ) -> dict:
        trace = _literal_trace(
            arm="C",
            reason="success" if overall_success else "window",
            overall_success=overall_success,
        )
        trace["episode_id"] = source_episode_id or (
            f"C-env{env_id}-episode{episode_ordinal}"
        )
        trace["env_id"] = env_id
        trace["episode_ordinal"] = episode_ordinal
        trace["first_strike"]["v_precontact_m_s"] = useful_speed
        trace["reward"]["impact_payout"] = [useful_speed * 1000.0]
        trace["physical"]["net_axial_force_n"][30] = useful_speed * 100.0
        trace["physical"]["head_position_m"][0][1] = useful_speed
        trace["trace_digest"] = _literal_digest(
            {
                "physical": trace["physical"],
                "event_trace": trace["event_trace"],
            }
        )
        return trace

    shared_legacy_id = "C-env2-episode0"
    seed_zero = [
        candidate(
            env_id=7,
            episode_ordinal=0,
            overall_success=True,
            useful_speed=0.01,
        ),
        candidate(
            env_id=2,
            episode_ordinal=1,
            overall_success=True,
            useful_speed=100.0,
        ),
        candidate(
            env_id=2,
            episode_ordinal=0,
            overall_success=True,
            useful_speed=10.0,
            source_episode_id=shared_legacy_id,
        ),
    ]
    seed_one = [
        candidate(
            env_id=0,
            episode_ordinal=0,
            overall_success=True,
            useful_speed=1000.0,
        ),
        candidate(
            env_id=2,
            episode_ordinal=0,
            overall_success=False,
            useful_speed=9999.0,
            source_episode_id=shared_legacy_id,
        ),
    ]
    rows = []
    payloads = []
    for seed, episodes, filename in (
        (1, seed_one, "aaa-seed-one.npz"),
        (0, seed_zero, "zzz-seed-zero.npz"),
    ):
        path = tmp_path / filename
        payload = {
            "treatment": "C",
            "training_seed": seed,
            "episodes": episodes,
        }
        payloads.append(payload)
        np.savez_compressed(
            path,
            payload_json=np.asarray(
                json.dumps(
                    payload,
                    sort_keys=True,
                    allow_nan=False,
                ),
                dtype=np.str_,
            ),
        )
        rows.append(
            {
                "treatment": "C",
                "training_seed": seed,
                "sampled_trace_path": str(path),
            }
        )

    selection = (
        first_strike_campaign._select_representative_traces_from_payloads(
            rows,
            payloads,
        )
    )
    by_outcome = {
        bool(trace["overall_success"]): trace
        for trace in selection["traces"]
        if trace["arm"] == "C"
    }

    assert selection["selection_rule"] == (
        "lexicographically_smallest_training_seed_env_id_episode_ordinal"
    )
    assert (
        by_outcome[True]["training_seed"],
        by_outcome[True]["env_id"],
        by_outcome[True]["episode_ordinal"],
    ) == (0, 2, 0)
    assert by_outcome[True]["first_strike"]["v_precontact_m_s"] == 10.0
    assert (
        by_outcome[False]["training_seed"],
        by_outcome[False]["env_id"],
        by_outcome[False]["episode_ordinal"],
    ) == (1, 2, 0)
    assert by_outcome[True]["source_episode_id"] == shared_legacy_id
    assert by_outcome[False]["source_episode_id"] == shared_legacy_id
    assert by_outcome[True]["episode_id"] != by_outcome[False]["episode_id"]
    assert "seed0" in by_outcome[True]["episode_id"]
    assert "seed1" in by_outcome[False]["episode_id"]

    with np.load(rows[1]["sampled_trace_path"], allow_pickle=False) as saved:
        source = json.loads(str(saved["payload_json"]))["episodes"][2]
    assert source["episode_id"] == shared_legacy_id
    assert "training_seed" not in source
    assert "source_episode_id" not in source


def test_representative_figure_reports_empty_outcome_strata():
    incomplete = [
        _literal_trace(arm=arm, reason="success", overall_success=True)
        for arm in ARM_TASKS
    ]
    omissions = [
        {
            "arm": arm,
            "outcome": "failure",
            "overall_success": False,
            "reason": "no sampled episodes in outcome stratum",
        }
        for arm in ARM_TASKS
    ]

    figure = build_trajectory_figure(
        incomplete,
        omitted_strata=omissions,
    )

    assert "Omitted empty strata" in figure.layout.title.text
    for arm in ARM_TASKS:
        assert f"{arm} failure" in figure.layout.title.text


def test_representative_top_view_uses_equal_spatial_units():
    traces = [
        _literal_trace(arm=arm, reason="success", overall_success=True)
        for arm in ARM_TASKS
    ]
    omissions = [
        {
            "arm": arm,
            "outcome": "failure",
            "overall_success": False,
            "reason": "no sampled episodes in outcome stratum",
        }
        for arm in ARM_TASKS
    ]

    figure = build_trajectory_figure(traces, omitted_strata=omissions)

    # The x-y top view is subplot x2/y2.  Tying y2 to the bottom-row time
    # axis x3 expands metres to a multi-second range and makes the path flat.
    assert figure.layout.yaxis2.scaleanchor == "x2"


def test_no_contact_representative_skips_only_contact_aligned_panels():
    failure = _literal_trace(
        arm="D-prime",
        reason="none",
        overall_success=False,
        no_contact=True,
    )
    omissions = [
        {
            "arm": arm,
            "outcome": "success" if success else "failure",
            "overall_success": success,
            "reason": "no sampled episodes in outcome stratum",
        }
        for arm in ARM_TASKS
        for success in (False, True)
        if (arm, success) != ("D-prime", False)
    ]

    figure = build_trajectory_figure([failure], omitted_strata=omissions)

    # A no-contact timeout belongs in the state-path panels, but it has no
    # meaningful contact-aligned force/depth/impulse timeline.  Its recorded
    # nonzero reward stream is absolute-time evidence and must not be hidden.
    assert not any(
        getattr(trace, "xaxis", None) in {"x3", "x4"}
        for trace in figure.data
    )
    assert any(
        getattr(trace, "xaxis", None) == "x5"
        for trace in figure.data
    )


def test_all_zero_no_contact_payout_is_omitted_with_disclosure():
    failure = _literal_trace(
        arm="D-prime",
        reason="none",
        overall_success=False,
        no_contact=True,
    )
    failure["reward"]["impact_payout"] = [0.0] * 200
    failure["reward"]["delivered_payout"] = [0.0] * 200
    omissions = [
        {
            "arm": arm,
            "outcome": "success" if success else "failure",
            "overall_success": success,
            "reason": "no sampled episodes in outcome stratum",
        }
        for arm in ARM_TASKS
        for success in (False, True)
        if (arm, success) != ("D-prime", False)
    ]

    figure = build_trajectory_figure([failure], omitted_strata=omissions)

    assert not any(
        getattr(trace, "xaxis", None) == "x5"
        for trace in figure.data
    )
    assert "Omitted all-zero no-contact payout timelines: 1" in (
        figure.layout.title.text
    )


def test_fixed_reset_xz_grid_uses_one_common_coordinate_and_spatial_window():
    rows = []
    payloads = []
    for arm in ARM_TASKS:
        for training_seed in range(8):
            trace = _literal_trace(arm=arm)
            trace["episode_id"] = f"{arm}-seed{training_seed}-env0-episode0"
            trace["training_seed"] = training_seed
            rows.append({"treatment": arm, "training_seed": training_seed})
            payloads.append(
                {
                    "treatment": arm,
                    "training_seed": training_seed,
                    "nail_geometry": copy.deepcopy(trace["nail_geometry"]),
                    "episodes": [trace],
                }
            )

    figure = build_fixed_reset_xz_grid(rows, payloads)

    assert len(figure.axes) == 32
    assert len({axis.get_xlim() for axis in figure.axes}) == 1
    assert len({axis.get_ylim() for axis in figure.axes}) == 1
    assert all(axis.get_aspect() == 1.0 for axis in figure.axes)
    assert [axis.get_title().splitlines()[0] for axis in figure.axes[:8]] == [
        f"C · seed {seed}" for seed in range(8)
    ]
    assert all("env 0 · episode 0" in text.get_text() for text in figure.texts)


def test_fixed_reset_grid_excludes_decoys_and_labels_actual_event_geometry():
    rows = []
    payloads = []
    for arm in ARM_TASKS:
        for training_seed in range(8):
            trace = _literal_trace(arm=arm)
            trace["episode_id"] = f"{arm}-seed{training_seed}-env0-episode0"
            if (arm, training_seed) == ("E", 0):
                trace["first_strike"]["productive"] = False
                trace["first_strike"]["v_precontact_m_s"] = 2.5
                trace["physical"]["head_position_m"][30][:2] = [0.509, 0.012]
                decoy = copy.deepcopy(trace)
                decoy["episode_id"] = "E-seed0-env1-episode0"
                decoy["env_id"] = 1
                decoy["first_strike"]["v_precontact_m_s"] = 9.99
            else:
                decoy = None
            trace["training_seed"] = training_seed
            rows.append({"treatment": arm, "training_seed": training_seed})
            payloads.append(
                {
                    "treatment": arm,
                    "training_seed": training_seed,
                    "nail_geometry": copy.deepcopy(trace["nail_geometry"]),
                    "episodes": ([decoy] if decoy is not None else []) + [trace],
                }
            )

    figure = build_fixed_reset_xz_grid(rows, payloads)
    axis = figure.axes[first_strike_campaign.ARM_ORDER.index("E") * 8]

    assert "offset 15.0 mm" in axis.get_title()
    assert "v_pre 2.50 m/s" in axis.get_title()
    assert "9.99" not in axis.get_title()
    disclosure = figure._suptitle.get_text()
    assert "normalized episode time" in disclosure
    assert "radial x-y offset" in disclosure
    assert "nail axis (x only; z not stored)" in disclosure


def test_fixed_reset_grid_requires_complete_campaign_and_streams_payloads():
    rows = []
    payloads = []
    for arm in ARM_TASKS:
        for training_seed in range(8):
            trace = _literal_trace(arm=arm)
            trace["training_seed"] = training_seed
            rows.append({"treatment": arm, "training_seed": training_seed})
            payloads.append(
                {
                    "treatment": arm,
                    "training_seed": training_seed,
                    "episodes": [trace],
                }
            )

    with pytest.raises(ValueError, match="complete 4×8"):
        build_fixed_reset_xz_grid(rows[:-1], payloads[:-1])

    class NoLengthHintIterator:
        def __init__(self, values):
            self.values = iter(values)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.values)

        def __length_hint__(self):
            raise AssertionError("payload iterator must not be materialized")

    figure = build_fixed_reset_xz_grid(rows, NoLengthHintIterator(payloads))

    assert len(figure.axes) == 32


def test_campaign_figure_coerces_csv_string_numeric_fields():
    rows = [
        {
            "treatment": arm,
            "training_seed": "0",
            "first_strike_useful_speed_mean_sampled": "1.25",
            "first_strike_success_rate_sampled": "0.5",
            "impact_return_discounted_mean_sampled": "2.0",
            "delivered_return_discounted_mean_sampled": "3.0",
        }
        for arm in ARM_TASKS
    ]

    figure = build_campaign_figure(rows)

    plotted_y = [
        value
        for trace in figure.data
        for value in trace.y
    ]
    assert plotted_y == [
        value
        for _ in ARM_TASKS
        for value in (1.25, 0.5, 2.0, 3.0)
    ]
    assert all(isinstance(value, float) for value in plotted_y)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda omissions: omissions[0].__setitem__("outcome", "success"),
        lambda omissions: omissions[0].__setitem__("reason", "invented"),
        lambda omissions: omissions.append(copy.deepcopy(omissions[0])),
    ),
)
def test_representative_figure_rejects_noncanonical_or_duplicate_omissions(
    mutate,
):
    traces = [
        _literal_trace(arm=arm, reason="success", overall_success=True)
        for arm in ARM_TASKS
    ]
    omissions = [
        {
            "arm": arm,
            "outcome": "failure",
            "overall_success": False,
            "reason": "no sampled episodes in outcome stratum",
        }
        for arm in ARM_TASKS
    ]
    mutate(omissions)

    with pytest.raises(ValueError, match="omitted"):
        build_trajectory_figure(traces, omitted_strata=omissions)


def test_seed_qualification_preserves_raw_seed_presence_and_rejects_shadow_fields():
    raw = _literal_trace(arm="C")
    without_raw_seed = first_strike_campaign._seed_qualified_representative_trace(
        raw,
        training_seed=0,
    )
    raw_with_seed = copy.deepcopy(raw)
    raw_with_seed["training_seed"] = 0
    with_raw_seed = first_strike_campaign._seed_qualified_representative_trace(
        raw_with_seed,
        training_seed=0,
    )

    assert without_raw_seed["source_training_seed"] is None
    assert with_raw_seed["source_training_seed"] == 0
    assert without_raw_seed != with_raw_seed

    for reserved in ("source_episode_id", "source_training_seed"):
        shadowed = copy.deepcopy(raw)
        shadowed[reserved] = "attacker-controlled"
        with pytest.raises(ValueError, match="reserved"):
            first_strike_campaign._seed_qualified_representative_trace(
                shadowed,
                training_seed=0,
            )


def test_plotly_trajectory_and_video_overlay_keep_labels_and_trace_digest(tmp_path):
    traces = [
        _literal_trace(
            arm=arm,
            reason=reason,
            overall_success=reason == "success",
        )
        for arm in ("C", "D-prime", "F", "E")
        for reason in ("success", "window")
    ]
    # Plot markers follow the tracker-accepted onset, never an earlier raw hit.
    traces[0]["physical"]["contact"][0] = True
    traces[0]["trace_digest"] = _literal_digest(
        {
            "physical": traces[0]["physical"],
            "event_trace": traces[0]["event_trace"],
        }
    )
    figure = build_trajectory_figure(traces)

    labels = " ".join(
        [trace.name or "" for trace in figure.data]
        + [annotation.text for annotation in figure.layout.annotations]
    )
    for arm, task in ARM_TASKS.items():
        assert arm in labels
        assert task in labels
    assert "Frozen nail axis/radius" in labels
    assert "pre-integration depth" in labels
    assert "tracker post-integration depth" in labels
    assert "tracker success (post-depth transition)" in labels
    assert all(trace.type != "scattergl" for trace in figure.data)
    contact_marker = next(
        item
        for item in figure.data
        if item.name == f"{traces[0]['arm']} — {traces[0]['task']} first contact"
    )
    assert list(contact_marker.x) == pytest.approx(
        [traces[0]["physical"]["head_position_m"][30][0]]
    )
    cumulative = next(
        item
        for item in figure.data
        if item.name
        == f"{traces[0]['arm']} — {traces[0]['task']} cumulative event impulse"
    )
    assert list(cumulative.y) == pytest.approx(
        traces[0]["event_trace"]["event_cumulative_impulse_n_s"]
    )
    assert cumulative.y[-1] == cumulative.y[31]

    source = traces[0]
    video = tmp_path / "representative.mp4"
    video.write_bytes(b"literal-rendered-video")
    output = tmp_path / "representative_overlay.html"
    write_video_overlay_html(
        source,
        video_artifact={
            "path": video,
            "episode_id": source["episode_id"],
            "trace_digest": source["trace_digest"],
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "timing": {
                "frame_timestamps_s": [
                    index * source["physics_dt_s"]
                    for index in range(
                        len(source["physical"]["contact"])
                    )
                ]
            },
        },
        output_path=output,
    )
    html_text = output.read_text()
    assert source["trace_digest"] in html_text
    assert source["arm"] in html_text
    assert source["task"] in html_text
    assert "state ground truth" in html_text.lower()
    assert '<div id="state-overlay"' in html_text
    assert "position: absolute" in html_text
    assert "Plotly.newPlot" in html_text
    assert 'video.addEventListener("timeupdate", syncPlotlyOverlay)' in html_text
    assert "<canvas" not in html_text
    assert hashlib.sha256(video.read_bytes()).hexdigest() in html_text
    overlay_line = next(
        line.strip()
        for line in html_text.splitlines()
        if line.strip().startswith("const overlay = ")
    )
    overlay = json.loads(
        overlay_line.removeprefix("const overlay = ").removesuffix(";")
    )
    assert overlay["contact_index"] == 30
    assert overlay["success_index"] == 31
    assert overlay["frame_trace_indices"][30:32] == [30, 31]

    with pytest.raises(ValueError, match="explicit render timing"):
        write_video_overlay_html(
            source,
            video_artifact={
                "path": video,
                "episode_id": source["episode_id"],
                "trace_digest": source["trace_digest"],
                "video_sha256": hashlib.sha256(
                    video.read_bytes()
                ).hexdigest(),
            },
            output_path=tmp_path / "missing-timing.html",
        )

    with pytest.raises(ValueError, match="video trace digest"):
        write_video_overlay_html(
            source,
            video_artifact={
                "path": video,
                "episode_id": source["episode_id"],
                "trace_digest": "wrong",
                "video_sha256": hashlib.sha256(
                    video.read_bytes()
                ).hexdigest(),
                "timing": {
                    "frame_timestamps_s": [0.0, 0.002],
                },
            },
            output_path=tmp_path / "wrong.html",
        )
    original_video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    video.write_bytes(b"unbound-replacement-video")
    with pytest.raises(ValueError, match="video artifact SHA-256"):
        write_video_overlay_html(
            source,
            video_artifact={
                "path": video,
                "episode_id": source["episode_id"],
                "trace_digest": source["trace_digest"],
                "video_sha256": original_video_sha,
                "timing": {
                    "frame_timestamps_s": [0.0, 0.002],
                },
            },
            output_path=tmp_path / "wrong-video-bytes.html",
        )


def test_fixture_report_html_and_video_overlays_remain_digest_bound(
    tmp_path, campaign_rows_fixture
):
    traces = _artifact_representatives(campaign_rows_fixture)
    video_artifacts = {}
    for trace in traces:
        video = tmp_path / f"{trace['episode_id']}.mp4"
        video.write_bytes(f"video-{trace['episode_id']}".encode())
        video_artifacts[trace["episode_id"]] = {
            "path": video,
            "episode_id": trace["episode_id"],
            "trace_digest": trace["trace_digest"],
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "timing": {"frame_timestamps_s": [0.0, 0.002]},
        }

    result = generate_report(
        campaign_rows_fixture,
        traces,
        output_dir=tmp_path,
        video_artifacts=video_artifacts,
        bootstrap_samples=1000,
    )

    for key in ("campaign_html", "trajectory_html"):
        html_text = Path(result["artifacts"][key]).read_text()
        for arm, task in ARM_TASKS.items():
            assert arm in html_text
            assert task in html_text
    assert Path(result["artifacts"]["fixed_reset_xz_grid_png"]).is_file()
    by_episode = {trace["episode_id"]: trace for trace in traces}
    for overlay in result["artifacts"]["video_overlays"]:
        source = by_episode[overlay["episode_id"]]
        assert overlay["trace_digest"] == source["trace_digest"]
        overlay_html = Path(overlay["html"]).read_text()
        assert source["trace_digest"] in overlay_html
        assert source["arm"] in overlay_html
        assert source["task"] in overlay_html
        assert '<div id="state-overlay"' in overlay_html
        assert "Plotly.newPlot" in overlay_html
        assert "<canvas" not in overlay_html
        assert overlay["video_sha256"] == hashlib.sha256(
            Path(overlay["video"]).read_bytes()
        ).hexdigest()


def test_report_keeps_seed_analysis_valid_when_a_representative_stratum_is_empty(
    tmp_path, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)

    def make_c_overall_success(trace):
        trace["overall_success"] = True
        trace["episode_peak_lambda"] = [0.1] * 6
        trace["episode_delivered_accumulator_n_s"] = max(
            0.1, float(trace["episode_delivered_accumulator_n_s"])
        )
        trace["episode_depth_m"] = 0.032

    _rewrite_campaign_artifacts(
        rows,
        treatment="C",
        output_dir=tmp_path,
        transform=make_c_overall_success,
    )

    result = generate_report(
        rows,
        output_dir=tmp_path,
        bootstrap_samples=1000,
    )

    assert result["analysis"]["valid"] is True
    assert result["representative_trajectories"]["omitted_strata"] == [
        {
            "arm": "C",
            "outcome": "failure",
            "overall_success": False,
            "reason": "no sampled episodes in outcome stratum",
        }
    ]
    selected = result["representative_trajectories"]["selected"]
    assert not any(
        row["arm"] == "C" and row["overall_success"] is False
        for row in selected
    )
    assert "C failure" in Path(
        result["artifacts"]["trajectory_html"]
    ).read_text()


def test_report_rejects_sampled_artifact_mutated_after_campaign_analysis(
    monkeypatch, tmp_path, campaign_rows_fixture
):
    rows = copy.deepcopy(campaign_rows_fixture)
    target = next(
        row
        for row in rows
        if row["treatment"] == "C" and row["training_seed"] == 0
    )
    source_path = Path(target["sampled_trace_path"])
    copied_path = tmp_path / source_path.name
    copied_path.write_bytes(source_path.read_bytes())
    target["sampled_trace_path"] = str(copied_path)
    real_analyze = plot_first_strike_campaign.analyze_campaign

    def analyze_then_mutate(*args, **kwargs):
        result = real_analyze(*args, **kwargs)
        with np.load(copied_path, allow_pickle=False) as saved:
            payload = json.loads(str(saved["payload_json"]))
        payload["episodes"][0]["reward"]["impact_payout"] = [999_999.0]
        payload_without_digest = dict(payload)
        payload_without_digest.pop("payload_digest")
        payload["payload_digest"] = _literal_digest(payload_without_digest)
        np.savez_compressed(
            copied_path,
            payload_json=np.asarray(
                json.dumps(payload, sort_keys=True, allow_nan=False),
                dtype=np.str_,
            ),
        )
        return result

    monkeypatch.setattr(
        plot_first_strike_campaign,
        "analyze_campaign",
        analyze_then_mutate,
    )

    with pytest.raises(
        ValueError,
        match="sampled trace artifact SHA-256 mismatch",
    ):
        plot_first_strike_campaign.generate_report(
            rows,
            output_dir=tmp_path,
            bootstrap_samples=1000,
        )


def test_report_rejects_representative_not_in_validated_campaign_artifact(
    tmp_path, campaign_rows_fixture
):
    traces = _artifact_representatives(campaign_rows_fixture)
    traces[0]["physical"]["head_position_m"][0][0] += 0.001
    traces[0]["trace_digest"] = _literal_digest(
        {
            "physical": traces[0]["physical"],
            "event_trace": traces[0]["event_trace"],
        }
    )

    with pytest.raises(ValueError, match="not exact members"):
        generate_report(
            campaign_rows_fixture,
            traces,
            output_dir=tmp_path,
            bootstrap_samples=1000,
        )


_TASK5_ARM_CONTRACT = {
    "C": (
        "Unitree-Z1-Hammer-CaT-Impulse",
        "legacy_50hz_repeated_credit",
        "c",
    ),
    "D-prime": (
        "Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy",
        "legacy_50hz_first_event_censored_one_shot",
        "dprime",
    ),
    "F": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
        "event_500hz_first_event_linear_delivered",
        "f",
    ),
    "E": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event",
        "event_500hz_first_event_saturated_delivered",
        "e",
    ),
}


def test_frozen_campaign_matrix_is_the_exact_32_job_training_contract():
    rows = list(FROZEN_CAMPAIGN_MATRIX)

    assert len(rows) == 32
    assert len({row["run_name"] for row in rows}) == 32
    assert {(row["arm"], row["training_seed"]) for row in rows} == {
        (arm, seed) for arm in _TASK5_ARM_CONTRACT for seed in range(8)
    }

    for row in rows:
        task, semantics, slug = _TASK5_ARM_CONTRACT[row["arm"]]
        seed = row["training_seed"]
        assert row == {
            "arm": row["arm"],
            "task": task,
            "arm_semantics": semantics,
            "training_seed": seed,
            "impact_weight": 8.0,
            "delivered_weight": 2.0,
            "imp_max_p": 0.0,
            "impedance_mode": "fixed",
            "fixed_action_signature": EXPECTED_FIXED_ACTION_SIGNATURE,
            "impulse_limits_n_m_s": (
                1.64,
                3.28,
                1.64,
                1.64,
                1.64,
                1.64,
            ),
            "training_iterations": 500,
            "training_num_envs": 4096,
            "checkpoint_filename": "model_499.pt",
            "run_name": f"fsr4x8_{slug}_seed{seed}",
            "expected_checkpoint_glob": (
                "logs/rsl_rl/z1_hammer/"
                f"*_fsr4x8_{slug}_seed{seed}/model_499.pt"
            ),
            "eval_num_envs": 256,
            "eval_episodes_per_env": 2,
            "eval_episode_count": 512,
            "eval_episode_len_s": 4.0,
            "eval_mean_nsteps": 400,
            "eval_completion_rule": "first_two_completions_per_environment",
            "eval_stochastic_actions": True,
            "eval_reset_noise_rad": (-0.05, 0.05),
            "eval_base_rng_seed": 2_026_072_900,
            "eval_reset_rng_seed": 2_036_072_919,
            "eval_observation_rng_seed": 2_046_072_933,
            "eval_action_rng_seed": 2_056_072_941,
            "eval_actor_observation_corruption": True,
            "eval_critic_observation_corruption": False,
        }


@pytest.mark.parametrize(
    ("field", "invalid_value", "reason"),
    [
        ("task", "Unitree-Z1-Hammer-CaT-Impulse", "task"),
        ("arm_semantics", "legacy_50hz_repeated_credit", "arm semantics"),
        ("impact_weight", 7.9, "impact weight"),
        ("delivered_weight", 2.1, "delivered weight"),
        ("imp_max_p", 0.1, "imp_max_p"),
        ("impedance_mode", "variable", "fixed impedance"),
        ("fixed_action_signature", ("set_gains",), "fixed action"),
        ("impulse_limits_n_m_s", (0.1,) * 6, "impulse limits"),
        ("training_iterations", 499, "500 iterations"),
        ("training_num_envs", 256, "4096 environments"),
        ("checkpoint_filename", "model_500.pt", "model_499.pt"),
        ("run_name", "duplicate", "run name"),
        ("expected_checkpoint_glob", "invented/model.pt", "checkpoint location"),
        ("eval_num_envs", 128, "256 evaluation environments"),
        ("eval_episodes_per_env", 1, "two completions"),
        ("eval_episode_count", 256, "512 sampled episodes"),
        ("eval_episode_len_s", 3.0, "4.0 second evaluation horizon"),
        ("eval_mean_nsteps", 399, "400 mean-action steps"),
        ("eval_completion_rule", "fixed_control_steps", "completion rule"),
        ("eval_stochastic_actions", False, "stochastic actions"),
        ("eval_reset_noise_rad", (0.0, 0.0), "reset noise"),
        ("eval_base_rng_seed", 42, "base evaluator RNG"),
        ("eval_reset_rng_seed", 0, "reset RNG"),
        ("eval_observation_rng_seed", 0, "observation RNG"),
        ("eval_action_rng_seed", 0, "action RNG"),
        (
            "eval_actor_observation_corruption",
            False,
            "actor observation corruption",
        ),
        (
            "eval_critic_observation_corruption",
            True,
            "critic observation corruption",
        ),
    ],
)
def test_frozen_campaign_validator_rejects_any_contract_drift(
    field, invalid_value, reason
):
    rows = copy.deepcopy(list(FROZEN_CAMPAIGN_MATRIX))
    rows[8][field] = invalid_value

    with pytest.raises(ValueError, match=reason):
        validate_frozen_campaign_matrix(rows)


def test_frozen_campaign_validator_rejects_missing_duplicate_or_extra_jobs():
    rows = copy.deepcopy(list(FROZEN_CAMPAIGN_MATRIX))
    with pytest.raises(ValueError, match="exactly 32 jobs"):
        validate_frozen_campaign_matrix(rows[:-1])

    duplicate = copy.deepcopy(rows)
    duplicate[1]["training_seed"] = 0
    with pytest.raises(ValueError, match="arm/seed"):
        validate_frozen_campaign_matrix(duplicate)

    duplicate_name = copy.deepcopy(rows)
    duplicate_name[1]["run_name"] = duplicate_name[0]["run_name"]
    with pytest.raises(ValueError, match="run name"):
        validate_frozen_campaign_matrix(duplicate_name)


_ACCEPTED_MANIFEST_HEADER = (
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


def _write_filled_accepted_manifest(tmp_path: Path) -> tuple[Path, list[dict]]:
    rows = []
    for frozen in FROZEN_CAMPAIGN_MATRIX:
        attempt_path = (
            tmp_path
            / "logs"
            / f"2026-07-25_12-00-00_{frozen['run_name']}"
        )
        attempt_path.mkdir(parents=True)
        checkpoint_path = attempt_path / "model_499.pt"
        checkpoint_path.write_bytes(
            f"{frozen['arm']}/seed{frozen['training_seed']}".encode()
        )
        rows.append(
            {
                "arm": frozen["arm"],
                "training_seed": frozen["training_seed"],
                "task": frozen["task"],
                "run_name": frozen["run_name"],
                "attempt_path": str(attempt_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint_path.read_bytes()
                ).hexdigest(),
                "training_code_revision": "a" * 40,
                "training_asset_revision": "b" * 40,
                "disposition": "accepted",
                "reason": "completed frozen attempt",
            }
        )
    path = tmp_path / "accepted_attempts.tsv"
    lines = ["\t".join(_ACCEPTED_MANIFEST_HEADER)]
    lines.extend(
        "\t".join(str(row[field]) for field in _ACCEPTED_MANIFEST_HEADER)
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n")
    return path, rows


def test_accepted_attempt_manifest_validates_literal_32_row_selection(tmp_path):
    path, rows = _write_filled_accepted_manifest(tmp_path)

    accepted = load_accepted_attempt_manifest(
        path,
        expected_code_revision="a" * 40,
        expected_asset_revision="b" * 40,
    )

    assert len(accepted) == 32
    assert [
        (row["arm"], row["training_seed"]) for row in accepted
    ] == [
        (row["arm"], row["training_seed"]) for row in FROZEN_CAMPAIGN_MATRIX
    ]
    assert {row["checkpoint_path"] for row in accepted} == {
        row["checkpoint_path"] for row in rows
    }


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "exactly 32 rows"),
        ("duplicate", "duplicate accepted arm/seed"),
        ("task", "exact task mismatch"),
        ("hash", "checkpoint SHA-256 mismatch"),
    ],
)
def test_accepted_attempt_manifest_rejects_missing_duplicate_task_or_hash(
    tmp_path, mutation, reason
):
    path, rows = _write_filled_accepted_manifest(tmp_path)
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows[-1]["arm"] = rows[0]["arm"]
        rows[-1]["training_seed"] = rows[0]["training_seed"]
    elif mutation == "task":
        rows[0]["task"] = ARM_TASKS["E"]
    else:
        rows[0]["checkpoint_sha256"] = "f" * 64
    lines = ["\t".join(_ACCEPTED_MANIFEST_HEADER)]
    lines.extend(
        "\t".join(str(row[field]) for field in _ACCEPTED_MANIFEST_HEADER)
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match=reason):
        load_accepted_attempt_manifest(
            path,
            expected_code_revision="a" * 40,
            expected_asset_revision="b" * 40,
        )


def test_accepted_attempt_manifest_rejects_duplicate_checkpoint_bytes(
    tmp_path,
):
    path, rows = _write_filled_accepted_manifest(tmp_path)
    source = Path(rows[0]["checkpoint_path"])
    duplicate = Path(rows[1]["checkpoint_path"])
    duplicate.write_bytes(source.read_bytes())
    rows[1]["checkpoint_sha256"] = hashlib.sha256(
        duplicate.read_bytes()
    ).hexdigest()
    lines = ["\t".join(_ACCEPTED_MANIFEST_HEADER)]
    lines.extend(
        "\t".join(str(row[field]) for field in _ACCEPTED_MANIFEST_HEADER)
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="checkpoint SHA-256 values"):
        load_accepted_attempt_manifest(
            path,
            expected_code_revision="a" * 40,
            expected_asset_revision="b" * 40,
        )


def test_frozen_campaign_matrix_matches_live_reward_and_fixed_gain_configs():
    reward_contract = {
        "C": ("ImpactProgressTerm", "DeliveredImpulseTerm", None),
        "D-prime": (
            "FirstStrikeLegacyImpactRewardTerm",
            "FirstStrikeLegacyDeliveredRewardTerm",
            None,
        ),
        "F": (
            "FirstStrikeImpactRewardTerm",
            "FirstStrikeDeliveredRewardTerm",
            False,
        ),
        "E": (
            "FirstStrikeImpactRewardTerm",
            "FirstStrikeDeliveredRewardTerm",
            True,
        ),
    }
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

    actuator_signatures = set()
    for arm in _TASK5_ARM_CONTRACT:
        cfg = eval_impulse.load_env_cfg(ARM_TASKS[arm], play=False)
        impact = cfg.rewards["impact_progress"]
        delivered = cfg.rewards["delivered_impulse"]
        want_impact, want_delivered, want_saturate = reward_contract[arm]

        assert impact.func.__name__ == want_impact
        assert delivered.func.__name__ == want_delivered
        assert delivered.params.get("saturate") is want_saturate
        assert (impact.weight, delivered.weight) == (8.0, 2.0)
        assert cfg.metrics["cat_soft"].params["imp_max_p"] == 0.0
        assert tuple(cfg.metrics["cat_soft"].params["imp_limit"]) == (
            1.64,
            3.28,
            1.64,
            1.64,
            1.64,
            1.64,
        )

        signature = tuple(
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
        actuator_signatures.add(signature)
        assert signature == expected_actuators

    assert len(actuator_signatures) == 1


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


def _fake_git(path: Path) -> None:
    _write_executable(
        path,
        """#!/bin/sh
case "$*" in
  *"safe_impact_manipulation"*"rev-parse"*) printf '%s\\n' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb ;;
  *"rev-parse"*) printf '%s\\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
  *"status"*) exit 0 ;;
esac
""",
    )


def _run_training_launcher(root: Path, **overrides) -> tuple[subprocess.CompletedProcess, str]:
    capture = root / "python-calls.txt"
    fake_home = root.parent / f"{root.name}-home"
    (root.parent / "safe_impact_manipulation").mkdir(parents=True, exist_ok=True)
    _write_executable(
        root / ".venv" / "bin" / "python",
        """#!/bin/sh
mkdir -p "$WARP_CACHE_PATH"
touch "$WARP_CACHE_PATH/probe"
printf '%s\\n' "$*" >> "$CAPTURE"
exit 0
""",
    )
    _fake_git(root / "bin" / "git")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{root / 'bin'}:/usr/bin:/bin",
            "HOME": str(fake_home),
            "SLURM_SUBMIT_DIR": str(root),
            "SLURM_ARRAY_TASK_ID": "3",
            "SLURM_JOB_ID": "123",
            "CAMPAIGN": "fsr4x8",
            "SEEDS": "0 1 2 3 4 5 6 7",
            "SINGLE_TASK": ARM_TASKS["D-prime"],
            "SINGLE_SHORT": "dprime",
            "IMPACT_W": "8",
            "DELIVERED_W": "2",
            "ITERS": "500",
            "EXPECTED_CODE_REVISION": "a" * 40,
            "EXPECTED_ASSET_REVISION": "b" * 40,
            "CAPTURE": str(capture),
        }
    )
    env.pop("NAIL_DRIVEN_W", None)
    env.pop("ASSET_REPO", None)
    env.update({key: str(value) for key, value in overrides.items()})
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts" / "slurm" / "vega_train.sbatch")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, capture.read_text() if capture.exists() else ""


def test_training_launcher_executes_one_exact_frozen_matrix_row(tmp_path):
    root = tmp_path / "valid"
    result, calls = _run_training_launcher(root)

    assert result.returncode == 0, result.stderr + result.stdout
    train_calls = [line for line in calls.splitlines() if "scripts/train.py" in line]
    assert len(train_calls) == 1
    call = train_calls[0]
    assert f"scripts/train.py {ARM_TASKS['D-prime']}" in call
    assert "--agent.run-name fsr4x8_dprime_seed3" in call
    assert "--agent.seed 3" in call
    assert "--agent.max-iterations 500" in call
    assert "--env.scene.num-envs 4096" in call
    assert "--env.metrics.cat-soft.params.imp-max-p 0" in call
    assert "--env.rewards.impact-progress.weight 8" in call
    assert "--env.rewards.delivered-impulse.weight 2" in call
    assert not (root / ".warp_cache").exists()
    assert (
        tmp_path / "valid-home" / ".cache" / "unitree_rl_mjlab" / "warp" / "probe"
    ).is_file()


def test_training_launcher_rejects_legacy_reward_override(tmp_path):
    result, calls = _run_training_launcher(
        tmp_path / "invalid",
        NAIL_DRIVEN_W="0.5",
    )

    assert result.returncode == 2
    assert "NAIL_DRIVEN_W" in result.stdout
    assert "scripts/train.py" not in calls


def test_training_launcher_rejects_wrong_asset_repo_or_expected_revision(tmp_path):
    wrong_asset = tmp_path / "wrong-asset"
    wrong_asset.mkdir()
    wrong_repo, wrong_repo_calls = _run_training_launcher(
        tmp_path / "wrong-repo",
        ASSET_REPO=wrong_asset,
    )
    assert wrong_repo.returncode == 2
    assert "canonical sibling" in wrong_repo.stdout
    assert "scripts/train.py" not in wrong_repo_calls

    wrong_revision, wrong_revision_calls = _run_training_launcher(
        tmp_path / "wrong-revision",
        EXPECTED_CODE_REVISION="b" * 40,
    )
    assert wrong_revision.returncode == 2
    assert "expected code revision" in wrong_revision.stdout
    assert "scripts/train.py" not in wrong_revision_calls


def _run_eval_launcher(
    root: Path,
    *,
    precreate_attempt: bool = False,
    attempt_label: str = "attempt1",
    eval_root: Path | None = None,
    env_overrides: dict | None = None,
) -> tuple[subprocess.CompletedProcess, list[str]]:
    capture = root / "eval-calls.txt"
    fake_home = root.parent / f"{root.name}-home"
    (root.parent / "safe_impact_manipulation").mkdir(parents=True, exist_ok=True)
    _write_executable(
        root / ".venv" / "bin" / "python",
        """#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "evaluation.analysis.first_strike_campaign" ]; then
  PYTHONPATH="$TASK5_SOURCE_ROOT" exec "$TASK5_REAL_PYTHON" "$@"
fi
mkdir -p "$WARP_CACHE_PATH"
touch "$WARP_CACHE_PATH/probe"
exit 0
""",
    )
    _fake_git(root / "bin" / "git")
    _write_executable(
        root / "scripts" / "eval_impulse.sh",
        """#!/bin/sh
count=$(printf '%s' "$CKPTS" | grep -c 'model_499.pt')
[ -f "$OUT/accepted_attempts.tsv" ] || exit 9
bad=$(printf '%s' "$CKPTS" | awk -F: 'NF && (NF != 3 || length($3) != 64) {n++} END{print n+0}')
[ "$bad" -eq 0 ] || exit 10
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' \
  "$TASK" "$FRESH" "$OUT" "$count" "$NENVS" "$NSTEPS" "$EPLEN" "$SEED" \
  "$RUN_LEGACY_DIAG" "$ACCEPTED_MANIFEST_SHA256" \
  "$TRAINING_CODE_REVISION" "$TRAINING_ASSET_REVISION" >> "$CAPTURE"
exit 0
""",
    )
    if eval_root is None:
        eval_root = root.parent / f"{root.name}-durable-eval"
    internal_manifest, _ = _write_filled_accepted_manifest(root)
    accepted_manifest = root.parent / f"{root.name}-accepted-attempts.tsv"
    accepted_manifest.write_text(internal_manifest.read_text())
    if precreate_attempt:
        (eval_root / "fsr4x8" / attempt_label).mkdir(parents=True)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{root / 'bin'}:/usr/bin:/bin",
            "HOME": str(fake_home),
            "SLURM_SUBMIT_DIR": str(root),
            "SLURM_JOB_ID": "456",
            "CAMPAIGN": "fsr4x8",
            "EVAL_ATTEMPT": attempt_label,
            "EVAL_ROOT": str(eval_root),
            "EXPECTED_CODE_REVISION": "a" * 40,
            "EXPECTED_ASSET_REVISION": "b" * 40,
            "ACCEPTED_MANIFEST": str(accepted_manifest),
            "CAPTURE": str(capture),
            "TASK5_SOURCE_ROOT": str(ROOT),
            "TASK5_REAL_PYTHON": sys.executable,
        }
    )
    env.pop("ASSET_REPO", None)
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items()})
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts" / "slurm" / "vega_eval.sbatch")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = capture.read_text().splitlines() if capture.exists() else []
    return result, calls


def test_eval_launcher_uses_four_exact_tasks_and_attempt_specific_output(tmp_path):
    root = tmp_path / "eval"
    eval_root = tmp_path / "durable-eval"
    result, calls = _run_eval_launcher(root, eval_root=eval_root)
    manifest_sha = hashlib.sha256(
        (tmp_path / "eval-accepted-attempts.tsv").read_bytes()
    ).hexdigest()

    assert result.returncode == 0, result.stderr + result.stdout
    assert calls == [
        (
            f"{ARM_TASKS['C']}|0|"
            f"{eval_root / 'fsr4x8' / 'attempt1'}|8|256|400|4.0|"
            f"2026072900|0|{manifest_sha}|{'a' * 40}|{'b' * 40}"
        ),
        (
            f"{ARM_TASKS['D-prime']}|0|"
            f"{eval_root / 'fsr4x8' / 'attempt1'}|8|256|400|4.0|"
            f"2026072900|0|{manifest_sha}|{'a' * 40}|{'b' * 40}"
        ),
        (
            f"{ARM_TASKS['F']}|0|"
            f"{eval_root / 'fsr4x8' / 'attempt1'}|8|256|400|4.0|"
            f"2026072900|0|{manifest_sha}|{'a' * 40}|{'b' * 40}"
        ),
        (
            f"{ARM_TASKS['E']}|0|"
            f"{eval_root / 'fsr4x8' / 'attempt1'}|8|256|400|4.0|"
            f"2026072900|0|{manifest_sha}|{'a' * 40}|{'b' * 40}"
        ),
    ]
    assert not (root / ".warp_cache").exists()
    assert (
        tmp_path / "eval-home" / ".cache" / "unitree_rl_mjlab" / "warp" / "probe"
    ).is_file()
    copied_manifest = (
        eval_root / "fsr4x8" / "attempt1" / "accepted_attempts.tsv"
    )
    assert hashlib.sha256(copied_manifest.read_bytes()).hexdigest() == manifest_sha
    assert (
        eval_root
        / "fsr4x8"
        / "attempt1"
        / "accepted_attempts.tsv.sha256"
    ).read_text().startswith(manifest_sha)


def test_eval_launcher_never_overwrites_a_prior_attempt(tmp_path):
    result, calls = _run_eval_launcher(
        tmp_path / "eval-existing", precreate_attempt=True
    )

    assert result.returncode == 2
    assert "already exists" in result.stdout
    assert calls == []


def test_eval_launcher_rejects_attempt_label_path_traversal(tmp_path):
    result, calls = _run_eval_launcher(
        tmp_path / "eval-traversal",
        attempt_label="..",
    )

    assert result.returncode == 2
    assert "path-safe attempt label" in result.stdout
    assert calls == []


def test_eval_launcher_rejects_output_root_inside_code_repository(tmp_path):
    root = tmp_path / "eval-inside"
    result, calls = _run_eval_launcher(
        root,
        eval_root=root / "unsafe-output",
    )

    assert result.returncode == 2
    assert "outside the code repository" in result.stdout
    assert calls == []


def test_eval_launcher_rejects_output_root_inside_asset_repository(tmp_path):
    root = tmp_path / "eval-asset-output"
    asset_repo = tmp_path / "safe_impact_manipulation"
    result, calls = _run_eval_launcher(
        root,
        eval_root=asset_repo / "unsafe-output",
    )

    assert result.returncode == 2
    assert "outside the asset repository" in result.stdout
    assert calls == []


def test_eval_launcher_rejects_wrong_asset_repo_or_expected_revision(tmp_path):
    wrong_asset = tmp_path / "wrong-eval-asset"
    wrong_asset.mkdir()
    wrong_repo, wrong_repo_calls = _run_eval_launcher(
        tmp_path / "eval-wrong-repo",
        env_overrides={"ASSET_REPO": wrong_asset},
    )
    assert wrong_repo.returncode == 2
    assert "canonical sibling" in wrong_repo.stdout
    assert wrong_repo_calls == []

    wrong_revision, wrong_revision_calls = _run_eval_launcher(
        tmp_path / "eval-wrong-revision",
        env_overrides={"EXPECTED_ASSET_REVISION": "f" * 40},
    )
    assert wrong_revision.returncode == 2
    assert "expected asset revision" in wrong_revision.stdout
    assert wrong_revision_calls == []


def test_eval_launcher_requires_explicit_accepted_manifest_and_never_globs(tmp_path):
    missing, calls = _run_eval_launcher(
        tmp_path / "eval-no-manifest",
        env_overrides={"ACCEPTED_MANIFEST": tmp_path / "missing.tsv"},
    )
    assert missing.returncode == 2
    assert "accepted-attempt manifest" in (missing.stdout + missing.stderr)
    assert calls == []

    source = (ROOT / "scripts" / "slurm" / "vega_eval.sbatch").read_text()
    fsr_block = source.split('if [ "$CAMPAIGN" = "fsr4x8" ]; then', 1)[1].split(
        "\nelse\n", 1
    )[0]
    assert "*_fsr4x8_" not in fsr_block
    assert "eval_fsr_arm c Unitree-Z1-Hammer-CaT-Impulse 0" in fsr_block
    assert "eval_fsr_arm c Unitree-Z1-Hammer-CaT-Impulse 1" not in fsr_block


def test_slurm_logs_and_warp_cache_are_outside_unignored_repo_paths():
    train = (ROOT / "scripts" / "slurm" / "vega_train.sbatch").read_text()
    evaluate = (ROOT / "scripts" / "slurm" / "vega_eval.sbatch").read_text()
    driver = (ROOT / "scripts" / "eval_impulse.sh").read_text()

    assert "#SBATCH --output=logs/" in train
    assert "#SBATCH --output=logs/" in evaluate
    assert 'WARP_CACHE_PATH="$HOME/.cache/unitree_rl_mjlab/warp"' in train
    assert 'WARP_CACHE_PATH="$HOME/.cache/unitree_rl_mjlab/warp"' in evaluate
    assert 'RUN_LEGACY_DIAG="${RUN_LEGACY_DIAG:-1}"' in driver
    assert '--accepted-manifest-sha256 "$ACCEPTED_MANIFEST_SHA256"' in driver
    assert '--training-code-revision "$TRAINING_CODE_REVISION"' in driver
    assert '--training-asset-revision "$TRAINING_ASSET_REVISION"' in driver
    assert '--expected-checkpoint-sha256 "$expected_checkpoint_sha256"' in driver
