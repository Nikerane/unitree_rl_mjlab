import copy
import csv
import hashlib
import json
import sys

import numpy as np
import pytest
from matplotlib.patches import Circle

from evaluation.analysis import presentation3_results as p3
from evaluation.analysis.presentation3_results import (
    assemble_result_row,
    canonical_policy_rows,
    cap_utilization_figure,
    dose_response_figure,
    fixed_impulse_metrics,
    peak_post_integration_qvel,
    precontact_geometry_metrics,
    sampled_first_event_metrics,
    sampled_waypoints_by_contact,
    trajectory_grid_figure,
    validate_artifact_join,
    velocity_cat_figure,
    waypoints_reached_by_contact,
    write_result_package,
)


NAIL_GEOMETRY = {
    "nail_axis": [0.0, 0.0, -1.0],
    "nail_xy_m": [0.0, 0.0],
    "nail_radius_m": 0.012,
    "source_sha256": "9" * 64,
}


def _valid_sampled_contract_inputs():
    task = (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered0"
    )
    checkpoint = "a" * 64
    manifest = "b" * 64
    training_revision = "c" * 40
    evaluation_revision = "d" * 40
    asset_revision = "e" * 40
    campaign_config = "1" * 64
    treatment_config = "2" * 64
    nail_asset = "3" * 64
    caps = [1.64, 3.28, 1.64, 1.64, 1.64, 1.64]
    summary = {
        "name": "presentation3_pvd0_seed2",
        "task": task,
        "treatment": "V+D0",
        "training_seed": "2",
        "checkpoint_sha256": checkpoint,
        "accepted_checkpoint_sha256": checkpoint,
        "accepted_manifest_sha256": manifest,
        "training_code_revision": training_revision,
        "training_asset_revision": asset_revision,
        "git_revision": evaluation_revision,
        "git_dirty": "False",
        "asset_git_revision": asset_revision,
        "asset_git_dirty": "False",
        "campaign_config_sha256": campaign_config,
        "treatment_config_sha256": treatment_config,
        "nail_asset_sha256": nail_asset,
        "seed": "2026072900",
        "num_envs": "256",
        "episodes_per_env_sampled": "2",
        "episode_len_s": "4.0",
        "nsteps": "400",
        "sampled_completion_rule": "first_two_completions_per_environment",
        "sampled_actions_stochastic": "True",
        "reset_position_noise_min_rad": "0.0",
        "reset_position_noise_max_rad": "0.0",
        "actor_observation_corruption": "True",
        "critic_observation_corruption": "False",
        "physics_dt_s": "0.002",
        "control_decimation": "10",
        "fixed_impedance_signature_sha256": (
            "a8252c853dd0059c89cff357e8e542fc6d097ecc9ef2652768ca0ef83d8aa269"
        ),
        "fixed_action_signature_sha256": (
            "56e59da46050a16c2005cb1872632ed81d41f82b48ae5c94beaca8fea44790ec"
        ),
        "reset_rng_seed": "2036072919",
        "observation_rng_seed": "2046072933",
        "action_rng_seed": "2056072941",
        "impact_weight": "8.0",
        "delivered_weight": "0.0",
        "r_waypoint_progress_weight": "8.0",
        "event_i_ref_n_s": "0.3088",
        "imp_max_p": "0.0",
        "n_episodes_sampled": "512",
    }
    payload = {
        "schema_version": 3,
        "selection": "first two completed episodes from each of 256 environments",
        "expected_episode_count": 512,
        "task": task,
        "treatment": "V+D0",
        "training_seed": 2,
        "event_i_ref_n_s": 0.3088,
        "imp_max_p": 0.0,
        "weights": {
            "impact_progress": 8.0,
            "delivered_impulse": 0.0,
            "r_waypoint_progress": 8.0,
        },
        "impulse_limits_n_m_s": caps,
        "rng_streams": {
            "reset": 2036072919,
            "observation": 2046072933,
            "action": 2056072941,
        },
        "evaluation_contract": {
            "base_rng_seed": 2026072900,
            "num_envs": 256,
            "episodes_per_env": 2,
            "episode_len_s": 4.0,
            "mean_nsteps": 400,
            "completion_rule": "first_two_completions_per_environment",
            "stochastic_actions": True,
            "reset_position_noise_rad": [0.0, 0.0],
            "actor_observation_corruption": True,
            "critic_observation_corruption": False,
            "physics_dt_s": 0.002,
            "control_decimation": 10,
            "fixed_impedance_signature_sha256": summary[
                "fixed_impedance_signature_sha256"
            ],
            "fixed_action_signature_sha256": summary[
                "fixed_action_signature_sha256"
            ],
            "strict_config_identities": {},
        },
        "provenance": {
            "checkpoint_sha256": checkpoint,
            "accepted_checkpoint_sha256": checkpoint,
            "accepted_manifest_sha256": manifest,
            "campaign_config_sha256": campaign_config,
            "treatment_config_sha256": treatment_config,
            "nail_asset_sha256": nail_asset,
            "training_code_revision": training_revision,
            "training_asset_revision": asset_revision,
            "code_git": {
                "revision": evaluation_revision,
                "dirty": False,
                "status": "",
            },
            "asset_git": {
                "revision": asset_revision,
                "dirty": False,
                "status": "",
            },
        },
        "episodes": [
            {
                "episode_id": f"V+D0-env{env_id}-episode{ordinal}",
                "env_id": env_id,
                "episode_ordinal": ordinal,
                "arm": "V+D0",
                "task": task,
                "impulse_limits_n_m_s": caps,
            }
            for env_id in range(256)
            for ordinal in (0, 1)
        ],
    }
    expected = {
        "manifest": manifest,
        "training_revision": training_revision,
        "evaluation_revision": evaluation_revision,
        "asset_revision": asset_revision,
        "manifest_row": {
            "array_index": "0",
            "arm": "V+D0",
            "name": summary["name"],
            "task": task,
            "training_seed": "2",
            "training_code_revision": training_revision,
            "training_asset_revision": asset_revision,
            "checkpoint_path": "/frozen/model_199.pt",
            "checkpoint_sha256": checkpoint,
        },
        "config_identity": {
            "campaign_config_sha256": campaign_config,
            "treatment_config_sha256": treatment_config,
            "nail_asset_sha256": nail_asset,
        },
    }
    return summary, payload, expected


def _raw_sampled_episode(
    *,
    task,
    reason,
    delivered,
    accumulated,
    qvel_peak,
    recontact,
    peak_lambda,
    next_gate,
):
    reason_code = {"success": 1, "window": 2}[reason]
    states = [[0.0] * 6 for _ in range(5)]
    states[1][0] = qvel_peak
    contact = [False, True, False, recontact]
    return {
        "episode_id": f"fixture-{reason}",
        "env_id": 0 if reason == "success" else 1,
        "episode_ordinal": 0,
        "arm": "V+D0",
        "task": task,
        "trace_digest": "8" * 64,
        "physics_dt_s": 0.002,
        "phase_contract": {
            "physical_channels": "pre_integration",
            "tracker_derived_channels": "pre_integration",
            "tracker_nail_depth": "post_integration",
            "tracker_depth_lead_substeps": 1,
        },
        "nail_geometry": NAIL_GEOMETRY,
        "physical": {
            "contact": contact,
            "head_position_m": [
                [0.0, 0.0, 0.10],
                [0.0, 0.0, 0.08],
                [0.0, 0.0, 0.07],
                [0.0, 0.0, 0.06],
            ],
            "clamped_depth_m": [0.0, 0.01, 0.01, 0.01],
            "net_axial_force_n": [0.0, 40.0, 0.0, 10.0 if recontact else 0.0],
            "joint_speed_rad_s": states[:-1],
            "post_step_joint_speed_rad_s": states[1:],
            "tracker_depth_post_integration_m": [0.0, 0.01, 0.01, 0.01],
        },
        "event_trace": {
            "tracker_started": [False, True, True, True],
            "tracker_finalized": [False, True, True, True],
            "tracker_productive": [False, True, True, True],
            "tracker_reason": [0, reason_code, reason_code, reason_code],
            "event_cumulative_impulse_n_s": [
                0.0,
                delivered,
                delivered,
                delivered,
            ],
        },
        "first_strike": {
            "started": True,
            "accepted_onset_index": 1,
            "finalized": True,
            "productive": True,
            "reason": reason,
            "v_precontact_m_s": 1.0,
            "delivered_n_s": delivered,
            "saturated": False,
        },
        "overall_success": True,
        "episode_peak_lambda": peak_lambda,
        "episode_delivered_accumulator_n_s": accumulated,
        "episode_depth_m": 0.032,
        "guideline": {"next_gate": next_gate},
        "reward": {
            "gamma": 0.99,
            "impact_payout": [0.0] * 4,
            "delivered_payout": [0.0] * 4,
        },
    }


@pytest.fixture
def result_inputs():
    checkpoint = "a" * 64
    training_revision = "b" * 40
    asset_revision = "c" * 40
    reset_digest = "d" * 64
    task = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered0"
    sampled_summary = {
        "name": "presentation3_pvd0_seed2",
        "task": task,
        "treatment": "V+D0",
        "training_seed": "2",
        "checkpoint_sha256": checkpoint,
        "accepted_checkpoint_sha256": checkpoint,
        "accepted_manifest_sha256": "e" * 64,
        "sampled_trace_digest": "f" * 64,
        "sampled_trace_artifact_sha256": "1" * 64,
        "training_code_revision": training_revision,
        "training_asset_revision": asset_revision,
        "git_revision": "2" * 40,
        "git_dirty": "False",
        "asset_git_dirty": "False",
        "reset_digest": reset_digest,
        "n_episodes_sampled": "2",
        "qvel_violation_rate_sampled": "0.5",
        "qvel_max_abs_rad_s_sampled": "3.20",
        "worst_ratio_max_sampled": "0.50",
        "recontact_rate_sampled": "0.50",
        "tail_fraction_mean_sampled": "0.10",
        "first_strike_v_precontact_mean_sampled": "1.0",
    }
    sampled_payload = {
        "task": task,
        "treatment": "V+D0",
        "training_seed": 2,
        "imp_max_p": 0.0,
        "weights": {
            "impact_progress": 8.0,
            "delivered_impulse": 0.0,
            "r_waypoint_progress": 8.0,
        },
        "impulse_limits_n_m_s": [1.64, 3.28, 1.64, 1.64, 1.64, 1.64],
        "nail_geometry": NAIL_GEOMETRY,
        "provenance": {
            "checkpoint_sha256": checkpoint,
            "training_code_revision": training_revision,
            "training_asset_revision": asset_revision,
        },
        "episodes": [
            _raw_sampled_episode(
                task=task,
                reason="success",
                delivered=0.30,
                accumulated=0.50,
                qvel_peak=3.20,
                recontact=True,
                peak_lambda=[0.82, 0.0, 0.0, 0.0, 0.0, 0.0],
                next_gate=[2, 6, 6, 6],
            ),
            _raw_sampled_episode(
                task=task,
                reason="window",
                delivered=0.10,
                accumulated=0.30,
                qvel_peak=3.1415,
                recontact=False,
                peak_lambda=[0.41, 0.0, 0.0, 0.0, 0.0, 0.0],
                next_gate=[1, 2, 6, 6],
            ),
        ],
    }
    video_metadata = {
        "campaign": "presentation3",
        "arm": "P+V+D0",
        "training_seed": 2,
        "task": task,
        "checkpoint_sha256": checkpoint,
        "training_revision": training_revision,
        "asset_revision": asset_revision,
        "analysis_revision": "3" * 40,
        "reset_state_digest": reset_digest,
    }
    video_trace = {
        "substep_head_position_m": np.array(
            [[0.0, 0.0, 0.10], [0.0, 0.0, 0.08], [0.01, 0.0, 0.06]]
        ),
        "substep_contact": np.array([False, True, True]),
        "substep_nail_depth_m": np.array([0.0, 0.01, 0.04]),
        "substep_arm_qvel_rad_s": np.array(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 3.1, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "substep_gate_index": np.array([2, 6, 6]),
        "substep_perpendicular_error_m": np.array([0.0, 0.001, 0.010]),
        "guideline_entry_m": np.array([0.0, 0.0, 0.10]),
        "guideline_nail_m": np.array([0.0, 0.0, 0.06]),
        "guideline_gate_centers_m": np.array(
            [[0.0, 0.0, z] for z in np.linspace(0.095, 0.065, 6)]
        ),
    }
    impulse_metadata = {
        "campaign": "presentation3",
        "arm": "P+V+D0",
        "training_seed": 2,
        "task": task,
        "checkpoint_sha256": checkpoint,
        "asset_revision": asset_revision,
        "code_revision": p3.EXPECTED_D0_RENDERER,
        "reset_state_digest": reset_digest,
        "j_limit_n_m_s": [1.64, 3.28, 1.64, 1.64, 1.64, 1.64],
    }
    impulse_trace = {
        "head_position_m": video_trace["substep_head_position_m"].astype(np.float32),
        "contact": video_trace["substep_contact"],
        # The diagnostic's pre-integration channel is intentionally one substep
        # behind; only its explicitly post-integration channel aligns to the renderer.
        "nail_depth_m": np.array([0.0, 0.0, 0.01], dtype=np.float32),
        "nail_depth_post_integration_m": np.minimum(
            video_trace["substep_nail_depth_m"], 0.032
        ).astype(np.float32),
        "joint_velocity_post_integration_rad_s": video_trace[
            "substep_arm_qvel_rad_s"
        ].astype(np.float32),
        "tracker_finalized": np.array([False, True, True]),
        "tracker_productive": np.array([False, True, True]),
        "tracker_delivered_n_s": np.array([0.0, 0.31, 0.31]),
        "tracker_v_precontact_m_s": np.array([0.0, 1.25, 1.25]),
        "delivered_impulse_n_s": np.array([0.0, 0.40, 0.45]),
        "lambda_windowed_constraint_read_n_m_s": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.82, 0.82, 0.0, 0.0, 0.0, 0.0],
                [0.82, 0.82, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    }

    return (
        sampled_summary,
        sampled_payload,
        video_metadata,
        video_trace,
        impulse_metadata,
        impulse_trace,
    )


def test_sampled_contract_accepts_only_the_frozen_complete_population():
    summary, payload, expected = _valid_sampled_contract_inputs()

    p3.validate_sampled_population_contract(summary, payload, **expected)


def test_frozen_manifest_loader_hash_binds_the_exact_policy_matrix(tmp_path):
    manifest = tmp_path / "accepted.tsv"
    manifest.write_text(
        "\t".join(
            (
                "array_index",
                "arm",
                "name",
                "task",
                "training_seed",
                "training_code_revision",
                "training_asset_revision",
                "checkpoint_path",
                "checkpoint_sha256",
            )
        )
        + "\n"
        + "\t".join(
            (
                "0",
                "V+D0",
                "presentation3_pvd0_seed2",
                "task",
                "2",
                "c" * 40,
                "e" * 40,
                "/frozen/model_199.pt",
                "a" * 64,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    rows = p3.load_frozen_training_manifest(
        manifest,
        expected_sha256=digest,
        expected_keys={("V+D0", 2)},
    )

    assert rows[("V+D0", 2)]["checkpoint_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        p3.load_frozen_training_manifest(
            manifest,
            expected_sha256="f" * 64,
            expected_keys={("V+D0", 2)},
        )


def test_sampled_input_inventory_binds_the_exact_24_file_pairs():
    records = []
    for arm_index, arm in enumerate(("M", "V+D0", "V", "V+M")):
        for seed in range(2, 8):
            index = arm_index * 6 + seed
            records.append(
                {
                    "arm": arm,
                    "training_seed": seed,
                    "summary_sha256": f"{index:064x}",
                    "artifact_sha256": f"{index + 100:064x}",
                }
            )
    canonical = "".join(
        f"{row['arm']}\t{row['training_seed']}\t{row['summary_sha256']}\t"
        f"{row['artifact_sha256']}\n"
        for row in records
    ).encode()
    expected = hashlib.sha256(canonical).hexdigest()

    assert p3.validate_sampled_input_inventory(
        records, expected_sha256=expected
    ) == expected

    records[0]["artifact_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="inventory SHA-256"):
        p3.validate_sampled_input_inventory(records, expected_sha256=expected)


def test_sampled_contract_rejects_jointly_mutated_config_and_asset_identity():
    summary, payload, expected = _valid_sampled_contract_inputs()
    for field, value in (
        ("campaign_config_sha256", "a" * 64),
        ("treatment_config_sha256", "b" * 64),
        ("nail_asset_sha256", "c" * 64),
    ):
        summary[field] = value
        payload["provenance"][field] = value

    with pytest.raises(ValueError, match="config identity"):
        p3.validate_sampled_population_contract(summary, payload, **expected)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda payload: payload["provenance"].__setitem__(
                "accepted_manifest_sha256", "f" * 64
            ),
            "accepted_manifest_sha256",
        ),
        (
            lambda payload: payload["provenance"].__setitem__(
                "training_code_revision", "f" * 40
            ),
            "training_code_revision",
        ),
        (
            lambda payload: payload["rng_streams"].__setitem__("action", 7),
            "RNG",
        ),
        (
            lambda payload: payload["evaluation_contract"].__setitem__(
                "episodes_per_env", 1
            ),
            "evaluation contract",
        ),
        (
            lambda payload: payload.__setitem__("selection", "best 512 episodes"),
            "selection",
        ),
        (
            lambda payload: payload.__setitem__(
                "impulse_limits_n_m_s", [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
            ),
            "impulse caps",
        ),
    ),
)
def test_sampled_contract_rejects_stale_or_mutated_bindings(mutation, match):
    summary, payload, expected = _valid_sampled_contract_inputs()
    mutation(payload)

    with pytest.raises(ValueError, match=match):
        p3.validate_sampled_population_contract(summary, payload, **expected)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda payload: payload.__setitem__("schema_version", 3.0), "schema_version"),
        (lambda payload: payload.__setitem__("imp_max_p", False), "imp_max_p"),
        (
            lambda payload: payload["evaluation_contract"].__setitem__(
                "stochastic_actions", 1
            ),
            "evaluation contract",
        ),
        (
            lambda payload: payload["episodes"][34].__setitem__("env_id", 17.9),
            "environment/ordinal",
        ),
        (
            lambda payload: payload.__setitem__(
                "impulse_limits_n_m_s", {"not": "a vector"}
            ),
            "impulse caps",
        ),
    ),
)
def test_sampled_contract_rejects_permissive_json_types(mutation, match):
    summary, payload, expected = _valid_sampled_contract_inputs()
    mutation(payload)

    with pytest.raises(ValueError, match=match):
        p3.validate_sampled_population_contract(summary, payload, **expected)


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("action_rng_seed", "7"),
        ("episodes_per_env_sampled", "1"),
        ("sampled_completion_rule", "best_two"),
        ("delivered_weight", "9.0"),
        ("imp_max_p", "0.5"),
    ),
)
def test_sampled_contract_rejects_summary_contract_drift(field, bad_value):
    summary, payload, expected = _valid_sampled_contract_inputs()
    summary[field] = bad_value

    with pytest.raises(ValueError, match=field):
        p3.validate_sampled_population_contract(summary, payload, **expected)


def test_sampled_contract_rejects_a_duplicated_stream_coordinate():
    summary, payload, expected = _valid_sampled_contract_inputs()
    payload["episodes"][-1] = copy.deepcopy(payload["episodes"][0])

    with pytest.raises(ValueError, match="environment/ordinal"):
        p3.validate_sampled_population_contract(summary, payload, **expected)


def test_sampled_contract_rejects_episode_task_arm_and_cap_drift():
    summary, payload, expected = _valid_sampled_contract_inputs()
    payload["episodes"][10]["task"] = "wrong-task"

    with pytest.raises(ValueError, match="episode identity"):
        p3.validate_sampled_population_contract(summary, payload, **expected)

    summary, payload, expected = _valid_sampled_contract_inputs()
    payload["episodes"][10]["impulse_limits_n_m_s"] = [1.0] * 6

    with pytest.raises(ValueError, match="episode impulse caps"):
        p3.validate_sampled_population_contract(summary, payload, **expected)


@pytest.mark.parametrize(
    "drift,match",
    (
        ("duplicate_coordinate", "environment/ordinal"),
        ("joint_config_identity", "config identity"),
    ),
)
def test_main_applies_population_contract_before_reducing_raw_episodes(
    tmp_path, monkeypatch, drift, match
):
    summary, payload, _ = _valid_sampled_contract_inputs()
    manifest_fields = list(p3.FROZEN_MANIFEST_FIELDS)

    def write_manifest(path, arms, training_revision, sampled_row=None):
        rows = []
        for arm in arms:
            for seed in range(2, 8):
                row = {
                    "array_index": str(len(rows)),
                    "arm": arm,
                    "name": f"unused_{arm}_{seed}",
                    "task": "unused-task",
                    "training_seed": str(seed),
                    "training_code_revision": training_revision,
                    "training_asset_revision": p3.EXPECTED_ASSET_REVISION,
                    "checkpoint_path": f"/frozen/{arm}_{seed}/model_199.pt",
                    "checkpoint_sha256": f"{len(rows) + 1:064x}",
                }
                if sampled_row is not None and (arm, seed) == (
                    sampled_row["treatment"],
                    int(sampled_row["training_seed"]),
                ):
                    row.update(
                        {
                            "name": sampled_row["name"],
                            "task": sampled_row["task"],
                            "checkpoint_sha256": sampled_row["checkpoint_sha256"],
                        }
                    )
                rows.append(row)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=manifest_fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    existing_manifest_path = tmp_path / "existing.tsv"
    d0_manifest_path = tmp_path / "d0.tsv"
    existing_manifest_sha = write_manifest(
        existing_manifest_path,
        ("M", "V", "V+M"),
        p3.EXPECTED_EXISTING_TRAINING,
    )
    d0_manifest_sha = write_manifest(
        d0_manifest_path,
        ("V+D0",),
        p3.EXPECTED_D0_TRAINING,
        sampled_row=summary,
    )
    monkeypatch.setattr(p3, "EXPECTED_EXISTING_MANIFEST", existing_manifest_sha)
    monkeypatch.setattr(p3, "EXPECTED_D0_MANIFEST", d0_manifest_sha)
    config_identity = p3.EXPECTED_CONFIG_IDENTITY_BY_TREATMENT["V+D0"]
    summary.update(config_identity)
    payload["provenance"].update(config_identity)
    summary.update(
        {
            "accepted_manifest_sha256": d0_manifest_sha,
            "training_code_revision": p3.EXPECTED_D0_TRAINING,
            "training_asset_revision": p3.EXPECTED_ASSET_REVISION,
            "git_revision": p3.EXPECTED_D0_EVALUATION,
            "asset_git_revision": p3.EXPECTED_ASSET_REVISION,
            "reset_digest": p3.EXPECTED_RESET_DIGEST,
        }
    )
    payload["provenance"].update(
        {
            "accepted_manifest_sha256": d0_manifest_sha,
            "training_code_revision": p3.EXPECTED_D0_TRAINING,
            "training_asset_revision": p3.EXPECTED_ASSET_REVISION,
        }
    )
    payload["provenance"]["code_git"]["revision"] = p3.EXPECTED_D0_EVALUATION
    payload["provenance"]["asset_git"]["revision"] = p3.EXPECTED_ASSET_REVISION
    if drift == "duplicate_coordinate":
        payload["episodes"][-1] = copy.deepcopy(payload["episodes"][0])
    else:
        for field, value in (
            ("campaign_config_sha256", "a" * 64),
            ("treatment_config_sha256", "b" * 64),
            ("nail_asset_sha256", "c" * 64),
        ):
            summary[field] = value
            payload["provenance"][field] = value
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    payload["payload_digest"] = digest

    existing_root = tmp_path / "existing"
    d0_root = tmp_path / "d0"
    row_root = d0_root / summary["name"]
    existing_root.mkdir()
    row_root.mkdir(parents=True)
    trace_path = row_root / "fixture_sampled_traces.npz"
    encoded = np.frombuffer(json.dumps(payload).encode(), dtype=np.uint8)
    np.savez_compressed(trace_path, payload_json=encoded)
    summary["sampled_trace_digest"] = digest
    summary["sampled_trace_artifact_sha256"] = hashlib.sha256(
        trace_path.read_bytes()
    ).hexdigest()
    with (row_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "presentation3_results.py",
            "--existing-fixed-root",
            str(tmp_path / "unused-existing-fixed"),
            "--d0-fixed-root",
            str(tmp_path / "unused-d0-fixed"),
            "--existing-sampled-root",
            str(existing_root),
            "--d0-sampled-root",
            str(d0_root),
            "--existing-manifest",
            str(existing_manifest_path),
            "--d0-manifest",
            str(d0_manifest_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    with pytest.raises(ValueError, match=match):
        p3.main()


def test_main_requires_the_exact_sampled_input_inventory_before_writing(
    tmp_path, monkeypatch
):
    existing_root = tmp_path / "existing"
    d0_root = tmp_path / "d0"
    existing_root.mkdir()
    d0_root.mkdir()
    monkeypatch.setattr(p3, "load_frozen_training_manifest", lambda *args, **kwargs: {})

    def reject_unpinned_inventory(records, *, expected_sha256):
        assert records == []
        assert expected_sha256 == p3.EXPECTED_SAMPLED_INPUT_INVENTORY_SHA256
        raise ValueError("sampled inventory sentinel")

    monkeypatch.setattr(
        p3, "validate_sampled_input_inventory", reject_unpinned_inventory
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "presentation3_results.py",
            "--existing-fixed-root",
            str(tmp_path / "unused-existing-fixed"),
            "--d0-fixed-root",
            str(tmp_path / "unused-d0-fixed"),
            "--existing-sampled-root",
            str(existing_root),
            "--d0-sampled-root",
            str(d0_root),
            "--existing-manifest",
            str(tmp_path / "unused-existing.tsv"),
            "--d0-manifest",
            str(tmp_path / "unused-d0.tsv"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    with pytest.raises(ValueError, match="sampled inventory sentinel"):
        p3.main()


def test_assemble_result_row_keeps_sampled_and_fixed_protocols_separate(result_inputs):
    row, trajectory = assemble_result_row(*result_inputs)

    assert row["sampled_first_event_impulse_mean_n_s"] == pytest.approx(0.20)
    assert row["sampled_success_finalized_n"] == 1
    assert row["sampled_window_finalized_n"] == 1
    assert row["sampled_all_six_by_contact_n"] == 1
    assert row["sampled_qvel_legal_n"] == 1
    assert row["sampled_qvel_max_rad_s"] == pytest.approx(3.20)
    assert row["sampled_max_lambda_cap_ratio"] == pytest.approx(0.50)
    assert row["sampled_post_finalization_contact_rate"] == pytest.approx(0.50)
    assert "sampled_recontact_rate" not in row
    assert row["sampled_tail_fraction_mean"] == pytest.approx(0.10)
    assert row["fixed_first_event_impulse_n_s"] == pytest.approx(0.31)
    assert row["fixed_cumulative_impulse_n_s"] == pytest.approx(0.45)
    assert row["fixed_qvel_legal"] is True
    assert row["fixed_all_six_by_contact"] is True
    assert row["fixed_precontact_perpendicular_max_m"] == pytest.approx(0.001)
    assert trajectory["waypoints_m"].shape == (6, 3)


def test_assemble_result_row_emits_sampled_and_fixed_precontact_speeds(result_inputs):
    row, _ = assemble_result_row(*result_inputs)

    assert row["sampled_precontact_axial_speed_mean_m_s"] == pytest.approx(1.0)
    assert row["fixed_precontact_axial_speed_m_s"] == pytest.approx(1.25)


def test_assemble_result_row_rejects_sampled_precontact_speed_summary_mismatch(
    result_inputs,
):
    mutated = copy.deepcopy(result_inputs)
    mutated[0]["first_strike_v_precontact_mean_sampled"] = "0.99"

    with pytest.raises(
        ValueError, match="first_strike_v_precontact_mean_sampled"
    ):
        assemble_result_row(*mutated)


@pytest.mark.parametrize(
    "malformed_speed",
    (
        np.array([[0.0, 1.25, 1.25]]),
        np.array([0.0, np.nan, 1.25]),
    ),
)
def test_assemble_result_row_rejects_malformed_fixed_precontact_speed(
    result_inputs, malformed_speed
):
    mutated = copy.deepcopy(result_inputs)
    mutated[5]["tracker_v_precontact_m_s"] = malformed_speed

    with pytest.raises(ValueError, match="fixed-reset tracker speed"):
        assemble_result_row(*mutated)


def test_assemble_result_row_rejects_fixed_cap_drift(result_inputs):
    mutated = copy.deepcopy(result_inputs)
    mutated[4]["j_limit_n_m_s"][0] = 9.0

    with pytest.raises(ValueError, match="fixed-reset impulse caps"):
        assemble_result_row(*mutated)


def test_assemble_result_row_rejects_unapproved_impulse_analysis_revision(
    result_inputs,
):
    mutated = copy.deepcopy(result_inputs)
    mutated[4]["code_revision"] = "4" * 40

    with pytest.raises(ValueError, match="impulse analysis revision"):
        assemble_result_row(*mutated)


@pytest.mark.parametrize(
    "field,mutated_value",
    (
        ("qvel_violation_rate_sampled", "0.0"),
        ("qvel_max_abs_rad_s_sampled", "3.19"),
        ("worst_ratio_max_sampled", "0.40"),
        ("recontact_rate_sampled", "0.0"),
        ("tail_fraction_mean_sampled", "0.0"),
    ),
)
def test_assemble_result_row_rejects_raw_inconsistent_sampled_summary(
    result_inputs, field, mutated_value
):
    mutated = copy.deepcopy(result_inputs)
    mutated[0][field] = mutated_value

    with pytest.raises(ValueError, match=field):
        assemble_result_row(*mutated)


def test_fixed_impulse_keeps_productive_first_event_separate_from_larger_cumulative():
    diagnostic = {
        "tracker_finalized": np.array([False, True]),
        "tracker_productive": np.array([False, True]),
        "tracker_delivered_n_s": np.array([0.0, 0.30]),
        "tracker_v_precontact_m_s": np.array([0.0, 1.10]),
        "delivered_impulse_n_s": np.array([0.0, 0.52]),
        "lambda_windowed_constraint_read_n_m_s": np.array(
            [[0.0, 0.0], [0.82, 0.82]]
        ),
    }

    actual = fixed_impulse_metrics(diagnostic, caps=np.array([1.64, 3.28]))

    assert actual["first_event_impulse_n_s"] == pytest.approx(0.30)
    assert actual["cumulative_impulse_n_s"] == pytest.approx(0.52)
    assert actual["post_event_tail_n_s"] == pytest.approx(0.22)
    assert actual["max_lambda_cap_ratio"] == pytest.approx(0.5)


def test_fixed_impulse_rejects_a_first_event_value_that_drifted_after_finalization():
    diagnostic = {
        "tracker_finalized": np.array([False, True, True]),
        "tracker_productive": np.array([False, True, True]),
        "tracker_delivered_n_s": np.array([0.0, 0.30, 0.45]),
        "tracker_v_precontact_m_s": np.array([0.0, 1.10, 1.10]),
        "delivered_impulse_n_s": np.array([0.0, 0.30, 0.52]),
        "lambda_windowed_constraint_read_n_m_s": np.array(
            [[0.0, 0.0], [0.82, 0.82], [0.82, 0.82]]
        ),
    }

    with pytest.raises(ValueError, match="first-event latch"):
        fixed_impulse_metrics(diagnostic, caps=np.array([1.64, 3.28]))


def test_waypoint_count_stops_at_contact_onset_instead_of_using_follow_through():
    next_waypoint = np.array([0, 1, 2, 6])
    contact = np.array([False, False, True, True])

    assert waypoints_reached_by_contact(next_waypoint, contact) == 2


def test_waypoints_by_contact_is_zero_when_contact_never_occurs():
    next_waypoint = np.array([0, 1, 2, 6])
    contact = np.array([False, False, False, False])

    assert waypoints_reached_by_contact(next_waypoint, contact) == 0


def test_precontact_geometry_uses_recorded_finite_segment_error_and_contact_window():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.04],
            [0.01, 0.0, -0.08],
            [0.20, 0.0, -0.20],
        ]
    )
    recorded_segment_error = np.array([0.0, 0.001, 0.010, 0.150])
    contact = np.array([False, False, True, True])

    actual = precontact_geometry_metrics(
        positions,
        recorded_segment_error,
        contact,
        entry=np.array([0.0, 0.0, 0.0]),
    )

    assert actual["window_substeps"] == 3
    assert actual["perpendicular_max_m"] == pytest.approx(0.010)
    assert actual["perpendicular_rms_m"] == pytest.approx(
        np.sqrt((0.0**2 + 0.001**2 + 0.010**2) / 3.0)
    )
    assert actual["path_ratio"] == pytest.approx(
        (0.04 + np.sqrt(0.01**2 + 0.04**2)) / np.sqrt(0.01**2 + 0.08**2)
    )


def test_qvel_legality_uses_500hz_post_integration_samples():
    post_integration = np.array([[0.0, 2.0], [3.20, 1.0]])

    peak, legal = peak_post_integration_qvel(post_integration, limit=3.1415)

    assert peak == pytest.approx(3.20)
    assert legal is False


def test_qvel_legality_is_inclusive_at_the_limit_and_catches_the_last_sample():
    limit = 3.1415
    at_limit = np.array([[0.0, limit]])
    last_sample_over = np.array([[0.0, 1.0], [0.0, np.nextafter(limit, np.inf)]])

    assert peak_post_integration_qvel(at_limit, limit=limit)[1] is True
    assert peak_post_integration_qvel(last_sample_over, limit=limit)[1] is False


def test_sampled_first_event_mean_includes_a_productive_window_finalization():
    payload = {
        "episodes": [
            {
                "overall_success": True,
                "episode_delivered_accumulator_n_s": 0.50,
                "first_strike": {
                    "finalized": True,
                    "productive": True,
                    "reason": "success",
                    "delivered_n_s": 0.30,
                },
            },
            {
                "overall_success": True,
                "episode_delivered_accumulator_n_s": 0.30,
                "first_strike": {
                    "finalized": True,
                    "productive": True,
                    "reason": "window",
                    "delivered_n_s": 0.10,
                },
            },
        ]
    }

    actual = sampled_first_event_metrics(payload)

    assert actual == {
        "episode_n": 2,
        "overall_success_n": 2,
        "finalized_n": 2,
        "productive_n": 2,
        "success_finalized_n": 1,
        "window_finalized_n": 1,
        "first_event_impulse_mean_n_s": pytest.approx(0.20),
        "cumulative_impulse_mean_n_s": pytest.approx(0.40),
        "post_event_tail_mean_n_s": pytest.approx(0.20),
    }


def test_sampled_waypoint_rate_stops_at_each_episode_contact_onset():
    payload = {
        "episodes": [
            {
                "first_strike": {"accepted_onset_index": 2},
                "guideline": {"next_gate": [0, 1, 2, 6]},
            },
            {
                "first_strike": {"accepted_onset_index": 3},
                "guideline": {"next_gate": [0, 2, 5, 6]},
            },
        ]
    }

    assert sampled_waypoints_by_contact(payload) == {
        "all_six_by_contact_n": 1,
        "all_six_by_contact_rate": pytest.approx(0.5),
    }


def test_policy_matrix_requires_all_four_arms_and_six_seeds_without_selection():
    rows = [
        {
            "arm": arm,
            "training_seed": seed,
            "checkpoint_sha256": f"{index:064x}",
        }
        for index, (arm, seed) in enumerate(
            (
                (arm, seed)
                for arm in ("M", "V+D0", "V", "V+M")
                for seed in range(2, 8)
            ),
            start=1,
        )
    ]
    scrambled = list(reversed(rows))

    ordered = canonical_policy_rows(scrambled)

    assert [(row["arm"], row["training_seed"]) for row in ordered] == [
        (arm, seed)
        for arm in ("M", "V+D0", "V", "V+M")
        for seed in range(2, 8)
    ]

    with pytest.raises(ValueError, match="exactly 24"):
        canonical_policy_rows(rows[:-1])

    duplicate_checkpoint = [dict(row) for row in rows]
    duplicate_checkpoint[-1]["checkpoint_sha256"] = rows[0]["checkpoint_sha256"]
    with pytest.raises(ValueError, match="checkpoint"):
        canonical_policy_rows(duplicate_checkpoint)


def test_presentation_figures_use_first_event_qvel_cap_and_canonical_grid_fields(tmp_path):
    rows = []
    traces = {}
    for arm_index, arm in enumerate(("M", "V+D0", "V", "V+M")):
        for seed in range(2, 8):
            dose = {"M": 4, "V+D0": 0, "V": 2, "V+M": 4}[arm]
            rows.append(
                {
                    "arm": arm,
                    "training_seed": seed,
                    "checkpoint_sha256": f"{arm_index * 6 + seed:064x}",
                    "delivered_reward_dose": dose,
                    "velocity_cat_enabled": arm in {"V+D0", "V", "V+M"},
                    "sampled_first_event_impulse_mean_n_s": 0.10 * dose + seed / 1000,
                    "sampled_cumulative_impulse_mean_n_s": 1.0 + 0.10 * dose + seed / 1000,
                    "fixed_first_event_impulse_n_s": 0.05 * dose + seed / 1000,
                    "fixed_cumulative_impulse_n_s": 0.5 + 0.05 * dose + seed / 1000,
                    "sampled_qvel_max_rad_s": 5.0 if arm == "M" else 3.0,
                    "sampled_max_lambda_cap_ratio": 0.1 * (arm_index + 1),
                }
            )
            traces[(arm, seed)] = {
                "head_position_m": np.array(
                    [[0.0, 0.0, 0.10], [0.001 * seed, 0.0, 0.05]]
                ),
                "entry_m": np.array([0.0, 0.0, 0.10]),
                "nail_m": np.array([0.0, 0.0, 0.05]),
                "waypoints_m": np.array(
                    [[0.0, 0.0, z] for z in np.linspace(0.095, 0.055, 6)]
                ),
            }

    dose = dose_response_figure(rows)
    sampled_first = next(
        line for line in dose.axes[0].lines if line.get_label() == "median first event"
    )
    sampled_cumulative = next(
        line for line in dose.axes[0].lines if line.get_label() == "median cumulative"
    )
    assert sampled_first.get_ydata() == pytest.approx([0.0045, 0.2045, 0.4045])
    assert sampled_cumulative.get_ydata() == pytest.approx([1.0045, 1.2045, 1.4045])

    velocity = velocity_cat_figure(rows)
    assert any(
        np.allclose(line.get_ydata(), [3.1415, 3.1415])
        for line in velocity.axes[0].lines
    )

    cap = cap_utilization_figure(rows)
    assert any(
        len(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), [1.0, 1.0])
        for line in cap.axes[0].lines
    )

    grid = trajectory_grid_figure(rows, traces)
    assert len(grid.axes) == 24
    assert [axis.get_title() for axis in grid.axes] == [
        f"{description} | seed {seed}"
        for description in (
            "D4, no velocity CaT",
            "D0 + 500 Hz velocity CaT",
            "D2 + 500 Hz velocity CaT",
            "D4 + 500 Hz velocity CaT",
        )
        for seed in range(2, 8)
    ]
    assert not any(
        isinstance(patch, Circle) for axis in grid.axes for patch in axis.patches
    )

    outputs = write_result_package(rows, traces, tmp_path)
    assert set(outputs) == {
        "table",
        "dose_response",
        "velocity_cat",
        "cap_utilization",
        "trajectory_grid",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    with outputs["table"].open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 24
    assert [(row["arm"], int(row["training_seed"])) for row in written] == [
        (arm, seed)
        for arm in ("M", "V+D0", "V", "V+M")
        for seed in range(2, 8)
    ]


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("checkpoint_sha256", "b" * 64),
        ("reset_state_digest", "c" * 64),
        ("training_revision", "d" * 40),
        ("asset_revision", "e" * 40),
    ),
)
def test_artifact_join_rejects_mismatched_identity(field, bad_value):
    sampled = {
        "checkpoint_sha256": "a" * 64,
        "reset_state_digest": "1" * 64,
        "training_revision": "2" * 40,
        "asset_revision": "3" * 40,
        "training_seed": 4,
    }
    fixed = dict(sampled)
    fixed[field] = bad_value

    with pytest.raises(ValueError, match=field):
        validate_artifact_join(sampled, fixed)
