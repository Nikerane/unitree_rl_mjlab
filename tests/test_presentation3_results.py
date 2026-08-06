import copy
import csv

import numpy as np
import pytest
from matplotlib.patches import Circle

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
        "substep_arm_qvel_rad_s": np.array([[1.0, 2.0], [2.0, 3.0], [2.0, 3.1]]),
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
        "code_revision": "4" * 40,
        "reset_state_digest": reset_digest,
        "j_limit_n_m_s": [1.64, 3.28],
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
        "delivered_impulse_n_s": np.array([0.0, 0.40, 0.45]),
        "lambda_windowed_constraint_read_n_m_s": np.array(
            [[0.0, 0.0], [0.82, 0.82], [0.82, 0.82]]
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


def test_assemble_result_row_keeps_sampled_and_fixed_protocols_separate(result_inputs):
    row, trajectory = assemble_result_row(*result_inputs)

    assert row["sampled_first_event_impulse_mean_n_s"] == pytest.approx(0.20)
    assert row["sampled_success_finalized_n"] == 1
    assert row["sampled_window_finalized_n"] == 1
    assert row["sampled_all_six_by_contact_n"] == 1
    assert row["sampled_qvel_legal_n"] == 1
    assert row["sampled_qvel_max_rad_s"] == pytest.approx(3.20)
    assert row["sampled_max_lambda_cap_ratio"] == pytest.approx(0.50)
    assert row["sampled_recontact_rate"] == pytest.approx(0.50)
    assert row["sampled_tail_fraction_mean"] == pytest.approx(0.10)
    assert row["fixed_first_event_impulse_n_s"] == pytest.approx(0.31)
    assert row["fixed_cumulative_impulse_n_s"] == pytest.approx(0.45)
    assert row["fixed_qvel_legal"] is True
    assert row["fixed_all_six_by_contact"] is True
    assert row["fixed_precontact_perpendicular_max_m"] == pytest.approx(0.001)
    assert trajectory["waypoints_m"].shape == (6, 3)


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


def test_waypoint_count_stops_at_contact_onset_instead_of_using_follow_through():
    next_waypoint = np.array([0, 1, 2, 6])
    contact = np.array([False, False, True, True])

    assert waypoints_reached_by_contact(next_waypoint, contact) == 2


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
