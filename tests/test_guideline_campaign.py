from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from evaluation.analysis import guideline_campaign
from evaluation.analysis.guideline_campaign import (
    aggregate_guideline_seed,
    summarize_guideline_episode,
    validate_guideline_pilot_rows,
)


def _trace(
    *,
    next_gate: list[int] | None = None,
    contact_index: int | None = 4,
    errors: list[float] | None = None,
) -> dict:
    next_gate = [0, 1, 2, 4, 6] if next_gate is None else next_gate
    count = len(next_gate)
    errors = [0.001, 0.002, 0.006, 0.060, 0.004] if errors is None else errors
    assert len(errors) == count
    contact = [False] * count
    if contact_index is not None:
        contact[contact_index] = True
    return {
        "physical": {
            "contact": contact,
            "head_position_m": [[0.0, 0.0, float(index)] for index in range(count)],
            "joint_speed_rad_s": [[0.1] * 6 for _ in range(count)],
            "post_step_joint_speed_rad_s": [[0.2] * 6 for _ in range(count)],
        },
        "event_trace": {"tracker_started": contact},
        "guideline": {
            "entry_m": [0.0, 0.0, 0.0],
            "nail_m": [0.0, 0.0, 1.0],
            "next_gate": next_gate,
            "perpendicular_error_m": errors,
            "disarmed": [False] * count,
            "gate_reward_present": True,
            "gate_payout": [0.08, 0.08],
        },
        "first_strike": {
            "accepted_onset_index": contact_index,
            "v_precontact_m_s": 1.5,
        },
        "overall_success": True,
        "episode_peak_lambda": [0.82, 1.64, 0.82, 0.82, 0.82, 0.82],
        "episode_depth_m": 0.03,
        "impulse_limits_n_m_s": [1.64, 3.28, 1.64, 1.64, 1.64, 1.64],
    }


def test_episode_without_gate_one_gets_fixed_failure_value() -> None:
    summary = summarize_guideline_episode(_trace(next_gate=[0, 0, 0, 0, 0]))

    assert summary["q90_terminal_descent_perpendicular_error_m"] == pytest.approx(0.050)


def test_contact_before_gate_one_gets_fixed_failure_value() -> None:
    summary = summarize_guideline_episode(
        _trace(next_gate=[0, 0, 1, 2, 6], contact_index=1)
    )

    assert summary["q90_terminal_descent_perpendicular_error_m"] == pytest.approx(0.050)


def test_episode_endpoint_is_q90_of_capped_window_errors() -> None:
    trace = _trace()

    summary = summarize_guideline_episode(trace)

    expected = np.quantile(np.minimum(trace["guideline"]["perpendicular_error_m"][1:5], 0.050), 0.90)
    assert summary["q90_terminal_descent_perpendicular_error_m"] == pytest.approx(expected)
    assert summary["all_six_gates"] is True
    assert summary["corridor_occupancy"] == pytest.approx(0.5)
    assert summary["actual_gate_return"] == pytest.approx(0.16)
    assert summary["success"] is True
    assert summary["useful_precontact_speed_m_s"] == pytest.approx(1.5)
    assert summary["lambda_cap_ratios"] == pytest.approx([0.5] * 6)
    assert summary["qvel_finite"] is True


def test_no_contact_scores_through_episode_end() -> None:
    trace = _trace(contact_index=None)
    trace["guideline"]["perpendicular_error_m"][-1] = 0.050

    summary = summarize_guideline_episode(trace)

    expected = np.quantile(np.minimum(trace["guideline"]["perpendicular_error_m"][1:], 0.050), 0.90)
    assert summary["q90_terminal_descent_perpendicular_error_m"] == pytest.approx(expected)


def test_episode_retains_a_nonfinite_qvel_failure_for_pilot_validation() -> None:
    trace = _trace()
    trace["physical"]["joint_speed_rad_s"][0][0] = math.nan

    summary = summarize_guideline_episode(trace)

    assert summary["qvel_finite"] is False


def test_episode_rejects_any_drift_from_frozen_manufacturer_impulse_caps() -> None:
    trace = _trace()
    trace["impulse_limits_n_m_s"][3] = 1.63

    with pytest.raises(ValueError, match="frozen manufacturer impulse caps"):
        summarize_guideline_episode(trace)


@pytest.mark.parametrize("accepted_onset", (1.9, "1", math.nan))
def test_episode_rejects_nonintegral_accepted_contact_indices(accepted_onset: object) -> None:
    trace = _trace()
    trace["first_strike"]["accepted_onset_index"] = accepted_onset

    with pytest.raises(ValueError, match="literal integral index"):
        summarize_guideline_episode(trace)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda trace: trace["guideline"].__setitem__("entry_m", [0.0, math.nan, 0.0]),
        lambda trace: trace["guideline"].__setitem__("nail_m", [0.0, 0.0, 0.0]),
        lambda trace: trace["guideline"].__setitem__("perpendicular_error_m", [0.0]),
        lambda trace: trace["physical"].__setitem__("joint_speed_rad_s", [[0.1] * 6]),
    ),
    ids=("nonfinite-geometry", "degenerate-geometry", "misaligned-error", "misaligned-qvel"),
)
def test_episode_rejects_malformed_or_misaligned_stored_trace(mutate) -> None:
    trace = _trace()
    mutate(trace)

    with pytest.raises(ValueError):
        summarize_guideline_episode(trace)


def test_seed_aggregates_episode_values_not_pooled_substeps() -> None:
    short = _trace(errors=[0.001, 0.001, 0.001, 0.001, 0.001])
    long = _trace(
        next_gate=[0, 1] + [6] * 98,
        contact_index=99,
        errors=[0.049] * 100,
    )

    summary = aggregate_guideline_seed([short, long], expected_count=2)

    assert summary["q90_terminal_descent_perpendicular_error_m_sampled"] == pytest.approx(
        np.quantile([0.001, 0.049], 0.90)
    )


def test_seed_requires_its_exact_production_episode_count() -> None:
    with pytest.raises(ValueError, match="exactly 512"):
        aggregate_guideline_seed([_trace()])


def _pilot_rows() -> list[dict]:
    common = {
        "checkpoint_filename": "model_499.pt",
        "num_envs": 256,
        "episode_len_s": 4.0,
        "n_episodes_sampled": 512,
        "episodes_per_env_sampled": 2,
        "reset_digest": "a" * 64,
        "guideline_geometry_digest": "b" * 64,
        "reset_position_range_rad": (0.0, 0.0),
        "windup_enabled": False,
        "impedance_mode": "fixed",
        "imp_max_p": 0.0,
        "treatment_base_identity": "c" * 64,
        "reset_rng_seed": 2036072919,
        "observation_rng_seed": 2046072933,
        "action_rng_seed": 2056072941,
        "git_dirty": False,
        "asset_git_dirty": False,
        "git_revision": "1" * 40,
        "asset_git_revision": "2" * 40,
        "accepted_manifest_sha256": "d" * 64,
        "training_code_revision": "3" * 40,
        "training_asset_revision": "2" * 40,
        "campaign_config_sha256": "e" * 64,
        "impossible_success_n": 0,
        "lambda_dead_n": 0,
        "qvel_nonfinite_rate_sampled": 0.0,
        "q90_terminal_descent_perpendicular_error_m_sampled": 0.004,
        "all_six_gates_rate_sampled": 0.75,
        "corridor_occupancy_mean_sampled": 0.80,
        "backward_progress_count_mean_sampled": 0.25,
    }
    return [
        dict(
            common,
            task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
            treatment="C0",
            training_seed=0,
            checkpoint_sha256="4" * 64,
            accepted_checkpoint_sha256="4" * 64,
            treatment_config_sha256="f" * 64,
            sampled_trace_digest="5" * 64,
            sampled_trace_artifact_sha256="6" * 64,
            r_gate_present=False,
            r_gate_weight=None,
            gate_reward_present=False,
            actual_gate_return_total_sampled=0.0,
            success_rate_sampled=0.30,
            first_strike_useful_speed_mean_sampled=0.41,
            worst_ratio_max_sampled=0.61,
            qvel_finite_exceedance_rate_sampled=0.01,
            qvel_max_abs_rad_s_sampled=2.1,
        ),
        dict(
            common,
            task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
            treatment="C0",
            training_seed=1,
            checkpoint_sha256="7" * 64,
            accepted_checkpoint_sha256="7" * 64,
            treatment_config_sha256="f" * 64,
            sampled_trace_digest="8" * 64,
            sampled_trace_artifact_sha256="9" * 64,
            r_gate_present=False,
            r_gate_weight=None,
            gate_reward_present=False,
            actual_gate_return_total_sampled=0.0,
            success_rate_sampled=0.10,
            first_strike_useful_speed_mean_sampled=0.42,
            worst_ratio_max_sampled=0.62,
            qvel_finite_exceedance_rate_sampled=0.02,
            qvel_max_abs_rad_s_sampled=2.2,
        ),
        dict(
            common,
            task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
            treatment="C-Gate",
            training_seed=0,
            checkpoint_sha256="a" * 64,
            accepted_checkpoint_sha256="a" * 64,
            treatment_config_sha256="0" * 64,
            sampled_trace_digest="b" * 64,
            sampled_trace_artifact_sha256="c" * 64,
            r_gate_present=True,
            r_gate_weight=8.0,
            gate_reward_present=True,
            actual_gate_return_total_sampled=1.0,
            success_rate_sampled=0.25,
            first_strike_useful_speed_mean_sampled=0.43,
            worst_ratio_max_sampled=0.63,
            qvel_finite_exceedance_rate_sampled=0.03,
            qvel_max_abs_rad_s_sampled=2.3,
        ),
        dict(
            common,
            task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
            treatment="C-Gate",
            training_seed=1,
            checkpoint_sha256="d" * 64,
            accepted_checkpoint_sha256="d" * 64,
            treatment_config_sha256="0" * 64,
            sampled_trace_digest="e" * 64,
            sampled_trace_artifact_sha256="f" * 64,
            r_gate_present=True,
            r_gate_weight=8.0,
            gate_reward_present=True,
            actual_gate_return_total_sampled=0.5,
            success_rate_sampled=0.10,
            first_strike_useful_speed_mean_sampled=0.44,
            worst_ratio_max_sampled=0.64,
            qvel_finite_exceedance_rate_sampled=0.04,
            qvel_max_abs_rad_s_sampled=2.4,
        ),
    ]


def _write_pilot_csv(path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("row_index", "field", "malformed"),
    (
        (0, "training_seed", "0.0"),
        (0, "git_dirty", "false"),
        (0, "reset_position_range_rad", "[0.0, 0.0]"),
        (0, "windup_enabled", ""),
        (2, "r_gate_weight", "nan"),
    ),
)
def test_csv_loader_rejects_malformed_typed_fields(
    tmp_path, row_index: int, field: str, malformed: str
) -> None:
    rows = _pilot_rows()
    rows[row_index][field] = malformed
    path = tmp_path / "summary.csv"
    _write_pilot_csv(path, rows)

    with pytest.raises(ValueError, match=field):
        guideline_campaign.load_and_validate_guideline_pilot_csv(path)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: [row.__setitem__("reset_rng_seed", 999) for row in rows],
        lambda rows: rows[0].pop("git_dirty"),
        lambda rows: rows[0].__setitem__("asset_git_dirty", 0),
        lambda rows: rows[0].pop("checkpoint_sha256"),
        lambda rows: rows[0].__setitem__("accepted_manifest_sha256", "not-a-sha"),
        lambda rows: rows[0].__setitem__("task", "wrong-task"),
        lambda rows: rows[0].__setitem__("accepted_checkpoint_sha256", "9" * 64),
        lambda rows: rows[0].__setitem__("campaign_config_sha256", "9" * 64),
        lambda rows: rows[1].__setitem__("treatment_config_sha256", "9" * 64),
        lambda rows: [row.__setitem__("treatment_config_sha256", "f" * 64) for row in rows[2:]],
        lambda rows: rows[0].__setitem__("training_code_revision", "not-a-revision"),
        lambda rows: rows[0].__setitem__("training_asset_revision", "3" * 40),
        lambda rows: rows[0].pop("q90_terminal_descent_perpendicular_error_m_sampled"),
        lambda rows: rows[0].__setitem__("q90_terminal_descent_perpendicular_error_m_sampled", math.nan),
        lambda rows: rows[0].__setitem__("q90_terminal_descent_perpendicular_error_m_sampled", 0.051),
        lambda rows: rows[0].__setitem__("episode_len_s", 3.9),
        lambda rows: rows[0].__setitem__("sampled_trace_digest", "bad"),
        lambda rows: rows[1].__setitem__("sampled_trace_artifact_sha256", rows[0]["sampled_trace_artifact_sha256"]),
    ),
    ids=(
        "wrong-frozen-rng",
        "missing-dirty",
        "nonbool-dirty",
        "missing-checkpoint-sha",
        "invalid-manifest-sha",
        "wrong-task",
        "checkpoint-sha-mismatch",
        "campaign-config-drift",
        "within-arm-config-drift",
        "same-config-across-arms",
        "invalid-training-code-revision",
        "training-asset-mismatch",
        "missing-primary-q90",
        "nonfinite-primary-q90",
        "out-of-range-primary-q90",
        "episode-horizon-drift",
        "invalid-trace-digest",
        "duplicate-trace-artifact",
    ),
)
def test_pilot_contract_rejects_load_bearing_identity_or_result_drift(mutate) -> None:
    rows = _pilot_rows()
    mutate(rows)

    with pytest.raises(ValueError):
        validate_guideline_pilot_rows(rows)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: [row.__setitem__("action_rng_seed", 999) for row in rows],
        lambda rows: [row.pop("q90_terminal_descent_perpendicular_error_m_sampled") for row in rows],
        lambda rows: rows[0].__setitem__("checkpoint_sha256", "not-a-sha"),
        lambda rows: rows[0].__setitem__("task", "wrong-task"),
        lambda rows: rows[0].__setitem__("accepted_checkpoint_sha256", "9" * 64),
        lambda rows: rows[0].__setitem__("campaign_config_sha256", "9" * 64),
        lambda rows: rows[0].__setitem__("q90_terminal_descent_perpendicular_error_m_sampled", "nan"),
    ),
    ids=(
        "wrong-frozen-rng",
        "missing-primary-q90",
        "invalid-checkpoint-sha",
        "wrong-task",
        "checkpoint-sha-mismatch",
        "campaign-config-drift",
        "nonfinite-primary-q90",
    ),
)
def test_csv_loader_rejects_persisted_identity_or_result_tampering(
    tmp_path, mutate
) -> None:
    rows = _pilot_rows()
    mutate(rows)
    path = tmp_path / "summary.csv"
    _write_pilot_csv(path, rows)

    with pytest.raises(ValueError):
        guideline_campaign.load_and_validate_guideline_pilot_csv(path)


@pytest.mark.parametrize(
    ("field", "mode"),
    (
        ("first_strike_useful_speed_mean_sampled", "missing"),
        ("worst_ratio_max_sampled", "missing"),
        ("qvel_finite_exceedance_rate_sampled", "missing"),
        ("qvel_max_abs_rad_s_sampled", "missing"),
        ("first_strike_useful_speed_mean_sampled", "nonfinite"),
        ("worst_ratio_max_sampled", "nonfinite"),
        ("qvel_finite_exceedance_rate_sampled", "nonfinite"),
        ("qvel_max_abs_rad_s_sampled", "nonfinite"),
        ("first_strike_useful_speed_mean_sampled", "negative"),
        ("worst_ratio_max_sampled", "negative"),
        ("qvel_finite_exceedance_rate_sampled", "negative"),
        ("qvel_max_abs_rad_s_sampled", "negative"),
        ("qvel_finite_exceedance_rate_sampled", "above-one"),
    ),
)
def test_direct_pilot_contract_rejects_invalid_secondary_outputs(
    field: str, mode: str
) -> None:
    rows = _pilot_rows()
    if mode == "missing":
        rows[0].pop(field)
    elif mode == "nonfinite":
        rows[0][field] = math.nan
    elif mode == "negative":
        rows[0][field] = -0.01
    else:
        rows[0][field] = 1.01

    with pytest.raises(ValueError, match=field):
        validate_guideline_pilot_rows(rows)


def test_pilot_contract_accepts_only_the_literal_four_rows_and_reports_continuation() -> None:
    summary = validate_guideline_pilot_rows(_pilot_rows())

    assert summary == {
        "valid": True,
        "row_identities": [("C0", 0), ("C0", 1), ("C-Gate", 0), ("C-Gate", 1)],
        "continuation_by_arm": {"C0": True, "C-Gate": True},
        "continuation_allowed": True,
        "validated_secondary_results": [
            {
                "treatment": "C0",
                "training_seed": 0,
                "first_strike_useful_speed_mean_sampled": 0.41,
                "worst_ratio_max_sampled": 0.61,
                "qvel_finite_exceedance_rate_sampled": 0.01,
                "qvel_max_abs_rad_s_sampled": 2.1,
            },
            {
                "treatment": "C0",
                "training_seed": 1,
                "first_strike_useful_speed_mean_sampled": 0.42,
                "worst_ratio_max_sampled": 0.62,
                "qvel_finite_exceedance_rate_sampled": 0.02,
                "qvel_max_abs_rad_s_sampled": 2.2,
            },
            {
                "treatment": "C-Gate",
                "training_seed": 0,
                "first_strike_useful_speed_mean_sampled": 0.43,
                "worst_ratio_max_sampled": 0.63,
                "qvel_finite_exceedance_rate_sampled": 0.03,
                "qvel_max_abs_rad_s_sampled": 2.3,
            },
            {
                "treatment": "C-Gate",
                "training_seed": 1,
                "first_strike_useful_speed_mean_sampled": 0.44,
                "worst_ratio_max_sampled": 0.64,
                "qvel_finite_exceedance_rate_sampled": 0.04,
                "qvel_max_abs_rad_s_sampled": 2.4,
            },
        ],
    }


def test_pilot_contract_accepts_a_permutation_and_returns_canonical_identities() -> None:
    rows = _pilot_rows()
    rows.reverse()

    summary = validate_guideline_pilot_rows(rows)

    assert summary["row_identities"] == [("C0", 0), ("C0", 1), ("C-Gate", 0), ("C-Gate", 1)]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: rows.__setitem__(1, dict(rows[0])),
        lambda rows: rows.append(dict(rows[0], treatment="other", training_seed=2)),
        lambda rows: rows.__setitem__(0, dict(rows[0], treatment="C-Gate")),
        lambda rows: rows.__setitem__(0, dict(rows[0], checkpoint_filename="model_498.pt")),
        lambda rows: rows.__setitem__(0, dict(rows[0], reset_digest="drift")),
        lambda rows: rows.__setitem__(0, dict(rows[0], guideline_geometry_digest="drift")),
        lambda rows: rows.__setitem__(0, dict(rows[0], reset_position_range_rad=[-0.05, 0.05])),
        lambda rows: rows.__setitem__(0, dict(rows[0], windup_enabled=True)),
        lambda rows: rows.__setitem__(0, dict(rows[0], impedance_mode="variable")),
        lambda rows: rows.__setitem__(0, dict(rows[0], imp_max_p=0.01)),
        lambda rows: rows.__setitem__(0, dict(rows[0], treatment_base_identity="drift")),
        lambda rows: rows.__setitem__(0, dict(rows[0], r_gate_weight=8.0)),
        lambda rows: rows.__setitem__(2, dict(rows[2], r_gate_weight=7.9)),
        lambda rows: rows.__setitem__(0, dict(rows[0], action_rng_seed=14)),
        lambda rows: rows.__setitem__(0, dict(rows[0], git_revision="other-clean-code")),
        lambda rows: rows.__setitem__(0, dict(rows[0], training_seed=0.0)),
        lambda rows: rows.__setitem__(0, dict(rows[0], git_dirty=True)),
        lambda rows: rows.__setitem__(0, dict(rows[0], impossible_success_n=1)),
        lambda rows: rows.__setitem__(0, dict(rows[0], lambda_dead_n=1)),
        lambda rows: rows.__setitem__(0, dict(rows[0], qvel_nonfinite_rate_sampled=math.nan)),
        lambda rows: rows.__setitem__(0, dict(rows[0], actual_gate_return_total_sampled=0.1)),
        lambda rows: rows.__setitem__(2, dict(rows[2], actual_gate_return_total_sampled=0.0)),
    ),
    ids=("duplicate", "extra", "replacement", "nonfinal", "reset-drift", "geometry-drift", "randomized-reset", "windup", "variable-impedance", "impulse-enforcement", "treatment-base-drift", "c0-r-gate", "cgate-r-gate-weight", "rng-drift", "provenance-drift", "noninteger-seed", "dirty-provenance", "impossible-success", "lambda-dead", "nonfinite-qvel", "c0-payout", "cgate-zero-payout"),
)
def test_pilot_contract_rejects_identity_and_validity_drift(mutate) -> None:
    rows = _pilot_rows()
    mutate(rows)

    with pytest.raises(ValueError):
        validate_guideline_pilot_rows(rows)


def test_pilot_continuation_requires_a_successful_seed_in_each_arm() -> None:
    rows = _pilot_rows()
    rows[0]["success_rate_sampled"] = 0.249

    summary = validate_guideline_pilot_rows(rows)

    assert summary["valid"] is True
    assert summary["continuation_by_arm"] == {"C0": False, "C-Gate": True}
    assert summary["continuation_allowed"] is False


@pytest.mark.parametrize("success_rate", (-0.01, 1.01, math.nan))
def test_pilot_contract_rejects_out_of_range_success_rates(success_rate: float) -> None:
    rows = _pilot_rows()
    rows[0]["success_rate_sampled"] = success_rate

    with pytest.raises(ValueError, match="success rate"):
        validate_guideline_pilot_rows(rows)
