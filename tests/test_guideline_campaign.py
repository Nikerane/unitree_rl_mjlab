from __future__ import annotations

import math

import numpy as np
import pytest

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
        "n_episodes_sampled": 512,
        "reset_digest": "reset-fixed",
        "guideline_geometry_digest": "geometry-fixed",
        "reset_rng_seed": 11,
        "observation_rng_seed": 12,
        "action_rng_seed": 13,
        "git_dirty": False,
        "asset_git_dirty": False,
        "git_revision": "code-clean",
        "asset_git_revision": "asset-clean",
        "impossible_success_n": 0,
        "lambda_dead_n": 0,
        "qvel_nonfinite_rate_sampled": 0.0,
    }
    return [
        dict(common, treatment="C0", training_seed=0, gate_reward_present=False,
             actual_gate_return_total_sampled=0.0, success_rate_sampled=0.30),
        dict(common, treatment="C0", training_seed=1, gate_reward_present=False,
             actual_gate_return_total_sampled=0.0, success_rate_sampled=0.10),
        dict(common, treatment="C-Gate", training_seed=0, gate_reward_present=True,
             actual_gate_return_total_sampled=1.0, success_rate_sampled=0.25),
        dict(common, treatment="C-Gate", training_seed=1, gate_reward_present=True,
             actual_gate_return_total_sampled=0.5, success_rate_sampled=0.10),
    ]


def test_pilot_contract_accepts_only_the_literal_four_rows_and_reports_continuation() -> None:
    summary = validate_guideline_pilot_rows(_pilot_rows())

    assert summary == {
        "valid": True,
        "row_identities": [("C0", 0), ("C0", 1), ("C-Gate", 0), ("C-Gate", 1)],
        "continuation_by_arm": {"C0": True, "C-Gate": True},
        "continuation_allowed": True,
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: rows.__setitem__(1, dict(rows[0])),
        lambda rows: rows.append(dict(rows[0], treatment="other", training_seed=2)),
        lambda rows: rows.__setitem__(0, dict(rows[0], treatment="C-Gate")),
        lambda rows: rows.__setitem__(0, dict(rows[0], checkpoint_filename="model_498.pt")),
        lambda rows: rows.__setitem__(0, dict(rows[0], reset_digest="drift")),
        lambda rows: rows.__setitem__(0, dict(rows[0], guideline_geometry_digest="drift")),
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
    ids=("duplicate", "extra", "replacement", "nonfinal", "reset-drift", "geometry-drift", "rng-drift", "provenance-drift", "noninteger-seed", "dirty-provenance", "impossible-success", "lambda-dead", "nonfinite-qvel", "c0-payout", "cgate-zero-payout"),
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
