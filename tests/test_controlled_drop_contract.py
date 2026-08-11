from dataclasses import replace
import json

import pytest

from src.tasks.hammer.calibration.controlled_drop_contract import (
    ControlledDropExecution,
    ControlledDropTrial,
    build_primary_result,
    summarize_primary_trials,
)


def _trial(index: int, impulse: float) -> ControlledDropTrial:
    return ControlledDropTrial(
        trial_index=index,
        h0_m=0.150,
        release_velocity_m_s=0.0,
        precontact_velocity_m_s=1.70,
        impulse_n_s=impulse,
        contacted=True,
        finalized=True,
        productive=True,
        reason="success",
        depth_at_contact_m=0.0,
        peak_depth_m=0.031,
    )


def _valid_trials() -> tuple[ControlledDropTrial, ...]:
    return tuple(
        _trial(index, impulse)
        for index, impulse in enumerate((0.10, 0.20, 0.30, 0.40, 0.50), start=1)
    )


def _execution() -> ControlledDropExecution:
    return ControlledDropExecution(
        code_revision="a" * 40,
        asset_revision="b" * 40,
        device="cpu",
        backend="mujoco",
        mujoco_version="3.8.1",
        mujoco_warp_version="3.8.1",
        mjlab_version="1.4.0",
        physics_dt_s=0.002,
        h0_m=0.150,
        mass_kg=0.2,
        radius_m=0.02,
        half_height_m=0.10,
        friction=(1.0, 0.005, 0.0001),
        slide_axis="z",
        slide_damping=0.1,
        slide_frictionloss=0.0,
        tracker_axis="z",
        tracker_window_substeps=25,
        tracker_progress_eps=0.001,
    )


def test_primary_summary_keeps_all_five_trials_and_uses_arithmetic_mean():
    trials = _valid_trials()

    summary = summarize_primary_trials(trials)

    assert summary.trials == trials
    assert summary.i_ref_mean_n_s == 0.30


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[:-1], "exactly five"),
        (lambda rows: rows + (rows[-1],), "exactly five"),
        (
            lambda rows: rows[:2] + (replace(rows[2], trial_index=4),) + rows[3:],
            "indices",
        ),
        (lambda rows: (replace(rows[0], trial_index=True),) + rows[1:], "numeric type"),
        (
            lambda rows: rows[:2] + (replace(rows[2], h0_m=float("nan")),) + rows[3:],
            "h0",
        ),
        (lambda rows: rows[:2] + (replace(rows[2], h0_m=True),) + rows[3:], "h0"),
        (lambda rows: tuple(replace(row, h0_m=0.149) for row in rows), "exactly 0.150"),
        (
            lambda rows: rows[:2]
            + (replace(rows[2], release_velocity_m_s=0.01),)
            + rows[3:],
            "released from rest",
        ),
        (
            lambda rows: rows[:2]
            + (replace(rows[2], release_velocity_m_s=True),)
            + rows[3:],
            "released from rest",
        ),
        (
            lambda rows: rows[:2]
            + (replace(rows[2], precontact_velocity_m_s=float("inf")),)
            + rows[3:],
            "pre-contact velocity",
        ),
        (
            lambda rows: rows[:2]
            + (replace(rows[2], precontact_velocity_m_s=True),)
            + rows[3:],
            "pre-contact velocity",
        ),
        (
            lambda rows: rows[:2] + (replace(rows[2], impulse_n_s=float("nan")),) + rows[3:],
            "impulse",
        ),
        (lambda rows: rows[:2] + (replace(rows[2], impulse_n_s=True),) + rows[3:], "impulse"),
        (lambda rows: rows[:2] + (replace(rows[2], impulse_n_s=0.0),) + rows[3:], "impulse"),
        (
            lambda rows: rows[:2]
            + (replace(rows[2], precontact_velocity_m_s=0.0),)
            + rows[3:],
            "pre-contact velocity",
        ),
        (
            lambda rows: rows[:2]
            + (replace(rows[2], precontact_velocity_m_s=-1.0),)
            + rows[3:],
            "pre-contact velocity",
        ),
        (lambda rows: rows[:2] + (replace(rows[2], contacted=False),) + rows[3:], "contacted"),
        (lambda rows: rows[:2] + (replace(rows[2], finalized=False),) + rows[3:], "finalized"),
        (lambda rows: rows[:2] + (replace(rows[2], productive=False),) + rows[3:], "productive"),
        (lambda rows: rows[:2] + (replace(rows[2], reason="other"),) + rows[3:], "reason"),
        (
            lambda rows: rows[:2] + (replace(rows[2], reason="window"),) + rows[3:],
            "one finalization reason",
        ),
        (
            lambda rows: rows[:2]
            + (replace(rows[2], depth_at_contact_m=float("nan")),)
            + rows[3:],
            "depth",
        ),
        (
            lambda rows: rows[:2]
            + (replace(rows[2], depth_at_contact_m=True),)
            + rows[3:],
            "depth",
        ),
        (
            lambda rows: rows[:2] + (replace(rows[2], peak_depth_m=0.0005),) + rows[3:],
            "depth progress",
        ),
        (
            lambda rows: rows[:2] + (replace(rows[2], peak_depth_m=True),) + rows[3:],
            "depth",
        ),
    ],
)
def test_primary_summary_fails_closed_on_invalid_trial_sets(mutate, message):
    with pytest.raises(ValueError, match=message):
        summarize_primary_trials(mutate(_valid_trials()))


def test_primary_summary_uses_strict_tracker_depth_progress_threshold():
    at_tracker_eps = _valid_trials()[:2] + (
        replace(_valid_trials()[2], peak_depth_m=0.0005),
    ) + _valid_trials()[3:]
    just_above_tracker_eps = _valid_trials()[:2] + (
        replace(_valid_trials()[2], peak_depth_m=0.0005001),
    ) + _valid_trials()[3:]

    with pytest.raises(ValueError, match="depth progress"):
        summarize_primary_trials(at_tracker_eps)

    summary = summarize_primary_trials(just_above_tracker_eps)

    assert summary.trials == just_above_tracker_eps


def test_primary_result_json_has_one_execution_and_nested_unfiltered_summary():
    result = build_primary_result(_execution(), _valid_trials())

    payload = result.to_json_dict()

    assert set(payload) == {"schema_version", "execution", "summary"}
    assert payload["schema_version"] == 1
    assert payload["execution"]["code_revision"] == "a" * 40
    assert [row["trial_index"] for row in payload["summary"]["trials"]] == [1, 2, 3, 4, 5]
    assert [row["impulse_n_s"] for row in payload["summary"]["trials"]] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert payload["summary"]["i_ref_mean_n_s"] == 0.3
    json.dumps(payload, sort_keys=True, allow_nan=False)
