"""Decision rules for the Cartesian-reference CPU qualification gate."""

from __future__ import annotations

import copy
import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.guideline.qualify_reference import (
    REQUIRED_SEEDS,
    RESET_POSITION_RANGE_RAD,
    TASK_ID,
    _configure_projection_x_ticks,
    _gate_disk_points,
    canonical_qvel_trace,
    corridor_window_errors,
    load_qualification_cfg,
    representative_plot_title,
    run_required_seeds,
    summarize_qualification,
    trace_numeric_is_finite,
    write_result_tables,
)


def _passing_rows() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "gates_crossed": 6,
            "contact_seen": True,
            "nail_progress_m": 0.03,
            "corridor_max_m": 0.004,
            "qvel_peak_rad_s": 3.0,
            "finite": True,
            "reset_arm_qpos_rad": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        }
        for seed in REQUIRED_SEEDS
    ]


def test_summary_schema_accepts_only_the_complete_preregistered_seed_set() -> None:
    summary = summarize_qualification(_passing_rows())

    assert summary == {
        "schema_version": 1,
        "required_seeds": list(range(1000, 1016)),
        "thresholds": {
            "required_gates": 6,
            "corridor_radius_m": 0.005,
            "qvel_limit_rad_s": 3.1415,
        },
        "passed": True,
        "qualified_resets": 16,
        "required_resets": 16,
        "identical_fixed_reset": True,
        "failures": [],
        "rows": [dict(row, qualifies=True, failures=[]) for row in _passing_rows()],
    }


@pytest.mark.parametrize(
    ("field", "bad_value", "reason"),
    (
        ("gates_crossed", 5, "fewer than six gates"),
        ("contact_seen", False, "no contact"),
        ("nail_progress_m", 0.0, "no nail progress"),
        ("corridor_max_m", 0.0050001, "corridor above 0.005 m"),
        ("qvel_peak_rad_s", 3.1415001, "qvel above 3.1415 rad/s"),
        ("qvel_peak_rad_s", math.nan, "non-finite value"),
        ("finite", False, "non-finite value"),
    ),
)
def test_any_single_reset_failure_fails_the_aggregate(
    field: str, bad_value: object, reason: str
) -> None:
    rows = _passing_rows()
    rows[7][field] = bad_value

    summary = summarize_qualification(rows)

    assert summary["passed"] is False
    assert summary["qualified_resets"] == 15
    assert summary["failures"] == [f"seed 1007: {reason}"]
    assert summary["rows"][7]["qualifies"] is False
    assert summary["rows"][7]["failures"] == [reason]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: rows.pop(),
        lambda rows: rows.__setitem__(-1, copy.deepcopy(rows[-2])),
        lambda rows: rows.append(dict(rows[-1], seed=1016)),
        lambda rows: rows.reverse(),
    ),
    ids=("success-selected-subset", "duplicate-seed", "extra-seed", "reordered"),
)
def test_missing_duplicate_or_extra_seed_fails_closed(mutate) -> None:
    rows = _passing_rows()
    mutate(rows)

    summary = summarize_qualification(rows)

    assert summary["passed"] is False
    assert summary["qualified_resets"] == 0
    assert summary["failures"] == [
        "seed set must be exactly 1000--1015 with one row per seed"
    ]


def test_missing_required_row_field_fails_closed() -> None:
    rows = _passing_rows()
    del rows[0]["corridor_max_m"]

    summary = summarize_qualification(rows)

    assert summary["passed"] is False
    assert summary["qualified_resets"] == 15
    assert summary["failures"] == ["seed 1000: missing corridor_max_m"]


def test_missing_seed_fails_closed_instead_of_raising() -> None:
    rows = _passing_rows()
    del rows[0]["seed"]

    summary = summarize_qualification(rows)

    assert summary["passed"] is False
    assert summary["failures"] == [
        "seed set must be exactly 1000--1015 with one row per seed"
    ]


@pytest.mark.parametrize(
    ("bad_reset", "aggregate_reason"),
    (
        (
            [0.1, -0.2, 0.3, -0.4, 0.5],
            "reset_arm_qpos_rad must be six finite values",
        ),
        (
            [0.1, -0.2, math.nan, -0.4, 0.5, -0.6],
            "reset_arm_qpos_rad must be six finite values",
        ),
        (
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.5],
            "reset_arm_qpos_rad must be identical across all execution seeds",
        ),
    ),
    ids=("wrong-length", "non-finite", "physical-drift"),
)
def test_invalid_or_drifting_realized_fixed_reset_fails_closed(
    bad_reset: list[float], aggregate_reason: str
) -> None:
    rows = _passing_rows()
    rows[1]["reset_arm_qpos_rad"] = bad_reset

    summary = summarize_qualification(rows)

    assert summary["passed"] is False
    assert summary["qualified_resets"] == 0
    assert summary["identical_fixed_reset"] is False
    assert summary["failures"] == [
        aggregate_reason
    ]


def test_qualification_loads_training_cfg_and_disables_auto_reset() -> None:
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=99),
        auto_reset=True,
        metrics={
            "cat_soft": SimpleNamespace(params={"imp_max_p": 0.0}),
            "waypoint_progress": object(),
            "first_strike": object(),
        },
        events={
            "reset_robot_joints": SimpleNamespace(
                params={"position_range": (0.0, 0.0)}
            )
        },
    )
    calls = []

    def loader(task_id: str, *, play: bool):
        calls.append((task_id, play))
        return cfg

    loaded = load_qualification_cfg(loader)

    assert calls == [(TASK_ID, False)]
    assert loaded is cfg
    assert loaded.scene.num_envs == 1
    assert loaded.auto_reset is False
    assert RESET_POSITION_RANGE_RAD == (0.0, 0.0)


def test_qualification_rejects_wrong_nonzero_training_reset_range() -> None:
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1),
        auto_reset=True,
        metrics={
            "cat_soft": SimpleNamespace(params={"imp_max_p": 0.0}),
            "waypoint_progress": object(),
            "first_strike": object(),
        },
        events={
            "reset_robot_joints": SimpleNamespace(
                params={"position_range": (-0.01, 0.01)}
            )
        },
    )

    with pytest.raises(RuntimeError, match="reset position_range.*0.0.*0.0"):
        load_qualification_cfg(lambda _task_id, *, play: cfg)


def test_canonical_qvel_trace_is_all_preintegration_states_plus_terminal_post() -> None:
    pre = np.array([[1.0, 2.0], [3.0, 4.0]])
    post = np.array([[3.0, 4.0], [5.0, 6.0]])

    np.testing.assert_array_equal(
        canonical_qvel_trace(pre, post),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )


def test_corridor_window_uses_gate_one_through_accepted_contact_inclusively() -> None:
    samples = [
        {"gates_crossed": 0, "accepted_contact": False, "raw_contact": False, "error_m": 0.009},
        {"gates_crossed": 0, "accepted_contact": False, "raw_contact": True, "error_m": 0.008},
        {"gates_crossed": 1, "accepted_contact": False, "raw_contact": False, "error_m": 0.004},
        {"gates_crossed": 6, "accepted_contact": True, "raw_contact": True, "error_m": 0.003},
        {"gates_crossed": 6, "accepted_contact": True, "raw_contact": False, "error_m": 0.020},
    ]

    assert corridor_window_errors(samples) == [0.004, 0.003]


def test_raw_trace_finiteness_rejects_interior_pre_gate_nan() -> None:
    trace = {
        "qvel_pre": [np.zeros(6), np.zeros(6)],
        "qvel_post": [np.zeros(6), np.zeros(6)],
        "path": [np.zeros(3), np.ones(3)],
        "depth": [0.0, 0.03],
        "actions": [np.zeros(3)],
        "samples": [
            {"gates_crossed": 0, "error_m": math.nan},
            {"gates_crossed": 1, "error_m": 0.004},
        ],
        "gate_centers": {0: np.zeros(3)},
        "contact_point": np.zeros(3),
        "physics_dt_s": 0.002,
    }

    assert not trace_numeric_is_finite(
        trace,
        initial_depth=0.0,
        reset_head=np.zeros(3),
        reset_arm_qpos=np.zeros(6),
        entry=np.zeros(3),
        nail=np.ones(3),
    )


def test_required_seed_runner_never_selects_a_successful_subset() -> None:
    seen = []

    rows = run_required_seeds(lambda seed: seen.append(seed) or {"seed": seed})

    assert seen == list(range(1000, 1016))
    assert rows == [{"seed": seed} for seed in range(1000, 1016)]


def test_json_and_csv_bank_identical_rows_with_explicit_si_columns(tmp_path) -> None:
    summary = summarize_qualification(_passing_rows())

    write_result_tables(summary, tmp_path)

    saved = json.loads((tmp_path / "qualification.json").read_text())
    with (tmp_path / "per_reset.csv").open(newline="") as stream:
        csv_rows = [
            {key: json.loads(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    assert csv_rows == saved["rows"]
    assert {
        "reset_arm_qpos_rad",
        "nail_progress_m",
        "corridor_max_m",
        "qvel_peak_rad_s",
    } <= set(csv_rows[0])
    assert b"\r\n" not in (tmp_path / "per_reset.csv").read_bytes()


def test_gate_disk_projection_is_not_a_spherical_circle_shortcut() -> None:
    center = np.array((0.5, 0.0, 0.2))
    points = _gate_disk_points(
        center,
        entry=np.array((0.5, 0.0, 0.25)),
        nail=np.array((0.5, 0.0, 0.10)),
        radius=0.015,
    )

    np.testing.assert_allclose(points[:, 2], center[2], atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(points[:, :2] - center[:2], axis=1), 0.015, atol=1e-12
    )


def test_xz_projection_uses_one_centered_x_tick() -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    try:
        axis.set_xlim(0.490, 0.520)
        _configure_projection_x_ticks(axis, centered=True)

        np.testing.assert_allclose(axis.get_xticks(), [0.505])
    finally:
        plt.close(figure)


def test_representative_plot_title_states_fixed_seed_and_aggregate_failure() -> None:
    rows = _passing_rows()
    for row in rows[:12]:
        row["corridor_max_m"] = 0.007523 if row["seed"] == 1000 else 0.006
    summary = summarize_qualification(rows)

    assert representative_plot_title(summary) == (
        "Representative 1/16, seed 1000 (chosen a priori): FAIL "
        "(7.523 mm vs 5.000 mm)\n"
        "Aggregate 4/16: FAIL | production guideline: frozen reset-head anchor "
        "to frozen nail top | scored interval: gate 1 to accepted contact"
    )
