"""Behavioral contract for the causal joint-position qualification artifact."""

from __future__ import annotations

import copy
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import evaluation.joint_position.qualify_joint_action as qualifier
from evaluation.joint_position.qualify_joint_action import (
    derive_scale_by_joint,
    normalized_action_for_target,
    qualification_passes,
    reduce_first_target_per_interval,
)
from src.tasks.hammer.config.z1.joint_position_contract import (
    load_joint_position_contract,
)


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
PHYSICAL_CLIPS = [
    [-2.61799, 2.61799],
    [0.0, 2.96706],
    [-2.87979, 0.0],
    [-1.51844, 1.51844],
    [-1.34390, 1.34390],
    [-2.79253, 2.79253],
]
TAPE = [
    [0.0, 0.5, -0.5, 0.0, 0.1, -0.1],
    [0.1, 0.6, -0.6, 0.1, 0.2, -0.2],
]
SOURCE_TASK_CONFIG_PROJECTION = {
    "action": {"name": "hammer_ik", "dimension": 3},
    "observations": {"policy": {"dimension": 44}},
    "rewards": {"nail_depth_delta": {"weight": 600.0}},
    "metrics": {"first_strike": {"enabled": True}},
    "events": {"reset": {"mode": "fixed"}},
    "actuators": {"arm": {"joint_names": JOINT_NAMES}},
    "timing": {"physics_dt_s": 0.002, "control_decimation": 10},
    "reset": {"joint_pos_offset": [0.0, 0.0]},
}


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


TAPE_SHA256 = _canonical_sha256(TAPE)


def _canonical_payload_sha256(payload: dict[str, object]) -> str:
    """Independent implementation of the documented non-self-referential digest."""
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _passing_rows() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "passed": True,
            "source_target_tape_sha256": TAPE_SHA256,
        }
        for seed in range(1000, 1016)
    ]


def _passing_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "action_semantics": "absolute_default_offset_joint_position",
        "source_task_id": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
        "source_code_revision": "a" * 40,
        "source_asset_revision": "b" * 40,
        "source_task_config_projection": copy.deepcopy(SOURCE_TASK_CONFIG_PROJECTION),
        "source_task_config_sha256": _canonical_sha256(SOURCE_TASK_CONFIG_PROJECTION),
        "joint_names": copy.deepcopy(JOINT_NAMES),
        "actuator_names": copy.deepcopy(JOINT_NAMES),
        "default_joint_pos_rad": [0.0, 0.5, -0.5, 0.0, 0.1, -0.1],
        "physical_clip_rad": copy.deepcopy(PHYSICAL_CLIPS),
        "scale_rad": [0.165, 0.165, 0.165, 0.165, 0.165, 0.165],
        "physics_dt_s": 0.002,
        "control_decimation": 10,
        "post_reference_hold_control_steps": 10,
        "seeds": list(range(1000, 1016)),
        "source_target_tape_rad": copy.deepcopy(TAPE),
        "source_target_tape_sha256": TAPE_SHA256,
        "per_seed_replay_rows": _passing_rows(),
        "decision": "PASS",
    }
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    return payload


def _add_source_task_config_projection(payload: dict[str, object]) -> None:
    projection = copy.deepcopy(SOURCE_TASK_CONFIG_PROJECTION)
    payload["source_task_config_projection"] = projection
    payload["source_task_config_sha256"] = _canonical_sha256(projection)
    payload["payload_sha256"] = _canonical_payload_sha256(payload)


def _refresh_tape_binding(payload: dict[str, object]) -> None:
    tape_sha256 = _canonical_sha256(payload["source_target_tape_rad"])
    payload["source_target_tape_sha256"] = tape_sha256
    for row in payload["per_seed_replay_rows"]:
        row["source_target_tape_sha256"] = tape_sha256
    payload["payload_sha256"] = _canonical_payload_sha256(payload)


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")


def test_reduction_uses_first_target_not_interval_future() -> None:
    """Selecting an interval's final DiffIK target would make this fail."""
    targets = np.arange(20 * 6, dtype=np.float64).reshape(20, 6)

    reduced = reduce_first_target_per_interval(targets, decimation=10)

    np.testing.assert_array_equal(reduced, targets[[0, 10]])


class _FakeRobotData:
    def __init__(self) -> None:
        self.joint_pos_target = np.zeros((1, 6), dtype=np.float64)
        self.joint_pos = np.full((1, 6), -99.0, dtype=np.float64)


class _FakeActionTerm:
    def __init__(self, data: _FakeRobotData) -> None:
        self._data = data
        self.calls = 0

    def apply_actions(self) -> None:
        self.calls += 1
        self._data.joint_pos_target[0] = float(self.calls)
        self._data.joint_pos[0] = 100.0 + float(self.calls)


class _FakeActionManager:
    def __init__(self, term: _FakeActionTerm) -> None:
        self._term = term

    def get_term(self, name: str) -> _FakeActionTerm:
        assert name == "ik_hammer_head"
        return self._term


class _FakeCfg:
    auto_reset = False
    decimation = 10


class _FakeCaptureEnv:
    def __init__(self) -> None:
        self.cfg = _FakeCfg()
        self.robot_data = _FakeRobotData()
        self.term = _FakeActionTerm(self.robot_data)
        self.action_manager = _FakeActionManager(self.term)
        self.reset_buf = np.array([False])
        self.reset_calls = 0
        self.nail_depth = 0.0
        self.first_strike = {"finalized": False, "productive": False}
        self.live_tracking_input = 0.0

    def step(self, action: object) -> None:
        del action
        for _ in range(self.cfg.decimation):
            self.term.apply_actions()
        self.robot_data.joint_pos[0] = np.arange(6, dtype=np.float64) + 0.25
        self.nail_depth = 0.032
        self.first_strike = {"finalized": True, "productive": True}
        self.reset_buf[0] = True
        self.live_tracking_input = float(
            np.sum(
                (
                    self.robot_data.joint_pos_target[0]
                    - self.robot_data.joint_pos[0]
                )
                ** 2
            )
        )

    def reset(self) -> None:
        self.reset_calls += 1
        self.robot_data.joint_pos.fill(-123.0)
        self.nail_depth = 0.0
        self.first_strike = {"finalized": False, "productive": False}


def test_capture_hook_records_first_applied_target_not_realized_or_final() -> None:
    """Moving capture after the decimation loop would record target ten, not one."""
    env = _FakeCaptureEnv()

    captured = qualifier.capture_control_step(
        env,
        np.zeros((1, 3), dtype=np.float64),
        action_term_name="ik_hammer_head",
        robot_data=env.robot_data,
        joint_ids=np.arange(6),
    )

    assert captured["applied_substeps"] == 10
    np.testing.assert_array_equal(captured["first_applied_target"], np.ones(6))
    assert not np.array_equal(captured["first_applied_target"], np.full(6, 10.0))
    assert not np.array_equal(captured["first_applied_target"], env.robot_data.joint_pos[0])


def test_terminal_transition_is_read_before_reset_and_matches_tracking_input() -> None:
    """Auto-resetting before terminal capture would erase the successful transition."""
    env = _FakeCaptureEnv()

    captured = qualifier.capture_control_step(
        env,
        np.zeros((1, 3), dtype=np.float64),
        action_term_name="ik_hammer_head",
        robot_data=env.robot_data,
        joint_ids=np.arange(6),
        terminal_reader=lambda: {
            "q_next": env.robot_data.joint_pos[0].copy(),
            "nail_depth": env.nail_depth,
            "first_strike": dict(env.first_strike),
            "applied_target": env.robot_data.joint_pos_target[0].copy(),
            "live_tracking_input": env.live_tracking_input,
        },
    )

    terminal = captured["terminal"]
    assert env.reset_calls == 0
    np.testing.assert_array_equal(terminal["q_next"], np.arange(6) + 0.25)
    assert terminal["nail_depth"] == 0.032
    assert terminal["first_strike"] == {"finalized": True, "productive": True}
    np.testing.assert_array_equal(terminal["applied_target"], np.full(6, 10.0))
    assert qualifier.squared_tracking_error(
        terminal["applied_target"], terminal["q_next"]
    ) == pytest.approx(terminal["live_tracking_input"])


def _passing_rollout_record(*, mode: str = "source") -> dict[str, object]:
    geometry = {
        "entry_m": [0.1, 0.2, 0.3],
        "nail_m": [0.1, 0.2, 0.0],
        "gate_centers_m": [
            [0.1, 0.2, 0.3 - (index + 1) * 0.3 / 7.0]
            for index in range(6)
        ],
    }
    return {
        "seed": 1000,
        "mode": mode,
        "joint_names": copy.deepcopy(JOINT_NAMES),
        "actuator_names": copy.deepcopy(JOINT_NAMES),
        "source_target_tape_sha256": TAPE_SHA256,
        "finite": True,
        "target_count": 12,
        "scheduled_post_reference_hold_control_steps": 10,
        "executed_post_reference_hold_control_steps": 2,
        "terminal_transition_captured": True,
        "terminal_reason": "success",
        "applied_substeps_per_target": [10] * 12,
        "normalized_action_saturation_count": 0,
        "normalized_action_peak_abs": 0.91,
        "cartesian_action_saturation_count": 4 if mode == "source" else 0,
        "cartesian_action_peak_abs": 1.32 if mode == "source" else None,
        "physical_target_clipping_count": 0,
        "joint_limit_violation_count": 0,
        "qvel_peak_rad_s": 3.0,
        "production_contact_seen": True,
        "first_strike_finalized": True,
        "first_strike_productive": True,
        "accepted_onset_control_step": 11,
        "playback_length": 10,
        "productive_precontact_axial_speed_m_s": 0.5,
        "first_event_nail_axis_impulse_n_s": 0.1,
        "success_within_first_event": True,
        "success": True,
        "ordered_waypoints_before_contact": 6,
        "post_gate_1_corridor_max_m": 0.005,
        "multi_gate_crossings": 0,
        "waypoint_credit_after_contact": False,
        "geometry": geometry,
    }


@pytest.mark.parametrize(
    ("field", "bad_value", "failure_fragment"),
    (
        ("joint_names", list(reversed(JOINT_NAMES)), "joint names"),
        ("finite", False, "non-finite"),
        ("applied_substeps_per_target", [10] * 11 + [9], "ten applied substeps"),
        ("normalized_action_saturation_count", 1, "normalized-action saturation"),
        ("physical_target_clipping_count", 1, "physical target clipping"),
        ("joint_limit_violation_count", 1, "joint-limit violation"),
        ("qvel_peak_rad_s", 3.1415001, "qvel"),
        ("production_contact_seen", False, "production-accepted contact"),
        ("first_strike_finalized", False, "did not finalize"),
        ("first_strike_productive", False, "not productive"),
        ("accepted_onset_control_step", 13, "accepted onset"),
        ("productive_precontact_axial_speed_m_s", 0.499, "precontact"),
        ("first_event_nail_axis_impulse_n_s", 0.0, "impulse"),
        ("success_within_first_event", False, "first event"),
        ("success", False, "success"),
        ("ordered_waypoints_before_contact", 5, "six ordered"),
        ("post_gate_1_corridor_max_m", 0.005001, "corridor"),
        ("multi_gate_crossings", 1, "multi-gate"),
        ("waypoint_credit_after_contact", True, "after contact"),
    ),
)
def test_rollout_gate_is_fail_closed(
    field: str, bad_value: object, failure_fragment: str
) -> None:
    """Dropping any frozen G1 condition would let its bad fixture pass."""
    record = _passing_rollout_record()
    record[field] = bad_value

    failures = qualifier.rollout_gate_failures(record)

    assert any(failure_fragment in failure for failure in failures)


def test_replay_gate_requires_source_geometry_equality() -> None:
    """Self-consistent but shifted replay gates must not qualify as the same path."""
    source = _passing_rollout_record()
    replay = _passing_rollout_record(mode="replay")
    replay["geometry"]["nail_m"][0] += 2e-6

    failures = qualifier.rollout_gate_failures(
        replay, expected_geometry=source["geometry"]
    )

    assert failures == ["source/replay geometry differs by more than 1e-6 m"]


def test_shortened_hold_requires_real_terminal_evidence() -> None:
    """A truncated hold without a terminal transition would fabricate missing commands."""
    record = _passing_rollout_record()
    record["terminal_transition_captured"] = False
    record["terminal_reason"] = None

    failures = qualifier.rollout_gate_failures(record)

    assert "shortened hold lacks captured terminal evidence" in failures


def test_executed_hold_count_must_match_executed_target_count() -> None:
    """Claiming ten scheduled holds as executed would misstate the causal tape."""
    record = _passing_rollout_record()
    record["executed_post_reference_hold_control_steps"] = 10

    failures = qualifier.rollout_gate_failures(record)

    assert "executed hold count does not match executed targets" in failures


def test_source_cartesian_clamp_is_diagnostic_not_joint_normalization_gate() -> None:
    """The exact legacy parent saturates 3-D playback without saturating joint replay."""
    record = _passing_rollout_record()

    assert record["cartesian_action_saturation_count"] == 4
    assert qualifier.rollout_gate_failures(record) == []


def test_replay_schedule_zero_order_holds_last_observed_causal_target() -> None:
    """Inventing a new post-terminal target instead of holding the last one would fail."""
    observed = np.arange(8 * 6, dtype=np.float64).reshape(8, 6)

    scheduled = qualifier.scheduled_replay_targets(
        observed, playback_length=6, hold_control_steps=10
    )

    assert scheduled.shape == (16, 6)
    np.testing.assert_array_equal(scheduled[:8], observed)
    np.testing.assert_array_equal(
        scheduled[8:], np.repeat(observed[-1:], 8, axis=0)
    )
    np.testing.assert_array_equal(observed[-1], np.arange(42, 48))


def test_contract_builder_is_deterministic_and_loader_compatible(
    tmp_path: Path,
) -> None:
    """Nondeterministic hashes or an incompatible Task-2 schema would fail loading."""
    source_rows = []
    replay_rows = []
    for seed in range(1000, 1016):
        source = _passing_rollout_record()
        replay = _passing_rollout_record(mode="replay")
        source["seed"] = seed
        replay["seed"] = seed
        source_rows.append(source)
        replay_rows.append(replay)

    kwargs = {
        "source_code_revision": "a" * 40,
        "source_asset_revision": "b" * 40,
        "source_task_config_projection": copy.deepcopy(
            SOURCE_TASK_CONFIG_PROJECTION
        ),
        "default_joint_pos_rad": [0.0, 0.5, -0.5, 0.0, 0.1, -0.1],
        "physical_clip_rad": copy.deepcopy(PHYSICAL_CLIPS),
        "source_target_tape_rad": copy.deepcopy(TAPE),
        "source_rows": source_rows,
        "replay_rows": replay_rows,
    }
    first = qualifier.build_contract_payload(**kwargs)
    second = qualifier.build_contract_payload(**kwargs)

    assert first == second
    assert first["decision"] == "PASS"
    assert first["source_target_tape_sha256"] == TAPE_SHA256
    assert first["source_task_config_sha256"] == _canonical_sha256(
        SOURCE_TASK_CONFIG_PROJECTION
    )
    assert first["payload_sha256"] == _canonical_payload_sha256(first)
    artifact = tmp_path / "built.json"
    qualifier.write_canonical_json(artifact, first)
    loaded = load_joint_position_contract(artifact)
    assert loaded.payload_sha256 == first["payload_sha256"]
    assert loaded.per_seed_replay_rows[0]["source"]["mode"] == "source"
    assert loaded.per_seed_replay_rows[0]["replay"]["mode"] == "replay"


def test_failed_contract_still_serializes_but_loader_rejects_it(
    tmp_path: Path,
) -> None:
    """A failed diagnostic must remain inspectable without becoming bankable evidence."""
    source_rows = []
    replay_rows = []
    for seed in range(1000, 1016):
        source = _passing_rollout_record()
        replay = _passing_rollout_record(mode="replay")
        source["seed"] = seed
        replay["seed"] = seed
        source_rows.append(source)
        replay_rows.append(replay)
    replay_rows[4]["success"] = False

    payload = qualifier.build_contract_payload(
        source_code_revision="a" * 40,
        source_asset_revision="b" * 40,
        source_task_config_projection=copy.deepcopy(SOURCE_TASK_CONFIG_PROJECTION),
        default_joint_pos_rad=[0.0, 0.5, -0.5, 0.0, 0.1, -0.1],
        physical_clip_rad=copy.deepcopy(PHYSICAL_CLIPS),
        source_target_tape_rad=copy.deepcopy(TAPE),
        source_rows=source_rows,
        replay_rows=replay_rows,
    )
    artifact = tmp_path / "failed.json"
    qualifier.write_canonical_json(artifact, payload)

    assert payload["decision"] == "FAIL"
    assert payload["per_seed_replay_rows"][4]["passed"] is False
    assert "success threshold not reached" in payload["per_seed_replay_rows"][4][
        "failures"
    ]
    assert json.loads(artifact.read_text())["decision"] == "FAIL"
    with pytest.raises(ValueError, match="decision must be PASS"):
        load_joint_position_contract(artifact)


def test_seed_range_parser_is_half_open_and_rejects_selection() -> None:
    """Inclusive end parsing or duplicate seed selection would alter the frozen bank."""
    assert qualifier.parse_seed_spec("1000:1016") == tuple(range(1000, 1016))
    with pytest.raises(ValueError):
        qualifier.parse_seed_spec("1000,1001")
    with pytest.raises(ValueError):
        qualifier.parse_seed_spec("1016:1000")


def test_source_config_projection_binds_all_eight_scientific_sections() -> None:
    """Omitting a task-defining section would leave config drift outside the digest."""
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
        play=True,
    )
    cfg.auto_reset = False
    projection = qualifier.source_task_config_projection(cfg)

    assert set(projection) == {
        "action",
        "observations",
        "rewards",
        "metrics",
        "events",
        "actuators",
        "timing",
        "reset",
    }
    assert projection["action"]["ik_hammer_head"]["actuator_names"] == JOINT_NAMES
    assert projection["timing"] == {
        "physics_dt_s": 0.002,
        "control_decimation": 10,
    }
    assert projection["reset"]["auto_reset"] is False
    assert projection["reset"]["reset_robot_joints"]["params"][
        "position_range"
    ] == [0.0, 0.0]
    assert "first_strike" in projection["metrics"]
    assert "r_waypoint_progress" in projection["rewards"]
    assert "actor" in projection["observations"]
    assert len(projection["actuators"]["robot"]) == 3
    json.dumps(projection, sort_keys=True, allow_nan=False)

    changed = copy.deepcopy(cfg)
    changed.actions["ik_hammer_head"].delta_pos_scale = 0.14
    assert qualifier.canonical_sha256(
        qualifier.source_task_config_projection(changed)
    ) != qualifier.canonical_sha256(projection)


def test_replay_cfg_uses_formula_scales_and_live_physical_clips() -> None:
    """A relative/current-position action or duplicated clip would break tape replay."""
    import src.tasks  # noqa: F401
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
        play=True,
    )
    scale = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
    replay = qualifier.configure_replay_cfg(
        cfg, scale_rad=scale, physical_clip_rad=np.asarray(PHYSICAL_CLIPS)
    )

    assert replay.auto_reset is False
    assert list(replay.actions) == ["joint_pos"]
    action = replay.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    assert tuple(action.actuator_names) == tuple(JOINT_NAMES)
    assert action.use_default_offset is True
    assert action.preserve_order is True
    assert action.scale == dict(zip(JOINT_NAMES, scale.tolist(), strict=True))
    assert action.clip == {
        name: tuple(bounds)
        for name, bounds in zip(JOINT_NAMES, PHYSICAL_CLIPS, strict=True)
    }


def test_live_physical_limits_must_equal_frozen_contract() -> None:
    """Silently trusting copied limits would hide a changed compiled asset."""
    assert qualifier.validate_live_physical_limits(PHYSICAL_CLIPS).shape == (6, 2)
    drifted = copy.deepcopy(PHYSICAL_CLIPS)
    drifted[2][0] += 1e-5
    with pytest.raises(RuntimeError, match="compiled physical joint limits"):
        qualifier.validate_live_physical_limits(drifted)


def test_compiled_limits_resolve_local_joints_through_raw_mujoco_indices() -> None:
    """Reading the float32 entity mirror would lose the exact XML-decimal contract."""
    raw_ranges = np.asarray(PHYSICAL_CLIPS + [[-0.01, 0.0]], dtype=np.float64)
    env = SimpleNamespace(
        sim=SimpleNamespace(mj_model=SimpleNamespace(jnt_range=raw_ranges))
    )
    robot = SimpleNamespace(
        indexing=SimpleNamespace(joint_ids=np.array([5, 4, 3, 2, 1, 0, 6]))
    )
    local_ids = np.array([5, 4, 3, 2, 1, 0])

    resolved = qualifier.read_compiled_physical_limits(env, robot, local_ids)

    np.testing.assert_array_equal(resolved, np.asarray(PHYSICAL_CLIPS))
    assert resolved.dtype == np.float64


def test_manager_snapshot_marks_nonfinite_terms_without_emitting_nan() -> None:
    """Checking only the summed reward would hide a non-finite individual term."""
    manager = SimpleNamespace(
        get_active_iterable_terms=lambda env_index: (
            [("finite_term", [1.25]), ("bad_term", [float("nan")])]
            if env_index == 0
            else []
        )
    )

    values, finite = qualifier.manager_term_snapshot(manager, env_index=0)

    assert values == {"finite_term": 1.25, "bad_term": None}
    assert finite is False
    json.dumps(values, allow_nan=False)


def test_numeric_tree_finiteness_covers_all_observation_groups() -> None:
    """Ignoring a non-policy observation group would let a corrupt state qualify."""
    assert qualifier.numeric_tree_is_finite(
        {"actor": np.array([[1.0, 2.0]]), "critic": np.array([[3.0]])}
    )
    assert not qualifier.numeric_tree_is_finite(
        {"actor": np.array([[1.0, 2.0]]), "critic": np.array([[float("inf")]])}
    )


def test_scale_has_exploration_floor_and_margin() -> None:
    """Dropping either the 0.05-rad floor or 1.10 margin would make this fail."""
    tape = np.array([[0.2, -0.1], [0.4, 0.3]])
    default = np.array([0.1, 0.0])
    expected = 1.10 * (np.array([0.3, 0.3]) + 0.05)

    np.testing.assert_allclose(derive_scale_by_joint(tape, default), expected)


def test_normalized_tape_round_trips_without_saturation() -> None:
    """Using current state or a mismatched affine sign would make this fail."""
    tape = np.array([[0.2, -0.1], [0.4, 0.3]])
    default = np.array([0.1, 0.0])
    scale = np.array([0.4, 0.4])

    action = normalized_action_for_target(tape, default, scale)

    assert np.max(np.abs(action)) < 1.0
    np.testing.assert_allclose(default + scale * action, tape)


def test_qualification_requires_exact_passing_seed_sequence() -> None:
    """Accepting a missing seed or a failed replay row would make this fail."""
    rows = _passing_rows()

    assert qualification_passes(rows) is True
    assert qualification_passes(rows[:-1]) is False
    rows[7]["passed"] = False
    assert qualification_passes(rows) is False


def test_loader_accepts_complete_canonical_contract(tmp_path: Path) -> None:
    """Rejecting a complete, internally consistent artifact would make this fail."""
    artifact = tmp_path / "joint_position.json"
    _write_payload(artifact, _passing_payload())

    contract = load_joint_position_contract(artifact)

    assert contract.joint_names == tuple(JOINT_NAMES)
    assert contract.actuator_names == tuple(JOINT_NAMES)
    np.testing.assert_allclose(contract.default_joint_pos_rad, [0.0, 0.5, -0.5, 0.0, 0.1, -0.1])
    assert contract.payload_sha256 == _passing_payload()["payload_sha256"]


def test_loader_returns_read_only_scientific_arrays(tmp_path: Path) -> None:
    """Mutating validated evidence after its digest check would make this fail."""
    artifact = tmp_path / "joint_position.json"
    _write_payload(artifact, _passing_payload())

    contract = load_joint_position_contract(artifact)

    for array in (
        contract.default_joint_pos_rad,
        contract.physical_clip_rad,
        contract.scale_rad,
        contract.source_target_tape_rad,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 123.0


@pytest.mark.parametrize(
    "field",
    (
        "default_joint_pos_rad",
        "physical_clip_rad",
        "scale_rad",
        "source_target_tape_rad",
    ),
)
def test_loader_scientific_arrays_cannot_be_made_writeable(
    tmp_path: Path, field: str
) -> None:
    """Backing validated arrays with owned mutable memory would make this fail."""
    artifact = tmp_path / "joint_position.json"
    _write_payload(artifact, _passing_payload())
    array = getattr(load_joint_position_contract(artifact), field)

    try:
        array.setflags(write=True)
    except ValueError:
        assert array.flags.writeable is False
    else:
        before = float(array.flat[0])
        array.flat[0] = before + 1.0
        assert float(array.flat[0]) != before
        pytest.fail(f"{field} accepted setflags(write=True) and mutation")


def test_loader_accepts_projection_bound_config_identity(tmp_path: Path) -> None:
    """Ignoring the explicit source-config projection would make this fail."""
    payload = _passing_payload()
    _add_source_task_config_projection(payload)
    artifact = tmp_path / "projected_config.json"
    _write_payload(artifact, payload)

    contract = load_joint_position_contract(artifact)

    assert contract.source_task_config_projection["timing"]["control_decimation"] == 10
    assert contract.source_task_config_sha256 == _canonical_sha256(
        SOURCE_TASK_CONFIG_PROJECTION
    )
    with pytest.raises(TypeError):
        contract.source_task_config_projection["timing"]["control_decimation"] = 11


def test_loader_rejects_arbitrary_rehashed_task_config_digest(tmp_path: Path) -> None:
    """Trusting a well-formed but projection-unbound config digest would make this fail."""
    payload = _passing_payload()
    _add_source_task_config_projection(payload)
    payload["source_task_config_sha256"] = "d" * 64
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "unbound_config_digest.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="does not match source_task_config_projection"):
        load_joint_position_contract(artifact)


@pytest.mark.parametrize(
    ("field", "index"),
    (
        ("default_joint_pos_rad", 0),
        ("physical_clip_rad", (0, 0)),
        ("scale_rad", 0),
        ("source_target_tape_rad", (0, 0)),
    ),
)
def test_loader_rejects_json_booleans_in_numeric_arrays(
    tmp_path: Path, field: str, index: int | tuple[int, int]
) -> None:
    """Letting NumPy coerce JSON booleans into physical numbers would make this fail."""
    payload = _passing_payload()
    if isinstance(index, tuple):
        payload[field][index[0]][index[1]] = False
    else:
        payload[field][index] = False
    _refresh_tape_binding(payload)
    artifact = tmp_path / f"boolean_{field}.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="must not contain JSON booleans"):
        load_joint_position_contract(artifact)


def test_loader_returns_deeply_immutable_replay_rows(tmp_path: Path) -> None:
    """Mutating validated replay evidence through nested mappings would make this fail."""
    payload = _passing_payload()
    payload["per_seed_replay_rows"][0]["metrics"] = {"waypoint_hits": [1, 2, 3]}
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "nested_rows.json"
    _write_payload(artifact, payload)

    contract = load_joint_position_contract(artifact)
    row = contract.per_seed_replay_rows[0]

    assert isinstance(row, Mapping)
    assert row.get("seed") == 1000
    assert row["metrics"]["waypoint_hits"] == (1, 2, 3)
    with pytest.raises(TypeError):
        row["passed"] = False
    with pytest.raises(TypeError):
        row["metrics"]["waypoint_hits"] = ()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.update(default_joint_pos_rad=[0.0] * 5),
        lambda payload: payload.update(joint_names=JOINT_NAMES[:-1]),
        lambda payload: payload.update(joint_names=list(reversed(JOINT_NAMES))),
    ),
    ids=("wrong-width", "missing-joint", "permuted-joints"),
)
def test_loader_rejects_wrong_joint_contract_width_or_order(
    tmp_path: Path, mutate
) -> None:
    """Weakening the ordered six-joint identity check would make this fail."""
    payload = _passing_payload()
    mutate(payload)
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "invalid.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_missing_seed(tmp_path: Path) -> None:
    """Allowing incomplete deterministic-repeatability evidence would make this fail."""
    payload = _passing_payload()
    payload["per_seed_replay_rows"] = _passing_rows()[:-1]
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "missing_seed.json"
    _write_payload(artifact, payload)

    assert qualification_passes(payload["per_seed_replay_rows"]) is False
    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_unknown_schema_field(tmp_path: Path) -> None:
    """Silently accepting an uninterpreted artifact field would make this fail."""
    payload = _passing_payload()
    payload["scale_derivation"] = "hand_tuned"
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "unknown_field.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="unexpected fields"):
        load_joint_position_contract(artifact)


def test_loader_rejects_boolean_schema_version(tmp_path: Path) -> None:
    """Treating JSON true as integer schema version one would make this fail."""
    payload = _passing_payload()
    payload["schema_version"] = True
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "boolean_schema.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="schema_version"):
        load_joint_position_contract(artifact)


def test_loader_rejects_nonhex_task_config_digest(tmp_path: Path) -> None:
    """Accepting a signed integer-shaped string as a SHA-256 digest would make this fail."""
    payload = _passing_payload()
    payload["source_task_config_sha256"] = "-" + "c" * 63
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "nonhex_digest.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_joint_position_contract(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_code_revision", "main"),
        ("source_asset_revision", "b" * 39),
    ),
)
def test_loader_rejects_unpinned_source_revision(
    tmp_path: Path, field: str, value: str
) -> None:
    """Accepting a symbolic or truncated revision would make this fail."""
    payload = _passing_payload()
    payload[field] = value
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "unpinned_revision.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="40-character lowercase Git commit"):
        load_joint_position_contract(artifact)


def test_loader_rejects_wrong_source_task_identity(tmp_path: Path) -> None:
    """Loading evidence captured from a different task would make this fail."""
    payload = _passing_payload()
    payload["source_task_id"] = "Unitree-Z1-Hammer-Other-Task"
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "wrong_source_task.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="source_task_id"):
        load_joint_position_contract(artifact)


def test_loader_rejects_nonfinite_json_values(tmp_path: Path) -> None:
    """Permitting JSON NaN in a scientific artifact would make this fail."""
    artifact = tmp_path / "nonfinite.json"
    artifact.write_text('{"scale_rad":[NaN]}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_nonphysical_clips(tmp_path: Path) -> None:
    """Accepting an inverted physical interval would make this fail."""
    payload = _passing_payload()
    payload["physical_clip_rad"] = copy.deepcopy(PHYSICAL_CLIPS)
    payload["physical_clip_rad"][2] = [0.0, -2.87979]
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "inverted_clip.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_scale_that_violates_frozen_derivation(tmp_path: Path) -> None:
    """Accepting hand-tuned scales instead of the frozen formula would make this fail."""
    payload = _passing_payload()
    payload["scale_rad"] = [0.2] * 6
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "hand_tuned_scale.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="frozen derivation"):
        load_joint_position_contract(artifact)


def test_loader_rejects_inconsistent_source_tape_hashes(tmp_path: Path) -> None:
    """Not binding every replay row to the recorded causal tape would make this fail."""
    payload = _passing_payload()
    payload["per_seed_replay_rows"] = _passing_rows()
    payload["per_seed_replay_rows"][3]["source_target_tape_sha256"] = "d" * 64
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "drifted_tape.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_nonpassing_decision(tmp_path: Path) -> None:
    """Treating a FAIL artifact as banked evidence would make this fail."""
    payload = _passing_payload()
    payload["decision"] = "FAIL"
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "failed.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_mismatched_canonical_payload_sha256(tmp_path: Path) -> None:
    """Skipping the non-self-referential canonical digest check would make this fail."""
    payload = _passing_payload()
    payload["scale_rad"][0] = 0.23
    artifact = tmp_path / "tampered.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)
