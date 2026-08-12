"""Additive config contract for the fixed-gain Z1 joint-position arm."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import dataclasses
import math
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from mjlab.envs.mdp.actions import (
    DifferentialIKActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from mjlab.envs.mdp.observations import last_action
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks import registry as task_registry
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from src.tasks.hammer.config.z1.joint_position_contract import (
    JOINT_NAMES,
    load_joint_position_contract,
)
from src.tasks.hammer.mdp.trackability import joint_trackability_cost
from src.tasks.hammer.mdp.rewards import ImitationPriorTerm, action_rate_penalty
from src.tasks.hammer.mdp.variable_impedance import (
    JointStiffnessActionCfg,
    expand_variable_impedance_model_fields,
)


PARENT_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    "CProgress-Vel-Delivered4"
)
FIC0_TASK = f"{PARENT_TASK}-JointPosition-Fixed"
FICTT_TASK = f"{FIC0_TASK}-TT"
DIRECT_FIC0_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-Fixed"
)
DIRECT_FICTT_TASK = f"{DIRECT_FIC0_TASK}-TT"
VIC_TT_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-VariableImpedance-TT"
)
JOINT_POLICY_TASKS = frozenset(
    (FIC0_TASK, FICTT_TASK, DIRECT_FIC0_TASK, DIRECT_FICTT_TASK, VIC_TT_TASK)
)
FIC_CONTROLLED_DROP_I_REF_N_S = 0.2799950838088989
ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json"
)


def _load_fic0(*, play: bool = False):
    # Keep a missing registration as an intentional test failure, not a fixture
    # setup error or an opaque registry KeyError during the RED phase.
    assert FIC0_TASK in list_tasks(), f"missing registered task {FIC0_TASK}"
    return load_env_cfg(FIC0_TASK, play=play)


def _load_fictt(*, play: bool = False):
    assert FICTT_TASK in list_tasks(), f"missing registered task {FICTT_TASK}"
    return load_env_cfg(FICTT_TASK, play=play)


def _load_direct_fic0(*, play: bool = False):
    assert DIRECT_FIC0_TASK in list_tasks(), (
        f"missing registered task {DIRECT_FIC0_TASK}"
    )
    return load_env_cfg(DIRECT_FIC0_TASK, play=play)


def _load_direct_fictt(*, play: bool = False):
    assert DIRECT_FICTT_TASK in list_tasks(), (
        f"missing registered task {DIRECT_FICTT_TASK}"
    )
    return load_env_cfg(DIRECT_FICTT_TASK, play=play)


def _load_victt(*, play: bool = False):
    assert VIC_TT_TASK in list_tasks(), f"missing registered task {VIC_TT_TASK}"
    return load_env_cfg(VIC_TT_TASK, play=play)


def _canonicalize(value):
    """Turn config trees into stable, content-only values for exact diffing."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return tuple((key, _canonicalize(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_canonicalize(item) for item in value)
    if isinstance(value, np.ndarray):
        return (str(value.dtype), tuple(value.shape), tuple(value.reshape(-1).tolist()))
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu()
        return (str(cpu.dtype), tuple(cpu.shape), tuple(cpu.reshape(-1).tolist()))
    if callable(value):
        return (value.__module__, value.__qualname__)
    return value


def test_config_canonicalizer_detects_pure_mapping_insertion_order_changes() -> None:
    """Reordering a manager term must not disappear inside exact config diffing."""
    first = {"nested": {"alpha": 1, "beta": 2}}
    reordered = {"nested": {"beta": 2, "alpha": 1}}

    assert _canonicalize(first) != _canonicalize(reordered)


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_direct_fic_treatment_registration_and_reference_contract(play: bool) -> None:
    """Direct-reference FIC is launchable without waypoint guidance state."""
    fic0 = _load_direct_fic0(play=play)
    fictt = _load_direct_fictt(play=play)

    assert _canonicalize(load_rl_cfg(DIRECT_FIC0_TASK)) == _canonicalize(
        load_rl_cfg(PARENT_TASK)
    )
    assert _canonicalize(load_rl_cfg(DIRECT_FICTT_TASK)) == _canonicalize(
        load_rl_cfg(PARENT_TASK)
    )
    assert tuple(fic0.observations["actor"].terms) == (
        "joint_pos", "joint_vel", "ee_pos", "ee_vel", "head_pos", "head_vel",
        "nail_top_pos", "nail_depth", "strike_phase", "strike_ref_error", "actions",
    )
    assert tuple(fic0.observations["critic"].terms) == tuple(
        fic0.observations["actor"].terms
    )
    assert not {"next_gate_vector", "completed_gate_fraction",
                "guideline_perpendicular_error", "waypoint_progress_state"} & set(
        fic0.observations["actor"].terms
    )
    assert "waypoint_progress" not in fic0.metrics
    assert {"r_gate", "r_waypoint_progress"}.isdisjoint(fic0.rewards)
    assert "r_imit" in fic0.rewards
    imit = fic0.rewards["r_imit"]
    assert imit.func is ImitationPriorTerm
    assert imit.weight == pytest.approx(0.10)
    assert imit.params["sigma"] == pytest.approx(0.05)
    assert imit.params["sensor_name"] == "hammer_nail_contact"
    assert imit.params["robot_cfg"].site_names == ("hammer_head_site",)

    reset = fic0.events["reset_robot_joints"].params
    assert reset["position_range"] == (0.0, 0.0)
    assert reset["velocity_range"] == (0.0, 0.0)
    assert tuple(fictt.observations["actor"].terms) == tuple(
        fic0.observations["actor"].terms
    )

    if play:
        assert not fic0.curriculum
        assert not fictt.curriculum
    else:
        assert fic0.curriculum["r_imit_anneal"].params["stages"] == [
            {"step": 0, "weight": 0.10},
            {"step": 1200, "weight": 0.08},
            {"step": 2400, "weight": 0.06},
            {"step": 3600, "weight": 0.04},
            {"step": 4800, "weight": 0.02},
            {"step": 6000, "weight": 0.00},
        ]


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_direct_fic_preserves_banked_identity_and_tt_isolation(play: bool) -> None:
    """Direct FIC keeps D4/impulse machinery while TT adds only r_tt."""
    fic0 = _load_direct_fic0(play=play)
    fictt = _load_direct_fictt(play=play)

    action = fic0.actions["joint_position"]
    contract = load_joint_position_contract(ARTIFACT)
    assert isinstance(action, JointPositionActionCfg)
    assert tuple(action.actuator_names) == JOINT_NAMES == contract.joint_names
    assert action.use_default_offset is True
    assert action.scale == dict(zip(JOINT_NAMES, contract.scale_rad.tolist(), strict=True))
    assert action.clip == {
        name: tuple(bounds)
        for name, bounds in zip(
            JOINT_NAMES, contract.physical_clip_rad.tolist(), strict=True
        )
    }
    assert _actuator_signature(fic0) == (
        ("BuiltinPositionActuatorCfg", 1000.0, 100.0, 30.0, 0.01,
         ("joint1", "joint3", "joint4", "joint5", "joint6")),
        ("BuiltinPositionActuatorCfg", 1500.0, 150.0, 60.0, 0.02, ("joint2",)),
        ("BuiltinPositionActuatorCfg", 100.0, 20.0, 30.0, 0.005, ("jointGripper",)),
    )
    assert fic0.rewards["delivered_impulse"].params["i_ref"] == (
        FIC_CONTROLLED_DROP_I_REF_N_S
    )
    assert fic0.metrics["substep_impulse_rows"].params["enabled"] is False
    assert "substep_impulse" in fic0.metrics
    cat = fic0.metrics["cat_soft"].params
    assert cat["use_vel"] is True
    assert cat["vel_detection"] == "substep"
    assert cat["use_impulse"] is True
    assert tuple(cat["imp_limit"]) == (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
    assert cat["imp_max_p"] == 0.0
    assert "vel_hard" not in fic0.terminations
    assert "cat_vel" not in fic0.terminations

    assert tuple(fictt.rewards) == (*tuple(fic0.rewards), "r_tt")
    term = fictt.rewards["r_tt"]
    assert term.func is joint_trackability_cost
    assert term.weight == -1.0
    assert term.params["k_tt"] == 1.0
    assert tuple(term.params["robot_cfg"].joint_names) == JOINT_NAMES
    assert term.params["robot_cfg"].preserve_order is True
    fictt_tree = _canonicalize(fictt)
    fictt_tree["rewards"] = fictt_tree["rewards"][:-1]
    assert fictt_tree == _canonicalize(fic0)


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_victt_registration_has_the_approved_ordered_policy_contract(play: bool) -> None:
    """VIC-TT adds one bounded gain term while keeping FIC-TT's learning inputs."""
    cfg = _load_victt(play=play)

    assert tuple(cfg.actions) == ("joint_position", "joint_stiffness")
    stiffness = cfg.actions["joint_stiffness"]
    assert isinstance(stiffness, JointStiffnessActionCfg)
    assert stiffness.entity_name == "robot"
    assert tuple(stiffness.joint_names) == JOINT_NAMES
    assert stiffness.C == pytest.approx(1.25)

    expansion = cfg.events["expand_variable_impedance_model_fields"]
    assert isinstance(expansion, EventTermCfg)
    assert expansion.func is expand_variable_impedance_model_fields
    assert expansion.mode == "startup"
    assert expansion.params == {}
    assert expansion.func.model_fields == (
        "actuator_gainprm",
        "actuator_biasprm",
    )

    for group_name in ("actor", "critic"):
        actions = cfg.observations[group_name].terms["actions"]
        assert actions.func is last_action
        assert actions.params == {"action_name": "joint_position"}
    action_rate = cfg.rewards["action_rate"]
    assert action_rate.func is action_rate_penalty
    assert action_rate.weight == pytest.approx(-0.01)
    assert action_rate.params == {"action_name": "joint_position"}

    r_tt = cfg.rewards["r_tt"]
    assert r_tt.func is joint_trackability_cost
    assert r_tt.weight == pytest.approx(-1.0)
    assert r_tt.params["k_tt"] == pytest.approx(1.0)
    reset = cfg.events["reset_robot_joints"].params
    assert reset["position_range"] == (0.0, 0.0)
    assert reset["velocity_range"] == (0.0, 0.0)
    assert _canonicalize(load_rl_cfg(VIC_TT_TASK)) == _canonicalize(
        load_rl_cfg(DIRECT_FICTT_TASK)
    )


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_victt_is_the_exact_approved_delta_from_direct_fictt(play: bool) -> None:
    """No reward, constraint, reset, plant, or learner drift can hide in VIC-TT."""
    fictt = _load_direct_fictt(play=play)
    victt = _load_victt(play=play)
    fictt_tree = _canonicalize(fictt)
    victt_tree = _canonicalize(victt)

    assert {
        field for field in fictt_tree if fictt_tree[field] != victt_tree[field]
    } == {"actions", "events", "observations", "rewards"}

    reconstructed = copy.deepcopy(victt)
    reconstructed.actions.pop("joint_stiffness")
    reconstructed.events.pop("expand_variable_impedance_model_fields")
    for group_name in ("actor", "critic"):
        reconstructed.observations[group_name].terms["actions"].params.clear()
    reconstructed.rewards["action_rate"].params.clear()
    assert _canonicalize(reconstructed) == fictt_tree


@pytest.mark.integration
def test_victt_live_two_world_action_and_observation_contract() -> None:
    """The registered task constructs live with isolated arm gains and 40 inputs."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = _load_victt(play=True)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        observations, _ = env.reset(seed=20260813)
        assert set(observations) == {"actor", "critic"}
        for observation in observations.values():
            assert observation.shape == (2, 40)
            assert bool(torch.isfinite(observation).all())
        assert env.action_manager.action.shape == (2, 12)
        assert tuple(env.action_manager.active_terms) == (
            "joint_position",
            "joint_stiffness",
        )

        stiffness = env.action_manager.get_term("joint_stiffness")
        assert tuple(stiffness.control_ids.tolist()) == (0, 5, 1, 2, 3, 4)
        torch.testing.assert_close(
            stiffness.nominal_kp,
            torch.tensor(
                [1000.0, 1500.0, 1000.0, 1000.0, 1000.0, 1000.0],
                device=env.device,
            ),
        )
        torch.testing.assert_close(
            stiffness.nominal_kd,
            torch.tensor(
                [100.0, 150.0, 100.0, 100.0, 100.0, 100.0],
                device=env.device,
            ),
        )
        assert {"actuator_gainprm", "actuator_biasprm"}.issubset(
            env.sim.expanded_fields
        )

        model = env.sim.model
        force_range_before = model.actuator_forcerange.clone()
        force_limited_before = model.actuator_forcelimited.clone()
        gripper_ids = tuple(
            sorted(set(range(model.nu)) - set(stiffness.control_ids.tolist()))
        )
        assert gripper_ids == (6,)
        gripper_gain_before = model.actuator_gainprm[:, gripper_ids].clone()
        gripper_bias_before = model.actuator_biasprm[:, gripper_ids].clone()

        action = torch.zeros((2, 12), device=env.device)
        action[0, 6:] = -1.0
        action[1, 6:] = 1.0
        observations, _, _, _, _ = env.step(action)
        for observation in observations.values():
            assert observation.shape == (2, 40)
            assert bool(torch.isfinite(observation).all())

        expected_kp = torch.stack(
            (stiffness.nominal_kp / 1.25, stiffness.nominal_kp * 1.25)
        )
        expected_kd = torch.stack(
            (
                stiffness.nominal_kd / math.sqrt(1.25),
                stiffness.nominal_kd * math.sqrt(1.25),
            )
        )
        control_ids = stiffness.control_ids
        torch.testing.assert_close(
            model.actuator_gainprm[:, control_ids, 0], expected_kp
        )
        torch.testing.assert_close(
            model.actuator_biasprm[:, control_ids, 1], -expected_kp
        )
        torch.testing.assert_close(
            model.actuator_biasprm[:, control_ids, 2], -expected_kd
        )
        torch.testing.assert_close(
            model.actuator_forcerange[:], force_range_before, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            model.actuator_forcelimited[:],
            force_limited_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.actuator_gainprm[:][:, gripper_ids],
            gripper_gain_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.actuator_biasprm[:][:, gripper_ids],
            gripper_bias_before,
            rtol=0.0,
            atol=0.0,
        )
    finally:
        env.close()


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_direct_and_waypoint_fic_differ_only_at_guidance_seams(play: bool) -> None:
    """The direct arm trades waypoint guidance only for the weak reference prior."""
    direct = _load_direct_fic0(play=play)
    waypoint = _load_fic0(play=play)

    for group in ("actor", "critic"):
        for name in (
            "next_gate_vector", "completed_gate_fraction",
            "guideline_perpendicular_error", "waypoint_progress_state",
        ):
            waypoint.observations[group].terms.pop(name)
    waypoint.metrics.pop("waypoint_progress")
    waypoint.rewards.pop("r_waypoint_progress")
    waypoint.rewards["r_imit"] = direct.rewards["r_imit"]
    waypoint.curriculum = direct.curriculum
    # ``imitation=True`` creates r_imit before z1_hammer_env_cfg installs the
    # delivered term; the waypoint factory appends its reward after delivery.
    # Reward insertion order is not a treatment seam, so align it for content
    # comparison after removing the two distinct guidance readers.
    direct.rewards["r_imit"] = direct.rewards.pop("r_imit")
    assert _canonicalize(waypoint) == _canonicalize(direct)


def test_fictt_is_registered_for_train_and_play_with_the_parent_learner() -> None:
    """The calibrated treatment must be launchable under the matched CatPPO."""
    assert FICTT_TASK in list_tasks()
    train = _load_fictt()
    play = _load_fictt(play=True)
    assert list(train.actions) == ["joint_position"]
    assert list(play.actions) == ["joint_position"]
    assert _canonicalize(load_rl_cfg(FICTT_TASK)) == _canonicalize(
        load_rl_cfg(PARENT_TASK)
    )
    assert (
        load_rl_cfg(FICTT_TASK).algorithm.class_name
        == "src.tasks.hammer.rl.cat_ppo:CatPPO"
    )


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_fictt_installs_only_the_strict_banked_trackability_cost(play: bool) -> None:
    """Wrong gain, sign, function, joint order, or a second mutation confounds TT."""
    fic0 = _load_fic0(play=play)
    fictt = _load_fictt(play=play)

    assert "r_tt" not in fic0.rewards
    assert tuple(fictt.rewards) == (*tuple(fic0.rewards), "r_tt")
    term = fictt.rewards["r_tt"]
    assert isinstance(term, RewardTermCfg)
    assert term.func is joint_trackability_cost
    assert term.weight == -1.0
    assert set(term.params) == {"robot_cfg", "k_tt"}
    assert term.params["k_tt"] == 1.0
    robot_cfg = term.params["robot_cfg"]
    assert isinstance(robot_cfg, SceneEntityCfg)
    assert robot_cfg.name == "robot"
    assert tuple(robot_cfg.joint_names) == JOINT_NAMES
    assert robot_cfg.preserve_order is True

    fic0_tree = _canonicalize(fic0)
    fictt_tree = _canonicalize(fictt)
    assert {
        field for field in fic0_tree if fic0_tree[field] != fictt_tree[field]
    } == {"rewards"}
    assert fictt_tree["rewards"] == (
        *fic0_tree["rewards"],
        ("r_tt", _canonicalize(term)),
    )
    fictt_tree["rewards"] = fictt_tree["rewards"][:-1]
    assert fictt_tree == fic0_tree


def test_registered_fic0_and_fictt_train_play_configs_do_not_alias() -> None:
    """Mutating one registered cell must never mutate another study cell."""
    registered = (
        task_registry._REGISTRY[FIC0_TASK].env_cfg,
        task_registry._REGISTRY[FIC0_TASK].play_env_cfg,
        task_registry._REGISTRY[FICTT_TASK].env_cfg,
        task_registry._REGISTRY[FICTT_TASK].play_env_cfg,
    )
    assert len({id(cfg) for cfg in registered}) == 4
    assert len({id(cfg.rewards) for cfg in registered}) == 4
    for name in registered[0].rewards:
        assert len({id(cfg.rewards[name]) for cfg in registered}) == 4
    assert (
        registered[2].rewards["r_tt"]
        is not registered[3].rewards["r_tt"]
    )


def _actuator_signature(cfg) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            type(actuator).__name__,
            actuator.stiffness,
            actuator.damping,
            actuator.effort_limit,
            actuator.armature,
            tuple(actuator.target_names_expr),
        )
        for actuator in cfg.scene.entities["robot"].articulation.actuators
    )


def test_fic0_is_registered_for_train_and_play_with_the_parent_learner() -> None:
    """Dropping the additive registration would make the treatment unlaunchable."""
    assert FIC0_TASK in list_tasks()
    train = load_env_cfg(FIC0_TASK)
    play = load_env_cfg(FIC0_TASK, play=True)
    assert list(train.actions) == ["joint_position"]
    assert list(play.actions) == ["joint_position"]
    assert _canonicalize(load_rl_cfg(FIC0_TASK)) == _canonicalize(
        load_rl_cfg(PARENT_TASK)
    )
    assert (
        load_rl_cfg(FIC0_TASK).algorithm.class_name
        == "src.tasks.hammer.rl.cat_ppo:CatPPO"
    )


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_fic0_action_is_the_exact_banked_absolute_joint_contract(play: bool) -> None:
    """A copied, reordered, relative, or incompletely mapped action breaks G1."""
    contract = load_joint_position_contract(ARTIFACT)
    action = _load_fic0(play=play).actions["joint_position"]

    assert isinstance(action, JointPositionActionCfg)
    assert not isinstance(action, RelativeJointPositionActionCfg)
    assert tuple(action.actuator_names) == JOINT_NAMES == contract.joint_names
    assert action.use_default_offset is True
    assert action.offset == 0.0
    assert action.scale == dict(
        zip(JOINT_NAMES, contract.scale_rad.tolist(), strict=True)
    )
    assert action.clip == {
        name: tuple(bounds)
        for name, bounds in zip(
            JOINT_NAMES, contract.physical_clip_rad.tolist(), strict=True
        )
    }
    assert tuple(action.scale) == JOINT_NAMES
    assert tuple(action.clip) == JOINT_NAMES
    for mapping in (action.scale, action.clip):
        assert all(
            sum(re.fullmatch(pattern, target) is not None for target in JOINT_NAMES)
            == 1
            for pattern in mapping
        )


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_fic_reference_and_row_diagnostic_are_isolated_from_cartesian_tasks(
    play: bool,
) -> None:
    """Only the joint pilot receives the calibrated normalizer and scale-safe diagnostic flag."""
    parent = load_env_cfg(PARENT_TASK, play=play)
    fic0 = _load_fic0(play=play)
    fictt = _load_fictt(play=play)

    parent_delivered = parent.rewards["delivered_impulse"]
    assert parent_delivered.params["i_ref"] == 0.3088
    assert parent_delivered.weight == 4.0
    assert parent.metrics["substep_impulse_rows"].params["enabled"] is True

    for task_id in list_tasks():
        if task_id in JOINT_POLICY_TASKS:
            continue
        cartesian = load_env_cfg(task_id, play=play)
        delivered = cartesian.rewards.get("delivered_impulse")
        if delivered is None or delivered.func.__name__ != "FirstStrikeDeliveredRewardTerm":
            continue
        assert delivered.params["i_ref"] == 0.3088

    for fic in (fic0, fictt):
        delivered = fic.rewards["delivered_impulse"]
        assert delivered.params["i_ref"] == FIC_CONTROLLED_DROP_I_REF_N_S
        assert delivered.weight == 4.0
        assert fic.metrics["substep_impulse_rows"].params["enabled"] is False


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_fic0_differs_from_the_exact_cartesian_parent_only_at_approved_seams(
    play: bool,
) -> None:
    """Any change beyond action, calibrated normalizer, and row diagnostics confounds FIC-0."""
    parent = load_env_cfg(PARENT_TASK, play=play)
    fic0 = _load_fic0(play=play)
    parent_tree = _canonicalize(parent)
    fic0_tree = _canonicalize(fic0)

    assert set(parent_tree) == set(fic0_tree)
    assert {
        field for field in parent_tree if parent_tree[field] != fic0_tree[field]
    } == {"actions", "metrics", "rewards"}

    parent.actions = fic0.actions
    parent.rewards["delivered_impulse"].params["i_ref"] = (
        FIC_CONTROLLED_DROP_I_REF_N_S
    )
    parent.metrics["substep_impulse_rows"].params["enabled"] = False
    assert _canonicalize(parent) == fic0_tree


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_fic0_retains_the_p_v_d4_scientific_identity(play: bool) -> None:
    """Reward, constraint, reset, timing, and plant drift would invalidate FIC-0."""
    cfg = _load_fic0(play=play)
    assert list(cfg.actions) == ["joint_position"]
    assert not any(
        isinstance(action, (DifferentialIKActionCfg, RelativeJointPositionActionCfg))
        for action in cfg.actions.values()
    )
    assert "set_gains" not in cfg.actions
    for section in (
        cfg.events,
        cfg.metrics,
        cfg.rewards,
        cfg.terminations,
        cfg.curriculum,
    ):
        for term in section.values():
            func = getattr(term, "func", None)
            if callable(func):
                assert "set_gains" not in func.__qualname__

    assert {name: term.weight for name, term in cfg.rewards.items()} == {
        "approach": 0.1,
        "nail_driven": 0.5,
        "nail_depth_delta": 600.0,
        "impact_progress": 8.0,
        "completion": 100.0,
        "action_rate": -0.01,
        "joint_pos_limits": -10.0,
        "delivered_impulse": 4.0,
        "r_waypoint_progress": 8.0,
    }
    assert "r_imit" not in cfg.rewards
    assert "r_gate" not in cfg.rewards

    expected_terms = (
        "joint_pos",
        "joint_vel",
        "ee_pos",
        "ee_vel",
        "head_pos",
        "head_vel",
        "nail_top_pos",
        "nail_depth",
        "strike_phase",
        "strike_ref_error",
        "actions",
        "next_gate_vector",
        "completed_gate_fraction",
        "guideline_perpendicular_error",
        "waypoint_progress_state",
    )
    for group in ("actor", "critic"):
        assert tuple(cfg.observations[group].terms) == expected_terms
    p_widths = tuple(
        cfg.observations["actor"].terms[name].params["width"]
        for name in (
            "next_gate_vector",
            "completed_gate_fraction",
            "guideline_perpendicular_error",
            "waypoint_progress_state",
        )
    )
    assert p_widths == (3, 1, 1, 2)
    assert sum(p_widths) == 7

    cat = cfg.metrics["cat_soft"].params
    assert cat["use_vel"] is True
    assert cat["vel_detection"] == "substep"
    assert cat["limit"] == pytest.approx(3.1415)
    assert cat["max_p"] == pytest.approx(0.5)
    assert cat["min_p"] == pytest.approx(0.0)
    assert cat["tau"] == pytest.approx(0.95)
    assert cat["use_impulse"] is True
    assert tuple(cat["imp_limit"]) == (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
    assert cat["imp_max_p"] == 0.0
    assert isinstance(cat["imp_max_p"], float)
    assert "substep_peak_qv" in cfg.metrics
    assert "vel_hard" not in cfg.terminations
    assert "cat_vel" not in cfg.terminations

    reset = cfg.events["reset_robot_joints"].params
    assert reset["position_range"] == (0.0, 0.0)
    assert reset["velocity_range"] == (0.0, 0.0)
    assert cfg.sim.mujoco.timestep == pytest.approx(0.002)
    assert cfg.decimation == 10
    assert _actuator_signature(cfg) == (
        (
            "BuiltinPositionActuatorCfg",
            1000.0,
            100.0,
            30.0,
            0.01,
            ("joint1", "joint3", "joint4", "joint5", "joint6"),
        ),
        ("BuiltinPositionActuatorCfg", 1500.0, 150.0, 60.0, 0.02, ("joint2",)),
        (
            "BuiltinPositionActuatorCfg",
            100.0,
            20.0,
            30.0,
            0.005,
            ("jointGripper",),
        ),
    )


@pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
def test_cartesian_parent_keeps_its_checkpoint_contract(play: bool) -> None:
    """Adding FIC-0 must not retrofit any Cartesian registration."""
    parent = load_env_cfg(PARENT_TASK, play=play)
    assert list(parent.actions) == ["ik_hammer_head"]
    assert isinstance(parent.actions["ik_hammer_head"], DifferentialIKActionCfg)
    assert tuple(parent.observations["actor"].terms) == tuple(
        parent.observations["critic"].terms
    )
