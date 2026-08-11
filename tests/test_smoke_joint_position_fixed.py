"""Live metadata contract for the fixed-impedance joint-policy arms."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.tasks.registry import load_env_cfg

from src.tasks.hammer.config.z1.joint_position_contract import (
    JOINT_NAMES,
    load_joint_position_contract,
)
from src.tasks.hammer.rl.runner import _get_hammer_metadata


PARENT_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    "CProgress-Vel-Delivered4"
)
FIC0_TASK = f"{PARENT_TASK}-JointPosition-Fixed"
FICTT_TASK = f"{FIC0_TASK}-TT"
RAW_POLICY_CLIP = 1.0
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json"
)


@pytest.fixture(scope="module")
def metadata_envs():
    """Construct the Cartesian parent and both fixed-impedance live environments."""
    import src.tasks  # noqa: F401  # populate the isolated task registry

    envs = {}
    try:
        for task_id in (PARENT_TASK, FIC0_TASK, FICTT_TASK):
            cfg = load_env_cfg(task_id, play=True)
            cfg.scene.num_envs = 1
            envs[task_id] = ManagerBasedRlEnv(cfg, device="cpu")
        yield envs
    finally:
        for env in envs.values():
            env.close()


def _row(value: torch.Tensor | float) -> list[float]:
    if isinstance(value, torch.Tensor):
        return value[0].detach().cpu().tolist()
    return [float(value)] * len(JOINT_NAMES)


def _clip_rows(value: torch.Tensor) -> list[list[float]]:
    return value[0].detach().cpu().tolist()


def _fixed_actuator_signature(env) -> list[dict[str, object]]:
    actuators = env.cfg.scene.entities["robot"].articulation.actuators
    return [
        {
            "type": type(actuator).__name__,
            "target_names_expr": list(actuator.target_names_expr),
            "stiffness": float(actuator.stiffness),
            "damping": float(actuator.damping),
            "effort_limit": float(actuator.effort_limit),
            "armature": float(actuator.armature),
        }
        for actuator in actuators
    ]


def _make_env(cfg) -> ManagerBasedRlEnv:
    cfg.scene.num_envs = 1
    return ManagerBasedRlEnv(cfg, device="cpu")


def test_cartesian_metadata_is_preserved_byte_for_byte(metadata_envs) -> None:
    """Joint dispatch must not alter the established Cartesian export payload."""
    env = metadata_envs[PARENT_TASK]
    action = env.action_manager.get_term("ik_hammer_head")
    robot = env.scene["robot"]

    assert _get_hammer_metadata(
        env, "test-run", raw_policy_clip=RAW_POLICY_CLIP
    ) == {
        "run_path": "test-run",
        "action_type": "ik_delta_pos",
        "frame_name": action.cfg.frame_name,
        "delta_pos_scale": action.cfg.delta_pos_scale,
        "joint_names": list(robot.joint_names),
        "observation_names": env.observation_manager.active_terms["actor"],
    }


@pytest.mark.parametrize(
    ("task_id", "r_tt_enabled", "r_tt_k_tt"),
    ((FIC0_TASK, False, "not_applicable"), (FICTT_TASK, True, 1.0)),
    ids=("fic0", "fictt"),
)
def test_joint_metadata_is_resolved_from_the_live_action_and_robot(
    metadata_envs, task_id: str, r_tt_enabled: bool, r_tt_k_tt: float | str
) -> None:
    """A stale config, payload hash, or fixed-plant signature must not reach ONNX."""
    env = metadata_envs[task_id]
    action = env.action_manager.get_term("joint_position")
    contract = load_joint_position_contract(_CONTRACT_PATH)

    metadata = _get_hammer_metadata(
        env, "test-run", raw_policy_clip=RAW_POLICY_CLIP
    )

    assert metadata == {
        "run_path": "test-run",
        "action_type": "joint_position",
        "action_term": "joint_position",
        "action_dim": 6,
        "target_names": list(action.target_names),
        "target_ids": action.target_ids.detach().cpu().tolist(),
        "actuator_names": list(action.cfg.actuator_names),
        "use_default_offset": True,
        "default_offsets": _row(action.offset),
        "action_scale": _row(action.scale),
        "physical_clips": _clip_rows(action._clip),
        "raw_policy_clip": RAW_POLICY_CLIP,
        "physics_dt_s": float(env.cfg.sim.mujoco.timestep),
        "control_decimation": int(env.cfg.decimation),
        "fixed_actuator_signature": _fixed_actuator_signature(env),
        "joint_action_qualification_payload_sha256": contract.payload_sha256,
        "r_tt_enabled": r_tt_enabled,
        "r_tt_k_tt": r_tt_k_tt,
    }
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize(
    ("variant", "match"),
    (("unknown", "unsupported action"), ("multiple", "exactly one")),
    ids=("unknown", "multiple"),
)
def test_metadata_rejects_unknown_or_multiple_live_actions(
    variant: str, match: str
) -> None:
    """Export must fail closed rather than guess an action contract."""
    import src.tasks  # noqa: F401  # populate the isolated task registry

    cfg = load_env_cfg(FIC0_TASK, play=True)
    action = cfg.actions["joint_position"]
    if variant == "unknown":
        cfg.actions = {
            "joint_velocity": JointVelocityActionCfg(
                entity_name="robot",
                actuator_names=action.actuator_names,
                scale=action.scale,
                clip=action.clip,
                use_default_offset=True,
                preserve_order=True,
            )
        }
    else:
        cfg.actions = {
            "joint_position": action,
            "joint_position_duplicate": copy.deepcopy(action),
        }
    env = _make_env(cfg)
    try:
        with pytest.raises(ValueError, match=match):
            _get_hammer_metadata(env, "test-run", raw_policy_clip=RAW_POLICY_CLIP)
    finally:
        env.close()


def test_metadata_rejects_wrong_live_joint_order() -> None:
    """A reordered action would make policy columns target the wrong joints."""
    import src.tasks  # noqa: F401  # populate the isolated task registry

    cfg = load_env_cfg(FIC0_TASK, play=True)
    cfg.actions["joint_position"].actuator_names = tuple(reversed(JOINT_NAMES))
    env = _make_env(cfg)
    try:
        with pytest.raises(ValueError, match="order"):
            _get_hammer_metadata(env, "test-run", raw_policy_clip=RAW_POLICY_CLIP)
    finally:
        env.close()


def test_metadata_rejects_nonfinite_live_joint_values() -> None:
    """NaN action mappings are invalid ONNX metadata, not a warning-only export failure."""
    import src.tasks  # noqa: F401  # populate the isolated task registry

    cfg = load_env_cfg(FIC0_TASK, play=True)
    cfg.actions["joint_position"].scale = dict(
        cfg.actions["joint_position"].scale, joint1=float("nan")
    )
    env = _make_env(cfg)
    try:
        with pytest.raises(ValueError, match="finite"):
            _get_hammer_metadata(env, "test-run", raw_policy_clip=RAW_POLICY_CLIP)
    finally:
        env.close()
