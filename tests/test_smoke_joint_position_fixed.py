"""Live metadata contract for the fixed-impedance joint-policy arms."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
from typing import NoReturn

import onnx
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.hammer.config.z1.joint_position_contract import (
    JOINT_NAMES,
    load_joint_position_contract,
)
from src.tasks.hammer.rl.runner import _get_hammer_metadata
import scripts.smoke_cat_soft as smoke_cat_soft
import scripts.smoke_joint_position_fixed as smoke_joint_position_fixed
from scripts.smoke_joint_position_fixed import run_checks


pytestmark = pytest.mark.integration


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


@pytest.mark.parametrize(
    ("kwargs", "invalid_name"),
    (
        ({"num_envs": True}, "num_envs"),
        ({"num_envs": 1.5}, "num_envs"),
        ({"num_envs": 0}, "num_envs"),
        ({"num_envs": -1}, "num_envs"),
        ({"steps": True}, "steps"),
        ({"steps": 1.5}, "steps"),
        ({"steps": 0}, "steps"),
        ({"steps": -1}, "steps"),
    ),
)
def test_live_smoke_rejects_invalid_sizes_before_environment_creation(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    invalid_name: str,
) -> None:
    """The importable live gate must fail malformed public inputs before MuJoCo setup."""

    def unexpected_environment_creation(*args, **kwargs):
        del args, kwargs
        pytest.fail("environment was created before public inputs were validated")

    monkeypatch.setattr(
        smoke_joint_position_fixed,
        "ManagerBasedRlEnv",
        unexpected_environment_creation,
    )

    with pytest.raises(ValueError, match=invalid_name):
        run_checks(task=FIC0_TASK, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "invalid_name"),
    (
        ({"num_envs": True}, "num_envs"),
        ({"num_envs": 1.5}, "num_envs"),
        ({"num_envs": 0}, "num_envs"),
        ({"num_envs": -1}, "num_envs"),
        ({"iters": True}, "iters"),
        ({"iters": 1.5}, "iters"),
        ({"iters": 0}, "iters"),
        ({"iters": -1}, "iters"),
    ),
)
def test_catppo_smoke_rejects_invalid_sizes_before_environment_creation(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    invalid_name: str,
) -> None:
    """The importable train gate must reject malformed sizes before MuJoCo setup."""

    def unexpected_environment_creation(*args, **kwargs):
        del args, kwargs
        pytest.fail("environment was created before public inputs were validated")

    monkeypatch.setattr(
        smoke_cat_soft,
        "ManagerBasedRlEnv",
        unexpected_environment_creation,
    )

    with pytest.raises(ValueError, match=invalid_name):
        smoke_cat_soft.run_smoke(task=FIC0_TASK, **kwargs)


@pytest.mark.parametrize("task_id", (FIC0_TASK, FICTT_TASK), ids=("fic0", "fictt"))
def test_live_joint_position_smoke_passes_every_check(task_id: str) -> None:
    """Both registered fixed-impedance arms must pass the live manager gate."""
    results = run_checks(task=task_id, device="cpu", num_envs=4, steps=2)

    assert results
    assert all(passed for _, passed, _ in results), results


def test_live_smoke_uses_the_registered_raw_policy_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed registered clip must reach metadata validation instead of a literal."""
    drifted_rl_cfg = copy.deepcopy(load_rl_cfg(FIC0_TASK))
    drifted_rl_cfg.clip_actions = 0.5
    monkeypatch.setattr(
        smoke_joint_position_fixed,
        "load_rl_cfg",
        lambda task: drifted_rl_cfg,
    )

    with pytest.raises(ValueError, match="raw policy clip 1.0"):
        run_checks(task=FIC0_TASK, device="cpu", num_envs=1, steps=1)


def test_live_smoke_rejects_nonfinite_output_from_an_earlier_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later healthy manager buffer must not hide an earlier corrupt transition."""
    original_step = ManagerBasedRlEnv.step
    call_count = 0

    def step_with_early_nonfinite(self, action):
        nonlocal call_count
        transition = original_step(self, action)
        call_count += 1
        if call_count != 1:
            return transition

        observations, reward, terminated, truncated, extras = transition
        poisoned_observations = dict(observations)
        poisoned_observations["actor"] = observations["actor"].clone()
        poisoned_observations["actor"][0, 0] = float("nan")
        poisoned_reward = reward.clone()
        poisoned_reward[0] = float("nan")
        poisoned_terminated = terminated.float()
        poisoned_terminated[0] = float("nan")
        poisoned_truncated = truncated.float()
        poisoned_truncated[0] = float("nan")
        return (
            poisoned_observations,
            poisoned_reward,
            poisoned_terminated,
            poisoned_truncated,
            extras,
        )

    monkeypatch.setattr(ManagerBasedRlEnv, "step", step_with_early_nonfinite)

    results = run_checks(task=FIC0_TASK, device="cpu", num_envs=1, steps=2)
    passed_by_name = {name: passed for name, passed, _ in results}

    assert call_count == 2
    assert passed_by_name["reset and stepped observations are finite (N,47)"] is False
    assert passed_by_name["step rewards and done flags are finite"] is False


@pytest.mark.parametrize("failure_stage", ("wrapper", "runner"))
def test_catppo_smoke_closes_raw_env_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Setup failures after real environment creation must not leak the raw environment."""
    closed_envs: list[ManagerBasedRlEnv] = []
    original_close = smoke_cat_soft.ManagerBasedRlEnv.close

    def recording_close(env: ManagerBasedRlEnv) -> None:
        closed_envs.append(env)
        original_close(env)

    def fail_construction(*args, **kwargs) -> NoReturn:
        del args, kwargs
        raise RuntimeError(f"injected {failure_stage} construction failure")

    monkeypatch.setattr(smoke_cat_soft.ManagerBasedRlEnv, "close", recording_close)
    if failure_stage == "wrapper":
        monkeypatch.setattr(smoke_cat_soft, "RslRlVecEnvWrapper", fail_construction)
    else:
        monkeypatch.setattr(smoke_cat_soft, "load_runner_cls", lambda task: fail_construction)

    with pytest.raises(RuntimeError, match=f"injected {failure_stage}"):
        smoke_cat_soft.run_smoke(
            task=FIC0_TASK,
            device="cpu",
            num_envs=1,
            iters=1,
        )

    assert len(closed_envs) == 1


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


@pytest.fixture(scope="module")
def joint_runner(tmp_path_factory):
    """Create the registered FIC-0 runner through its real wrapper/logger path."""
    import src.tasks  # noqa: F401  # populate the isolated task registry

    cfg = load_env_cfg(FIC0_TASK, play=True)
    env = _make_env(cfg)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=RAW_POLICY_CLIP)
    runner_cfg = asdict(load_rl_cfg(FIC0_TASK))
    runner_cfg["logger"] = "tensorboard"
    runner_cfg["upload_model"] = False
    run_root = tmp_path_factory.mktemp("joint-policy-save")
    runner_cls = load_runner_cls(FIC0_TASK)
    assert runner_cls is not None
    runner = runner_cls(
        wrapper,
        runner_cfg,
        log_dir=str(run_root / "logs"),
        device="cpu",
    )
    runner.logger.init_logging_writer()
    try:
        yield runner, run_root
    finally:
        runner.logger.stop_logging_writer()
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


def test_live_smoke_actuator_signature_includes_literal_armature(metadata_envs) -> None:
    """The live fixed-plant gate must pin rotor inertia as well as PD and effort."""
    env = metadata_envs[FIC0_TASK]

    assert smoke_joint_position_fixed._actuator_signature(env) == (
        (
            "BuiltinPositionActuatorCfg",
            ("joint1", "joint3", "joint4", "joint5", "joint6"),
            1000.0,
            100.0,
            30.0,
            0.01,
        ),
        ("BuiltinPositionActuatorCfg", ("joint2",), 1500.0, 150.0, 60.0, 0.02),
        ("BuiltinPositionActuatorCfg", ("jointGripper",), 100.0, 20.0, 30.0, 0.005),
    )


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


def test_metadata_rejects_cartesian_action_registered_as_joint_position() -> None:
    """Action class and manager-term identity must agree before export."""
    import src.tasks  # noqa: F401  # populate the isolated task registry

    cfg = load_env_cfg(PARENT_TASK, play=True)
    cfg.actions = {"joint_position": cfg.actions["ik_hammer_head"]}
    env = _make_env(cfg)
    try:
        with pytest.raises(ValueError, match="action term"):
            _get_hammer_metadata(env, "test-run", raw_policy_clip=RAW_POLICY_CLIP)
    finally:
        env.close()


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
        "delivered_impulse_i_ref_n_s": 0.2799950838088989,
        "r_tt_enabled": r_tt_enabled,
        "r_tt_k_tt": r_tt_k_tt,
    }
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("invalid_timestep", (float("nan"), 0.001))
def test_metadata_rejects_noncanonical_live_physics_timestep(
    metadata_envs, invalid_timestep: float
) -> None:
    """Joint-policy metadata is valid only for the qualified 500 Hz plant."""
    env = metadata_envs[FIC0_TASK]
    original = env.cfg.sim.mujoco.timestep
    env.cfg.sim.mujoco.timestep = invalid_timestep
    try:
        with pytest.raises(ValueError, match="physics timestep"):
            _get_hammer_metadata(env, "test-run", raw_policy_clip=RAW_POLICY_CLIP)
    finally:
        env.cfg.sim.mujoco.timestep = original


@pytest.mark.parametrize(
    "invalid_decimation", (float("nan"), 10.5, 5, True)
)
def test_metadata_rejects_invalid_or_noncanonical_live_control_decimation(
    metadata_envs, invalid_decimation: float | bool
) -> None:
    """No coercion may turn malformed timing into the qualified 10 substeps."""
    env = metadata_envs[FIC0_TASK]
    original = env.cfg.decimation
    env.cfg.decimation = invalid_decimation
    try:
        with pytest.raises(ValueError, match="control decimation"):
            _get_hammer_metadata(env, "test-run", raw_policy_clip=RAW_POLICY_CLIP)
    finally:
        env.cfg.decimation = original


@pytest.mark.parametrize("invalid_clip", (float("nan"), 0.5, True))
def test_metadata_rejects_noncanonical_wrapper_action_clip(
    metadata_envs, invalid_clip: float | bool
) -> None:
    """The exported policy contract is qualified only for raw clip 1.0."""
    env = metadata_envs[FIC0_TASK]
    with pytest.raises(ValueError, match="raw policy clip 1.0"):
        _get_hammer_metadata(env, "test-run", raw_policy_clip=invalid_clip)


def test_save_attaches_joint_metadata_with_the_wrapper_owned_clip(joint_runner) -> None:
    """The public save path must write the live wrapper contract into ONNX."""
    runner, run_root = joint_runner
    export_dir = run_root / "success"
    export_dir.mkdir()
    checkpoint = export_dir / "model_0.pt"

    runner.save(str(checkpoint))

    onnx_path = export_dir / "success.onnx"
    metadata = {
        entry.key: entry.value
        for entry in onnx.load(onnx_path).metadata_props
    }
    live_metadata = _get_hammer_metadata(
        runner.env.unwrapped,
        "local",
        raw_policy_clip=runner.env.clip_actions,
    )
    assert checkpoint.is_file()
    assert metadata["action_type"] == "joint_position"
    assert metadata["action_term"] == "joint_position"
    assert metadata["raw_policy_clip"] == "1.0"
    assert metadata["physics_dt_s"] == "0.002"
    assert metadata["control_decimation"] == "10"
    assert metadata["delivered_impulse_i_ref_n_s"] == "0.2799950838088989"
    for key, live_value in live_metadata.items():
        if isinstance(live_value, (list, dict)):
            assert json.loads(metadata[key]) == live_value


def test_save_propagates_metadata_contract_errors_before_onnx_export(
    joint_runner,
) -> None:
    """Metadata drift must escape the warning-only ONNX serialization boundary."""
    runner, run_root = joint_runner
    export_dir = run_root / "invalid-clip"
    export_dir.mkdir()
    checkpoint = export_dir / "model_0.pt"
    original_clip = runner.env.clip_actions
    runner.env.clip_actions = 0.5
    try:
        with pytest.raises(ValueError, match="raw policy clip 1.0"):
            runner.save(str(checkpoint))
    finally:
        runner.env.clip_actions = original_clip

    assert checkpoint.is_file()
    assert not (export_dir / "invalid-clip.onnx").exists()


@pytest.mark.parametrize(
    "invalid_i_ref", (float("nan"), 0.0, -1.0, True, "not-a-number")
)
def test_joint_metadata_rejects_invalid_live_delivered_impulse_reference(
    metadata_envs, invalid_i_ref: object
) -> None:
    """Joint policy exports must fail closed on malformed live normalization."""
    env = metadata_envs[FIC0_TASK]
    term = env.reward_manager.get_term_cfg("delivered_impulse")
    original = term.params["i_ref"]
    term.params["i_ref"] = invalid_i_ref
    try:
        with pytest.raises(ValueError, match="delivered impulse i_ref"):
            _get_hammer_metadata(env, "test-run", raw_policy_clip=RAW_POLICY_CLIP)
    finally:
        term.params["i_ref"] = original


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
