"""Evaluate one matched fixed-impedance FIC pilot checkpoint.

This deliberately small evaluator is separate from the historical Cartesian
campaign evaluator: it admits only the two joint-position treatments and
records the first completed episode from each member of one fixed 64-world
population.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.curriculums import reward_curriculum
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from src.tasks.hammer.cat.hook import CatSoftHook, _NEG_TERMS
from src.tasks.hammer.config.z1.joint_position_contract import (
    JOINT_NAMES,
    load_joint_position_contract,
)
from src.tasks.hammer.mdp.first_strike import (
    REASON_SUCCESS,
    REASON_WINDOW,
    _ENV_FIRST_STRIKE_ATTR,
)
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.references import get_strike_reference
from src.tasks.hammer.mdp.rewards import ImitationPriorTerm
from src.tasks.hammer.mdp.trackability import joint_trackability_cost
from src.tasks.hammer.mdp.velocity_bound import _ENV_SUBSTEP_ATTR
from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH


PARENT_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4"
)
FIC0_TASK = f"{PARENT_TASK}-JointPosition-Fixed"
FICTT_TASK = f"{FIC0_TASK}-TT"
TASKS = (FIC0_TASK, FICTT_TASK)

OBSERVATION_TERMS = (
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
)
OBSERVATION_WIDTH = 40
IMITATION_CURRICULUM = (
    (0, 0.10),
    (1200, 0.08),
    (2400, 0.06),
    (3600, 0.04),
    (4800, 0.02),
    (6000, 0.00),
)

SEED = 2026081202
NUM_ENVS = 64
EPISODE_LENGTH_S = 4.0
I_REF_N_S = 0.2799950838088989
JOINT_VELOCITY_LIMIT_RAD_S = 3.1415
JOINT_IMPULSE_CAP_N_M_S = (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_JOINT_POSITION_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json"
)
_FIXED_ACTUATOR_SIGNATURE = (
    (
        "BuiltinPositionActuatorCfg",
        "TransmissionType",
        TransmissionType.JOINT,
        1000.0,
        100.0,
        30.0,
        0.01,
        None,
        None,
        0,
        0,
        0.0,
        0,
        True,
        ("joint1", "joint3", "joint4", "joint5", "joint6"),
    ),
    (
        "BuiltinPositionActuatorCfg",
        "TransmissionType",
        TransmissionType.JOINT,
        1500.0,
        150.0,
        60.0,
        0.02,
        None,
        None,
        0,
        0,
        0.0,
        0,
        True,
        ("joint2",),
    ),
    (
        "BuiltinPositionActuatorCfg",
        "TransmissionType",
        TransmissionType.JOINT,
        100.0,
        20.0,
        30.0,
        0.005,
        None,
        None,
        0,
        0,
        0.0,
        0,
        True,
        ("jointGripper",),
    ),
)


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(path: Path) -> dict[str, str]:
    revision = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if _REVISION_RE.fullmatch(revision) is None or status:
        raise RuntimeError(f"repository must be clean at a full commit: {path}")
    return {"revision": revision, "status": status}


def _cfg_value(mapping: Mapping[str, Any], name: str) -> Any:
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"missing required {name}") from exc


def _fixed_actuator_signature(env_cfg) -> tuple[tuple[object, ...], ...]:
    try:
        actuators = env_cfg.scene.entities["robot"].articulation.actuators
    except (AttributeError, KeyError, TypeError):
        return ()
    return tuple(
        (
            type(actuator).__name__,
            type(getattr(actuator, "transmission_type", None)).__name__,
            getattr(actuator, "transmission_type", None),
            getattr(actuator, "stiffness", None),
            getattr(actuator, "damping", None),
            getattr(actuator, "effort_limit", None),
            getattr(actuator, "armature", None),
            getattr(actuator, "frictionloss", None),
            getattr(actuator, "viscous_damping", None),
            getattr(actuator, "delay_min_lag", None),
            getattr(actuator, "delay_max_lag", None),
            getattr(actuator, "delay_hold_prob", None),
            getattr(actuator, "delay_update_period", None),
            getattr(actuator, "delay_per_env_phase", None),
            tuple(getattr(actuator, "target_names_expr", ())),
        )
        for actuator in actuators
    )


def validate_fic_contract(task: str, env_cfg, agent_cfg) -> dict[str, object]:
    """Fail closed unless a loaded config is exactly one calibrated FIC arm."""
    if task not in TASKS:
        raise ValueError(f"unsupported FIC pilot task: {task}")
    if _finite_number(getattr(agent_cfg, "clip_actions", None), name="clip_actions") != 1.0:
        raise ValueError("FIC pilot requires raw policy clip 1.0")
    if getattr(getattr(agent_cfg, "actor", None), "obs_normalization", None) is not True:
        raise ValueError("FIC pilot requires actor observation normalization")
    if getattr(getattr(agent_cfg, "critic", None), "obs_normalization", None) is not True:
        raise ValueError("FIC pilot requires critic observation normalization")

    observations = getattr(env_cfg, "observations", {})
    if tuple(observations) != ("actor", "critic") or any(
        tuple(getattr(observations[group], "terms", ())) != OBSERVATION_TERMS
        for group in ("actor", "critic")
    ):
        raise ValueError("FIC pilot direct-reference observation contract drift")

    actions = getattr(env_cfg, "actions", {})
    if tuple(actions) != ("joint_position",):
        raise ValueError("FIC pilot requires exactly the joint_position action")
    action = actions["joint_position"]
    qualified = load_joint_position_contract(_JOINT_POSITION_CONTRACT_PATH)
    qualified_scale = dict(
        zip(JOINT_NAMES, qualified.scale_rad.tolist(), strict=True)
    )
    qualified_clip = {
        name: tuple(bounds)
        for name, bounds in zip(
            JOINT_NAMES, qualified.physical_clip_rad.tolist(), strict=True
        )
    }
    action_offset = getattr(action, "offset", None)
    if (
        type(action) is not JointPositionActionCfg
        or getattr(action, "transmission_type", None) is not TransmissionType.JOINT
        or getattr(action, "entity_name", None) != "robot"
        or tuple(getattr(action, "actuator_names", ())) != JOINT_NAMES
        or tuple(getattr(action, "scale", ())) != JOINT_NAMES
        or getattr(action, "scale", None) != qualified_scale
        or tuple(getattr(action, "clip", ())) != JOINT_NAMES
        or getattr(action, "clip", None) != qualified_clip
        or getattr(action, "use_default_offset", None) is not True
        or getattr(action, "preserve_order", None) is not True
        or isinstance(action_offset, bool)
        or action_offset != 0.0
    ):
        raise ValueError("FIC pilot joint action contract drift")
    physics_dt = _finite_number(
        getattr(getattr(getattr(env_cfg, "sim", None), "mujoco", None), "timestep", None),
        name="physics timestep",
    )
    decimation = getattr(env_cfg, "decimation", None)
    if (
        physics_dt != qualified.physics_dt_s
        or type(decimation) is not int
        or decimation != qualified.control_decimation
    ):
        raise ValueError("FIC pilot control timing drift")
    if _fixed_actuator_signature(env_cfg) != _FIXED_ACTUATOR_SIGNATURE:
        raise ValueError("FIC pilot fixed actuator signature drift")

    reset = getattr(env_cfg, "events", {}).get("reset_robot_joints")
    reset_params = getattr(reset, "params", {}) if reset is not None else {}
    if (
        reset is None
        or reset_params.get("position_range") != (0.0, 0.0)
        or reset_params.get("velocity_range") != (0.0, 0.0)
    ):
        raise ValueError("FIC pilot fixed joint reset drift")

    rewards = getattr(env_cfg, "rewards", {})
    expected_rewards = (
        "approach",
        "nail_driven",
        "nail_depth_delta",
        "impact_progress",
        "completion",
        "action_rate",
        "joint_pos_limits",
        "r_imit",
        "delivered_impulse",
        *(("r_tt",) if task == FICTT_TASK else ()),
    )
    if tuple(rewards) != expected_rewards:
        raise ValueError("FIC pilot direct-reference reward or waypoint behavior drift")
    delivered = _cfg_value(rewards, "delivered_impulse")
    if _finite_number(getattr(delivered, "weight", None), name="D4 weight") != 4.0:
        raise ValueError("FIC pilot requires D4 weight 4.0")
    i_ref = _finite_number(
        getattr(delivered, "params", {}).get("i_ref"), name="delivered i_ref"
    )
    if i_ref != I_REF_N_S:
        raise ValueError(f"FIC pilot delivered i_ref drift: {i_ref}")

    imitation = _cfg_value(rewards, "r_imit")
    imitation_params = getattr(imitation, "params", {})
    robot_cfg = imitation_params.get("robot_cfg")
    nail_cfg = imitation_params.get("nail_cfg")
    if (
        getattr(imitation, "func", None) is not ImitationPriorTerm
        or _finite_number(getattr(imitation, "weight", None), name="r_imit weight") != 0.10
        or set(imitation_params) != {"sensor_name", "robot_cfg", "nail_cfg", "sigma"}
        or imitation_params.get("sensor_name") != "hammer_nail_contact"
        or _finite_number(imitation_params.get("sigma"), name="r_imit sigma") != 0.05
        or getattr(robot_cfg, "name", None) != "robot"
        or tuple(getattr(robot_cfg, "site_names", ())) != ("hammer_head_site",)
        or getattr(nail_cfg, "name", None) != "nail_block"
        or tuple(getattr(nail_cfg, "site_names", ())) != ("nail_top",)
    ):
        raise ValueError("FIC pilot r_imit contract drift")

    curriculum = getattr(env_cfg, "curriculum", {})
    anneal = curriculum.get("r_imit_anneal") if isinstance(curriculum, Mapping) else None
    anneal_params = getattr(anneal, "params", {}) if anneal is not None else {}
    stages = anneal_params.get("stages")
    live_stages = None
    if isinstance(stages, list) and all(
        isinstance(stage, dict)
        and set(stage) == {"step", "weight"}
        and type(stage["step"]) is int
        for stage in stages
    ):
        live_stages = tuple((stage["step"], stage["weight"]) for stage in stages)
    if (
        tuple(curriculum) != ("r_imit_anneal",)
        or anneal is None
        or getattr(anneal, "func", None) is not reward_curriculum
        or set(anneal_params) != {"reward_name", "stages"}
        or anneal_params.get("reward_name") != "r_imit"
        or live_stages != IMITATION_CURRICULUM
    ):
        raise ValueError("FIC pilot r_imit curriculum drift")

    metrics = getattr(env_cfg, "metrics", {})
    if "waypoint_progress" in metrics or {"r_gate", "r_waypoint_progress"} & set(rewards):
        raise ValueError("FIC pilot forbids waypoint behavior")
    cat = _cfg_value(metrics, "cat_soft")
    cat_reduce = getattr(cat, "reduce", None)
    if (
        getattr(cat, "per_substep", None) is not False
        or type(cat_reduce) is not str
        or cat_reduce != "mean"
    ):
        raise ValueError("FIC pilot CaT scheduling drift")
    cat_params = getattr(cat, "params", {})
    cat_robot_cfg = cat_params.get("robot_cfg")
    if (
        getattr(cat, "func", None) is not CatSoftHook
        or set(cat_params)
        != {
            "use_vel",
            "use_impulse",
            "imp_limit",
            "imp_max_p",
            "imp_seed",
            "robot_cfg",
            "limit",
            "max_p",
            "min_p",
            "tau",
            "vel_detection",
        }
        or cat_params.get("use_vel") is not True
        or cat_params.get("use_impulse") is not True
        or getattr(cat_robot_cfg, "name", None) != "robot"
        or tuple(getattr(cat_robot_cfg, "joint_names", ())) != JOINT_NAMES
        or getattr(cat_robot_cfg, "preserve_order", None) is not False
        or cat_params.get("vel_detection") != "substep"
        or _finite_number(cat_params.get("limit"), name="velocity limit") != 3.1415
        or _finite_number(cat_params.get("max_p"), name="velocity max_p") != 0.5
        or _finite_number(cat_params.get("min_p"), name="CaT min_p") != 0.0
        or _finite_number(cat_params.get("tau"), name="CaT tau") != 0.95
        or tuple(cat_params.get("imp_limit", ()))
        != (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
        or _finite_number(cat_params.get("imp_max_p"), name="imp_max_p") != 0.0
        or _finite_number(cat_params.get("imp_seed"), name="imp_seed") != 0.001
    ):
        raise ValueError("FIC pilot CaT contract drift")
    if (
        getattr(_cfg_value(metrics, "substep_impulse_rows"), "params", {}).get(
            "enabled"
        )
        is not False
    ):
        raise ValueError("FIC pilot requires row-attribution disabled")

    r_tt = rewards.get("r_tt")
    if task == FIC0_TASK:
        if r_tt is not None:
            raise ValueError("FIC-0 must not contain r_tt")
    else:
        rtt_params = getattr(r_tt, "params", {}) if r_tt is not None else {}
        rtt_robot_cfg = rtt_params.get("robot_cfg")
        if (
            r_tt is None
            or getattr(r_tt, "func", None) is not joint_trackability_cost
            or _finite_number(getattr(r_tt, "weight", None), name="r_tt weight") != -1.0
            or set(rtt_params) != {"robot_cfg", "k_tt"}
            or _finite_number(rtt_params.get("k_tt"), name="r_tt k_tt") != 1.0
            or getattr(rtt_robot_cfg, "name", None) != "robot"
            or tuple(getattr(rtt_robot_cfg, "joint_names", ())) != JOINT_NAMES
            or getattr(rtt_robot_cfg, "preserve_order", None) is not True
        ):
            raise ValueError("FIC-TT r_tt contract drift")

    expected_negative_terms = (
        ("action_rate", "joint_pos_limits", "r_tt")
        if task == FICTT_TASK
        else ("action_rate", "joint_pos_limits")
    )
    live_negative_terms = tuple(
        name
        for name, term in rewards.items()
        if _finite_number(getattr(term, "weight", None), name=f"{name} weight") < 0.0
    )
    if (
        _NEG_TERMS != ("action_rate", "joint_pos_limits", "r_tt")
        or live_negative_terms != expected_negative_terms
    ):
        raise ValueError("FIC pilot CaT negative-term split drift")

    return {
        "task": task,
        "action_term": "joint_position",
        "joint_names": list(JOINT_NAMES),
        "observation_names": list(OBSERVATION_TERMS),
        "observation_width": OBSERVATION_WIDTH,
        "actor_obs_normalization": True,
        "critic_obs_normalization": True,
        "d4_weight": 4.0,
        "delivered_impulse_i_ref_n_s": i_ref,
        "r_imit_weight": 0.10,
        "r_imit_sigma_m": 0.05,
        "r_imit_curriculum": [
            {"step": step, "weight": weight}
            for step, weight in IMITATION_CURRICULUM
        ],
        "velocity_cat_substep": True,
        "impulse_cat_log_only": True,
        "row_attribution_enabled": False,
        "r_tt_enabled": task == FICTT_TASK,
        "r_tt_k_tt": 1.0 if task == FICTT_TASK else None,
    }


def _validate_live_observation_contract(env) -> None:
    """Require the constructed policy boundary to retain the qualified width."""
    manager = env.observation_manager
    active_terms = manager.active_terms
    dimensions = manager.group_obs_dim
    if set(active_terms) != {"actor", "critic"} or any(
        tuple(active_terms.get(group, ())) != OBSERVATION_TERMS
        for group in ("actor", "critic")
    ):
        raise RuntimeError("live FIC observation names drift")
    if set(dimensions) != {"actor", "critic"} or any(
        tuple(dimensions.get(group, ())) != (OBSERVATION_WIDTH,)
        for group in ("actor", "critic")
    ):
        raise RuntimeError("live FIC observation width drift")


def _reason(value: object) -> str:
    code = int(value)
    if code == REASON_SUCCESS:
        return "success"
    if code == REASON_WINDOW:
        return "window"
    return "none"


@dataclass
class EpisodeAccumulators:
    qvel_peak: torch.Tensor
    target_sq_sum: torch.Tensor
    target_abs_max: torch.Tensor
    target_samples: torch.Tensor
    reference_sum: torch.Tensor
    reference_max: torch.Tensor
    reference_samples: torch.Tensor


def _new_episode_accumulators(
    num_envs: int, device: torch.device | str
) -> EpisodeAccumulators:
    if type(num_envs) is not int or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    kwargs = {"device": device, "dtype": torch.float32}
    return EpisodeAccumulators(
        qvel_peak=torch.zeros((num_envs, len(JOINT_NAMES)), **kwargs),
        target_sq_sum=torch.zeros(num_envs, **kwargs),
        target_abs_max=torch.zeros(num_envs, **kwargs),
        target_samples=torch.zeros(num_envs, device=device, dtype=torch.long),
        reference_sum=torch.zeros(num_envs, **kwargs),
        reference_max=torch.zeros(num_envs, **kwargs),
        reference_samples=torch.zeros(num_envs, device=device, dtype=torch.long),
    )


def _reset_episode_accumulators(
    accumulators: EpisodeAccumulators, env_ids: torch.Tensor
) -> None:
    if env_ids.ndim != 1 or env_ids.dtype != torch.long:
        raise ValueError("episode accumulator reset ids must be a one-dimensional int64 tensor")
    for value in vars(accumulators).values():
        value[env_ids] = 0


def _accumulate_episode_measurements(
    accumulators: EpisodeAccumulators,
    *,
    qvel_peak: torch.Tensor,
    target_error: torch.Tensor,
    reference_error: torch.Tensor,
    reference_eligible: torch.Tensor,
) -> None:
    num_envs = accumulators.qvel_peak.shape[0]
    if qvel_peak.shape != (num_envs, len(JOINT_NAMES)):
        raise RuntimeError("qvel peak must have shape (num_envs, 6)")
    if target_error.shape != (num_envs, len(JOINT_NAMES)):
        raise RuntimeError("target error must have shape (num_envs, 6)")
    if reference_error.shape != (num_envs,):
        raise RuntimeError("reference error must have shape (num_envs,)")
    if reference_eligible.shape != (num_envs,) or reference_eligible.dtype != torch.bool:
        raise RuntimeError("reference eligibility must be a bool vector over environments")
    inputs = (qvel_peak, target_error, reference_error)
    if any(
        value.device != accumulators.qvel_peak.device
        for value in (*inputs, reference_eligible)
    ):
        raise RuntimeError("episode measurement tensors must share one device")
    if not all(bool(torch.isfinite(value).all()) for value in inputs):
        raise RuntimeError("episode measurements must be finite")

    torch.maximum(accumulators.qvel_peak, qvel_peak, out=accumulators.qvel_peak)
    accumulators.target_sq_sum += target_error.square().sum(dim=1)
    torch.maximum(
        accumulators.target_abs_max,
        target_error.abs().amax(dim=1),
        out=accumulators.target_abs_max,
    )
    accumulators.target_samples += 1
    accumulators.reference_sum[reference_eligible] += reference_error[reference_eligible]
    accumulators.reference_max[reference_eligible] = torch.maximum(
        accumulators.reference_max[reference_eligible],
        reference_error[reference_eligible],
    )
    accumulators.reference_samples[reference_eligible] += 1


def _finalize_episode_measurements(
    accumulators: EpisodeAccumulators, env_id: int, episode_steps: int
) -> dict[str, object]:
    if type(env_id) is not int or not 0 <= env_id < accumulators.qvel_peak.shape[0]:
        raise ValueError("episode accumulator env_id out of range")
    if type(episode_steps) is not int or episode_steps <= 0:
        raise RuntimeError("episode steps must be a positive integer")
    if int(accumulators.target_samples[env_id].item()) != episode_steps:
        raise RuntimeError("target samples must equal episode steps")
    reference_samples = int(accumulators.reference_samples[env_id].item())
    if reference_samples <= 0:
        raise RuntimeError("reference samples must be positive")
    finite_values = (
        accumulators.qvel_peak[env_id],
        accumulators.target_sq_sum[env_id],
        accumulators.target_abs_max[env_id],
        accumulators.reference_sum[env_id],
        accumulators.reference_max[env_id],
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise RuntimeError("final episode measurements must be finite")

    qvel = _finite_six(
        accumulators.qvel_peak[env_id], name="joint velocity peak"
    )
    target_sq_sum = float(accumulators.target_sq_sum[env_id].item())
    reference_sum = float(accumulators.reference_sum[env_id].item())
    return {
        "joint_velocity_peak_rad_s": qvel,
        "joint_velocity_legal": all(
            value <= JOINT_VELOCITY_LIMIT_RAD_S for value in qvel
        ),
        "joint_target_rmse_rad": math.sqrt(
            target_sq_sum / (len(JOINT_NAMES) * episode_steps)
        ),
        "joint_target_error_max_rad": float(
            accumulators.target_abs_max[env_id].item()
        ),
        "reference_error_mean_m": reference_sum / reference_samples,
        "reference_error_max_m": float(accumulators.reference_max[env_id].item()),
        "reference_error_samples": reference_samples,
    }


def _finite_six(value: torch.Tensor, *, name: str = "joint impulse peak") -> list[float]:
    values = [float(item) for item in value.detach().cpu().tolist()]
    if len(values) != len(JOINT_NAMES) or not all(math.isfinite(item) for item in values):
        raise RuntimeError(f"{name} must contain six finite values")
    return values


def capture_first_terminals(
    *,
    records: dict[int, dict[str, object]],
    done: torch.Tensor,
    terminated: torch.Tensor,
    timed_out: torch.Tensor,
    steps: torch.Tensor,
    tracker,
    nail_depth: torch.Tensor,
    joint_impulse_peak: torch.Tensor,
    episode_accumulators: EpisodeAccumulators,
) -> tuple[int, ...]:
    """Copy each first terminal state before any caller-reset can mutate it."""
    ids = torch.nonzero(done, as_tuple=False).flatten().detach().cpu().tolist()
    for env_id in ids:
        if env_id in records:
            continue
        measurements = _finalize_episode_measurements(
            episode_accumulators,
            int(env_id),
            int(steps[env_id].item()),
        )
        impulse_peak = _finite_six(joint_impulse_peak[env_id])
        record = {
            "env_id": int(env_id),
            "success": bool(terminated[env_id]),
            "timeout": bool(timed_out[env_id]),
            "episode_steps": int(steps[env_id].item()),
            "first_strike_started": bool(tracker.started[env_id]),
            "first_strike_finalized": bool(tracker.finalized[env_id]),
            "first_strike_productive": bool(tracker.productive[env_id]),
            "first_strike_reason": _reason(tracker.reason[env_id]),
            "precontact_velocity_m_s": float(tracker.v_precontact[env_id]),
            "first_event_impulse_n_s": float(tracker.delivered[env_id]),
            "nail_depth_m": float(nail_depth[env_id]),
            "joint_impulse_peak_n_m_s": impulse_peak,
            "joint_impulse_utilization": [
                value / cap
                for value, cap in zip(
                    impulse_peak, JOINT_IMPULSE_CAP_N_M_S, strict=True
                )
            ],
            **measurements,
        }
        if not all(
            math.isfinite(float(record[key]))
            for key in ("precontact_velocity_m_s", "first_event_impulse_n_s", "nail_depth_m")
        ):
            raise RuntimeError("terminal first-strike metrics must be finite")
        records[env_id] = record
    return tuple(int(env_id) for env_id in ids)


def reset_done_envs_preserving_unfinished(
    env, observations: TensorDict, done_ids: torch.Tensor
) -> TensorDict:
    """Reset terminal worlds without perturbing unfinished inputs or Torch RNG."""
    device = torch.device(env.device)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    try:
        reset_obs_dict, _ = env.reset(env_ids=done_ids)
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

    reset_observations = TensorDict(reset_obs_dict, batch_size=[env.num_envs])
    merged = observations.clone()
    merged[done_ids] = reset_observations[done_ids]
    return merged


def _population_hash(env) -> str:
    robot = env.scene["robot"].data
    nail = env.scene["nail_block"].data
    payload = {
        "robot_joint_pos": robot.joint_pos.detach().cpu().tolist(),
        "robot_joint_vel": robot.joint_vel.detach().cpu().tolist(),
        "nail_joint_pos": nail.joint_pos.detach().cpu().tolist(),
        "nail_joint_vel": nail.joint_vel.detach().cpu().tolist(),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _publish_new_json(output: Path, payload: Mapping[str, object]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluator output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(payload) + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise RuntimeError("cannot summarize an empty population")
    impulses = np.asarray([row["first_event_impulse_n_s"] for row in records], dtype=float)
    peaks = np.asarray([row["joint_impulse_peak_n_m_s"] for row in records], dtype=float)
    utilization = np.asarray(
        [row["joint_impulse_utilization"] for row in records], dtype=float
    )
    qvel = np.asarray(
        [row["joint_velocity_peak_rad_s"] for row in records], dtype=float
    )
    target_rmse = np.asarray(
        [row["joint_target_rmse_rad"] for row in records], dtype=float
    )
    target_max = np.asarray(
        [row["joint_target_error_max_rad"] for row in records], dtype=float
    )
    reference_mean = np.asarray(
        [row["reference_error_mean_m"] for row in records], dtype=float
    )
    reference_max = np.asarray(
        [row["reference_error_max_m"] for row in records], dtype=float
    )
    vectors = (peaks, utilization, qvel)
    scalars = (impulses, target_rmse, target_max, reference_mean, reference_max)
    if any(value.shape != (len(records), len(JOINT_NAMES)) for value in vectors):
        raise RuntimeError("summary vector endpoints must have six columns")
    if not all(np.isfinite(value).all() for value in (*vectors, *scalars)):
        raise RuntimeError("summary endpoint values must be finite")
    expected_utilization = peaks / np.asarray(JOINT_IMPULSE_CAP_N_M_S, dtype=float)
    if not np.array_equal(utilization, expected_utilization):
        raise RuntimeError("joint impulse utilization must match the qualified caps")
    qvel_legal = np.all(qvel <= JOINT_VELOCITY_LIMIT_RAD_S, axis=1)
    if [bool(value) for value in qvel_legal] != [
        bool(row["joint_velocity_legal"]) for row in records
    ]:
        raise RuntimeError("joint velocity legal flags must match the six-joint peaks")
    impulse_legal = np.all(utilization <= 1.0, axis=1)

    def scalar_summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }

    return {
        "task_success_n": sum(bool(row["success"]) for row in records),
        "task_success_rate": float(np.mean([bool(row["success"]) for row in records])),
        "productive_first_strike_n": sum(
            bool(row["first_strike_productive"]) for row in records
        ),
        "productive_first_strike_rate": float(
            np.mean([bool(row["first_strike_productive"]) for row in records])
        ),
        "first_event_impulse_n_s": {
            "mean": float(np.mean(impulses)),
            "std": float(np.std(impulses)),
            "min": float(np.min(impulses)),
            "max": float(np.max(impulses)),
        },
        "joint_impulse_peak_n_m_s": {
            "p95": np.quantile(peaks, 0.95, axis=0).tolist(),
            "max": np.max(peaks, axis=0).tolist(),
        },
        "joint_impulse_utilization": {
            "p95": np.quantile(utilization, 0.95, axis=0).tolist(),
            "max": np.max(utilization, axis=0).tolist(),
            "all_joints_at_or_below_cap_n": int(np.sum(impulse_legal)),
            "all_joints_at_or_below_cap_rate": float(np.mean(impulse_legal)),
        },
        "joint_velocity_peak_rad_s": {
            "p95": np.quantile(qvel, 0.95, axis=0).tolist(),
            "max": np.max(qvel, axis=0).tolist(),
            "all_joints_legal_n": int(np.sum(qvel_legal)),
            "all_joints_legal_rate": float(np.mean(qvel_legal)),
        },
        "joint_target_rmse_rad": scalar_summary(target_rmse),
        "joint_target_error_max_rad": scalar_summary(target_max),
        "reference_error_mean_m": scalar_summary(reference_mean),
        "reference_error_max_m": scalar_summary(reference_max),
    }


def _ordered_complete_records(records: Mapping[int, dict[str, object]]) -> list[dict[str, object]]:
    """Return the fixed population only when every expected first episode exists."""
    expected = set(range(NUM_ENVS))
    actual = set(records)
    if actual != expected:
        raise RuntimeError(
            f"incomplete first-episode population: {len(actual)}/{NUM_ENVS}"
        )
    return [records[index] for index in range(NUM_ENVS)]


def evaluate_checkpoint(
    task: str,
    checkpoint: Path,
    output: Path,
    device: str = "cpu",
    *,
    training_seed: int,
) -> dict[str, object]:
    """Evaluate a mean policy on the frozen 64-world first-episode population."""
    checkpoint = Path(checkpoint)
    output = Path(output)
    if type(training_seed) is not int or training_seed not in (2, 3, 4):
        raise ValueError("training seed must be one of 2, 3, or 4")
    if checkpoint.name != "model_499.pt":
        raise ValueError("FIC pilot requires final checkpoint basename model_499.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluator output: {output}")

    env_cfg = load_env_cfg(task, play=False)
    agent_cfg = load_rl_cfg(task)
    contract = validate_fic_contract(task, env_cfg, agent_cfg)
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.episode_length_s = EPISODE_LENGTH_S
    env_cfg.auto_reset = False
    env_cfg.seed = SEED
    torch.manual_seed(SEED)

    root = Path(__file__).resolve().parents[2]
    from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

    code_git = _git_identity(root)
    asset_git = _git_identity(Z1_HAMMER_XML.parents[2])
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        _validate_live_observation_contract(env)
        tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR, None)
        impulse_accumulator = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
        qvel_tracker = getattr(env, _ENV_SUBSTEP_ATTR, None)
        if tracker is None or impulse_accumulator is None or qvel_tracker is None:
            raise RuntimeError(
                "FIC pilot requires first-strike, joint-impulse, and substep-qvel instrumentation"
            )
        action_term = env.action_manager.get_term("joint_position")
        r_imit_cfg = env.reward_manager.get_term_cfg("r_imit")
        r_imit_term = r_imit_cfg.func
        robot = env.scene["robot"]
        if not isinstance(r_imit_term, ImitationPriorTerm):
            raise RuntimeError("live r_imit term must be ImitationPriorTerm")
        if tuple(getattr(action_term, "target_names", ())) != JOINT_NAMES:
            raise RuntimeError("live joint-position targets must retain canonical order")
        target_ids = getattr(action_term, "target_ids", None)
        if not isinstance(target_ids, torch.Tensor) or target_ids.shape != (len(JOINT_NAMES),):
            raise RuntimeError("live joint-position target ids must have shape (6,)")
        expected_ids, expected_names = robot.find_joints(JOINT_NAMES)
        if (
            target_ids.detach().cpu().tolist() != expected_ids
            or tuple(expected_names) != JOINT_NAMES
        ):
            raise RuntimeError("live joint-position target ids must retain canonical order")
        if (
            qvel_tracker.peak_qv_joint.shape != (NUM_ENVS, len(JOINT_NAMES))
            or r_imit_term._contacted.shape != (NUM_ENVS,)
            or r_imit_term._contacted.dtype != torch.bool
        ):
            raise RuntimeError("live FIC measurement instrumentation shape drift")
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        # Runner construction can consume Torch RNG state; reset the global stream at
        # the population boundary as well as supplying the environment seed above.
        torch.manual_seed(SEED)
        observations, _ = wrapped.reset()
        if (
            robot.data.joint_pos_target[:, target_ids].shape
            != (NUM_ENVS, len(JOINT_NAMES))
            or robot.data.joint_pos[:, target_ids].shape
            != (NUM_ENVS, len(JOINT_NAMES))
        ):
            raise RuntimeError("live joint-position measurement shape drift")
        head_site_ids = r_imit_cfg.params["robot_cfg"].site_ids
        head = robot.data.site_pos_w[:, head_site_ids].squeeze(1)
        if head.shape != (NUM_ENVS, 3):
            raise RuntimeError("live hammer-head measurement shape drift")
        reference = get_strike_reference(env)
        population_hash = _population_hash(env)
        records: dict[int, dict[str, object]] = {}
        steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=env.device)
        episode_accumulators = _new_episode_accumulators(NUM_ENVS, env.device)
        max_steps = env.max_episode_length
        for _ in range(max_steps):
            with torch.no_grad():
                actions = policy(observations, stochastic_output=False)
            observations, _, _, _ = wrapped.step(actions)
            steps += 1
            target_error = (
                robot.data.joint_pos_target[:, target_ids]
                - robot.data.joint_pos[:, target_ids]
            )
            contacted = r_imit_term._contacted
            reference_eligible = ~contacted
            head = robot.data.site_pos_w[:, head_site_ids].squeeze(1)
            reference_waypoint = reference.waypoint(
                reference.preview(head, env.episode_length_buf)
            )
            reference_error = (head - reference_waypoint).norm(dim=1)
            _accumulate_episode_measurements(
                episode_accumulators,
                qvel_peak=qvel_tracker.peak_qv_joint,
                target_error=target_error,
                reference_error=reference_error,
                reference_eligible=reference_eligible,
            )
            done = env.reset_terminated | env.reset_time_outs
            done_ids = capture_first_terminals(
                records=records,
                done=done,
                terminated=env.reset_terminated,
                timed_out=env.reset_time_outs,
                steps=steps,
                tracker=tracker,
                nail_depth=env.scene["nail_block"].data.joint_pos[:, 0].clamp(
                    0.0, NAIL_GOAL_DEPTH
                ),
                joint_impulse_peak=impulse_accumulator._episode_peak_perjoint,
                episode_accumulators=episode_accumulators,
            )
            if len(records) == NUM_ENVS:
                break
            if done_ids:
                ids = torch.tensor(done_ids, device=env.device, dtype=torch.long)
                observations = reset_done_envs_preserving_unfinished(
                    env, observations, ids
                )
                steps[ids] = 0
                _reset_episode_accumulators(episode_accumulators, ids)
        ordered = _ordered_complete_records(records)
        payload: dict[str, object] = {
            "schema_version": 2,
            "task": task,
            "training_seed": training_seed,
            "protocol": {
                "evaluation_seed": SEED,
                "num_envs": NUM_ENVS,
                "episodes_per_env": 1,
                "episode_length_s": EPISODE_LENGTH_S,
                "auto_reset": False,
                "policy_mode": "mean",
            },
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256_file(checkpoint)},
            "code_git": code_git,
            "asset_git": asset_git,
            "treatment_contract": contract,
            "initial_population_sha256": population_hash,
            "episodes": ordered,
            "summary": _summary(ordered),
        }
    finally:
        wrapped.close()
    _publish_new_json(output, payload)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-seed", required=True, type=int, choices=(2, 3, 4))
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    payload = evaluate_checkpoint(
        args.task,
        args.checkpoint,
        args.output,
        args.device,
        training_seed=args.training_seed,
    )
    print(
        f"FIC pilot evaluation complete output={args.output} "
        f"success={payload['summary']['task_success_n']}/{NUM_ENVS}"
    )


if __name__ == "__main__":
    main()
