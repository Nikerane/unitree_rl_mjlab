"""On-policy runner for the hammer-nail task."""

import json
import math
from pathlib import Path

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import DifferentialIKAction, JointPositionAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.hammer.config.z1.joint_position_contract import (
    JOINT_NAMES,
    load_joint_position_contract,
)


_JOINT_POSITION_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/z1/data/z1_joint_position_stage1.json"
)


def _finite_row(value: torch.Tensor | float, *, name: str) -> list[float]:
    """Return the first resolved action row after checking the full live tensor."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 2 or value.shape[1] != len(JOINT_NAMES):
            raise ValueError(f"{name} must contain one value for each Z1 arm joint")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return [float(item) for item in value[0].detach().cpu().tolist()]
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return [float(value)] * len(JOINT_NAMES)


def _fixed_actuator_signature(env) -> list[dict[str, object]]:
    """Serialize the fixed plant's resolved actuator configuration safely."""
    actuators = env.cfg.scene.entities["robot"].articulation.actuators
    signature = []
    for actuator in actuators:
        values = {
            "stiffness": float(actuator.stiffness),
            "damping": float(actuator.damping),
            "effort_limit": float(actuator.effort_limit),
            "armature": float(actuator.armature),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("fixed actuator signature must contain only finite values")
        signature.append(
            {
                "type": type(actuator).__name__,
                "target_names_expr": list(actuator.target_names_expr),
                **values,
            }
        )
    return signature


def _get_hammer_metadata(env, run_path: str, *, raw_policy_clip: float) -> dict:
    action_terms = list(env.action_manager.active_terms)
    if len(action_terms) != 1:
        raise ValueError("hammer ONNX metadata requires exactly one active action")
    action_term = action_terms[0]
    if action_term not in ("ik_hammer_head", "joint_position"):
        raise ValueError(f"unsupported action term for hammer ONNX metadata: {action_term}")

    robot: Entity = env.scene["robot"]
    action = env.action_manager.get_term(action_term)
    if isinstance(action, DifferentialIKAction):
        if action_term != "ik_hammer_head":
            raise ValueError(
                f"unsupported action term for Cartesian metadata: {action_term}"
            )
        return {
            "run_path": run_path,
            "action_type": "ik_delta_pos",
            "frame_name": action.cfg.frame_name,
            "delta_pos_scale": action.cfg.delta_pos_scale,
            "joint_names": list(robot.joint_names),
            "observation_names": env.observation_manager.active_terms["actor"],
        }
    if not isinstance(action, JointPositionAction):
        raise ValueError(f"unsupported action type for hammer ONNX metadata: {type(action).__name__}")
    if action_term != "joint_position":
        raise ValueError(f"unsupported action term for joint metadata: {action_term}")
    if action.action_dim != len(JOINT_NAMES):
        raise ValueError("joint metadata must contain exactly six action dimensions")
    if tuple(action.target_names) != JOINT_NAMES:
        raise ValueError("joint metadata requires the canonical Z1 joint order")
    target_ids = action.target_ids.detach().cpu().tolist()
    expected_ids, expected_names = robot.find_joints(JOINT_NAMES)
    if target_ids != expected_ids or tuple(expected_names) != JOINT_NAMES:
        raise ValueError("joint metadata requires the canonical Z1 joint order")
    if tuple(action.cfg.actuator_names) != JOINT_NAMES:
        raise ValueError("joint metadata requires the canonical Z1 actuator order")
    if not action.cfg.use_default_offset:
        raise ValueError("joint metadata requires default-offset action semantics")
    physics_dt_s = env.cfg.sim.mujoco.timestep
    if (
        isinstance(physics_dt_s, bool)
        or not isinstance(physics_dt_s, (int, float))
        or not math.isfinite(float(physics_dt_s))
        or physics_dt_s != 0.002
    ):
        raise ValueError("joint metadata requires physics timestep 0.002 s")
    control_decimation = env.cfg.decimation
    if (
        isinstance(control_decimation, bool)
        or not isinstance(control_decimation, int)
        or control_decimation != 10
    ):
        raise ValueError("joint metadata requires control decimation 10")
    if (
        isinstance(raw_policy_clip, bool)
        or not isinstance(raw_policy_clip, (int, float))
        or not math.isfinite(float(raw_policy_clip))
        or raw_policy_clip != 1.0
    ):
        raise ValueError("joint metadata requires raw policy clip 1.0")
    scales = _finite_row(action.scale, name="joint action scale")
    offsets = _finite_row(action.offset, name="joint action default offsets")
    clips = action._clip
    if clips.ndim != 3 or clips.shape[1:] != (len(JOINT_NAMES), 2):
        raise ValueError("joint physical clips must contain six [min, max] pairs")
    if not torch.isfinite(clips).all() or not torch.all(clips[:, :, 0] < clips[:, :, 1]):
        raise ValueError("joint physical clips must be finite increasing intervals")
    contract = load_joint_position_contract(_JOINT_POSITION_CONTRACT_PATH)

    delivered_i_ref = env.reward_manager.get_term_cfg(
        "delivered_impulse"
    ).params["i_ref"]
    if (
        isinstance(delivered_i_ref, bool)
        or not isinstance(delivered_i_ref, (int, float))
        or not math.isfinite(float(delivered_i_ref))
        or float(delivered_i_ref) <= 0.0
    ):
        raise ValueError("delivered impulse i_ref must be finite and strictly positive")

    r_tt_enabled = "r_tt" in env.reward_manager.active_terms
    if r_tt_enabled:
        r_tt_k_tt = float(env.reward_manager.get_term_cfg("r_tt").params["k_tt"])
        if not math.isfinite(r_tt_k_tt):
            raise ValueError("r_tt k_tt must be finite")
    else:
        r_tt_k_tt = "not_applicable"
    return {
        "run_path": run_path,
        "action_type": "joint_position",
        "action_term": action_term,
        "action_dim": action.action_dim,
        "target_names": list(action.target_names),
        "target_ids": target_ids,
        "actuator_names": list(action.cfg.actuator_names),
        "use_default_offset": action.cfg.use_default_offset,
        "default_offsets": offsets,
        "action_scale": scales,
        "physical_clips": clips[0].detach().cpu().tolist(),
        "raw_policy_clip": float(raw_policy_clip),
        "physics_dt_s": float(physics_dt_s),
        "control_decimation": control_decimation,
        "fixed_actuator_signature": _fixed_actuator_signature(env),
        "joint_action_qualification_payload_sha256": contract.payload_sha256,
        "delivered_impulse_i_ref_n_s": float(delivered_i_ref),
        "r_tt_enabled": r_tt_enabled,
        "r_tt_k_tt": r_tt_k_tt,
    }


def _metadata_for_onnx(metadata: dict) -> dict:
    """Encode structured joint-policy fields without upstream CSV rounding."""
    if metadata.get("action_type") != "joint_position":
        return metadata
    return {
        key: json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if isinstance(value, (list, dict))
        else value
        for key, value in metadata.items()
    }


class HammerOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    is_wandb = self.logger.logger_type == "wandb"
    if is_wandb:
      import wandb  # lazy: a tensorboard run (the Lightning default) must not require wandb installed
      run_name = wandb.run.name if wandb.run else "local"
    else:
      run_name = "local"
    metadata = _get_hammer_metadata(
      self.env.unwrapped, run_name, raw_policy_clip=self.env.clip_actions
    )
    metadata_for_onnx = _metadata_for_onnx(metadata)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      attach_metadata_to_onnx(str(onnx_path), metadata_for_onnx)
      if is_wandb and self.cfg.get("upload_model"):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
