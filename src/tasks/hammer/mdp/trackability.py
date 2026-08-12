"""Post-decimation readers for one-step joint-command trackability."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
_QUALIFIED_ACTION_SIGNATURES = (
  ("joint_position",),
  ("joint_position", "joint_stiffness"),
)


def _id_tuple(value: Any) -> tuple[int, ...]:
  if isinstance(value, slice):
    if value.stop is None:
      raise ValueError("trackability requires six explicitly resolved joint ids")
    return tuple(range(value.start or 0, value.stop, value.step or 1))
  if hasattr(value, "detach"):
    value = value.detach().cpu().tolist()
  return tuple(int(item) for item in value)


def _validated_joint_state(
  env: "ManagerBasedRlEnv", robot_cfg: "SceneEntityCfg"
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return public applied targets and post-step positions for the six-joint arm."""
  joint_ids = _id_tuple(robot_cfg.joint_ids)
  joint_names = tuple(robot_cfg.joint_names or ())
  if len(joint_ids) != len(_ARM_JOINT_NAMES) or joint_names != _ARM_JOINT_NAMES:
    raise ValueError(
      "trackability robot_cfg must resolve exactly joint1 through joint6 in order"
    )

  active_terms = tuple(env.action_manager.active_terms)
  if active_terms not in _QUALIFIED_ACTION_SIGNATURES:
    raise ValueError(
      "joint trackability requires exactly ('joint_position',) or "
      "('joint_position', 'joint_stiffness') in that order"
    )
  action_term = env.action_manager.get_term("joint_position")
  if tuple(action_term.target_names) != _ARM_JOINT_NAMES:
    raise ValueError(
      "joint-position action target order must match trackability joint1 through joint6"
    )

  robot = env.scene[robot_cfg.name]
  applied = robot.data.joint_pos_target[:, list(joint_ids)]
  actual = robot.data.joint_pos[:, list(joint_ids)]
  if applied.ndim != 2 or applied.shape != actual.shape:
    raise ValueError(
      "applied joint target and post-decimation joint position shape must match"
    )
  if applied.shape[1] != len(_ARM_JOINT_NAMES):
    raise ValueError("trackability state must contain exactly six arm joints")
  return applied, actual


def joint_target_squared_error(
  env: "ManagerBasedRlEnv", robot_cfg: "SceneEntityCfg"
) -> torch.Tensor:
  """Return ``sum_j (q_des_applied(t) - q_actual(t+1))**2`` for joint1…joint6."""
  applied, actual = _validated_joint_state(env, robot_cfg)
  return torch.square(applied - actual).sum(dim=1)


def joint_target_rmse(
  env: "ManagerBasedRlEnv", robot_cfg: "SceneEntityCfg"
) -> torch.Tensor:
  """Return the six-joint root-mean-square one-step command error."""
  return torch.sqrt(joint_target_squared_error(env, robot_cfg) / len(_ARM_JOINT_NAMES))


def joint_trackability_cost(
  env: "ManagerBasedRlEnv", robot_cfg: "SceneEntityCfg", k_tt: float
) -> torch.Tensor:
  """Return the nonnegative pre-RewardManager-dt tracking cost."""
  if isinstance(k_tt, bool) or not isinstance(k_tt, (int, float)):
    raise ValueError("k_tt must be a finite, strictly positive number")
  gain = float(k_tt)
  if not math.isfinite(gain) or gain <= 0.0:
    raise ValueError("k_tt must be finite and strictly positive")
  return gain * joint_target_squared_error(env, robot_cfg)
