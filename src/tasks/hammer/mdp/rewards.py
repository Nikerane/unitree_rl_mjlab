"""Reward terms for the Z1 hammer-nail task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_NAIL_CFG = SceneEntityCfg("nail_block", joint_names=("nail_slide",))


def nail_driven_reward(
  env: ManagerBasedRlEnv,
  goal_depth: float,
  std: float,
  nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
  """Gaussian reward shaped on how far the nail has been driven.

  Returns exp(-error^2 / std^2) where error = goal_depth - current_depth.
  Shape: (B,).
  """
  nail_entity: Entity = env.scene[nail_cfg.name]
  current_depth = nail_entity.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
  error = goal_depth - current_depth
  return torch.exp(-(error**2) / std**2)


def hammer_approach_reward(
  env: ManagerBasedRlEnv,
  std: float,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
  """Gaussian reward for bringing hammer head close to nail top.

  Encourages the policy to align the hammer with the nail before striking.
  Shape: (B,).
  """
  robot: Entity = env.scene[robot_cfg.name]
  nail_entity: Entity = env.scene[nail_cfg.name]

  head_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
  nail_pos_w = nail_entity.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

  dist_sq = torch.sum((head_pos_w - nail_pos_w) ** 2, dim=-1)
  return torch.exp(-dist_sq / std**2)


def action_rate_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """L2 penalty on the change in actions between consecutive steps.

  Penalises jerky motions. Shape: (B,).
  """
  return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)
