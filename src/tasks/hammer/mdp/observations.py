"""Observation terms for the Z1 hammer-nail task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_NAIL_CFG = SceneEntityCfg("nail_block", joint_names=("nail_slide",))


def hammer_head_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Position of hammer_head_site in robot base frame. Shape: (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  head_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  base_pos_w = robot.data.root_link_pos_w
  return head_pos_w - base_pos_w


def hammer_head_vel_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Linear velocity of hammer_head_site in world frame. Shape: (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  site_vel_w = robot.data.site_vel_w[:, asset_cfg.site_ids].squeeze(1)
  return site_vel_w[:, :3]


def ee_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Position of ee_center_site relative to robot base. Shape: (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  base_pos_w = robot.data.root_link_pos_w
  return ee_pos_w - base_pos_w


def ee_vel_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Linear velocity of ee_center_site in world frame. Shape: (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  site_vel_w = robot.data.site_vel_w[:, asset_cfg.site_ids].squeeze(1)
  return site_vel_w[:, :3]


def nail_top_pos_w(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
  """World position of nail_top site. Shape: (B, 3)."""
  nail_entity: Entity = env.scene[asset_cfg.name]
  return nail_entity.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def nail_depth(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
  """Current nail slide qpos (0 = not driven, 0.075 = fully driven). Shape: (B, 1)."""
  nail_entity: Entity = env.scene[asset_cfg.name]
  # joint_pos has shape (B, n_joints); nail_slide is the only joint.
  depth = nail_entity.data.joint_pos[:, asset_cfg.joint_ids]
  return depth
