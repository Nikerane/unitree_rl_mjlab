"""Observation terms for the Z1 hammer-nail task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.hammer.mdp.references import get_strike_reference

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_NAIL_CFG = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
# NOTE: strike_phase / strike_ref_error take robot_cfg/nail_cfg as REQUIRED
# params (no defaults): a site-less SceneEntityCfg("robot") default would
# resolve to ALL sites and silently return (B, nsites, 3) (review finding).


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


def strike_phase(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  nail_cfg: SceneEntityCfg,
  horizontal_detour_m: float = 0.0,
  followthrough_mode: str = "vertical",
) -> torch.Tensor:
  """Phase of the direct scripted single-strike reference. Shape: (B, 1).

  ``phi`` is monotone-latched spatial progress on the reset-head-to-
  follow-through segment, re-anchored on reset.
  """
  robot: Entity = env.scene[robot_cfg.name]
  nail_entity: Entity = env.scene[nail_cfg.name]
  head_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
  nail_top_w = nail_entity.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)
  ref = get_strike_reference(
    env,
    horizontal_detour_m=horizontal_detour_m,
    followthrough_mode=followthrough_mode,
  )
  phi = ref.update(head_w, nail_top_w, env.episode_length_buf)
  return phi.unsqueeze(-1)


def strike_ref_error(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  nail_cfg: SceneEntityCfg,
  horizontal_detour_m: float = 0.0,
  followthrough_mode: str = "vertical",
) -> torch.Tensor:
  """Vector from the hammer head to the current direct-strike waypoint.

  Gives the policy the reference to anticipate (anchor, not cage). Shape: (B, 3).
  """
  robot: Entity = env.scene[robot_cfg.name]
  nail_entity: Entity = env.scene[nail_cfg.name]
  head_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
  nail_top_w = nail_entity.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)
  ref = get_strike_reference(
    env,
    horizontal_detour_m=horizontal_detour_m,
    followthrough_mode=followthrough_mode,
  )
  phi = ref.update(head_w, nail_top_w, env.episode_length_buf)
  return ref.waypoint(phi) - head_w


def nail_depth(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
  """Current nail slide qpos (0 = not driven, NAIL_GOAL_DEPTH=0.032 = fully driven). Shape: (B, 1)."""
  nail_entity: Entity = env.scene[asset_cfg.name]
  # joint_pos has shape (B, n_joints); nail_slide is the only joint.
  depth = nail_entity.data.joint_pos[:, asset_cfg.joint_ids]
  # Clamp the OBSERVED depth to >= 0. The nail's declared joint range is
  # [0, 0.032], but the soft lower-limit constraint lets a hooking hammer claw
  # extract the nail a few mm past 0 under adversarial action (audit #4,
  # 2026-06-16: a sustained pure-up command pulls a seated nail to ~-9 mm). The
  # policy is never rewarded for that — the depth-progress terms ratchet on
  # max-depth-so-far so re-driving re-covered ground earns nothing, and the
  # Gaussian nail_driven term (which keeps reading the raw qpos) actually drops
  # when the nail is pulled up. Clamping only the obs keeps the network's input
  # in the physical [0, .] range without touching physics/reward.
  return depth.clamp_min(0.0)
