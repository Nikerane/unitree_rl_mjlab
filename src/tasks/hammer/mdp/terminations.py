"""Termination terms for the Z1 hammer-nail task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_NAIL_CFG = SceneEntityCfg("nail_block", joint_names=("nail_slide",))


def nail_fully_driven(
  env: ManagerBasedRlEnv,
  success_depth: float,
  nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
  """Terminate (success) when the nail has been driven to success_depth.

  Shape: (B,) bool tensor.
  """
  nail_entity: Entity = env.scene[nail_cfg.name]
  current_depth = nail_entity.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
  return current_depth >= success_depth
