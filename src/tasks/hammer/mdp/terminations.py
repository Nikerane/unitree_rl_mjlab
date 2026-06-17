"""Termination terms for the Z1 hammer-nail task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.hammer.mdp.rewards import clamped_nail_depth

if TYPE_CHECKING:
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
  current_depth = clamped_nail_depth(env, nail_cfg)
  return current_depth >= success_depth
