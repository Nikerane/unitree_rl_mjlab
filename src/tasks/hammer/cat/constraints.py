"""CaT constraint functions for the Z1 hammer task.

Each returns the RAW signed margin ``c = value − limit`` (NOT clamped, NOT a probability) — the
``CaT`` manager does the clamping/normalization. Shape is (B, J) per-joint, so each joint is
normalized by its own EMA ``c_max``. Mirrors the reference constraint convention
(constraints-as-terminations/.../cat/constraints.py: joint_velocity returns ``|q̇| − limit``).

The substep-accumulated per-joint IMPULSE constraint (the thesis target) plugs in here as another
func reading a substep-accumulated buffer; for C0 we provide the joint-velocity margin, reusing the
existing arm config + limit from the velocity-bound ablation module.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT, _ARM_CFG


def joint_velocity_excess(
  env,
  limit: float = Z1_JOINT_VEL_LIMIT,
  robot_cfg: SceneEntityCfg = _ARM_CFG,
) -> torch.Tensor:
  """Raw per-joint velocity margin ``|q̇_j| − limit``, shape (B, J). Positive ⇒ over the limit."""
  robot: Entity = env.scene[robot_cfg.name]
  qv = robot.data.joint_vel[:, robot_cfg.joint_ids].abs()  # (B, J)
  return qv - limit
