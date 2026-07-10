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

from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR
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


def joint_impulse_excess(env, limit: float | torch.Tensor) -> torch.Tensor:
  """Raw per-joint impact-impulse margin ``Λ_j − limit``, shape (B, J). Positive ⇒ over the limit.

  Reads the contact-anchored, substep-accumulated per-joint reaction impulse from the
  ``SubstepImpulseAccumulator`` stashed on the env (the accumulator's own robot_cfg fixes the joint
  set — there is deliberately no robot_cfg here). Returns the RAW signed margin only — the CaT
  manager does all clamp/EMA/normalization. NEVER a probability, NEVER a reward term (the per-joint
  impact impulse must be enforced via the CaT termination probability, not penalized; hard
  constraint #1). ``limit`` is REQUIRED (a scalar or a per-joint (J,) tensor, broadcasts) — the
  ``Z1_JOINT_IMPULSE_LIMIT`` placeholder must never be reachable as a silent fallback.
  """
  acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
  if acc is None:
    raise RuntimeError(
      "joint_impulse_excess needs the SubstepImpulseAccumulator per_substep metric wired into "
      "cfg.metrics (see env_cfgs.py cat_impulse) so it stashes itself on the env."
    )
  return acc.impulse - limit  # (B, J) − scalar/(J,) broadcasts to per-joint thresholds
