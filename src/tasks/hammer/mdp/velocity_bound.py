"""Joint-velocity-bound ablation terms (Z1 hammer, Phase-0).

b_strike showed the trained strike drives arm joints to 4.3-4.65 rad/s worst-case,
over the real Z1 limit of 3.1415 rad/s (URDF; see [[z1-hardware-limits]]). These terms
are the soft-constraint ablation arms that try to make the strike honest WITHOUT
lowering delta_pos_scale (so the hard-strike scale is kept and the limit is enforced
as a constraint):

  - A2  joint_vel_excess_penalty  -- reward term: -w * Sum_j max(0,|qdot_j| - limit*beta)^2
        (excess-over-threshold, NEVER raw ||qdot||^2 per the repo rule; silent below the
        limit, only bites the overshoot). Annealed in after the strike is learned.

  - A3  CaTJointVelConstraint     -- termination term implementing Constraints-as-Terminations
        (Chane-Sane et al., IROS 2024, [[cat-constraints-as-terminations]]): terminate with
        probability delta = p_max * clip(excess / EMA(batch-max excess), 0, 1). A CaT
        termination is NON-timeout, so PPO does not bootstrap it -> the policy sees the lost
        future reward (incl. completion=100) and learns to keep the wind-up under the limit.

WAVE-1 SIGNAL = CONTROL-RATE joint velocity (post-decimation robot.data.joint_vel), the
same quantity diag_policy_trace.py reports, so the bound target and the eval are consistent.
The substep-peak refinement (a per_substep MetricsTerm) and per-joint impulse Lambda_j
logging are deferred to v2 (Lambda_j needs qfrc_constraint, not exposed on the Entity).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Real Unitree Z1 joint-velocity limit (rad/s), every joint (z1_description URDF).
Z1_JOINT_VEL_LIMIT: float = 3.1415

# Arm joints (exclude the vestigial gripper, which carries reset-noise velocity).
_ARM_CFG = SceneEntityCfg("robot", joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"))


def joint_vel_excess_penalty(
  env: "ManagerBasedRlEnv",
  limit: float = Z1_JOINT_VEL_LIMIT,
  beta: float = 0.9,
  robot_cfg: SceneEntityCfg = _ARM_CFG,
) -> torch.Tensor:
  """A2: squared excess of arm joint speed over ``limit*beta``. Shape (B,).

  Excess-over-threshold (NOT raw ||qdot||^2): silent while every joint is under
  ``limit*beta`` (engages at ~2.83 rad/s for beta=0.9, leaving honest sub-limit
  strikes untouched), squared so the worst-case tail (4.65) is crushed hardest.
  The RewardTermCfg weight must be NEGATIVE.
  """
  robot: Entity = env.scene[robot_cfg.name]
  qv = robot.data.joint_vel[:, robot_cfg.joint_ids].abs()  # (B, J)
  excess = (qv - limit * beta).clamp_min(0.0)
  return torch.sum(excess**2, dim=-1)


class CaTJointVelConstraint(ManagerTermBase):
  """A3: Constraints-as-Terminations on arm joint velocity.

  Per control step: c = max(0, max_j|qdot_j| - limit) is the worst-joint excess;
  delta = p_max * clip(c / c_max, 0, 1) where c_max is an EMA of the batch-max excess
  (CaT Eq. 6-7). Returns a sampled Bernoulli(delta) termination mask (bool, NON-timeout,
  so PPO does not bootstrap it). p_max in (0,1]: 1.0 = hard, <1 = soft. EMA persists
  across episodes (it is a batch statistic), so reset() is a no-op.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    # EMA of the batch-max excess; seed at the limit's order so early big violations
    # (excess ~1.5) still map to a meaningful delta before the EMA warms up.
    self._excess_max: torch.Tensor = torch.ones((), device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    return None

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    limit: float = Z1_JOINT_VEL_LIMIT,
    p_max: float = 0.5,
    tau: float = 0.95,
    robot_cfg: SceneEntityCfg = _ARM_CFG,
  ) -> torch.Tensor:
    """Returns a bool termination mask, shape (B,)."""
    robot: Entity = env.scene[robot_cfg.name]
    qv = robot.data.joint_vel[:, robot_cfg.joint_ids].abs()  # (B, J)
    c = (qv.amax(dim=1) - limit).clamp_min(0.0)              # (B,) worst-joint excess
    # Update the EMA of the batch-max excess (detached; a normalizer, not a gradient path).
    batch_max = c.max().detach()
    self._excess_max = tau * self._excess_max + (1.0 - tau) * batch_max
    denom = self._excess_max.clamp_min(1e-6)
    delta = p_max * (c / denom).clamp(0.0, 1.0)              # (B,) termination probability
    return torch.rand_like(delta) < delta
