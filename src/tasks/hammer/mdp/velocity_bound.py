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
The substep-peak refinement (a per_substep MetricsTerm) is SubstepPeakJointVel below; the
per-joint impulse Lambda_j accumulator is now implemented in impulse_bound.py. (CORRECTION:
qfrc_constraint IS accessible per-DoF via entity.data._joint_dof_field("qfrc_constraint") --
the earlier "not exposed on the Entity" note was outdated; no site-packages edit is needed.)
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

# Where SubstepPeakJointVel stashes itself on the env so the (control-rate) CaT termination
# can read the substep-peak |q̇| instead of the aliased post-decimation sample.
_ENV_SUBSTEP_ATTR = "_hammer_substep_peak_qv"


class SubstepPeakJointVel(ManagerTermBase):
  """per_substep MetricsTerm: running MAX of |arm joint vel| over each control window.

  CaT's termination manager runs once per CONTROL step (after the decimation loop), so a
  control-rate read sees only the last substep and ALIASES the 500 Hz peak (the spike
  implicated in A3's residual). This term, evaluated every physics substep inside the
  decimation loop, peak-holds |q̇| over the window and stashes itself on the env so the CaT
  termination can read `peak_qv` (the true within-window peak). Returns the per-substep
  |q̇|.amax for logging (the manager means it over substeps).

  Pattern mirrors the ContactSensor history+max. Window resets at the first substep of each
  control step (i % decimation == 0); episode reset clears it too.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    arm = SceneEntityCfg("robot", joint_names=_ARM_CFG.joint_names)
    arm.resolve(env.scene)
    self._robot: Entity = env.scene["robot"]
    self._joint_ids = arm.joint_ids
    self._dec = int(env.cfg.decimation)
    self.peak_qv: torch.Tensor = torch.zeros(env.num_envs, device=env.device)
    self._i = 0
    setattr(env, _ENV_SUBSTEP_ATTR, self)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self.peak_qv.zero_()
    else:
      self.peak_qv[env_ids] = 0.0
    return None

  def __call__(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    if self._i % self._dec == 0:        # first substep of a new control window -> reset peak
      self.peak_qv.zero_()
    qv = self._robot.data.joint_vel[:, self._joint_ids].abs().amax(dim=1)  # (B,)
    torch.maximum(self.peak_qv, qv, out=self.peak_qv)                       # in-place: keep identity
    self._i += 1
    return qv


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
    detection: str = "control_rate",
  ) -> torch.Tensor:
    """Returns a bool termination mask, shape (B,).

    detection: "control_rate" reads the post-decimation joint_vel (aliases the 500 Hz peak);
    "substep" reads the within-window peak from the SubstepPeakJointVel metric (must be wired
    into cfg.metrics) -- the honest signal that catches the spike A3 missed.
    """
    if detection == "substep":
      tracker = getattr(env, _ENV_SUBSTEP_ATTR, None)
      if tracker is None:
        raise RuntimeError(
          "CaTJointVelConstraint detection='substep' requires the SubstepPeakJointVel "
          "per_substep metric wired into cfg.metrics (see env_cfgs.py cat_substep)."
        )
      qv_peak = tracker.peak_qv                                # (B,) within-control-window peak
    else:
      robot: Entity = env.scene[robot_cfg.name]
      qv_peak = robot.data.joint_vel[:, robot_cfg.joint_ids].abs().amax(dim=1)  # (B,) control-rate
    c = (qv_peak - limit).clamp_min(0.0)                       # (B,) worst-joint excess
    # Update the EMA of the batch-max excess (detached; a normalizer, not a gradient path).
    batch_max = c.max().detach()
    self._excess_max = tau * self._excess_max + (1.0 - tau) * batch_max
    denom = self._excess_max.clamp_min(1e-6)
    delta = p_max * (c / denom).clamp(0.0, 1.0)              # (B,) termination probability
    return torch.rand_like(delta) < delta


def joint_vel_hard_termination(
  env: "ManagerBasedRlEnv",
  limit: float = Z1_JOINT_VEL_LIMIT,
  warmup_steps: int = 3600,
  robot_cfg: SceneEntityCfg = _ARM_CFG,
  detection: str = "substep",
) -> torch.Tensor:
  """DETERMINISTIC hard termination: any arm joint over ``limit`` ends the episode.

  The strongest *learned* enforcement (top of the soft->hard sweep; the "just end the episode
  when it speeds" idea). NOTE it does NOT physically prevent the spike -- it ends the episode the
  control step AFTER the joint already crossed ``limit``, so it is a maximal learning signal
  ("never go there or lose the whole episode + the completion bonus"), not a brake. Whether it
  bounds velocity depends on feasibility: if a limit-respecting hard strike exists on fixed PD the
  policy learns it; if not, it can only comply by striking softly (the strike-vs-honesty tension
  VIC is meant to resolve).

  CURRICULUM (mandatory): disabled for the first ``warmup_steps`` control steps so the strike skill
  forms first -- a hard cut from scratch makes every early (flailing) episode die before the policy
  ever learns to strike (the documented all-hard collapse). ``detection='substep'`` reads the true
  within-window peak (needs the SubstepPeakJointVel metric wired).
  """
  if env.common_step_counter < warmup_steps:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if detection == "substep":
    tracker = getattr(env, _ENV_SUBSTEP_ATTR, None)
    if tracker is None:
      raise RuntimeError(
        "joint_vel_hard_termination detection='substep' requires the SubstepPeakJointVel "
        "per_substep metric wired into cfg.metrics (see env_cfgs.py vel_hard_term)."
      )
    qv_peak = tracker.peak_qv
  else:
    robot: Entity = env.scene[robot_cfg.name]
    qv_peak = robot.data.joint_vel[:, robot_cfg.joint_ids].abs().amax(dim=1)
  return qv_peak > limit
