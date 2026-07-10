"""Reward terms for the Z1 hammer-nail task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR
from src.tasks.hammer.mdp.references import get_strike_reference
from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_NAIL_CFG = SceneEntityCfg("nail_block", joint_names=("nail_slide",))


def clamped_nail_depth(env: "ManagerBasedRlEnv", nail_cfg: SceneEntityCfg) -> torch.Tensor:
  """Nail slide depth clamped to the physical joint range [0, NAIL_GOAL_DEPTH]. Shape (B,).

  The slide's soft limits let a hard strike transiently overshoot the 0.032 m stop to
  ~63 mm (and a hooking claw pull it below 0). Reward/termination terms must read the
  PHYSICAL depth, not the elastic excursion -- otherwise a harder strike inflates the
  progress reward and any windowed impulse. (Only the observation was clamped before,
  and only on the low side; observations.py:129.)
  """
  nail: Entity = env.scene[nail_cfg.name]
  depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
  return depth.clamp(0.0, NAIL_GOAL_DEPTH)


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
  current_depth = clamped_nail_depth(env, nail_cfg)
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


def completion_bonus(
  env: ManagerBasedRlEnv,
  success_depth: float,
  nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
  """Sparse +1 task-completion reward when nail_slide qpos >= success_depth.

  This is the actual task reward (sparse goal signal), not shaping. Multiplied
  by weight in the RewardTermCfg. The env terminates on success via the
  nail_driven TerminationTermCfg, so this fires at most once per episode.
  Shape: (B,).
  """
  depth = clamped_nail_depth(env, nail_cfg)
  return (depth >= success_depth).float()


class NailDepthDeltaTerm(ManagerTermBase):
  """Progress reward: only positive changes in nail depth are rewarded.

  Tracks max_depth_so_far per environment across the episode and returns
  max(0, current_depth - max_depth_so_far) each step. Provides a non-zero
  gradient from the very first mm of nail travel, unlike the Gaussian
  nail_driven_reward which is near-zero at 0mm depth.

  Stateful: requires per-episode reset of _max_depth via reset(env_ids).
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._max_depth: torch.Tensor = torch.full(
      (env.num_envs,), self._SETTLE_OFFSET, dtype=torch.float32, device=env.device
    )

  # WHY THIS EXISTS (now belt-and-suspenders):
  # Originally absorbed a gravity-creep artefact — the nail drifting down a few mm
  # at reset because MuJoCo joint frictionloss does NOT statically hold it. As of
  # 2026-06-15 that is fixed at the SOURCE with gravcomp="1" on the nail body
  # (nail_block_scene.xml), so the nail now holds at ~0 and there is no settling to
  # absorb. The dead zone is kept as a cheap guard; it can be lowered toward 0 in a
  # future change (would also shift validate_rewards Phase C/E/F expected values).
  # Cost of keeping it: the first 4 mm of genuine nail progress earns no
  # nail_depth_delta — negligible, since a real strike clears 4 mm on first contact.
  _SETTLE_OFFSET: float = 0.004  # 4 mm dead zone (legacy gravity-settling guard)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self._max_depth.fill_(self._SETTLE_OFFSET)
    else:
      self._max_depth[env_ids] = self._SETTLE_OFFSET

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
  ) -> torch.Tensor:
    """Returns shape (B,)."""
    depth = clamped_nail_depth(env, nail_cfg)
    delta = (depth - self._max_depth).clamp_min(0.0)
    self._max_depth = torch.maximum(self._max_depth, depth)
    return delta


class ImpactProgressTerm(ManagerTermBase):
  """Double-gated momentum reward: (v_axial / v_expected) · 1[first_contact] · 1[Δdepth > ε].

  Pays end-effector axial (downward) impact speed, but ONLY on the control step a
  fresh hammer→nail contact begins AND only when that contact drives the nail past
  its previous max depth. The two indicator gates make the speed bonus unfarmable by
  scraping/tapping that produces no progress (Skalse et al. 2022; Pan et al. 2022).

  On the position-only DifferentialIK action space the controllable impact lever is
  end-effector momentum, so this term rewards pre-impact axial speed gated on real
  nail progress. Velocity is finite-differenced from the hammer-head site position
  (robust to a lazily-updated ``site_vel_w``).

  Stateful: stores the previous head position, the max nail depth so far, and a
  per-env init flag; all reset per-episode via reset(env_ids).
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._prev_head: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
    self._prev_depth: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
    self._init: torch.Tensor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self._init.fill_(False)
      self._prev_depth.zero_()
    else:
      self._init[env_ids] = False
      self._prev_depth[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
    axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    eps: float = 5e-4,
    v_expected: float = 1.0,
  ) -> torch.Tensor:
    """Returns shape (B,)."""
    robot: Entity = env.scene[robot_cfg.name]
    sensor = env.scene[sensor_name]
    dt = env.step_dt

    # Finite-difference axial speed (downward, along the nail axis). Zeroed on the
    # first step after a reset so the spawn/teleport cannot register as velocity.
    head = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    vel = torch.where(
      self._init[:, None], (head - self._prev_head) / dt, torch.zeros_like(head)
    )
    self._prev_head = head.clone()
    self._init.fill_(True)

    n = torch.tensor(axis, device=head.device, dtype=head.dtype)
    v_axial = (vel * n).sum(-1).clamp_min(0.0)

    # Progress gate: nail must advance past its max-so-far by more than eps.
    depth = clamped_nail_depth(env, nail_cfg)
    advanced = (depth - self._prev_depth > eps).to(head.dtype)
    self._prev_depth = torch.maximum(self._prev_depth, depth)

    # Contact gate: fire only on the step a fresh contact begins.
    fc = sensor.compute_first_contact(dt=dt).any(-1).to(head.dtype)
    return (v_axial / v_expected) * fc * advanced


class DeliveredImpulseTerm(ManagerTermBase):
  """Maximize the OBJECT-side delivered impact impulse: pay the positive increment of the
  episode-cumulative axial impulse I_total = Σ F_axial·dt, DEPTH-GATED and normalized by I_ref.

  The maximize half of the thesis's two-sides-of-one-collision claim (the constraint half is the
  robot-side per-joint reaction impulse bounded by soft-CaT). Reads ``SubstepDeliveredImpulse``:
  episode-cumulative (monotone, so the delta-credit below can never re-pay or zero-pay across
  windows) with a per-event accrual cap (~50 ms) so a slow quasi-static press cannot out-earn
  striking (the anti-press rule of the v2 reward deep-dive). Per ``TRACKING_IMPACT_IMPULSE_
  IMPL_PLAN.md`` T3; on the Z1 (posture ≈ fixed at contact, m_eff ≈ const) this delivered-impulse
  measure needs no effective-mass / J·M⁻¹·Jᵀ computation (reserve hitting-flux for the G1).

  Delta-tracking (mirrors NailDepthDeltaTerm): credits only the positive increase in delivered
  impulse since last paid, and ONLY on steps the nail advances past its max depth by > eps.
  Augment-not-replace: a NEW positive term; impact_progress is untouched. Stateful: _credited
  (delivered credited so far), _prev_depth; reset per episode.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._credited: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
    self._prev_depth: torch.Tensor = torch.zeros(self.num_envs, device=self.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self._credited.zero_()
      self._prev_depth.zero_()
    else:
      self._credited[env_ids] = 0.0
      self._prev_depth[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    i_ref: float = 1.0,
    eps: float = 5e-4,
    nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
  ) -> torch.Tensor:
    """Returns shape (B,)."""
    acc = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, None)
    if acc is None:
      raise RuntimeError(
        "DeliveredImpulseTerm needs the SubstepDeliveredImpulse per_substep metric wired into "
        "cfg.metrics (see env_cfgs.py cat_impulse) so it stashes itself on the env."
      )
    cur = acc.delivered  # (B,) EPISODE-CUMULATIVE delivered axial impulse (monotone)
    depth = clamped_nail_depth(env, nail_cfg)
    advanced = (depth - self._prev_depth) > eps
    self._prev_depth = torch.maximum(self._prev_depth, depth)
    delta = (cur - self._credited).clamp_min(0.0)
    # Credit (and pay) only on a depth-advancing step; otherwise hold _credited so the as-yet-unpaid
    # impulse is paid once the nail actually moves. The accumulator is monotone BY CONTRACT, so the
    # torch.maximum is purely defensive: credit must never lower even if that contract ever breaks
    # (a lowered credit re-pays already-paid impulse — the farming exploit).
    self._credited = torch.where(
      advanced, torch.maximum(self._credited, cur), self._credited
    )
    return torch.where(advanced, delta, torch.zeros_like(delta)) / i_ref


class ImitationPriorTerm(ManagerTermBase):
  """Weak ante-impact tracking prior (plan T2): exp(-||p_head - p*(phi)||^2 / sigma^2) * 1[pre-contact].

  Rewards the hammer head for following the scripted SingleStrikeReference waypoint
  p*(phi), but ONLY before the first hammer->nail contact of the episode (ante-impact
  latch), so it shapes the approach/wind-up and never the impact. Position-only,
  task-space; no velocity imitation (Biemond/TAC: ill-posed through contact). Intended
  to run at a small weight that ANNEALS to 0 via mjlab's reward_curriculum, so the policy
  stays free to deviate and beat the reference (online RL, not DeepMimic).

  Stateful: a per-env "has contacted this episode" latch, reset per episode.

  KNOWN RISK (watch-item, not yet guarded): the latch bounds accumulation only once
  contact occurs. A policy that hovers near the wind-up apex without contacting keeps
  earning ~weight/step. Mitigated by (a) the anneal to 0 by step 6000, and (b) the +100
  completion bonus that terminates the episode (striking dominates hovering). The
  deviation-norm / press-watchdog training metrics surface it; add a per-episode cap or
  a phi-descent gate only if observed (augment-not-replace).
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._contacted: torch.Tensor = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self._contacted.fill_(False)
    else:
      self._contacted[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
    sigma: float = 0.05,
  ) -> torch.Tensor:
    """Returns shape (B,)."""
    robot: Entity = env.scene[robot_cfg.name]
    nail: Entity = env.scene[nail_cfg.name]
    sensor = env.scene[sensor_name]

    head_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    nail_top_w = nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

    ref = get_strike_reference(env)
    phi = ref.update(head_w, nail_top_w, env.episode_length_buf)
    p_star = ref.waypoint(phi)

    dist_sq = torch.sum((head_w - p_star) ** 2, dim=-1)
    gauss = torch.exp(-dist_sq / sigma**2)

    # Ante-impact latch: zero from the first contact of the episode onward. Level-
    # triggered on current contact state (data.found), not the first-contact rising
    # edge -- we want "has contacted at all this episode", OR-accumulated below.
    found = (sensor.data.found > 0).any(-1)
    self._contacted = self._contacted | found
    gate = (~self._contacted).to(head_w.dtype)
    return gauss * gate
