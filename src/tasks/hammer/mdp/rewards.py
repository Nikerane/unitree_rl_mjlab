"""Reward terms for the Z1 hammer-nail task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
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
  nail: Entity = env.scene[nail_cfg.name]
  depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
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

  # WHY THIS EXISTS:
  # At qpos=0 (episode reset), MuJoCo's constraint solver has gravity and the
  # joint-limit spring active simultaneously. The solver is inherently compliant
  # (all constraints are soft springs via solref) so it finds equilibrium at
  # ~3.5 mm rather than exactly 0. This drift happens every episode in the first
  # few physics steps with no arm contact.
  #
  # Without this offset, _max_depth starts at 0 and the settling looks like real
  # progress: delta = 0.0035 → reward = 2000 × 0.0035 = 7.0 per episode for free.
  # That free reward is consistent but it dilutes the striking signal.
  #
  # Setting _max_depth to 0.004 (just above 3.5 mm) creates a dead zone that
  # absorbs the settling. Reward only fires when the arm drives the nail past 4 mm,
  # which requires real hammer contact. Training impact is negligible: a single
  # real strike drives ~66 mm, so the 4 mm threshold is cleared on first contact.
  _SETTLE_OFFSET: float = 0.004  # 4 mm dead zone above gravity-settling artefact

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
    nail: Entity = env.scene[nail_cfg.name]
    depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
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
    nail: Entity = env.scene[nail_cfg.name]
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
    depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
    advanced = (depth - self._prev_depth > eps).to(head.dtype)
    self._prev_depth = torch.maximum(self._prev_depth, depth)

    # Contact gate: fire only on the step a fresh contact begins.
    fc = sensor.compute_first_contact(dt=dt).any(-1).to(head.dtype)
    return (v_axial / v_expected) * fc * advanced
