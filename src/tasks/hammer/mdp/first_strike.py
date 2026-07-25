"""Shared 500 Hz first-strike event tracker for event-correct rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH, NAIL_SUCCESS_THRESHOLD

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


REASON_NONE = 0
REASON_SUCCESS = 1
REASON_WINDOW = 2

_STATE_UNARMED = 0
_STATE_ARMED = 1
_STATE_ACTIVE = 2
_STATE_FINALIZED = 3

_ENV_FIRST_STRIKE_ATTR = "_hammer_first_strike"


class FirstStrikeEventTracker(ManagerTermBase):
  """Track one immutable first-strike snapshot per environment.

  After reset, two consecutive off-contact physics substeps are required before
  the first rising edge can start an event.  The event then spans wall time,
  including contact gaps, and finalizes inclusively on success or on its window
  boundary.  A finalized environment cannot rearm until reset.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    params = cfg.params
    self._contact_sensor = env.scene[params["contact_sensor_name"]]
    self._impulse_sensor = env.scene[params["impulse_sensor_name"]]
    self._robot = env.scene[params["robot_cfg"].name]
    self._nail = env.scene[params["nail_cfg"].name]
    self._site_ids = params["robot_cfg"].site_ids
    self._joint_ids = params["nail_cfg"].joint_ids
    self._axis = torch.tensor(
      params.get("axis", (0.0, 0.0, -1.0)), device=env.device, dtype=torch.float32
    )
    self._window = int(params.get("window_substeps", 25))
    if self._window < 1:
      raise ValueError(f"window_substeps must be >= 1, got {self._window}")
    self._progress_eps = float(params.get("progress_eps", 5e-4))

    self._state = torch.full(
      (env.num_envs,), _STATE_UNARMED, dtype=torch.long, device=env.device
    )
    self._off_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._event_age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._prev_head = torch.zeros(env.num_envs, 3, device=env.device)
    self._prev_depth = torch.zeros(env.num_envs, device=env.device)

    self._finalized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._productive = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._v_precontact = torch.zeros(env.num_envs, device=env.device)
    self._delivered = torch.zeros(env.num_envs, device=env.device)
    self._peak_depth = torch.zeros(env.num_envs, device=env.device)
    self._depth_at_contact = torch.zeros(env.num_envs, device=env.device)
    self._reason = torch.full(
      (env.num_envs,), REASON_NONE, dtype=torch.long, device=env.device
    )
    setattr(env, _ENV_FIRST_STRIKE_ATTR, self)

  @property
  def finalized(self) -> torch.Tensor:
    return self._finalized

  @property
  def productive(self) -> torch.Tensor:
    return self._productive

  @property
  def v_precontact(self) -> torch.Tensor:
    return self._v_precontact

  @property
  def delivered(self) -> torch.Tensor:
    return self._delivered

  @property
  def peak_depth(self) -> torch.Tensor:
    return self._peak_depth

  @property
  def depth_at_contact(self) -> torch.Tensor:
    return self._depth_at_contact

  @property
  def reason(self) -> torch.Tensor:
    return self._reason

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self._state[idx] = _STATE_UNARMED
    self._off_streak[idx] = 0
    self._event_age[idx] = 0
    self._prev_head[idx] = 0.0
    self._prev_depth[idx] = 0.0
    self._finalized[idx] = False
    self._productive[idx] = False
    self._v_precontact[idx] = 0.0
    self._delivered[idx] = 0.0
    self._peak_depth[idx] = 0.0
    self._depth_at_contact[idx] = 0.0
    self._reason[idx] = REASON_NONE
    return None

  def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
    del params
    head = self._robot.data.site_pos_w[:, self._site_ids].squeeze(1)
    depth = self._nail.data.joint_pos[:, self._joint_ids].squeeze(1)
    depth = depth.clamp(0.0, NAIL_GOAL_DEPTH)
    in_contact = (self._contact_sensor.data.found > 0).any(dim=-1)

    force = self._impulse_sensor.data.force
    axis = self._axis.to(dtype=force.dtype)
    force_axial = (force * axis).sum(dim=-1).sum(dim=-1).clamp_min(0.0)

    unarmed = self._state == _STATE_UNARMED
    armed = self._state == _STATE_ARMED
    active = self._state == _STATE_ACTIVE

    # Keep the immediately previous free-flight sample.  UNARMED needs two
    # consecutive such samples before a contact can be accepted.
    off_unarmed = unarmed & ~in_contact
    self._off_streak = torch.where(
      unarmed,
      torch.where(
        in_contact,
        torch.zeros_like(self._off_streak),
        self._off_streak + 1,
      ),
      self._off_streak,
    )
    cache_previous = ~in_contact & (unarmed | armed)
    self._prev_head = torch.where(cache_previous[:, None], head, self._prev_head)
    self._prev_depth = torch.where(cache_previous, depth, self._prev_depth)
    newly_armed = off_unarmed & (self._off_streak >= 2)
    self._state = torch.where(
      newly_armed, torch.full_like(self._state, _STATE_ARMED), self._state
    )

    # The rising-edge position is the pre-force onset sample in mjlab's real
    # callback order, so use onset minus the immediately previous off-contact
    # position rather than the one-substep-stale off/off finite difference.
    onset = armed & in_contact
    velocity = ((head - self._prev_head) / env.physics_dt * self._axis).sum(dim=-1)
    velocity = velocity.clamp_min(0.0)
    self._v_precontact = torch.where(onset, velocity, self._v_precontact)
    self._depth_at_contact = torch.where(onset, self._prev_depth, self._depth_at_contact)
    onset_peak = torch.maximum(self._prev_depth, depth)
    self._peak_depth = torch.where(onset, onset_peak, self._peak_depth)
    self._state = torch.where(
      onset, torch.full_like(self._state, _STATE_ACTIVE), self._state
    )

    # The onset substep is age one.  Every subsequent physics substep advances
    # age, even across raw-contact gaps.  Force is accumulated only while the
    # contact sensor is on; depth is peak-tracked throughout the window.
    active_now = active | onset
    self._event_age = self._event_age + active_now.long()
    contribution = force_axial * env.physics_dt
    contribution = contribution * (active_now & in_contact).to(contribution.dtype)
    self._delivered = self._delivered + contribution
    self._peak_depth = torch.where(
      active_now, torch.maximum(self._peak_depth, depth), self._peak_depth
    )

    success = active_now & (depth >= NAIL_SUCCESS_THRESHOLD)
    window_done = active_now & (self._event_age >= self._window)
    finalize = success | window_done
    self._productive = torch.where(
      finalize,
      (self._peak_depth - self._depth_at_contact) > self._progress_eps,
      self._productive,
    )
    final_reason = torch.where(
      success,
      torch.full_like(self._reason, REASON_SUCCESS),
      torch.full_like(self._reason, REASON_WINDOW),
    )
    self._reason = torch.where(finalize, final_reason, self._reason)
    self._finalized = self._finalized | finalize
    self._state = torch.where(
      finalize, torch.full_like(self._state, _STATE_FINALIZED), self._state
    )
    return self._finalized.to(dtype=head.dtype)
