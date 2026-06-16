"""Phase-indexed single-strike reference for the hammer head (plan stage T1).

This module provides the "given trajectory" of the tracking + impact + impulse
design (docs/research/tracking_impact_impulse_design_research.md, decisions
D2/D5): a scripted lift -> strike path that the policy receives through
observations now (``strike_phase``, ``strike_ref_error``) and through the weak
annealed imitation prior in stage T2, and that
docs/research/reward-design/playback_reference.py executes open-loop to certify
reference quality (validation Phase M) and log the I_ref strike baseline.

Design constraints honoured here:
- The wind-up segment is TIME-indexed (no contact can happen there).
- The descent segment is DISTANCE-indexed (projection onto the strike axis),
  never clock time: under contact-timing variation, time-indexed references
  make tracking errors spike for every policy (Biemond et al., TAC 2013).
- Phase is monotone-latched per env so a post-impact bounce cannot rewind it.
- Anchors (start pose, apex, strike target) are rebuilt from live state on
  every episode reset, so reset randomization moves the reference with the arm
  and the nail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class SingleStrikeReference:
  """Two-segment hammer-head reference: wind-up (head0 -> apex above the nail),
  then strike descent (apex -> just below nail top).

  Phase convention: phi in [0, 0.5] is the wind-up (step-indexed), [0.5, 1] the
  descent (strike-axis projection). All tensors are batched (num_envs, ...).
  """

  def __init__(
    self,
    num_envs: int,
    device: str | torch.device,
    # Apex height above nail_top. 0.15 keeps the apex ABOVE the near-nail reset
    # head (z≈0.228 vs nail_top 0.102): at 0.10 the apex sat 2.6 cm BELOW the
    # reset head and the wind-up segment degenerated to a 2-step nudge (review
    # finding, 2026-06-10). Wind-up stays short until the start-pose curriculum
    # (FUTURE_UPDATES #3) moves the reset away from the nail.
    approach_height: float = 0.15,
    # Strike target this far below nail_top (hammer follow-through). Playback-
    # calibrated 2026-06-10: 0.005 stalls at ~25.5 mm (target ends above the
    # driven nail head); 0.015 drives past the 30 mm threshold in 13 steps at
    # approach heights 0.06-0.10 with a single contact event.
    # 0.015 -> 0.035 (2026-06-15, real-hammer fix): the box-hammer site sat at
    # the head-box CENTRE, 25 mm above the contact face, so 0.015 of follow-
    # through reached ~30 mm. The claw-mesh site now sits ON the striking face
    # (z1_hammer_robot.xml, hammer_head_site = face centroid), so overshoot must
    # equal the desired drive depth: sweep gave 0.015->21 mm, 0.030->success,
    # 0.040->success; 0.035 lands ~35 mm with margin and stays above block top.
    overshoot: float = 0.035,
    windup_speed: float = 0.02,     # metres per control step during wind-up
    descent_speed: float = 0.05,    # metres per control step during descent (= IK max)
    # Descent phase engages only within this lateral distance of the strike
    # axis (review finding, 2026-06-10): without it, a head 14 cm off-axis at
    # nail height read phi≈0.93 ("strike nearly complete") and the monotone
    # latch made the aliasing irreversible — poison for the T2 imitation prior.
    axis_tol: float = 0.05,
  ):
    self.num_envs = num_envs
    self.device = device
    self.approach_height = float(approach_height)
    self.overshoot = float(overshoot)
    self.windup_speed = float(windup_speed)
    self.descent_speed = float(descent_speed)
    self.axis_tol = float(axis_tol)

    self._head0 = torch.zeros(num_envs, 3, device=device)
    self._apex = torch.zeros(num_envs, 3, device=device)
    self._target = torch.zeros(num_envs, 3, device=device)
    self._n_windup = torch.ones(num_envs, device=device)
    self._step0 = torch.zeros(num_envs, device=device)  # episode step at anchor time
    self._phi = torch.zeros(num_envs, device=device)
    self._anchored = torch.zeros(num_envs, dtype=torch.bool, device=device)

  # -- anchoring ------------------------------------------------------------

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Forget anchors so the next update() rebuilds them from live state."""
    if env_ids is None:
      self._anchored.fill_(False)
      self._phi.zero_()
    else:
      self._anchored[env_ids] = False
      self._phi[env_ids] = 0.0

  def _anchor(
    self,
    mask: torch.Tensor,
    head_w: torch.Tensor,
    nail_top_w: torch.Tensor,
    step: torch.Tensor,
  ) -> None:
    self._head0[mask] = head_w[mask]
    apex = nail_top_w[mask].clone()
    apex[:, 2] += self.approach_height
    self._apex[mask] = apex
    target = nail_top_w[mask].clone()
    target[:, 2] -= self.overshoot
    self._target[mask] = target
    dist = (apex - head_w[mask]).norm(dim=-1)
    self._n_windup[mask] = (dist / self.windup_speed).ceil().clamp(min=1.0)
    # Phase counts steps SINCE anchoring, so a mid-episode re-anchor (explicit
    # reset(); future cyclic re-arm) restarts the wind-up cleanly.
    self._step0[mask] = step[mask]
    self._phi[mask] = 0.0
    self._anchored[mask] = True

  # -- phase ------------------------------------------------------------------

  def update(
    self,
    head_w: torch.Tensor,
    nail_top_w: torch.Tensor,
    episode_step: torch.Tensor,
  ) -> torch.Tensor:
    """Advance phase from live state; re-anchors any env with episode_step == 0.

    Idempotent within a control step (safe to call from several obs/reward
    terms in the same step). Returns phi with shape (num_envs,).
    """
    step = episode_step.to(self._phi.dtype)
    fresh = (~self._anchored) | (episode_step == 0)
    if bool(fresh.any()):
      self._anchor(fresh, head_w, nail_top_w, step)
    rel_step = (step - self._step0).clamp(min=0.0)
    # Wind-up: step-indexed first half of phase.
    phi_w = 0.5 * (rel_step / self._n_windup).clamp(max=1.0)
    # Descent: projection of the head onto the apex->target strike axis,
    # gated on actually being NEAR that axis — lateral wandering must not
    # read as strike progress (and must not latch as such).
    axis = self._target - self._apex
    length = axis.norm(dim=-1).clamp(min=1e-9)
    u = axis / length.unsqueeze(-1)
    s = (((head_w - self._apex) * u).sum(-1) / length).clamp(0.0, 1.0)
    perp = (head_w - self._apex) - (s * length).unsqueeze(-1) * u
    on_axis = perp.norm(dim=-1) <= self.axis_tol
    phi_d = 0.5 + 0.5 * s
    in_descent = (rel_step >= self._n_windup) & on_axis
    phi_now = torch.where(in_descent, phi_d, phi_w)
    self._phi = torch.maximum(self._phi, phi_now)  # monotone latch per env
    return self._phi.clone()

  # -- waypoints --------------------------------------------------------------

  def waypoint(self, phi: torch.Tensor) -> torch.Tensor:
    """Reference head position at phase phi. Shape (num_envs, 3)."""
    t_w = (phi / 0.5).clamp(0.0, 1.0).unsqueeze(-1)
    p_w = self._head0 + t_w * (self._apex - self._head0)
    t_d = ((phi - 0.5) / 0.5).clamp(0.0, 1.0).unsqueeze(-1)
    p_d = self._apex + t_d * (self._target - self._apex)
    return torch.where((phi >= 0.5).unsqueeze(-1), p_d, p_w)

  # -- scripted playback --------------------------------------------------------

  def playback_length(self) -> int:
    """Number of scripted steps to traverse wind-up + descent (max over envs)."""
    axis_len = (self._target - self._apex).norm(dim=-1)
    n_descent = (axis_len / self.descent_speed).ceil()
    return int((self._n_windup + n_descent).max().item())

  def playback_target(self, k: int) -> torch.Tensor:
    """Scripted waypoint at control step k (k >= 1): wind-up at windup_speed,
    then descent at descent_speed; clamps at the strike target."""
    k_t = torch.full((self.num_envs,), float(k), device=self.device)
    t_w = (k_t / self._n_windup).clamp(max=1.0).unsqueeze(-1)
    p_w = self._head0 + t_w * (self._apex - self._head0)
    axis = self._target - self._apex
    length = axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    steps_d = (k_t - self._n_windup).clamp(min=0.0).unsqueeze(-1)
    frac = (steps_d * self.descent_speed / length).clamp(max=1.0)
    # Lerp form (not apex + frac*axis) so the clamped endpoint equals the
    # target bit-exactly (fuzz finding: the additive form was 1 ulp off).
    p_d = self._apex * (1.0 - frac) + self._target * frac
    in_descent = (k_t > self._n_windup).unsqueeze(-1)
    return torch.where(in_descent, p_d, p_w)


def get_strike_reference(env: "ManagerBasedRlEnv", **kwargs) -> SingleStrikeReference:
  """Lazily attach one shared SingleStrikeReference to the env instance.

  Obs terms (and later the T2 imitation reward) share this single instance so
  phase state is computed once per env, idempotently.
  """
  ref = getattr(env, "_strike_reference", None)
  if ref is None or ref.num_envs != env.num_envs:
    ref = SingleStrikeReference(env.num_envs, env.device, **kwargs)
    env._strike_reference = ref
  elif kwargs:
    # Guard against silently inheriting another caller's parameters: the
    # first construction wins, later callers must agree (review finding).
    for key, value in kwargs.items():
      cached = getattr(ref, key)
      if cached != value:
        raise ValueError(
          f"get_strike_reference: cached instance has {key}={cached}, "
          f"caller requested {value}. Construct SingleStrikeReference "
          "directly if you need different parameters."
        )
  return ref
