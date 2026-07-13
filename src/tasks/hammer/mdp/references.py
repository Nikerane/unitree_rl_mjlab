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
    # Apex height above nail_top. NOTE (2026-07-13): this alone no longer
    # guarantees a real wind-up — the 2026-07-06 L6 NEAR_NAIL re-solve moved
    # the reset head to z≈0.250, i.e. 2 mm BELOW nail_top+0.15=0.252, so the
    # wind-up degenerated to a 1-step nudge (adversarial review F3; the old
    # z≈0.228 reset this comment used to cite is history). The apex is now
    # additionally floored at head0_z + min_windup_clearance in _anchor(), so
    # the reference always encodes a genuine lift-then-strike regardless of
    # where the reset places the head.
    approach_height: float = 0.15,
    # Minimum apex clearance ABOVE the anchored head (2026-07-13, F3 fix):
    # apex_z = max(nail_top_z + approach_height, head0_z + min_windup_clearance).
    # 0.05 → ≥3 wind-up steps at windup_speed=0.02 and a descent long enough to
    # build genuine strike speed from the near-nail reset.
    min_windup_clearance: float = 0.05,
    # Strike target this far below nail_top (hammer FOLLOW-THROUGH).
    # 0.035 -> 0.15 (2026-07-13, adversarial review F3): the old value was
    # calibrated as "overshoot = desired drive depth" — an ENDPOINT-SERVO
    # model in which playback reached depth by holding the final waypoint and
    # pressing through. With the carrot parked 35 mm below the nail, the arm
    # decelerated into contact at ~0.46-0.49 m/s at EVERY pacing (measured
    # 15-combo scan) — a press-through, not a strike. A real strike commands
    # THROUGH the nail: with 0.15 of follow-through the carrot is still moving
    # at contact and open-loop playback strikes IN-SCRIPT at 1.37 m/s (the
    # ~1.35 m/s delta-scale design intent) and drives to success (overshoot
    # sweep 2026-07-13: 0.035->0.49 m/s post-script; 0.08->0.98; 0.10->1.19;
    # 0.12->1.34; 0.15->1.37 m/s, contact@11 of n=10+2 slack). Depth now comes
    # from impact momentum, not from the endpoint hold. The prior is
    # ante-impact gated, so below-nail waypoints are never rewarded; the
    # descent LINE through the nail is unchanged — only its parameterization
    # (phi at contact reads ~0.78 instead of ~0.93).
    overshoot: float = 0.15,
    windup_speed: float = 0.02,     # metres per control step during wind-up
    descent_speed: float = 0.05,    # metres per control step during descent (= IK max)
    # Descent phase engages only within this lateral distance of the strike
    # axis (review finding, 2026-06-10): without it, a head 14 cm off-axis at
    # nail height read phi≈0.93 ("strike nearly complete") and the monotone
    # latch made the aliasing irreversible — poison for the T2 imitation prior.
    axis_tol: float = 0.05,
    # Descent additionally requires the head to have descended at least this
    # far BELOW its own anchor projection on the strike axis (2026-07-14,
    # adversarial review R2-F2): rel_step >= n_windup alone let a head that
    # anchors ON the descent axis (the L6 near-nail reset sits INSIDE the
    # corridor at s0≈0.14) hold perfectly still, "enter descent" by clock,
    # latch its own projection (phi≈0.57), and collect the full prior forever
    # without moving. Gating on progress-below-anchor kills that free credit:
    # a stationary head never enters descent (waypoint stays at the apex, the
    # prior decays to exp(-1)≈0.37 at the L6 pose) while a genuinely striking
    # head crosses its anchor altitude immediately and gets full credit.
    # NOTE a spatial apex-REACHED latch (the reviewer's suggestion) was tried
    # and MEASURED unworkable: the physical head's closest approach to the
    # apex during the genuine scripted strike is 23-46 mm (servo lag is
    # structural to open-loop playback), overlapping the idle head's 30-50 mm
    # — and it would punish legitimate corner-cutting strikes.
    descent_margin: float = 0.01,
  ):
    self.num_envs = num_envs
    self.device = device
    self.approach_height = float(approach_height)
    self.min_windup_clearance = float(min_windup_clearance)
    self.overshoot = float(overshoot)
    self.windup_speed = float(windup_speed)
    self.descent_speed = float(descent_speed)
    self.axis_tol = float(axis_tol)
    self.descent_margin = float(descent_margin)

    self._head0 = torch.zeros(num_envs, 3, device=device)
    self._apex = torch.zeros(num_envs, 3, device=device)
    self._target = torch.zeros(num_envs, 3, device=device)
    self._n_windup = torch.ones(num_envs, device=device)
    self._step0 = torch.zeros(num_envs, device=device)  # episode step at anchor time
    self._phi = torch.zeros(num_envs, device=device)
    self._anchored = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # Anchor projection s0 on the apex->target axis (R2-F2): descent-phase
    # credit requires the head to be descent_margin BELOW this — the anchor's
    # own corridor position is never free strike progress.
    self._s0 = torch.zeros(num_envs, device=device)

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
    # Wind-up degeneracy guard (2026-07-13, adversarial review F3): the apex
    # must clear the ANCHORED head, or a near-nail reset (L6: head z≈0.250 vs
    # nail_top+0.15=0.252) collapses the wind-up to a 1-step nudge and the
    # prior degenerates to axis-centering with no lift-then-strike content.
    apex[:, 2] = torch.maximum(
      apex[:, 2], head_w[mask][:, 2] + self.min_windup_clearance
    )
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
    # Anchor projection on the strike axis (R2-F2 descent gate): where the
    # head STARTS in the corridor; only progress BELOW this earns descent phase.
    axis_m = target - apex
    len_m = axis_m.norm(dim=-1).clamp(min=1e-9)
    u_m = axis_m / len_m.unsqueeze(-1)
    self._s0[mask] = (
      ((head_w[mask] - apex) * u_m).sum(-1) / len_m
    ).clamp(0.0, 1.0)

  # -- phase ------------------------------------------------------------------

  def _phi_now(
    self,
    head_w: torch.Tensor,
    rel_step: torch.Tensor,
  ) -> torch.Tensor:
    """Instantaneous (un-latched) phase from live state — PURE function of its
    inputs and the anchors; shared by update() and preview() so the two paths
    can never drift (R2-F1 fix discipline)."""
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
    # Descent entry needs elapsed wind-up AND real progress BELOW the anchor's
    # own corridor projection (R2-F2): the clock alone let an on-axis head that
    # never moved latch its anchor projection as "strike progress" and collect
    # the full prior forever. Only descent EARNED below s0 counts.
    below_anchor = (s - self._s0) * length >= self.descent_margin
    in_descent = (rel_step >= self._n_windup) & on_axis & below_anchor
    return torch.where(in_descent, phi_d, phi_w)

  def update(
    self,
    head_w: torch.Tensor,
    nail_top_w: torch.Tensor,
    episode_step: torch.Tensor,
  ) -> torch.Tensor:
    """Advance phase from live state; re-anchors any env with episode_step == 0.

    Idempotent within a control step (safe to call from several obs terms in
    the same step). THE ONLY WRITER of the shared phase state — reward terms
    must use preview() (see there). Returns phi with shape (num_envs,).
    """
    step = episode_step.to(self._phi.dtype)
    fresh = (~self._anchored) | (episode_step == 0)
    if bool(fresh.any()):
      self._anchor(fresh, head_w, nail_top_w, step)
    rel_step = (step - self._step0).clamp(min=0.0)
    phi_now = self._phi_now(head_w, rel_step)
    self._phi = torch.maximum(self._phi, phi_now)  # monotone latch per env
    return self._phi.clone()

  def preview(
    self,
    head_w: torch.Tensor,
    episode_step: torch.Tensor,
  ) -> torch.Tensor:
    """Reward-time phase: computed from CURRENT kinematics, WRITES NOTHING.

    History (two adversarial-review rounds):
    - Round 1 (F1): the reward path called update() on kinematics one physics
      substep stale (mjlab order: reward → sim.forward → obs), committing a
      stale phase into the shared monotone latch the obs terms re-read — a
      policy-visible side channel between the imitation arm and its twin.
    - The first fix (peek(): read the phase the LAST obs pass committed) closed
      the side channel but made the reward compare the current head against the
      PREVIOUS step's waypoint (R2-F1): faithful reference-following then
      scored exp(-1)…0.74 per step while hovering scored 1.0 — a wrong-sign,
      anti-motion shaping gradient the pre-fix code never had.
    This preview computes the instantaneous phase from the reward-time head
    (restoring current-progress semantics, follower ≥ hoverer) via the same
    _phi_now() as update(), but never anchors, never latches, never writes —
    the obs pass remains the sole committer, so the side channel stays closed.
    Envs not yet anchored return their latch (zeros); in mjlab the reset-time
    obs pass always anchors before any reward computation.
    """
    step = episode_step.to(self._phi.dtype)
    rel_step = (step - self._step0).clamp(min=0.0)
    phi_now = self._phi_now(head_w, rel_step)
    phi_now = torch.where(self._anchored, phi_now, torch.zeros_like(phi_now))
    return torch.maximum(self._phi, phi_now)

  def peek(self) -> torch.Tensor:
    """Latched phase as committed by the last update() — non-mutating read."""
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
