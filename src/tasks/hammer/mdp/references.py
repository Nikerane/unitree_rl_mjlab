"""Phase-indexed direct single-strike reference for the hammer head.

This module provides the "given trajectory" of the tracking + impact + impulse
design (docs/research/tracking_impact_impulse_design_research.md, decisions
D2/D5): a scripted reset-head -> follow-through path that the policy receives
through observations now (``strike_phase``, ``strike_ref_error``) and through
the weak annealed imitation prior in stage T2, and that
docs/research/reward-design/playback_reference.py executes open-loop to certify
reference quality (validation Phase M) and log the I_ref strike baseline.

Design constraints honoured here:
- The single segment is DISTANCE-indexed (projection onto the strike axis),
  never clock time: stationary heads cannot earn reference progress and
  contact-timing variation cannot create tracking-error spikes.
- Phase is monotone-latched per env so a post-impact bounce cannot rewind it.
- Anchors (start pose, strike target) are rebuilt from live state on
  every episode reset, so reset randomization moves the reference with the arm
  and the nail.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class SingleStrikeReference:
  """One direct hammer-head segment: reset head -> below-nail follow-through.

  ``phi`` is the monotone-latched spatial projection onto that finite segment.
  All tensors are batched ``(num_envs, ...)``.
  """

  def __init__(
    self,
    num_envs: int,
    device: str | torch.device,
    # Follow-through distance.  ``vertical`` places the target this far below
    # nail_top; ``strike_axis`` extends the reset-head -> nail ray by this
    # distance so an offset-start straight guide crosses the nail exactly.
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
    descent_speed: float = 0.05,    # metres per control step during descent (= IK max)
    # Phase advances only within this lateral distance of the strike
    # axis (review finding, 2026-06-10): without it, a head 14 cm off-axis at
    # nail height read phi≈0.93 ("strike nearly complete") and the monotone
    # latch made the aliasing irreversible — poison for the T2 imitation prior.
    axis_tol: float = 0.05,
    *,
    horizontal_detour_m: float = 0.0,
    followthrough_mode: Literal["vertical", "strike_axis"] = "vertical",
  ):
    self.num_envs = num_envs
    self.device = device
    self.overshoot = float(overshoot)
    self.descent_speed = float(descent_speed)
    self.axis_tol = float(axis_tol)
    self.horizontal_detour_m = float(horizontal_detour_m)
    if followthrough_mode not in ("vertical", "strike_axis"):
      raise ValueError(
        "followthrough_mode must be either 'vertical' or 'strike_axis'"
      )
    self.followthrough_mode = followthrough_mode

    self._head0 = torch.zeros(num_envs, 3, device=device)
    self._target = torch.zeros(num_envs, 3, device=device)
    self._phi = torch.zeros(num_envs, device=device)
    self._anchored = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self._route_sign = torch.zeros(num_envs, dtype=torch.int8, device=device)

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
  ) -> None:
    self._head0[mask] = head_w[mask]
    nail = nail_top_w[mask]
    head = head_w[mask]
    if self.followthrough_mode == "strike_axis":
      # Extend the realized reset-head -> nail ray beyond contact.  With an
      # offset reset, a vertically lowered target would make the straight
      # segment pass beside the nail; the ray extension keeps the nail exactly
      # on the one-segment guide while retaining nonzero motion through it.
      incoming = nail - head
      length = incoming.norm(dim=-1, keepdim=True)
      direction = incoming / length.clamp_min(1e-18)
      target = nail + self.overshoot * direction
      target = torch.where(length > 1e-18, target, head)
    else:
      target = nail.clone()
      target[:, 2] -= self.overshoot
      # A low realized reset must never turn the direct strike into an upward
      # command. Preserve the nail-derived x/y follow-through while clamping its
      # height to the frozen reset head.
      target[:, 2] = torch.minimum(target[:, 2], head[:, 2])
    self._target[mask] = target
    self._phi[mask] = 0.0
    self._anchored[mask] = True

  # -- phase ------------------------------------------------------------------

  def _phi_now(
    self,
    head_w: torch.Tensor,
  ) -> torch.Tensor:
    """Instantaneous (un-latched) phase from live state — PURE function of its
    inputs and the anchors; shared by update() and preview() so the two paths
    can never drift (R2-F1 fix discipline)."""
    # Projection onto the finite reset-head -> follow-through segment.  The
    # closest-point residual enforces axis_tol even beyond either endpoint.
    axis = self._target - self._head0
    length_sq = (axis * axis).sum(-1)
    safe_length_sq = length_sq.clamp_min(1e-18)
    raw_s = ((head_w - self._head0) * axis).sum(-1) / safe_length_sq
    s = raw_s.clamp(0.0, 1.0)
    closest = self._head0 + s.unsqueeze(-1) * axis
    on_segment = (head_w - closest).norm(dim=-1) <= self.axis_tol
    nonzero = length_sq > 1e-18
    return torch.where(on_segment & nonzero, s, torch.zeros_like(s))

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
    fresh = (~self._anchored) | (episode_step == 0)
    if bool(fresh.any()):
      self._anchor(fresh, head_w, nail_top_w)
    phi_now = self._phi_now(head_w)
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
    del episode_step
    phi_now = self._phi_now(head_w)
    phi_now = torch.where(self._anchored, phi_now, torch.zeros_like(phi_now))
    return torch.maximum(self._phi, phi_now)

  def peek(self) -> torch.Tensor:
    """Latched phase as committed by the last update() — non-mutating read."""
    return self._phi.clone()

  def set_route_signs(self, signs: torch.Tensor) -> None:
    """Force one route sign per environment for deterministic playback."""
    valid = (signs == -1) | (signs == 0) | (signs == 1)
    if signs.shape != (self.num_envs,) or not bool(valid.all()):
      raise ValueError(
        "route signs must have shape (num_envs,) with values in {-1, 0, +1}"
      )
    self._route_sign.copy_(signs.to(device=self.device, dtype=self._route_sign.dtype))

  def sample_route_signs(self, env_ids: torch.Tensor) -> None:
    """Sample R-/R0/R+ only for environments entering a new episode."""
    sampled = (
      torch.randint(0, 3, (len(env_ids),), device=self.device, dtype=torch.int8)
      - 1
    )
    self._route_sign[env_ids] = sampled

  def route_signs(self) -> torch.Tensor:
    """Return a defensive copy of the reset-stable route assignments."""
    return self._route_sign.clone()

  # -- waypoints --------------------------------------------------------------

  def waypoint(self, phi: torch.Tensor) -> torch.Tensor:
    """Reference head position at phase phi. Shape (num_envs, 3)."""
    t = phi.clamp(0.0, 1.0).unsqueeze(-1)
    direct = self._head0 * (1.0 - t) + self._target * t
    if self.horizontal_detour_m == 0.0:
      return direct
    phase = t.squeeze(-1)
    active = (phase > 0.0) & (phase < 0.5)
    amplitude = torch.where(
      active,
      self.horizontal_detour_m * torch.sin(2.0 * math.pi * phase).square(),
      torch.zeros_like(phase),
    )
    routed = direct.clone()
    routed[:, 0] += self._route_sign.to(dtype=routed.dtype) * amplitude
    return routed

  def reference_polyline(
    self,
    env_id: int = 0,
    num_points: int = 65,
  ) -> torch.Tensor:
    """Sample one environment's commanded route without changing reference state.

    The returned ``(num_points, 3)`` tensor is in world coordinates and includes
    both endpoints.  It is intended for visualization and diagnostics; sampling
    never advances phase, changes route assignment, or re-anchors the reference.
    """
    if not 0 <= env_id < self.num_envs:
      raise IndexError(f"env_id {env_id} is outside [0, {self.num_envs})")
    if num_points < 2:
      raise ValueError("num_points must be at least 2")
    if not bool(self._anchored[env_id]):
      raise RuntimeError("reference must be anchored before sampling its polyline")

    phases = torch.linspace(
      0.0,
      1.0,
      num_points,
      device=self._head0.device,
      dtype=self._head0.dtype,
    )
    t = phases.unsqueeze(-1)
    head0 = self._head0[env_id]
    target = self._target[env_id]
    points = head0 * (1.0 - t) + target * t
    if self.horizontal_detour_m != 0.0:
      active = (phases > 0.0) & (phases < 0.5)
      amplitude = torch.where(
        active,
        self.horizontal_detour_m * torch.sin(2.0 * math.pi * phases).square(),
        torch.zeros_like(phases),
      )
      points[:, 0] += self._route_sign[env_id].to(points.dtype) * amplitude
    return points

  # -- scripted playback --------------------------------------------------------

  def playback_length(self) -> int:
    """Number of scripted steps to traverse the segment (max over envs)."""
    length = (self._target - self._head0).norm(dim=-1)
    return int((length / self.descent_speed).ceil().max().item())

  def playback_target(self, k: int) -> torch.Tensor:
    """Scripted waypoint at control step k, advancing at descent_speed."""
    k_t = torch.full((self.num_envs,), float(k), device=self.device)
    axis = self._target - self._head0
    length = axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    frac = (k_t.unsqueeze(-1) * self.descent_speed / length).clamp(0.0, 1.0)
    # Lerp form (not head0 + frac*axis) so the clamped endpoint equals the
    # target bit-exactly (fuzz finding: the additive form was 1 ulp off).
    if self.horizontal_detour_m == 0.0:
      return self._head0 * (1.0 - frac) + self._target * frac
    return self.waypoint(frac.squeeze(-1))


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


def sample_strike_route_signs(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  horizontal_detour_m: float,
  followthrough_mode: Literal["vertical", "strike_axis"] = "vertical",
  fixed_route_sign: int | None = None,
) -> None:
  """Reset event that samples one cached strike route per requested env.

  A fixed specialist still consumes the ordinary route-sampler draw before its
  selected sign is overwritten, keeping subsequent reset randomization matched
  with the mixed-route treatments.
  """
  if fixed_route_sign is not None and fixed_route_sign not in (-1, 0, 1):
    raise ValueError("fixed_route_sign must be one of {-1, 0, +1}")
  ref = get_strike_reference(
    env,
    horizontal_detour_m=horizontal_detour_m,
    followthrough_mode=followthrough_mode,
  )
  ref.reset(env_ids)
  ref.sample_route_signs(env_ids)
  if fixed_route_sign is not None:
    signs = ref.route_signs()
    signs[env_ids] = fixed_route_sign
    ref.set_route_signs(signs)


def reset_joints_to_strike_route_starts(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  route_joint_positions: tuple[tuple[float, ...], ...],
  asset_cfg,
) -> None:
  """Write the cached R-/R0/R+ arm pose for each resetting environment.

  The route sampler must run first in the same reset event sequence.  Keeping
  sampling and state writing separate preserves the ordinary reset manager
  ordering while making partial resets select only their already-cached rows.
  """
  ref = getattr(env, "_strike_reference", None)
  if not isinstance(ref, SingleStrikeReference):
    raise RuntimeError("strike route signs must be sampled before joint reset")

  joint_ids = asset_cfg.joint_ids
  poses = torch.as_tensor(
    route_joint_positions,
    dtype=torch.float32,
    device=env.device,
  )
  expected_shape = (3, len(joint_ids))
  if poses.shape != expected_shape or not bool(torch.isfinite(poses).all()):
    raise ValueError(
      "route_joint_positions must be a finite "
      f"{expected_shape} R-/R0/R+ table"
    )

  signs = ref.route_signs()[env_ids].to(dtype=torch.long)
  valid = (signs >= -1) & (signs <= 1)
  if not bool(valid.all()):
    raise ValueError("cached strike route signs must lie in {-1, 0, +1}")
  joint_pos = poses[signs + 1]
  joint_vel = torch.zeros_like(joint_pos)
  env.scene[asset_cfg.name].write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    joint_ids=joint_ids,
    env_ids=env_ids,
  )
