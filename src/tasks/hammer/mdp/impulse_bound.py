"""Substep impact-impulse accumulators for the Z1 hammer task (robot-side Λ_j + object-side I).

The thesis bounds the **robot-side per-joint reaction impulse** absorbed by each harmonic drive
at impact — the opposite side of the same collision from the object-side delivered impulse on the
nail (the maximize objective; see ``rewards.DeliveredImpulseTerm``). The quantity, per
``IMPULSE_CAT_IMPL_PLAN.md`` §1 and ``tracking_impact_impulse_design_research.md`` SQ3:

    Λ_j = Σ_{substeps s in a CONTACT window}  |qfrc_constraint_{s,j}|  · physics_dt   [N·m·s]

accumulated at the **500 Hz substep rate** (never the 50 Hz control rate, which aliases the brief
impact), over a **contact-anchored window** (first→last contact substep, NOT the 20 ms control grid).

Correctness points (IMPULSE_CAT_IMPL_PLAN.md §1.1/§4, tightened after the 2026-07 deep review):

  * CONTACT-ANCHORED window — anchored to the ``hammer_nail_contact`` sensor, not the control grid,
    so a policy cannot split one impact across two 20 ms windows to stay under threshold. The sensor
    reads a live view into ``data.sensordata`` (refreshed by ``sim.step()`` every substep).
  * WELD/FRICTION immunity — ``qfrc_constraint`` sums ALL constraints (weld + dof friction + limits
    + contacts); the arm joints carry a constant ±frictionloss while moving. The contact-sensor gate
    zeros every OFF-contact substep; ``subtract_baseline`` additionally removes the frozen
    pre-contact reaction baseline during the window. The C0 quantity gate
    (``derive_impulse_thresholds.py``) measures the residual contamination against the object-side
    ground truth and certifies the SHIPPED accumulators, not a parallel reimplementation.
  * PER-EVENT PULSE (constraint side) — a completed window's Λ_j is visible to the 50 Hz constraint
    read for the REMAINDER of the control step it closed in (max-combined if several windows close
    within one step), then cleared at the next step boundary. An in-progress window is visible via
    its running sum. This is CaT's per-violation semantics: δ pressure exactly during and at the
    close of a violating window — no episode-long persistence (which the review showed causes
    duration-dependent punishment and EMA-normalizer saturation), and no latched value a later
    gentle tap could erase.
  * EPISODE-CUMULATIVE (object side) — the delivered impulse I_total is monotone non-decreasing
    (Σ over all contact substeps of the episode), so the delta-crediting reward can never re-pay
    or zero-pay across windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.hammer.mdp.velocity_bound import _ARM_CFG

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Per-joint contact-reaction impulse limit (N·m·s). C0 PLACEHOLDER — enforcement must pass the
# REAL per-joint caps from a fresh derive_impulse_thresholds.py run (τ_rated × 2 Harmonic-Drive
# Repeated-Peak × measured contact window) explicitly via cfg. This constant is NEVER a silent
# default: joint_impulse_excess requires ``limit`` and CatSoftHook requires ``imp_limit``.
Z1_JOINT_IMPULSE_LIMIT: float = 0.1

# Where the accumulators stash themselves on the env so full-step consumers (joint_impulse_excess,
# DeliveredImpulseTerm) can read the substep-accumulated buffers.
_ENV_SUBSTEP_IMPULSE_ATTR = "_hammer_substep_impulse"
_ENV_SUBSTEP_DELIVERED_ATTR = "_hammer_substep_delivered"


def _joint_count(joint_pos: torch.Tensor, joint_ids) -> int:
  """Number of selected joints, safe for both list ids and the slice(None) optimization
  SceneEntityCfg.resolve applies when the requested joints cover the whole entity."""
  return joint_pos[:1, joint_ids].shape[-1]


class SubstepImpulseAccumulator(ManagerTermBase):
  """per_substep MetricsTerm: contact-anchored Σ |qfrc_constraint_j| · dt per arm joint,
  exposed with PER-EVENT PULSE semantics (see module docstring).

  Per env, per substep: a window OPENS on the rising contact edge (running sum reset),
  ACCUMULATES the rectified per-joint reaction while in contact, and on the falling edge
  MAX-combines the completed window into ``_pulse`` — which is cleared at the first substep of
  each control step, so a completed window is visible to exactly one 50 Hz constraint read.
  ``impulse`` = max(pulse, running): an in-progress window is always visible too (including a
  strike that terminates the episode while still in contact).

  ``_last_off_qfrc`` (the most recent off-contact reaction) doubles as the frozen pre-contact
  baseline during a window — it only updates while OFF contact.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    p = cfg.params
    self._subtract_baseline: bool = bool(p.get("subtract_baseline", False))
    arm = SceneEntityCfg("robot", joint_names=p.get("robot_cfg", _ARM_CFG).joint_names)
    arm.resolve(env.scene)
    self._joint_ids = arm.joint_ids
    self._robot = env.scene["robot"]
    self._sensor = env.scene[p.get("sensor_name", "hammer_nail_contact")]
    self._dec = int(env.cfg.decimation)
    self._i = 0
    J = _joint_count(self._robot.data.joint_pos, self._joint_ids)
    self._pulse = torch.zeros(env.num_envs, J, device=env.device)
    self._running = torch.zeros(env.num_envs, J, device=env.device)
    self._last_off_qfrc = torch.zeros(env.num_envs, J, device=env.device)
    self._in_contact_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Episode-peak worst-joint Λ — the LOGGED metric (cfg reduce="last"): the C0 log-only run must
    # be able to observe learned Λ against J_limit; a substep-mean of the transient pulse dilutes
    # it ~100× and is phase-dependent.
    self._episode_peak = torch.zeros(env.num_envs, device=env.device)
    # Per-JOINT episode-peak Λ (Task 6 observability): J_limit differs 2× across joints (joint2
    # τ_rated=60 vs 30 elsewhere), so the worst-joint scalar above can't be compared per-column to
    # the cap vector. Read by the module-level joint_impulse_peak reader below (one TB key/joint).
    self._episode_peak_perjoint = torch.zeros(env.num_envs, J, device=env.device)
    setattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, self)

  @property
  def impulse(self) -> torch.Tensor:
    """Per-joint contact-window impulse Λ_j the constraint reads, shape (B, J):
    max(pulse of windows closed this control step, running sum of an open window)."""
    return torch.maximum(self._pulse, self._running)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self._pulse[idx] = 0.0
    self._running[idx] = 0.0
    self._last_off_qfrc[idx] = 0.0
    self._in_contact_prev[idx] = False
    self._episode_peak[idx] = 0.0
    self._episode_peak_perjoint[idx] = 0.0
    return None

  def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
    # **params absorbs the MetricsTermCfg.params the manager re-passes each substep.
    # Branchless masked math throughout: no data-dependent `.any()` host syncs in the 500 Hz loop.
    if self._i % self._dec == 0:  # first substep of a control step: last step's pulse was consumed
      self._pulse.zero_()
    qfrc = self._robot.data._joint_dof_field("qfrc_constraint")[:, self._joint_ids]  # (B,J)
    in_contact = (self._sensor.data.found > 0).any(dim=-1)  # (B,)
    in_c = in_contact[:, None]
    falling = (~in_contact & self._in_contact_prev)[:, None]

    # ACCUMULATE the rectified per-joint reaction while in contact (baseline frozen off-contact).
    # No rising-edge reset is needed: _running is provably 0 at every window open (the falling edge
    # below zeroes it, episode reset zeroes it, and accumulation is contact-masked).
    signal = qfrc - self._last_off_qfrc if self._subtract_baseline else qfrc
    self._running = self._running + signal.abs() * (env.physics_dt) * in_c
    # CLOSE: max-combine the completed window into the pulse; zero the running sum.
    self._pulse = torch.where(falling, torch.maximum(self._pulse, self._running), self._pulse)
    self._running = torch.where(falling, torch.zeros_like(self._running), self._running)
    # Track the rolling pre-contact baseline (updates only while OFF contact).
    self._last_off_qfrc = torch.where(in_c, self._last_off_qfrc, qfrc)

    self._in_contact_prev = in_contact
    self._i += 1
    # Logged metric: EPISODE-PEAK worst-joint Λ (pair with MetricsTermCfg reduce="last").
    imp = torch.maximum(self._pulse, self._running).amax(dim=1)  # (B,)
    torch.maximum(self._episode_peak, imp, out=self._episode_peak)
    # Per-joint episode-peak Λ (Task 6): same monotone-max update, one column per arm joint.
    torch.maximum(
      self._episode_peak_perjoint, torch.maximum(self._pulse, self._running), out=self._episode_peak_perjoint
    )
    return self._episode_peak


class SubstepDeliveredImpulse(ManagerTermBase):
  """per_substep MetricsTerm: EPISODE-CUMULATIVE object-side delivered axial impulse
  I_total = Σ_episode F_axial · dt over all contact substeps.

  The OTHER side of the same collision from ``SubstepImpulseAccumulator``: the impulse the hammer
  delivers to the nail (the maximize OBJECTIVE's signal; see ``rewards.DeliveredImpulseTerm``) and
  the weld/friction-IMMUNE ground truth for the C0 quantity gate (the contact sensor sees ONLY the
  hammer↔nail contact). Monotone non-decreasing within an episode, so the delta-crediting reward
  can never re-pay or zero-pay across windows. Reads a ``reduce="netforce"`` contact sensor (net
  contact wrench per primary, world frame); only the downward (delivering) axial component counts.

  PRESS-FARMING CAP: each contact event accrues for at most ``event_window_substeps`` substeps
  (default 25 = 50 ms, above the measured p95 strike window of ~20 substeps). Without the cap a
  slow quasi-static press (long duration × moderate force) accrues unbounded ∫F·dt and out-earns
  striking — the anti-press failure mode the v2 reward deep-dive forbids. Strikes are unaffected;
  presses pay only their first 50 ms.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    p = cfg.params
    axis = p.get("axis", (0.0, 0.0, -1.0))
    self._axis = torch.tensor(axis, device=env.device, dtype=torch.float32)
    self._sensor = env.scene[p.get("sensor_name", "hammer_nail_impulse")]
    self._window = int(p.get("event_window_substeps", 25))
    self._total = torch.zeros(env.num_envs, device=env.device)
    self._event_age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._in_contact_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, self)

  @property
  def delivered(self) -> torch.Tensor:
    """Cumulative delivered axial impulse this episode, shape (B,). Monotone non-decreasing."""
    return self._total

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self._total[idx] = 0.0
    self._event_age[idx] = 0
    self._in_contact_prev[idx] = False
    return None

  def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
    f = self._sensor.data.force  # (B, N, 3) world-frame net contact force per primary
    f_axial = (f * self._axis).sum(dim=-1).sum(dim=-1).clamp_min(0.0)  # (B,) net downward force
    in_contact = (self._sensor.data.found > 0).any(dim=-1)  # (B,)
    # Event age: 0 on the rising edge, +1 per in-contact substep (branchless).
    rising = in_contact & ~self._in_contact_prev
    self._event_age = torch.where(rising, torch.zeros_like(self._event_age), self._event_age)
    payable = in_contact & (self._event_age < self._window)
    self._total += f_axial * env.physics_dt * payable  # in-place: keep tensor identity
    self._event_age = self._event_age + in_contact.long()
    self._in_contact_prev = in_contact
    return self._total  # (B,) cumulative — pair with MetricsTermCfg reduce="last"


def joint_impulse_peak(env: "ManagerBasedRlEnv", joint: int) -> torch.Tensor:
  """Full-step metric: episode-peak Λ for ONE arm joint, from the stashed accumulator.
  reduce="last" on a FULL-STEP term logs the exact compute-time value of the monotone buffer —
  the authoritative per-joint episode peak (the per-substep worst-joint scalar is substep-mean
  diluted in the terminal control step)."""
  acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
  if acc is None:
    raise RuntimeError("joint_impulse_peak requires the SubstepImpulseAccumulator metric (cat_impulse).")
  return acc._episode_peak_perjoint[:, joint]


class CatDeltaPeak(ManagerTermBase):
  """Full-step metric: episode-peak δ from env.extras['cat_delta'] (episode-MEAN δ dilutes
  strike-time δ by ~episode length). Register AFTER cfg.metrics['cat_soft'] — the manager
  evaluates in insertion order, so the hook has written this step's δ."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    self._peak = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self._peak[idx] = 0.0

  def __call__(self, env, **params) -> torch.Tensor:
    delta = env.extras.get("cat_delta")
    if delta is not None:
      torch.maximum(self._peak, delta.reshape(self._peak.shape), out=self._peak)
    return self._peak
