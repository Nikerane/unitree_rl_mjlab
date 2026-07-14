"""Substep impact-impulse accumulators for the Z1 hammer task (robot-side Λ_j + object-side I).

The thesis bounds the **robot-side per-joint reaction impulse** absorbed by each harmonic drive
at impact — the opposite side of the same collision from the object-side delivered impulse on the
nail (the maximize objective; see ``rewards.DeliveredImpulseTerm``). The quantity, per
``IMPULSE_CAT_IMPL_PLAN.md`` §1 and ``tracking_impact_impulse_design_research.md`` SQ3:

    Λ_j = max over any ``event_window_substeps`` interval of  Σ |qfrc_constraint_{s,j}| · dt · 1[contact_s]

accumulated at the **500 Hz substep rate** (never the 50 Hz control rate, which aliases the brief
impact), over a **TIME-based sliding window** (2026-07-13; see the SLIDING WINDOW bullet below —
NOT first→last contact-event anchoring, and NOT the 20 ms control grid).

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
  * SLIDING WINDOW (constraint side, 2026-07-13) — Λ_j is the contact-masked reaction impulse over
    the most recent ``event_window_substeps`` (≈50 ms), latched per control step as the max
    window-sum seen. TIME-based, not event-based: it has no gentle-prefix masking blind spot (the
    2026-07-13 Codex finding falsified the short-lived first-N-substeps prefix cap), bounds a
    sustained press at one window's worth, aggregates flickered sub-events, and leaves brief
    impacts (< window) reading their full event sum. Visibility after contact decays naturally
    within one window (≤ ~2.5 control steps) — bounded, not the episode-long persistence the
    2026-07 deep review rejected, and a later gentle tap cannot erase a violent window mid-decay
    (the max latch and the window still contain it).
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
  """per_substep MetricsTerm: SLIDING-WINDOW Σ |qfrc_constraint_j| · dt per arm joint —
  Λ_j(t) = the contact-masked reaction impulse accumulated over the most recent
  ``event_window_substeps`` (default 25 ≈ 50 ms) substeps.

  SEMANTICS (2026-07-13, replaces both the original unbounded per-event window and the
  short-lived first-N-substeps prefix cap): the window is TIME-based, not event-based.
  Off-contact substeps contribute zero but do NOT reset or shift the window. Consequences,
  each verified empirically (Codex adversarial review + verification workflow, 2026-07-13):

    * NO MASKING BLIND SPOT — a gentle 50 ms touch followed by a force spike in unbroken
      contact registers the spike (the prefix cap read exactly 0 for it, missing 99.75% of
      the true integral: the ``hold-then-spike`` bypass).
    * PRESS STILL BOUNDED — a constant press reads at most F̄·window·dt, bit-identical to the
      prefix cap for steady presses (the original 18×-inflation fix is preserved).
    * FLICKER-PROOF — two sub-events inside one window aggregate (a 1-substep contact flicker
      can no longer halve the read via per-event splitting).
    * BRIEF IMPACTS UNCHANGED — a strike shorter than the window (reference strikes are
      ~9-20 substeps) reads its full event sum, exactly as before.

  The 50 Hz read: ``_pulse`` latches the max window-sum seen during the current control step
  (cleared at each step boundary); ``impulse`` = max(pulse, current window sum). After contact
  ends the reading decays naturally as the window slides past (≤ ``window`` substeps ≈ 2.5
  control steps) — bounded visibility, NOT the episode-long persistence the 2026-07 deep
  review rejected.

  δ MULTI-READ (calibration note for enforcement, Fable verification 2026-07-13): because
  window (25) > decimation (10), one brief violation stays at full magnitude for ~3 consecutive
  50 Hz reads (+1 partial), vs exactly 1 under the old pulse semantics — per-event survival under
  CaT becomes ≈(1−δ)³, i.e. ~3× stronger termination pressure per violation, and the logged mean
  δ inflates ~3× per event. A task-COMPLETING strike gets only 1 read (env reset truncates the
  tail). Harmless while log-only (imp_max_p=0) but MUST be folded into any imp_max_p calibration
  (the C2 value 0.5 was chosen under single-read pulse semantics).

  NOTE this deliberately COUNTS sustained press reaction (up to one window's worth). Whether
  the enforced quantity should be this windowed reaction, a ballistic-only impulse, or split
  constraints is an OPEN thesis decision (Khadiv decision (e), 2026-07-13 Λ-quantity plan);
  this class is the mechanism, log-only until that is settled (imp_max_p=0).

  ``_last_off_qfrc`` (the most recent off-contact reaction) doubles as the frozen pre-contact
  baseline during contact — it only updates while OFF contact.
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
    self._last_off_qfrc = torch.zeros(env.num_envs, J, device=env.device)
    # SLIDING WINDOW state (see class docstring): ring buffer of the last `window` per-substep
    # contributions + their current sum. `_buf_i` is a single global ring index (the substep clock
    # is shared across envs); per-env reset just zeroes that env's buffer slice, which is correct
    # at any ring position. `_rolling` is recomputed as buf.sum(-1) each substep — exact (no
    # float-drift from incremental add/subtract) and trivially parallel at (B, J, window).
    self._window = int(p.get("event_window_substeps", 25))
    if self._window <= 0:
      raise ValueError(f"event_window_substeps must be >= 1, got {self._window}")
    self._buf = torch.zeros(env.num_envs, J, self._window, device=env.device)
    self._buf_i = 0
    self._rolling = torch.zeros(env.num_envs, J, device=env.device)
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
    """Per-joint windowed impulse Λ_j the constraint reads, shape (B, J):
    max(peak window-sum latched this control step, current window sum)."""
    return torch.maximum(self._pulse, self._rolling)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self._pulse[idx] = 0.0
    self._buf[idx] = 0.0
    self._rolling[idx] = 0.0
    self._last_off_qfrc[idx] = 0.0
    self._episode_peak[idx] = 0.0
    self._episode_peak_perjoint[idx] = 0.0
    return None

  def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
    # **params absorbs the MetricsTermCfg.params the manager re-passes each substep.
    # Branchless masked math throughout: no data-dependent `.any()` host syncs in the 500 Hz loop.
    if self._i % self._dec == 0:  # first substep of a control step: last step's latch was consumed
      self._pulse.zero_()
    qfrc = self._robot.data._joint_dof_field("qfrc_constraint")[:, self._joint_ids]  # (B,J)
    in_contact = (self._sensor.data.found > 0).any(dim=-1)  # (B,)
    in_c = in_contact[:, None]

    # This substep's contribution: rectified (optionally baseline-subtracted) reaction × dt,
    # contact-masked (off-contact substeps contribute 0 but still SLIDE the window — time-based).
    signal = qfrc - self._last_off_qfrc if self._subtract_baseline else qfrc
    contrib = signal.abs() * (env.physics_dt) * in_c  # (B,J)
    # Slide the window: overwrite the slot from `window` substeps ago, recompute the sum (exact).
    self._buf[:, :, self._buf_i] = contrib
    self._buf_i = (self._buf_i + 1) % self._window
    self._rolling = self._buf.sum(dim=-1)
    # Latch the peak window-sum seen this control step (the 50 Hz constraint read).
    torch.maximum(self._pulse, self._rolling, out=self._pulse)
    # Track the pre-contact baseline (updates only while OFF contact — frozen during contact).
    self._last_off_qfrc = torch.where(in_c, self._last_off_qfrc, qfrc)

    self._i += 1
    # Logged metric: EPISODE-PEAK worst-joint Λ (pair with MetricsTermCfg reduce="last").
    imp = self._pulse.amax(dim=1)  # (B,) — _pulse >= _rolling after the latch above
    torch.maximum(self._episode_peak, imp, out=self._episode_peak)
    # Per-joint episode-peak Λ (Task 6): same monotone-max update, one column per arm joint.
    torch.maximum(self._episode_peak_perjoint, self._pulse, out=self._episode_peak_perjoint)
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

  RE-ARM DEBOUNCE (2026-07-14, closes the C1 flicker-press reward farm): a rising contact edge
  re-arms the event age (opening a fresh payable window) ONLY after ``rearm_gap_substeps``
  (default = the window) consecutive OFF-contact substeps. Previously EVERY rising edge re-armed,
  so a press chopped by 1-substep releases (2 ms flicker — physical bounce chatter, or in
  principle an adversarial policy) re-opened the 50 ms cap indefinitely and paid at ~100% duty
  cycle. With the debounce, a flicker-chopped press pays exactly one window total (the age
  persists across sub-gap flickers), and the best possible farm rate is bounded at
  window/(window+gap) ≤ 50% duty. A genuine re-strike (retreat + wind-up = hundreds of
  substeps off contact) always re-arms; a quick post-strike bounce continues paying from the
  first event's remaining age (partial credit — undercounting reward is the safe direction).
  Reachability context: the 2026-07-12 probe found substep-rate toggling is NOT
  policy-commandable (20 ms action floor), so this is defense-in-depth against bounce chatter
  and future action-space changes, not a live exploit.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    p = cfg.params
    axis = p.get("axis", (0.0, 0.0, -1.0))
    self._axis = torch.tensor(axis, device=env.device, dtype=torch.float32)
    self._sensor = env.scene[p.get("sensor_name", "hammer_nail_impulse")]
    self._window = int(p.get("event_window_substeps", 25))
    self._rearm_gap = int(p.get("rearm_gap_substeps", self._window))
    if self._rearm_gap < 1:
      raise ValueError(f"rearm_gap_substeps must be >= 1, got {self._rearm_gap}")
    self._total = torch.zeros(env.num_envs, device=env.device)
    self._event_age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # Consecutive off-contact substeps; starts (and resets) ARMED so an episode's first contact
    # always opens a payable window.
    self._off_streak = torch.full(
      (env.num_envs,), self._rearm_gap, dtype=torch.long, device=env.device
    )
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
    self._off_streak[idx] = self._rearm_gap  # armed: the first contact opens a payable window
    self._in_contact_prev[idx] = False
    return None

  def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
    f = self._sensor.data.force  # (B, N, 3) world-frame net contact force per primary
    f_axial = (f * self._axis).sum(dim=-1).sum(dim=-1).clamp_min(0.0)  # (B,) net downward force
    in_contact = (self._sensor.data.found > 0).any(dim=-1)  # (B,)
    # Event age: 0 on a DEBOUNCED rising edge (see class docstring: re-arm requires >= rearm_gap
    # consecutive off-contact substeps, so flicker/bounce chatter cannot re-open the payable
    # window), +1 per in-contact substep (branchless).
    rising = in_contact & ~self._in_contact_prev
    rearm = rising & (self._off_streak >= self._rearm_gap)
    self._event_age = torch.where(rearm, torch.zeros_like(self._event_age), self._event_age)
    payable = in_contact & (self._event_age < self._window)
    self._total += f_axial * env.physics_dt * payable  # in-place: keep tensor identity
    self._event_age = self._event_age + in_contact.long()
    self._off_streak = torch.where(
      in_contact, torch.zeros_like(self._off_streak), self._off_streak + 1
    )
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


def delivered_impulse_total(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Full-step metric: the EXACT episode-cumulative delivered impulse I_total, from the stashed
  ``SubstepDeliveredImpulse`` (adversarial-review I3 fix, 2026-07-14). The per-substep
  ``substep_delivered`` metric with reduce="last" logs the TERMINAL control step's substep-MEAN
  of the ramping cumulative signal — for a strike-terminated episode (the success case, where
  most of I_total accrues inside that very step) it undercounts by up to ~half, and the
  undercount fraction differs across arms (strike-and-terminate vs timeout episodes), biasing
  any C3 cross-arm delivered-impulse comparison. Same full-step reduce="last" idiom as
  ``joint_impulse_peak``: logs the exact compute-time value of the monotone buffer."""
  dacc = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, None)
  if dacc is None:
    raise RuntimeError(
      "delivered_impulse_total requires the SubstepDeliveredImpulse metric (cat_impulse)."
    )
  return dacc.delivered


def contact_seen(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  """Full-step SENTINEL metric (Tier-3 safety net, 2026-07-14): 1.0 if the contact SENSOR
  registered ANY hammer↔nail contact this episode, else 0.0 (reduce="last" logs the terminal
  value, and the per-episode-reset air-time fields accumulate within the episode).

  This is an INDEPENDENT liveness signal for the object-side contact path — a healthy trained
  policy sits near 1.0 (it strikes every episode); a live drop, visible in TensorBoard mid-run,
  means contact sensing died on this device. Reads current/last_contact_time (track_air_time=True)
  so it never touches the qfrc accumulator it is meant to corroborate."""
  sensor = env.scene[sensor_name]
  seen = torch.zeros(env.num_envs, device=env.device)
  cct = getattr(sensor.data, "current_contact_time", None)
  lct = getattr(sensor.data, "last_contact_time", None)
  if cct is not None:
    seen = torch.maximum(seen, (cct > 0).any(dim=-1).float())
  if lct is not None:
    seen = torch.maximum(seen, (lct > 0).any(dim=-1).float())
  return seen


def impossible_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Full-step SENTINEL/ALARM metric (Tier-3 safety net, 2026-07-14): 1.0 on an episode that
  terminated on SUCCESS (env.reset_terminated -- the non-timeout nail_driven termination) while
  the robot-side per-joint peak Λ is EXACTLY zero. A nail cannot be driven to success without a
  reaction impulse, so this is dead instrumentation, never physics -- the exact false-negative
  that produced Λ≡0 on the first GPU smoke while success=1.0.

  reduce="last": at the terminal step, metrics_manager.compute() runs AFTER
  termination_manager.compute() and BEFORE the in-step reset, so reset_terminated is fresh and
  the accumulator peak is pre-reset. On timeout episodes reset_terminated is False -> 0.0.
  A healthy run logs a flat 0.0; ANY uptick in TensorBoard means kill the run now, don't wait
  for eval."""
  acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
  if acc is None:
    raise RuntimeError("impossible_success requires the SubstepImpulseAccumulator metric (cat_impulse).")
  lam_worst = acc._episode_peak_perjoint.amax(dim=1)  # (B,)
  terminated = env.reset_terminated  # (B,) bool: non-timeout success termination
  return (terminated & (lam_worst <= 0.0)).float()


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
