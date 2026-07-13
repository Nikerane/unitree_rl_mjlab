"""Unit tests for the substep impact-impulse accumulators (robot-side Λ_j + object-side I).

SEMANTICS (2026-07-13 sliding-window redesign; supersedes both the per-event pulse and the
short-lived first-N-substeps prefix cap the Codex adversarial review falsified):

* ``SubstepImpulseAccumulator`` (robot side, the CONSTRAINT signal) computes a **TIME-based
  sliding window**: Λ_j(t) = contact-masked Σ|qfrc|·dt over the most recent
  ``event_window_substeps`` (default 25 ≈ 50 ms). Off-contact substeps contribute 0 but do NOT
  reset or shift the window. ``_pulse`` latches the max window-sum per control step (cleared at
  each step boundary); ``impulse`` = max(pulse, current window sum). Encoded properties:
  - NO masking blind spot: a gentle prefix + late spike in unbroken contact registers the spike
    (the prefix cap read exactly 0 for it — the hold-then-spike bypass);
  - a sustained press is bounded at F̄·window·dt (identical to the prefix cap for steady presses);
  - flickered sub-events inside one window AGGREGATE (fragmentation can no longer halve the read);
  - brief impacts (< window) read their full event sum, unchanged;
  - visibility after contact decays within ≤ window substeps (bounded — not episode persistence),
    so a later gentle tap cannot erase a violent reading mid-decay.

* ``SubstepDeliveredImpulse`` (object side, the REWARD signal) is **episode-cumulative** with a
  per-event PREFIX cap (undercounting reward is the safe direction — anti-press-farming):
  I_total = Σ over payable contact substeps of F_axial·dt — monotone non-decreasing,
  so the delta-crediting reward can never re-pay or zero-pay (review finding #4). Since
  2026-07-14 the cap's re-arm is DEBOUNCED: a rising edge opens a fresh payable window only
  after ``rearm_gap_substeps`` (default = window) consecutive off-contact substeps, closing
  the C1 flicker-press farm (1-substep releases used to re-arm the cap indefinitely).

Tests fake the env in the repo's SimpleNamespace style and drive one substep at a time. The
accumulators cache the robot/sensor handles at construction, so the helpers expose mutable fake
data objects and a ``feed(...)`` closure.
"""

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.impulse_bound import (
  _ENV_SUBSTEP_IMPULSE_ATTR,
  CatDeltaPeak,
  SubstepDeliveredImpulse,
  SubstepImpulseAccumulator,
  joint_impulse_peak,
)

DT = 0.002  # physics_dt @ 500 Hz
DEC = 10    # decimation (substeps per control step)


def _acc(B: int, subtract_baseline: bool = False, n_joints: int = 6, window: int = 25):
  """Accumulator + feed(qfrc, found) closure driving one substep per call. `window` must be set
  here (not mutated afterwards): the ring buffer is sized to it at construction."""
  a = object.__new__(SubstepImpulseAccumulator)
  a._joint_ids = list(range(n_joints))
  a._subtract_baseline = subtract_baseline
  a._dec = DEC
  a._i = 0
  a._pulse = torch.zeros(B, n_joints)
  a._last_off_qfrc = torch.zeros(B, n_joints)
  a._window = window
  a._buf = torch.zeros(B, n_joints, a._window)
  a._buf_i = 0
  a._rolling = torch.zeros(B, n_joints)
  a._episode_peak = torch.zeros(B)
  a._episode_peak_perjoint = torch.zeros(B, n_joints)
  robot_data = SimpleNamespace(_joint_dof_field=None)
  sensor_data = SimpleNamespace(found=None)
  a._robot = SimpleNamespace(data=robot_data)
  a._sensor = SimpleNamespace(data=sensor_data)
  env = SimpleNamespace(physics_dt=DT)

  def feed(qfrc: torch.Tensor, found: torch.Tensor) -> torch.Tensor:
    robot_data._joint_dof_field = lambda name: qfrc
    sensor_data.found = found
    return a(env)

  return a, feed


def test_off_contact_accumulates_nothing():
  acc, feed = _acc(B=2)
  for _ in range(5):
    feed(torch.full((2, 6), 5.0), torch.zeros(2, 1))  # big qfrc but no contact
  assert torch.allclose(acc.impulse, torch.zeros(2, 6)), acc.impulse


def test_accumulates_rectified_qfrc_dt_during_contact():
  acc, feed = _acc(B=1)
  q = torch.tensor([[1.0, -2.0, 3.0, 0.0, 0.0, 0.0]])
  for _ in range(3):
    feed(q, torch.ones(1, 1))
  expected = q.abs() * DT * 3  # rectified per-joint, summed over 3 substeps
  assert torch.allclose(acc.impulse, expected, atol=1e-9), (acc.impulse, expected)


def test_shape_is_per_joint_B_by_J():
  acc, feed = _acc(B=4, n_joints=6)
  feed(torch.ones(4, 6), torch.ones(4, 1))
  assert acc.impulse.shape == (4, 6)


def test_window_spanning_control_steps_stays_visible_while_open():
  # A window open across a control-step boundary keeps its running sum (no reset on the grid).
  acc, feed = _acc(B=1)
  q = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
  for _ in range(25):  # 2.5 control steps of sustained contact
    feed(q, torch.ones(1, 1))
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(1.0 * DT * 25), atol=1e-9), acc.impulse


def test_reading_decays_within_one_window_after_contact():
  # Bounded visibility: after contact ends, the reading persists only while the sliding window
  # still contains the event, then decays to zero within `window` substeps — NOT episode-long
  # persistence (2026-07 deep-review rejection), NOT single-step consumption (old pulse).
  acc, feed = _acc(B=1, window=10)
  q = torch.tensor([[2.0, 0, 0, 0, 0, 0]])
  for _ in range(3):
    feed(q, torch.ones(1, 1))          # substeps 0-2: in contact
  total = torch.tensor(2.0 * DT * 3)
  assert torch.allclose(acc.impulse[0, 0], total, atol=1e-6)
  for _ in range(7):                    # substeps 3-9: window still contains all 3 contributions
    feed(torch.zeros(1, 6), torch.zeros(1, 1))
    assert acc._rolling[0, 0] > 0.0     # event still inside the window
  for _ in range(3):                    # substeps 10-12: contributions slide out one by one
    feed(torch.zeros(1, 6), torch.zeros(1, 1))
  assert acc._rolling[0, 0].item() == 0.0  # window fully past the event
  for _ in range(DEC):                  # a full control step later, the read is 0 too
    feed(torch.zeros(1, 6), torch.zeros(1, 1))
  assert acc.impulse[0, 0].item() == 0.0


def test_gentle_tap_cannot_erase_a_violent_reading():
  # Review finding #2 held under sliding-window semantics, and more strongly: a later gentle tap
  # cannot REPLACE a violent reading — while the window still contains the violent event the
  # reading includes it (a tap only ADDS), and the per-step max latch never overwrites downward.
  acc, feed = _acc(B=1)
  big = torch.tensor([[5.0, 0, 0, 0, 0, 0]])
  small = torch.tensor([[0.1, 0, 0, 0, 0, 0]])
  for _ in range(4):
    feed(big, torch.ones(1, 1))         # violent event, substeps 0-3
  violent = 5.0 * DT * 4
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(violent), atol=1e-6)
  for _ in range(6):                     # substeps 4-9 off contact: still inside the window
    feed(torch.zeros(1, 6), torch.zeros(1, 1))
  feed(small, torch.ones(1, 1))          # substep 10 (next step): gentle tap while window holds it
  assert acc.impulse[0, 0] >= violent - 1e-9, acc.impulse  # tap cannot lower the reading


def test_sustained_press_is_bounded_at_one_window():
  # A bottomed-out press cannot grow Λ without bound: a constant press reads exactly F̄·window·dt
  # under the sliding window — bit-identical to the old prefix cap for steady presses (verified
  # 2026-07-13), preserving the 18×-inflation fix without the masking blind spot.
  acc, feed = _acc(B=1, window=10)
  q = torch.tensor([[3.0, 0, 0, 0, 0, 0]])
  for _ in range(40):  # 40-substep sustained press, far past the 10-substep window
    feed(q, torch.ones(1, 1))
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(3.0 * DT * 10), atol=1e-6), acc.impulse


def test_brief_impact_under_window_reads_full_event_sum():
  # A genuine impact (shorter than the window) is untouched — reads its full event sum.
  acc, feed = _acc(B=1, window=25)
  q = torch.tensor([[4.0, 0, 0, 0, 0, 0]])
  for _ in range(8):  # brief impact, well under the window
    feed(q, torch.ones(1, 1))
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(4.0 * DT * 8), atol=1e-6), acc.impulse


def test_hold_then_spike_bypass_is_closed():
  # THE Codex adversarial regression (2026-07-13): under the falsified prefix cap, holding gentle
  # contact past the window then spiking force in unbroken contact contributed EXACTLY 0 to Λ
  # (99.75% of the true integral missed). The sliding window must register the spike: the peak
  # window covers the last 25 substeps = 5 gentle + 20 spike.
  acc, feed = _acc(B=1, window=25)
  gentle = torch.tensor([[0.1, 0, 0, 0, 0, 0]])
  spike = torch.tensor([[50.0, 0, 0, 0, 0, 0]])
  for _ in range(30):
    feed(gentle, torch.ones(1, 1))      # 30 substeps of gentle hold (past the 25 window)
  for _ in range(20):
    feed(spike, torch.ones(1, 1))       # late spike, contact never breaks
  expected = (5 * 0.1 + 20 * 50.0) * DT  # the window at peak: 5 gentle + 20 spike substeps
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(expected), atol=1e-5), acc.impulse
  # and the episode-peak logged metric saw it too
  assert acc._episode_peak[0] >= expected - 1e-5


def test_flickered_subevents_within_window_aggregate():
  # Fragmentation exploit closed: two sub-events split by a 1-substep contact flicker inside one
  # window SUM (time-based window), instead of MAX-combining as separate events (which let a
  # flicker halve the constraint read under per-event semantics).
  acc, feed = _acc(B=1, window=25)
  q = torch.tensor([[10.0, 0, 0, 0, 0, 0]])
  for _ in range(12):
    feed(q, torch.ones(1, 1))           # sub-event 1
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # 1-substep flicker gap
  for _ in range(12):
    feed(q, torch.ones(1, 1))           # sub-event 2 (still inside the 25-substep window)
  expected = 24 * 10.0 * DT             # both sub-events aggregate
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(expected), atol=1e-5), acc.impulse


def test_bounce_chatter_within_window_aggregates():
  # Bounce chatter within one window AGGREGATES (it is real reaction within the interval); a tap
  # after a bigger event can only add to the reading, never replace it (old latch-overwrite bug).
  acc, feed = _acc(B=1)
  big = torch.tensor([[3.0, 0, 0, 0, 0, 0]])
  small = torch.tensor([[0.2, 0, 0, 0, 0, 0]])
  for _ in range(3):
    feed(big, torch.ones(1, 1))          # substeps 0-2: big event (3*3*dt)
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # substep 3: gap
  feed(small, torch.ones(1, 1))          # substep 4: tap
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # substep 5
  expected = (3 * 3.0 + 0.2) * DT        # window covers big + tap
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(expected), atol=1e-6), acc.impulse
  assert acc.impulse[0, 0] >= 3 * 3.0 * DT - 1e-9  # never below the big event alone


def test_per_env_contact_independence():
  acc, feed = _acc(B=2)
  q = torch.tensor([[3.0, 0, 0, 0, 0, 0], [3.0, 0, 0, 0, 0, 0]])
  found = torch.tensor([[1.0], [0.0]])  # env0 in contact, env1 not
  feed(q, found)
  assert acc.impulse[0, 0] > 0 and acc.impulse[1, 0] == 0, acc.impulse


def test_reset_zeros_buffers():
  acc, feed = _acc(B=2)
  feed(torch.ones(2, 6), torch.ones(2, 1))
  assert acc.impulse.abs().sum() > 0
  acc.reset(None)
  assert torch.allclose(acc.impulse, torch.zeros(2, 6))


def test_reset_subset_of_envs():
  acc, feed = _acc(B=2)
  feed(torch.ones(2, 6), torch.ones(2, 1))
  acc.reset(torch.tensor([0]))
  assert torch.allclose(acc.impulse[0], torch.zeros(6))
  assert acc.impulse[1].abs().sum() > 0


def test_found_multislot_reduced_with_any():
  acc, feed = _acc(B=1)
  q = torch.tensor([[2.0, 0, 0, 0, 0, 0]])
  feed(q, torch.tensor([[0.0, 0.0, 1.0]]))  # 3 slots, last in contact
  assert acc.impulse[0, 0] > 0


def test_baseline_subtraction_removes_precontact_offset():
  acc, feed = _acc(B=1, subtract_baseline=True)
  base = torch.tensor([[1.0, 0, 0, 0, 0, 0]])  # friction baseline at the last off-contact substep
  feed(base, torch.zeros(1, 1))                 # off-contact: sets rolling baseline
  feed(torch.tensor([[5.0, 0, 0, 0, 0, 0]]), torch.ones(1, 1))  # contact: 5 = baseline 1 + 4
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(4.0 * DT), atol=1e-9), acc.impulse


def test_returns_episode_peak_shape_B():
  # The logged metric is the EPISODE-PEAK worst-joint Λ (reduce="last" in the cfg) — the old
  # mean-of-running-max diluted Λ ~100× and made the C0 log-only run unobservable.
  acc, feed = _acc(B=3)
  out = feed(torch.ones(3, 6), torch.ones(3, 1))
  assert out.shape == (3,)
  # peak persists after the window is consumed (pulse cleared), unlike .impulse
  for _ in range(DEC + 2):
    out = feed(torch.zeros(3, 6), torch.zeros(3, 1))
  assert torch.allclose(out, torch.full((3,), 1.0 * DT), atol=1e-9), out
  acc.reset(None)
  assert acc._episode_peak.abs().sum() == 0


def test_reset_mid_contact_no_phantom_window():
  # Terminating an episode while in contact must not leave a phantom reading: reset zeroes the
  # ring buffer + rolling sum, so the next substep (still touching, post-reset) starts fresh.
  acc, feed = _acc(B=1)
  q = torch.tensor([[4.0, 0, 0, 0, 0, 0]])
  feed(q, torch.ones(1, 1))
  feed(q, torch.ones(1, 1))
  acc.reset(None)  # episode ends mid-contact
  assert acc.impulse.abs().sum() == 0
  feed(q, torch.ones(1, 1))  # contact persists into the new episode -> new window, 1 substep
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(4.0 * DT), atol=1e-9), acc.impulse


# --- Object-side delivered axial impulse: EPISODE-CUMULATIVE I_total = Σ F_axial·dt -------------


def _dacc(B: int, window: int = 25, rearm_gap: int | None = None):
  """rearm_gap defaults to the window, mirroring the class's own default."""
  a = object.__new__(SubstepDeliveredImpulse)
  a._axis = torch.tensor([0.0, 0.0, -1.0])
  a._total = torch.zeros(B)
  a._event_age = torch.zeros(B, dtype=torch.long)
  a._window = window
  a._rearm_gap = window if rearm_gap is None else rearm_gap
  a._off_streak = torch.full((B,), a._rearm_gap, dtype=torch.long)  # armed at episode start
  a._in_contact_prev = torch.zeros(B, dtype=torch.bool)
  sensor_data = SimpleNamespace(force=None, found=None)
  a._sensor = SimpleNamespace(data=sensor_data)
  env = SimpleNamespace(physics_dt=DT)

  def feed(force: torch.Tensor, found: torch.Tensor) -> torch.Tensor:
    sensor_data.force = force
    sensor_data.found = found
    return a(env)

  return a, feed


def test_delivered_off_contact_zero():
  acc, feed = _dacc(1)
  for _ in range(3):
    feed(torch.zeros(1, 1, 3), torch.zeros(1, 1))
  assert acc.delivered.item() == 0.0


def test_delivered_accumulates_downward_axial_force_impulse():
  acc, feed = _dacc(1)
  f = torch.tensor([[[0.0, 0.0, -10.0]]])  # 10 N downward (along the -Z strike axis)
  for _ in range(3):
    feed(f, torch.ones(1, 1))
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT * 3]), atol=1e-9), acc.delivered


def test_delivered_ignores_upward_force():
  acc, feed = _dacc(1)
  feed(torch.tensor([[[0.0, 0.0, 5.0]]]), torch.ones(1, 1))  # +Z upward
  assert acc.delivered.item() == 0.0


def test_delivered_sums_over_multiple_primary_contacts():
  acc, feed = _dacc(1)
  f = torch.tensor([[[0.0, 0.0, -6.0], [0.0, 0.0, -4.0]]])  # two primaries: net axial = 10 down
  feed(f, torch.ones(1, 2))
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT]), atol=1e-9)


def test_delivered_is_cumulative_and_monotone_across_windows():
  # Review finding #4's root fix: I_total sums across ALL windows and never decreases, so the
  # delta-crediting reward can neither re-pay nor zero-pay.
  acc, feed = _dacc(1)
  f1 = torch.tensor([[[0.0, 0.0, -8.0]]])
  f2 = torch.tensor([[[0.0, 0.0, -2.0]]])
  feed(f1, torch.ones(1, 1))                   # window 1: 8*dt
  feed(torch.zeros(1, 1, 3), torch.zeros(1, 1))  # air
  after_w1 = acc.delivered.clone()
  assert torch.allclose(after_w1, torch.tensor([8.0 * DT]), atol=1e-9)
  feed(f2, torch.ones(1, 1))                   # window 2 (weaker): ADDS, does not replace
  assert torch.allclose(acc.delivered, torch.tensor([8.0 * DT + 2.0 * DT]), atol=1e-9)
  assert (acc.delivered >= after_w1).all()     # monotone


def test_delivered_event_window_cap_blocks_press_farming():
  # xhigh-review finding: unbounded cumulative ∫F·dt lets a slow quasi-static PRESS accrue huge
  # 'delivered impulse' (long duration × moderate force) and out-earn striking. Each contact event
  # pays only its first `window` substeps — covers real strikes (measured p95 ≈ 20 substeps),
  # truncates presses.
  acc, feed = _dacc(1, window=3, rearm_gap=2)
  f = torch.tensor([[[0.0, 0.0, -10.0]]])
  for _ in range(8):  # sustained press, 8 substeps in one event
    feed(f, torch.ones(1, 1))
  # only the first 3 substeps accrued
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT * 3]), atol=1e-9), acc.delivered
  # a GENUINE new event (release for >= rearm_gap substeps, then re-contact) gets a fresh window
  for _ in range(2):
    feed(torch.zeros(1, 1, 3), torch.zeros(1, 1))
  feed(f, torch.ones(1, 1))
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT * 4]), atol=1e-9), acc.delivered


def test_delivered_flicker_rearm_farm_is_closed():
  # C1 adversarial-review finding (fixed 2026-07-14): _event_age used to reset on EVERY rising
  # edge, so a press chopped by 1-substep releases (2 ms flicker / bounce chatter) re-armed the
  # anti-press cap indefinitely — an unbounded reward farm at ~100% duty. Now a rising edge
  # re-arms only after >= rearm_gap consecutive off-contact substeps: the flicker-chopped press
  # pays exactly one window total, no matter how many flicker cycles run.
  acc, feed = _dacc(1, window=3, rearm_gap=5)
  f = torch.tensor([[[0.0, 0.0, -10.0]]])
  for _ in range(3):
    feed(f, torch.ones(1, 1))            # event pays its full window (3 substeps)
  paid = acc.delivered.clone()
  assert torch.allclose(paid, torch.tensor([10.0 * DT * 3]), atol=1e-9)
  for _ in range(6):                      # 6 flicker cycles: 1 substep off, 3 substeps pressing
    feed(torch.zeros(1, 1, 3), torch.zeros(1, 1))  # sub-gap release — must NOT re-arm
    for _ in range(3):
      feed(f, torch.ones(1, 1))
  assert torch.allclose(acc.delivered, paid, atol=1e-9), (
    f"flicker farm re-opened the payable window: {acc.delivered} vs {paid}"
  )


def test_delivered_rearm_requires_full_gap_and_reset_rearms():
  # The debounce boundary: a gap one substep short of rearm_gap does not re-arm; a gap of exactly
  # rearm_gap does. Episode reset() re-arms immediately (off_streak restored to the gap).
  acc, feed = _dacc(1, window=2, rearm_gap=4)
  f = torch.tensor([[[0.0, 0.0, -10.0]]])
  for _ in range(2):
    feed(f, torch.ones(1, 1))            # window exhausted
  for _ in range(3):                      # gap of 3 < 4: NOT re-armed
    feed(torch.zeros(1, 1, 3), torch.zeros(1, 1))
  feed(f, torch.ones(1, 1))
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT * 2]), atol=1e-9), acc.delivered
  for _ in range(4):                      # gap of exactly 4: re-armed
    feed(torch.zeros(1, 1, 3), torch.zeros(1, 1))
  feed(f, torch.ones(1, 1))
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT * 3]), atol=1e-9), acc.delivered
  acc.reset(None)
  assert acc.delivered.item() == 0.0
  feed(f, torch.ones(1, 1))               # first contact after reset pays immediately (armed)
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT]), atol=1e-9), acc.delivered


def test_delivered_reset_and_shape():
  acc, feed = _dacc(2)
  feed(torch.tensor([[[0.0, 0.0, -3.0]]]).expand(2, 1, 3).contiguous(), torch.ones(2, 1))
  assert acc.delivered.shape == (2,) and acc.delivered.sum() > 0
  acc.reset(None)
  assert torch.allclose(acc.delivered, torch.zeros(2))


# --- Per-joint episode-peak Λ + module-level readers (Task 6 observability) ----------------------


def test_episode_peak_perjoint_matches_window_and_survives_reset():
  # New buffer: per-JOINT episode-peak Λ (not just the worst-joint scalar _episode_peak) — the
  # authoritative full-step reader (joint_impulse_peak) logs this directly via reduce="last".
  acc, feed = _acc(B=1, n_joints=6)
  q = torch.tensor([[1.0, -2.0, 3.0, 0.0, 0.0, 0.0]])
  for _ in range(3):
    feed(q, torch.ones(1, 1))
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # falling edge: window closes -> pulse latched
  expected = (q.abs() * DT * 3)[0]
  assert torch.allclose(acc._episode_peak_perjoint[0], expected, atol=1e-9), acc._episode_peak_perjoint
  # survives well past the point where .impulse clears (the sliding window passes the event
  # within `window` substeps; + DEC crosses a control-step boundary so the latch clears too)
  for _ in range(acc._window + DEC):
    feed(torch.zeros(1, 6), torch.zeros(1, 1))
  assert acc.impulse[0, 0].item() == 0.0  # transient reading has decayed to zero...
  assert torch.allclose(acc._episode_peak_perjoint[0], expected, atol=1e-9)  # ...but the peak persists
  acc.reset(None)
  assert torch.allclose(acc._episode_peak_perjoint, torch.zeros(1, 6))


def test_episode_peak_perjoint_reset_subset_of_envs():
  acc, feed = _acc(B=2, n_joints=6)
  feed(torch.ones(2, 6), torch.ones(2, 1))
  assert acc._episode_peak_perjoint.abs().sum() > 0
  acc.reset(torch.tensor([0]))
  assert torch.allclose(acc._episode_peak_perjoint[0], torch.zeros(6))
  assert acc._episode_peak_perjoint[1].abs().sum() > 0


def test_joint_impulse_peak_reads_requested_column():
  acc, feed = _acc(B=2, n_joints=6)
  q = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]])
  feed(q, torch.ones(2, 1))
  env = SimpleNamespace()
  setattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, acc)
  out1 = joint_impulse_peak(env, joint=1)
  assert out1.shape == (2,)
  assert torch.allclose(out1, acc._episode_peak_perjoint[:, 1])
  out0 = joint_impulse_peak(env, joint=0)
  assert torch.allclose(out0, acc._episode_peak_perjoint[:, 0])
  assert not torch.allclose(out0, out1)


def test_joint_impulse_peak_requires_accumulator():
  env = SimpleNamespace()  # no _hammer_substep_impulse stashed
  with pytest.raises(RuntimeError):
    joint_impulse_peak(env, joint=0)


def test_cat_delta_peak_tracks_running_max_and_resets():
  peak_term = object.__new__(CatDeltaPeak)
  peak_term._peak = torch.zeros(2)

  env = SimpleNamespace(extras={"cat_delta": torch.tensor([0.1, 0.0])})
  out = peak_term(env)
  assert torch.allclose(out, torch.tensor([0.1, 0.0]))

  env.extras["cat_delta"] = torch.tensor([0.05, 0.3])  # smaller in col0, bigger in col1
  out = peak_term(env)
  assert torch.allclose(out, torch.tensor([0.1, 0.3])), out  # running MAX, not the last value

  env.extras["cat_delta"] = torch.tensor([0.4, 0.0])
  out = peak_term(env)
  assert torch.allclose(out, torch.tensor([0.4, 0.3])), out

  peak_term.reset(None)
  assert torch.allclose(peak_term._peak, torch.zeros(2))


def test_cat_delta_peak_missing_key_is_a_noop():
  peak_term = object.__new__(CatDeltaPeak)
  peak_term._peak = torch.tensor([0.2, 0.0])
  out = peak_term(SimpleNamespace(extras={}))  # no "cat_delta" key yet this step
  assert torch.allclose(out, torch.tensor([0.2, 0.0]))


def test_cat_delta_peak_reset_subset_of_envs():
  peak_term = object.__new__(CatDeltaPeak)
  peak_term._peak = torch.tensor([0.5, 0.7])
  peak_term.reset(torch.tensor([0]))
  assert torch.allclose(peak_term._peak, torch.tensor([0.0, 0.7]))
