"""Unit tests for the substep impact-impulse accumulators (robot-side Λ_j + object-side I).

SEMANTICS (post-review redesign, 2026-07):

* ``SubstepImpulseAccumulator`` (robot side, the CONSTRAINT signal) uses **per-event pulse**
  semantics: while a contact window is open the running sum is visible; when the window closes
  (falling edge) its total is latched into a PULSE that stays visible for the REMAINDER of that
  control step only, and is cleared at the first substep of the next control step. Several windows
  closing within one control step MAX-combine. Consequences (the review findings this encodes):
  - a violation is visible to the 50 Hz constraint read for exactly the steps it happens in — no
    episode-long persistence (no duration-dependent punishment, no EMA saturation), and
  - a later gentle tap cannot erase a violent window's value (nothing persists to erase).

* ``SubstepDeliveredImpulse`` (object side, the REWARD signal) is **episode-cumulative**:
  I_total = Σ over ALL contact substeps of the episode of F_axial·dt — monotone non-decreasing,
  so the delta-crediting reward can never re-pay or zero-pay (review finding #4).

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


def _acc(B: int, subtract_baseline: bool = False, n_joints: int = 6):
  """Accumulator + feed(qfrc, found) closure driving one substep per call."""
  a = object.__new__(SubstepImpulseAccumulator)
  a._joint_ids = list(range(n_joints))
  a._subtract_baseline = subtract_baseline
  a._dec = DEC
  a._i = 0
  a._pulse = torch.zeros(B, n_joints)
  a._running = torch.zeros(B, n_joints)
  a._last_off_qfrc = torch.zeros(B, n_joints)
  a._in_contact_prev = torch.zeros(B, dtype=torch.bool)
  a._window = 25
  a._event_age = torch.zeros(B, dtype=torch.long)
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


def test_pulse_visible_for_remainder_of_closing_step_only():
  # Window closes at substep 3 of a control step -> value visible through substep 9 (the 50 Hz
  # read), then cleared at the first substep of the NEXT control step. NO persistence (review
  # finding: episode-long latched margin caused duration-dependent punishment + EMA saturation).
  acc, feed = _acc(B=1)
  q = torch.tensor([[2.0, 0, 0, 0, 0, 0]])
  for _ in range(3):
    feed(q, torch.ones(1, 1))          # substeps 0-2: in contact
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # substep 3: falling edge -> pulse latched
  total = torch.tensor(2.0 * DT * 3)
  for _ in range(6):                    # substeps 4-9: off contact, same control step
    feed(torch.zeros(1, 6), torch.zeros(1, 1))
    assert torch.allclose(acc.impulse[0, 0], total, atol=1e-9)  # visible at the step's read
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # substep 10 = next control step -> pulse cleared
  assert acc.impulse[0, 0].item() == 0.0


def test_gentle_tap_cannot_erase_a_violent_window():
  # Review finding #2: the old latch-overwrite let a later gentle touch REPLACE a violent value.
  # With pulse semantics the violent window is consumed at its own control step; a later tap makes
  # its own small pulse and there is nothing to erase.
  acc, feed = _acc(B=1)
  big = torch.tensor([[5.0, 0, 0, 0, 0, 0]])
  small = torch.tensor([[0.1, 0, 0, 0, 0, 0]])
  for _ in range(4):
    feed(big, torch.ones(1, 1))         # violent window, substeps 0-3
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # substep 4: closes -> pulse = 5*4*dt
  violent = 5.0 * DT * 4
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(violent), atol=1e-9)
  for _ in range(5):                     # substeps 5-9 off: still visible this step
    feed(torch.zeros(1, 6), torch.zeros(1, 1))
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(violent), atol=1e-9)
  # next control step: a gentle tap makes its OWN tiny pulse (nothing to erase — no persistence).
  feed(small, torch.ones(1, 1))          # substep 10: pulse cleared at the control-step boundary
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # substep 11: tap closes -> its own tiny pulse
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(0.1 * DT), atol=1e-9), acc.impulse


def test_press_cap_bounds_a_sustained_press():
  # Robot-side press guard (2026-07-12 vacuity §4/§5): a contact EVENT accumulates for at most
  # _window substeps, so a bottomed-out press cannot grow Λ without bound (mirrors the object side).
  acc, feed = _acc(B=1)
  acc._window = 10  # short window for the test
  q = torch.tensor([[3.0, 0, 0, 0, 0, 0]])
  for _ in range(40):  # 40-substep sustained press, far past the 10-substep window
    feed(q, torch.ones(1, 1))
  # only the first 10 substeps accumulate; _running (uncleared by the control-step grid) holds it
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(3.0 * DT * 10), atol=1e-9), acc.impulse


def test_brief_impact_under_window_is_uncapped():
  # A genuine impact (window < cap) is untouched by the press guard.
  acc, feed = _acc(B=1)
  acc._window = 25
  q = torch.tensor([[4.0, 0, 0, 0, 0, 0]])
  for _ in range(8):  # brief impact, well under the window
    feed(q, torch.ones(1, 1))
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(4.0 * DT * 8), atol=1e-9), acc.impulse


def test_press_cap_re_arms_on_a_new_contact_event():
  # The event age RESETS on the rising edge, so a second contact event gets its own fresh window.
  # Assert acc._running directly (not impulse) so the check ISOLATES re-arm and is robust to the
  # pulse-grid clear: if the age did NOT reset, event 2 would inherit event 1's saturated age and
  # accumulate nothing (running would stay 0).
  acc, feed = _acc(B=1)
  acc._window = 3
  q = torch.tensor([[2.0, 0, 0, 0, 0, 0]])
  for _ in range(5):  # event 1: 5 substeps, capped at 3
    feed(q, torch.ones(1, 1))
  assert torch.allclose(acc._running[0, 0], torch.tensor(2.0 * DT * 3), atol=1e-9), acc._running
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # falling edge closes event 1 (running -> 0)
  for _ in range(2):  # event 2: 2 substeps -> a fresh window pays BOTH only if the age re-armed
    feed(q, torch.ones(1, 1))
  assert torch.allclose(acc._running[0, 0], torch.tensor(2.0 * DT * 2), atol=1e-9), acc._running


def test_two_windows_closing_in_one_step_max_combine():
  # Bounce chatter within one control step must not REPLACE the bigger window (old bug).
  acc, feed = _acc(B=1)
  big = torch.tensor([[3.0, 0, 0, 0, 0, 0]])
  small = torch.tensor([[0.2, 0, 0, 0, 0, 0]])
  for _ in range(3):
    feed(big, torch.ones(1, 1))          # substeps 0-2: big window (3*3*dt)
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # substep 3: big closes
  feed(small, torch.ones(1, 1))          # substep 4: tap opens
  feed(torch.zeros(1, 6), torch.zeros(1, 1))  # substep 5: tap closes (0.2*dt)
  # both closed within the SAME control step: pulse = max(big, tap) = big
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(3.0 * DT * 3), atol=1e-9), acc.impulse


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
  # Terminating an episode while in contact must not leave a phantom window: reset clears
  # _in_contact_prev, so the next substep (still touching, post-reset state) opens a FRESH window.
  acc, feed = _acc(B=1)
  q = torch.tensor([[4.0, 0, 0, 0, 0, 0]])
  feed(q, torch.ones(1, 1))
  feed(q, torch.ones(1, 1))
  acc.reset(None)  # episode ends mid-contact
  assert acc.impulse.abs().sum() == 0
  feed(q, torch.ones(1, 1))  # contact persists into the new episode -> new window, 1 substep
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(4.0 * DT), atol=1e-9), acc.impulse


# --- Object-side delivered axial impulse: EPISODE-CUMULATIVE I_total = Σ F_axial·dt -------------


def _dacc(B: int, window: int = 25):
  a = object.__new__(SubstepDeliveredImpulse)
  a._axis = torch.tensor([0.0, 0.0, -1.0])
  a._total = torch.zeros(B)
  a._event_age = torch.zeros(B, dtype=torch.long)
  a._window = window
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
  acc, feed = _dacc(1, window=3)
  f = torch.tensor([[[0.0, 0.0, -10.0]]])
  for _ in range(8):  # sustained press, 8 substeps in one event
    feed(f, torch.ones(1, 1))
  # only the first 3 substeps accrued
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT * 3]), atol=1e-9), acc.delivered
  # a NEW event (release then re-contact) gets a fresh window
  feed(torch.zeros(1, 1, 3), torch.zeros(1, 1))
  feed(f, torch.ones(1, 1))
  assert torch.allclose(acc.delivered, torch.tensor([10.0 * DT * 4]), atol=1e-9), acc.delivered


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
  # survives well past the control-step boundary where .impulse (the pulse) clears
  for _ in range(DEC + 2):
    feed(torch.zeros(1, 6), torch.zeros(1, 1))
  assert acc.impulse[0, 0].item() == 0.0  # pulse has cleared...
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
