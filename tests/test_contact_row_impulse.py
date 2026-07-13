"""Task 8 (Track 2): efc layout probe + exact-reconstruction invariant.

Drives the OPEN-LOOP single-strike reference against a live Z1 hammer env (modeled on
``docs/research/reward-design/derive_impulse_thresholds.py:main()`` / the ``auto_reset=False``
idiom from ``docs/research/reward-design/reward_design_util.py:run_reference_strikes``) and
certifies, at EVERY control step:

1. ``reconstruct_qfrc_from_efc`` exactly reproduces ``qfrc_constraint`` on the robot's arm dofs
   (the foundation invariant — if this doesn't hold, the measured efc layout in
   ``contact_row_impulse.py``'s module docstring is wrong and nothing downstream can be trusted).
2. ``contact_row_qfrc`` (the geom-pair-filtered subset of those same rows) is exactly zero off
   contact and nonzero on some in-contact step.
3. With ``num_envs=2`` and two temporally-offset strike phases, per-world attribution via
   ``contact.worldid`` correctly isolates each world's contact rows — the invariant holds
   per-world, and a world with no active contact reports an exactly-zero contact-row signal even
   while its sibling world is mid-impact.

All tests here require the full mjlab + Warp stack (``@pytest.mark.integration`` / module-level
``pytestmark``); the single-env and two-env envs are each built ONCE per test session (module
scope) and the strike driven once, since a physics rollout takes real wall-clock time.

Task 9 (Track 2) extends this file with ``ContactRowImpulseAccumulator``: (a) a driven-strike
comparison against the RAW-signal shipped ``SubstepImpulseAccumulator`` (upper bound), and (b) a
FAKE-env synthetic pulse-semantics unit test mirroring ``tests/test_impulse_bound.py`` — the
latter needs no physics and is fast despite inheriting the module's ``integration`` marker (the
marker is documentation-only in this repo's ``pytest.ini``, not filtered by ``addopts``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp import contact_row_impulse
from src.tasks.hammer.mdp.contact_row_impulse import (
  ContactRowImpulseAccumulator,
  arm_dof_cols,
  contact_row_qfrc,
  reconstruct_qfrc_from_efc,
)
from src.tasks.hammer.mdp.references import SingleStrikeReference

pytestmark = pytest.mark.integration

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# Post-playback hold steps, mirrors derive_impulse_thresholds.py's HOLD_STEPS (lets a strike that
# lands on the last scripted waypoint finish settling / the success termination fire).
HOLD_STEPS = 6


def _build_driving_helpers(env):
  """Site/joint SceneEntityCfg resolution + head()/nail_top() closures, shared by both fixtures."""
  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  arm_cfg = SceneEntityCfg("robot", joint_names=ARM)
  head_cfg.resolve(env.scene)
  arm_cfg.resolve(env.scene)
  nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  nail_cfg.resolve(env.scene)

  def head() -> torch.Tensor:
    return robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)

  def nail_top() -> torch.Tensor:
    return nail_e.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

  return head, nail_top, arm_cfg.joint_ids


@pytest.fixture(scope="module")
def strike_trace_1env():
  """Drive one open-loop reference strike (1 env); record the per-control-step reconstruction,
  ground-truth qfrc, contact-row signal, and contact-sensor flag. Assertions live in the test
  functions below, not here, so a failure attributes to the specific property being checked."""
  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device="cpu")
  robot = env.scene["robot"]
  contact_sensor = env.scene["hammer_nail_contact"]
  head, nail_top, arm_joint_ids = _build_driving_helpers(env)

  env.reset()
  ref = SingleStrikeReference(1, env.device, approach_height=0.10)
  ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
  n = ref.playback_length()
  cols = arm_dof_cols(env)

  recon_cols: list[torch.Tensor] = []
  qfrc_cols: list[torch.Tensor] = []
  contact_flags: list[bool] = []
  cr_traces: list[torch.Tensor] = []

  for k in range(1, n + HOLD_STEPS + 1):
    target = ref.playback_target(min(k, n))
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    env.step(action)

    recon = reconstruct_qfrc_from_efc(env)
    qfrc = robot.data._joint_dof_field("qfrc_constraint")
    recon_cols.append(recon[:, cols].clone())
    qfrc_cols.append(qfrc[:, arm_joint_ids].clone())

    in_c = bool((contact_sensor.data.found > 0).any())
    contact_flags.append(in_c)
    cr_traces.append(contact_row_qfrc(env).clone())

    if bool(env.reset_terminated.any()):  # success fired; auto_reset=False keeps state readable
      break

  env.close()
  return {
    "recon_cols": recon_cols,
    "qfrc_cols": qfrc_cols,
    "contact_flags": contact_flags,
    "cr_traces": cr_traces,
  }


@pytest.fixture(scope="module")
def strike_trace_2env():
  """2-env attribution scenario (task brief step 5): a single ``SingleStrikeReference(2, ...)``
  is batched, but ``playback_target(k)`` takes one scalar ``k`` for ALL envs — so the two worlds
  are given DIFFERENT phases by row-selecting two separate calls (env 0 at phase ``k``, env 1
  lagged to phase ``k-1``), which staggers their impact windows enough to observe a step where
  exactly one world is in contact."""
  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 2
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device="cpu")
  robot = env.scene["robot"]
  contact_sensor = env.scene["hammer_nail_contact"]
  head, nail_top, arm_joint_ids = _build_driving_helpers(env)

  env.reset()
  ref = SingleStrikeReference(2, env.device, approach_height=0.10)
  ref.update(head(), nail_top(), torch.zeros(2, dtype=torch.long, device=env.device))
  n = ref.playback_length()
  cols = arm_dof_cols(env)

  recon_cols: list[torch.Tensor] = []
  qfrc_cols: list[torch.Tensor] = []
  contact_flags: list[torch.Tensor] = []  # (2,) bool per step
  cr_traces: list[torch.Tensor] = []  # (2, nv) per step

  for k in range(1, n + HOLD_STEPS + 1):
    t = ref.playback_target(min(k, n))
    t_lag = ref.playback_target(min(max(k - 1, 1), n))
    target = torch.cat([t[:1], t_lag[1:2]])
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    env.step(action)

    recon = reconstruct_qfrc_from_efc(env)
    qfrc = robot.data._joint_dof_field("qfrc_constraint")
    recon_cols.append(recon[:, cols].clone())
    qfrc_cols.append(qfrc[:, arm_joint_ids].clone())

    in_c = (contact_sensor.data.found > 0).any(dim=-1)  # (2,)
    contact_flags.append(in_c.clone())
    cr_traces.append(contact_row_qfrc(env).clone())

    # Break on ANY termination (2026-07-13): the follow-through reference now
    # terminates IN-SCRIPT, so the lead world finishes ~2 steps before its
    # lagged twin — stepping a terminated world with auto_reset=False raises.
    # The 1-step stagger still guarantees, BEFORE the lead world terminates,
    # (a) a step where exactly one world is in contact (the lead contacts one
    # phase-step early) and (b) first contact in BOTH worlds (contact spans
    # ~3 steps; the lagged world contacts one step after the lead).
    if bool(env.reset_terminated.any()):
      break

  env.close()
  return {
    "recon_cols": recon_cols,
    "qfrc_cols": qfrc_cols,
    "contact_flags": contact_flags,
    "cr_traces": cr_traces,
  }


# --- Step 1/2/3: exact-reconstruction invariant (1 env) -----------------------------------------


class TestExactReconstructionInvariant:
  def test_saw_contact(self, strike_trace_1env):
    assert any(strike_trace_1env["contact_flags"]), (
      "reference strike never reached the nail — nothing exercised the contact path"
    )

  def test_invariant_holds_every_control_step(self, strike_trace_1env):
    recon_cols = strike_trace_1env["recon_cols"]
    qfrc_cols = strike_trace_1env["qfrc_cols"]
    assert len(recon_cols) > 0
    for step, (recon, qfrc) in enumerate(zip(recon_cols, qfrc_cols)):
      torch.testing.assert_close(
        recon, qfrc, rtol=1e-4, atol=1e-6,
        msg=lambda m, step=step: f"control step {step}: {m}",
      )


# --- Step 4: contact-row filter -------------------------------------------------------------


class TestContactRowFilter:
  def test_nonzero_on_some_in_contact_step(self, strike_trace_1env):
    contact_flags = strike_trace_1env["contact_flags"]
    cr_traces = strike_trace_1env["cr_traces"]
    nonzero_seen = any(
      bool(cr.abs().sum() > 0) for cr, in_c in zip(cr_traces, contact_flags) if in_c
    )
    assert nonzero_seen, "contact_row_qfrc was zero on every in-contact step"

  def test_exactly_zero_off_contact(self, strike_trace_1env):
    contact_flags = strike_trace_1env["contact_flags"]
    cr_traces = strike_trace_1env["cr_traces"]
    for step, (cr, in_c) in enumerate(zip(cr_traces, contact_flags)):
      if not in_c:
        assert float(cr.abs().sum()) == 0.0, (
          f"control step {step}: contact_row_qfrc nonzero with no contact ({cr})"
        )


# --- Step 5: num_envs=2 attribution ---------------------------------------------------------


class TestTwoEnvAttribution:
  def test_invariant_holds_per_world(self, strike_trace_2env):
    recon_cols = strike_trace_2env["recon_cols"]
    qfrc_cols = strike_trace_2env["qfrc_cols"]
    assert len(recon_cols) > 0
    for step, (recon, qfrc) in enumerate(zip(recon_cols, qfrc_cols)):
      torch.testing.assert_close(
        recon, qfrc, rtol=1e-4, atol=1e-6,
        msg=lambda m, step=step: f"control step {step} (both worlds): {m}",
      )

  def test_saw_contact_both_worlds(self, strike_trace_2env):
    contact_flags = strike_trace_2env["contact_flags"]
    ever_contact = torch.zeros(2, dtype=torch.bool)
    for in_c in contact_flags:
      ever_contact |= in_c
    assert bool(ever_contact.all()), (
      f"expected both worlds to strike the nail at some point, got {ever_contact.tolist()}"
    )

  def test_rows_attributed_to_the_contacting_world_via_worldid(self, strike_trace_2env):
    """The core Task-9-gating assertion: find a step where EXACTLY ONE world is in contact
    (the two envs' phases are offset by one control step precisely so this occurs), and confirm
    contact_row_qfrc attributes nonzero force to that world only and exactly zero to the other —
    i.e. per-world efc attribution via contact.worldid is real, not an artifact of both worlds
    happening to be in contact/not-in-contact together."""
    contact_flags = strike_trace_2env["contact_flags"]
    cr_traces = strike_trace_2env["cr_traces"]

    mismatch_steps = [
      step
      for step, in_c in enumerate(contact_flags)
      if bool(in_c[0]) != bool(in_c[1])
    ]
    assert mismatch_steps, (
      "no control step found where exactly one world was in contact — cannot exercise "
      "per-world attribution with this phase offset"
    )
    for step in mismatch_steps:
      in_c = contact_flags[step]
      cr = cr_traces[step]
      contacting = 0 if bool(in_c[0]) else 1
      other = 1 - contacting
      assert float(cr[contacting].abs().sum()) > 0.0, (
        f"control step {step}: world {contacting} is in contact but contact_row_qfrc is zero"
      )
      assert float(cr[other].abs().sum()) == 0.0, (
        f"control step {step}: world {other} is NOT in contact but contact_row_qfrc is "
        f"nonzero ({cr[other]}) — attribution leaked across worlds"
      )


# --- Task 9 Step 1(a): ContactRowImpulseAccumulator vs. the RAW-signal shipped peak -------------


@pytest.fixture(scope="module")
def strike_trace_1env_rows_vs_shipped():
  """Drive the same open-loop reference strike as ``strike_trace_1env`` with the SHIPPED config
  (``subtract_baseline=True``, the C2 enforced mode). Both the shipped
  ``SubstepImpulseAccumulator`` and the ``ContactRowImpulseAccumulator`` are wired simultaneously
  by ``z1_hammer_env_cfg(cat_impulse=True)``, so a single env exercises both.

  History (adversarial-review I10, fixed 2026-07-14): this fixture used to FORCE
  ``subtract_baseline=False`` and the test asserted ``rows <= 1.05 x raw`` — a FALSIFIED
  invariant (joint3's dof-friction partially CANCELS the contact reaction inside raw
  ``qfrc_constraint``, so rows can genuinely EXCEED raw; measured ~1.2-2.9x over — see
  IMPULSE_CAT_IMPL_PLAN.md "Track-2 finding" / task-10-report.md, adjudicated 2026-07-10).
  It passed on this fixture's straight-down strike by luck — a latent landmine teaching the
  wrong invariant. The replacement asserts what Task 10 actually established: the shipped
  baseline-subtracted quantity CONVERGES to the rigorous contact-row ground truth on the
  load-bearing joints (joint3 measured sub=0.0420 vs rows=0.0420, ~0% residual)."""
  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device="cpu")
  head, nail_top, _ = _build_driving_helpers(env)

  env.reset()
  ref = SingleStrikeReference(1, env.device, approach_height=0.10)
  ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
  n = ref.playback_length()

  for k in range(1, n + HOLD_STEPS + 1):
    target = ref.playback_target(min(k, n))
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    env.step(action)
    if bool(env.reset_terminated.any()):
      break

  acc = env._hammer_substep_impulse
  robot_joint_names = env.scene["robot"].joint_names
  resolved_order = [robot_joint_names[int(i)] for i in acc._joint_ids]
  shipped_perjoint = acc._episode_peak_perjoint[0].clone()  # (6,) baseline-subtracted peaks
  shipped_peak = acc._episode_peak.clone()
  rows_peak = env._hammer_substep_impulse_rows._episode_peak.clone()
  env.close()
  return {
    "shipped_peak": shipped_peak,
    "shipped_perjoint": shipped_perjoint,
    "rows_peak": rows_peak,
    "resolved_order": resolved_order,
  }


class TestContactRowAccumulatorVsShipped:
  def test_rows_peak_positive_and_shape_matches_shipped(self, strike_trace_1env_rows_vs_shipped):
    shipped_peak = strike_trace_1env_rows_vs_shipped["shipped_peak"]
    rows_peak = strike_trace_1env_rows_vs_shipped["rows_peak"]
    assert rows_peak.shape == shipped_peak.shape
    assert bool((rows_peak > 0).all()), rows_peak

  def test_shipped_subtracted_converges_to_rows_ground_truth(
    self, strike_trace_1env_rows_vs_shipped
  ):
    # The I10 replacement invariant (see fixture docstring): on the worst (binding) joint, the
    # shipped baseline-subtracted episode-peak must agree with the contact-row ground truth's
    # worst-joint episode-peak. Both are per-event sums for this sub-window single strike
    # (sliding window == full event sum below 25 substeps), so they are directly comparable.
    # Tolerance 25%: Task-10 measured ~0% residual on the load-bearing joints; 25% still
    # falsifies both failure modes this guards (baseline subtraction broken -> residual jumps to
    # the ~45% dof-friction share; row attribution broken -> rows collapses toward 0).
    shipped_perjoint = strike_trace_1env_rows_vs_shipped["shipped_perjoint"]
    rows_peak = float(strike_trace_1env_rows_vs_shipped["rows_peak"][0])
    shipped_worst = float(shipped_perjoint.max())
    assert rows_peak > 0.0
    rel = abs(shipped_worst - rows_peak) / rows_peak
    assert rel <= 0.25, (
      f"shipped baseline-subtracted worst-joint peak {shipped_worst:.4f} vs contact-row ground "
      f"truth {rows_peak:.4f}: residual {rel:.1%} > 25% — the enforced quantity has drifted "
      "from the rigorous ground truth (Task 10 measured ~0% on the binding joints)."
    )


class TestJointOrderPin:
  """Adversarial-review I11 (2026-07-14): IMP_J_LIMIT[1]=3.28 (joint2's 2x cap, tau_rated=60)
  rides on POSITIONAL alignment between the cap vector and the accumulator's resolved joint
  order. test_configs.py only set-compares the names; a chain rebuild (the L6 fixture change
  already happened once) that reorders joints in the compiled model would silently misassign
  every cap. This pins the end-to-end chain on the COMPILED model: entity joint order ->
  SceneEntityCfg resolution -> the accumulator's _joint_ids -> positional cap pairing."""

  def test_resolved_joint_order_matches_cap_vector_order(
    self, strike_trace_1env_rows_vs_shipped
  ):
    from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT

    order = strike_trace_1env_rows_vs_shipped["resolved_order"]
    assert order == list(ARM), (
      f"resolved arm-joint order {order} != expected {list(ARM)} — IMP_J_LIMIT and every "
      "per-joint cap/metric would be positionally misassigned."
    )
    assert len(IMP_J_LIMIT) == len(ARM)
    # joint2 (index 1) is the heavy shoulder: tau_rated 60 vs 30 -> its cap must be exactly 2x
    # the others', and it must sit at the position where "joint2" resolved.
    assert order[1] == "joint2"
    others = [v for k, v in enumerate(IMP_J_LIMIT) if k != 1]
    assert all(abs(v - others[0]) < 1e-9 for v in others)
    assert abs(IMP_J_LIMIT[1] - 2.0 * others[0]) < 1e-9


# --- Task 9 Step 1(b): synthetic pulse-semantics unit test (fake env, mirrors --------------------
# --- tests/test_impulse_bound.py's SubstepImpulseAccumulator pattern) ----------------------------

DT = 0.002  # physics_dt @ 500 Hz
DEC = 10    # decimation (substeps per control step)


def _rows_acc(B: int, n_joints: int = 6):
  """ContactRowImpulseAccumulator + feed(qfrc_rows, found) closure driving one substep per call.
  Bypasses __init__ (object.__new__) the same way tests/test_impulse_bound.py's ``_acc`` does,
  and monkeypatches the module-level ``contact_row_qfrc`` the class calls internally."""
  a = object.__new__(ContactRowImpulseAccumulator)
  a._enabled = True
  a._cols = torch.arange(n_joints)
  a._dec = DEC
  a._i = 0
  a._pulse = torch.zeros(B, n_joints)
  a._running = torch.zeros(B, n_joints)
  a._in_contact_prev = torch.zeros(B, dtype=torch.bool)
  a._episode_peak = torch.zeros(B)
  sensor_data = SimpleNamespace(found=None)
  a._sensor = SimpleNamespace(data=sensor_data)
  env = SimpleNamespace(physics_dt=DT)

  def feed(qfrc_rows: torch.Tensor, found: torch.Tensor, monkeypatch) -> torch.Tensor:
    # qfrc_rows must be full (B, nv)-shaped so `[:, self._cols]` (cols=arange(n_joints)) selects
    # it identity-wise, mirroring how the real class indexes contact_row_qfrc(env)'s (B, nv) output.
    monkeypatch.setattr(contact_row_impulse, "contact_row_qfrc", lambda e: qfrc_rows)
    sensor_data.found = found
    return a(env)

  return a, feed


def test_rows_off_contact_accumulates_nothing(monkeypatch):
  acc, feed = _rows_acc(B=2)
  for _ in range(5):
    feed(torch.full((2, 6), 5.0), torch.zeros(2, 1), monkeypatch)
  assert torch.allclose(acc.impulse, torch.zeros(2, 6)), acc.impulse


def test_rows_accumulates_rectified_qfrc_dt_during_contact(monkeypatch):
  acc, feed = _rows_acc(B=1)
  q = torch.tensor([[1.0, -2.0, 3.0, 0.0, 0.0, 0.0]])
  for _ in range(3):
    feed(q, torch.ones(1, 1), monkeypatch)
  expected = q.abs() * DT * 3
  assert torch.allclose(acc.impulse, expected, atol=1e-9), (acc.impulse, expected)


def test_rows_rising_edge_opens_a_fresh_window(monkeypatch):
  # A window must start from 0 at the first in-contact substep, not carry stale running state.
  acc, feed = _rows_acc(B=1)
  q = torch.tensor([[2.0, 0, 0, 0, 0, 0]])
  out = feed(q, torch.ones(1, 1), monkeypatch)
  assert torch.allclose(out, torch.tensor([2.0 * DT]), atol=1e-9)


def test_rows_pulse_visible_for_remainder_of_closing_step_only(monkeypatch):
  acc, feed = _rows_acc(B=1)
  q = torch.tensor([[2.0, 0, 0, 0, 0, 0]])
  for _ in range(3):
    feed(q, torch.ones(1, 1), monkeypatch)          # substeps 0-2: in contact
  feed(torch.zeros(1, 6), torch.zeros(1, 1), monkeypatch)  # substep 3: falling edge -> pulse latched
  total = torch.tensor(2.0 * DT * 3)
  for _ in range(6):                                 # substeps 4-9: off contact, same control step
    feed(torch.zeros(1, 6), torch.zeros(1, 1), monkeypatch)
    assert torch.allclose(acc.impulse[0, 0], total, atol=1e-9)
  feed(torch.zeros(1, 6), torch.zeros(1, 1), monkeypatch)  # substep 10 = next control step -> cleared
  assert acc.impulse[0, 0].item() == 0.0


def test_rows_two_windows_closing_in_one_step_max_combine(monkeypatch):
  acc, feed = _rows_acc(B=1)
  big = torch.tensor([[3.0, 0, 0, 0, 0, 0]])
  small = torch.tensor([[0.2, 0, 0, 0, 0, 0]])
  for _ in range(3):
    feed(big, torch.ones(1, 1), monkeypatch)                 # substeps 0-2: big window (3*3*dt)
  feed(torch.zeros(1, 6), torch.zeros(1, 1), monkeypatch)    # substep 3: big closes
  feed(small, torch.ones(1, 1), monkeypatch)                 # substep 4: tap opens
  feed(torch.zeros(1, 6), torch.zeros(1, 1), monkeypatch)    # substep 5: tap closes (0.2*dt)
  # both closed within the SAME control step: pulse = max(big, tap) = big
  assert torch.allclose(acc.impulse[0, 0], torch.tensor(3.0 * DT * 3), atol=1e-9), acc.impulse


def test_rows_per_env_contact_independence(monkeypatch):
  acc, feed = _rows_acc(B=2)
  q = torch.tensor([[3.0, 0, 0, 0, 0, 0], [3.0, 0, 0, 0, 0, 0]])
  found = torch.tensor([[1.0], [0.0]])  # env0 in contact, env1 not
  feed(q, found, monkeypatch)
  assert acc.impulse[0, 0] > 0 and acc.impulse[1, 0] == 0, acc.impulse


def test_rows_reset_zeros_buffers(monkeypatch):
  acc, feed = _rows_acc(B=2)
  feed(torch.ones(2, 6), torch.ones(2, 1), monkeypatch)
  assert acc.impulse.abs().sum() > 0
  acc.reset(None)
  assert torch.allclose(acc.impulse, torch.zeros(2, 6))
  assert acc._episode_peak.abs().sum() == 0


def test_rows_reset_subset_of_envs(monkeypatch):
  acc, feed = _rows_acc(B=2)
  feed(torch.ones(2, 6), torch.ones(2, 1), monkeypatch)
  acc.reset(torch.tensor([0]))
  assert torch.allclose(acc.impulse[0], torch.zeros(6))
  assert acc.impulse[1].abs().sum() > 0


def test_rows_returns_episode_peak_shape_B_and_survives_pulse_clear(monkeypatch):
  acc, feed = _rows_acc(B=3)
  out = feed(torch.ones(3, 6), torch.ones(3, 1), monkeypatch)
  assert out.shape == (3,)
  for _ in range(DEC + 2):
    out = feed(torch.zeros(3, 6), torch.zeros(3, 1), monkeypatch)
  assert torch.allclose(out, torch.full((3,), 1.0 * DT), atol=1e-9), out
  acc.reset(None)
  assert acc._episode_peak.abs().sum() == 0
