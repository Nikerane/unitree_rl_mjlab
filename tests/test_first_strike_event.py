"""Unit tests for the shared 500 Hz first-strike event tracker.

The fixtures drive mutable tensor state one physics substep at a time.  Contact
onset comes from one sensor while object-side force comes from a distinct
net-force sensor, matching the real task wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.first_strike import (
  REASON_NONE,
  REASON_SUCCESS,
  REASON_WINDOW,
  FirstStrikeEventTracker,
  _ENV_FIRST_STRIKE_ATTR,
)


DT = 0.002
_ROBOT_CFG = SimpleNamespace(name="robot", site_ids=[0])
_NAIL_CFG = SimpleNamespace(name="nail_block", joint_ids=[0])


def _values(value, batch: int, *, dtype=torch.float32) -> torch.Tensor:
  tensor = torch.as_tensor(value, dtype=dtype)
  if tensor.ndim == 0:
    tensor = tensor.repeat(batch)
  return tensor


def _tracker(batch: int = 1, *, window: int = 25, progress_eps: float = 5e-4):
  robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(batch, 1, 3)))
  nail = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(batch, 1)))
  contact = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(batch, 1)))
  impulse = SimpleNamespace(data=SimpleNamespace(force=torch.zeros(batch, 1, 3)))
  env = SimpleNamespace(
    num_envs=batch,
    device="cpu",
    physics_dt=DT,
    scene={
      "robot": robot,
      "nail_block": nail,
      "hammer_nail_contact": contact,
      "hammer_nail_impulse": impulse,
    },
  )
  cfg = SimpleNamespace(
    params={
      "contact_sensor_name": "hammer_nail_contact",
      "impulse_sensor_name": "hammer_nail_impulse",
      "robot_cfg": _ROBOT_CFG,
      "nail_cfg": _NAIL_CFG,
      "axis": (0.0, 0.0, -1.0),
      "window_substeps": window,
      "progress_eps": progress_eps,
    }
  )
  tracker = FirstStrikeEventTracker(cfg=cfg, env=env)
  assert getattr(env, _ENV_FIRST_STRIKE_ATTR) is tracker

  def step(*, head_z, depth, contact_on, downward_force=0.0):
    robot.data.site_pos_w[:, 0, 2] = _values(head_z, batch)
    nail.data.joint_pos[:, 0] = _values(depth, batch)
    contact.data.found[:, 0] = _values(contact_on, batch)
    impulse.data.force.zero_()
    impulse.data.force[:, 0, 2] = -_values(downward_force, batch)
    return tracker(env)

  return tracker, step


def _arm(step, *, head_z=(0.100, 0.096), depth=(0.001, 0.002)) -> None:
  step(head_z=head_z[0], depth=depth[0], contact_on=False)
  step(head_z=head_z[1], depth=depth[1], contact_on=False)


def test_first_contact_uses_onset_minus_previous_position():
  tracker, step = _tracker()
  _arm(step)

  step(head_z=0.090, depth=0.002, contact_on=True, downward_force=10.0)

  # (0.090 - 0.096) / 0.002 projected onto -Z = 3 m/s.  The stale
  # off/off difference would be only 2 m/s.
  assert tracker.v_precontact[0].item() == pytest.approx(3.0)
  assert not tracker.finalized[0]


def test_started_is_false_until_onset_and_stays_true_after_finalization():
  tracker, step = _tracker(window=2)
  assert not tracker.started[0]

  _arm(step, depth=(0.0, 0.0))
  assert not tracker.started[0]

  step(head_z=0.090, depth=0.001, contact_on=True, downward_force=10.0)
  assert tracker.started[0]
  assert not tracker.finalized[0]

  step(head_z=0.089, depth=0.002, contact_on=False)
  assert tracker.finalized[0]
  assert tracker.started[0]

  step(head_z=0.088, depth=0.030, contact_on=True, downward_force=100.0)
  assert tracker.started[0]


def test_contact_depth_is_previous_substep_depth():
  tracker, step = _tracker()
  _arm(step, depth=(0.001, 0.002))

  step(head_z=0.090, depth=0.007, contact_on=True, downward_force=10.0)

  assert tracker.depth_at_contact[0].item() == pytest.approx(0.002)
  assert tracker.peak_depth[0].item() == pytest.approx(0.007)


def test_window_age_counts_wall_time_across_contact_gaps():
  tracker, step = _tracker(window=4)
  _arm(step, depth=(0.0, 0.0))

  step(head_z=0.090, depth=0.001, contact_on=True, downward_force=10.0)   # age 1
  step(head_z=0.089, depth=0.002, contact_on=False, downward_force=100.0)  # age 2
  step(head_z=0.088, depth=0.003, contact_on=False, downward_force=100.0)  # age 3
  assert not tracker.finalized[0]
  step(head_z=0.087, depth=0.004, contact_on=True, downward_force=20.0)   # age 4

  assert tracker.finalized[0]
  assert tracker.reason[0].item() == REASON_WINDOW
  assert tracker.delivered[0].item() == pytest.approx((10.0 + 20.0) * DT)
  assert tracker.peak_depth[0].item() == pytest.approx(0.004)


def test_success_substep_is_included_and_later_samples_are_frozen():
  tracker, step = _tracker()
  _arm(step, depth=(0.0, 0.0))

  step(head_z=0.090, depth=0.005, contact_on=True, downward_force=10.0)
  step(head_z=0.089, depth=0.030, contact_on=True, downward_force=20.0)

  assert tracker.finalized[0]
  assert tracker.reason[0].item() == REASON_SUCCESS
  assert tracker.productive[0]
  assert tracker.delivered[0].item() == pytest.approx((10.0 + 20.0) * DT)
  assert tracker.peak_depth[0].item() == pytest.approx(0.030)
  frozen = (
    tracker.v_precontact.clone(),
    tracker.delivered.clone(),
    tracker.peak_depth.clone(),
    tracker.depth_at_contact.clone(),
    tracker.productive.clone(),
    tracker.reason.clone(),
  )

  step(head_z=0.070, depth=0.032, contact_on=True, downward_force=100.0)

  current = (
    tracker.v_precontact,
    tracker.delivered,
    tracker.peak_depth,
    tracker.depth_at_contact,
    tracker.productive,
    tracker.reason,
  )
  for actual, expected in zip(current, frozen, strict=True):
    assert torch.equal(actual, expected)


def test_success_on_substep_25_precedes_window_finalization():
  tracker, step = _tracker(window=25)
  _arm(step, depth=(0.0, 0.0))

  step(head_z=0.090, depth=0.001, contact_on=True, downward_force=1.0)  # age 1
  for age in range(2, 25):
    step(
      head_z=0.090 - age * 1e-4,
      depth=0.001 + age * 1e-5,
      contact_on=True,
      downward_force=1.0,
    )
  assert not tracker.finalized[0]

  step(head_z=0.0875, depth=0.030, contact_on=True, downward_force=1.0)  # age 25

  assert tracker.finalized[0]
  assert tracker.reason[0].item() == REASON_SUCCESS
  assert tracker.delivered[0].item() == pytest.approx(25.0 * DT)
  assert tracker.peak_depth[0].item() == pytest.approx(0.030)


def test_delayed_recontact_cannot_change_final_snapshot():
  tracker, step = _tracker(window=3)
  _arm(step, depth=(0.0, 0.0))

  step(head_z=0.090, depth=0.001, contact_on=True, downward_force=10.0)
  step(head_z=0.091, depth=0.001, contact_on=False)
  step(head_z=0.092, depth=0.001, contact_on=False)
  assert tracker.finalized[0]
  before = (
    tracker.v_precontact.clone(),
    tracker.delivered.clone(),
    tracker.peak_depth.clone(),
    tracker.depth_at_contact.clone(),
    tracker.productive.clone(),
    tracker.reason.clone(),
  )

  for _ in range(30):
    step(head_z=0.100, depth=0.001, contact_on=False)
  step(head_z=0.080, depth=0.032, contact_on=True, downward_force=500.0)

  after = (
    tracker.v_precontact,
    tracker.delivered,
    tracker.peak_depth,
    tracker.depth_at_contact,
    tracker.productive,
    tracker.reason,
  )
  for actual, expected in zip(after, before, strict=True):
    assert torch.equal(actual, expected)


def test_unproductive_window_finalizes_but_is_not_productive():
  tracker, step = _tracker(window=3, progress_eps=5e-4)
  _arm(step, depth=(0.002, 0.002))

  step(head_z=0.090, depth=0.0022, contact_on=True, downward_force=5.0)
  step(head_z=0.089, depth=0.0024, contact_on=False)
  step(head_z=0.088, depth=0.0024, contact_on=False)

  assert tracker.finalized[0]
  assert tracker.reason[0].item() == REASON_WINDOW
  assert not tracker.productive[0]
  assert tracker.peak_depth[0].item() == pytest.approx(0.0024)


def test_reset_mid_contact_requires_clean_free_flight_rearm():
  tracker, step = _tracker()
  _arm(step, depth=(0.0, 0.0))
  step(head_z=0.090, depth=0.005, contact_on=True, downward_force=10.0)

  tracker.reset(None)
  step(head_z=0.080, depth=0.006, contact_on=True, downward_force=100.0)
  step(head_z=0.078, depth=0.006, contact_on=False)
  step(head_z=0.076, depth=0.007, contact_on=True, downward_force=100.0)
  assert tracker.v_precontact[0].item() == 0.0
  assert tracker.delivered[0].item() == 0.0
  assert tracker.reason[0].item() == REASON_NONE

  step(head_z=0.074, depth=0.007, contact_on=False)
  step(head_z=0.072, depth=0.007, contact_on=False)
  step(head_z=0.068, depth=0.008, contact_on=True, downward_force=5.0)

  assert tracker.v_precontact[0].item() == pytest.approx(2.0, abs=1e-5)
  assert tracker.delivered[0].item() == pytest.approx(5.0 * DT)
  assert tracker.depth_at_contact[0].item() == pytest.approx(0.007)


def test_subset_reset_does_not_change_other_environment():
  tracker, step = _tracker(batch=2)
  step(head_z=[0.100, 0.200], depth=[0.0, 0.0], contact_on=[False, False])
  step(head_z=[0.096, 0.194], depth=[0.0, 0.0], contact_on=[False, False])
  step(
    head_z=[0.090, 0.184],
    depth=[0.005, 0.030],
    contact_on=[True, True],
    downward_force=[10.0, 20.0],
  )
  assert tracker.finalized.tolist() == [False, True]
  env1 = (
    tracker.finalized[1].clone(),
    tracker.productive[1].clone(),
    tracker.v_precontact[1].clone(),
    tracker.delivered[1].clone(),
    tracker.peak_depth[1].clone(),
    tracker.depth_at_contact[1].clone(),
    tracker.reason[1].clone(),
  )

  tracker.reset(torch.tensor([0]))

  assert not tracker.finalized[0]
  assert tracker.reason[0].item() == REASON_NONE
  current_env1 = (
    tracker.finalized[1],
    tracker.productive[1],
    tracker.v_precontact[1],
    tracker.delivered[1],
    tracker.peak_depth[1],
    tracker.depth_at_contact[1],
    tracker.reason[1],
  )
  for actual, expected in zip(current_env1, env1, strict=True):
    assert torch.equal(actual, expected)
