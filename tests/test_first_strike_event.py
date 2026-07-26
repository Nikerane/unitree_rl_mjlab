"""Unit tests for the shared 500 Hz first-strike event tracker.

The fixtures drive mutable tensor state one physics substep at a time.  Contact
onset comes from one sensor while object-side force comes from a distinct
net-force sensor, matching the real task wiring.
"""

from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.first_strike import (
  REASON_NONE,
  REASON_SUCCESS,
  REASON_WINDOW,
  FirstStrikeEventTracker,
  _ENV_FIRST_STRIKE_ATTR,
)


DT = 0.002
QUALITY_SLOTS = 2
_ROBOT_CFG = SimpleNamespace(name="robot", site_ids=[0])
_NAIL_CFG = SimpleNamespace(name="nail_block", joint_ids=[0], site_ids=[0])


def _values(value, batch: int, *, dtype=torch.float32) -> torch.Tensor:
  tensor = torch.as_tensor(value, dtype=dtype)
  if tensor.ndim == 0:
    tensor = tensor.repeat(batch)
  return tensor


def _vectors(value, batch: int, width: int) -> torch.Tensor:
  tensor = torch.as_tensor(value, dtype=torch.float32)
  if tensor.ndim == 1:
    assert tensor.shape == (width,)
    tensor = tensor.unsqueeze(0).expand(batch, -1)
  assert tensor.shape == (batch, width)
  return tensor


def _tracker(
  batch: int = 1,
  *,
  window: int = 25,
  progress_eps: float = 5e-4,
  quality_instrumentation: bool = True,
):
  robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(batch, 1, 3)))
  nail = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=torch.zeros(batch, 1),
      site_pos_w=torch.zeros(batch, 1, 3),
    )
  )
  contact = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(batch, 1)))
  impulse = SimpleNamespace(data=SimpleNamespace(force=torch.zeros(batch, 1, 3)))
  quality = SimpleNamespace(
    cfg=SimpleNamespace(num_slots=QUALITY_SLOTS),
    data=SimpleNamespace(
      found=torch.zeros(batch, QUALITY_SLOTS),
      force=torch.zeros(batch, QUALITY_SLOTS, 3),
      pos=torch.zeros(batch, QUALITY_SLOTS, 3),
      normal=torch.zeros(batch, QUALITY_SLOTS, 3),
    ),
  )

  def geom(name):
    assert name == "nail_block/nail_head"
    return SimpleNamespace(size=(0.012,))

  scene = {
    "robot": robot,
    "nail_block": nail,
    "hammer_nail_contact": contact,
    "hammer_nail_impulse": impulse,
  }
  if quality_instrumentation:
    scene["hammer_nail_quality"] = quality
  env = SimpleNamespace(
    num_envs=batch,
    device="cpu",
    physics_dt=DT,
    sim=SimpleNamespace(mj_model=SimpleNamespace(geom=geom)),
    scene=scene,
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
  if quality_instrumentation:
    cfg.params["quality_sensor_name"] = "hammer_nail_quality"
  tracker = FirstStrikeEventTracker(cfg=cfg, env=env)
  assert getattr(env, _ENV_FIRST_STRIKE_ATTR) is tracker

  def step(
    *,
    head_z,
    depth,
    contact_on,
    downward_force=0.0,
    transverse_force=(0.0, 0.0),
    quality_pos=(0.0, 0.0, 0.1),
    quality_normal_force=None,
    quality_normal=(0.0, 0.0, -1.0),
    quality_found=None,
    nail_top=(0.0, 0.0, 0.1),
  ):
    robot.data.site_pos_w[:, 0, 2] = _values(head_z, batch)
    nail.data.joint_pos[:, 0] = _values(depth, batch)
    nail.data.site_pos_w[:, 0, :] = _vectors(nail_top, batch, 3)
    contact.data.found[:, 0] = _values(contact_on, batch)

    impulse.data.force.zero_()
    impulse.data.force[:, 0, :2] = _vectors(transverse_force, batch, 2)
    impulse.data.force[:, 0, 2] = -_values(downward_force, batch)

    quality.data.found.zero_()
    quality.data.force.zero_()
    quality.data.pos.zero_()
    quality.data.normal.zero_()
    found_value = contact_on if quality_found is None else quality_found
    normal_force = (
      downward_force
      if quality_normal_force is None
      else quality_normal_force
    )
    quality.data.found[:, 0] = _values(found_value, batch)
    quality.data.force[:, 0, 0] = _values(normal_force, batch)
    quality.data.pos[:, 0, :] = _vectors(quality_pos, batch, 3)
    quality.data.normal[:, 0, :] = _vectors(quality_normal, batch, 3)
    return tracker(env)

  return tracker, step


def _arm(step, *, head_z=(0.100, 0.096), depth=(0.001, 0.002)) -> None:
  step(head_z=head_z[0], depth=depth[0], contact_on=False)
  step(head_z=head_z[1], depth=depth[1], contact_on=False)


def test_event_config_adds_dedicated_quality_sensor_without_changing_contact_shape():
  cfg = z1_hammer_env_cfg(
    play=True,
    cat_impulse=True,
    event_correct=True,
    quality_instrumentation=True,
  )
  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}

  contact = sensors["hammer_nail_contact"]
  quality = sensors["hammer_nail_quality"]
  assert contact.fields == ("found", "force")
  assert contact.reduce == "maxforce"
  assert contact.num_slots == 1
  assert contact.track_air_time is True
  assert quality.primary == contact.primary
  assert quality.secondary == contact.secondary
  assert quality.fields == ("found", "force", "pos", "normal")
  assert quality.reduce == "maxforce"
  assert quality.num_slots == 8
  assert quality.track_air_time is False
  assert quality.global_frame is False
  assert (
    cfg.metrics["first_strike"].params["quality_sensor_name"]
    == "hammer_nail_quality"
  )


def test_uninstrumented_tracker_preserves_event_semantics_without_quality_sensor():
  """Removing passive geometry instrumentation must not disturb event credit."""
  tracker, step = _tracker(window=1, quality_instrumentation=False)
  _arm(step)
  step(head_z=0.090, depth=0.003, contact_on=True, downward_force=10.0)

  assert tracker.finalized[0]
  assert tracker.productive[0]
  assert tracker.v_precontact[0].item() > 0.0
  assert tracker.delivered[0].item() > 0.0
  torch.testing.assert_close(tracker.contact_quality, torch.zeros(1))
  assert not tracker.contact_quality_valid[0]
  assert not tracker.contact_quality_overflow[0]
  torch.testing.assert_close(tracker.contact_normal_axiality, torch.zeros(1))


def test_slot_probe_rejects_an_overflowing_next_larger_comparator():
  probe_path = (
    Path(__file__).resolve().parents[1]
    / "docs/results/assets/2026-07-17_fixed_impedance_diag/probes"
    / "contact_quality_sensor.py"
  )
  qualify = runpy.run_path(str(probe_path))["qualify"]

  def row(slots, *, overflow):
    return {
      "candidate_slots": slots,
      "contact_substeps": 1,
      "found_finite": True,
      "found_integral": True,
      "found_nonnegative": True,
      "normal_force_positive_n": 1,
      "normal_force_negative_n": 0,
      "contact_quality_valid": True,
      "overflow_n": overflow,
      "contact_point_w": [0.0, 0.0, 0.1],
      "contact_error_m": 0.0,
    }

  chosen, _, failures = qualify(
    [
      row(8, overflow=0),
      row(16, overflow=1),
      row(64, overflow=0),
    ]
  )

  assert chosen is None
  assert any("no candidate" in failure for failure in failures)


def test_slot_probe_can_fallback_when_a_smaller_candidate_overflows():
  probe_path = (
    Path(__file__).resolve().parents[1]
    / "docs/results/assets/2026-07-17_fixed_impedance_diag/probes"
    / "contact_quality_sensor.py"
  )
  qualify = runpy.run_path(str(probe_path))["qualify"]

  def row(slots, *, overflow, valid):
    return {
      "candidate_slots": slots,
      "contact_substeps": 1,
      "found_finite": True,
      "found_integral": True,
      "found_nonnegative": True,
      "normal_force_positive_n": 1,
      "normal_force_negative_n": 0,
      "contact_quality_valid": valid,
      "overflow_n": overflow,
      "contact_point_w": [0.0, 0.0, 0.1],
      "contact_error_m": 0.0,
    }

  chosen, _, failures = qualify(
    [
      row(8, overflow=1, valid=False),
      row(16, overflow=0, valid=True),
      row(64, overflow=0, valid=True),
    ]
  )

  assert chosen == 16
  assert failures == []


def test_onset_latches_contact_quality_time_and_normal_axiality():
  tracker, step = _tracker()
  _arm(step)

  step(
    head_z=0.090,
    depth=0.002,
    contact_on=True,
    downward_force=10.0,
    quality_pos=(0.003, 0.0, 0.1),
    quality_normal=(0.6, 0.0, -0.8),
  )

  torch.testing.assert_close(
    tracker.contact_point_w[0], torch.tensor([0.003, 0.0, 0.1])
  )
  assert tracker.contact_error_m[0].item() == pytest.approx(0.003)
  assert tracker.contact_quality[0].item() == pytest.approx(0.9375)
  assert tracker.contact_quality_valid[0]
  assert not tracker.contact_quality_overflow[0]
  assert tracker.first_contact_time_s[0].item() == pytest.approx(3 * DT)
  assert tracker.contact_normal_axiality[0].item() == pytest.approx(0.8)


def test_contact_geometry_is_latched_only_at_onset():
  tracker, step = _tracker()
  _arm(step)
  step(
    head_z=0.090,
    depth=0.002,
    contact_on=True,
    downward_force=10.0,
    quality_pos=(0.003, 0.0, 0.1),
  )
  frozen = (
    tracker.contact_point_w.clone(),
    tracker.contact_error_m.clone(),
    tracker.contact_quality.clone(),
    tracker.contact_quality_valid.clone(),
    tracker.contact_quality_overflow.clone(),
    tracker.first_contact_time_s.clone(),
    tracker.contact_normal_axiality.clone(),
  )

  step(
    head_z=0.089,
    depth=0.003,
    contact_on=True,
    downward_force=20.0,
    quality_pos=(0.020, 0.0, 0.1),
    quality_normal=(1.0, 0.0, 0.0),
  )

  current = (
    tracker.contact_point_w,
    tracker.contact_error_m,
    tracker.contact_quality,
    tracker.contact_quality_valid,
    tracker.contact_quality_overflow,
    tracker.first_contact_time_s,
    tracker.contact_normal_axiality,
  )
  for actual, expected in zip(current, frozen, strict=True):
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("normal_force", [0.0, -10.0])
def test_nonpositive_quality_force_marks_onset_geometry_invalid(normal_force):
  tracker, step = _tracker()
  _arm(step)

  step(
    head_z=0.090,
    depth=0.002,
    contact_on=True,
    downward_force=10.0,
    quality_normal_force=normal_force,
    quality_pos=(0.003, 0.0, 0.1),
  )

  assert not tracker.contact_quality_valid[0]
  assert not tracker.contact_quality_overflow[0]
  torch.testing.assert_close(tracker.contact_point_w, torch.zeros(1, 3))
  torch.testing.assert_close(tracker.contact_error_m, torch.zeros(1))
  torch.testing.assert_close(tracker.contact_quality, torch.zeros(1))


def test_quality_slot_overflow_fails_closed_at_onset():
  tracker, step = _tracker()
  _arm(step)

  step(
    head_z=0.090,
    depth=0.002,
    contact_on=True,
    downward_force=10.0,
    quality_found=QUALITY_SLOTS + 1,
    quality_pos=(0.003, 0.0, 0.1),
  )

  assert tracker.contact_quality_overflow[0]
  assert not tracker.contact_quality_valid[0]
  torch.testing.assert_close(tracker.contact_point_w, torch.zeros(1, 3))
  torch.testing.assert_close(tracker.contact_error_m, torch.zeros(1))
  torch.testing.assert_close(tracker.contact_quality, torch.zeros(1))


def test_transverse_impulse_integrates_only_during_active_raw_contact():
  tracker, step = _tracker(window=4)
  _arm(step, depth=(0.0, 0.0))

  step(
    head_z=0.090,
    depth=0.001,
    contact_on=True,
    downward_force=10.0,
    transverse_force=(3.0, 4.0),
  )
  step(
    head_z=0.089,
    depth=0.002,
    contact_on=False,
    downward_force=100.0,
    transverse_force=(60.0, 80.0),
  )
  step(
    head_z=0.088,
    depth=0.003,
    contact_on=True,
    downward_force=20.0,
    transverse_force=(0.0, 6.0),
  )
  step(
    head_z=0.087,
    depth=0.004,
    contact_on=False,
    downward_force=100.0,
    transverse_force=(60.0, 80.0),
  )

  assert tracker.finalized[0]
  assert tracker.delivered_transverse[0].item() == pytest.approx(
    (5.0 + 6.0) * DT
  )


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
    tracker.delivered_transverse.clone(),
    tracker.peak_depth.clone(),
    tracker.depth_at_contact.clone(),
    tracker.contact_point_w.clone(),
    tracker.contact_error_m.clone(),
    tracker.contact_quality.clone(),
    tracker.contact_quality_valid.clone(),
    tracker.contact_quality_overflow.clone(),
    tracker.first_contact_time_s.clone(),
    tracker.contact_normal_axiality.clone(),
    tracker.productive.clone(),
    tracker.reason.clone(),
  )

  step(
    head_z=0.070,
    depth=0.032,
    contact_on=True,
    downward_force=100.0,
    transverse_force=(60.0, 80.0),
    quality_pos=(0.020, 0.0, 0.1),
  )

  current = (
    tracker.v_precontact,
    tracker.delivered,
    tracker.delivered_transverse,
    tracker.peak_depth,
    tracker.depth_at_contact,
    tracker.contact_point_w,
    tracker.contact_error_m,
    tracker.contact_quality,
    tracker.contact_quality_valid,
    tracker.contact_quality_overflow,
    tracker.first_contact_time_s,
    tracker.contact_normal_axiality,
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
    tracker.delivered_transverse[1].clone(),
    tracker.peak_depth[1].clone(),
    tracker.depth_at_contact[1].clone(),
    tracker.contact_point_w[1].clone(),
    tracker.contact_error_m[1].clone(),
    tracker.contact_quality[1].clone(),
    tracker.contact_quality_valid[1].clone(),
    tracker.contact_quality_overflow[1].clone(),
    tracker.first_contact_time_s[1].clone(),
    tracker.contact_normal_axiality[1].clone(),
    tracker.reason[1].clone(),
  )

  tracker.reset(torch.tensor([0]))

  assert not tracker.finalized[0]
  assert tracker.reason[0].item() == REASON_NONE
  torch.testing.assert_close(tracker.contact_point_w[0], torch.zeros(3))
  assert tracker.contact_error_m[0].item() == 0.0
  assert tracker.contact_quality[0].item() == 0.0
  assert not tracker.contact_quality_valid[0]
  assert not tracker.contact_quality_overflow[0]
  assert tracker.first_contact_time_s[0].item() == 0.0
  assert tracker.contact_normal_axiality[0].item() == 0.0
  assert tracker.delivered_transverse[0].item() == 0.0
  current_env1 = (
    tracker.finalized[1],
    tracker.productive[1],
    tracker.v_precontact[1],
    tracker.delivered[1],
    tracker.delivered_transverse[1],
    tracker.peak_depth[1],
    tracker.depth_at_contact[1],
    tracker.contact_point_w[1],
    tracker.contact_error_m[1],
    tracker.contact_quality[1],
    tracker.contact_quality_valid[1],
    tracker.contact_quality_overflow[1],
    tracker.first_contact_time_s[1],
    tracker.contact_normal_axiality[1],
    tracker.reason[1],
  )
  for actual, expected in zip(current_env1, env1, strict=True):
    assert torch.equal(actual, expected)
