"""Tests for Cartesian guideline geometry and tracker lifecycle."""

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.guideline import (
  GUIDELINE_NUM_GATES,
  WaypointProgressTracker,
  advance_ordered_gates,
  completed_gate_fraction,
  guideline_perpendicular_error,
  next_gate_vector,
  ordered_gate_progress_reward,
  project_to_reference,
)


_ROBOT_CFG = SimpleNamespace(name="robot", site_ids=[0])
_NAIL_CFG = SimpleNamespace(name="nail_block", site_ids=[0])
METRIC_CFG = SimpleNamespace(
  params={"robot_cfg": _ROBOT_CFG, "nail_cfg": _NAIL_CFG},
)


@pytest.fixture
def tracker_env():
  batch = 2
  robot = SimpleNamespace(
    data=SimpleNamespace(site_pos_w=torch.zeros(batch, 1, 3)),
  )
  nail = SimpleNamespace(
    data=SimpleNamespace(site_pos_w=torch.zeros(batch, 1, 3)),
  )
  robot.data.site_pos_w[:, 0, 2] = 0.7
  first_strike = SimpleNamespace(
    started=torch.zeros(batch, dtype=torch.bool),
  )
  return SimpleNamespace(
    num_envs=batch,
    device="cpu",
    cfg=SimpleNamespace(decimation=2),
    scene={"robot": robot, "nail_block": nail},
    _hammer_first_strike=first_strike,
  )


def _set_head_z(env, values) -> None:
  env.scene["robot"].data.site_pos_w[:, 0, 2] = torch.as_tensor(values)


def test_projection_returns_progress_and_perpendicular_distance():
  """Changing the projection or radial-distance calculation breaks this."""
  entry = torch.tensor([[0.0, 0.0, 1.0]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])
  points = torch.tensor([[0.003, 0.004, 0.50]])

  s, d = project_to_reference(points, entry, nail)

  torch.testing.assert_close(s, torch.tensor([0.5]))
  torch.testing.assert_close(d, torch.tensor([0.005]))


def test_projection_clamps_to_the_finite_reference_segment():
  """Removing segment clamping would report progress outside the reference."""
  entry = torch.tensor([[0.0, 0.0, 1.0]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])
  points = torch.tensor([[0.0, 0.0, -0.20]])

  s, d = project_to_reference(points, entry, nail)

  torch.testing.assert_close(s, torch.tensor([1.0]))
  torch.testing.assert_close(d, torch.tensor([0.20]))


def test_projection_handles_a_zero_length_reference_line():
  """Dividing by the reference length would make a coincident entry/nail NaN."""
  entry = torch.tensor([[1.0, 2.0, 3.0]])
  nail = entry.clone()
  points = torch.tensor([[1.0, 2.0, 3.005]])

  s, d = project_to_reference(points, entry, nail)

  torch.testing.assert_close(s, torch.tensor([0.0]))
  torch.testing.assert_close(d, torch.tensor([0.005]))


def test_swept_crossing_consumes_multiple_ordered_gates():
  """Stopping after one swept gate would lose progress during a large control step."""
  entry = torch.tensor([[0.0, 0.0, 0.7]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])

  count, index = advance_ordered_gates(
    torch.tensor([[0.0, 0.0, 0.65]]),
    torch.tensor([[0.0, 0.0, 0.35]]),
    entry,
    nail,
    torch.tensor([0]),
  )

  assert count.tolist() == [3]
  assert index.tolist() == [3]


def test_first_gate_is_at_one_seventh_of_the_reference():
  """Moving the first gate from 1/7 to 1/6 would miss this swept crossing."""
  entry = torch.tensor([[0.0, 0.0, 0.7]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])

  count, index = advance_ordered_gates(
    torch.tensor([[0.0, 0.0, 0.62]]),
    torch.tensor([[0.0, 0.0, 0.595]]),
    entry,
    nail,
    torch.tensor([0]),
  )

  assert count.tolist() == [1]
  assert index.tolist() == [1]


def test_sixth_gate_is_at_six_sevenths_of_the_reference():
  """Putting the sixth gate at the nail instead of 6/7 would miss this crossing."""
  entry = torch.tensor([[0.0, 0.0, 0.7]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])

  count, index = advance_ordered_gates(
    torch.tensor([[0.0, 0.0, 0.12]]),
    torch.tensor([[0.0, 0.0, 0.08]]),
    entry,
    nail,
    torch.tensor([5]),
  )

  assert count.tolist() == [1]
  assert index.tolist() == [6]


def test_radial_miss_outside_gate_disk_does_not_advance():
  """Treating a gate as an infinite plane would accept an off-path 15.1 mm miss."""
  entry = torch.tensor([[0.0, 0.0, 0.6]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])

  count, index = advance_ordered_gates(
    torch.tensor([[0.0151, 0.0, 0.55]]),
    torch.tensor([[0.0151, 0.0, 0.45]]),
    entry,
    nail,
    torch.tensor([0]),
  )

  assert count.tolist() == [0]
  assert index.tolist() == [0]


def test_backward_crossing_does_not_advance_an_ordered_gate():
  """Ignoring travel direction would reward a retreat through an earlier gate plane."""
  entry = torch.tensor([[0.0, 0.0, 0.6]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])

  count, index = advance_ordered_gates(
    torch.tensor([[0.0, 0.0, 0.45]]),
    torch.tensor([[0.0, 0.0, 0.55]]),
    entry,
    nail,
    torch.tensor([0]),
  )

  assert count.tolist() == [0]
  assert index.tolist() == [0]


def test_revisiting_consumed_gates_does_not_advance():
  """Restarting gate search at zero would let a later path revisit earn duplicate credit."""
  entry = torch.tensor([[0.0, 0.0, 0.6]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])

  count, index = advance_ordered_gates(
    torch.tensor([[0.0, 0.0, 0.55]]),
    torch.tensor([[0.0, 0.0, 0.35]]),
    entry,
    nail,
    torch.tensor([2]),
  )

  assert count.tolist() == [0]
  assert index.tolist() == [2]


def test_segment_that_never_reaches_next_gate_does_not_advance():
  """Using endpoint proximity rather than a swept-plane crossing would advance early."""
  entry = torch.tensor([[0.0, 0.0, 0.6]])
  nail = torch.tensor([[0.0, 0.0, 0.0]])

  count, index = advance_ordered_gates(
    torch.tensor([[0.0, 0.0, 0.55]]),
    torch.tensor([[0.0, 0.0, 0.52]]),
    entry,
    nail,
    torch.tensor([0]),
  )

  assert count.tolist() == [0]
  assert index.tolist() == [0]


def test_zero_length_gate_reference_does_not_advance():
  """A degenerate guideline must be safe and leave per-environment gate state unchanged."""
  entry = torch.tensor([[0.0, 0.0, 0.0]])
  nail = entry.clone()

  count, index = advance_ordered_gates(
    torch.tensor([[0.0, 0.0, 0.1]]),
    torch.tensor([[0.0, 0.0, 0.0]]),
    entry,
    nail,
    torch.tensor([1]),
  )

  assert count.tolist() == [0]
  assert index.tolist() == [1]


def test_partial_reset_clears_only_selected_tracker_rows(tracker_env):
  """Resetting one environment must not erase another environment's progress."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  _set_head_z(tracker_env, [0.59, 0.59])
  tracker(tracker_env)
  tracker_env._hammer_first_strike.started[1] = True
  tracker(tracker_env)

  preserved = {
    "initialized": tracker.initialized[1].clone(),
    "entry": tracker.entry[1].clone(),
    "nail": tracker.nail[1].clone(),
    "previous_head": tracker.previous_head[1].clone(),
    "next_gate": tracker.next_gate[1].clone(),
    "newly_crossed": tracker.newly_crossed[1].clone(),
    "disarmed": tracker.disarmed[1].clone(),
  }
  tracker.reset(torch.tensor([0]))

  assert not tracker.initialized[0]
  torch.testing.assert_close(tracker.entry[0], torch.zeros(3))
  torch.testing.assert_close(tracker.nail[0], torch.zeros(3))
  torch.testing.assert_close(tracker.previous_head[0], torch.zeros(3))
  assert tracker.next_gate[0].item() == 0
  assert tracker.newly_crossed[0].item() == 0
  assert not tracker.disarmed[0]
  for name, expected in preserved.items():
    torch.testing.assert_close(getattr(tracker, name)[1], expected)


def test_partial_reset_row_reanchors_while_initialized_row_advances(tracker_env):
  """A mixed reset mask must suppress only the uninitialized row's crossing."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  _set_head_z(tracker_env, [0.70, 0.59])
  tracker(tracker_env)
  assert tracker.next_gate.tolist() == [0, 1]
  tracker.reset(torch.tensor([0]))
  _set_head_z(tracker_env, [0.40, 0.49])

  tracker(tracker_env)

  assert tracker.initialized.tolist() == [True, True]
  torch.testing.assert_close(tracker.entry[:, 2], torch.tensor([0.40, 0.70]))
  assert tracker.next_gate.tolist() == [0, 2]
  assert tracker.newly_crossed.tolist() == [0, 1]


def test_first_post_reset_call_lazily_freezes_current_anchors(tracker_env):
  """Eager construction/reset anchors would retain stale pre-forward positions."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  tracker.reset(None)
  _set_head_z(tracker_env, [0.75, 0.80])
  tracker_env.scene["nail_block"].data.site_pos_w[:, 0, 2] = torch.tensor([0.05, 0.10])

  tracker(tracker_env)

  torch.testing.assert_close(tracker.entry[:, 2], torch.tensor([0.75, 0.80]))
  torch.testing.assert_close(tracker.nail[:, 2], torch.tensor([0.05, 0.10]))
  assert tracker.initialized.tolist() == [True, True]


def test_first_post_reset_substep_emits_zero_progress(tracker_env):
  """Comparing an uninitialized zero position to the head would create false credit."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  tracker.reset(None)
  _set_head_z(tracker_env, [0.40, 0.45])

  value = tracker(tracker_env)

  assert value.tolist() == [0.0, 0.0]
  assert tracker.newly_crossed.tolist() == [0, 0]
  assert tracker.next_gate.tolist() == [0, 0]


def test_uninitialized_readers_preview_post_forward_geometry_without_mutation(tracker_env):
  """Initial observations must use live post-forward geometry, not zero anchors."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  tracker.reset(None)
  tracker_env.scene["robot"].data.site_pos_w[:, 0, :] = torch.tensor(
    [[0.02, 0.03, 0.70], [-0.01, 0.04, 0.80]],
  )
  tracker_env.scene["nail_block"].data.site_pos_w[:, 0, :] = torch.tensor(
    [[0.02, 0.03, 0.00], [-0.01, 0.04, 0.10]],
  )
  before = (
    tracker.initialized.clone(),
    tracker.entry.clone(),
    tracker.nail.clone(),
    tracker.previous_head.clone(),
    tracker.next_gate.clone(),
    tracker.newly_crossed.clone(),
    tracker.disarmed.clone(),
  )

  vector = next_gate_vector(tracker_env)
  fraction = completed_gate_fraction(tracker_env)
  error = guideline_perpendicular_error(tracker_env)

  torch.testing.assert_close(
    vector,
    torch.tensor([[0.0, 0.0, -0.10], [0.0, 0.0, -0.10]]),
  )
  torch.testing.assert_close(fraction, torch.zeros(2, 1))
  torch.testing.assert_close(error, torch.zeros(2, 1))
  after = (
    tracker.initialized,
    tracker.entry,
    tracker.nail,
    tracker.previous_head,
    tracker.next_gate,
    tracker.newly_crossed,
    tracker.disarmed,
  )
  for actual, expected in zip(after, before, strict=True):
    torch.testing.assert_close(actual, expected)


def test_reward_and_observation_reads_are_idempotent(tracker_env):
  """Readers must not advance gates or consume the one-control-window pulse."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  _set_head_z(tracker_env, [0.59, 0.59])
  tracker_env.scene["robot"].data.site_pos_w[:, 0, 0] = torch.tensor([0.003, 0.004])
  tracker(tracker_env)
  before_index = tracker.next_gate.clone()
  before_pulse = tracker.newly_crossed.clone()

  reward = ordered_gate_progress_reward(tracker_env)
  vector = next_gate_vector(tracker_env)
  fraction = completed_gate_fraction(tracker_env)
  error = guideline_perpendicular_error(tracker_env)

  torch.testing.assert_close(reward, torch.full((2,), 1.0 / 6.0))
  torch.testing.assert_close(
    vector,
    torch.tensor([[-0.003, 0.0, -0.09], [-0.004, 0.0, -0.09]]),
  )
  torch.testing.assert_close(fraction, torch.full((2, 1), 1.0 / 6.0))
  torch.testing.assert_close(error, torch.tensor([[0.003], [0.004]]))
  torch.testing.assert_close(tracker.next_gate, before_index)
  torch.testing.assert_close(tracker.newly_crossed, before_pulse)


def test_next_control_window_clears_the_crossing_pulse(tracker_env):
  """Leaving a pulse latched past decimation would reward one gate twice."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  _set_head_z(tracker_env, [0.59, 0.59])
  tracker(tracker_env)
  assert tracker.newly_crossed.tolist() == [1, 1]

  tracker(tracker_env)

  assert tracker.newly_crossed.tolist() == [0, 0]
  assert ordered_gate_progress_reward(tracker_env).tolist() == [0.0, 0.0]


@pytest.mark.parametrize(("head_z", "crossed"), ((0.35, 3), (0.05, 6)))
def test_tracker_emits_multi_gate_pulse_and_clears_it_next_window(
  tracker_env,
  head_z,
  crossed,
):
  """The stateful tracker must preserve pure geometry's multi-gate sweep behavior."""
  tracker_env.cfg.decimation = 1
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  _set_head_z(tracker_env, [head_z, head_z])

  tracker(tracker_env)

  assert tracker.newly_crossed.tolist() == [crossed, crossed]
  assert tracker.next_gate.tolist() == [crossed, crossed]
  torch.testing.assert_close(
    ordered_gate_progress_reward(tracker_env),
    torch.full((2,), crossed / GUIDELINE_NUM_GATES),
  )
  tracker(tracker_env)
  assert tracker.newly_crossed.tolist() == [0, 0]
  assert ordered_gate_progress_reward(tracker_env).tolist() == [0.0, 0.0]


def test_completed_gate_observations_are_zero_vector_and_unit_fraction(tracker_env):
  """A completed path needs a finite terminal observation with no nonexistent gate."""
  tracker_env.cfg.decimation = 1
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  _set_head_z(tracker_env, [0.05, 0.05])
  tracker(tracker_env)

  vector = next_gate_vector(tracker_env)
  fraction = completed_gate_fraction(tracker_env)

  assert vector.shape == (2, 3)
  assert torch.isfinite(vector).all()
  torch.testing.assert_close(vector, torch.zeros(2, 3))
  torch.testing.assert_close(fraction, torch.ones(2, 1))


def test_degenerate_tracker_reference_stays_finite_and_never_advances(tracker_env):
  """Coincident reset anchors must remain safe through tracker and reader paths."""
  tracker_env.scene["nail_block"].data.site_pos_w[:, 0, 2] = 0.7
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  tracker_env.scene["robot"].data.site_pos_w[:, 0, :] = torch.tensor(
    [[0.10, 0.0, 0.60], [0.0, 0.20, 0.50]],
  )

  tracker(tracker_env)

  readers = (
    next_gate_vector(tracker_env),
    completed_gate_fraction(tracker_env),
    guideline_perpendicular_error(tracker_env),
    ordered_gate_progress_reward(tracker_env),
  )
  assert all(torch.isfinite(value).all() for value in readers)
  assert tracker.next_gate.tolist() == [0, 0]
  assert tracker.newly_crossed.tolist() == [0, 0]


def test_same_substep_contact_disarms_before_gate_credit(tracker_env):
  """Checking contact after gate advancement would pay for a contacted segment."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  tracker_env._hammer_first_strike.started[:] = True
  _set_head_z(tracker_env, [0.59, 0.59])

  tracker(tracker_env)

  assert tracker.newly_crossed.tolist() == [0, 0]
  assert tracker.next_gate.tolist() == [0, 0]
  assert tracker.disarmed.tolist() == [True, True]


def test_contact_disarm_is_permanent_after_first_strike_latch_falls(tracker_env):
  """A later false contact latch must not rearm one-shot gate progress."""
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  tracker_env._hammer_first_strike.started[:] = True
  _set_head_z(tracker_env, [0.59, 0.59])
  tracker(tracker_env)
  tracker_env._hammer_first_strike.started[:] = False
  _set_head_z(tracker_env, [0.05, 0.05])

  tracker(tracker_env)

  assert tracker.disarmed.tolist() == [True, True]
  assert tracker.next_gate.tolist() == [0, 0]
  assert tracker.newly_crossed.tolist() == [0, 0]
  assert ordered_gate_progress_reward(tracker_env).tolist() == [0.0, 0.0]


def test_six_gate_cumulative_raw_payout_is_exactly_one(tracker_env):
  """Wrong normalization or repeat credit would change the episode reward dose."""
  tracker_env.cfg.decimation = 1
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  tracker(tracker_env)
  total = torch.zeros(tracker_env.num_envs)

  for z in (0.59, 0.49, 0.39, 0.29, 0.19, 0.09):
    _set_head_z(tracker_env, [z, z])
    tracker(tracker_env)
    total += ordered_gate_progress_reward(tracker_env)

  torch.testing.assert_close(total, torch.ones(tracker_env.num_envs))
  assert tracker.next_gate.tolist() == [GUIDELINE_NUM_GATES] * tracker_env.num_envs
  tracker(tracker_env)
  assert ordered_gate_progress_reward(tracker_env).tolist() == [0.0, 0.0]


def test_tracker_and_reader_device_dtype_contract(tracker_env):
  """Tracker state must not allocate off-device or silently narrow geometry."""
  tracker_env.scene["robot"].data.site_pos_w = (
    tracker_env.scene["robot"].data.site_pos_w.double()
  )
  tracker_env.scene["nail_block"].data.site_pos_w = (
    tracker_env.scene["nail_block"].data.site_pos_w.double()
  )
  tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
  metric_value = tracker(tracker_env)

  mutable = (
    tracker.initialized,
    tracker.entry,
    tracker.nail,
    tracker.previous_head,
    tracker.next_gate,
    tracker.newly_crossed,
    tracker.disarmed,
  )
  assert all(value.device == torch.device(tracker_env.device) for value in mutable)
  assert tracker.entry.dtype == torch.float64
  assert tracker.nail.dtype == torch.float64
  assert tracker.previous_head.dtype == torch.float64
  assert metric_value.dtype == torch.float64
  assert next_gate_vector(tracker_env).dtype == torch.float64
  assert guideline_perpendicular_error(tracker_env).dtype == torch.float64
  assert completed_gate_fraction(tracker_env).dtype == torch.float32
  assert ordered_gate_progress_reward(tracker_env).dtype == torch.float32


@pytest.mark.parametrize(
  "reader",
  (
    next_gate_vector,
    completed_gate_fraction,
    guideline_perpendicular_error,
    ordered_gate_progress_reward,
  ),
)
def test_readers_fail_loudly_when_tracker_is_missing(tracker_env, reader):
  """Missing always-on tracker wiring must not silently return dummy observations."""
  del tracker_env._hammer_first_strike

  with pytest.raises(RuntimeError, match="WaypointProgressTracker"):
    reader(tracker_env)
