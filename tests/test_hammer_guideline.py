"""Tests for pure Cartesian guideline geometry."""

import torch

from src.tasks.hammer.mdp.guideline import advance_ordered_gates, project_to_reference


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
