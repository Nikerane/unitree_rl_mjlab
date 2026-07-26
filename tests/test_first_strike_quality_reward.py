"""Unit tests for the pure batched first-contact quality kernel."""

import pytest
import torch

from src.tasks.hammer.mdp.contact_quality import contact_point_quality
from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
from src.tasks.hammer.mdp.rewards import FirstStrikeQualityImpactRewardTerm


NAIL_RADIUS_M = 0.012


def _single_contact(
  position_w: torch.Tensor,
  *,
  force_normal: float = 1.0,
  found: float = 1.0,
  nail_top_w: torch.Tensor | None = None,
  nail_axis_w: torch.Tensor | None = None,
):
  return contact_point_quality(
    found=torch.tensor([[found]]),
    force_contact=torch.tensor([[[force_normal, 7.0, -4.0]]]),
    position_w=position_w.reshape(1, 1, 3),
    nail_top_w=(
      torch.zeros(1, 3) if nail_top_w is None else nail_top_w.reshape(1, 3)
    ),
    nail_axis_w=(
      torch.tensor([0.0, 0.0, -1.0])
      if nail_axis_w is None
      else nail_axis_w
    ),
    nail_radius_m=NAIL_RADIUS_M,
    num_slots=1,
  )


def test_center_edge_and_outside_contacts_have_expected_shapes_and_quality():
  centroid, error, quality, valid, overflow = contact_point_quality(
    found=torch.ones(3, 1),
    force_contact=torch.tensor(
      [[[2.0, 9.0, -3.0]], [[2.0, 9.0, -3.0]], [[2.0, 9.0, -3.0]]]
    ),
    position_w=torch.tensor(
      [[[0.0, 0.0, 0.1]], [[0.012, 0.0, 0.1]], [[0.018, 0.0, 0.1]]]
    ),
    nail_top_w=torch.tensor(
      [[0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]
    ),
    nail_axis_w=torch.tensor([0.0, 0.0, -1.0]),
    nail_radius_m=NAIL_RADIUS_M,
    num_slots=1,
  )

  assert centroid.shape == (3, 3)
  assert error.shape == quality.shape == valid.shape == overflow.shape == (3,)
  torch.testing.assert_close(error, torch.tensor([0.0, 0.012, 0.018]))
  torch.testing.assert_close(quality, torch.tensor([1.0, 0.0, 0.0]))
  assert valid.tolist() == [True, True, True]
  assert overflow.tolist() == [False, False, False]


def test_two_contacts_use_positive_normal_force_weighted_centroid():
  centroid, error, quality, valid, overflow = contact_point_quality(
    found=torch.tensor([[1.0, 1.0]]),
    force_contact=torch.tensor(
      [[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    ),
    position_w=torch.tensor(
      [[[0.0, 0.0, 0.1], [0.008, 0.0, 0.1]]]
    ),
    nail_top_w=torch.tensor([[0.0, 0.0, 0.1]]),
    nail_axis_w=torch.tensor([0.0, 0.0, -1.0]),
    nail_radius_m=NAIL_RADIUS_M,
    num_slots=2,
  )

  assert centroid[0, 0].item() == pytest.approx(0.002)
  assert error[0].item() == pytest.approx(0.002)
  assert quality[0].item() == pytest.approx(
    1.0 - (0.002 / NAIL_RADIUS_M) ** 2
  )
  assert valid.tolist() == [True]
  assert overflow.tolist() == [False]


def test_world_translation_does_not_change_radial_error_or_quality():
  base = _single_contact(torch.tensor([0.006, 0.0, 0.0]))
  translated = _single_contact(
    torch.tensor([4.006, -2.0, 7.0]),
    nail_top_w=torch.tensor([4.0, -2.0, 7.0]),
  )

  torch.testing.assert_close(translated[1], base[1])
  torch.testing.assert_close(translated[2], base[2])
  torch.testing.assert_close(
    translated[0], torch.tensor([[4.006, -2.0, 7.0]])
  )


def test_nonunit_rotated_nail_axis_removes_axial_offset():
  centroid, error, quality, valid, _ = _single_contact(
    torch.tensor([0.1, 0.006, 0.0]),
    nail_axis_w=torch.tensor([2.0, 0.0, 0.0]),
  )

  torch.testing.assert_close(centroid, torch.tensor([[0.1, 0.006, 0.0]]))
  assert error.item() == pytest.approx(0.006)
  assert quality.item() == pytest.approx(0.75)
  assert valid.tolist() == [True]


def test_no_contact_returns_zero_outputs_and_is_invalid():
  centroid, error, quality, valid, overflow = _single_contact(
    torch.tensor([0.006, 0.0, 0.0]),
    force_normal=10.0,
    found=0.0,
  )

  torch.testing.assert_close(centroid, torch.zeros(1, 3))
  torch.testing.assert_close(error, torch.zeros(1))
  torch.testing.assert_close(quality, torch.zeros(1))
  assert valid.tolist() == [False]
  assert overflow.tolist() == [False]


@pytest.mark.parametrize("force_normal", [0.0, -3.0])
def test_nonpositive_normal_force_returns_zero_outputs_and_is_invalid(
  force_normal,
):
  centroid, error, quality, valid, overflow = _single_contact(
    torch.tensor([0.006, 0.0, 0.0]),
    force_normal=force_normal,
  )

  torch.testing.assert_close(centroid, torch.zeros(1, 3))
  torch.testing.assert_close(error, torch.zeros(1))
  torch.testing.assert_close(quality, torch.zeros(1))
  assert valid.tolist() == [False]
  assert overflow.tolist() == [False]


def test_nonfinite_input_returns_zero_outputs_and_is_invalid():
  centroid, error, quality, valid, overflow = _single_contact(
    torch.tensor([float("nan"), 0.0, 0.0])
  )

  torch.testing.assert_close(centroid, torch.zeros(1, 3))
  torch.testing.assert_close(error, torch.zeros(1))
  torch.testing.assert_close(quality, torch.zeros(1))
  assert valid.tolist() == [False]
  assert overflow.tolist() == [False]


def test_contact_slot_overflow_returns_zero_outputs_and_is_invalid():
  centroid, error, quality, valid, overflow = contact_point_quality(
    found=torch.tensor([[3.0, 3.0]]),
    force_contact=torch.tensor(
      [[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    ),
    position_w=torch.tensor(
      [[[0.0, 0.0, 0.1], [0.008, 0.0, 0.1]]]
    ),
    nail_top_w=torch.tensor([[0.0, 0.0, 0.1]]),
    nail_axis_w=torch.tensor([0.0, 0.0, -1.0]),
    nail_radius_m=NAIL_RADIUS_M,
    num_slots=2,
  )

  torch.testing.assert_close(centroid, torch.zeros(1, 3))
  torch.testing.assert_close(error, torch.zeros(1))
  torch.testing.assert_close(quality, torch.zeros(1))
  assert valid.tolist() == [False]
  assert overflow.tolist() == [True]


def _quality_reward_env(
  *,
  finalized: bool = True,
  productive: bool = True,
  quality: float = 0.25,
  quality_valid: bool = True,
  v_precontact: float = 3.0,
  delivered: float = 10.0,
):
  env = type("QualityRewardEnv", (), {"num_envs": 1, "device": "cpu"})()
  tracker = type(
    "QualityRewardTracker",
    (),
    {
      "finalized": torch.tensor([finalized]),
      "productive": torch.tensor([productive]),
      "contact_quality": torch.tensor([quality]),
      "contact_quality_valid": torch.tensor([quality_valid]),
      "v_precontact": torch.tensor([v_precontact]),
      "delivered": torch.tensor([delivered]),
    },
  )()
  setattr(env, _ENV_FIRST_STRIKE_ATTR, tracker)
  return env, tracker


def test_quality_speed_reader_pays_bounded_contact_quality_once():
  """Removing quality multiplication, speed saturation, or one-shot credit must fail."""
  env, _ = _quality_reward_env(
    finalized=True,
    productive=True,
    quality=0.25,
    quality_valid=True,
    v_precontact=3.0,
  )
  term = FirstStrikeQualityImpactRewardTerm(cfg=None, env=env)

  assert term(env, v_expected=1.4598331451416016).item() == pytest.approx(0.25)
  assert term(env, v_expected=1.4598331451416016).item() == pytest.approx(0.0)


@pytest.mark.parametrize(
  "tracker_state",
  [
    {"quality_valid": False},
    {"finalized": False},
    {"productive": False},
  ],
  ids=["invalid-quality", "no-contact", "nonproductive-event"],
)
def test_quality_speed_reader_rejects_ineligible_event(tracker_state):
  """A missing quality/contact/finalized productive event must pay no maximize reward."""
  env, _ = _quality_reward_env(**tracker_state)
  term = FirstStrikeQualityImpactRewardTerm(cfg=None, env=env)

  assert term(env, v_expected=1.4598331451416016).item() == pytest.approx(0.0)


def test_quality_speed_reader_cannot_trade_extreme_speed_for_quality():
  """Increasing speed beyond the knee must never raise payout above contact quality."""
  env, _ = _quality_reward_env(quality=0.25, v_precontact=300.0)
  term = FirstStrikeQualityImpactRewardTerm(cfg=None, env=env)

  assert term(env, v_expected=1.4598331451416016).item() == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_quality_speed_reader_rejects_nonpositive_or_nonfinite_normalizer(bad):
  """A bad normalizer must fail loudly rather than bypass boundedness."""
  env, _ = _quality_reward_env()
  term = FirstStrikeQualityImpactRewardTerm(cfg=None, env=env)

  with pytest.raises(ValueError):
    term(env, v_expected=bad)
