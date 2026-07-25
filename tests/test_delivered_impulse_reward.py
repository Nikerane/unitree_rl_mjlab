"""Unit tests for DeliveredImpulseTerm — the object-side maximize-impulse reward.

Pays the positive increment of the contact-window delivered axial impulse (read from
SubstepDeliveredImpulse), DEPTH-GATED (only credited on steps the nail actually advances, so force
without progress is unfarmable) and normalized by I_ref. Mirrors the validated NailDepthDeltaTerm
delta-tracking pattern. Stateful: _credited (max delivered credited so far), _prev_depth.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.helpers import stub
from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR
from src.tasks.hammer.mdp.rewards import (
  DeliveredImpulseTerm,
  FirstStrikeDeliveredRewardTerm,
  FirstStrikeLegacyDeliveredRewardTerm,
)

NAIL_CFG = SimpleNamespace(name="nail_block", joint_ids=[0])


def _renv(delivered: torch.Tensor, depth: torch.Tensor):
  B = delivered.shape[0]
  nail = SimpleNamespace(data=SimpleNamespace(joint_pos=depth.reshape(B, 1)))
  env = SimpleNamespace(scene={"nail_block": nail}, num_envs=B, device="cpu")
  setattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, SimpleNamespace(delivered=delivered))
  return env


def _rterm(B: int) -> DeliveredImpulseTerm:
  # helpers.stub fails loudly if DeliveredImpulseTerm.__init__ gains a field.
  return stub(DeliveredImpulseTerm, _credited=torch.zeros(B), _prev_depth=torch.zeros(B))


def test_pays_delivered_delta_when_nail_advances():
  t = _rterm(1)
  out = t(_renv(torch.tensor([0.10]), torch.tensor([0.010])), i_ref=1.0, eps=5e-4, nail_cfg=NAIL_CFG)
  assert torch.allclose(out, torch.tensor([0.10]), atol=1e-6), out


def test_normalized_by_i_ref():
  t = _rterm(1)
  out = t(_renv(torch.tensor([0.10]), torch.tensor([0.010])), i_ref=0.05, eps=5e-4, nail_cfg=NAIL_CFG)
  assert torch.allclose(out, torch.tensor([2.0]), atol=1e-6), out


def test_depth_gate_blocks_payout_without_progress():
  # A press with no depth advance pays nothing.
  t = _rterm(1)
  out = t(_renv(torch.tensor([0.10]), torch.tensor([0.0])), i_ref=1.0, eps=5e-4, nail_cfg=NAIL_CFG)
  assert torch.allclose(out, torch.tensor([0.0])), out  # impulse delivered but nail didn't advance


def test_no_press_backlog_farm():
  # F1 fix (2026-07-19 audit): non-progress impulse is DISCARDED, not escrowed. A press with no depth
  # advance must NOT be collectable on a later nudge — else park+press+nudge farms the delivered reward.
  # (The OLD escrow code held _credited on non-advancing steps and paid the whole backlog here.)
  t = _rterm(1)
  # press: 0.10 accrues while the nail is static -> pays 0 AND is discarded (baseline advances to 0.10)
  out = t(_renv(torch.tensor([0.10]), torch.tensor([0.0])), i_ref=1.0, eps=5e-4, nail_cfg=NAIL_CFG)
  assert torch.allclose(out, torch.tensor([0.0])), out
  # nudge with NO new impulse: the 0.10 press backlog must NOT be paid (old code paid 0.10 here)
  out2 = t(_renv(torch.tensor([0.10]), torch.tensor([0.010])), i_ref=1.0, eps=5e-4, nail_cfg=NAIL_CFG)
  assert torch.allclose(out2, torch.tensor([0.0]), atol=1e-6), out2
  # advance WITH fresh impulse: pays only the new increment (0.14 - 0.10), not the discarded backlog
  out3 = t(_renv(torch.tensor([0.14]), torch.tensor([0.020])), i_ref=1.0, eps=5e-4, nail_cfg=NAIL_CFG)
  assert torch.allclose(out3, torch.tensor([0.04]), atol=1e-6), out3


def test_one_payout_no_double_credit():
  t = _rterm(1)
  t(_renv(torch.tensor([0.10]), torch.tensor([0.010])), i_ref=1.0, nail_cfg=NAIL_CFG)
  out = t(_renv(torch.tensor([0.10]), torch.tensor([0.020])), i_ref=1.0, nail_cfg=NAIL_CFG)
  assert torch.allclose(out, torch.tensor([0.0]), atol=1e-6), out  # no NEW impulse -> no pay


def test_credit_is_monotone_no_repay_after_signal_drop():
  # Review finding #4: if the accumulator signal ever DROPS (it should be monotone now, but the
  # reward must be robust regardless), credit must NOT be lowered — otherwise a later stronger
  # window re-pays impulse that was already paid (reward farming).
  t = _rterm(1)
  t(_renv(torch.tensor([0.20]), torch.tensor([0.010])), i_ref=1.0, nail_cfg=NAIL_CFG)  # pay 0.20
  out_drop = t(_renv(torch.tensor([0.05]), torch.tensor([0.020])), i_ref=1.0, nail_cfg=NAIL_CFG)
  assert torch.allclose(out_drop, torch.tensor([0.0]), atol=1e-6), out_drop  # no negative/zero-pay
  assert torch.allclose(t._credited, torch.tensor([0.20]), atol=1e-6), t._credited  # NOT lowered
  out_again = t(_renv(torch.tensor([0.18]), torch.tensor([0.030])), i_ref=1.0, nail_cfg=NAIL_CFG)
  assert torch.allclose(out_again, torch.tensor([0.0]), atol=1e-6), out_again  # 0.18 < 0.20: no re-pay


def test_incremental_delivered_pays_only_the_increment():
  t = _rterm(1)
  o1 = t(_renv(torch.tensor([0.04]), torch.tensor([0.010])), i_ref=1.0, nail_cfg=NAIL_CFG)
  o2 = t(_renv(torch.tensor([0.10]), torch.tensor([0.020])), i_ref=1.0, nail_cfg=NAIL_CFG)
  assert torch.allclose(o1, torch.tensor([0.04]), atol=1e-6), o1
  assert torch.allclose(o2, torch.tensor([0.06]), atol=1e-6), o2  # only the +0.06 increment


def test_reset_zeros_state():
  t = _rterm(2)
  t(_renv(torch.tensor([0.1, 0.1]), torch.tensor([0.01, 0.01])), i_ref=1.0, nail_cfg=NAIL_CFG)
  t.reset(None)
  assert torch.allclose(t._credited, torch.zeros(2))
  assert torch.allclose(t._prev_depth, torch.zeros(2))


def test_bad_i_ref_raises():
  # F6 (2026-07-19 audit): a CLI sweep passing a zero/NaN normalizer must fail loud, not emit inf/NaN.
  t = _rterm(1)
  for bad in (0.0, -1.0, float("inf"), float("nan")):
    with pytest.raises(ValueError, match="i_ref"):
      t(_renv(torch.tensor([0.10]), torch.tensor([0.010])), i_ref=bad, nail_cfg=NAIL_CFG)


def test_raises_without_accumulator():
  t = _rterm(1)
  env = SimpleNamespace(
    scene={"nail_block": SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 1)))},
    num_envs=1, device="cpu",
  )
  with pytest.raises(RuntimeError, match="SubstepDeliveredImpulse"):
    t(env, i_ref=1.0, nail_cfg=NAIL_CFG)


def test_shape_B():
  t = _rterm(3)
  out = t(_renv(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.01, 0.01, 0.01])), i_ref=1.0, nail_cfg=NAIL_CFG)
  assert out.shape == (3,)


def _event_env(delivered: torch.Tensor, *, productive=True, attach_tracker=True):
  B = delivered.shape[0]
  env = SimpleNamespace(num_envs=B, device="cpu")
  tracker = SimpleNamespace(
    finalized=torch.ones(B, dtype=torch.bool),
    productive=torch.full((B,), bool(productive), dtype=torch.bool),
    v_precontact=torch.zeros(B),
    delivered=delivered,
  )
  if attach_tracker:
    setattr(env, _ENV_FIRST_STRIKE_ATTR, tracker)
  return env


def test_event_delivered_saturates_at_one_i_ref():
  """First-window impulse is reference-sufficient, not unbounded maximization."""
  env = _event_env(torch.tensor([0.30, 0.75]))
  term = FirstStrikeDeliveredRewardTerm(cfg=None, env=env)

  assert torch.allclose(term(env, i_ref=0.50), torch.tensor([0.60, 1.00]))


def test_event_delivered_linear_mode_preserves_excess_over_i_ref():
  """Arm F must remain linear above the event reference instead of silently capping."""
  env = _event_env(torch.tensor([0.30, 0.75]))
  term = FirstStrikeDeliveredRewardTerm(cfg=None, env=env)

  assert torch.allclose(
    term(env, i_ref=0.50, saturate=False),
    torch.tensor([0.60, 1.50]),
  )


@pytest.mark.parametrize(
  "bad",
  [None, "false", 0, 1, np.bool_(True), torch.tensor(False)],
)
def test_event_delivered_requires_real_bool_saturate(bad):
  """Ambiguous config values must fail instead of silently selecting a payout branch."""
  env = _event_env(torch.tensor([0.75]))
  term = FirstStrikeDeliveredRewardTerm(cfg=None, env=env)

  with pytest.raises(TypeError, match="saturate.*bool"):
    term(env, i_ref=0.50, saturate=bad)


def test_event_delivered_pays_once_only():
  env = _event_env(torch.tensor([0.25]))
  term = FirstStrikeDeliveredRewardTerm(cfg=None, env=env)

  assert torch.equal(term(env, i_ref=0.50), torch.tensor([0.50]))
  assert torch.equal(term(env, i_ref=0.50), torch.tensor([0.0]))


def test_event_delivered_requires_shared_tracker():
  env = _event_env(torch.tensor([0.25]), attach_tracker=False)
  term = FirstStrikeDeliveredRewardTerm(cfg=None, env=env)

  with pytest.raises(RuntimeError, match="FirstStrikeEventTracker"):
    term(env, i_ref=0.50)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_event_delivered_rejects_bad_normalizer(bad):
  env = _event_env(torch.tensor([0.25]))
  term = FirstStrikeDeliveredRewardTerm(cfg=None, env=env)

  with pytest.raises(ValueError, match="i_ref"):
    term(env, i_ref=bad)


def test_legacy_delivered_sums_positive_increments_through_finalization(monkeypatch):
  env = SimpleNamespace(num_envs=1, device="cpu")
  tracker = SimpleNamespace(
    started=torch.tensor([False]),
    finalized=torch.tensor([False]),
    productive=torch.tensor([False]),
  )
  setattr(env, _ENV_FIRST_STRIKE_ATTR, tracker)
  sequence = iter(torch.tensor([value]) for value in (0.0, 0.20, 0.30, 0.10))
  monkeypatch.setattr(
    DeliveredImpulseTerm,
    "__call__",
    lambda self, env, **params: next(sequence),
  )
  term = FirstStrikeLegacyDeliveredRewardTerm(cfg=None, env=env)
  outputs = []
  for started, finalized, productive in (
    (False, False, False),
    (True, False, False),
    (True, True, True),
    (True, True, True),
  ):
    tracker.started.fill_(started)
    tracker.finalized.fill_(finalized)
    tracker.productive.fill_(productive)
    outputs.append(float(term(env)))
  assert outputs == pytest.approx([0.0, 0.0, 0.50, 0.0])


def _legacy_delivered_tracker_env(num_envs=1):
  env = SimpleNamespace(num_envs=num_envs, device="cpu")
  tracker = SimpleNamespace(
    started=torch.zeros(num_envs, dtype=torch.bool),
    finalized=torch.zeros(num_envs, dtype=torch.bool),
    productive=torch.zeros(num_envs, dtype=torch.bool),
  )
  setattr(env, _ENV_FIRST_STRIKE_ATTR, tracker)
  return env, tracker


def _patch_delivered_raw(monkeypatch, values):
  sequence = iter(torch.as_tensor(value, dtype=torch.float32) for value in values)
  monkeypatch.setattr(
    DeliveredImpulseTerm,
    "__call__",
    lambda self, env, **params: next(sequence),
  )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_legacy_delivered_ignores_nonfinite_then_sums_finite_positives(
  monkeypatch, nonfinite
):
  env, tracker = _legacy_delivered_tracker_env()
  _patch_delivered_raw(monkeypatch, [[nonfinite], [0.20], [0.30]])
  term = FirstStrikeLegacyDeliveredRewardTerm(cfg=None, env=env)
  tracker.started.fill_(True)

  assert float(term(env)) == 0.0
  assert float(term(env)) == 0.0
  tracker.finalized.fill_(True)
  tracker.productive.fill_(True)
  assert float(term(env)) == pytest.approx(0.50)


def test_legacy_delivered_consumes_unproductive_finalization(monkeypatch):
  env, tracker = _legacy_delivered_tracker_env()
  _patch_delivered_raw(monkeypatch, [[1.0], [5.0]])
  term = FirstStrikeLegacyDeliveredRewardTerm(cfg=None, env=env)
  tracker.started.fill_(True)
  tracker.finalized.fill_(True)

  assert float(term(env)) == 0.0
  tracker.productive.fill_(True)
  assert float(term(env)) == 0.0


def test_legacy_delivered_ignores_delayed_recontact(monkeypatch):
  env, tracker = _legacy_delivered_tracker_env()
  _patch_delivered_raw(monkeypatch, [[1.0], [7.0]])
  term = FirstStrikeLegacyDeliveredRewardTerm(cfg=None, env=env)
  tracker.started.fill_(True)
  tracker.finalized.fill_(True)
  tracker.productive.fill_(True)

  assert float(term(env)) == pytest.approx(1.0)
  assert float(term(env)) == 0.0


def test_legacy_delivered_reset_selected_environments_only(monkeypatch):
  env, tracker = _legacy_delivered_tracker_env(num_envs=2)
  _patch_delivered_raw(monkeypatch, [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
  term = FirstStrikeLegacyDeliveredRewardTerm(cfg=None, env=env)
  tracker.started.fill_(True)
  tracker.finalized.fill_(True)
  tracker.productive.fill_(True)
  assert torch.equal(term(env), torch.tensor([1.0, 2.0]))

  term.reset(torch.tensor([0]))
  tracker.finalized[:] = torch.tensor([False, True])
  assert torch.equal(term(env), torch.zeros(2))
  tracker.finalized[:] = True
  assert torch.equal(term(env), torch.tensor([3.0, 0.0]))


def test_legacy_delivered_real_parent_outputs_match_standalone_term():
  standalone_env = _renv(torch.zeros(1), torch.zeros(1))
  wrapped_env = _renv(torch.zeros(1), torch.zeros(1))
  tracker = SimpleNamespace(
    started=torch.tensor([False]),
    finalized=torch.tensor([False]),
    productive=torch.tensor([False]),
  )
  setattr(wrapped_env, _ENV_FIRST_STRIKE_ATTR, tracker)
  standalone = DeliveredImpulseTerm(cfg=None, env=standalone_env)
  wrapped = FirstStrikeLegacyDeliveredRewardTerm(cfg=None, env=wrapped_env)

  assert torch.equal(
    standalone(standalone_env, i_ref=1.0, nail_cfg=NAIL_CFG),
    wrapped(wrapped_env, i_ref=1.0, nail_cfg=NAIL_CFG),
  )

  tracker.started.fill_(True)
  getattr(standalone_env, _ENV_SUBSTEP_DELIVERED_ATTR).delivered.fill_(0.20)
  getattr(wrapped_env, _ENV_SUBSTEP_DELIVERED_ATTR).delivered.fill_(0.20)
  standalone_env.scene["nail_block"].data.joint_pos.fill_(0.01)
  wrapped_env.scene["nail_block"].data.joint_pos.fill_(0.01)
  legacy_first = standalone(standalone_env, i_ref=1.0, nail_cfg=NAIL_CFG)
  assert torch.equal(
    wrapped(wrapped_env, i_ref=1.0, nail_cfg=NAIL_CFG), torch.zeros(1)
  )

  getattr(standalone_env, _ENV_SUBSTEP_DELIVERED_ATTR).delivered.fill_(0.50)
  getattr(wrapped_env, _ENV_SUBSTEP_DELIVERED_ATTR).delivered.fill_(0.50)
  standalone_env.scene["nail_block"].data.joint_pos.fill_(0.02)
  wrapped_env.scene["nail_block"].data.joint_pos.fill_(0.02)
  legacy_final = standalone(standalone_env, i_ref=1.0, nail_cfg=NAIL_CFG)
  tracker.finalized.fill_(True)
  tracker.productive.fill_(True)
  assert torch.equal(
    wrapped(wrapped_env, i_ref=1.0, nail_cfg=NAIL_CFG),
    legacy_first + legacy_final,
  )
