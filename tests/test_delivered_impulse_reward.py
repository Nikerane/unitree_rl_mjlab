"""Unit tests for DeliveredImpulseTerm — the object-side maximize-impulse reward.

Pays the positive increment of the contact-window delivered axial impulse (read from
SubstepDeliveredImpulse), DEPTH-GATED (only credited on steps the nail actually advances, so force
without progress is unfarmable) and normalized by I_ref. Mirrors the validated NailDepthDeltaTerm
delta-tracking pattern. Stateful: _credited (max delivered credited so far), _prev_depth.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers import stub
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR
from src.tasks.hammer.mdp.rewards import DeliveredImpulseTerm

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
  t = _rterm(1)
  out = t(_renv(torch.tensor([0.10]), torch.tensor([0.0])), i_ref=1.0, eps=5e-4, nail_cfg=NAIL_CFG)
  assert torch.allclose(out, torch.tensor([0.0])), out  # impulse delivered but nail didn't advance
  out2 = t(_renv(torch.tensor([0.10]), torch.tensor([0.010])), i_ref=1.0, eps=5e-4, nail_cfg=NAIL_CFG)
  assert torch.allclose(out2, torch.tensor([0.10]), atol=1e-6), out2  # credited on the advancing step


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
