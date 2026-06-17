"""C0 unit tests for the pure CaT δ-math core (src/tasks/hammer/cat/constraint_manager.py).

These exercise the δ formula, max-combine, EMA persistence across reset, and the no-violation case
with no sim -- the CaT helper is pure torch. The constraint func is checked with a faked env in the
style of tests/test_velocity_bound.py.
"""

from types import SimpleNamespace

import torch

from src.tasks.hammer.cat import CaT, joint_velocity_excess
from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT


def test_no_violation_zero_delta():
  # All margins <= 0 -> no termination probability.
  cat = CaT(tau=0.95, min_p=0.0)
  cat.add("v", torch.tensor([-1.0, -0.5, 0.0]), max_p=1.0)
  assert torch.allclose(cat.get_probs(), torch.zeros(3))


def test_saturated_violation_hits_max_p():
  # c == batch-max -> normalized clamps to 1 -> delta = max_p (for the batch leader).
  cat = CaT(tau=0.95, min_p=0.0)
  cat.add("v", torch.full((4,), 5.0), max_p=0.5)  # first add seeds c_max=5; all at the max -> norm=1
  assert torch.allclose(cat.get_probs(), torch.full((4,), 0.5), atol=1e-6)


def test_min_p_floor_and_formula_exact():
  # tau=0 -> c_max is exactly the current batch max; check the closed-form delta with a min_p floor.
  cat = CaT(tau=0.0, min_p=0.1)
  cat.add("v", torch.tensor([10.0, 5.0]), max_p=1.0)  # c_max=10; norm=[1.0,0.5]
  # delta = 0.1 + clamp(norm,0,1)*(1.0-0.1) = [1.0, 0.55]
  assert torch.allclose(cat.get_probs(), torch.tensor([1.0, 0.55]), atol=1e-6)


def test_max_combine_over_terms_picks_larger_per_env():
  # Pre-warm both terms' c_max to 10, reset (keeps c_max), then add asymmetric violations so each
  # term dominates a different env -> get_probs must take the per-env max across terms.
  cat = CaT(tau=0.9, min_p=0.0)
  cat.add("a", torch.tensor([10.0, 10.0]), max_p=1.0)  # seed c_max_a=10
  cat.add("b", torch.tensor([10.0, 10.0]), max_p=1.0)  # seed c_max_b=10
  cat.reset()
  cat.add("a", torch.tensor([3.0, 1.0]), max_p=1.0)  # c_max_a=0.9*10+0.1*3=9.3 -> [0.32258, 0.10753]
  cat.add("b", torch.tensor([1.0, 4.0]), max_p=1.0)  # c_max_b=0.9*10+0.1*4=9.4 -> [0.10638, 0.42553]
  out = cat.get_probs()
  assert torch.allclose(out, torch.tensor([3.0 / 9.3, 4.0 / 9.4]), atol=1e-4), out


def test_ema_persists_across_reset_but_probs_cleared():
  cat = CaT(tau=0.9, min_p=0.0)
  cat.add("v", torch.full((3,), 10.0), max_p=1.0)  # seed c_max=10
  cmax_before = cat.running_maxes["v"].clone()
  cat.reset()
  assert "v" not in cat.probs  # per-step buffer cleared
  assert "v" in cat.running_maxes and torch.allclose(cat.running_maxes["v"], cmax_before)  # normalizer kept
  cat.add("v", torch.full((3,), 5.0), max_p=1.0)  # EMA: c_max=0.9*10+0.1*5=9.5
  assert torch.allclose(cat.get_probs(), torch.full((3,), 5.0 / 9.5), atol=1e-4)


def test_more_excess_more_delta():
  cat = CaT(tau=0.95, min_p=0.0)
  for _ in range(3):
    cat.add("v", torch.full((100,), 10.0), max_p=1.0)  # warm c_max to ~10
  cat.reset()
  cat.add("v", torch.tensor([1.0, 3.0]), max_p=1.0)  # same c_max for both envs
  out = cat.get_probs()
  assert out[1] > out[0] > 0.0, out


def test_empty_get_probs():
  cat = CaT()
  assert cat.get_probs().numel() == 0


def _env(qv: torch.Tensor):
  entity = SimpleNamespace(data=SimpleNamespace(joint_vel=qv))
  return SimpleNamespace(scene={"robot": entity})


def test_constraint_func_returns_raw_signed_margin():
  qv = torch.tensor([[4.0, 1.0, 1.0, 1.0, 1.0, 1.0]])  # joint0 over the limit, rest under
  rcfg = SimpleNamespace(name="robot", joint_ids=list(range(6)))
  c = joint_velocity_excess(_env(qv), limit=Z1_JOINT_VEL_LIMIT, robot_cfg=rcfg)
  assert torch.allclose(c, qv - Z1_JOINT_VEL_LIMIT)
  assert bool(c[0, 0] > 0) and bool((c[0, 1:] < 0).all())  # margin is signed, not clamped
