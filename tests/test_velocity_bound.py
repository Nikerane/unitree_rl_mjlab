"""Unit tests for the joint-velocity-bound ablation terms (A2 penalty, A3 CaT).

The terms read ``env.scene[robot_cfg.name].data.joint_vel[:, robot_cfg.joint_ids]``.
In the real env the manager resolves ``robot_cfg.joint_ids``; here we fake the env and
pass a SimpleNamespace robot_cfg so the math can be checked in isolation.
"""

from types import SimpleNamespace

import torch

from src.tasks.hammer.mdp.velocity_bound import (
  CaTJointVelConstraint,
  Z1_JOINT_VEL_LIMIT,
  joint_vel_excess_penalty,
  joint_vel_hard_termination,
)

LIM = Z1_JOINT_VEL_LIMIT  # 3.1415


def _env(qv: torch.Tensor):
  entity = SimpleNamespace(data=SimpleNamespace(joint_vel=qv))
  scene = {"robot": entity}
  return SimpleNamespace(scene=scene, device="cpu", num_envs=qv.shape[0])


def _rcfg(n_joints: int = 6):
  return SimpleNamespace(name="robot", joint_ids=list(range(n_joints)))


def test_penalty_silent_below_soft_limit():
  # All joints at 2.0 rad/s < 0.9*3.1415 = 2.827 -> zero penalty.
  qv = torch.full((4, 6), 2.0)
  pen = joint_vel_excess_penalty(_env(qv), limit=LIM, beta=0.9, robot_cfg=_rcfg())
  assert torch.allclose(pen, torch.zeros(4)), pen


def test_penalty_bites_above_soft_limit_squared():
  # One joint at 4.65 rad/s (the observed worst case); others under.
  qv = torch.full((1, 6), 1.0)
  qv[0, 2] = 4.65
  pen = joint_vel_excess_penalty(_env(qv), limit=LIM, beta=0.9, robot_cfg=_rcfg())
  expected = (4.65 - 0.9 * LIM) ** 2  # only the over-limit joint contributes
  assert torch.allclose(pen, torch.tensor([expected]), atol=1e-4), (pen, expected)


def test_penalty_is_monotone_and_squared():
  # Bigger overshoot -> disproportionately bigger penalty (squared crushes the tail).
  def pen_at(v):
    qv = torch.full((1, 6), 1.0)
    qv[0, 0] = v
    return float(joint_vel_excess_penalty(_env(qv), robot_cfg=_rcfg())[0])
  p35, p465 = pen_at(3.5), pen_at(4.65)
  assert p465 > p35 > 0.0
  # squared: doubling the excess ~quadruples the penalty
  e1, e2 = 3.5 - 0.9 * LIM, 4.65 - 0.9 * LIM
  assert abs((p465 / p35) - (e2 / e1) ** 2) < 1e-3


def test_cat_never_terminates_below_limit():
  torch.manual_seed(0)
  term = CaTJointVelConstraint(cfg=SimpleNamespace(params={}), env=_env(torch.zeros(8, 6)))
  qv = torch.full((8, 6), 2.5)  # under 3.1415
  for _ in range(5):
    out = term(_env(qv), limit=LIM, p_max=0.5)
    assert out.dtype == torch.bool and not out.any(), out


def test_cat_terminates_at_pmax_fraction_when_saturated():
  # All envs grossly over the limit -> c/EMA clips to 1 -> delta = p_max on every env.
  torch.manual_seed(0)
  n, p_max = 20000, 0.5
  qv = torch.full((n, 6), 6.0)  # excess ~2.86, >> EMA seed (1.0) -> clip to 1
  term = CaTJointVelConstraint(cfg=SimpleNamespace(params={}), env=_env(qv))
  out = term(_env(qv), limit=LIM, p_max=p_max)
  frac = out.float().mean().item()
  assert abs(frac - p_max) < 0.02, frac  # ~50% terminate


def test_cat_more_excess_more_termination():
  torch.manual_seed(0)
  n = 20000
  term = CaTJointVelConstraint(cfg=SimpleNamespace(params={}), env=_env(torch.zeros(n, 6)))
  # Warm the EMA with a big batch so normalization is stable.
  term(_env(torch.full((n, 6), 6.0)), limit=LIM, p_max=1.0)
  small = term(_env(torch.full((n, 6), LIM + 0.2)), limit=LIM, p_max=1.0).float().mean().item()
  big = term(_env(torch.full((n, 6), LIM + 1.4)), limit=LIM, p_max=1.0).float().mean().item()
  assert big > small >= 0.0, (small, big)


def _env_step(qv: torch.Tensor, step: int):
  e = _env(qv)
  e.common_step_counter = step
  return e


def test_hard_term_silent_during_warmup():
  # Even grossly over the limit, no termination before warmup_steps.
  qv = torch.full((4, 6), 5.0)
  out = joint_vel_hard_termination(_env_step(qv, 100), warmup_steps=3600, robot_cfg=_rcfg(), detection="control_rate")
  assert out.dtype == torch.bool and not out.any(), out


def test_hard_term_fires_past_warmup_only_over_limit():
  # env0 over (5 rad/s), env1 under (2 rad/s); past warmup.
  qv = torch.tensor([[5.0, 1, 1, 1, 1, 1], [2.0, 2, 2, 2, 2, 2]])
  out = joint_vel_hard_termination(_env_step(qv, 4000), warmup_steps=3600, robot_cfg=_rcfg(), detection="control_rate")
  assert bool(out[0]) and not bool(out[1]), out
