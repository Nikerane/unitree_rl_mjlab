"""Unit tests for the joint-velocity-bound ablation terms (A2 penalty, A3 CaT).

The terms read ``env.scene[robot_cfg.name].data.joint_vel[:, robot_cfg.joint_ids]``.
In the real env the manager resolves ``robot_cfg.joint_ids``; here we fake the env and
pass a SimpleNamespace robot_cfg so the math can be checked in isolation.
"""

from types import SimpleNamespace

import torch

from tests.helpers import stub
from src.tasks.hammer.mdp.velocity_bound import (
  CaTJointVelConstraint,
  SubstepPeakJointVel,
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


# --- Per-joint 500 Hz substep peak hold (P+V wave) -----------------------------------------------
# The scalar worst-joint `peak_qv` aliases WHICH joint sped: a per-joint soft-CaT margin needs its
# own column per joint, peak-held at substep rate over the same control window.

def _tracker(B: int = 3, J: int = 6, dec: int = 10):
  """SubstepPeakJointVel without __init__ (no real scene); drive it via `robot.data.joint_vel`."""
  robot = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(B, J)))
  return stub(
    SubstepPeakJointVel,
    _robot=robot,
    _joint_ids=list(range(J)),
    _dec=dec,
    peak_qv=torch.zeros(B),
    peak_qv_joint=torch.zeros(B, J),
    _i=0,
  )


def _feed(tracker, qv: torch.Tensor):
  tracker._robot.data.joint_vel = qv
  return tracker(None)


def test_substep_peak_holds_every_joint_independently():
  # Each joint peaks on a DIFFERENT substep; the per-joint buffer must keep all six maxima,
  # not just the column that happened to win the last substep.
  t = _tracker(B=1, J=6, dec=10)
  peaks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  for j, v in enumerate(peaks):
    qv = torch.full((1, 6), 0.1)
    qv[0, j] = v
    _feed(t, qv)
  assert torch.allclose(t.peak_qv_joint, torch.tensor([peaks])), t.peak_qv_joint


def test_substep_peak_uses_absolute_value_per_joint():
  # Negative joint velocity is just as illegal as positive.
  t = _tracker(B=1, J=6, dec=10)
  qv = torch.zeros(1, 6)
  qv[0, 2] = -4.2
  _feed(t, qv)
  assert torch.allclose(t.peak_qv_joint[0, 2], torch.tensor(4.2))


def test_substep_peak_scalar_interface_stays_the_worst_joint():
  # diag_policy_trace.py reads `.peak_qv` (B,) -- the legacy consumer must keep working.
  t = _tracker(B=2, J=6, dec=10)
  qv = torch.tensor([[0.5, 4.4, 0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0, 1.0, 2.7]])
  _feed(t, qv)
  assert t.peak_qv.shape == (2,)
  assert torch.allclose(t.peak_qv, torch.tensor([4.4, 2.7]))
  assert torch.allclose(t.peak_qv, t.peak_qv_joint.amax(dim=1))


def test_substep_peak_resets_at_every_control_window_boundary():
  # The window is `decimation` substeps long. Substep `dec` opens a NEW window: the previous
  # window's peak must not leak into it (otherwise the peak is monotone for the whole episode).
  dec = 4
  t = _tracker(B=1, J=6, dec=dec)
  hot = torch.full((1, 6), 5.0)
  cold = torch.full((1, 6), 0.5)
  for _ in range(dec):                       # window 0: hot
    _feed(t, hot)
  assert torch.allclose(t.peak_qv_joint, hot)
  _feed(t, cold)                             # window 1, first substep -> reset then record
  assert torch.allclose(t.peak_qv_joint, cold), t.peak_qv_joint
  assert torch.allclose(t.peak_qv, torch.tensor([0.5]))
  for _ in range(dec - 1):                   # rest of window 1 stays cold
    _feed(t, cold)
  assert torch.allclose(t.peak_qv_joint, cold)


def test_substep_peak_resets_across_episode_resets():
  t = _tracker(B=3, J=6, dec=10)
  _feed(t, torch.full((3, 6), 4.9))
  t.reset(torch.tensor([1]))                 # partial reset: only env 1 clears
  assert torch.allclose(t.peak_qv_joint[1], torch.zeros(6))
  assert torch.allclose(t.peak_qv_joint[0], torch.full((6,), 4.9))
  assert torch.allclose(t.peak_qv, torch.tensor([4.9, 0.0, 4.9]))
  t.reset(None)                              # full reset
  assert torch.allclose(t.peak_qv_joint, torch.zeros(3, 6))
  assert torch.allclose(t.peak_qv, torch.zeros(3))


def test_substep_peak_buffers_keep_their_identity_in_place():
  # CatSoftHook holds a reference to the tracker, not to the tensor -- but diag tooling grabs
  # `.peak_qv` once. Both buffers must be updated IN PLACE, never rebound.
  t = _tracker(B=2, J=6, dec=10)
  scalar_ref, joint_ref = t.peak_qv, t.peak_qv_joint
  _feed(t, torch.full((2, 6), 3.3))
  t.reset(None)
  _feed(t, torch.full((2, 6), 1.1))
  assert t.peak_qv is scalar_ref and t.peak_qv_joint is joint_ref
