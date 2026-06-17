"""C3 unit tests for the soft-CaT env hook (stub env; the full-sim check is scripts/verify_cat_soft.py).

The hook is a metrics term, so it structurally cannot trigger a reset (no reset assertion needed at
unit level — that invariant lives in the full-sim verify). These tests check the δ + r_pos + extras
contract via a faked env, in the style of tests/test_velocity_bound.py.
"""

from types import SimpleNamespace

import torch

from src.tasks.hammer.cat import CaT
from src.tasks.hammer.cat.hook import CatSoftHook, _NEG_TERMS
from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY
from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT

LIM = Z1_JOINT_VEL_LIMIT
ACTIVE = ["approach", "nail_driven", "nail_depth_delta", "impact_progress", "completion",
          "action_rate", "joint_pos_limits"]
I_ACTION_RATE, I_JOINT_LIM = ACTIVE.index("action_rate"), ACTIVE.index("joint_pos_limits")


def _hook(max_p=0.5, tau=0.95, min_p=0.0):
  h = object.__new__(CatSoftHook)               # bypass ManagerTermBase.__init__ (no real env)
  h._cat = CaT(tau=tau, min_p=min_p)
  h._max_p, h._limit = max_p, LIM
  h._robot_cfg = SimpleNamespace(name="robot", joint_ids=list(range(6)))
  h._neg_idx = None
  return h


def _env(qv, step_reward, step_dt=0.02):
  robot = SimpleNamespace(data=SimpleNamespace(joint_vel=qv))
  rm = SimpleNamespace(_step_reward=step_reward, active_terms=list(ACTIVE))
  return SimpleNamespace(scene={"robot": robot}, reward_manager=rm, step_dt=step_dt, extras={})


def test_writes_extras_keys_with_shape_and_range():
  B = 4
  env = _env(torch.full((B, 6), 5.0), torch.zeros(B, len(ACTIVE)))  # all over the limit
  out = _hook(max_p=0.5)(env)
  assert out.shape == (B,) and out.dtype == torch.float32
  assert CAT_DELTA_KEY in env.extras and CAT_R_POS_KEY in env.extras
  d = env.extras[CAT_DELTA_KEY]
  assert d.shape == (B,)
  assert (d >= 0).all() and (d <= 0.5 + 1e-6).all()
  assert (d > 0).all()                            # over the limit -> δ > 0


def test_delta_zero_under_limit():
  B = 3
  env = _env(torch.full((B, 6), 2.0), torch.zeros(B, len(ACTIVE)))  # 2.0 < 3.1415
  _hook()(env)
  assert torch.allclose(env.extras[CAT_DELTA_KEY], torch.zeros(B))


def test_r_pos_excludes_negative_terms_and_is_dt_scaled():
  B = 2
  step = torch.zeros(B, len(ACTIVE))
  step[:, ACTIVE.index("approach")] = 4.0          # +4
  step[:, ACTIVE.index("impact_progress")] = 6.0   # +6  -> positives = 10
  step[:, I_ACTION_RATE] = -0.1                     # negatives ...
  step[:, I_JOINT_LIM] = -5.0                       # -> r_neg = -5.1
  dt = 0.02
  env = _env(torch.full((B, 6), 1.0), step, step_dt=dt)
  _hook()(env)
  # r_total_rate=4.9, r_neg_rate=-5.1 -> r_pos_rate=10.0 -> *dt
  assert torch.allclose(env.extras[CAT_R_POS_KEY], torch.full((B,), 10.0 * dt), atol=1e-6)


def test_r_pos_equals_reward_minus_negatives_identity():
  # r_pos must equal reward_buf - (dt-scaled negative terms): the exact CatPPO consistency relation.
  B = 3
  torch.manual_seed(0)
  step = torch.randn(B, len(ACTIVE))
  dt = 0.02
  env = _env(torch.zeros(B, 6), step, step_dt=dt)
  _hook()(env)
  reward_buf = step.sum(dim=1) * dt                                  # what the env returns
  r_neg = step[:, [I_ACTION_RATE, I_JOINT_LIM]].sum(dim=1) * dt
  assert torch.allclose(env.extras[CAT_R_POS_KEY], reward_buf - r_neg, atol=1e-6)


def test_reset_clears_probs_keeps_ema():
  h = _hook()
  env = _env(torch.full((4, 6), 6.0), torch.zeros(4, len(ACTIVE)))
  h(env)
  assert "joint_velocity_excess" in h._cat.running_maxes
  h.reset(None)
  assert h._cat.probs == {}                         # per-step buffers cleared
  assert "joint_velocity_excess" in h._cat.running_maxes  # EMA normalizer persists


def test_neg_term_indices_resolved_lazily():
  h = _hook()
  assert h._neg_idx is None
  h(_env(torch.full((1, 6), 1.0), torch.zeros(1, len(ACTIVE))))
  assert h._neg_idx == [I_ACTION_RATE, I_JOINT_LIM]
  assert _NEG_TERMS == ("action_rate", "joint_pos_limits")
