"""C3 unit tests for the soft-CaT env hook (stub env; the full-sim check is scripts/verify_cat_soft.py).

The hook is a metrics term, so it structurally cannot trigger a reset (no reset assertion needed at
unit level — that invariant lives in the full-sim verify). These tests check the δ + r_pos + extras
contract via a faked env, in the style of tests/test_velocity_bound.py.
"""

from types import SimpleNamespace

import torch

from tests.helpers import stub
from src.tasks.hammer.cat import CaT
from src.tasks.hammer.cat.hook import (
  WINNER_IMPULSE,
  WINNER_NONE,
  WINNER_TIE,
  WINNER_VELOCITY,
  CatSoftHook,
  _NEG_TERMS,
)
from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.velocity_bound import _ENV_SUBSTEP_ATTR, Z1_JOINT_VEL_LIMIT

LIM = Z1_JOINT_VEL_LIMIT
ACTIVE = ["approach", "nail_driven", "nail_depth_delta", "impact_progress", "completion",
          "action_rate", "joint_pos_limits", "r_tt"]
I_ACTION_RATE = ACTIVE.index("action_rate")
I_JOINT_LIM = ACTIVE.index("joint_pos_limits")
I_R_TT = ACTIVE.index("r_tt")


def _hook(max_p=0.5, tau=0.95, min_p=0.0, vel_detection="control_rate"):
  # Built via helpers.stub (loud failure on __init__ drift). Historical note: the bare
  # object.__new__ version of this factory omitted all five _imp_* fields the real __init__
  # unconditionally assigns — safe only because the vel-only paths never read them. stub()
  # forces them to be set (to the class defaults) so the gap can never widen silently.
  return stub(
    CatSoftHook,
    _cat=CaT(tau=tau, min_p=min_p),
    _max_p=max_p,
    _limit=LIM,
    _robot_cfg=SimpleNamespace(name="robot", joint_ids=list(range(6))),
    _neg_idx=None,
    _use_vel=True,
    _use_impulse=False,
    _vel_detection=vel_detection,
    # __init__ defaults for the impulse fields (unread on the vel-only paths, but real):
    _imp_limit=torch.as_tensor(0.1, dtype=torch.float32),
    _imp_max_p=0.0,
    _imp_seed=1e-3,
    _imp_cmax=torch.full((1, 6), 1e-3),
    _imp_seeded=torch.zeros(1, 6, dtype=torch.bool),
    _last_impulse=None,
  )


def _ihook(imp_max_p=0.0, imp_seed=0.2, imp_limit=0.1, use_vel=False, tau=0.95, min_p=0.0, J=6,
           vel_detection="control_rate"):
  """Hook configured for the impulse arm (use_impulse=True). imp_max_p=0 ⇒ C0 log-only."""
  return stub(
    CatSoftHook,
    _cat=CaT(tau=tau, min_p=min_p),
    _max_p=0.5,
    _limit=LIM,
    _robot_cfg=SimpleNamespace(name="robot", joint_ids=list(range(6))),
    _neg_idx=None,
    _use_vel=use_vel,
    _use_impulse=True,
    _vel_detection=vel_detection,
    _imp_limit=imp_limit,
    _imp_max_p=imp_max_p,
    _imp_seed=imp_seed,
    _imp_cmax=torch.full((1, J), imp_seed),
    _imp_seeded=torch.zeros(1, J, dtype=torch.bool),
    _last_impulse=None,
  )


def _term_cfg(n):
  weights = {"action_rate": -0.01, "joint_pos_limits": -10.0, "r_tt": -1.0}
  w = weights.get(n, 1.0)
  return SimpleNamespace(weight=w)


def _env(qv, step_reward, step_dt=0.02, impulse=None):
  robot = SimpleNamespace(data=SimpleNamespace(joint_vel=qv, joint_pos=torch.zeros_like(qv)))
  rm = SimpleNamespace(_step_reward=step_reward, active_terms=list(ACTIVE),
                       get_term_cfg=_term_cfg, _scale_by_dt=True)
  env = SimpleNamespace(scene={"robot": robot}, reward_manager=rm, step_dt=step_dt, extras={})
  if impulse is not None:
    setattr(
      env,
      _ENV_SUBSTEP_IMPULSE_ATTR,
      SimpleNamespace(impulse=impulse, _joint_ids=list(range(impulse.shape[1]))),
    )
  return env


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
  step[:, I_R_TT] = -0.8                            # -> r_neg = -5.9
  dt = 0.02
  env = _env(torch.full((B, 6), 1.0), step, step_dt=dt)
  _hook()(env)
  # r_total_rate=4.1, r_neg_rate=-5.9 -> r_pos_rate=10.0 -> *dt
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
  r_neg = step[:, [I_ACTION_RATE, I_JOINT_LIM, I_R_TT]].sum(dim=1) * dt
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
  assert h._neg_idx == [I_ACTION_RATE, I_JOINT_LIM, I_R_TT]
  assert _NEG_TERMS == ("action_rate", "joint_pos_limits", "r_tt")


def test_nonzero_delta_leaves_all_three_penalties_unscaled() -> None:
  """CaT may discount only task return; r_tt cannot be evaded through velocity excess."""
  step = torch.zeros(1, len(ACTIVE))
  step[:, ACTIVE.index("approach")] = 10.0
  step[:, I_ACTION_RATE] = -0.1
  step[:, I_JOINT_LIM] = -5.0
  step[:, I_R_TT] = -0.8
  env = _env(torch.full((1, 6), 5.0), step)
  hook = _hook(max_p=0.5, tau=0.0)
  hook(env)
  delta = env.extras[CAT_DELTA_KEY]
  assert bool((delta > 0).all())
  r_pos = env.extras[CAT_R_POS_KEY]
  r_neg = step[:, [I_ACTION_RATE, I_JOINT_LIM, I_R_TT]].sum(dim=1) * env.step_dt
  cat_return = r_pos * (1.0 - delta) + r_neg
  expected = 10.0 * env.step_dt * (1.0 - delta) - 5.9 * env.step_dt
  torch.testing.assert_close(cat_return, expected)


def test_legacy_reward_table_without_optional_r_tt_keeps_previous_split() -> None:
  """FIC-0 and Cartesian tasks omit r_tt but retain the original two penalty columns."""
  legacy_active = [name for name in ACTIVE if name != "r_tt"]
  step = torch.zeros(1, len(legacy_active))
  step[:, legacy_active.index("approach")] = 4.0
  step[:, legacy_active.index("impact_progress")] = 6.0
  step[:, legacy_active.index("action_rate")] = -0.1
  step[:, legacy_active.index("joint_pos_limits")] = -5.0
  rm = SimpleNamespace(
    _step_reward=step,
    active_terms=legacy_active,
    get_term_cfg=_term_cfg,
    _scale_by_dt=True,
  )
  env = SimpleNamespace(
    scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_vel=torch.ones(1, 6)))},
    reward_manager=rm,
    step_dt=0.02,
    extras={},
  )
  hook = _hook()
  hook(env)
  assert hook._neg_idx == [
    legacy_active.index("action_rate"),
    legacy_active.index("joint_pos_limits"),
  ]
  torch.testing.assert_close(env.extras[CAT_R_POS_KEY], torch.tensor([0.2]))


def test_neg_sign_guard_raises_on_unregistered_negative_term():
  # MF-3: a future negative-weight reward term not in _NEG_TERMS would be silently discounted by
  # (1-δ) (penalty-evasion exploit). The hook must fail loudly on first __call__.
  import pytest

  active = list(ACTIVE) + ["rogue_penalty"]

  def term_cfg(n):
    w = -3.0 if n in ("action_rate", "joint_pos_limits", "r_tt", "rogue_penalty") else 1.0
    return SimpleNamespace(weight=w)

  rm = SimpleNamespace(_step_reward=torch.zeros(2, len(active)), active_terms=active,
                       get_term_cfg=term_cfg, _scale_by_dt=True)
  env = SimpleNamespace(
    scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_vel=torch.full((2, 6), 1.0)))},
    reward_manager=rm, step_dt=0.02, extras={})
  with pytest.raises(RuntimeError, match="rogue_penalty"):
    _hook()(env)


# --- Impulse arm (use_impulse=True) -------------------------------------------------------------

def test_impulse_log_only_delta_is_zero_even_over_limit():
  # C0 ships log-only (imp_max_p=0): δ must be identically 0 even when Λ is grossly over the limit.
  B = 4
  env = _env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.full((B, 6), 0.5))
  out = _ihook(imp_max_p=0.0)(env)
  assert out.shape == (B,)
  assert torch.allclose(out, torch.zeros(B)), out
  assert torch.allclose(env.extras[CAT_DELTA_KEY], torch.zeros(B))


def test_constraint_telemetry_log_only_preserves_exact_soft_or_and_shapes():
  """Pull-only telemetry exposes both arms without changing the log-only impulse invariant."""
  B, J = 3, 6
  caps = torch.tensor([0.82, 1.64, 0.82, 0.82, 0.82, 0.82])
  impulse = caps.unsqueeze(0).repeat(B, 1)
  impulse[:, 2] += 0.10
  env = _env(
    torch.full((B, J), 5.0),
    torch.zeros(B, len(ACTIVE)),
    impulse=impulse,
  )
  hook = _ihook(imp_max_p=0.0, imp_limit=caps, use_vel=True)
  returned = hook(env)
  telemetry = hook.constraint_telemetry()

  per_joint = (
    "delta_velocity_per_joint",
    "delta_impulse_per_joint",
    "lambda_per_joint",
    "cap_utilization_per_joint",
    "raw_margin_per_joint",
    "positive_margin_per_joint",
  )
  for name in per_joint:
    assert telemetry[name].shape == (B, J)
    assert torch.isfinite(telemetry[name]).all(), name
    assert telemetry[name].requires_grad is False
  for name in ("delta_velocity", "delta_impulse", "delta", "winner_constraint", "winner_joint"):
    assert telemetry[name].shape == (B,)
    assert torch.isfinite(telemetry[name]).all(), name

  assert torch.equal(telemetry["delta_impulse_per_joint"], torch.zeros(B, J))
  assert torch.equal(telemetry["delta_impulse"], torch.zeros(B))
  assert torch.equal(
    telemetry["delta"],
    torch.maximum(telemetry["delta_velocity"], telemetry["delta_impulse"]),
  )
  assert torch.equal(telemetry["delta"], returned)
  assert torch.equal(telemetry["winner_constraint"], torch.full((B,), WINNER_VELOCITY))
  assert torch.equal(telemetry["lambda_per_joint"], impulse)
  assert telemetry["active_limit_per_joint"].shape == (J,)
  assert torch.equal(telemetry["active_limit_per_joint"], caps)
  assert telemetry["active_limit_per_joint"].requires_grad is False
  torch.testing.assert_close(telemetry["raw_margin_per_joint"], impulse - caps)
  torch.testing.assert_close(
    telemetry["positive_margin_per_joint"], (impulse - caps).clamp_min(0.0)
  )
  torch.testing.assert_close(telemetry["cap_utilization_per_joint"], impulse / caps)

  # Every field is a detached clone: mutating a consumer's snapshot cannot corrupt the hook.
  telemetry["lambda_per_joint"].fill_(float("nan"))
  assert torch.isfinite(hook.constraint_telemetry()["lambda_per_joint"]).all()


def test_constraint_telemetry_returns_authoritative_nonroundtripping_lambda():
  """Lambda must come from the accumulator, not lossy (Lambda-cap)+cap reconstruction."""
  caps = torch.tensor([0.82, 1.64, 0.82, 0.82, 0.82, 0.82])
  impulse = torch.zeros(1, 6)
  impulse[0, 0] = 1.82
  reconstructed = (impulse - caps) + caps
  assert not torch.equal(reconstructed, impulse)
  env = _env(torch.zeros(1, 6), torch.zeros(1, len(ACTIVE)), impulse=impulse)
  hook = _ihook(imp_max_p=0.0, imp_limit=caps)

  hook(env)
  telemetry = hook.constraint_telemetry()

  assert torch.equal(telemetry["lambda_per_joint"], impulse)
  assert torch.equal(telemetry["raw_margin_per_joint"], impulse - caps)


def test_impulse_hook_rejects_same_width_permuted_accumulator_joint_ids():
  """A six-column Lambda tensor is unsafe unless its joint order exactly matches the hook."""
  import pytest

  impulse = torch.zeros(1, 6)
  env = _env(torch.zeros(1, 6), torch.zeros(1, len(ACTIVE)), impulse=impulse)
  getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)._joint_ids = [1, 0, 2, 3, 4, 5]

  with pytest.raises(RuntimeError, match="joint ids"):
    _ihook(imp_max_p=0.0, imp_limit=torch.ones(6))(env)


def test_constraint_telemetry_attributes_active_impulse_to_j3():
  B, J = 2, 6
  caps = torch.tensor([0.82, 1.64, 0.82, 0.82, 0.82, 0.82])
  impulse = torch.zeros(B, J)
  impulse[:, 2] = 1.02  # J3 exceeds its cap by 0.20 N.m.s.
  env = _env(torch.zeros(B, J), torch.zeros(B, len(ACTIVE)), impulse=impulse)
  hook = _ihook(imp_max_p=0.5, imp_seed=0.2, imp_limit=caps)
  hook(env)
  telemetry = hook.constraint_telemetry()

  assert torch.equal(telemetry["delta_velocity"], torch.zeros(B))
  assert torch.equal(telemetry["delta"], telemetry["delta_impulse"])
  assert torch.equal(telemetry["winner_constraint"], torch.full((B,), WINNER_IMPULSE))
  assert torch.equal(telemetry["winner_joint"], torch.full((B,), 2))
  torch.testing.assert_close(
    telemetry["positive_margin_per_joint"][:, 2], torch.full((B,), 0.20)
  )
  assert (telemetry["delta_impulse_per_joint"][:, 2] > 0.0).all()
  assert torch.equal(
    telemetry["delta_impulse_per_joint"][:, [0, 1, 3, 4, 5]],
    torch.zeros(B, J - 1),
  )


def test_constraint_telemetry_attributes_simultaneous_velocity_impulse_and_tie():
  """The soft-OR winner must remain attributable when both constraint arms are active."""
  B, J = 3, 6
  caps = torch.ones(J)
  qv = torch.zeros(B, J)
  qv[:, 0] = torch.tensor([1.0, 0.25, 0.5])
  impulse = torch.zeros(B, J)
  impulse[:, 0] = torch.tensor([1.25, 2.0, 1.5])
  env = _env(qv, torch.zeros(B, len(ACTIVE)), impulse=impulse)
  hook = _ihook(imp_max_p=0.5, imp_seed=0.1, imp_limit=caps, use_vel=True, tau=0.0)
  hook._limit = 0.0

  returned = hook(env)
  telemetry = hook.constraint_telemetry()

  assert torch.equal(
    telemetry["winner_constraint"],
    torch.tensor([WINNER_VELOCITY, WINNER_IMPULSE, WINNER_TIE]),
  )
  assert torch.equal(telemetry["winner_joint"], torch.tensor([0, 0, -1]))
  assert torch.equal(
    telemetry["delta"],
    torch.maximum(telemetry["delta_velocity"], telemetry["delta_impulse"]),
  )
  assert torch.equal(returned, telemetry["delta"])
  torch.testing.assert_close(telemetry["delta_velocity"], torch.tensor([0.5, 0.125, 0.25]))
  torch.testing.assert_close(telemetry["delta_impulse"], torch.tensor([0.125, 0.5, 0.25]))


def test_constraint_telemetry_reports_none_when_neither_constraint_activates():
  B, J = 2, 6
  caps = torch.tensor([0.82, 1.64, 0.82, 0.82, 0.82, 0.82])
  env = _env(torch.zeros(B, J), torch.zeros(B, len(ACTIVE)), impulse=torch.zeros(B, J))
  hook = _ihook(imp_max_p=0.5, imp_limit=caps)
  hook(env)
  telemetry = hook.constraint_telemetry()
  assert torch.equal(telemetry["winner_constraint"], torch.full((B,), WINNER_NONE))
  assert torch.equal(telemetry["winner_joint"], torch.full((B,), -1))


def test_impulse_raw_margin_logged_for_histogram():
  # Even log-only, the RAW per-joint margin must be stored (the C0 histogram / binding-ness gate).
  B = 4
  env = _env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.full((B, 6), 0.5))
  h = _ihook(imp_max_p=0.0, imp_limit=0.1)
  h(env)
  assert "joint_impulse_excess" in h._cat.raw_constraints
  assert torch.allclose(h._cat.raw_constraints["joint_impulse_excess"], torch.full((B, 6), 0.4))


def test_impulse_only_arm_returns_B_shaped_delta():
  # use_vel=False, use_impulse=True: get_probs must still yield (B,) (no empty-tensor bug when the
  # impulse term is the only term and it is always added).
  B = 3
  env = _env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.zeros(B, 6))
  out = _ihook(imp_max_p=0.5)(env)
  assert out.shape == (B,)
  assert torch.allclose(out, torch.zeros(B))  # under limit -> 0


def test_impulse_delta_graded_not_saturated():
  # imp_max_p>0 with a meaningful seed: a moderate excess yields δ well below max_p (GRADED), the
  # property the velocity batch-max EMA would destroy on a sparse signal.
  B = 4
  h = _ihook(imp_max_p=0.5, imp_seed=0.2, imp_limit=0.1)
  env = _env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.full((B, 6), 0.15))  # excess 0.05
  out = h(env)
  assert (out > 0).all() and (out < 0.5 * 0.9).all(), out  # graded, not pinned at max_p


def test_impulse_normalizer_no_collapse_after_idle():
  # The EMA-collapse guard: many idle (under-limit) steps must NOT decay the normalizer to ~1e-6
  # (which would saturate δ to max_p on the next contact). Seed-floored + contact-masked update.
  B = 4
  h = _ihook(imp_max_p=0.5, imp_seed=0.2, imp_limit=0.1)
  for _ in range(50):  # idle: Λ=0 -> margin negative -> no normalizer update
    h(_env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.zeros(B, 6)))
  out = h(_env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.full((B, 6), 0.2)))  # excess 0.1
  assert (out > 0).all() and (out < 0.5 * 0.95).all(), out  # NOT saturated => normalizer held at seed


def test_impulse_soft_or_combines_with_velocity():
  # Combined arm: a compliant velocity but over-limit impulse must still drive δ (soft-OR MAX).
  B = 4
  h = _ihook(imp_max_p=0.5, use_vel=True, imp_seed=0.2, imp_limit=0.1)
  env = _env(torch.full((B, 6), 2.0), torch.zeros(B, len(ACTIVE)), impulse=torch.full((B, 6), 0.6))
  out = h(env)  # vel 2.0 < 3.1415 (δ_vel=0); impulse over -> δ_imp>0
  assert (out > 0).all(), out


def test_impulse_log_only_zero_even_with_positive_min_p():
  # Review finding #7: with min_p>0 the old formula gave δ = min_p·(1−normalized) > 0 under
  # imp_max_p=0 (and INVERSELY graded). Log-only must mean δ_imp ≡ 0 regardless of min_p.
  B = 4
  h = _ihook(imp_max_p=0.0, min_p=0.05, imp_seed=0.2, imp_limit=0.1)
  env = _env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.full((B, 6), 0.15))
  out = h(env)  # over the limit, but log-only
  assert torch.allclose(out, torch.zeros(B)), out
  # ... while the RAW margin is still recorded for the C0 histogram.
  assert "joint_impulse_excess" in h._cat.raw_constraints


def test_impulse_ema_update_is_per_column():
  # Review hygiene: a violating joint must not decay a NON-violating joint's normalizer.
  B = 4
  h = _ihook(imp_max_p=0.5, imp_seed=0.1, imp_limit=0.1)
  imp = torch.full((B, 6), 0.05)  # under limit everywhere...
  imp[:, 0] = 1.2                 # ...except joint0, grossly over
  for _ in range(10):
    h(_env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=imp))
  # joint0's cmax grew toward its excess; every other column stayed at the seed (NOT decayed).
  assert h._imp_cmax[0, 0] > 0.1
  assert torch.allclose(h._imp_cmax[0, 1:], torch.full((5,), 0.1)), h._imp_cmax


def test_impulse_first_violation_seeds_scale():
  # xhigh-review finding: a tiny imp_seed (shipped 1e-3) must NOT cause long saturation at C2.
  # The first over-limit sample per column SEEDS cmax to max(batch_max, seed) — CaT.add's own
  # first-batch seeding — so grading is scale-correct from the second violation on.
  B = 4
  h = _ihook(imp_max_p=0.5, imp_seed=1e-3, imp_limit=0.1)
  h(_env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.full((B, 6), 1.1)))  # excess 1.0
  assert torch.allclose(h._imp_cmax[0, 0], torch.tensor(1.0), atol=1e-6), h._imp_cmax  # seeded, not 1e-3
  out = h(_env(torch.zeros(B, 6), torch.zeros(B, len(ACTIVE)), impulse=torch.full((B, 6), 0.6)))  # excess 0.5
  assert (out < 0.5 * 0.8).all() and (out > 0).all(), out  # graded (~0.25), not pinned at max_p


def test_validate_params_guards():
  import pytest

  # Both constraints off: expressible misconfiguration must fail loudly at construction.
  with pytest.raises(RuntimeError, match="use_vel"):
    CatSoftHook._validate_params({"use_vel": False, "use_impulse": False})
  # Enforcement against the not-to-ship placeholder limit must fail loudly (env_cfgs passes the
  # placeholder EXPLICITLY at C0, so presence-checking alone cannot catch the C2 flip).
  with pytest.raises(RuntimeError, match="placeholder"):
    CatSoftHook._validate_params(
      {"use_vel": False, "use_impulse": True, "imp_limit": 0.1, "imp_max_p": 0.5}
    )
  # …including the per-joint LIST and TENSOR forms (the original scalar-only type check let an
  # all-placeholder vector through silently).
  with pytest.raises(RuntimeError, match="placeholder"):
    CatSoftHook._validate_params(
      {"use_vel": False, "use_impulse": True, "imp_limit": [0.1] * 6, "imp_max_p": 0.5}
    )
  with pytest.raises(RuntimeError, match="placeholder"):
    CatSoftHook._validate_params(
      {"use_vel": False, "use_impulse": True, "imp_limit": torch.full((6,), 0.1), "imp_max_p": 0.5}
    )
  # A per-joint limit that merely CONTAINS 0.1 but is not all-placeholder is legitimate.
  CatSoftHook._validate_params(
    {"use_vel": False, "use_impulse": True, "imp_limit": [0.1, 3.28, 1.64, 1.64, 1.64, 1.64],
     "imp_max_p": 0.5}
  )
  # use_impulse without an explicit imp_limit: the placeholder must never be a silent fallback.
  with pytest.raises(RuntimeError, match="imp_limit"):
    CatSoftHook._validate_params({"use_vel": False, "use_impulse": True, "imp_max_p": 0.0})
  # Enforcement ceiling below the probability floor is incoherent (inverse grading).
  with pytest.raises(RuntimeError, match="min_p"):
    CatSoftHook._validate_params(
      {"use_vel": False, "use_impulse": True, "imp_limit": 1.0, "imp_max_p": 0.1, "min_p": 0.3}
    )
  # F5 (2026-07-19 audit): full domain validation. A δ ceiling > 1 (e.g. CLI --imp-max-p 2) would
  # make a termination probability > 1.
  with pytest.raises(RuntimeError, match=r"in \[0, 1\]"):
    CatSoftHook._validate_params(
      {"use_vel": False, "use_impulse": True, "imp_limit": 1.0, "imp_max_p": 2.0}
    )
  # tau outside [0, 1) breaks the EMA normalizer.
  with pytest.raises(RuntimeError, match="tau"):
    CatSoftHook._validate_params(
      {"use_vel": True, "use_impulse": False, "tau": 1.5}
    )
  # Non-positive normalizer seed collapses the per-column scale.
  with pytest.raises(RuntimeError, match="imp_seed"):
    CatSoftHook._validate_params(
      {"use_vel": True, "use_impulse": False, "imp_seed": 0.0}
    )
  # A zero/NaN per-joint cap makes the margin Λ−limit undefined or trivially violated.
  with pytest.raises(RuntimeError, match="finite"):
    CatSoftHook._validate_params(
      {"use_vel": False, "use_impulse": True, "imp_max_p": 0.5,
       "imp_limit": [0.0, 3.28, 1.64, 1.64, 1.64, 1.64]}
    )
  # Valid configs pass.
  CatSoftHook._validate_params({"use_vel": True, "use_impulse": False})
  CatSoftHook._validate_params(
    {"use_vel": False, "use_impulse": True, "imp_limit": 1.0, "imp_max_p": 0.5}
  )


# --- Substep (500 Hz) per-joint velocity detection: the P+V treatment ---------------------------
# The shipped hook reads CONTROL-RATE joint_vel, which aliases the within-window peak (the strike
# spike). P+V enforces on the peak-held per-joint |q̇| the SubstepPeakJointVel tracker records at
# 500 Hz. Detection must be OPT-IN and FAIL CLOSED: a missing tracker must never fall back to the
# aliased signal, because that would silently train a different (weaker) treatment than the one
# preregistered.

def _substep_env(control_rate_qv, peak_qv_joint, step_reward=None, impulse=None):
  """Env whose control-rate sample and substep peak DISAGREE, so the read source is provable."""
  B = control_rate_qv.shape[0]
  env = _env(control_rate_qv, torch.zeros(B, len(ACTIVE)) if step_reward is None else step_reward,
             impulse=impulse)
  setattr(env, _ENV_SUBSTEP_ATTR,
          SimpleNamespace(peak_qv_joint=peak_qv_joint,
                          peak_qv=peak_qv_joint.amax(dim=1),
                          _joint_ids=list(range(peak_qv_joint.shape[1]))))
  return env


def test_substep_detection_reads_the_peak_not_the_control_rate_sample():
  B = 4
  # Control rate says LEGAL (1.0); the 500 Hz peak says grossly illegal. δ must follow the peak.
  env = _substep_env(torch.full((B, 6), 1.0), torch.full((B, 6), 5.0))
  out = _hook(vel_detection="substep")(env)
  assert (out > 0).all(), out


def test_substep_detection_ignores_an_illegal_control_rate_sample():
  B = 4
  # The mirror image: control rate says illegal, the peak-held window says legal. Under substep
  # detection the peak is authoritative -- if this fired, the hook would be reading joint_vel.
  env = _substep_env(torch.full((B, 6), 5.0), torch.full((B, 6), 1.0))
  out = _hook(vel_detection="substep")(env)
  assert torch.allclose(out, torch.zeros(B)), out


def test_substep_detection_zero_delta_at_or_below_the_limit():
  B = 3
  # Control rate deliberately illegal, so this also fails if the peak is not the read source.
  env = _substep_env(torch.full((B, 6), 5.0), torch.full((B, 6), LIM))   # peak exactly at limit
  h = _hook(vel_detection="substep")
  assert torch.allclose(h(env), torch.zeros(B))
  assert torch.allclose(env.extras[CAT_DELTA_KEY], torch.zeros(B))


def test_substep_detection_any_single_joint_over_the_limit_gives_positive_delta():
  # Six independent one-joint violations: each joint on its own must be able to drive δ.
  for j in range(6):
    peak = torch.zeros(1, 6)
    peak[0, j] = LIM + 0.5
    out = _hook(vel_detection="substep")(_substep_env(torch.zeros(1, 6), peak))
    assert float(out[0]) > 0.0, (j, out)


def test_substep_detection_keeps_each_violating_joint_in_its_own_column():
  # Two joints over by DIFFERENT amounts: the raw margin must stay per-joint (c_j = peak_j - limit),
  # not be collapsed to the worst-joint scalar the legacy tracker interface exposes.
  B = 2
  peak = torch.full((B, 6), 1.0)
  peak[:, 1] = LIM + 0.4
  peak[:, 4] = LIM + 1.2
  h = _hook(vel_detection="substep")
  h(_substep_env(torch.zeros(B, 6), peak))
  c = h._cat.raw_constraints["joint_velocity_excess"]
  assert c.shape == (B, 6)
  assert torch.allclose(c, peak - LIM, atol=1e-5), c


def test_substep_detection_fails_closed_when_the_tracker_is_missing():
  import pytest

  B = 2
  env = _env(torch.full((B, 6), 1.0), torch.zeros(B, len(ACTIVE)))   # no tracker stashed
  with pytest.raises(RuntimeError, match="SubstepPeakJointVel"):
    _hook(vel_detection="substep")(env)


def test_substep_detection_fails_closed_on_a_joint_count_mismatch():
  import pytest

  B = 2
  env = _substep_env(torch.zeros(B, 6), torch.full((B, 3), 5.0))     # tracker on a 3-joint set
  with pytest.raises(RuntimeError, match="joint"):
    _hook(vel_detection="substep")(env)


def test_control_rate_detection_is_the_default_and_still_reads_joint_vel():
  # Regression guard for Unitree-Z1-Hammer-CaT-Soft: the shipped arm must be untouched. A tracker
  # is present and says LEGAL, but the control-rate sample says illegal -- default reads joint_vel.
  B = 3
  env = _substep_env(torch.full((B, 6), 5.0), torch.full((B, 6), 1.0))
  assert (_hook()(env) > 0).all()                                    # default == control_rate
  assert (_hook(vel_detection="control_rate")(env) > 0).all()


def test_substep_detection_composes_with_the_impulse_soft_or():
  # P+V is ONE hook carrying both constraints: substep velocity (enforcing) and impulse (log-only).
  B = 4
  peak = torch.full((B, 6), 5.0)                                     # velocity illegal
  env = _substep_env(torch.full((B, 6), 1.0), peak, impulse=torch.zeros(B, 6))
  h = _ihook(imp_max_p=0.0, use_vel=True, imp_limit=0.1, vel_detection="substep")
  out = h(env)
  assert (out > 0).all(), out                                        # velocity alone drives δ
  assert "joint_impulse_excess" in h._cat.raw_constraints            # impulse still logged


def test_validate_params_rejects_an_unknown_velocity_detection():
  import pytest

  with pytest.raises(RuntimeError, match="vel_detection"):
    CatSoftHook._validate_params({"use_vel": True, "use_impulse": False,
                                  "vel_detection": "substep_peak"})
  # The two supported modes pass.
  CatSoftHook._validate_params({"use_vel": True, "use_impulse": False,
                                "vel_detection": "control_rate"})
  CatSoftHook._validate_params({"use_vel": True, "use_impulse": False,
                                "vel_detection": "substep"})


def test_validate_params_rejects_substep_detection_without_velocity_enforcement():
  # FAIL-OPEN (review finding): use_vel and vel_detection are tyro-exposed params. With
  # use_vel=False the hook skips _vel_margin entirely and delta falls back to the impulse
  # term, which imp_max_p=0.0 hard-zeroes -- so the arm trains as a bit-for-bit copy of the
  # unenforced control while still LOOKING like the velocity arm. Six GPU runs later "CaT
  # did not bound velocity" would be indistinguishable from "CaT was never on".
  import pytest

  with pytest.raises(RuntimeError, match="use_vel"):
    CatSoftHook._validate_params(
      {"use_vel": False, "use_impulse": True, "imp_limit": 1.0,
       "vel_detection": "substep"}
    )


def test_validate_params_rejects_velocity_enforcement_at_a_zero_ceiling():
  # Same fail-open by a different flag: max_p=0 makes delta_vel identically zero.
  import pytest

  with pytest.raises(RuntimeError, match="max_p"):
    CatSoftHook._validate_params(
      {"use_vel": True, "use_impulse": False, "max_p": 0.0}
    )
  # ... and it must not fire for an arm that genuinely runs no velocity constraint.
  CatSoftHook._validate_params({"use_vel": False, "use_impulse": True,
                                "imp_limit": 1.0, "max_p": 0.0})


def test_substep_detection_checks_joint_IDENTITY_not_merely_column_count():
  # A same-width but DIFFERENT (or permuted) joint set passes a shape check while every
  # margin column is normalized by, and attributed to, the wrong joint's EMA -- silently.
  import pytest

  B = 2
  env = _substep_env(torch.zeros(B, 6), torch.full((B, 6), 5.0))
  getattr(env, _ENV_SUBSTEP_ATTR)._joint_ids = [5, 4, 3, 2, 1, 0]  # permuted, same width
  h = _hook(vel_detection="substep")
  with pytest.raises(RuntimeError, match="joint"):
    h(env)
