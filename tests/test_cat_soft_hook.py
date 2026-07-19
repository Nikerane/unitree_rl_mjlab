"""C3 unit tests for the soft-CaT env hook (stub env; the full-sim check is scripts/verify_cat_soft.py).

The hook is a metrics term, so it structurally cannot trigger a reset (no reset assertion needed at
unit level — that invariant lives in the full-sim verify). These tests check the δ + r_pos + extras
contract via a faked env, in the style of tests/test_velocity_bound.py.
"""

from types import SimpleNamespace

import torch

from tests.helpers import stub
from src.tasks.hammer.cat import CaT
from src.tasks.hammer.cat.hook import CatSoftHook, _NEG_TERMS
from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT

LIM = Z1_JOINT_VEL_LIMIT
ACTIVE = ["approach", "nail_driven", "nail_depth_delta", "impact_progress", "completion",
          "action_rate", "joint_pos_limits"]
I_ACTION_RATE, I_JOINT_LIM = ACTIVE.index("action_rate"), ACTIVE.index("joint_pos_limits")


def _hook(max_p=0.5, tau=0.95, min_p=0.0):
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
    # __init__ defaults for the impulse fields (unread on the vel-only paths, but real):
    _imp_limit=torch.as_tensor(0.1, dtype=torch.float32),
    _imp_max_p=0.0,
    _imp_seed=1e-3,
    _imp_cmax=torch.full((1, 6), 1e-3),
    _imp_seeded=torch.zeros(1, 6, dtype=torch.bool),
  )


def _ihook(imp_max_p=0.0, imp_seed=0.2, imp_limit=0.1, use_vel=False, tau=0.95, min_p=0.0, J=6):
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
    _imp_limit=imp_limit,
    _imp_max_p=imp_max_p,
    _imp_seed=imp_seed,
    _imp_cmax=torch.full((1, J), imp_seed),
    _imp_seeded=torch.zeros(1, J, dtype=torch.bool),
  )


def _term_cfg(n):
  w = -0.01 if n == "action_rate" else (-10.0 if n == "joint_pos_limits" else 1.0)
  return SimpleNamespace(weight=w)


def _env(qv, step_reward, step_dt=0.02, impulse=None):
  robot = SimpleNamespace(data=SimpleNamespace(joint_vel=qv))
  rm = SimpleNamespace(_step_reward=step_reward, active_terms=list(ACTIVE),
                       get_term_cfg=_term_cfg, _scale_by_dt=True)
  env = SimpleNamespace(scene={"robot": robot}, reward_manager=rm, step_dt=step_dt, extras={})
  if impulse is not None:
    setattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, SimpleNamespace(impulse=impulse))
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


def test_neg_sign_guard_raises_on_unregistered_negative_term():
  # MF-3: a future negative-weight reward term not in _NEG_TERMS would be silently discounted by
  # (1-δ) (penalty-evasion exploit). The hook must fail loudly on first __call__.
  import pytest

  active = list(ACTIVE) + ["rogue_penalty"]

  def term_cfg(n):
    w = -3.0 if n in ("action_rate", "joint_pos_limits", "rogue_penalty") else 1.0
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
  # …including the per-joint LIST and TENSOR forms (2026-07-14 audit: the original isinstance
  # (int, float) check let [0.1]*6 through to ~23-69x-too-tight enforcement silently).
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
