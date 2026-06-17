"""C2 unit tests for CatPPO: the scale-positives reward discount + the dual-mask γ(1−δ)(1−true) GAE.

Tested with no models/sim by calling the methods with a stub ``self`` (the GAE/scale logic only reads
plain attributes). The dual-mask GAE is cross-checked against (a) stock rsl_rl PPO.compute_returns
(δ=0 must reproduce it), (b) an INDEPENDENT re-implementation of the CaT reference's
`nextnonterminal * true_nextnonterminal` GAE (cleanrl/ppo.py:255-276), and (c) the
combined-float-done identity. The incentive-to-violate guard (test (iv) of the impl plan) is the
regression that motivated scale-positives over full-reward (1−δ).
"""

from types import SimpleNamespace

import torch

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage.rollout_storage import RolloutStorage

from src.tasks.hammer.rl.cat_ppo import CatPPO

GAMMA, LAM = 0.99, 0.95


# ----------------------------------------------------------------------------- helpers

def _storage(rewards, values, dones, soft_dones):
  T, B, _ = rewards.shape
  return SimpleNamespace(
    num_transitions_per_env=T,
    rewards=rewards.clone(), values=values.clone(),
    dones=dones.clone(), soft_dones=soft_dones.clone(),
    returns=torch.zeros(T, B, 1), advantages=torch.zeros(T, B, 1),
  )


def _alg(storage, last_value, gamma=GAMMA, lam=LAM):
  # normalize_advantage_per_mini_batch=True => compute_returns SKIPS advantage normalization, so the
  # raw returns are directly comparable across runs.
  return SimpleNamespace(
    storage=storage, gamma=gamma, lam=lam,
    normalize_advantage_per_mini_batch=True,
    critic=lambda obs: last_value,
  )


def _run(compute_returns_fn, rewards, values, dones, soft, last_value):
  st = _storage(rewards, values, dones, soft)
  compute_returns_fn(_alg(st, last_value), None)
  return st.returns


def _reference_dual_gae(rewards, values, soft, hard, last_value, gamma=GAMMA, lam=LAM):
  """Independent re-implementation of the CaT reference GAE (cleanrl/ppo.py:255-276), post-step
  indexed to match rsl_rl. A second source of truth for compute_returns."""
  T, B, _ = rewards.shape
  advantages = torch.zeros_like(rewards)
  lastgae = torch.zeros(B, 1)
  for t in reversed(range(T)):
    nextvalues = last_value if t == T - 1 else values[t + 1]
    nextnonterminal = 1.0 - soft[t]
    true_nextnonterminal = 1.0 - hard[t].float()
    mask = nextnonterminal * true_nextnonterminal
    delta = rewards[t] + gamma * nextvalues * mask - values[t]
    lastgae = delta + gamma * lam * mask * lastgae
    advantages[t] = lastgae
  return advantages + values


# ----------------------------------------------------------------------------- scale-positives reward

def test_scale_positives_formula():
  rewards = torch.tensor([2.0, -8.0, 5.0])
  delta = torch.tensor([0.5, 0.5, 0.0])
  r_pos = torch.tensor([12.0, 2.0, 5.0])
  out = CatPPO._cat_scale_reward(rewards, delta, r_pos)
  assert torch.allclose(out, rewards - delta * r_pos)


def test_scale_positives_preserves_penalty():
  # env: r_pos=+2 (nail progress), r_neg=-10 (joint limit) -> r_total=-8; violating (delta=0.5).
  # scale-positives: -8 - 0.5*2 = -9. The -10 penalty is fully present (it's not discounted).
  out = CatPPO._cat_scale_reward(torch.tensor([-8.0]), torch.tensor([0.5]), torch.tensor([2.0]))
  assert torch.allclose(out, torch.tensor([-9.0]))


def test_scale_delta_zero_is_identity():
  rewards = torch.tensor([1.0, -3.0, 7.0])
  out = CatPPO._cat_scale_reward(rewards, torch.zeros(3), torch.tensor([5.0, 5.0, 5.0]))
  assert torch.allclose(out, rewards)


def test_scale_handles_column_shapes():
  rewards = torch.tensor([[2.0], [-8.0]])          # (B,1)
  out = CatPPO._cat_scale_reward(rewards, torch.tensor([0.5, 0.5]), torch.tensor([4.0, 2.0]))
  assert out.shape == rewards.shape
  assert torch.allclose(out, torch.tensor([[0.0], [-9.0]]))


# ----------------------------------------------------------------------------- dual-mask GAE

def test_delta_zero_reproduces_stock_gae():
  torch.manual_seed(0)
  T, B = 4, 3
  rewards, values = torch.randn(T, B, 1), torch.randn(T, B, 1)
  dones = torch.zeros(T, B, 1); dones[3, 0, 0] = 1.0  # a real termination
  soft = torch.zeros(T, B, 1)                          # delta = 0
  last_v = torch.randn(B, 1)
  cat = _run(CatPPO.compute_returns, rewards, values, dones, soft, last_v)
  stock = _run(PPO.compute_returns, rewards, values, dones, soft, last_v)
  assert torch.allclose(cat, stock, atol=1e-6)


def test_dual_mask_matches_independent_reference_impl():
  torch.manual_seed(1)
  T, B = 5, 4
  rewards, values = torch.randn(T, B, 1), torch.randn(T, B, 1)
  hard = (torch.rand(T, B, 1) > 0.85).float()         # sparse real terminations
  soft = torch.rand(T, B, 1) * 0.6                     # soft deltas
  last_v = torch.randn(B, 1)
  cat = _run(CatPPO.compute_returns, rewards, values, hard, soft, last_v)
  ref = _reference_dual_gae(rewards, values, soft, hard, last_v)
  assert torch.allclose(cat, ref, atol=1e-6)


def test_dual_mask_equals_combined_float_done():
  # stock single-mask GAE fed a COMBINED float done dc = 1-(1-delta)(1-hard) must equal our dual mask.
  torch.manual_seed(2)
  T, B = 4, 3
  rewards, values = torch.randn(T, B, 1), torch.randn(T, B, 1)
  hard = (torch.rand(T, B, 1) > 0.8).float()
  soft = torch.rand(T, B, 1) * 0.5
  last_v = torch.randn(B, 1)
  dual = _run(CatPPO.compute_returns, rewards, values, hard, soft, last_v)
  combined = 1.0 - (1.0 - soft) * (1.0 - hard)         # float
  stock = _run(PPO.compute_returns, rewards, values, combined, torch.zeros(T, B, 1), last_v)
  assert torch.allclose(dual, stock, atol=1e-5)


def test_hard_done_cuts_bootstrap_across_reset():
  # A real termination at step 0 must zero the bootstrap of V[1] into return[0].
  rewards = torch.tensor([[[1.0]], [[1.0]]])
  values = torch.zeros(2, 1, 1)
  hard = torch.tensor([[[1.0]], [[0.0]]])              # step0 terminated
  soft = torch.zeros(2, 1, 1)
  out = _run(CatPPO.compute_returns, rewards, values, hard, soft, last_value=torch.tensor([[5.0]]))
  # return[0] = reward[0] + 0 = 1.0 (no bootstrap of step1/last value across the reset)
  assert torch.allclose(out[0], torch.tensor([[1.0]]), atol=1e-6)


def test_soft_delta_one_fully_cuts_bootstrap():
  # delta=1 at the last step => cont=0 => return = reward only (forfeits all future value).
  rewards = torch.tensor([[[2.0]]])
  out = _run(CatPPO.compute_returns, rewards, torch.zeros(1, 1, 1),
             dones=torch.zeros(1, 1, 1), soft=torch.ones(1, 1, 1), last_value=torch.tensor([[100.0]]))
  assert torch.allclose(out[0], torch.tensor([[2.0]]), atol=1e-6)


def test_multistep_lambda_recursion_handcomputed():
  rewards = torch.tensor([[[1.0]], [[2.0]]])
  values = torch.tensor([[[0.5]], [[0.3]]])
  out = _run(CatPPO.compute_returns, rewards, values,
             dones=torch.zeros(2, 1, 1), soft=torch.zeros(2, 1, 1),
             last_value=torch.tensor([[0.7]]))
  # step1: delta=2 + 0.99*0.7 - 0.3 = 2.393; return1 = 2.393 + 0.3 = 2.693
  # step0: delta=1 + 0.99*0.3 - 0.5 = 0.797; adv0 = 0.797 + (0.99*0.95)*2.393 = 0.797 + 2.250617 = 3.047617
  #        return0 = 3.047617 + 0.5 = 3.547617
  assert torch.allclose(out, torch.tensor([[[3.547617]], [[2.693]]]), atol=1e-4)


# ----------------------------------------------------------------------------- incentive-to-violate guard (iv)

def _one_step_incentive(stored_reward_d, stored_reward_0, delta, v_next):
  """return(δ) − return(0) for a single non-terminal step (current value cancels)."""
  ret_d = _run(CatPPO.compute_returns, torch.tensor([[[stored_reward_d]]]), torch.zeros(1, 1, 1),
               torch.zeros(1, 1, 1), torch.tensor([[[delta]]]), torch.tensor([[v_next]]))
  ret_0 = _run(CatPPO.compute_returns, torch.tensor([[[stored_reward_0]]]), torch.zeros(1, 1, 1),
               torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.tensor([[v_next]]))
  return (ret_d - ret_0).item()


def test_incentive_to_violate_nonpositive_scale_positives():
  # Across the grid, scale-positives NEVER pays the policy to violate when V(s_{t+1}) >= 0.
  delta = 0.5
  for r_pos in (0.0, 2.0, 60.0):
    for r_neg in (0.0, -8.0, -10.0):
      for v_next in (0.0, 2.0, 60.0):
        r_total = r_pos + r_neg
        inc = _one_step_incentive(r_total - delta * r_pos, r_total, delta, v_next)
        assert inc <= 1e-6, (r_pos, r_neg, v_next, inc)


def test_full_reward_has_penalty_evasion_exploit_that_scale_positives_fixes():
  # Cold critic (V=0), penalty-heavy step: r_pos=2, r_neg=-10 -> r_total=-8; delta=0.5.
  delta, r_pos, r_total, v_next = 0.5, 2.0, -8.0, 0.0
  full = _one_step_incentive(r_total * (1.0 - delta), r_total, delta, v_next)   # full-reward (1-δ)
  scaled = _one_step_incentive(r_total - delta * r_pos, r_total, delta, v_next)  # scale-positives
  assert full > 1e-6      # full-reward PAYS to violate (+4): the overturned-design exploit
  assert scaled <= 1e-6   # scale-positives does not (-1)


def test_incentive_residual_only_when_value_negative_and_uncoupled_from_penalty():
  # The only residual positive incentive needs V(s_{t+1}) < 0 AND small r_pos -- and it does NOT
  # grow with the -10 penalty (penalty is no longer in the bracket). Contrast the two r_neg rows.
  delta, v_next = 0.5, -1.0
  inc_small_pen = _one_step_incentive((0.0 - 8.0) - delta * 0.0, -8.0, delta, v_next)   # r_pos=0,r_neg=-8
  inc_big_pen = _one_step_incentive((0.0 - 10.0) - delta * 0.0, -10.0, delta, v_next)   # r_pos=0,r_neg=-10
  # residual is exactly -delta*(r_pos + gamma*v_next) = -0.5*(0 + 0.99*-1) = +0.495, INDEPENDENT of r_neg
  assert abs(inc_small_pen - inc_big_pen) < 1e-6
  assert abs(inc_small_pen - 0.495) < 1e-3


# ----------------------------------------------------------------------------- process_env_step plumbing

def test_process_env_step_scales_reward_and_carries_delta(monkeypatch):
  recorded = {}

  def spy(self, obs, rewards, dones, extras):
    recorded["rewards"] = rewards
    recorded["soft_dones"] = getattr(self.transition, "soft_dones", None)

  monkeypatch.setattr(PPO, "process_env_step", spy)
  obj = object.__new__(CatPPO)                 # bypass heavy PPO.__init__; super() still resolves
  obj.transition = RolloutStorage.Transition()
  rewards = torch.tensor([2.0, -8.0])
  delta, r_pos = torch.tensor([0.5, 0.5]), torch.tensor([12.0, 2.0])
  CatPPO.process_env_step(obj, None, rewards, torch.zeros(2),
                          {"cat_delta": delta, "cat_r_pos": r_pos})
  assert torch.allclose(recorded["rewards"], rewards - delta * r_pos)
  assert torch.allclose(recorded["soft_dones"], delta)


def test_first_step_without_delta_raises():
  # env/alg mismatch guard: CatPPO selected but the hook never fed δ -> loud crash, not a silent
  # PPO-vs-PPO baseline.
  import pytest
  obj = object.__new__(CatPPO)
  obj.transition = RolloutStorage.Transition()
  with pytest.raises(RuntimeError, match="cat_delta"):
    CatPPO.process_env_step(obj, None, torch.tensor([1.0, 2.0]), torch.zeros(2), extras={})


def test_noop_without_delta_after_first_cat_step(monkeypatch):
  recorded = {}

  def spy(self, obs, rewards, dones, extras):
    recorded["rewards"] = rewards

  monkeypatch.setattr(PPO, "process_env_step", spy)
  obj = object.__new__(CatPPO)
  obj.transition = RolloutStorage.Transition()
  obj.gamma = 0.99
  # first step carries δ (latch passes), then a later δ-absent step degrades gracefully (no raise)
  CatPPO.process_env_step(obj, None, torch.tensor([1.0, 2.0]), torch.zeros(2),
                          {"cat_delta": torch.zeros(2), "cat_r_pos": torch.zeros(2)})
  r = torch.tensor([3.0, -4.0])
  CatPPO.process_env_step(obj, None, r, torch.zeros(2), extras={})
  assert torch.allclose(recorded["rewards"], r)  # unscaled fallback


def test_success_terminal_keeps_discount_no_bootstrap():
  # Terminal step (hard_done=1, no timeout): target = stored (already (1-δ)-discounted) reward, NO
  # bootstrap. Documents the user-confirmed convention: a violating success is worth (1-δ)*reward
  # (e.g. 50, not 100) -- NOT the panel's undiscounted r_total. last_value is ignored (bootstrap cut).
  scaled = torch.tensor([[[50.0]]])  # = r_total(100) - δ(0.5)*r_pos(100)
  out = _run(CatPPO.compute_returns, scaled, torch.zeros(1, 1, 1),
             dones=torch.ones(1, 1, 1), soft=torch.full((1, 1, 1), 0.5),
             last_value=torch.tensor([[999.0]]))
  assert torch.allclose(out[0], torch.tensor([[50.0]]), atol=1e-6)


def test_timeout_bootstrap_is_delta_discounted(monkeypatch):
  # At a timeout, the injected time-limit bootstrap is (1-δ)·γ·V_t (NOT rsl_rl's full γ·V_t), and the
  # time_outs key is removed so super() does not double-inject.
  recorded = {}

  def spy(self, obs, rewards, dones, extras):
    recorded["rewards"] = rewards
    recorded["extras"] = extras

  monkeypatch.setattr(PPO, "process_env_step", spy)
  obj = object.__new__(CatPPO)
  obj.transition = RolloutStorage.Transition()
  obj.transition.values = torch.tensor([[10.0], [20.0]])  # V(s_t)
  obj.gamma = 0.99
  rewards = torch.tensor([2.0, -8.0])
  delta = torch.tensor([0.5, 0.0])
  r_pos = torch.tensor([4.0, 2.0])
  CatPPO.process_env_step(obj, None, rewards, torch.ones(2),
                          {"cat_delta": delta, "cat_r_pos": r_pos, "time_outs": torch.tensor([1.0, 1.0])})
  # scaled = [2-0.5*4, -8-0]=[0,-8]; +(1-δ)γV_t = [0.5*0.99*10, 1.0*0.99*20]=[4.95,19.8] -> [4.95,11.8]
  assert torch.allclose(recorded["rewards"], torch.tensor([4.95, 11.8]), atol=1e-4)
  assert "time_outs" not in recorded["extras"]
