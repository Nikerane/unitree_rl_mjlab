"""rsl_rl PPO adapter for faithful soft γ(1−δ) CaT (C2).

A thin subclass (the CaT authors' own pattern for encapsulated-buffer libraries — never edit the
installed package; inject via the ``algorithm.class_name`` config string, resolved by
``OnPolicyRunner`` at ``on_policy_runner.py:39``). Overrides three seams:

  • ``construct_algorithm`` — inject the float-dones :class:`CatRolloutStorage` (C1) so δ survives.
  • ``process_env_step``    — apply the SCALE-POSITIVES reward discount ``r_total − δ·r_pos`` (Decision 1;
    NO clip) and carry δ into the storage's soft-done buffer.
  • ``compute_returns``     — DUAL-MASK GAE bootstrap ``γ·(1−δ)·(1−true_done)`` (Decisions 3–4).

See docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md §4.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv

from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY
from src.tasks.hammer.rl.cat_storage import CatRolloutStorage


class CatPPO(PPO):
  """PPO with the faithful soft-CaT reward discount + dual-mask bootstrap."""

  # Keys the CaT env hook (CatSoftHook) stashes into ``extras`` each step (shared via cat.keys).
  DELTA_KEY = CAT_DELTA_KEY   # δ ∈ [0, max_p], shape (B,)
  R_POS_KEY = CAT_R_POS_KEY   # dt-scaled sum of the positive reward terms, shape (B,)

  @staticmethod
  def _cat_scale_reward(rewards: torch.Tensor, delta: torch.Tensor, r_pos: torch.Tensor) -> torch.Tensor:
    """Scale-positives-only discount: ``r_total − δ·r_pos`` ( == r_pos·(1−δ) + r_neg ). NO clip.

    Applies the CaT survival weight (1−δ) to the positive task return only; the negative regularizers
    in ``r_neg`` ride through unscaled (Decision 1 — avoids the penalty-evasion exploit of full-reward
    ``r_total·(1−δ)``).
    """
    d = delta.reshape(-1)
    rp = r_pos.reshape(-1)
    return rewards - (d * rp).reshape(rewards.shape)

  def process_env_step(
    self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
  ) -> None:
    """Scale-positives discount + a soft-CaT-consistent timeout bootstrap; carry δ; then defer to
    stock PPO bookkeeping."""
    delta = extras.get(self.DELTA_KEY)
    if not getattr(self, "_cat_checked", False):
      # Fail loudly on the env/alg mismatch: CatPPO selected but the CatSoftHook never fed δ (would
      # otherwise be a silent, uninstrumented PPO-vs-PPO baseline).
      self._cat_checked = True
      if delta is None:
        raise RuntimeError(
          "CatPPO is selected (algorithm.class_name) but extras['cat_delta'] is absent on the first "
          "step -- the CatSoftHook env hook is not wired. Set cat_soft=True on BOTH env_cfg and "
          "rl_cfg (FAITHFUL_SOFT_CAT_IMPL_PLAN.md), or use stock PPO."
        )
    if delta is None:
      super().process_env_step(obs, rewards, dones, extras)
      return

    d = delta.reshape(-1)
    # Scale-positives discount (Decision 1), applied on EVERY step incl. terminals (convention 2,
    # faithful to cat_env.py reward*(1-δ)). At a true terminal the dual mask cuts the bootstrap, so a
    # violating success is worth (1-δ)*reward -- the intended safety incentive (user-confirmed).
    rewards = self._cat_scale_reward(rewards, delta, extras[self.R_POS_KEY])
    self.transition.soft_dones = d

    # Soft-CaT-consistent time-limit bootstrap. rsl_rl injects the FULL gamma*V_t at a timeout
    # (ppo.py:151-155); under soft-CaT the future is reached only w.p. (1-δ), so inject
    # (1-δ)*gamma*V_t and suppress rsl_rl's full-weight injection. A timeout is a non-terminal cutoff,
    # so the step is handled like a continuing step: reward*(1-δ) + (1-δ)*gamma*V_t.
    time_outs = extras.get("time_outs")
    if time_outs is not None:
      v_t = self.transition.values.reshape(-1)  # V(s_t), set in act() before the env step
      to = time_outs.to(rewards.device).reshape(-1).to(rewards.dtype)
      rewards = rewards + (1.0 - d) * self.gamma * v_t * to
      extras = {k: v for k, v in extras.items() if k != "time_outs"}  # prevent super double-inject

    super().process_env_step(obs, rewards, dones, extras)

  def compute_returns(self, obs: TensorDict) -> None:
    """GAE with the dual continuation mask ``(1−δ)·(1−true_done)``.

    Mirrors stock ``PPO.compute_returns`` (ppo.py:163-185) except the single ``1−dones`` mask is
    replaced by ``(1−soft_dones)·(1−hard_dones)``: the soft δ discounts the bootstrap (CaT), the hard
    done cuts it at a real episode boundary (no bootstrap across a reset).
    """
    st = self.storage
    last_values = self.critic(obs).detach()
    advantage = 0
    for step in reversed(range(st.num_transitions_per_env)):
      next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
      soft_cont = 1.0 - st.soft_dones[step]            # (1 − δ)  — the CaT soft termination
      hard_cont = 1.0 - st.dones[step].float()         # (1 − true_done) — real episode boundary
      cont = soft_cont * hard_cont                     # dual mask
      delta_gae = st.rewards[step] + cont * self.gamma * next_values - st.values[step]
      advantage = delta_gae + cont * self.gamma * self.lam * advantage
      st.returns[step] = advantage + st.values[step]
    st.advantages = st.returns - st.values
    if not self.normalize_advantage_per_mini_batch:
      st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "CatPPO":
    """Build via the stock factory, then swap in the float-dones storage so δ is not truncated."""
    alg = PPO.construct_algorithm(obs, env, cfg, device)  # resolves class_name -> CatPPO instance
    alg.storage = CatRolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
    return alg  # type: ignore[return-value]
