"""C3/C4 end-to-end verification of the faithful soft-CaT arm on the REAL env (CPU, no GPU).

Confirms the C3 contract that unit tests (stub env) cannot: that δ + r_pos actually flow out of the
real mjlab env.step through the rsl_rl wrapper, that a soft violation never resets the episode
(Decision 5), and that the dt-scaled r_pos identity holds against the real reward_buf. Run this before
any GPU smoke train.

Usage:
    python scripts/verify_cat_soft.py --num-envs 8 --nsteps 40 --device cpu
"""

from __future__ import annotations

import argparse

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import src.tasks.hammer.config.z1  # noqa: F401  (registers the tasks)
from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY

TASK = "Unitree-Z1-Hammer-CaT-Soft"
BASE = "Unitree-Z1-Hammer"
NEG_TERMS = ("action_rate", "joint_pos_limits")
MAX_P = 0.5


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-envs", type=int, default=8)
  ap.add_argument("--nsteps", type=int, default=40)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--limit", type=float, default=0.5,
                  help="Constraint limit for the check. Lowered from the real 3.1415 so the OPEN-LOOP "
                       "downward motion (self-limits ~2.4 rad/s; only the trained closed-loop policy "
                       "exceeds 3.1415) triggers δ>0 and exercises the δ path.")
  args = ap.parse_args()

  # --- (1) cfg-level: Decision 5 (no constraint-induced termination) + the hook is wired ---
  cat_cfg = load_env_cfg(TASK)
  base_cfg = load_env_cfg(BASE)
  assert set(cat_cfg.terminations.keys()) == set(base_cfg.terminations.keys()), (
    "cat_soft added a termination term -- a soft violation could end the episode (violates Decision 5)."
  )
  assert "cat_soft" in cat_cfg.metrics, "cat_soft hook not wired into cfg.metrics"
  assert load_rl_cfg(TASK).algorithm.class_name.endswith(":CatPPO"), "rl_cfg does not select CatPPO"
  print(f"[cfg] terminations match baseline ({sorted(cat_cfg.terminations)}); hook wired; CatPPO selected")

  # --- (2) build the real env on CPU + step it with a full-downward (fast) action ---
  cat_cfg.scene.num_envs = args.num_envs
  # Lower the constraint limit so the open-loop motion violates it (see --limit help). The real arm
  # keeps Z1_JOINT_VEL_LIMIT=3.1415; this only exercises the δ-computation path in the real env.
  cat_cfg.metrics["cat_soft"].params["limit"] = args.limit
  print(f"[cfg] constraint limit lowered to {args.limit} rad/s for the open-loop δ-path check")
  env = ManagerBasedRlEnv(cfg=cat_cfg, device=args.device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=1.0)
  u = env.unwrapped
  N, dev, dt = u.num_envs, u.device, u.step_dt
  neg_idx = [u.reward_manager.active_terms.index(n) for n in NEG_TERMS if n in u.reward_manager.active_terms]

  obs, _ = env.reset()
  act = torch.zeros(N, env.num_actions, device=dev)
  act[:, 2] = -1.0  # drive the hammer straight down -> fast arm motion -> over the velocity limit

  ep_len_prev = u.episode_length_buf.clone()
  max_delta = torch.zeros(N, device=dev)
  soft_violation_no_reset = True
  rpos_max_err = 0.0

  for t in range(args.nsteps):
    obs, rewards, dones, extras = env.step(act)
    # contract: keys present, float, shape (N,), δ ∈ [0, max_p]
    assert CAT_DELTA_KEY in extras and CAT_R_POS_KEY in extras, f"missing CaT extras at step {t}"
    delta, r_pos = extras[CAT_DELTA_KEY], extras[CAT_R_POS_KEY]
    assert delta.shape == (N,) and r_pos.shape == (N,), (delta.shape, r_pos.shape)
    assert delta.is_floating_point() and r_pos.is_floating_point()
    assert (delta >= -1e-6).all() and (delta <= MAX_P + 1e-6).all(), (delta.min().item(), delta.max().item())
    max_delta = torch.maximum(max_delta, delta)

    # r_pos identity: r_pos == reward_buf - (negative terms, dt-scaled). reward_buf is the env reward
    # (the timeout bootstrap is learner-side, NOT in the env reward), so the identity holds here.
    r_neg = u.reward_manager._step_reward[:, neg_idx].sum(dim=1) * dt if neg_idx else torch.zeros(N, device=dev)
    rpos_max_err = max(rpos_max_err, (r_pos - (rewards - r_neg)).abs().max().item())

    # Decision 5: a soft violation (δ>0) must NOT itself reset the episode. The env only resets on a
    # real termination term; episode_length increments otherwise. Flag any env that reset while δ>0
    # WITHOUT a real terminated/timeout flag (would mean the constraint reset it).
    real_done = u.reset_terminated | u.reset_time_outs
    reset_now = u.episode_length_buf < ep_len_prev  # counter went backwards => was reset
    if bool(((delta > 0) & reset_now & ~real_done).any()):
      soft_violation_no_reset = False
    ep_len_prev = u.episode_length_buf.clone()

  print(f"[step] {args.nsteps} steps, N={N}, dt={dt:.4f}")
  print(f"[delta] max over run = {max_delta.max().item():.3f} (max_p={MAX_P}); fired = {bool(max_delta.max() > 0)}")
  print(f"[r_pos] max identity error vs reward_buf - r_neg*dt = {rpos_max_err:.2e}")
  print(f"[reset] no soft-violation-induced reset (Decision 5) = {soft_violation_no_reset}")

  assert max_delta.max() > 0, "δ never fired -- the forced over-limit action did not trigger the constraint"
  assert rpos_max_err < 1e-4, f"r_pos identity broken (err={rpos_max_err:.2e}) -- dt-scale or _step_reward drift"
  assert soft_violation_no_reset, "a soft violation reset an episode -- violates Decision 5"
  print("\nVERIFY_CAT_SOFT: ALL CHECKS PASSED")


if __name__ == "__main__":
  main()
