"""Reward validation script for the Z1 hammer-nail task.

Runs scripted state sequences (including direct sim-state writes) and asserts
each reward term produces the expected value. Catches reward-correctness bugs
without requiring training.

Methodology: docs/research/reward-design/REWARD_VALIDATION_METHODOLOGY.md

Phases (expected values scale with the live env config weights, not hardcoded):
    A. Reset & hold               -> nail_depth_delta == 0
    B. Move action (no contact)   -> nail_depth_delta == 0
    C. Force depth -> 0.010 m     -> nail_depth_delta ~= 0.010 * W_delta
    D. Hold at 0.010 m            -> nail_depth_delta == 0
    E. Force depth -> 0.020 m     -> nail_depth_delta ~= 0.010 * W_delta  (new delta only)
    F. Reset + depth -> 0.005 m   -> nail_depth_delta ~= 0.005 * W_delta  (proves reset())
    G. Bounce-back 0.020 -> 0.015 -> nail_depth_delta == 0                (proves clamp_min)
    H. Completion bonus           -> completion == W_completion above threshold
    I. Strike (drive down)        -> impact_progress > 0 on fresh productive contact, 0 else

Run:
    /home/nikhil/miniconda3/envs/unitree_mjlab/bin/python \\
        docs/research/reward-design/validate_rewards.py
"""

from __future__ import annotations

import sys

import torch

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD


TOL_FRAC = 0.01    # fractional tolerance for nonzero assertions (1% of expected)
TOL_ABS_MIN = 0.05 # minimum absolute tolerance floor (covers small-value noise)
TOL_ZERO = 0.01    # absolute tolerance for "should be zero" assertions


def reward_dict(env: ManagerBasedRlEnv) -> dict[str, float]:
  """Return {term_name: float} of per-term reward for env 0."""
  rm = env.reward_manager
  return {
    name: rm._step_reward[0, idx].item() for idx, name in enumerate(rm.active_terms)
  }


def get_weights(env: ManagerBasedRlEnv) -> dict[str, float]:
  """Return {term_name: weight} read from the live reward manager config."""
  rm = env.reward_manager
  return {name: float(rm.get_term_cfg(name).weight) for name in rm.active_terms}


def assert_close(actual: float, expected: float, msg: str, tol: float | None = None) -> None:
  if tol is None:
    tol = max(TOL_ABS_MIN, abs(expected) * TOL_FRAC)
  if abs(actual - expected) > tol:
    print(f"\n[FAIL] {msg}")
    print(f"       expected: {expected:.4f} (± {tol:.4f})")
    print(f"       actual:   {actual:.4f}")
    sys.exit(1)


def assert_zero(actual: float, msg: str, tol: float = TOL_ZERO) -> None:
  if abs(actual) > tol:
    print(f"\n[FAIL] {msg}")
    print(f"       expected: 0.0000 (± {tol})")
    print(f"       actual:   {actual:.4f}")
    sys.exit(1)


def force_nail_depth(env: ManagerBasedRlEnv, depth: float) -> None:
  """Directly set nail_slide qpos (position only, velocity untouched).

  NOTE: The nail shaft interpenetrates the block by design (resting state).
  Running env.step() after this write will let the contact solver push the
  nail back toward qpos=0 because the block normal force exceeds the joint's
  frictionloss (0.3 N). Use recompute_rewards() — not env.step() — to evaluate
  the reward at the written state.
  """
  nail = env.scene["nail_block"]
  new_pos = torch.tensor([[depth]], dtype=torch.float32, device=env.device)
  nail.write_joint_position_to_sim(new_pos, joint_ids=[0])


def recompute_rewards(env: ManagerBasedRlEnv) -> dict[str, float]:
  """Recompute reward terms WITHOUT running physics.

  Used for state-write validation phases: we want the reward term to see the
  nail position we just wrote, before the contact solver undoes the write.
  """
  env.reward_manager.compute(dt=env.step_dt)
  return reward_dict(env)


def main() -> None:
  cfg = z1_hammer_env_cfg(play=True)
  cfg.scene.num_envs = 1
  device = "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  env.reset()

  action_dim = env.action_manager.total_action_dim
  zero_action = torch.zeros(1, action_dim, device=device)
  down_action = torch.zeros(1, action_dim, device=device)
  down_action[:, 2] = -1.0

  weights = get_weights(env)
  W_DELTA = weights["nail_depth_delta"]
  W_COMPLETION = weights["completion"]

  from src.tasks.hammer.mdp.rewards import NailDepthDeltaTerm
  SETTLE = NailDepthDeltaTerm._SETTLE_OFFSET

  summary: list[tuple[str, dict[str, float]]] = []
  print(f"Active reward terms: {env.reward_manager.active_terms}")
  print(f"Live weights read from env: {weights}\n")

  # --- Phase A: Reset & hold (5 zero-action steps) ---
  print("--- Phase A: Reset & hold ---")
  for step in range(5):
    env.step(zero_action)
    r = reward_dict(env)
    assert_zero(r["nail_depth_delta"], f"A.step{step}: nail_depth_delta should be 0")
  summary.append(("A. Reset & hold", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f}  approach={r['approach']:.4f}")

  # --- Phase B: Move action (no nail contact expected) ---
  print("\n--- Phase B: Move action (downward, no contact) ---")
  for step in range(5):
    env.step(down_action)
    r = reward_dict(env)
    assert_zero(r["nail_depth_delta"], f"B.step{step}: nail_depth_delta should be 0 (no contact)")
  summary.append(("B. Move action", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f}  approach={r['approach']:.4f}")

  # Phases C-G use direct state writes + recompute_rewards() (no env.step) to bypass
  # the contact solver that would undo our writes.

  # --- Phase C: Force nail depth to 0.010 m ---
  print("\n--- Phase C: Force nail depth -> 0.010 m ---")
  force_nail_depth(env, 0.010)
  r = recompute_rewards(env)
  expected_C = (0.010 - SETTLE) * W_DELTA
  assert_close(r["nail_depth_delta"], expected_C,
               f"C: nail_depth_delta should fire (0.010 - {SETTLE}) * {W_DELTA} = {expected_C}")
  summary.append(("C. Force depth 0.010", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f} (expected ~{expected_C:.4f})")

  # --- Phase D: Hold at 0.010 (no new progress) ---
  print("\n--- Phase D: Hold at 0.010 m ---")
  force_nail_depth(env, 0.010)
  r = recompute_rewards(env)
  assert_zero(r["nail_depth_delta"], "D: nail_depth_delta should be 0 (no new progress)")
  summary.append(("D. Hold at 0.010", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f}")

  # --- Phase E: Force nail depth to 0.020 m (delta should fire on the NEW progress) ---
  print("\n--- Phase E: Force nail depth -> 0.020 m ---")
  force_nail_depth(env, 0.020)
  r = recompute_rewards(env)
  expected_E = 0.010 * W_DELTA
  assert_close(r["nail_depth_delta"], expected_E,
               f"E: nail_depth_delta should fire (0.020 - 0.010) * {W_DELTA} = {expected_E}")
  summary.append(("E. Force depth 0.020", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f} (expected ~{expected_E:.4f})")

  # --- Phase F: Reset env, force depth 0.005 (tests reset() of _max_depth) ---
  print("\n--- Phase F: Reset env, force depth -> 0.005 m (tests reset()) ---")
  env.reset()
  force_nail_depth(env, 0.005)
  r = recompute_rewards(env)
  expected_F = (0.005 - SETTLE) * W_DELTA
  assert_close(r["nail_depth_delta"], expected_F,
               f"F: nail_depth_delta should fire (0.005 - {SETTLE}) * {W_DELTA} = {expected_F} (proves reset())")
  summary.append(("F. Reset + depth 0.005", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f} (expected ~{expected_F:.4f})")

  # --- Phase G: Bounce-back 0.020 -> 0.015 (clamp_min check) ---
  print("\n--- Phase G: Bounce-back 0.020 -> 0.015 m (tests clamp_min) ---")
  force_nail_depth(env, 0.020)
  recompute_rewards(env)  # establishes _max_depth = 0.020
  force_nail_depth(env, 0.015)
  r = recompute_rewards(env)
  assert_zero(r["nail_depth_delta"], "G: nail_depth_delta should be 0 (negative delta clamped)")
  summary.append(("G. Bounce-back", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f}")

  # --- Phase H: Completion bonus (sparse task reward) ---
  print("\n--- Phase H: Completion bonus ---")
  env.reset()
  # H1: Below threshold -> completion = 0
  force_nail_depth(env, NAIL_SUCCESS_THRESHOLD - 0.001)
  r = recompute_rewards(env)
  assert_zero(r["completion"], "H1: completion should be 0 below success threshold")
  print(f"  H1 PASS  completion={r['completion']:.4f} (below threshold)")
  # H2: Above threshold -> completion = 1.0 * W_completion
  force_nail_depth(env, NAIL_SUCCESS_THRESHOLD + 0.001)
  r = recompute_rewards(env)
  assert_close(r["completion"], W_COMPLETION,
               f"H2: completion should fire 1.0 * {W_COMPLETION} = {W_COMPLETION} above threshold",
               tol=max(0.5, W_COMPLETION * TOL_FRAC))
  summary.append(("H. Completion bonus", r))
  print(f"  H2 PASS  completion={r['completion']:.4f} (expected {W_COMPLETION:.4f})")

  # --- Phase I: impact_progress end-to-end (real strike) ---
  # Unlike C-G, this needs real physics: first_contact + finite-diff velocity +
  # depth advance must all be genuine. Driving straight down yields free flight,
  # then a fresh productive contact (the strike), then continuous contact.
  print("\n--- Phase I: impact_progress end-to-end (real strike) ---")
  env.reset()
  contact_ever = False
  max_impact = 0.0
  strike_r: dict[str, float] | None = None
  sensor = env.scene["hammer_nail_contact"]
  for step in range(15):
    env.step(down_action)
    r = reward_dict(env)
    found = bool((sensor.data.found > 0).any())
    imp = r["impact_progress"]
    if imp < -TOL_ZERO:
      print(f"\n[FAIL] I.step{step}: impact_progress negative ({imp:.4f})")
      sys.exit(1)
    if imp > TOL_ZERO and not found:
      print(f"\n[FAIL] I.step{step}: impact_progress={imp:.4f} fired with NO contact (gate broken)")
      sys.exit(1)
    if not contact_ever and not found:
      assert_zero(imp, f"I.step{step}: impact_progress must be 0 in free flight")
    if imp > TOL_ZERO:
      strike_r = r
    contact_ever = contact_ever or found
    max_impact = max(max_impact, imp)
    if contact_ever and not found:
      break  # episode reset after the strike drove the nail home
  if max_impact <= TOL_ZERO:
    print("\n[FAIL] I: no productive strike produced impact_progress > 0")
    sys.exit(1)
  summary.append(("I. Strike (impact)", strike_r if strike_r is not None else r))
  print(f"  PASS  impact_progress peaked at {max_impact:.4f} on the strike, 0 in free flight")

  # --- Summary table ---
  print("\n" + "=" * 110)
  print(" ALL PHASES PASSED")
  print("=" * 110)
  header = f"{'Phase':<26} {'depth_delta':>12} {'impact':>9} {'completion':>12} {'approach':>10} {'nail_driven':>12} {'action_rate':>12} {'joint_lim':>10}"
  print(header)
  print("-" * 120)
  for phase, r in summary:
    print(
      f"{phase:<26} "
      f"{r.get('nail_depth_delta', 0):>12.4f} "
      f"{r.get('impact_progress', 0):>9.4f} "
      f"{r.get('completion', 0):>12.4f} "
      f"{r.get('approach', 0):>10.4f} "
      f"{r.get('nail_driven', 0):>12.4f} "
      f"{r.get('action_rate', 0):>12.4f} "
      f"{r.get('joint_pos_limits', 0):>10.4f}"
    )
  print()


if __name__ == "__main__":
  main()
