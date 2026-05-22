"""Reward validation script for the Z1 hammer-nail task.

Runs scripted state sequences (including direct sim-state writes) and asserts
each reward term produces the expected value. Catches reward-correctness bugs
without requiring training.

Methodology: docs/research/reward-design/REWARD_VALIDATION_METHODOLOGY.md

Phases:
    A. Reset & hold              -> nail_depth_delta == 0
    B. Move action (no contact)  -> nail_depth_delta == 0
    C. Force depth -> 0.010 m    -> nail_depth_delta ~= +5.0  (0.010 * 500)
    D. Hold at 0.010 m           -> nail_depth_delta == 0
    E. Force depth -> 0.020 m    -> nail_depth_delta ~= +5.0  (new delta only)
    F. Reset + depth -> 0.005 m  -> nail_depth_delta ~= +2.5  (proves reset())
    G. Bounce-back 0.020 -> 0.015 -> nail_depth_delta == 0    (proves clamp_min)

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


NAIL_DEPTH_DELTA_WEIGHT = 500.0
COMPLETION_WEIGHT = 100.0
TOL_DELTA = 0.05   # tolerance for nail_depth_delta assertions (covers tiny physics drift)
TOL_ZERO = 0.01    # tolerance for "should be zero" assertions


def reward_dict(env: ManagerBasedRlEnv) -> dict[str, float]:
  """Return {term_name: float} of per-term reward for env 0."""
  rm = env.reward_manager
  return {
    name: rm._step_reward[0, idx].item() for idx, name in enumerate(rm.active_terms)
  }


def assert_close(actual: float, expected: float, msg: str, tol: float = TOL_DELTA) -> None:
  if abs(actual - expected) > tol:
    print(f"\n[FAIL] {msg}")
    print(f"       expected: {expected:.4f} (± {tol})")
    print(f"       actual:   {actual:.4f}")
    sys.exit(1)


def assert_zero(actual: float, msg: str, tol: float = TOL_ZERO) -> None:
  assert_close(actual, 0.0, msg, tol)


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

  summary: list[tuple[str, dict[str, float]]] = []
  print(f"Active reward terms: {env.reward_manager.active_terms}\n")

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
  assert_close(r["nail_depth_delta"], 0.010 * NAIL_DEPTH_DELTA_WEIGHT,
               "C: nail_depth_delta should fire 0.010 * 500 = 5.0")
  summary.append(("C. Force depth 0.010", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f} (expected ~5.0)")

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
  assert_close(r["nail_depth_delta"], 0.010 * NAIL_DEPTH_DELTA_WEIGHT,
               "E: nail_depth_delta should fire the NEW delta (0.020 - 0.010) * 500 = 5.0")
  summary.append(("E. Force depth 0.020", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f} (expected ~5.0)")

  # --- Phase F: Reset env, force depth 0.005 (tests reset() of _max_depth) ---
  print("\n--- Phase F: Reset env, force depth -> 0.005 m (tests reset()) ---")
  env.reset()
  force_nail_depth(env, 0.005)
  r = recompute_rewards(env)
  assert_close(r["nail_depth_delta"], 0.005 * NAIL_DEPTH_DELTA_WEIGHT,
               "F: nail_depth_delta should fire 0.005 * 500 = 2.5 (proves reset() zeroed _max_depth)")
  summary.append(("F. Reset + depth 0.005", r))
  print(f"  PASS  nail_depth_delta={r['nail_depth_delta']:.4f} (expected ~2.5)")

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
  # H2: Above threshold -> completion = 1.0 * 100 = 100
  force_nail_depth(env, NAIL_SUCCESS_THRESHOLD + 0.001)
  r = recompute_rewards(env)
  assert_close(r["completion"], 1.0 * COMPLETION_WEIGHT,
               "H2: completion should fire 1.0 * 100 = 100 above success threshold",
               tol=0.5)
  summary.append(("H. Completion bonus", r))
  print(f"  H2 PASS  completion={r['completion']:.4f} (expected 100.0)")

  # --- Summary table ---
  print("\n" + "=" * 110)
  print(" ALL PHASES PASSED")
  print("=" * 110)
  header = f"{'Phase':<26} {'depth_delta':>12} {'completion':>12} {'approach':>10} {'nail_driven':>12} {'action_rate':>12} {'joint_lim':>10}"
  print(header)
  print("-" * 120)
  for phase, r in summary:
    print(
      f"{phase:<26} "
      f"{r.get('nail_depth_delta', 0):>12.4f} "
      f"{r.get('completion', 0):>12.4f} "
      f"{r.get('approach', 0):>10.4f} "
      f"{r.get('nail_driven', 0):>12.4f} "
      f"{r.get('action_rate', 0):>12.4f} "
      f"{r.get('joint_pos_limits', 0):>10.4f}"
    )
  print()


if __name__ == "__main__":
  main()
