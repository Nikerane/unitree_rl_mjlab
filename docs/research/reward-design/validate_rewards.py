"""Reward validation script for the Z1 hammer-nail task.

Runs scripted state sequences (including direct sim-state writes) and asserts
each reward term produces the expected value. Catches reward-correctness bugs
without requiring training.

Methodology: docs/archive/REWARD_VALIDATION_METHODOLOGY.md

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
    J. Phase machinery (T1)       -> strike_phase ~0 after reset, monotone under
                                     descent, reaches the descent half, re-anchors on reset
    K. Imitation prior (T2)       -> r_imit anchored at reset, ante-impact latch, budget cap
    L. Overshoot clamp            -> nail_depth_delta clamps to GOAL past the soft-limit stop
    M. Impulse-CaT arm (C0)       -> Λ_j>0 on a strike, cat_delta ≡ 0 under log-only (imp_max_p=0),
                                     delivered-impulse reward fires, joint_impulse_excess = Λ_j − limit

Run:
    /home/nikhil/miniconda3/envs/unitree_mjlab/bin/python \\
        docs/research/reward-design/validate_rewards.py
"""

from __future__ import annotations

import sys

import torch

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD, NAIL_GOAL_DEPTH


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
  Running env.step() after this write can let the contact solver push the
  nail back toward qpos=0 (interpenetration-recovery forces can exceed the
  joint's frictionloss — 30 N since the 2026-06-10 recalibration; the original
  observation was made at 0.3 N). Use recompute_rewards() — not env.step() —
  to evaluate the reward at the written state.
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
  cfg = z1_hammer_env_cfg(play=True, imitation=True)
  cfg.scene.num_envs = 1
  device = "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  env.reset()

  action_dim = env.action_manager.total_action_dim
  zero_action = torch.zeros(1, action_dim, device=device)
  down_action = torch.zeros(1, action_dim, device=device)
  down_action[:, 2] = -1.0
  up_action = torch.zeros(1, action_dim, device=device)
  up_action[:, 2] = 1.0

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
  # Drive UP/away from the nail: since 2026-06-15 the reset pose places the
  # striking face directly above the nail (head_site is now ON the face), so a
  # DOWNWARD move contacts almost immediately — it can no longer be the
  # "free, non-contacting move" this phase needs. Up is a genuine no-contact
  # move; we assert the sensor stays clear so the premise is verified, not assumed.
  print("\n--- Phase B: Move action (upward, no contact) ---")
  b_sensor = env.scene["hammer_nail_contact"]
  for step in range(5):
    env.step(up_action)
    r = reward_dict(env)
    if bool((b_sensor.data.found > 0).any()):
      print(f"\n[FAIL] B.step{step}: hammer contacted the nail while moving UP (premise broken)")
      sys.exit(1)
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

  # --- Phase J: strike-reference phase machinery (T1) ---
  # The obs manager drives the shared SingleStrikeReference every step; we read
  # its latched phase directly. Asserts: ~0 after reset, monotone under a
  # sustained descent, reaches the descent half, re-anchors to ~0 on reset.
  print("\n--- Phase J: strike-reference phase (anchored, monotone, resets) ---")
  from src.tasks.hammer.mdp.references import get_strike_reference

  env.reset()
  ref = get_strike_reference(env)
  phi_prev = float(ref._phi[0])
  if phi_prev > 0.05:
    print(f"\n[FAIL] J: phi should be ~0 right after reset, got {phi_prev:.4f}")
    sys.exit(1)
  phis = [phi_prev]
  for step in range(12):
    env.step(down_action)
    if env.episode_length_buf[0].item() == 0:
      # Episode auto-reset (success termination): phase re-anchors to ~0 by
      # design, so the monotone check applies only within an episode.
      break
    phi = float(ref._phi[0])
    if phi < phi_prev - 1e-6:
      print(f"\n[FAIL] J.step{step}: phi decreased {phi_prev:.4f} -> {phi:.4f} (latch broken)")
      sys.exit(1)
    phi_prev = phi
    phis.append(phi)
  if max(phis) <= 0.5:
    if env.episode_length_buf[0].item() == 0:
      # Success arrived before phi crossed into the descent half — possible if
      # nail physics are ever retuned to terminate in < n_windup steps. The
      # anchor/latch/re-anchor properties were still exercised; warn, not fail.
      print("  WARN  success terminated the episode before phi crossed 0.5; "
            "descent-half coverage skipped")
    else:
      print(f"\n[FAIL] J: phi never reached the descent half (max={max(phis):.3f})")
      sys.exit(1)
  env.reset()
  phi_reset = float(get_strike_reference(env)._phi[0])
  if phi_reset > 0.05:
    print(f"\n[FAIL] J: phi did not re-anchor on reset ({phi_reset:.3f})")
    sys.exit(1)
  summary.append(("J. Phase machinery", {}))
  print(f"  PASS  phi rose monotonically to {max(phis):.3f}, re-anchored to {phi_reset:.3f} on reset")

  # --- Phase K: imitation prior r_imit (T2) ---
  # K1: right after reset the head == the reference anchor (phi=0 -> waypoint=head0),
  #     so the Gaussian is 1 and (pre-contact) r_imit == its weight.
  # K2: driving down, from the first contact onward the ante-impact latch zeroes it.
  # K3: budget rule -- cumulative weighted r_imit over the strike < 0.35 * completion.
  print("\n--- Phase K: imitation prior (anchored, ante-impact latch, budget) ---")
  env.reset()
  W_IMIT = get_weights(env)["r_imit"]
  r = recompute_rewards(env)
  assert_close(r["r_imit"], W_IMIT,
               f"K1: r_imit should be ~{W_IMIT} (Gaussian=1 at the reference anchor)",
               tol=max(0.01, W_IMIT * TOL_FRAC))
  print(f"  K1 PASS  r_imit={r['r_imit']:.4f} at the reference anchor (weight {W_IMIT})")

  env.reset()
  k_sensor = env.scene["hammer_nail_contact"]
  contacted = False
  budget = 0.0
  for step in range(40):
    env.step(down_action)
    r = reward_dict(env)
    budget += r["r_imit"]
    found = bool((k_sensor.data.found > 0).any())
    if found or contacted:
      contacted = True
      assert_zero(r["r_imit"],
                  f"K2.step{step}: r_imit must be 0 from first contact onward (ante-impact latch)")
    if env.episode_length_buf[0].item() == 0:
      break  # episode reset after the strike drove the nail home
  if not contacted:
    print("\n[FAIL] K2: no contact within 40 steps; cannot verify the ante-impact latch")
    sys.exit(1)
  print(f"  K2 PASS  r_imit latched to 0 from first contact; pre-contact budget={budget:.4f}")

  budget_cap = 0.35 * W_COMPLETION
  if budget >= budget_cap:
    print(f"\n[FAIL] K3: imitation budget {budget:.4f} >= 0.35*completion ({budget_cap:.1f})")
    sys.exit(1)
  print(f"  K3 PASS  imitation budget {budget:.4f} < 0.35*completion ({budget_cap:.1f})")
  summary.append(("K. Imitation prior", {}))

  # --- Phase L: Soft-limit overshoot is clamped at the reward source ---
  # A hard strike transiently drives nail_slide past its 0.032 m stop (~63 mm).
  # Every depth-reading reward term must see the CLAMPED physical depth, not the
  # elastic excursion (else a harder strike inflates nail_depth_delta and any
  # future impulse integral). Previously only the OBSERVATION was clamped, low side only.
  print("\n--- Phase L: soft-limit overshoot clamp ---")
  expected_delta = (NAIL_GOAL_DEPTH - SETTLE) * W_DELTA
  r = {}
  for overshoot in (0.063, 0.10):
    env.reset()
    force_nail_depth(env, overshoot)
    r = recompute_rewards(env)
    assert_close(
      r["nail_depth_delta"], expected_delta,
      f"L@{overshoot}m: nail_depth_delta must clamp to GOAL ({NAIL_GOAL_DEPTH} m), "
      f"not pay the {overshoot} m overshoot",
    )
  print(f"  L PASS  nail_depth_delta clamps to {expected_delta:.4f} at overshoot (GOAL={NAIL_GOAL_DEPTH} m)")
  summary.append(("L. Overshoot clamp", r))

  # --- Phase M: impulse-CaT arm (Λ_j accumulation, LOG-ONLY δ≡0, delivered-impulse reward) ---
  # Builds the SEPARATE -CaT-Impulse arm (cat_impulse=True) and certifies the C0 invariants:
  #   M1  Λ_j (substep accumulator) goes positive on a productive strike;
  #   M2  cat_delta is PUBLISHED every step (a dead hook must FAIL, not vacuously pass) and is
  #       identically 0 under imp_max_p=0 (log-only is a TRUE no-op — the enforcement gate);
  #   M3  the object-side delivered-impulse maximize reward fires on the strike;
  #   M4  joint_impulse_excess = Λ_j − limit checked at a step with NONZERO Λ (not a tautology).
  # auto_reset is DISABLED for this phase (mjlab-native): nail_driven still TERMINATES (the MDP is
  # unchanged, and the metrics-before-reset ordering on the terminal strike is exercised), but the
  # in-step auto-reset — which zeroed the accumulators before step() returned and silently discarded
  # the home-driving strike's Λ (2026-07 review finding #5) — does not run, so terminal-step values
  # stay readable. Full histogram + shipped-accumulator cross-check: derive_impulse_thresholds.py.
  print("\n--- Phase M: impulse-CaT arm (Λ_j, log-only δ≡0, delivered impulse) ---")
  from src.tasks.hammer.cat.constraints import joint_impulse_excess
  from src.tasks.hammer.mdp.impulse_bound import (
    _ENV_SUBSTEP_IMPULSE_ATTR,
    Z1_JOINT_IMPULSE_LIMIT,
  )

  imp_cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  imp_cfg.scene.num_envs = 1
  imp_cfg.auto_reset = False  # terminal-step state stays readable; termination itself still fires
  imp_env = ManagerBasedRlEnv(imp_cfg, device=device)
  imp_env.reset()
  imp_down = torch.zeros(1, imp_env.action_manager.total_action_dim, device=device)
  imp_down[:, 2] = -1.0
  acc = getattr(imp_env, _ENV_SUBSTEP_IMPULSE_ATTR)
  imp_rm = imp_env.reward_manager
  didx = imp_rm.active_terms.index("delivered_impulse")
  peak_lambda = torch.zeros(6, device=device)
  max_delta, max_dimp_reward = 0.0, 0.0
  m4_margin = m4_impulse = None
  terminated = False
  for step in range(20):
    imp_env.step(imp_down)
    imp_now = acc.impulse[0].clone()
    peak_lambda = torch.maximum(peak_lambda, imp_now)
    delta = imp_env.extras.get("cat_delta")
    if delta is None:
      print(f"\n[FAIL] M2: extras['cat_delta'] missing at step {step} — the CaT hook is not wired")
      sys.exit(1)
    max_delta = max(max_delta, float(delta.abs().max()))
    max_dimp_reward = max(max_dimp_reward, float(imp_rm._step_reward[0, didx]))
    if m4_margin is None and float(imp_now.max()) > TOL_ZERO:
      # Capture the raw-margin identity at a step where Λ is genuinely nonzero (anti-tautology).
      m4_margin = joint_impulse_excess(imp_env, limit=Z1_JOINT_IMPULSE_LIMIT).clone()
      m4_impulse = acc.impulse.clone()
    if bool(imp_env.reset_terminated.any()):
      terminated = True  # success fired; with auto_reset=False the terminal state is still live
      break
  if terminated:
    print("  (success termination fired on the strike step; terminal-step Λ read pre-reset)")
  if not (peak_lambda.max() > TOL_ZERO):
    print(f"\n[FAIL] M1: per-joint impulse Λ_j never went positive on the strike ({peak_lambda.tolist()})")
    sys.exit(1)
  print(f"  M1 PASS  peak Λ_j = {[f'{v:.3f}' for v in peak_lambda.tolist()]} N·m·s")
  assert_zero(max_delta, "M2: cat_delta must be identically 0 under log-only imp_max_p=0", tol=1e-6)
  print(f"  M2 PASS  cat_delta published every step and ≡ 0 (log-only no-op): max|δ| = {max_delta:.1e}")
  if not (max_dimp_reward > TOL_ZERO):
    print(f"\n[FAIL] M3: delivered_impulse reward never fired (peak {max_dimp_reward:.4f})")
    sys.exit(1)
  print(f"  M3 PASS  delivered_impulse reward peaked at {max_dimp_reward:.4f} (object-side maximize)")
  if m4_margin is None:
    print("\n[FAIL] M4: never observed a nonzero Λ step to check the raw-margin identity on")
    sys.exit(1)
  if m4_margin.shape != (1, 6) or not torch.allclose(m4_margin, m4_impulse - Z1_JOINT_IMPULSE_LIMIT):
    print(f"\n[FAIL] M4: joint_impulse_excess must equal Λ_j − limit, shape (1,6); got {m4_margin.shape}")
    sys.exit(1)
  print(f"  M4 PASS  joint_impulse_excess = Λ_j − {Z1_JOINT_IMPULSE_LIMIT} at nonzero Λ (raw signed margin)")
  summary.append(("M. Impulse arm", {}))

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
