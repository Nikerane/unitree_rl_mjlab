"""C0 quantity gate for the impulse-CaT arm (IMPULSE_CAT_IMPL_PLAN.md C0).

Drives the OPEN-LOOP single-strike reference against the live env and certifies that the measured
per-joint reaction impulse Λ_j is the REAL impact impulse — not weld/friction pollution — before any
enforcement (max_p>0) is enabled. Per-substep (500 Hz) instrumentation via a scene.update wrapper
captures the full impact even on the success step (reset zeroes the accumulators AFTER the decimation
loop). Many reference strikes → a per-joint Λ_j histogram.

Reports / asserts (fail-not-warn → exit 1):
  1. OFF-CONTACT + SHIPPED-CODE CROSS-CHECK: the SHIPPED SubstepImpulseAccumulator (the buffer
     training actually uses) is read every substep and must be exactly 0 before any contact (a
     leaking sensor gate shows up here as the ±frictionloss baseline), AND its per-window Λ and the
     shipped delivered-impulse Δ must agree with this script's independent sums — the gate certifies
     the wired code, not a parallel reimplementation. ASSERT.
  2. CONTAMINATION (the "weld pollution" gate): during contact, raw |qfrc| includes the friction/weld
     baseline. Compare raw-gated Λ_j vs pre-contact-baseline-subtracted Λ_j; report the contamination
     fraction. ASSERT the baseline-subtracted signal is a strictly positive impact signal.
  3. GROUND TRUTH: the object-side delivered axial impulse ∫F_axial dt (contact sensor = weld/friction
     IMMUNE by construction) is the independent cross-check that an impact impulse of this scale is
     real. Pinocchio impulseDynamics(r_coeff=0) leg is scaffolded behind an availability check.
  4. THRESHOLDS: per-joint J_limit = τ_rated,j × 2 (Harmonic-Drive Repeated-Peak) × Δt_impact, with the
     1e4-event fatigue-budget note. Binding-ness: Λ_j(p95) / J_limit_j.
  5. NORMALIZER FLOOR: section [5] prints p95(Λ_j) for reference only — imp_seed stays a small decay
     floor (1e-3); the hook self-seeds from the first over-limit sample.

Run: ~/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/derive_impulse_thresholds.py
"""

from __future__ import annotations

import sys

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# Per-joint rated peak torque [N·m] from z1_constants effort_limit (joint2 = heavy shoulder).
TAU_RATED = torch.tensor([30.0, 60.0, 30.0, 30.0, 30.0, 30.0])
REPEATED_PEAK = 2.0  # Harmonic-Drive Repeated-Peak ≈ 2× rated (< Momentary-Peak ≈ 4×)
APPROACH_HEIGHTS = [0.06, 0.10, 0.15]
REPEATS = 5
HOLD_STEPS = 6
WELD_TOL = 0.5  # max acceptable off-contact-baseline fraction of the contact-window Λ (gate)


def _pct(x: torch.Tensor, q: float) -> torch.Tensor:
  return torch.quantile(x, q, dim=0) if x.numel() else torch.zeros(x.shape[1:])


def main() -> None:
  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  contact = env.scene["hammer_nail_contact"]
  netf = env.scene["hammer_nail_impulse"]
  rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  arm = SceneEntityCfg("robot", joint_names=ARM)
  rcfg.resolve(env.scene)
  arm.resolve(env.scene)
  jid = arm.joint_ids
  axis = torch.tensor([0.0, 0.0, -1.0])
  dt = env.physics_dt

  def head() -> torch.Tensor:
    return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)

  ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  ncfg.resolve(env.scene)

  def nail_top() -> torch.Tensor:
    return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

  # --- per-substep instrumentation (records the strike before the success-reset) ---
  # Each record also snapshots the SHIPPED accumulators (the buffers training actually uses), so
  # the gate certifies the wired code, not just this script's parallel computation (2026-07 review
  # finding #3: the old gate's off-contact check was vacuous and never read the shipped metric).
  from src.tasks.hammer.mdp.impulse_bound import (
    _ENV_SUBSTEP_DELIVERED_ATTR,
    _ENV_SUBSTEP_IMPULSE_ATTR,
  )

  acc_shipped = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)
  dacc_shipped = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR)
  # (in_contact, qfrc_arm(6), f_axial, shipped_impulse(6), shipped_delivered)
  rec: list[tuple[bool, torch.Tensor, float, torch.Tensor, float]] = []
  # Hook AFTER the shipped accumulators run (metrics_manager.compute_substep follows scene.update
  # in the decimation loop), so every record's shipped snapshot includes its OWN substep — the
  # cross-check is exact by construction, with no alignment tolerance to hide an off-by-one behind.
  orig_substep = env.metrics_manager.compute_substep

  def patched() -> None:
    orig_substep()
    qfrc = robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone()  # (6,)
    in_c = bool((contact.data.found > 0).any())
    f = netf.data.force  # (1, N, 3) world
    f_ax = float((f * axis).sum(-1).sum(-1).clamp_min(0.0)[0])
    rec.append((in_c, qfrc, f_ax, acc_shipped.impulse[0].clone(), float(dacc_shipped.delivered[0])))

  env.metrics_manager.compute_substep = patched  # type: ignore[method-assign]

  # --- collect many reference strikes ---
  per_joint_raw: list[torch.Tensor] = []      # Λ_j raw-gated, per strike (script-side)
  per_joint_sub: list[torch.Tensor] = []      # Λ_j baseline-subtracted, per strike (script-side)
  delivered: list[float] = []                 # object-side ∫F_axial dt, per strike (script-side)
  durations: list[int] = []                   # contact-window substeps, per strike
  friction_baseline: list[torch.Tensor] = []  # off-contact |qfrc| per joint (the contaminant scale)
  shipped_leak = 0.0                          # SHIPPED acc.impulse seen before any contact (must be 0)
  ximp_err: list[float] = []                  # |shipped window Λ − script raw Λ| per strike (abs, worst joint)
  xdel_err: list[float] = []                  # |shipped delivered Δ − script deliv| per strike

  for h in APPROACH_HEIGHTS:
    for _ in range(REPEATS):
      env.reset()
      ref = SingleStrikeReference(1, env.device, approach_height=h)
      ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
      n = ref.playback_length()
      rec.clear()
      for k in range(1, n + HOLD_STEPS + 1):
        target = ref.playback_target(min(k, n))
        action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
        env.step(action)
        if int(env.episode_length_buf[0]) == 0:  # success auto-reset; strike captured in rec
          break

      # Pre-contact baseline = last off-contact qfrc before the (first) contact window.
      baseline = torch.zeros(6)
      raw = torch.zeros(6)
      sub = torch.zeros(6)
      dur = 0
      deliv = 0.0
      seen_contact = False
      shipped_win = torch.zeros(6)  # per-joint max of the SHIPPED acc.impulse over the window
      shipped_del_start = 0.0
      shipped_del_end = 0.0
      deliv_capped = 0.0  # script-side mirror of the shipped per-event accrual cap (first 25 substeps)
      for in_c, qfrc, f_ax, ship_imp, ship_del in rec:
        if not in_c:
          baseline = qfrc  # rolling pre-contact reference (frozen once contact opens)
          friction_baseline.append(qfrc.abs())
          if not seen_contact:
            # REAL leak check: before any contact, the SHIPPED accumulator must be exactly 0 —
            # if its sensor gate regresses, the off-contact friction shows up right here.
            shipped_leak = max(shipped_leak, float(ship_imp.abs().max()))
            shipped_del_start = ship_del
          if seen_contact:
            break  # stop at first release (isolate the impact, ignore re-contact)
        else:
          seen_contact = True
          raw += qfrc.abs() * dt
          sub += (qfrc - baseline).abs() * dt
          deliv += f_ax * dt
          if dur < 25:  # mirrors SubstepDeliveredImpulse's event_window_substeps default
            deliv_capped += f_ax * dt
          dur += 1
          shipped_win = torch.maximum(shipped_win, ship_imp)
          shipped_del_end = ship_del
      if seen_contact and dur > 0:
        per_joint_raw.append(raw)
        per_joint_sub.append(sub)
        delivered.append(deliv)
        durations.append(dur)
        # Cross-check the SHIPPED buffers against this script's independent sums — records snapshot
        # AFTER metrics.compute_substep, so agreement must be exact (float tolerance only).
        ximp_err.append(max(0.0, float((shipped_win - raw).abs().max()) - 1e-4))
        xdel_err.append(max(0.0, abs((shipped_del_end - shipped_del_start) - deliv_capped) - 1e-4))

  env.metrics_manager.compute_substep = orig_substep  # type: ignore[method-assign]

  if not per_joint_raw:
    print("[FAIL] no contact windows captured — reference strike never contacted the nail.")
    sys.exit(1)

  RAW = torch.stack(per_joint_raw)   # (S, 6)
  SUB = torch.stack(per_joint_sub)   # (S, 6)
  DEL = torch.tensor(delivered)      # (S,)
  DUR = torch.tensor(durations, dtype=torch.float32)  # (S,)
  FB = torch.stack(friction_baseline) if friction_baseline else torch.zeros(1, 6)
  S = RAW.shape[0]
  dt_impact = float(DUR.mean()) * dt

  print(f"\n=== C0 IMPULSE QUANTITY GATE — {S} reference strikes ===")
  print(f"contact-window duration: mean {DUR.mean():.1f} substeps = {dt_impact * 1000:.1f} ms "
        f"(p95 {_pct(DUR.unsqueeze(1), 0.95)[0]:.0f})")

  print("\n[1] OFF-CONTACT (weld/friction-immunity of the SHIPPED sensor gate)")
  print(f"    off-contact raw |qfrc_constraint| per joint (the dof-friction baseline) mean: "
        f"{[f'{v:.3f}' for v in FB.mean(0).tolist()]}")
  print(f"    SHIPPED acc.impulse observed before any contact: {shipped_leak:.9f}  (must be 0)")
  print(f"    cross-check shipped vs script — window Λ worst excess-over-tol: {max(ximp_err):.6f}; "
        f"delivered Δ worst excess-over-tol: {max(xdel_err):.6f}  (0 = agree)")

  print("\n[2] PER-JOINT Λ_j [N·m·s] over the contact window")
  print(f"    {'joint':<8}{'raw mean':>10}{'raw p95':>10}{'raw max':>10}{'sub mean':>10}{'contam%':>9}")
  raw_mean, raw_p95, raw_max = RAW.mean(0), _pct(RAW, 0.95), RAW.amax(0)
  sub_mean = SUB.mean(0)
  contam = (1.0 - sub_mean / raw_mean.clamp_min(1e-9)) * 100.0
  for j in range(6):
    print(f"    {ARM[j]:<8}{raw_mean[j]:>10.4f}{raw_p95[j]:>10.4f}{raw_max[j]:>10.4f}"
          f"{sub_mean[j]:>10.4f}{contam[j]:>8.1f}%")
  print(f"    (contam% = friction/weld baseline share of the raw contact-window sum; "
        f"baseline-subtracted is the cleaner impact signal.)")

  print("\n[3] OBJECT-SIDE delivered axial impulse (weld/friction-IMMUNE ground truth)")
  print(f"    ∫F_axial dt over the window: mean {DEL.mean():.4f}  p95 {_pct(DEL.unsqueeze(1),0.95)[0]:.4f}  "
        f"max {DEL.amax():.4f} N·s")
  # Pinocchio independent cross-check (impulseDynamics, r_coeff=0) — scaffolded.
  try:
    import pinocchio  # noqa: F401
    print("    Pinocchio AVAILABLE — TODO: build Z1 model from URDF + impulseDynamics(r_coeff=0) "
          "cross-check (see report; not yet wired).")
  except ImportError:
    print("    Pinocchio NOT installed → independent impulseDynamics cross-check SKIPPED. "
          "The contact-sensor ∫F·dt above is the MuJoCo-native weld-immune ground truth; "
          "see the C0 report DECISION on installing Pinocchio (env + Vega).")

  print("\n[4] PER-JOINT J_limit (Harmonic-Drive Repeated-Peak × impact duration)")
  j_limit = TAU_RATED * REPEATED_PEAK * dt_impact  # (6,) N·m·s
  print(f"    τ_rated [N·m]: {TAU_RATED.tolist()}  ×{REPEATED_PEAK} (Repeated-Peak)  ×{dt_impact*1000:.1f} ms")
  print(f"    J_limit [N·m·s]: {[f'{v:.3f}' for v in j_limit.tolist()]}")
  binding = raw_p95 / j_limit.clamp_min(1e-9)
  print(f"    binding-ness Λ_j(p95)/J_limit: {[f'{v:.3f}' for v in binding.tolist()]}")
  print(f"    → reference strike sits at {binding.max()*100:.1f}% of the worst-joint hardware limit "
        f"(NON-binding for gentle reference strikes; binds for the ~{1/binding.max().clamp_min(1e-9):.0f}× "
        "more violent LEARNED strikes the velocity result showed). 1e4-event fatigue budget: a strike "
        f"at Λ_j(p95) uses 1/1e4 of the per-joint budget; the bound targets the aggressive tail.")

  print("\n[5] NORMALIZER FLOOR (C2; env_cfgs cat_soft imp_seed)")
  print(f"    reference p95(Λ_j) worst joint = {raw_p95.max():.4f}  (small-sample caveat: p95 of "
        f"{S} strikes ≈ the max)")
  print("    NOTE: imp_seed is only a DECAY FLOOR — the hook self-seeds each joint's scale from its "
        "first over-limit sample (CaT-style), so no excess-scale statistic is needed here; any "
        "small floor (e.g. 1e-3) is safe.")

  # --- gate verdict ---
  ok = True
  if shipped_leak > 1e-9:
    print(f"\n[GATE FAIL] SHIPPED accumulator nonzero before any contact ({shipped_leak:.2e}) — "
          "its contact-sensor gate is leaking off-contact friction into Λ_j.")
    ok = False
  if max(ximp_err) > 1e-6:
    print(f"\n[GATE FAIL] shipped window Λ disagrees with the script's independent sum "
          f"(worst excess-over-tolerance {max(ximp_err):.4f}) — the wired accumulator drifted "
          "from the audited semantics.")
    ok = False
  if max(xdel_err) > 1e-6:
    print(f"\n[GATE FAIL] shipped delivered-impulse Δ disagrees with the script's ∫F·dt "
          f"(worst excess-over-tolerance {max(xdel_err):.4f}).")
    ok = False
  if not (raw_mean.max() > 0 and sub_mean.max() > 0):
    print("\n[GATE FAIL] Λ_j is zero on the reference strike — no impact signal captured.")
    ok = False
  if not (DEL.mean() > 0):
    print("\n[GATE FAIL] object-side delivered impulse is zero — netforce sensor / axis wrong.")
    ok = False
  worst_contam = float(contam.max())
  if worst_contam > WELD_TOL * 100.0:
    print(f"\n[GATE WARN] worst-joint contamination {worst_contam:.0f}% > {WELD_TOL*100:.0f}% tol — "
          "prefer subtract_baseline=True for the shipped quantity (report decision).")
  print(f"\n=== C0 QUANTITY GATE: {'PASS' if ok else 'FAIL'} ===")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
