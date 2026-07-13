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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (macOS/CI safe) — must precede pyplot import

import matplotlib.pyplot as plt
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
ROWS_RAW_TOL = 1.05  # Track-2 hard gate (AMENDED 2026-07-10, sign-aware): contact-row Λ must
# satisfy rows ≤ raw + noncontact (triangle inequality; exact per substep given the Task-8
# reconstruction invariant) — NOT `rows ≤ raw` alone, which joint3's dof-friction sign-cancellation
# falsifies (raw undershoots the true reaction there; see IMPULSE_CAT_IMPL_PLAN.md Status area).
# noncontact_j = Σ_substeps |qfrc_constraint_j − contact_row_qfrc_j| · dt over the SAME window as
# raw/rows; 5% slack covers float/window-boundary noise.
TRACK2_RATIO_SPREAD_TOL = 5.0  # Track-2 cross-check: worst-joint contact-row Λ [N·m·s] and object-
# side ∫F·dt [N·s] are DIFFERENT units, so only the SPREAD of their ratio across strikes is
# meaningful (both should scale together with impact intensity). Wider than this across the
# APPROACH_HEIGHTS sweep signals a row-attribution bug (Task 8/9), not physical scaling.


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
  from src.tasks.hammer.mdp.contact_row_impulse import (
    _ENV_SUBSTEP_ROWS_ATTR,
    arm_dof_cols,
    contact_row_qfrc,
  )
  from src.tasks.hammer.mdp.impulse_bound import (
    _ENV_SUBSTEP_DELIVERED_ATTR,
    _ENV_SUBSTEP_IMPULSE_ATTR,
  )

  acc_shipped = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)
  dacc_shipped = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR)
  acc_rows = getattr(env, _ENV_SUBSTEP_ROWS_ATTR)  # Track 2 (Task 9): rigorous efc-row-only Λ
  cols = arm_dof_cols(env)  # GLOBAL dof columns for ARM (same order), so contact_row_qfrc's
  # (nworld, nv) output selects the identical 6 arm joints as `jid` does on qfrc_constraint.
  # (in_contact, qfrc_arm(6), contact_qfrc_arm(6), f_axial, shipped_impulse(6), shipped_delivered,
  #  shipped_rows_impulse(6)) — contact_qfrc_arm feeds the sign-aware bound's noncontact term.
  rec: list[tuple[bool, torch.Tensor, torch.Tensor, float, torch.Tensor, float, torch.Tensor]] = []
  # Hook AFTER the shipped accumulators run (metrics_manager.compute_substep follows scene.update
  # in the decimation loop), so every record's shipped snapshot includes its OWN substep — the
  # cross-check is exact by construction, with no alignment tolerance to hide an off-by-one behind.
  orig_substep = env.metrics_manager.compute_substep

  def patched() -> None:
    orig_substep()
    qfrc = robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone()  # (6,)
    contact_qfrc = contact_row_qfrc(env)[0, cols].clone()  # (6,) signed contact-row projection,
    # read at this SAME substep as `qfrc` above — required for the triangle-inequality bound
    # |contact| ≤ |qfrc| + |qfrc − contact| to hold (sign-aware rows≤raw+noncontact gate).
    in_c = bool((contact.data.found > 0).any())
    f = netf.data.force  # (1, N, 3) world
    f_ax = float((f * axis).sum(-1).sum(-1).clamp_min(0.0)[0])
    rec.append((
      in_c, qfrc, contact_qfrc, f_ax,
      acc_shipped.impulse[0].clone(), float(dacc_shipped.delivered[0]),
      acc_rows.impulse[0].clone(),
    ))

  env.metrics_manager.compute_substep = patched  # type: ignore[method-assign]

  # --- collect many reference strikes ---
  per_joint_raw: list[torch.Tensor] = []      # Λ_j raw-gated, per strike (script-side)
  per_joint_sub: list[torch.Tensor] = []      # Λ_j baseline-subtracted, per strike (script-side)
  per_joint_rows: list[torch.Tensor] = []     # Track 2: rigorous efc-row-only Λ_j, per strike (shipped)
  per_joint_noncontact: list[torch.Tensor] = []  # Σ|qfrc − contact_row_qfrc|·dt, per strike (sign-aware bound term)
  delivered: list[float] = []                 # object-side ∫F_axial dt, per strike (script-side)
  durations: list[int] = []                   # contact-window substeps, per strike
  friction_baseline: list[torch.Tensor] = []  # off-contact |qfrc| per joint (the contaminant scale)
  shipped_leak = 0.0                          # SHIPPED acc.impulse seen before any contact (must be 0)
  ximp_err: list[float] = []                  # |shipped window Λ − script baseline-subtracted Λ| per strike
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
      # Mirror of the SHIPPED accumulator's SLIDING window (2026-07-13): max over any
      # `imp_window`-substep interval of the baseline-subtracted sum. Window length read from the
      # runtime accumulator (never hardcoded) so the mirror cannot silently desync from the config.
      imp_window = int(acc_shipped._window)
      del_window = int(dacc_shipped._window)
      sub_win_buf: list[torch.Tensor] = []  # last `imp_window` per-substep contributions
      sub_capped = torch.zeros(6)           # max sliding-window sum seen (== full sum if dur < window)
      noncontact = torch.zeros(6)
      dur = 0
      deliv = 0.0
      seen_contact = False
      shipped_win = torch.zeros(6)  # per-joint max of the SHIPPED acc.impulse over the window
      shipped_rows_win = torch.zeros(6)  # per-joint max of the SHIPPED rows acc.impulse (Track 2)
      shipped_del_start = 0.0
      shipped_del_end = 0.0
      deliv_capped = 0.0  # script-side mirror of the shipped per-event accrual cap (first 25 substeps)
      for in_c, qfrc, contact_qfrc, f_ax, ship_imp, ship_del, ship_rows in rec:
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
          # Sliding-window mirror (impulse_bound.py, 2026-07-13): track the max window-sum so the
          # ximp_err gate compares like with like. For dur < window this equals the full sum.
          sub_win_buf.append((qfrc - baseline).abs() * dt)
          if len(sub_win_buf) > imp_window:
            sub_win_buf.pop(0)
          sub_capped = torch.maximum(sub_capped, torch.stack(sub_win_buf).sum(dim=0))
          # Sign-aware bound term (ADJUDICATED 2026-07-10): the non-contact-row component of qfrc
          # (dof-friction/limit contamination), accumulated over the IDENTICAL contact window as
          # raw/rows above — so rows_j ≤ raw_j + noncontact_j (triangle inequality) holds exactly.
          noncontact += (qfrc - contact_qfrc).abs() * dt
          deliv += f_ax * dt
          if dur < del_window:  # mirrors SubstepDeliveredImpulse's per-event PREFIX cap (unchanged)
            deliv_capped += f_ax * dt
          dur += 1
          shipped_win = torch.maximum(shipped_win, ship_imp)
          shipped_rows_win = torch.maximum(shipped_rows_win, ship_rows)
          shipped_del_end = ship_del
      if seen_contact and dur > 0:
        per_joint_raw.append(raw)
        per_joint_sub.append(sub)
        per_joint_rows.append(shipped_rows_win)
        per_joint_noncontact.append(noncontact)
        delivered.append(deliv)
        durations.append(dur)
        # Cross-check the SHIPPED buffers against this script's independent sums — records snapshot
        # AFTER metrics.compute_substep, so agreement must be exact (float tolerance only). The
        # shipped substep_impulse accumulator is configured subtract_baseline=True (C2, env_cfgs.py),
        # so its window value IS the baseline-subtracted sum — compare against `sub_capped`, not
        # `raw` (fixed 2026-07-10: previously compared `raw`, a stale leftover from before the C2
        # subtract_baseline=True switch). sub_capped (sliding-window since 2026-07-13) mirrors the
        # shipped accumulator's max window-sum with the window read from the runtime instance;
        # an uncapped `sub` would spuriously FAIL the gate on any strike whose contact outlives
        # the window (the reference strike's ~9-20 substeps is unaffected: windowed == full sum).
        ximp_err.append(max(0.0, float((shipped_win - sub_capped).abs().max()) - 1e-4))
        xdel_err.append(max(0.0, abs((shipped_del_end - shipped_del_start) - deliv_capped) - 1e-4))

  env.metrics_manager.compute_substep = orig_substep  # type: ignore[method-assign]

  if not per_joint_raw:
    print("[FAIL] no contact windows captured — reference strike never contacted the nail.")
    sys.exit(1)

  RAW = torch.stack(per_joint_raw)   # (S, 6)
  SUB = torch.stack(per_joint_sub)   # (S, 6)
  ROWS = torch.stack(per_joint_rows) # (S, 6) Track 2: rigorous efc-row-only Λ_j (Task 9)
  NONCONTACT = torch.stack(per_joint_noncontact)  # (S, 6) Σ|qfrc−contact_row_qfrc|·dt (sign-aware bound term)
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

  print("\n[6] TRACK 2 — THREE-WAY QUANTITY VALIDATION "
        "(raw Λ | baseline-subtracted Λ | contact-row Λ | object-side ∫F·dt)")
  rows_mean = ROWS.mean(0)
  noncont_mean = NONCONTACT.mean(0)  # sign-aware bound term (ADJUDICATED 2026-07-10): the
  # non-contact-row share of qfrc; rows_j ≤ raw_j + noncont_j is the amended hard gate below.
  # friction share: fraction of the raw contact-window sum that is dof-friction contamination, not
  # real hammer<->nail contact reaction (rows is the rigorous efc-row-only ground truth, Task 9).
  # NOTE: friction_share/residual_after_sub below are ratios-OF-MEANS (mean over strikes, THEN one
  # ratio) — NOT the mean of each strike's own ratio; quote per-strike numbers before citing these
  # percentages in the thesis.
  friction_share = (raw_mean - rows_mean) / raw_mean.clamp_min(1e-9) * 100.0
  # residual after subtraction: fraction the SHIPPED enforced quantity (baseline-subtracted Λ, C2)
  # still overshoots the rigorous ground truth by, after the cheaper baseline-subtraction correction.
  residual_after_sub = (sub_mean - rows_mean) / rows_mean.clamp_min(1e-9) * 100.0
  print(f"    {'joint':<8}{'raw':>9}{'sub':>9}{'rows':>9}{'noncont':>9}{'fric-shr%':>11}{'resid%':>9}")
  for j in range(6):
    print(f"    {ARM[j]:<8}{raw_mean[j]:>9.4f}{sub_mean[j]:>9.4f}{rows_mean[j]:>9.4f}"
          f"{noncont_mean[j]:>9.4f}{friction_share[j]:>10.1f}%{residual_after_sub[j]:>8.1f}%")
  print(f"    object-side ∫F_axial dt (task-space N·s, single scalar, NOT per-joint): "
        f"mean {DEL.mean():.4f}  max {DEL.amax():.4f}")
  print("    friction share = (raw − rows)/raw; residual after subtraction = (subtracted − rows)/rows "
        "(rows = the rigorous efc-row-only ground truth, Task 9; see the module docstring's [2]). "
        "noncont = Σ|qfrc − contact_row_qfrc|·dt, the sign-aware bound term (rows ≤ raw + noncont).")

  fig_dir = Path(__file__).parent / "figures"
  fig_dir.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(9, 5))
  x = list(range(6))
  w = 0.25
  ax.bar([xi - w for xi in x], raw_mean.tolist(), width=w, label="raw Λ (qfrc, uncorrected)")
  ax.bar(x, sub_mean.tolist(), width=w, label="baseline-subtracted Λ (shipped, enforced, C2)")
  ax.bar(
    [xi + w for xi in x], rows_mean.tolist(), width=w,
    label="contact-row Λ (efc rows, rigorous GT, Task 9)",
  )
  ax.set_xticks(x)
  ax.set_xticklabels(ARM)
  ax.set_ylabel(f"Λ_j  [N·m·s]  (mean over {S} reference strikes)")
  ax.set_title(
    "Impulse-CaT quantity-contamination gate\n"
    f"object-side ∫F_axial·dt (task-space GT) mean={DEL.mean():.3f} N·s"
  )
  ax.legend()
  fig.tight_layout()
  fig_path = fig_dir / "impulse_contamination.png"
  fig.savefig(fig_path, dpi=150)
  plt.close(fig)
  print(f"    figure saved: {fig_path}")

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

  # [6] hard gate (AMENDED 2026-07-10, sign-aware): rows ≤ raw + noncontact — the triangle-
  # inequality bound |contact| ≤ |qfrc| + |qfrc − contact| (exact per substep given the Task-8
  # reconstruction invariant), NOT `rows ≤ raw` alone (falsified by joint3's dof-friction sign-
  # cancellation, see IMPULSE_CAT_IMPL_PLAN.md Status area / task-10-report.md).
  rows_bound = (RAW + NONCONTACT) * ROWS_RAW_TOL + 1e-6
  rows_ok = bool((ROWS <= rows_bound).all())
  if not rows_ok:
    excess = float((ROWS - rows_bound).clamp_min(0.0).max())
    print(f"\n[GATE FAIL] contact-row Λ exceeds {ROWS_RAW_TOL:.2f}× (raw Λ + noncontact Λ) "
          f"(worst excess {excess:.4f}) — the sign-aware bound rows ≤ raw + noncontact "
          "(triangle inequality) is violated; this indicates a genuine attribution bug "
          "(double-counted rows, wrong-world reads), not physical sign-cancellation.")
    ok = False
  else:
    print(f"\n    [6] hard gate PASS: contact-row Λ ≤ {ROWS_RAW_TOL:.2f}× (raw Λ + noncontact Λ) "
          f"for all {S} strikes (sign-aware bound).")

  # [6] Track-2 cross-check: contact-row Λ vs. the weld/friction-immune object-side ∫F·dt. Different
  # units (N·m·s per joint vs N·s task-space) so we check (a) rows is never absent when a real strike
  # delivered impulse, and (b) the ratio between them doesn't swing wildly across strikes — a bug
  # (e.g. row misattribution) would decouple the two independent measurements of the SAME event.
  rows_worst = ROWS.amax(dim=1)  # (S,) worst-joint contact-row Λ per strike
  strike_mask = DEL > 1e-6
  if bool(strike_mask.any()):
    missing = strike_mask & (rows_worst <= 1e-9)
    if bool(missing.any()):
      print(f"\n[TRACK-2 BUG] contact-row Λ is zero on {int(missing.sum())}/{S} strike(s) where "
            "object-side ∫F·dt shows a real strike. This blocks TRACK-2 (contact-row / Task 8-10) "
            "conclusions ONLY — the enforced Track-1 quantity (raw / baseline-subtracted Λ, "
            "sections [1]-[2]) is measured independently and is UNAFFECTED.")
      ok = False
    elif int(strike_mask.sum()) >= 2:
      ratio = rows_worst[strike_mask] / DEL[strike_mask]
      spread = float(ratio.max() / ratio.min().clamp_min(1e-9))
      if spread > TRACK2_RATIO_SPREAD_TOL:
        print(f"\n[TRACK-2 BUG] contact-row Λ / object-side ∫F·dt ratio spreads {spread:.1f}× across "
              f"strikes (tol {TRACK2_RATIO_SPREAD_TOL:.1f}×; units differ so only the SPREAD is "
              "checked). This blocks TRACK-2 (contact-row / Task 8-10) conclusions ONLY — Track-1 "
              "(raw / baseline-subtracted Λ) is measured independently and is UNAFFECTED.")
        ok = False
      else:
        print(f"    Track-2 cross-check PASS: contact-row Λ(worst-joint)/object-side ∫F·dt ratio "
              f"spread {spread:.1f}× across {int(strike_mask.sum())} strikes "
              f"(tol {TRACK2_RATIO_SPREAD_TOL:.1f}×).")

  print(f"\n=== C0 QUANTITY GATE: {'PASS' if ok else 'FAIL'} ===")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
