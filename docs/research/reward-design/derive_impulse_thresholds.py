"""Log-only direct-reference impulse quantity cross-check (legacy filename).

Repeated executions of the one production direct strike are deterministic
repeatability observations, not an intensity sweep or a population sample.
This script checks accumulator/contact-row/delivered-impulse liveness and
reports measured Λ against the imported frozen ``IMP_J_LIMIT``.  Contact
duration is descriptive only: this script cannot calculate replacement caps,
normalizers, or enforcement settings, and writes no result asset.

Run: ~/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/derive_impulse_thresholds.py
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference

from reward_design_util import (
  assert_log_only_reference_contract,
  direct_reference_repeat_indices,
  load_direct_reference_c0_cfg,
)

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
REPEATS = 5
HOLD_STEPS = 6
WELD_TOL = 0.5  # max acceptable off-contact-baseline fraction of the contact-window Λ (gate)
ROWS_RAW_TOL = 1.05  # Track-2 triangle-inequality check (AMENDED 2026-07-10 sign-aware; DEMOTED
# to INFORMATIONAL 2026-07-14, adversarial-review I5): rows ≤ raw + noncontact is the algebraic
# identity |c| ≤ |q| + |q − c| — it holds by construction when all three sums share the same
# per-substep signals, so it certifies nothing (the 2026-07-10 amendment fixed the falsified
# `rows ≤ raw` by making it unfalsifiable). Kept as a drift tripwire only (a violation = shipped
# accumulator vs script-sum divergence); the ASSERTED Track-2 gates are the object-side ∫F·dt
# cross-checks. noncontact_j = Σ_substeps |qfrc_constraint_j − contact_row_qfrc_j| · dt over the
# SAME window as raw/rows; 5% slack covers float/window-boundary noise.
def direct_reference_measurement_summary(
  measured_lambda: torch.Tensor,
  *,
  contact_duration_s: list[float],
) -> dict[str, object]:
  """Return a log-only summary whose cap is independent of measured duration."""
  values = torch.as_tensor(measured_lambda, dtype=torch.float64)
  if values.ndim != 2 or values.shape[1] != len(IMP_J_LIMIT):
    raise ValueError(
      f"measured_lambda must have shape (samples, {len(IMP_J_LIMIT)}), got {tuple(values.shape)}"
    )
  if len(contact_duration_s) != values.shape[0]:
    raise ValueError("contact_duration_s must have one descriptive value per repeat")
  caps = torch.tensor(IMP_J_LIMIT, dtype=values.dtype)
  return {
    "sample_kind": "deterministic direct-reference repeatability",
    "contact_duration_s": [float(value) for value in contact_duration_s],
    "measured_lambda_nms": values.tolist(),
    "frozen_cap_nms": list(IMP_J_LIMIT),
    "lambda_to_frozen_cap": (values / caps).tolist(),
  }


def main() -> int:
  cfg = load_direct_reference_c0_cfg()
  env = ManagerBasedRlEnv(cfg, device="cpu")
  hook = env.metrics_manager.cfg["cat_soft"].func
  assert_log_only_reference_contract(cfg, live_imp_limit=hook._imp_limit)
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

  # --- collect fixed repeats of the one nominal direct strike ---
  per_joint_raw: list[torch.Tensor] = []      # Λ_j raw-gated, per strike (script-side)
  per_joint_sub: list[torch.Tensor] = []      # Λ_j baseline-subtracted, per strike (shipped/live)
  per_joint_rows: list[torch.Tensor] = []     # Track 2: rigorous efc-row-only Λ_j, per strike (shipped)
  per_joint_noncontact: list[torch.Tensor] = []  # Σ|qfrc − contact_row_qfrc|·dt, per strike (sign-aware bound term)
  delivered: list[float] = []                 # object-side ∫F_axial dt, per strike (script-side)
  durations: list[int] = []                   # contact-window substeps, per strike
  friction_baseline: list[torch.Tensor] = []  # off-contact |qfrc| per joint (the contaminant scale)
  shipped_leak = 0.0                          # SHIPPED acc.impulse seen before any contact (must be 0)
  ximp_err: list[float] = []                  # |shipped window Λ − script baseline-subtracted Λ| per strike
  xdel_err: list[float] = []                  # |shipped delivered Δ − script deliv| per strike

  for _repeat_index in direct_reference_repeat_indices(REPEATS):
    env.reset()
    ref = SingleStrikeReference(1, env.device)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()
    rec.clear()
    for k in range(1, n + HOLD_STEPS + 1):
      target = ref.playback_target(min(k, n))
      action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
      env.step(action)
      if bool(env.reset_terminated.any()):
        break

    # Pre-contact baseline = last off-contact qfrc before the first contact window.
    baseline = torch.zeros(6)
    raw = torch.zeros(6)
    imp_window = int(acc_shipped._window)
    del_window = int(dacc_shipped._window)
    sub_win_buf: list[torch.Tensor] = []
    sub_capped = torch.zeros(6)
    noncontact = torch.zeros(6)
    dur = 0
    deliv = 0.0
    seen_contact = False
    shipped_win = torch.zeros(6)
    shipped_rows_win = torch.zeros(6)
    shipped_del_start = 0.0
    shipped_del_end = 0.0
    deliv_capped = 0.0
    for in_c, qfrc, contact_qfrc, f_ax, ship_imp, ship_del, ship_rows in rec:
      if not in_c:
        baseline = qfrc
        friction_baseline.append(qfrc.abs())
        if not seen_contact:
          shipped_leak = max(shipped_leak, float(ship_imp.abs().max()))
          shipped_del_start = ship_del
        if seen_contact:
          break
      else:
        seen_contact = True
        raw += qfrc.abs() * dt
        sub_win_buf.append((qfrc - baseline).abs() * dt)
        if len(sub_win_buf) > imp_window:
          sub_win_buf.pop(0)
        sub_capped = torch.maximum(sub_capped, torch.stack(sub_win_buf).sum(dim=0))
        noncontact += (qfrc - contact_qfrc).abs() * dt
        deliv += f_ax * dt
        if dur < del_window:
          deliv_capped += f_ax * dt
        dur += 1
        shipped_win = torch.maximum(shipped_win, ship_imp)
        shipped_rows_win = torch.maximum(shipped_rows_win, ship_rows)
        shipped_del_end = ship_del
    if seen_contact and dur > 0:
      per_joint_raw.append(raw)
      per_joint_sub.append(shipped_win)
      per_joint_rows.append(shipped_rows_win)
      per_joint_noncontact.append(noncontact)
      delivered.append(deliv)
      durations.append(dur)
      # The script-side sliding-window and prefix sums independently mirror the
      # shipped accumulators.  Only float tolerance is discounted here.
      ximp_err.append(max(0.0, float((shipped_win - sub_capped).abs().max()) - 1e-4))
      xdel_err.append(max(0.0, abs((shipped_del_end - shipped_del_start) - deliv_capped) - 1e-4))

  env.metrics_manager.compute_substep = orig_substep  # type: ignore[method-assign]

  if not per_joint_raw:
    print("[FAIL] no contact windows captured — reference strike never contacted the nail.")
    return 1
  if len(per_joint_raw) != REPEATS:
    print(
      f"[FAIL] captured {len(per_joint_raw)}/{REPEATS} required direct-reference repeats."
    )
    return 1

  RAW = torch.stack(per_joint_raw)   # (S, 6)
  SUB = torch.stack(per_joint_sub)   # (S, 6), shipped baseline-subtracted sliding-window quantity
  ROWS = torch.stack(per_joint_rows) # (S, 6) Track 2: rigorous efc-row-only Λ_j (Task 9)
  NONCONTACT = torch.stack(per_joint_noncontact)  # (S, 6) Σ|qfrc−contact_row_qfrc|·dt (sign-aware bound term)
  DEL = torch.tensor(delivered)      # (S,)
  DUR = torch.tensor(durations, dtype=torch.float32)  # (S,)
  FB = torch.stack(friction_baseline) if friction_baseline else torch.zeros(1, 6)
  S = RAW.shape[0]
  duration_s = [float(value) * dt for value in durations]
  measurement = direct_reference_measurement_summary(
    SUB, contact_duration_s=duration_s
  )
  frozen_cap = torch.tensor(IMP_J_LIMIT, dtype=SUB.dtype)

  print(
    f"\n=== DIRECT-REFERENCE LOG-ONLY QUANTITY CROSS-CHECK — "
    f"{S} deterministic repeatability samples ==="
  )
  print(
    "contact-window duration (descriptive only; never used to derive a cap): "
    f"{[f'{value * 1000:.1f}' for value in duration_s]} ms"
  )

  print("\n[1] OFF-CONTACT (weld/friction-immunity of the SHIPPED sensor gate)")
  print(f"    off-contact raw |qfrc_constraint| per joint (the dof-friction baseline) mean: "
        f"{[f'{v:.3f}' for v in FB.mean(0).tolist()]}")
  print(f"    SHIPPED acc.impulse observed before any contact: {shipped_leak:.9f}  (must be 0)")
  print(f"    cross-check shipped vs script — window Λ worst excess-over-tol: {max(ximp_err):.6f}; "
        f"delivered Δ worst excess-over-tol: {max(xdel_err):.6f}  (0 = agree)")

  print("\n[2] PER-JOINT Λ_j [N·m·s] over the contact window")
  print(
    f"    {'joint':<8}{'raw mean':>10}{'raw max':>10}{'sub mean':>10}"
    f"{'frozen cap':>12}{'max Λ/cap':>12}{'contam%':>9}"
  )
  raw_mean, raw_max = RAW.mean(0), RAW.amax(0)
  sub_mean = SUB.mean(0)
  contam = (1.0 - sub_mean / raw_mean.clamp_min(1e-9)) * 100.0
  lambda_to_cap_max = (SUB / frozen_cap).amax(0)
  for j in range(6):
    print(
      f"    {ARM[j]:<8}{raw_mean[j]:>10.4f}{raw_max[j]:>10.4f}"
      f"{sub_mean[j]:>10.4f}{frozen_cap[j]:>12.3f}"
      f"{lambda_to_cap_max[j]:>12.4f}{contam[j]:>8.1f}%"
    )
  print(
    "    frozen cap is imported IMP_J_LIMIT; measured duration cannot change it. "
    "raw is diagnostic; sub is the shipped baseline-subtracted window quantity."
  )

  print("\n[3] OBJECT-SIDE delivered axial impulse (weld/friction-IMMUNE ground truth)")
  print(
    f"    per-repeat ∫F_axial dt: {[f'{value:.4f}' for value in DEL.tolist()]} N·s; "
    f"mean {DEL.mean():.4f} N·s (measurement only; no normalizer calibration)"
  )

  print("\n[4] TRACK 2 — THREE-WAY QUANTITY VALIDATION "
        "(raw Λ | baseline-subtracted Λ | contact-row Λ | object-side ∫F·dt)")
  rows_mean = ROWS.mean(0)
  noncont_mean = NONCONTACT.mean(0)
  friction_share = (raw_mean - rows_mean) / raw_mean.clamp_min(1e-9) * 100.0
  residual_after_sub = (sub_mean - rows_mean) / rows_mean.clamp_min(1e-9) * 100.0
  print(f"    {'joint':<8}{'raw':>9}{'sub':>9}{'rows':>9}{'noncont':>9}{'fric-shr%':>11}{'resid%':>9}")
  for j in range(6):
    print(f"    {ARM[j]:<8}{raw_mean[j]:>9.4f}{sub_mean[j]:>9.4f}{rows_mean[j]:>9.4f}"
          f"{noncont_mean[j]:>9.4f}{friction_share[j]:>10.1f}%{residual_after_sub[j]:>8.1f}%")
  print(f"    object-side ∫F_axial dt (task-space N·s, single scalar, NOT per-joint): "
        f"mean {DEL.mean():.4f}  max {DEL.amax():.4f}")
  print("    friction share = (raw − rows)/raw; residual after subtraction = (subtracted − rows)/rows "
        "(rows = the rigorous efc-row-only ground truth, Task 9). "
        "noncont = Σ|qfrc − contact_row_qfrc|·dt, the sign-aware bound term (rows ≤ raw + noncont).")

  # --- gate verdict ---
  ok = True
  if not all(
    bool(torch.isfinite(value).all())
    for value in (RAW, SUB, ROWS, NONCONTACT, DEL, DUR)
  ):
    print("\n[GATE FAIL] a direct-reference measurement is non-finite.")
    ok = False
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
  if not bool((DEL > 0).all()):
    print("\n[GATE FAIL] object-side delivered impulse is not positive on every repeat.")
    ok = False
  worst_contam = float(contam.max())
  if worst_contam > WELD_TOL * 100.0:
    print(f"\n[GATE WARN] worst-joint contamination {worst_contam:.0f}% > {WELD_TOL*100:.0f}% tol — "
          "prefer subtract_baseline=True for the shipped quantity (report decision).")

  # Triangle-inequality check is informational: it is a drift tripwire, not a
  # calibration or independent physics certificate.
  # rows ≤ raw + noncontact is the algebraic identity |c| ≤ |q| + |q − c| — when all three sums
  # come from the same per-substep signals it holds BY CONSTRUCTION and certifies nothing about
  # measurement quality (the 2026-07-10 amendment made the old falsified `rows ≤ raw` gate
  # unfalsifiable rather than correct). A violation can still flag gross implementation drift
  # between the SHIPPED rows accumulator and this script's sums (different windowing, double
  # counting), so it is still computed and printed.  The asserted independent checks are the
  # object-side/contact-row liveness below and the shipped-vs-script agreement gates in [1].
  rows_bound = (RAW + NONCONTACT) * ROWS_RAW_TOL + 1e-6
  rows_ok = bool((ROWS <= rows_bound).all())
  if not rows_ok:
    excess = float((ROWS - rows_bound).clamp_min(0.0).max())
    print(f"\n[GATE WARN — informational] contact-row Λ exceeds {ROWS_RAW_TOL:.2f}× "
          f"(raw Λ + noncontact Λ) (worst excess {excess:.4f}). This triangle-inequality identity "
          "should be unviolable when rows/raw/noncontact share the same substeps — a violation "
          "means the shipped rows accumulator and this script's sums have drifted apart "
          "(windowing/double-count drift), NOT a physics finding. Investigate, but the verdict "
          "rests on the asserted ∫F·dt cross-checks below.")
  else:
    print(f"\n    [4] triangle-inequality check (informational): contact-row Λ ≤ "
          f"{ROWS_RAW_TOL:.2f}× (raw Λ + noncontact Λ) for all {S} strikes — holds by "
          "construction; the asserted Track-2 gates are the ∫F·dt cross-checks below.")

  # Track-2 liveness: contact-row Λ must be present whenever object-side
  # delivered impulse reports a strike.  Repeats do not support an
  # intensity-response or population-spread claim.
  rows_worst = ROWS.amax(dim=1)  # (S,) worst-joint contact-row Λ per strike
  strike_mask = DEL > 1e-6
  if bool(strike_mask.any()):
    missing = strike_mask & (rows_worst <= 1e-9)
    if bool(missing.any()):
      print(f"\n[TRACK-2 BUG] contact-row Λ is zero on {int(missing.sum())}/{S} strike(s) where "
            "object-side ∫F·dt shows a real strike. This blocks TRACK-2 (contact-row / Task 8-10) "
            "conclusions ONLY — the shipped Track-1 quantity (raw / baseline-subtracted Λ, "
            "sections [1]-[2]) is measured independently and is UNAFFECTED.")
      ok = False
    else:
      print(
        f"    Track-2 liveness PASS: contact-row Λ is positive on all "
        f"{int(strike_mask.sum())} delivered-impulse repeats."
      )

  assert measurement["frozen_cap_nms"] == IMP_J_LIMIT
  print(f"\n=== DIRECT-REFERENCE LOG-ONLY CROSS-CHECK: {'PASS' if ok else 'FAIL'} ===")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
