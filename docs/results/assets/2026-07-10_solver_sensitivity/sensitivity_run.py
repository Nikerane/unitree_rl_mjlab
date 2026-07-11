"""Solver-parameter sensitivity run for the impulse-CaT quantity (one-off thesis defense artifact).

Copy of docs/research/reward-design/derive_impulse_thresholds.py (2026-07-10 working tree),
parametrized into three configurations (QFRC_MEASUREMENT_AUDIT.md section 2.1 recommendation):

  --config baseline   unmodified (timestep 0.002, decimation 10, XML solref)
  --config halfdt     physics timestep halved (0.001), decimation doubled (20) so control dt
                      stays 0.02 s and the control-rate reference trajectory is unchanged.
                      All accumulators (script-side and shipped) read env.physics_dt =
                      cfg.sim.mujoco.timestep, so the accumulation dt follows automatically.
  --config solref2x   hammer<->nail contact solref timeconst doubled on BOTH contacting geom
                      sets (nail_head 0.008->0.016; hammer_head_0/1 default 0.02->0.04), via a
                      spec_fn wrapper BEFORE compile (no repo/asset file is touched). MuJoCo
                      mixes geom solref by solmix-weighted average (equal solmix here), so the
                      effective contact timeconst doubles: 0.014 -> 0.028. Both remain valid
                      (>= 2x timestep 0.002).

Additional recording vs the original gate script (same 15-strike protocol, 3 heights x 5):
  - per-strike PEAK object-side axial contact force  max_t F_axial(t)      [N]
  - per-strike per-joint PEAK |qfrc_constraint_j| over the contact window  [N.m]
  - contact-window length per strike (substeps and ms)
Results are dumped as JSON (--out) for report assembly. The figure and the sys.exit gate
verdict of the original are dropped (this is a measurement run, not a gate); the shipped-vs-
script cross-checks are still computed and reported for integrity.

Run (from the repo root, PYTHONPATH=repo root):
  ~/miniconda3/envs/unitree_mjlab/bin/python sensitivity_run.py --config baseline --out baseline.json
"""

from __future__ import annotations

import argparse
import json
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
APPROACH_HEIGHTS = [0.06, 0.10, 0.15]
REPEATS = 5
HOLD_STEPS = 6

# Geoms whose solref timeconst is doubled in the solref2x arm: the two sides of the
# hammer<->nail contact (the ONLY contact the sensors gate on).
SOLREF_GEOM_SUFFIXES = ("nail_head", "hammer_head_0", "hammer_head_1")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", choices=["baseline", "halfdt", "solref2x"], required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  meta: dict = {"config": args.config}

  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1

  # --- configuration overrides (runtime only; no repo file is modified) ---
  assert cfg.sim.mujoco.timestep == 0.002, cfg.sim.mujoco.timestep
  assert cfg.decimation == 10, cfg.decimation
  if args.config == "halfdt":
    cfg.sim.mujoco.timestep = 0.001  # halved physics step
    cfg.decimation = 20              # control dt stays 0.02 s -> identical reference trajectory
  meta["timestep"] = cfg.sim.mujoco.timestep
  meta["decimation"] = cfg.decimation
  meta["control_dt"] = cfg.sim.mujoco.timestep * cfg.decimation

  if args.config == "solref2x":
    meta["solref_spec_edits"] = {}

    def wrap_spec(entity_cfg) -> None:
      orig_fn = entity_cfg.spec_fn

      def patched_spec():
        spec = orig_fn()
        for g in spec.geoms:
          if g.name in SOLREF_GEOM_SUFFIXES:
            old = [float(v) for v in g.solref]
            if old[0] <= 0:
              raise RuntimeError(
                f"geom {g.name} uses direct-stiffness solref {old}; doubling timeconst "
                "is undefined for this form — abort rather than misconfigure."
              )
            new = list(old)
            new[0] = old[0] * 2.0
            g.solref = new
            meta["solref_spec_edits"][g.name] = {"before": old, "after": new}
        return spec

      entity_cfg.spec_fn = patched_spec

    wrap_spec(cfg.scene.entities["robot"])
    wrap_spec(cfg.scene.entities["nail_block"])

  env = ManagerBasedRlEnv(cfg, device="cpu")

  # --- verify the overrides actually landed in the built env/model ---
  assert abs(env.physics_dt - meta["timestep"]) < 1e-12, (env.physics_dt, meta["timestep"])
  print(f"[cfg] config={args.config}  physics_dt={env.physics_dt}  decimation={cfg.decimation}  "
        f"control_dt={env.step_dt}")

  mj_model = None
  for attr in ("mj_model", "model"):
    cand = getattr(env.sim, attr, None)
    if cand is not None and hasattr(cand, "geom_solref"):
      mj_model = cand
      break
  if mj_model is not None:
    import mujoco  # noqa: PLC0415

    meta["compiled_geom_solref"] = {}
    for gid in range(mj_model.ngeom):
      name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
      if any(name.endswith(suf) for suf in SOLREF_GEOM_SUFFIXES):
        meta["compiled_geom_solref"][name] = [float(v) for v in mj_model.geom_solref[gid]]
        print(f"[cfg] compiled geom_solref[{name}] = {mj_model.geom_solref[gid].tolist()}")
    assert abs(float(mj_model.opt.timestep) - meta["timestep"]) < 1e-12
    if args.config == "solref2x":
      # hard verification: every targeted geom carries the doubled timeconst in the COMPILED model
      for name, sr in meta["compiled_geom_solref"].items():
        suf = next(s for s in SOLREF_GEOM_SUFFIXES if name.endswith(s))
        expected = meta["solref_spec_edits"][suf]["after"][0]
        assert abs(sr[0] - expected) < 1e-12, (name, sr, expected)
      print("[cfg] solref2x override VERIFIED in the compiled model.")
  else:
    print("[cfg] WARNING: could not locate compiled mj_model on env.sim — solref override "
          "verified only at spec level (spec edits recorded in meta).")
    if args.config == "solref2x" and not meta["solref_spec_edits"]:
      print("[FAIL] solref2x requested but no spec geom was edited.")
      sys.exit(1)
  if args.config == "solref2x" and len(meta.get("solref_spec_edits", {})) != 3:
    print(f"[FAIL] solref2x expected 3 geom edits, got {meta.get('solref_spec_edits')}")
    sys.exit(1)

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

  # --- per-substep instrumentation (identical to the gate script) ---
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
  acc_rows = getattr(env, _ENV_SUBSTEP_ROWS_ATTR)
  cols = arm_dof_cols(env)
  rec: list[tuple[bool, torch.Tensor, torch.Tensor, float, torch.Tensor, float, torch.Tensor]] = []
  orig_substep = env.metrics_manager.compute_substep

  def patched() -> None:
    orig_substep()
    qfrc = robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone()  # (6,)
    contact_qfrc = contact_row_qfrc(env)[0, cols].clone()  # (6,)
    in_c = bool((contact.data.found > 0).any())
    f = netf.data.force  # (1, N, 3) world
    f_ax = float((f * axis).sum(-1).sum(-1).clamp_min(0.0)[0])
    rec.append((
      in_c, qfrc, contact_qfrc, f_ax,
      acc_shipped.impulse[0].clone(), float(dacc_shipped.delivered[0]),
      acc_rows.impulse[0].clone(),
    ))

  env.metrics_manager.compute_substep = patched  # type: ignore[method-assign]

  # --- collect many reference strikes (identical protocol to the gate script) ---
  per_joint_raw: list[torch.Tensor] = []
  per_joint_sub: list[torch.Tensor] = []
  per_joint_rows: list[torch.Tensor] = []
  per_joint_noncontact: list[torch.Tensor] = []
  per_joint_peak_qfrc: list[torch.Tensor] = []   # NEW: max |qfrc_j| over the window [N.m]
  peak_force: list[float] = []                   # NEW: max object-side F_axial over the window [N]
  delivered: list[float] = []
  durations: list[int] = []
  friction_baseline: list[torch.Tensor] = []
  shipped_leak = 0.0
  ximp_err: list[float] = []
  xdel_err: list[float] = []
  strike_heights: list[float] = []

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

      baseline = torch.zeros(6)
      raw = torch.zeros(6)
      sub = torch.zeros(6)
      noncontact = torch.zeros(6)
      pk_qfrc = torch.zeros(6)
      pk_f = 0.0
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
            break  # stop at first release (isolate the impact, ignore re-contact)
        else:
          seen_contact = True
          raw += qfrc.abs() * dt
          sub += (qfrc - baseline).abs() * dt
          noncontact += (qfrc - contact_qfrc).abs() * dt
          deliv += f_ax * dt
          if dur < 25:
            deliv_capped += f_ax * dt
          dur += 1
          pk_qfrc = torch.maximum(pk_qfrc, qfrc.abs())
          pk_f = max(pk_f, f_ax)
          shipped_win = torch.maximum(shipped_win, ship_imp)
          shipped_rows_win = torch.maximum(shipped_rows_win, ship_rows)
          shipped_del_end = ship_del
      if seen_contact and dur > 0:
        per_joint_raw.append(raw)
        per_joint_sub.append(sub)
        per_joint_rows.append(shipped_rows_win)
        per_joint_noncontact.append(noncontact)
        per_joint_peak_qfrc.append(pk_qfrc)
        peak_force.append(pk_f)
        delivered.append(deliv)
        durations.append(dur)
        strike_heights.append(h)
        ximp_err.append(max(0.0, float((shipped_win - sub).abs().max()) - 1e-4))
        xdel_err.append(max(0.0, abs((shipped_del_end - shipped_del_start) - deliv_capped) - 1e-4))

  env.metrics_manager.compute_substep = orig_substep  # type: ignore[method-assign]

  if not per_joint_raw:
    print("[FAIL] no contact windows captured — reference strike never contacted the nail.")
    sys.exit(1)

  RAW = torch.stack(per_joint_raw)
  SUB = torch.stack(per_joint_sub)
  ROWS = torch.stack(per_joint_rows)
  NONCONTACT = torch.stack(per_joint_noncontact)
  PKQ = torch.stack(per_joint_peak_qfrc)
  PKF = torch.tensor(peak_force)
  DEL = torch.tensor(delivered)
  DUR = torch.tensor(durations, dtype=torch.float32)
  S = RAW.shape[0]

  raw_mean, sub_mean = RAW.mean(0), SUB.mean(0)
  rows_mean, noncont_mean = ROWS.mean(0), NONCONTACT.mean(0)
  pkq_mean = PKQ.mean(0)

  print(f"\n=== SENSITIVITY [{args.config}] — {S} reference strikes "
        f"(dt={dt}, decimation={cfg.decimation}) ===")
  print(f"contact window: mean {DUR.mean():.1f} substeps = {DUR.mean() * dt * 1000:.2f} ms "
        f"(min {DUR.min():.0f}, max {DUR.max():.0f} substeps)")
  print(f"shipped leak (must be 0): {shipped_leak:.9f}; xcheck worst excess: "
        f"imp {max(ximp_err):.6f}, del {max(xdel_err):.6f}")
  print(f"\n{'joint':<8}{'raw mean':>10}{'sub mean':>10}{'rows mean':>10}{'noncont':>10}"
        f"{'peak|qfrc|':>11}")
  for j in range(6):
    print(f"{ARM[j]:<8}{raw_mean[j]:>10.4f}{sub_mean[j]:>10.4f}{rows_mean[j]:>10.4f}"
          f"{noncont_mean[j]:>10.4f}{pkq_mean[j]:>11.2f}")
  print(f"\nobject-side ∫F_axial dt: mean {DEL.mean():.4f}  max {DEL.amax():.4f} N·s")
  print(f"object-side PEAK F_axial: mean {PKF.mean():.1f}  max {PKF.amax():.1f} N")

  out = {
    "meta": meta,
    "n_strikes": int(S),
    "physics_dt": float(dt),
    "arm": list(ARM),
    "raw_mean": raw_mean.tolist(),
    "sub_mean": sub_mean.tolist(),
    "rows_mean": rows_mean.tolist(),
    "noncontact_mean": noncont_mean.tolist(),
    "peak_qfrc_mean": pkq_mean.tolist(),
    "peak_qfrc_max": PKQ.amax(0).tolist(),
    "peak_force_mean": float(PKF.mean()),
    "peak_force_max": float(PKF.amax()),
    "delivered_mean": float(DEL.mean()),
    "delivered_max": float(DEL.amax()),
    "dur_substeps_mean": float(DUR.mean()),
    "dur_ms_mean": float(DUR.mean() * dt * 1000.0),
    "dur_substeps_min": float(DUR.min()),
    "dur_substeps_max": float(DUR.max()),
    "per_strike": {
      "height": strike_heights,
      "raw": RAW.tolist(),
      "sub": SUB.tolist(),
      "rows": ROWS.tolist(),
      "delivered": DEL.tolist(),
      "peak_force": PKF.tolist(),
      "peak_qfrc": PKQ.tolist(),
      "dur_substeps": DUR.tolist(),
    },
    "integrity": {
      "shipped_leak": float(shipped_leak),
      "ximp_err_max": float(max(ximp_err)),
      "xdel_err_max": float(max(xdel_err)),
    },
  }
  with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
  print(f"\nJSON written: {args.out}")


if __name__ == "__main__":
  main()
