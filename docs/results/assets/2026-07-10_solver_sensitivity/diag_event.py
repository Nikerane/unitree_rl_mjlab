"""Whole-event adjudication diagnostic for the sensitivity run.

The main sensitivity_run.py (gate-identical semantics) measures Λ / ∫F·dt over the FIRST
contact window only ("stop at first release"). The audit's momentum-pinning defense is
conditioned on the accumulation window covering the WHOLE event. This diagnostic runs ONE
strike per approach height and records, over the ENTIRE recorded episode (up to the success
reset), the full per-substep time series, so we can separate:
  (a) "impulse genuinely changed" (physical momentum transfer differs under the new params:
      different nail depth, different hammer Δv), from
  (b) "event leaked out of the first window" (re-contact windows carry the missing ∫F·dt).

Per strike it reports: number of contact windows + lengths; first-window vs whole-episode
∫F_axial·dt; per-joint raw Λ (first window vs whole episode); nail depth at end; hammer-head
z-velocity at first touch and at first release; peak F_axial.
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
HOLD_STEPS = 6
SOLREF_GEOM_SUFFIXES = ("nail_head", "hammer_head_0", "hammer_head_1")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", choices=["baseline", "halfdt", "solref2x"], required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1
  assert cfg.sim.mujoco.timestep == 0.002 and cfg.decimation == 10
  if args.config == "halfdt":
    cfg.sim.mujoco.timestep = 0.001
    cfg.decimation = 20
  if args.config == "solref2x":
    def wrap_spec(entity_cfg) -> None:
      orig_fn = entity_cfg.spec_fn

      def patched_spec():
        spec = orig_fn()
        for g in spec.geoms:
          if g.name in SOLREF_GEOM_SUFFIXES:
            old = [float(v) for v in g.solref]
            assert old[0] > 0, (g.name, old)
            new = list(old)
            new[0] = old[0] * 2.0
            g.solref = new
        return spec

      entity_cfg.spec_fn = patched_spec

    wrap_spec(cfg.scene.entities["robot"])
    wrap_spec(cfg.scene.entities["nail_block"])

  env = ManagerBasedRlEnv(cfg, device="cpu")
  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  contact = env.scene["hammer_nail_contact"]
  netf = env.scene["hammer_nail_impulse"]
  rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  arm = SceneEntityCfg("robot", joint_names=ARM)
  nailj = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
  rcfg.resolve(env.scene)
  arm.resolve(env.scene)
  nailj.resolve(env.scene)
  jid = arm.joint_ids
  njid = nailj.joint_ids
  axis = torch.tensor([0.0, 0.0, -1.0])
  dt = env.physics_dt

  def head() -> torch.Tensor:
    return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)

  ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  ncfg.resolve(env.scene)

  def nail_top() -> torch.Tensor:
    return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

  rec: list[dict] = []
  orig_substep = env.metrics_manager.compute_substep

  def patched() -> None:
    orig_substep()
    qfrc = robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone()
    in_c = bool((contact.data.found > 0).any())
    f = netf.data.force
    f_ax = float((f * axis).sum(-1).sum(-1).clamp_min(0.0)[0])
    rec.append({
      "in_c": in_c,
      "qfrc": qfrc,
      "f_ax": f_ax,
      "head_z": float(head()[0, 2]),
      "nail_q": float(nail_e.data.joint_pos[0, njid].reshape(-1)[0]),
    })

  env.metrics_manager.compute_substep = patched  # type: ignore[method-assign]

  strikes = []
  for h in APPROACH_HEIGHTS:
    env.reset()
    ref = SingleStrikeReference(1, env.device, approach_height=h)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()
    rec.clear()
    for k in range(1, n + HOLD_STEPS + 1):
      target = ref.playback_target(min(k, n))
      action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
      env.step(action)
      if int(env.episode_length_buf[0]) == 0:
        break

    # segment contact windows over the WHOLE recorded episode
    windows: list[list[int]] = []
    cur: list[int] = []
    for i, r in enumerate(rec):
      if r["in_c"]:
        cur.append(i)
      elif cur:
        windows.append(cur)
        cur = []
    if cur:
      windows.append(cur)

    total_int = sum(r["f_ax"] * dt for r in rec if r["in_c"])
    first_int = sum(rec[i]["f_ax"] * dt for i in windows[0]) if windows else 0.0
    raw_first = torch.zeros(6)
    raw_total = torch.zeros(6)
    for i, r in enumerate(rec):
      if r["in_c"]:
        raw_total += r["qfrc"].abs() * dt
        if windows and i <= windows[0][-1]:
          raw_first += r["qfrc"].abs() * dt

    # hammer head z-velocity by finite difference around first touch / first release
    v_in = v_out = float("nan")
    if windows:
      i0, i1 = windows[0][0], windows[0][-1]
      if i0 >= 1:
        v_in = (rec[i0]["head_z"] - rec[i0 - 1]["head_z"]) / dt
      if i1 + 2 < len(rec):
        v_out = (rec[i1 + 2]["head_z"] - rec[i1 + 1]["head_z"]) / dt
    strikes.append({
      "height": h,
      "n_windows": len(windows),
      "window_lengths_substeps": [len(w) for w in windows],
      "window_gaps_substeps": [windows[i + 1][0] - windows[i][-1] - 1 for i in range(len(windows) - 1)],
      "first_window_int_Fdt": first_int,
      "total_int_Fdt": total_int,
      "raw_first": raw_first.tolist(),
      "raw_total": raw_total.tolist(),
      "peak_f": max((r["f_ax"] for r in rec), default=0.0),
      "nail_q_final": rec[-1]["nail_q"] if rec else float("nan"),
      "nail_q_max": max((r["nail_q"] for r in rec), default=float("nan")),
      "head_v_at_touch": v_in,
      "head_v_after_release": v_out,
      "n_rec_substeps": len(rec),
    })
    s = strikes[-1]
    print(f"[{args.config}] h={h}: windows={s['n_windows']} lens={s['window_lengths_substeps']} "
          f"gaps={s['window_gaps_substeps']} firstInt={first_int:.4f} totalInt={total_int:.4f} "
          f"peakF={s['peak_f']:.2f} nail_q={s['nail_q_final']:.4f} "
          f"v_touch={v_in:.3f} v_release={v_out:.3f}")

  env.metrics_manager.compute_substep = orig_substep  # type: ignore[method-assign]
  with open(args.out, "w") as fh:
    json.dump({"config": args.config, "physics_dt": dt, "strikes": strikes}, fh, indent=1)
  print(f"JSON written: {args.out}")


if __name__ == "__main__":
  sys.exit(main())
