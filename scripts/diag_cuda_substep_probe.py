"""CUDA substep-instrumentation probe (2026-07-14, Lightning smoke zero-Λ triage).

The first-ever GPU run of the impulse eval returned Λ_j ≡ 0.0 and delivered ≡ 0.0 on
EVERY joint of BOTH arms while success_rate=1.0 and nail depth=32 mm — physically
impossible if the instrumentation were live (the nail cannot be driven without contact
reaction). Depth (read at full-step rate) was correct; the two dead signals
(contact-sensor ``found``/``force`` via sensordata, and the raw ``qfrc_constraint``
field) are the only quantities read at SUBSTEP rate, inside the decimation loop,
immediately after ``sim.step()`` — which on CUDA is an ASYNCHRONOUS captured-graph
launch. June's A100 runs prove contact sensing works at CONTROL-step rate
(TB impact_progress ≠ 0), so the suspect is specifically the substep-time read.

This probe discriminates the failure layer in one run: it drives the scripted
reference strike and, at every substep, reads the signal stack THREE ways —
  (a) immediate       (exactly what the shipped accumulators do),
  (b) after torch.cuda.synchronize()  (torch-stream settled),
  (c) after wp.synchronize_device()   (warp-stream settled — full sync),
plus the shipped accumulators' own outputs. Interpretation:
  - (a)=0 but (c)>0 during contact  -> warp<->torch stream race; accumulators need a sync
                                       (or mjlab-level fix) on CUDA.
  - (a)=(b)=(c)=0 during contact, but depth advances -> the FIELDS are dead on CUDA
    (sensordata / qfrc_constraint not populated at substep time) -> mjlab/mujoco_warp issue.
  - all three > 0 and accumulators > 0 -> instrumentation fine; the bug is in the eval
    readout path instead.
Run on the GPU box (and once with --device cpu as control):
    python scripts/diag_cuda_substep_probe.py --device cuda:0
    python scripts/diag_cuda_substep_probe.py --device cpu
"""

from __future__ import annotations

import argparse

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.impulse_bound import (
  _ENV_SUBSTEP_DELIVERED_ATTR,
  _ENV_SUBSTEP_IMPULSE_ATTR,
)
from src.tasks.hammer.mdp.references import SingleStrikeReference

try:
  import warp as wp
except Exception:  # pragma: no cover - warp is always present with mjlab
  wp = None


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--hold-steps", type=int, default=10)
  args = ap.parse_args()
  dev = args.device

  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1
  cfg.auto_reset = False  # keep the terminal strike's state readable
  env = ManagerBasedRlEnv(cfg, device=dev)
  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  contact = env.scene["hammer_nail_contact"]
  netf = env.scene["hammer_nail_impulse"]
  acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)
  dacc = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR)

  rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  rcfg.resolve(env.scene)
  ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  ncfg.resolve(env.scene)
  arm = SceneEntityCfg("robot", joint_names=tuple(f"joint{i}" for i in range(1, 7)))
  arm.resolve(env.scene)
  jid = arm.joint_ids

  def head() -> torch.Tensor:
    return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)

  def nail_top() -> torch.Tensor:
    return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

  def read_stack() -> tuple[int, float, float]:
    """(found_any, |F| max, sum|qfrc_arm|) — one immediate read of the signal stack."""
    found = int((contact.data.found > 0).any())
    f = netf.data.force
    fmag = float(f.norm(dim=-1).max())
    q = float(robot.data._joint_dof_field("qfrc_constraint")[0, jid].abs().sum())
    return found, fmag, q

  # Per-substep triple reads via a metrics hook (runs exactly where the shipped
  # accumulators run, AFTER them).
  rows: list[tuple] = []
  orig = env.metrics_manager.compute_substep

  def patched() -> None:
    orig()
    imm = read_stack()
    if dev.startswith("cuda"):
      torch.cuda.synchronize()
    tsync = read_stack()
    if wp is not None and dev.startswith("cuda"):
      wp.synchronize_device()
    wsync = read_stack()
    rows.append((
      imm, tsync, wsync,
      float(acc.impulse.max()), float(dacc.delivered.max()),
      float(nail_e.data.joint_pos[0, 0]),
    ))

  env.metrics_manager.compute_substep = patched  # type: ignore[method-assign]

  env.reset()
  ref = SingleStrikeReference(1, env.device)
  ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
  n = ref.playback_length()
  for k in range(1, n + args.hold_steps + 1):
    target = ref.playback_target(min(k, n))
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    env.step(action)
    if bool(env.reset_terminated.any()):
      break

  n_sub = len(rows)
  c_imm = sum(r[0][0] for r in rows)
  c_tsy = sum(r[1][0] for r in rows)
  c_wsy = sum(r[2][0] for r in rows)
  fmax_imm = max(r[0][1] for r in rows)
  fmax_wsy = max(r[2][1] for r in rows)
  qmax_imm = max(r[0][2] for r in rows)
  qmax_wsy = max(r[2][2] for r in rows)
  acc_final = max(r[3] for r in rows)
  del_final = max(r[4] for r in rows)
  depth_final = max(r[5] for r in rows)

  print(f"\n=== SUBSTEP PROBE ({dev}) — {n_sub} substeps recorded ===")
  print(f"contact substeps  immediate={c_imm}  torch-sync={c_tsy}  warp-sync={c_wsy}")
  print(f"max |F|           immediate={fmax_imm:.3f}  warp-sync={fmax_wsy:.3f} N")
  print(f"max sum|qfrc_arm| immediate={qmax_imm:.3f}  warp-sync={qmax_wsy:.3f} N·m")
  print(f"shipped acc Λ max={acc_final:.4f} N·m·s   delivered max={del_final:.4f} N·s")
  print(f"nail depth max={depth_final * 1000:.1f} mm   terminated={bool(env.reset_terminated.any())}")
  print("\nVERDICT:")
  if depth_final < 0.005:
    print("  strike did not land — probe inconclusive, check the scripted drive first")
  elif c_imm == 0 and c_wsy > 0:
    print("  STREAM RACE: signals appear only after wp.synchronize_device() — the shipped")
    print("  substep accumulators read warp-stream-unsettled memory on CUDA.")
  elif c_wsy == 0 and depth_final > 0.02:
    print("  FIELDS DEAD ON CUDA: nail driven but contact/qfrc read zero even fully synced")
    print("  — sensordata/qfrc_constraint not populated at substep time on this backend.")
  elif acc_final == 0.0 and c_imm > 0:
    print("  ACCUMULATOR BUG: signals live at substep time but the shipped accumulator")
    print("  stayed zero — inspect SubstepImpulseAccumulator on CUDA.")
  else:
    print("  INSTRUMENTATION LIVE at substep rate — the zero-Λ came from the eval readout")
    print("  path, not the accumulators; re-check eval_impulse.py on this device.")


if __name__ == "__main__":
  main()
