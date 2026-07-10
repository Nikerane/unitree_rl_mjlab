"""Impulse-CaT force-propagation diagnostics: the impulse analogue of diag_policy_trace.py.

Drives a SINGLE strike -- either the open-loop scripted reference (``--reference``, the Task-5
playback idiom) or a trained checkpoint (``--ckpt``, loaded exactly like diag_policy_trace.py) --
against a ``cat_impulse=True`` task (default ``Unitree-Z1-Hammer-CaT-Impulse``, which wires BOTH
the ``SubstepImpulseAccumulator`` / ``SubstepDeliveredImpulse`` accumulators AND the ``CatSoftHook``)
and records, at the true 500 Hz SUBSTEP rate:

  1. per-joint |qfrc_constraint_j|  -- the raw reaction, propagating down the arm chain
  2. object-side F_axial            -- the delivered contact force (weld/friction-immune)
  3. the SHIPPED accumulator's running per-joint Λ_j (``acc.impulse``, per-event-pulse semantics)
  4. δ  -- env.extras["cat_delta"], written once per CONTROL step by CatSoftHook and naturally
     forward-filled across this step's substeps (env.extras is not cleared between steps; see
     manager_based_rl_env.py -- metrics_manager.compute_substep() runs INSIDE the decimation loop,
     metrics_manager.compute() -- which calls CatSoftHook -- runs once AFTER it)
  5. per-joint |q̇_j|

via a ``metrics_manager.compute_substep`` monkeypatch, hooked AFTER the shipped accumulators run
(mirrors ``derive_impulse_thresholds.py``:100-111 verbatim: patched() calls orig_substep() first).

Output: a 5-panel matplotlib figure (``trace.png``) plus the raw substep arrays (``trace.npz``).

Usage:
  python scripts/diag_impulse_trace.py --reference --out /tmp/impulse_trace_ref \
      --j-limit 1.640,3.280,1.640,1.640,1.640,1.640
  python scripts/diag_impulse_trace.py --ckpt /tmp/ckpt/c2/model_499.pt --out /tmp/impulse_trace_ckpt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

import mjlab.tasks  # noqa: F401  (register builtin tasks)
import src.tasks  # noqa: F401  (register hammer tasks)
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR, _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
HOLD_STEPS = 6  # post-playback settle steps -- mirrors derive_impulse_thresholds.py / reward_design_util.py


def _install_substep_hook(env: ManagerBasedRlEnv, env_idx: int) -> list[tuple]:
  """Wrap ``env.metrics_manager.compute_substep`` to record one row per physics substep.

  Pattern reused VERBATIM from derive_impulse_thresholds.py:100-111: patched() calls
  orig_substep() FIRST, so this hook's snapshot of the shipped accumulators (acc_shipped.impulse,
  dacc_shipped.delivered) always includes the substep that just ran -- no alignment tolerance to
  hide an off-by-one behind.
  """
  robot = env.scene["robot"]
  contact = env.scene["hammer_nail_contact"]
  netf = env.scene["hammer_nail_impulse"]
  arm = SceneEntityCfg("robot", joint_names=ARM)
  arm.resolve(env.scene)
  jid = arm.joint_ids
  axis = torch.tensor([0.0, 0.0, -1.0], device=env.device)

  acc_shipped = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
  dacc_shipped = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, None)
  if acc_shipped is None or dacc_shipped is None:
    raise RuntimeError(
      "diag_impulse_trace requires BOTH the SubstepImpulseAccumulator and SubstepDeliveredImpulse "
      "metrics (cat_impulse=True) -- use --task Unitree-Z1-Hammer-CaT-Impulse (the default) or "
      "another task built with cat_impulse=True."
    )

  rec: list[tuple] = []
  orig_substep = env.metrics_manager.compute_substep

  def patched() -> None:
    orig_substep()
    qfrc = robot.data._joint_dof_field("qfrc_constraint")[env_idx, jid].abs().clone()  # (6,)
    in_c = bool((contact.data.found[env_idx] > 0).any())
    f = netf.data.force  # (B, N, 3) world-frame net contact force per primary
    f_ax = float((f * axis).sum(-1).sum(-1).clamp_min(0.0)[env_idx])  # net downward axial force
    imp = acc_shipped.impulse[env_idx].clone()  # (6,) running per-event-pulse Λ_j
    deliv = float(dacc_shipped.delivered[env_idx])  # episode-cumulative delivered impulse
    qv = robot.data.joint_vel[env_idx, jid].abs().clone()  # (6,)
    delta = env.extras.get("cat_delta")
    d = float(delta[env_idx]) if delta is not None else 0.0
    rec.append((qfrc.numpy(), f_ax, imp.numpy(), deliv, in_c, qv.numpy(), d))

  env.metrics_manager.compute_substep = patched  # type: ignore[method-assign]
  return rec


def _run_reference(args: argparse.Namespace) -> tuple[list[tuple], float]:
  """Open-loop scripted single strike (Task-5 idiom: reward_design_util.run_reference_strikes)."""
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  # auto_reset=False: a strike that fires the nail_driven success termination is otherwise reset
  # IN-STEP, zeroing the accumulators before we get to read the terminal strike's Λ_j (the same
  # 2026-07 review finding documented in impulse_bound.py / reward_design_util.py).
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  rec = _install_substep_hook(env, args.env_idx)

  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  rcfg.resolve(env.scene)
  ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  ncfg.resolve(env.scene)

  def head() -> torch.Tensor:
    return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)

  def nail_top() -> torch.Tensor:
    return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

  env.reset()
  ref = SingleStrikeReference(args.num_envs, env.device, approach_height=args.approach_height)
  ref.update(head(), nail_top(), torch.zeros(args.num_envs, dtype=torch.long, device=env.device))
  n = ref.playback_length()
  for k in range(1, n + HOLD_STEPS + 1):
    target = ref.playback_target(min(k, n))
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    env.step(action)
    if bool(env.reset_terminated.any()):  # success fired; auto_reset=False keeps state readable
      break
  return rec, float(env.physics_dt)


def _run_ckpt(args: argparse.Namespace) -> tuple[list[tuple], float]:
  """Roll out a trained checkpoint (loaded exactly like diag_policy_trace.py)."""
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task, play=args.play)
  env_cfg.scene.num_envs = args.num_envs
  # auto_reset=False (Task 6 deferred fix, 2026-07 review finding): with the default auto_reset=True
  # a success mid-trace resets the env IN-STEP (_reset_idx runs before env.step() returns), zeroing
  # the accumulators/nail state and silently splicing a fresh episode into what is meant to be ONE
  # continuous single-strike trace. Mirrors the SAME idiom already used by _run_reference above /
  # reward_design_util.run_reference_strikes: auto_reset=False + break on the first success keeps the
  # terminal state readable and the trace honest.
  env_cfg.auto_reset = False
  agent_cfg = load_rl_cfg(args.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  rec = _install_substep_hook(env, args.env_idx)  # hook the raw env BEFORE wrapping
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  obs, _ = wrapped.reset()
  for _ in range(args.nsteps):
    with torch.no_grad():
      actions = policy(obs)
    obs, rew, dones, extras = wrapped.step(actions)
    # Break on success (reset_terminated) OR timeout (reset_time_outs): auto_reset=False keeps the
    # terminal state readable, but it also arms _manual_reset_pending for ANY done env -- one more
    # step() after a timeout (possible without --play: training cfg episode_length_s=20 s = 1000
    # control steps) would raise mjlab's manual-reset RuntimeError. Either way the episode is over;
    # tracing past it would splice a stale terminal state into the figure.
    if bool((env.reset_terminated | env.reset_time_outs).any()):
      break
  return rec, float(env.physics_dt)


def _make_plot(rec: list[tuple], dt: float, j_limit: list[float] | None, out_dir: Path) -> None:
  n = len(rec)
  if n == 0:
    raise RuntimeError("diag_impulse_trace: no substeps were recorded (empty rollout).")
  t = np.arange(n) * dt
  qfrc = np.stack([r[0] for r in rec])       # (n, 6)
  f_axial = np.array([r[1] for r in rec])    # (n,)
  impulse = np.stack([r[2] for r in rec])    # (n, 6) running Λ_j
  delivered = np.array([r[3] for r in rec])  # (n,)
  contact = np.array([r[4] for r in rec], dtype=bool)  # (n,)
  qv = np.stack([r[5] for r in rec])         # (n, 6)
  delta = np.array([r[6] for r in rec])      # (n,)

  labels = [f"joint{k + 1}" for k in range(6)]
  fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)

  ax = axes[0]
  for k in range(6):
    ax.plot(t, qfrc[:, k], label=labels[k])
  ax.set_ylabel("|qfrc_constraint|\n(N·m)")
  ax.set_title("(1) Per-joint reaction force -- chain propagation")
  ax.legend(loc="upper right", fontsize=7, ncol=3)

  ax = axes[1]
  ax.plot(t, f_axial, color="black")
  ax.set_ylabel("F_axial (N)")
  ax.set_title("(2) Object-side axial contact force (delivered, weld/friction-immune)")

  ax = axes[2]
  for k in range(6):
    ax.plot(t, impulse[:, k], label=labels[k])
  if j_limit is not None:
    for k in range(6):
      ax.axhline(j_limit[k], linestyle="--", color=f"C{k}", alpha=0.6, linewidth=1)
  ax.set_ylabel("Λ_j (N·m·s)")
  ax.set_title("(3) Running per-joint impulse (per-event pulse) vs J_limit" + (
    "" if j_limit is not None else "  [--j-limit not given: no cap lines]"
  ))
  ax.legend(loc="upper right", fontsize=7, ncol=3)

  ax = axes[3]
  ax.plot(t, delta, color="crimson")
  ax.set_ylabel("δ")
  ax.set_title("(4) CaT soft-violation probability (control-rate, forward-filled)")

  ax = axes[4]
  for k in range(6):
    ax.plot(t, qv[:, k], label=labels[k])
  ax.set_ylabel("|q̇_j| (rad/s)")
  ax.set_xlabel("time (s)")
  ax.set_title("(5) Per-joint speed")
  ax.legend(loc="upper right", fontsize=7, ncol=3)

  for ax in axes:  # shade contact windows on every panel
    ylo, yhi = ax.get_ylim()
    ax.fill_between(t, ylo, yhi, where=contact, color="gray", alpha=0.12, step="mid")
    ax.set_ylim(ylo, yhi)

  fig.tight_layout()
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / "trace.png", dpi=150)
  plt.close(fig)

  np.savez(
    out_dir / "trace.npz",
    t=t,
    qfrc=qfrc,
    f_axial=f_axial,
    impulse=impulse,
    delivered=delivered,
    contact=contact,
    qv=qv,
    delta=delta,
    j_limit=np.asarray(j_limit, dtype=np.float64) if j_limit is not None else np.array([]),
  )
  print(f"[diag_impulse_trace] wrote {out_dir / 'trace.png'} and {out_dir / 'trace.npz'} "
        f"({n} substeps, {n * dt:.3f}s, peak Λ_j={impulse.max(axis=0)})")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default=None, help="trained checkpoint (.pt); XOR with --reference")
  ap.add_argument("--reference", action="store_true", help="open-loop scripted single strike; XOR with --ckpt")
  ap.add_argument("--task", default="Unitree-Z1-Hammer-CaT-Impulse",
                  help="must be a cat_impulse=True task (both accumulators + the hook wired)")
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--env-idx", type=int, default=0)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--out", default="/tmp/impulse_trace")
  ap.add_argument("--j-limit", default=None,
                  help="CSV of 6 floats (N·m·s), e.g. 1.640,3.280,1.640,1.640,1.640,1.640 -- "
                       "REQUIRED to draw the J_limit cap lines in panel 3. NEVER defaulted "
                       "(the placeholder Z1_JOINT_IMPULSE_LIMIT=0.1 is not a real cap); omitting "
                       "this flag draws no cap lines.")
  ap.add_argument("--approach-height", type=float, default=0.10, help="--reference mode: strike apex height (m)")
  ap.add_argument("--nsteps", type=int, default=80, help="--ckpt mode: control steps to roll out")
  ap.add_argument("--play", action="store_true", help="play-mode env cfg (zeroed reset/obs noise)")
  args = ap.parse_args()

  if bool(args.ckpt) == bool(args.reference):
    raise SystemExit("diag_impulse_trace: pass exactly one of --ckpt or --reference.")

  j_limit: list[float] | None = None
  if args.j_limit:
    j_limit = [float(x) for x in args.j_limit.split(",")]
    if len(j_limit) != 6:
      raise SystemExit(f"--j-limit must be exactly 6 comma-separated floats, got {len(j_limit)}: {args.j_limit}")

  rec, dt = _run_reference(args) if args.reference else _run_ckpt(args)
  _make_plot(rec, dt, j_limit, Path(args.out))


if __name__ == "__main__":
  main()
