"""C3 checkpoint eval: per-joint impulse Λ peaks/p95 + delivered impulse summary (the C4 analogue).

The rollout collector + aggregator behind ``scripts/eval_impulse.sh`` (Task 11,
``.superpowers/sdd/task-11-brief.md``). Rolls a SINGLE checkpoint in the SAME instrumented env
(default ``Unitree-Z1-Hammer-CaT-Impulse``, play cfg, ``imp_max_p`` forced to 0 -- log-only / pure
instrumentation) regardless of which of the four C3 arms trained it -- mirrors the archived June
velocity eval's (``docs/archive/tooling/eval_peak_qv.sh``) same-env cross-arm protocol, extended
to the impulse metrics so the non-impulse
arms (``c3_track``, ``c3_catsoft``) get the SAME per-joint Λ tail extracted as ``c3_imp``/``c3_imp0``.
Training-logged ``Episode_Metrics/*`` cannot serve this purpose: they are env-MEANS (this needs the
tail: max/p95 across episodes and envs), and the non-impulse arms never wire the impulse metrics in
their own training env at all.

Per checkpoint, extracts ONE summary.csv row:
  per-joint Λ max & p95 (6+6 cols, from the ``SubstepImpulseAccumulator``'s (B,6) episode-peak
  buffer), worst-joint Λ_j/J_limit_j (max + mean ratio, IMP_J_LIMIT imported from env_cfgs -- never
  hardcoded), delivered impulse mean+-std, success rate, nail depth mean+-std, episode length
  mean+-std, episode count -- for a MEAN-ACTION rollout (primary, the row's numbers) plus a condensed
  SAMPLED-ACTION repeat (robustness check: success/worst-ratio/delivered/n_episodes, ``_sampled``
  suffix) -- appended to ``<out>/summary.csv`` with provenance columns (host/timestamp/git hash, plus
  the effective ``imp_max_p`` so a non-default override is distinguishable from the pinned 0).

Auto-reset correctness (Task 6 deferred fix -- the SAME bug class fixed in diag_impulse_trace.py's
--ckpt mode, one level up): ``metrics_manager.compute()`` (full-step) runs BEFORE the in-step
auto-reset (``_reset_idx``) that mjlab's default ``auto_reset=True`` performs on envs that just
terminated/timed-out -- so reading the shipped accumulators' buffers AFTER ``env.step()`` returns is
too late for an env that just finished an episode (its Λ_j/delivered/depth are already zeroed by
``.reset()``). This script installs a hook on ``metrics_manager.compute`` (mirrors
``diag_impulse_trace.py``'s ``compute_substep`` hook, at full-step granularity) that snapshots the
PRE-reset per-env buffers every control step -- so reading the snapshot right after ``wrapped.step()``
returns is always correct, and the standard ``auto_reset=True`` + ``RslRlVecEnvWrapper`` path can be
used untouched (no manual per-env reset bookkeeping, no ``RuntimeError`` risk from manual resets).

Episode boundary: play cfg disables the timeout (``episode_length_s`` -> ~1e9) so an undertrained
policy that never strikes would never produce a completed episode. ``--episode-len-s`` re-finitizes
the horizon (default 4.0 s = 200 control steps at 50 Hz) so EVERY episode is bounded regardless of
success -- a converged policy finishes long before the cap (single strikes run ~8-15 control steps),
while an undertrained/smoke checkpoint still yields bounded, success=0 rows instead of zero rows.
``--nsteps`` default (400) is >= 2x the episode cap, so even the pathological all-timeout case still
yields >=2 episodes/env (the brief's floor), and a converged policy yields far more.

Usage (single checkpoint; scripts/eval_impulse.sh loops the checkpoint list and calls this once per
checkpoint, appending one row each time):
  python scripts/eval_impulse.py --ckpt logs/rsl_rl/z1_hammer/.../model_4999.pt --name c3_imp_seed0 \
      --num-envs 256 --nsteps 400 --device cuda:0 --seed 42 --out /tmp/eval_impulse

Pinned protocol (mirrored in scripts/eval_impulse.sh's header -- keep both in sync):
  task=Unitree-Z1-Hammer-CaT-Impulse, play cfg, imp_max_p forced to 0 at eval time (equivalent to
  scripts/train.py's --env.metrics.cat-soft.params.imp-max-p 0), 256 envs, seed 42, mean-action
  rollout primary + one sampled-action repeat, >=512 episodes/policy (production sizing).
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab.tasks  # noqa: F401  (register builtin tasks)
import src.tasks  # noqa: F401  (register hammer tasks)
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR, _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.rewards import clamped_nail_depth

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

FIELDNAMES = [
  "name", "ckpt_path", "task",
  "num_envs", "nsteps", "episode_len_s", "seed", "imp_max_p",
  "n_episodes",
  *[f"lambda_max_{j}" for j in ARM_JOINTS],
  *[f"lambda_p95_{j}" for j in ARM_JOINTS],
  "worst_ratio_max", "worst_ratio_mean",
  "delivered_mean", "delivered_std",
  "success_rate", "nail_depth_mean_mm", "nail_depth_std_mm",
  "ep_len_mean", "ep_len_std",
  # sampled-action repeat (robustness check) -- condensed, NOT a second row (brief: rows=checkpoints)
  "n_episodes_sampled", "success_rate_sampled", "worst_ratio_max_sampled", "delivered_mean_sampled",
  "host", "timestamp_utc", "git_hash",
]


def _install_episode_hook(env: ManagerBasedRlEnv) -> dict:
  """Snapshot the shipped accumulators' PRE-auto-reset buffers once per control step.

  See module docstring "Auto-reset correctness". Hooking ``metrics_manager.compute`` (not
  ``compute_substep``) is deliberate: reading the MANAGER's own per_substep-averaged ``_step_values``
  would re-introduce the substep-dilution ``joint_impulse_peak`` was written to avoid (impulse_bound.py
  docstring); this hook instead reads the LIVE accumulator objects directly (their monotone-max /
  cumulative-sum buffers), exactly like diag_impulse_trace.py's substep hook does one level down.
  """
  acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
  dacc = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, None)
  if acc is None or dacc is None:
    raise RuntimeError(
      "eval_impulse requires BOTH the SubstepImpulseAccumulator and SubstepDeliveredImpulse metrics "
      "(cat_impulse=True) -- use --task Unitree-Z1-Hammer-CaT-Impulse (the default) or another task "
      "built with cat_impulse=True."
    )
  nail_cfg = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
  nail_cfg.resolve(env.scene)

  snap: dict = {"perjoint": None, "delivered": None, "depth": None}
  orig_compute = env.metrics_manager.compute

  def patched() -> None:
    orig_compute()
    snap["perjoint"] = acc._episode_peak_perjoint.detach().clone()  # (B,6) pre-reset
    snap["delivered"] = dacc.delivered.detach().clone()             # (B,) pre-reset
    snap["depth"] = clamped_nail_depth(env, nail_cfg).detach().clone()  # (B,) pre-reset

  env.metrics_manager.compute = patched  # type: ignore[method-assign]
  return snap


def _rollout(
  env_cfg,
  agent_cfg,
  runner_cls,
  ckpt: str,
  device: str,
  nsteps: int,
  seed: int,
  stochastic: bool,
) -> dict:
  """Roll out ONE checkpoint for ``nsteps`` control steps; return per-episode records."""
  # Fresh cfg per rollout: env construction mutates the cfg in place (the registry's load_env_cfg
  # deep-copies for exactly this reason), and this function runs TWICE per checkpoint (mean +
  # sampled) -- without the copy the second env would be built from a construction-mutated cfg.
  env_cfg = copy.deepcopy(env_cfg)
  torch.manual_seed(seed)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  snap = _install_episode_hook(env)  # hook the raw env BEFORE wrapping
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs, _ = wrapped.reset()
  ep_len = torch.zeros(env.num_envs, device=env.device)
  lam_records: list[torch.Tensor] = []
  delivered_records: list[float] = []
  depth_records: list[float] = []
  ep_len_records: list[float] = []
  succ_records: list[bool] = []

  for _ in range(nsteps):
    with torch.no_grad():
      actions = policy(obs, stochastic_output=stochastic)
    obs, rew, dones, extras = wrapped.step(actions)
    del rew, extras
    ep_len += 1.0
    # Exact success signal, set inside env.step() BEFORE the in-step auto-reset and not touched
    # again until the NEXT env.step() call -- same idiom as diag_policy_trace.py / diag_impulse_trace.py.
    succ_mask = env.reset_terminated.clone()
    done_idx = torch.nonzero(dones, as_tuple=False).flatten()
    if done_idx.numel() > 0:
      for i in done_idx.tolist():
        lam_records.append(snap["perjoint"][i].cpu())
        delivered_records.append(float(snap["delivered"][i]))
        depth_records.append(float(snap["depth"][i]))
        ep_len_records.append(float(ep_len[i]))
        succ_records.append(bool(succ_mask[i]))
      ep_len[done_idx] = 0.0

  env.close()
  return {
    "lam": lam_records,
    "delivered": delivered_records,
    "depth": depth_records,
    "ep_len": ep_len_records,
    "succ": succ_records,
  }


def _mean_std(xs: list[float]) -> tuple[float, float]:
  if not xs:
    return 0.0, 0.0
  t = torch.tensor(xs, dtype=torch.float32)
  return float(t.mean()), float(t.std() if len(xs) > 1 else 0.0)


def _worst_ratios(lam_records: list[torch.Tensor], j_limit: torch.Tensor) -> tuple[float, float]:
  """Per-episode worst-joint Λ_j/J_limit_j; returns (max over episodes, mean over episodes)."""
  if not lam_records:
    return 0.0, 0.0
  lam = torch.stack(lam_records)  # (N, 6)
  ratio = lam / j_limit.clamp_min(1e-9)
  worst_per_ep = ratio.amax(dim=1)  # (N,)
  return float(worst_per_ep.max()), float(worst_per_ep.mean())


def _git_hash(repo_root: Path) -> str:
  try:
    out = subprocess.run(
      ["git", "rev-parse", "--short", "HEAD"],
      cwd=repo_root, capture_output=True, text=True, timeout=5, check=True,
    )
    return out.stdout.strip()
  except Exception:
    return "unknown"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Unitree-Z1-Hammer-CaT-Impulse",
                  help="must be a cat_impulse=True task (both accumulators + the hook wired); ALL "
                       "checkpoints are evaluated in this SAME env regardless of training arm "
                       "(same-env cross-arm protocol -- obs/action space is shared across the hammer "
                       "arms; only reward/termination/metrics wiring differs)")
  ap.add_argument("--ckpt", required=True, help="checkpoint .pt path")
  ap.add_argument("--name", default=None, help="row label; defaults to the checkpoint's parent dir name")
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--nsteps", type=int, default=400,
                  help="control steps for the mean-action (primary) rollout; the sampled-action "
                       "repeat reuses this budget unless --sampled-nsteps overrides it")
  ap.add_argument("--sampled-nsteps", type=int, default=None,
                  help="control steps for the sampled-action robustness repeat (default: same as --nsteps)")
  ap.add_argument("--episode-len-s", type=float, default=4.0,
                  help="finite eval episode horizon (s), overriding the play cfg's near-infinite "
                       "default -- otherwise an undertrained policy that never strikes never yields a "
                       "completed episode. 4.0s = 200 control steps @ 50 Hz; converged single strikes "
                       "finish in ~8-15 steps, so this only bounds failure-mode episodes.")
  ap.add_argument("--imp-max-p", type=float, default=0.0,
                  help="forced eval-time value (default 0 -- log-only/pure instrumentation, "
                       "equivalent to --env.metrics.cat-soft.params.imp-max-p 0 on scripts/train.py)")
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out", default="/tmp/eval_impulse", help="directory for summary.csv (appended)")
  ap.add_argument("--csv-name", default="summary.csv")
  args = ap.parse_args()

  name = args.name or Path(args.ckpt).resolve().parent.name

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.episode_length_s = args.episode_len_s
  if "cat_soft" not in env_cfg.metrics:
    raise RuntimeError(f"--task {args.task} has no 'cat_soft' metric -- not a cat_impulse=True task.")
  env_cfg.metrics["cat_soft"].params["imp_max_p"] = args.imp_max_p
  agent_cfg = load_rl_cfg(args.task)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner

  j_limit = torch.tensor(IMP_J_LIMIT, dtype=torch.float32)
  assert j_limit.numel() == len(ARM_JOINTS), f"IMP_J_LIMIT must have {len(ARM_JOINTS)} entries"

  print(f"[eval_impulse] name={name} ckpt={args.ckpt} task={args.task} "
        f"num_envs={args.num_envs} nsteps={args.nsteps} episode_len_s={args.episode_len_s} "
        f"seed={args.seed} device={args.device} imp_max_p={args.imp_max_p}")

  # --- Primary: mean-action rollout ---
  mean_rec = _rollout(env_cfg, agent_cfg, runner_cls, args.ckpt, args.device,
                       args.nsteps, args.seed, stochastic=False)
  n_ep = len(mean_rec["ep_len"])
  if n_ep == 0:
    print(f"[eval_impulse] ERROR: ZERO completed episodes in the mean-action rollout for '{name}' "
          f"(num_envs={args.num_envs}, nsteps={args.nsteps}, episode_len_s={args.episode_len_s}) -- "
          f"NO row written. Keep nsteps >= 2 x episode_len_s x 50 control steps so even a "
          f"never-succeeding checkpoint times out into >=2 episodes/env (see the "
          f"scripts/eval_impulse.sh header).", file=sys.stderr)
    raise SystemExit(1)
  if n_ep < 2 * args.num_envs:
    print(f"[eval_impulse] WARNING: n_episodes={n_ep} < {2 * args.num_envs} -- below the pinned "
          f">=2 episodes/env floor for num_envs={args.num_envs}. The row is still written "
          f"(n_episodes is a column), but its tail statistics are under-populated vs the protocol.",
          file=sys.stderr)
  lam_max = [0.0] * len(ARM_JOINTS)
  lam_p95 = [0.0] * len(ARM_JOINTS)
  if n_ep > 0:
    lam_stack = torch.stack(mean_rec["lam"])  # (N, 6)
    lam_max = lam_stack.amax(dim=0).tolist()
    lam_p95 = torch.quantile(lam_stack, 0.95, dim=0).tolist()
  worst_max, worst_mean = _worst_ratios(mean_rec["lam"], j_limit)
  delivered_mean, delivered_std = _mean_std(mean_rec["delivered"])
  depth_mean, depth_std = _mean_std([d * 1000.0 for d in mean_rec["depth"]])
  ep_len_mean, ep_len_std = _mean_std(mean_rec["ep_len"])
  success_rate = (sum(mean_rec["succ"]) / n_ep) if n_ep > 0 else 0.0
  print(f"[eval_impulse] mean-action: n_episodes={n_ep} success_rate={success_rate:.3f} "
        f"worst_ratio_max={worst_max:.4f} delivered_mean={delivered_mean:.4f} "
        f"lambda_max={['%.4f' % v for v in lam_max]}")

  # --- Robustness repeat: sampled-action rollout (condensed stats, same row) ---
  sampled_nsteps = args.sampled_nsteps if args.sampled_nsteps is not None else args.nsteps
  sampled_rec = _rollout(env_cfg, agent_cfg, runner_cls, args.ckpt, args.device,
                          sampled_nsteps, args.seed + 1, stochastic=True)
  n_ep_s = len(sampled_rec["ep_len"])
  worst_max_s, _ = _worst_ratios(sampled_rec["lam"], j_limit)
  delivered_mean_s, _ = _mean_std(sampled_rec["delivered"])
  success_rate_s = (sum(sampled_rec["succ"]) / n_ep_s) if n_ep_s > 0 else 0.0
  print(f"[eval_impulse] sampled-action (robustness): n_episodes={n_ep_s} "
        f"success_rate={success_rate_s:.3f} worst_ratio_max={worst_max_s:.4f} "
        f"delivered_mean={delivered_mean_s:.4f}")

  row = {
    "name": name,
    "ckpt_path": str(Path(args.ckpt).resolve()),
    "task": args.task,
    "num_envs": args.num_envs,
    "nsteps": args.nsteps,
    "episode_len_s": args.episode_len_s,
    "seed": args.seed,
    "imp_max_p": args.imp_max_p,
    "n_episodes": n_ep,
    **{f"lambda_max_{j}": lam_max[k] for k, j in enumerate(ARM_JOINTS)},
    **{f"lambda_p95_{j}": lam_p95[k] for k, j in enumerate(ARM_JOINTS)},
    "worst_ratio_max": worst_max,
    "worst_ratio_mean": worst_mean,
    "delivered_mean": delivered_mean,
    "delivered_std": delivered_std,
    "success_rate": success_rate,
    "nail_depth_mean_mm": depth_mean,
    "nail_depth_std_mm": depth_std,
    "ep_len_mean": ep_len_mean,
    "ep_len_std": ep_len_std,
    "n_episodes_sampled": n_ep_s,
    "success_rate_sampled": success_rate_s,
    "worst_ratio_max_sampled": worst_max_s,
    "delivered_mean_sampled": delivered_mean_s,
    "host": socket.gethostname(),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_hash": _git_hash(Path(__file__).resolve().parents[1]),
  }

  # Finiteness guard: a silent NaN/inf in the thesis CSV is exactly the failure mode to prevent --
  # refuse the whole row loudly instead of appending a poisoned value.
  nonfinite = {
    k: v for k, v in row.items()
    if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(v)
  }
  if nonfinite:
    print(f"[eval_impulse] ERROR: non-finite value(s) in the row for '{name}': {nonfinite} -- "
          f"NO row written.", file=sys.stderr)
    raise SystemExit(1)

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  csv_path = out_dir / args.csv_name
  write_header = not csv_path.exists()
  if not write_header:
    # Schema drift must never misalign rows: an existing file appended to by a DIFFERENT
    # FIELDNAMES vintage would silently shift every column right of the drift point.
    with open(csv_path, newline="") as f:
      existing_header = next(csv.reader(f), None)
    if existing_header != FIELDNAMES:
      print(f"[eval_impulse] ERROR: {csv_path} has a different header schema "
            f"({len(existing_header) if existing_header else 0} cols) than this script's FIELDNAMES "
            f"({len(FIELDNAMES)} cols) -- appending would misalign rows. Re-run with FRESH=1 "
            f"(scripts/eval_impulse.sh) to wipe the output dir, or move the stale CSV aside.",
            file=sys.stderr)
      raise SystemExit(1)
  with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
      writer.writeheader()
    writer.writerow(row)
  print(f"[eval_impulse] appended row to {csv_path}")


if __name__ == "__main__":
  main()
