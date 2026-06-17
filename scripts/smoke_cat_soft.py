"""Tiny CPU smoke-train of the soft-CaT arm — builds the real runner directly (bypassing train.py's
GPU-only launcher) to exercise the one path the unit tests + verify_cat_soft.py do not:
CatPPO.construct_algorithm injecting CatRolloutStorage, then a full rollout -> (1-δ)-discount ->
dual-mask GAE -> PPO update cycle. Asserts the right classes are wired and learn() runs without
crashing or NaN. Not a quality check — just a "the whole pipeline runs" gate before GPU.

Usage:
    python scripts/smoke_cat_soft.py --num-envs 16 --iters 3 --device cpu
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.hammer.config.z1  # noqa: F401  (registers tasks)
from src.tasks.hammer.rl.cat_ppo import CatPPO
from src.tasks.hammer.rl.cat_storage import CatRolloutStorage

TASK = "Unitree-Z1-Hammer-CaT-Soft"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-envs", type=int, default=16)
  ap.add_argument("--iters", type=int, default=3)
  ap.add_argument("--device", default="cpu")
  args = ap.parse_args()

  env_cfg = load_env_cfg(TASK)
  env_cfg.scene.num_envs = args.num_envs
  agent = load_rl_cfg(TASK)
  agent.max_iterations = args.iters
  agent.logger = "tensorboard"  # avoid wandb network/prompt

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner_cls = load_runner_cls(TASK)
  log_dir = tempfile.mkdtemp(prefix="cat_smoke_")
  runner = runner_cls(env, asdict(agent), log_dir, args.device)

  # construct_algorithm wiring (the one untested-on-CPU path)
  assert isinstance(runner.alg, CatPPO), f"runner.alg is {type(runner.alg).__name__}, expected CatPPO"
  assert isinstance(runner.alg.storage, CatRolloutStorage), (
    f"storage is {type(runner.alg.storage).__name__}, expected CatRolloutStorage"
  )
  assert runner.alg.storage.soft_dones.dtype.is_floating_point, "soft_dones must be float"
  print(f"[smoke] CatPPO + CatRolloutStorage wired OK; running {args.iters} iters on {args.device} ...")

  runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
  print(f"\nSMOKE_CAT_SOFT: learn() completed {args.iters} iters with no crash (log_dir={log_dir})")


if __name__ == "__main__":
  main()
