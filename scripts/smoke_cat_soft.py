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
from collections.abc import Iterator, Mapping
import json
from pathlib import Path
import tempfile
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.hammer.config.z1  # noqa: F401  (registers tasks)
from src.tasks.hammer.rl.cat_ppo import CatPPO
from src.tasks.hammer.rl.cat_storage import CatRolloutStorage
from src.tasks.hammer.config.z1.joint_position_contract import JOINT_NAMES

TASK = "Unitree-Z1-Hammer-CaT-Soft"
FIC0_TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
  "CProgress-Vel-Delivered4-JointPosition-Fixed"
)
FICTT_TASK = f"{FIC0_TASK}-TT"
DIRECT_FIC0_TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-Fixed"
)
DIRECT_FICTT_TASK = f"{DIRECT_FIC0_TASK}-TT"
VIC_TT_TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT"
)
JOINT_TASK_OBSERVATION_WIDTHS = {
  FIC0_TASK: 47,
  FICTT_TASK: 47,
  DIRECT_FIC0_TASK: 40,
  DIRECT_FICTT_TASK: 40,
  VIC_TT_TASK: 40,
}
JOINT_TASK_ACTION_WIDTHS = {
  task: 12 if task == VIC_TT_TASK else 6
  for task in JOINT_TASK_OBSERVATION_WIDTHS
}
TASKS = (TASK, *JOINT_TASK_OBSERVATION_WIDTHS)


def _tensors(value: object) -> Iterator[torch.Tensor]:
  if isinstance(value, torch.Tensor):
    yield value
  elif isinstance(value, Mapping):
    for item in value.values():
      yield from _tensors(item)
  elif isinstance(value, (list, tuple)):
    for item in value:
      yield from _tensors(item)


def _assert_finite_tensors(value: object, *, state_name: str) -> None:
  tensors = list(_tensors(value))
  assert tensors, f"{state_name} contains no tensors"
  nonfinite = [tuple(tensor.shape) for tensor in tensors if not torch.isfinite(tensor).all()]
  assert not nonfinite, f"{state_name} contains non-finite tensors with shapes {nonfinite}"


def _assert_checkpoint_rollout_telemetry(
  task: str,
  checkpoint_state: Mapping[str, object],
  *,
  expected_sample_count: int,
) -> None:
  """Require the compact qualified rollout record for the VIC smoke only."""
  if task != VIC_TT_TASK:
    return
  infos = checkpoint_state.get("infos")
  assert isinstance(infos, Mapping), "VIC checkpoint telemetry infos are missing"
  telemetry = infos.get("vic_rollout_telemetry")
  assert isinstance(telemetry, Mapping), "VIC checkpoint telemetry record is missing"
  expected_keys = {
    "schema_version",
    "source",
    "action_terms",
    "joint_names",
    "gain_action_indices",
    "raw_action_clip",
    "sample_count",
    "deterministic_gaussian_mean",
    "sampled_action",
    "gaussian_exploration_std",
  }
  assert set(telemetry) == expected_keys, "VIC checkpoint telemetry schema drifted"
  assert telemetry["schema_version"] == 1
  assert telemetry["source"] == "cat_rollout_storage"
  assert telemetry["action_terms"] == ["joint_position", "joint_stiffness"]
  assert telemetry["joint_names"] == list(JOINT_NAMES)
  assert telemetry["gain_action_indices"] == [6, 7, 8, 9, 10, 11], (
    "VIC checkpoint gain action indices drifted"
  )
  assert telemetry["raw_action_clip"] == 1.0
  assert telemetry["sample_count"] == expected_sample_count
  summary_keys = {
    "mean",
    "std",
    "minimum",
    "p05",
    "median",
    "p95",
    "maximum",
    "lower_bound_occupancy",
    "upper_bound_occupancy",
  }
  for source_name in ("deterministic_gaussian_mean", "sampled_action"):
    source = telemetry[source_name]
    assert isinstance(source, Mapping) and set(source) == {"raw", "clipped"}
    for form in ("raw", "clipped"):
      summary = source[form]
      assert isinstance(summary, Mapping) and set(summary) == summary_keys
      assert all(
        isinstance(summary[key], list) and len(summary[key]) == len(JOINT_NAMES)
        for key in summary_keys
      )
  exploration_std = telemetry["gaussian_exploration_std"]
  assert isinstance(exploration_std, list) and len(exploration_std) == len(JOINT_NAMES)
  assert all(value > 0.0 for value in exploration_std)
  json.dumps(telemetry, allow_nan=False, sort_keys=True)


def run_smoke(
  task: str = TASK,
  device: str = "cpu",
  num_envs: int = 16,
  iters: int = 3,
) -> Path:
  """Run the registered CatPPO rollout/update path and return its temp checkpoint."""
  if task not in TASKS:
    raise ValueError(f"unsupported CatPPO smoke task: {task}")
  if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
    raise ValueError("num_envs must be a positive integer")
  if isinstance(iters, bool) or not isinstance(iters, int) or iters <= 0:
    raise ValueError("iters must be a positive integer")

  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = num_envs
  agent = load_rl_cfg(task)
  agent.max_iterations = iters
  agent.logger = "tensorboard"  # avoid wandb network/prompt

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  try:
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent.clip_actions)
    runner_cls = load_runner_cls(task)
    log_dir = tempfile.mkdtemp(prefix="cat_smoke_")
    runner_cfg = asdict(agent)
    runner_cfg["upload_model"] = False
    runner = runner_cls(env, runner_cfg, log_dir, device)

    # construct_algorithm wiring (the one untested-on-CPU path)
    assert isinstance(runner.alg, CatPPO), f"runner.alg is {type(runner.alg).__name__}, expected CatPPO"
    assert isinstance(runner.alg.storage, CatRolloutStorage), (
      f"storage is {type(runner.alg.storage).__name__}, expected CatRolloutStorage"
    )
    assert runner.alg.storage.soft_dones.dtype.is_floating_point, "soft_dones must be float"
    assert runner_cfg["num_steps_per_env"] == 24, "CatPPO smoke must use the registered 24-step rollout"
    assert runner.alg.storage.num_transitions_per_env == 24, "storage must hold the real 24-step rollout"

    observations = env.get_observations()
    requested_device = torch.empty(0, device=device).device
    actual_devices = {value.device for value in observations.values()}
    actual_devices.add(runner.alg.storage.actions.device)
    assert actual_devices == {requested_device}, (
      f"requested device {requested_device}, got tensors on {sorted(map(str, actual_devices))}"
    )
    if task in JOINT_TASK_OBSERVATION_WIDTHS:
      observation_width = JOINT_TASK_OBSERVATION_WIDTHS[task]
      action_width = JOINT_TASK_ACTION_WIDTHS[task]
      assert env.num_actions == action_width, (
        f"joint task has {env.num_actions} actions, expected {action_width}"
      )
      assert runner.alg.storage.actions.shape[-1] == action_width, (
        "CatPPO storage action width does not match the live joint task: "
        f"{runner.alg.storage.actions.shape[-1]} != {action_width}"
      )
      assert tuple(observations["actor"].shape) == (num_envs, observation_width)
      assert tuple(observations["critic"].shape) == (num_envs, observation_width)

    print(
      f"[smoke] {task}: CatPPO + CatRolloutStorage wired; "
      f"running {iters} update(s) on {device} ..."
    )
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)

    _assert_finite_tensors(runner.alg.save(), state_name="learned CatPPO state")
    checkpoints = sorted(Path(log_dir).glob("model_*.pt"))
    assert checkpoints, f"runner wrote no checkpoint under temporary log path {log_dir}"
    checkpoint = checkpoints[-1]
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _assert_finite_tensors(checkpoint_state, state_name=f"checkpoint {checkpoint}")
    _assert_checkpoint_rollout_telemetry(
      task,
      checkpoint_state,
      expected_sample_count=runner_cfg["num_steps_per_env"] * num_envs,
    )
  finally:
    raw_env.close()

  print(
    f"\nSMOKE_CAT_SOFT: learn() completed {iters} update(s); "
    f"learned/checkpoint tensors finite (checkpoint={checkpoint})"
  )
  return checkpoint


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", choices=TASKS, default=TASK)
  ap.add_argument("--num-envs", type=int, default=16)
  ap.add_argument("--iters", type=int, default=3)
  ap.add_argument("--device", default="cpu")
  args = ap.parse_args()
  run_smoke(
    task=args.task,
    device=args.device,
    num_envs=args.num_envs,
    iters=args.iters,
  )


if __name__ == "__main__":
  main()
