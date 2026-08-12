"""Lean live-manager gate for the two fixed-impedance joint-policy arms."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _REPO_ROOT_STR]
sys.path.insert(0, _REPO_ROOT_STR)

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import src.tasks.hammer.config.z1  # noqa: F401  (registers tasks)
from src.tasks.hammer.cat.hook import CatSoftHook
from src.tasks.hammer.cat.keys import CAT_R_POS_KEY
from src.tasks.hammer.config.z1.joint_position_contract import JOINT_NAMES
from src.tasks.hammer.mdp.velocity_bound import _ENV_SUBSTEP_ATTR, Z1_JOINT_VEL_LIMIT
from src.tasks.hammer.rl.runner import _get_hammer_metadata


PARENT_TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
  "CProgress-Vel-Delivered4"
)
FIC0_TASK = f"{PARENT_TASK}-JointPosition-Fixed"
FICTT_TASK = f"{FIC0_TASK}-TT"
DIRECT_FIC0_TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-Fixed"
)
DIRECT_FICTT_TASK = f"{DIRECT_FIC0_TASK}-TT"
TASK_CONTRACTS = {
  FIC0_TASK: {"width": 47, "guidance": "waypoint_progress", "tt": False},
  FICTT_TASK: {"width": 47, "guidance": "waypoint_progress", "tt": True},
  DIRECT_FIC0_TASK: {"width": 40, "guidance": "direct_reference", "tt": False},
  DIRECT_FICTT_TASK: {"width": 40, "guidance": "direct_reference", "tt": True},
}
TASKS = tuple(TASK_CONTRACTS)

_DEFAULT_OFFSETS = (
  0.0,
  1.6059999465942383,
  -0.4300999939441681,
  -1.19760000705719,
  -0.0013000000035390258,
  1.5543999671936035,
)
_FIXED_ACTUATOR_SIGNATURE = (
  (
    "BuiltinPositionActuatorCfg",
    ("joint1", "joint3", "joint4", "joint5", "joint6"),
    1000.0,
    100.0,
    30.0,
    0.01,
  ),
  ("BuiltinPositionActuatorCfg", ("joint2",), 1500.0, 150.0, 60.0, 0.02),
  ("BuiltinPositionActuatorCfg", ("jointGripper",), 100.0, 20.0, 30.0, 0.005),
)
_IMPULSE_LIMITS = (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
_FIC_CONTROLLED_DROP_I_REF_N_S = 0.2799950838088989


def _hook_of(env: ManagerBasedRlEnv) -> CatSoftHook:
  manager = env.metrics_manager
  instance = manager._term_cfgs[manager._term_names.index("cat_soft")].func
  if not isinstance(instance, CatSoftHook):
    raise RuntimeError(f"cat_soft resolved to {type(instance)}, not CatSoftHook")
  return instance


def _actuator_signature(env: ManagerBasedRlEnv) -> tuple[tuple[object, ...], ...]:
  return tuple(
    (
      type(actuator).__name__,
      tuple(actuator.target_names_expr),
      float(actuator.stiffness),
      float(actuator.damping),
      float(actuator.effort_limit),
      float(actuator.armature),
    )
    for actuator in env.cfg.scene.entities["robot"].articulation.actuators
  )


def run_checks(
  task: str,
  device: str = "cpu",
  num_envs: int = 8,
  steps: int = 3,
) -> list[tuple[str, bool, str]]:
  """Return ``(name, passed, detail)`` checks from one real registered environment."""
  if task not in TASK_CONTRACTS:
    raise ValueError(f"unsupported fixed joint-position smoke task: {task}")
  if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
    raise ValueError("num_envs must be a positive integer")
  if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
    raise ValueError("steps must be a positive integer")
  contract = TASK_CONTRACTS[task]
  observation_width = contract["width"]
  guidance = contract["guidance"]
  tt_enabled = contract["tt"]
  results: list[tuple[str, bool, str]] = []

  def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))

  cfg = load_env_cfg(task, play=True)
  raw_policy_clip = load_rl_cfg(task).clip_actions
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
  try:
    requested_device = torch.empty(0, device=device).device
    hook = _hook_of(env)
    tracker = getattr(env.unwrapped, _ENV_SUBSTEP_ATTR, None)
    obs, _ = env.reset()

    action = env.action_manager.get_term("joint_position")
    command = torch.full(
      (env.num_envs, env.action_manager.total_action_dim),
      0.25,
      device=env.device,
    )
    observation_tensors = [
      obs["actor"],
      obs["critic"],
    ]
    observation_records = [
      (tuple(value.shape), str(value.device)) for value in observation_tensors
    ]
    observations_are_finite = all(
      tuple(value.shape) == (num_envs, observation_width)
      and bool(torch.isfinite(value).all())
      for value in observation_tensors
    )
    step_outputs_are_finite = True
    actual_devices = {value.device for value in (*observation_tensors, command)}
    for _ in range(steps):
      step_obs, reward, terminated, truncated, _ = env.step(command)
      step_observations = [step_obs["actor"], step_obs["critic"]]
      observation_records.extend(
        (tuple(value.shape), str(value.device)) for value in step_observations
      )
      observations_are_finite = observations_are_finite and all(
        tuple(value.shape) == (num_envs, observation_width)
        and bool(torch.isfinite(value).all())
        for value in step_observations
      )
      step_outputs_are_finite = step_outputs_are_finite and (
        bool(torch.isfinite(reward).all())
        and bool(torch.isfinite(terminated.float()).all())
        and bool(torch.isfinite(truncated.float()).all())
      )
      actual_devices.update(
        value.device
        for value in (*step_observations, reward, terminated, truncated)
      )

    check(
      f"reset and stepped observations are finite (N,{observation_width})",
      observations_are_finite,
      str(observation_records),
    )
    check(
      "step rewards and done flags are finite",
      step_outputs_are_finite,
    )
    check(
      "actual tensor device matches the requested device",
      actual_devices == {requested_device},
      f"requested={requested_device}; actual={sorted(map(str, actual_devices))}",
    )

    check(
      "joint-position action is exactly six canonical arm joints",
      env.action_manager.total_action_dim == 6
      and action.action_dim == 6
      and tuple(action.target_names) == JOINT_NAMES
      and tuple(action.cfg.actuator_names) == JOINT_NAMES,
      f"dim={env.action_manager.total_action_dim}; names={tuple(action.target_names)}",
    )
    expected_offset = torch.tensor(
      _DEFAULT_OFFSETS, dtype=action.offset.dtype, device=action.offset.device
    ).expand_as(action.offset)
    check(
      "absolute action uses the qualified default offsets",
      action.cfg.use_default_offset is True
      and bool(torch.allclose(action.offset, expected_offset, rtol=0.0, atol=1e-7)),
      str(action.offset[0].detach().cpu().tolist()),
    )
    check(
      "fixed gain and effort signature is unchanged",
      _actuator_signature(env) == _FIXED_ACTUATOR_SIGNATURE,
      str(_actuator_signature(env)),
    )
    check(
      "there is no gain action",
      tuple(env.action_manager.active_terms) == ("joint_position",),
      str(env.action_manager.active_terms),
    )

    reward_weights = {
      name: env.reward_manager.get_term_cfg(name).weight
      for name in env.reward_manager.active_terms
    }
    if guidance == "waypoint_progress":
      check(
        "P is exactly 8",
        reward_weights.get("r_waypoint_progress") == 8.0,
        str(reward_weights.get("r_waypoint_progress")),
      )
    else:
      imitation_cfg = env.reward_manager.get_term_cfg("r_imit")
      check(
        "direct-reference reward is exactly r_imit=0.1 with sigma=0.05",
        reward_weights.get("r_imit") == 0.1
        and imitation_cfg.params["sigma"] == 0.05,
        f"weight={reward_weights.get('r_imit')}; "
        f"sigma={imitation_cfg.params.get('sigma')}",
      )
      observation_terms = {
        name
        for terms in env.observation_manager.active_terms.values()
        for name in terms
      }
      check(
        "direct-reference arm has no waypoint state, reward, or tracker",
        not {
          "next_gate_vector",
          "completed_gate_fraction",
          "guideline_perpendicular_error",
          "waypoint_progress_state",
        }
        & observation_terms
        and not {"r_gate", "r_waypoint_progress"} & set(reward_weights)
        and "waypoint_progress" not in env.metrics_manager.active_terms,
        f"observations={sorted(observation_terms)}; "
        f"rewards={tuple(reward_weights)}; metrics={tuple(env.metrics_manager.active_terms)}",
      )
      reset_cfg = env.cfg.events["reset_robot_joints"].params
      check(
        "direct-reference reset position and velocity are fixed",
        reset_cfg["position_range"] == (0.0, 0.0)
        and reset_cfg["velocity_range"] == (0.0, 0.0),
        f"position={reset_cfg['position_range']}; velocity={reset_cfg['velocity_range']}",
      )
    check(
      "D4 is exactly 4",
      reward_weights.get("delivered_impulse") == 4.0,
      str(reward_weights.get("delivered_impulse")),
    )
    delivered_cfg = env.reward_manager.get_term_cfg("delivered_impulse")
    check(
      "delivered impulse uses the corrected FIC-only controlled-drop reference",
      delivered_cfg.params["i_ref"] == _FIC_CONTROLLED_DROP_I_REF_N_S,
      str(delivered_cfg.params["i_ref"]),
    )
    check(
      "contact-row attribution is disabled while production impulse remains active",
      env.cfg.metrics["substep_impulse_rows"].params["enabled"] is False
      and "substep_impulse" in env.metrics_manager.active_terms,
      f"rows_enabled={env.cfg.metrics['substep_impulse_rows'].params['enabled']}; "
      f"metrics={tuple(env.metrics_manager.active_terms)}",
    )

    check(
      "velocity CaT reads a six-joint substep tracker",
      tracker is not None
      and hook._vel_detection == "substep"
      and tuple(tracker.peak_qv_joint.shape) == (num_envs, 6)
      and tracker.peak_qv_joint.device == requested_device,
      f"detection={hook._vel_detection!r}; "
      f"shape={tuple(tracker.peak_qv_joint.shape) if tracker is not None else None}",
    )
    tracker.peak_qv_joint.fill_(1.0)
    tracker.peak_qv_joint[:, 2] = Z1_JOINT_VEL_LIMIT + 0.75
    synthetic_delta = hook(env).clone()
    check(
      "synthetic positive substep velocity excess produces positive CaT delta",
      bool((synthetic_delta > 0.0).all()),
      f"min={float(synthetic_delta.min()):.6f}; max={float(synthetic_delta.max()):.6f}",
    )

    impulse_raw = hook._cat.raw_constraints.get("joint_impulse_excess")
    impulse_prob = hook._cat.probs.get("joint_impulse_excess")
    check(
      "per-joint impulse is measured against the fixed caps",
      hook._use_impulse is True
      and bool(torch.allclose(
        hook._imp_limit,
        torch.tensor(_IMPULSE_LIMITS, device=hook._imp_limit.device),
        rtol=0.0,
        atol=1e-7,
      ))
      and impulse_raw is not None
      and tuple(impulse_raw.shape) == (num_envs, 6)
      and bool(torch.isfinite(impulse_raw).all()),
      f"caps={hook._imp_limit.detach().cpu().tolist()}; "
      f"raw_shape={tuple(impulse_raw.shape) if impulse_raw is not None else None}",
    )
    check(
      "impulse CaT is log-only",
      hook._imp_max_p == 0.0
      and impulse_prob is not None
      and bool(torch.count_nonzero(impulse_prob) == 0),
      f"imp_max_p={hook._imp_max_p}; "
      f"prob_max={float(impulse_prob.max()) if impulse_prob is not None else None}",
    )

    reward_manager = env.reward_manager
    expected_negative_terms = (
      ("action_rate", "joint_pos_limits")
      if not tt_enabled
      else ("action_rate", "joint_pos_limits", "r_tt")
    )
    live_negative_terms = tuple(
      name
      for name in reward_manager.active_terms
      if reward_manager.get_term_cfg(name).weight < 0.0
    )
    check(
      "arm-specific negative reward split is exact",
      live_negative_terms == expected_negative_terms,
      str(live_negative_terms),
    )
    negative_indices = [
      reward_manager.active_terms.index(name) for name in expected_negative_terms
    ]
    expected_r_pos = (
      reward_manager._step_reward.sum(dim=1)
      - reward_manager._step_reward[:, negative_indices].sum(dim=1)
    ) * env.step_dt
    check(
      "CaT positive-return split excludes every negative arm term",
      CAT_R_POS_KEY in env.extras
      and bool(torch.allclose(env.extras[CAT_R_POS_KEY], expected_r_pos)),
    )

    if not tt_enabled:
      check(
        "FIC-0 has no trackability cost",
        "r_tt" not in reward_manager.active_terms,
      )
    else:
      r_tt_cfg = reward_manager.get_term_cfg("r_tt")
      r_tt_index = reward_manager.active_terms.index("r_tt")
      target_ids = action.target_ids
      robot = env.scene["robot"]
      expected_raw_cost = torch.square(
        robot.data.joint_pos_target[:, target_ids]
        - robot.data.joint_pos[:, target_ids]
      ).sum(dim=1)
      live_weighted_cost = reward_manager._step_reward[:, r_tt_index]
      check(
        "FIC-TT uses k_tt=1 with a nonpositive live reward contribution",
        r_tt_cfg.weight == -1.0
        and r_tt_cfg.params["k_tt"] == 1.0
        and bool((expected_raw_cost >= 0.0).all())
        and bool(torch.allclose(live_weighted_cost, -expected_raw_cost)),
        f"weight={r_tt_cfg.weight}; k_tt={r_tt_cfg.params['k_tt']}; "
        f"weighted_max={float(live_weighted_cost.max()):.6f}",
      )

    metadata = _get_hammer_metadata(
      env,
      "joint-position-smoke",
      raw_policy_clip=raw_policy_clip,
    )
    check(
      "live export metadata constructs successfully",
      metadata["action_type"] == "joint_position"
      and metadata["action_dim"] == 6
      and metadata["raw_policy_clip"] == raw_policy_clip == 1.0
      and metadata["delivered_impulse_i_ref_n_s"]
      == _FIC_CONTROLLED_DROP_I_REF_N_S
      and metadata["observation_widths"]
      == {"actor": observation_width, "critic": observation_width}
      and metadata["guidance_type"] == guidance
      and metadata["guidance_reward_key"]
      == ("r_imit" if guidance == "direct_reference" else "r_waypoint_progress")
      and metadata["r_tt_enabled"] is tt_enabled
      and metadata["r_tt_k_tt"] == (1.0 if tt_enabled else "not_applicable"),
      str({
        key: metadata[key]
        for key in (
          "action_dim",
          "delivered_impulse_i_ref_n_s",
          "observation_widths",
          "guidance_type",
          "guidance_reward_key",
          "r_tt_enabled",
          "r_tt_k_tt",
        )
      }),
    )
  finally:
    env.close()

  return results


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--steps", type=int, default=3)
  args = parser.parse_args()

  selected = TASKS if args.task == "all" else (args.task,)
  failed: list[str] = []
  for task in selected:
    print(f"\n[smoke] task={task} device={args.device} envs={args.num_envs}")
    results = run_checks(task, args.device, args.num_envs, args.steps)
    for name, passed, detail in results:
      print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
      if not passed:
        failed.append(f"{task}: {name}")
    print(f"=== {sum(passed for _, passed, _ in results)}/{len(results)} checks passed ===")

  if failed:
    print("FAILED:", *failed, sep="\n  ")
    return 1
  print("\nFixed-impedance joint-policy live managers verified.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
