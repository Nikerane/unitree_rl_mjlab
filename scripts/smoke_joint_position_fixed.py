"""Lean live-manager gate for the fixed- and variable-impedance joint policies."""

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
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
from src.tasks.hammer.config.z1.joint_position_contract import (
  JOINT_NAMES,
  load_joint_position_contract,
)
from src.tasks.hammer.mdp.impulse_bound import (
  _ENV_SUBSTEP_DELIVERED_ATTR,
  _ENV_SUBSTEP_IMPULSE_ATTR,
)
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
VIC_TT_TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT"
)
TASK_CONTRACTS = {
  FIC0_TASK: {
    "width": 47, "guidance": "waypoint_progress", "tt": False,
    "action_width": 6, "vic": False,
  },
  FICTT_TASK: {
    "width": 47, "guidance": "waypoint_progress", "tt": True,
    "action_width": 6, "vic": False,
  },
  DIRECT_FIC0_TASK: {
    "width": 40, "guidance": "direct_reference", "tt": False,
    "action_width": 6, "vic": False,
  },
  DIRECT_FICTT_TASK: {
    "width": 40, "guidance": "direct_reference", "tt": True,
    "action_width": 6, "vic": False,
  },
  VIC_TT_TASK: {
    "width": 40, "guidance": "direct_reference", "tt": True,
    "action_width": 12, "vic": True,
  },
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
_FIC_CONTROLLED_DROP_I_REF_N_S = 0.2799950838088989
_JOINT_POSITION_CONTRACT_PATH = (
  _REPO_ROOT / "src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json"
)
_ARM_CONTROL_IDS = (0, 5, 1, 2, 3, 4)
_CPU_PARITY_TOLERANCE = 1.0e-6
_AUTHORITY_POSITION_ERROR_RAD = 1.0e-3


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


def _stack_trace(rows: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
  return {name: torch.stack(values) for name, values in rows.items()}


def _require_complete_nominal_trace(
  task: str,
  trace: dict[str, torch.Tensor],
  *,
  expected_controls: int,
  decimation: int,
) -> None:
  """Reject a terminated prefix before it can serve as parity evidence."""
  captured_controls = int(trace["r_tt"].shape[0])
  captured_substeps = int(trace["qpos"].shape[0])
  expected_substeps = expected_controls * decimation
  if (
    captured_controls != expected_controls
    or captured_substeps != expected_substeps
  ):
    raise RuntimeError(
      f"{task} nominal trace is incomplete: captured "
      f"{captured_controls}/{expected_controls} control steps and "
      f"{captured_substeps}/{expected_substeps} physics substeps"
    )


def _capture_nominal_trace(
  task: str,
  device: str,
  *,
  num_envs: int,
  seed: int = 1000,
) -> dict[str, torch.Tensor]:
  """Run the banked direct-reference target tape and capture every physics substep."""
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
  original_compute_substep = None
  try:
    env.reset(seed=seed)
    position = env.action_manager.get_term("joint_position")
    robot = env.scene["robot"]
    contact = env.scene["hammer_nail_contact"]
    tracker = getattr(env, _ENV_SUBSTEP_ATTR)
    impulse = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)
    delivered = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR)
    hook = _hook_of(env)
    control_ids = torch.tensor(
      _ARM_CONTROL_IDS, dtype=torch.long, device=env.device
    )
    r_tt_index = env.reward_manager.active_terms.index("r_tt")
    substep_rows: dict[str, list[torch.Tensor]] = {
      name: []
      for name in (
        "qpos",
        "qvel",
        "qtarget",
        "gain_kp",
        "bias_kp",
        "bias_kd",
        "contact",
        "peak_qv",
        "impulse",
        "delivered",
      )
    }
    control_rows: dict[str, list[torch.Tensor]] = {
      name: []
      for name in (
        "expected_qtarget",
        "r_tt",
        "vel_raw",
        "imp_raw",
        "vel_prob",
        "imp_prob",
        "cat_delta",
      )
    }

    original_compute_substep = env.metrics_manager.compute_substep

    def capture_after_substep() -> None:
      original_compute_substep()
      model = env.sim.model
      values = {
        "qpos": robot.data.joint_pos[:, position.target_ids],
        "qvel": robot.data.joint_vel[:, position.target_ids],
        "qtarget": robot.data.joint_pos_target[:, position.target_ids],
        "gain_kp": model.actuator_gainprm[:, control_ids, 0],
        "bias_kp": model.actuator_biasprm[:, control_ids, 1],
        "bias_kd": model.actuator_biasprm[:, control_ids, 2],
        "contact": (contact.data.found > 0).any(dim=-1),
        "peak_qv": tracker.peak_qv_joint,
        "impulse": impulse.impulse,
        "delivered": delivered.delivered,
      }
      for name, value in values.items():
        substep_rows[name].append(value.detach().clone())

    env.metrics_manager.compute_substep = capture_after_substep
    contract = load_joint_position_contract(_JOINT_POSITION_CONTRACT_PATH)
    target_tape = torch.tensor(
      contract.source_target_tape_rad.copy(),
      dtype=position.scale.dtype,
      device=env.device,
    )
    for target in target_tape:
      expected_target = target.expand(num_envs, -1)
      raw_position = (expected_target - position.offset) / position.scale
      command = raw_position
      if task == VIC_TT_TASK:
        command = torch.cat((raw_position, torch.zeros_like(raw_position)), dim=1)
      _, _, terminated, truncated, _ = env.step(command)
      control_rows["expected_qtarget"].append(expected_target.detach().clone())
      control_rows["r_tt"].append(
        env.reward_manager._step_reward[:, r_tt_index].detach().clone()
      )
      control_rows["vel_raw"].append(
        hook._cat.raw_constraints["joint_velocity_excess"].detach().clone()
      )
      control_rows["imp_raw"].append(
        hook._cat.raw_constraints["joint_impulse_excess"].detach().clone()
      )
      control_rows["vel_prob"].append(
        hook._cat.probs["joint_velocity_excess"].detach().clone()
      )
      control_rows["imp_prob"].append(
        hook._cat.probs["joint_impulse_excess"].detach().clone()
      )
      control_rows["cat_delta"].append(hook._cat.get_probs().detach().clone())
      if bool((terminated | truncated).any()):
        break

    trace = _stack_trace(substep_rows) | _stack_trace(control_rows)
    _require_complete_nominal_trace(
      task,
      trace,
      expected_controls=len(target_tape),
      decimation=int(cfg.decimation),
    )
    if torch.device(env.device).type == "cuda":
      torch.cuda.synchronize(env.device)
    return trace
  finally:
    if original_compute_substep is not None:
      env.metrics_manager.compute_substep = original_compute_substep
    env.close()


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
  if left.shape != right.shape:
    return float("inf")
  if left.dtype == torch.bool:
    return float(torch.count_nonzero(left != right))
  return float((left - right).abs().max())


def _cuda_repeat_tolerances(
  first: dict[str, torch.Tensor],
  repeated: dict[str, torch.Tensor],
  names: tuple[str, ...],
) -> dict[str, float]:
  """Freeze one absolute CUDA parity tolerance from each same-arm repeat delta."""
  tolerances: dict[str, float] = {}
  for name in names:
    first_value = first[name]
    repeated_value = repeated[name]
    if first_value.shape != repeated_value.shape:
      raise ValueError(
        f"CUDA repeatability field {name!r} changed shape: "
        f"{tuple(first_value.shape)} != {tuple(repeated_value.shape)}"
      )
    if not bool(torch.isfinite(first_value).all()) or not bool(
      torch.isfinite(repeated_value).all()
    ):
      raise ValueError(
        f"CUDA repeatability field {name!r} contains a non-finite value"
      )
    tolerances[name] = max(
      _CPU_PARITY_TOLERANCE,
      2.0 * _max_abs_difference(first_value, repeated_value),
    )
  return tolerances


def _allclose_trace(
  left: dict[str, torch.Tensor],
  right: dict[str, torch.Tensor],
  names: tuple[str, ...],
  *,
  atol: float,
  rtol: float,
) -> bool:
  return all(
    left[name].shape == right[name].shape
    and torch.allclose(left[name], right[name], atol=atol, rtol=rtol)
    for name in names
  )


def _trace_matches_tolerances(
  left: dict[str, torch.Tensor],
  right: dict[str, torch.Tensor],
  tolerances: dict[str, float],
) -> bool:
  """Compare each CUDA trace field against its pre-frozen absolute tolerance."""
  return all(
    left[name].shape == right[name].shape
    and torch.allclose(left[name], right[name], atol=atol, rtol=0.0)
    for name, atol in tolerances.items()
  )


def _run_authority_probe(device: str) -> dict[str, object]:
  """Exercise C=1.25 corners, a selected reset, and a live alternating tape."""
  cfg = load_env_cfg(VIC_TT_TASK, play=True)
  cfg.scene.num_envs = 2
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
  original_compute_substep = None
  try:
    env.reset(seed=1000)
    position = env.action_manager.get_term("joint_position")
    stiffness = env.action_manager.get_term("joint_stiffness")
    robot = env.scene["robot"]
    control_ids = stiffness.control_ids
    model = env.sim.model
    force_range_before = model.actuator_forcerange.clone()
    force_limited_before = model.actuator_forcelimited.clone()
    p_tape = torch.tensor(
      (
        ((-1.0,) * 6, (1.0,) * 6),
        ((0.0,) * 6, (0.0,) * 6),
        ((1.0,) * 6, (-1.0,) * 6),
      ),
      dtype=position.scale.dtype,
      device=env.device,
    )
    physical_target = (
      robot.data.joint_pos[:, position.target_ids].clone()
      + _AUTHORITY_POSITION_ERROR_RAD
    )
    raw_position = (physical_target - position.offset) / position.scale
    static_rows: dict[str, list[torch.Tensor]] = {
      name: []
      for name in (
        "p",
        "kp",
        "kd",
        "bias_kp",
        "bias_kd",
        "force",
        "error",
      )
    }
    for p in p_tape:
      env.action_manager.process_action(torch.cat((raw_position, p), dim=1))
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      env.sim.forward()
      telemetry = stiffness.telemetry
      values = {
        "p": telemetry.p,
        "kp": model.actuator_gainprm[:, control_ids, 0],
        "kd": -model.actuator_biasprm[:, control_ids, 2],
        "bias_kp": model.actuator_biasprm[:, control_ids, 1],
        "bias_kd": model.actuator_biasprm[:, control_ids, 2],
        "force": env.sim.data.actuator_force[:, control_ids],
        "error": (
          robot.data.joint_pos_target[:, position.target_ids]
          - robot.data.joint_pos[:, position.target_ids]
        ),
      }
      for name, value in values.items():
        static_rows[name].append(value.detach().clone())

    kept_before = {
      "raw": stiffness.raw_action[1].detach().clone(),
      "p": stiffness.telemetry.p[1].detach().clone(),
      "multiplier": stiffness.telemetry.multiplier[1].detach().clone(),
      "kp": stiffness.telemetry.kp[1].detach().clone(),
      "kd": stiffness.telemetry.kd[1].detach().clone(),
      "gain": model.actuator_gainprm[1, control_ids].detach().clone(),
      "bias": model.actuator_biasprm[1, control_ids].detach().clone(),
    }
    env.reset(env_ids=torch.tensor((0,), dtype=torch.long, device=env.device))
    kept_after = {
      "raw": stiffness.raw_action[1].detach().clone(),
      "p": stiffness.telemetry.p[1].detach().clone(),
      "multiplier": stiffness.telemetry.multiplier[1].detach().clone(),
      "kp": stiffness.telemetry.kp[1].detach().clone(),
      "kd": stiffness.telemetry.kd[1].detach().clone(),
      "gain": model.actuator_gainprm[1, control_ids].detach().clone(),
      "bias": model.actuator_biasprm[1, control_ids].detach().clone(),
    }
    reset_world_is_nominal = (
      torch.equal(stiffness.raw_action[0], torch.zeros_like(stiffness.raw_action[0]))
      and torch.equal(stiffness.telemetry.p[0], torch.zeros_like(stiffness.telemetry.p[0]))
      and torch.equal(
        stiffness.telemetry.multiplier[0],
        torch.ones_like(stiffness.telemetry.multiplier[0]),
      )
      and torch.equal(stiffness.telemetry.kp[0], stiffness.nominal_kp)
      and torch.equal(stiffness.telemetry.kd[0], stiffness.nominal_kd)
      and torch.equal(
        model.actuator_gainprm[0, control_ids, 0], stiffness.nominal_kp
      )
      and torch.equal(
        model.actuator_biasprm[0, control_ids, 1], -stiffness.nominal_kp
      )
      and torch.equal(
        model.actuator_biasprm[0, control_ids, 2], -stiffness.nominal_kd
      )
    )

    env.reset()
    physical_target = (
      robot.data.joint_pos[:, position.target_ids].clone()
      + _AUTHORITY_POSITION_ERROR_RAD
    )
    raw_position = (physical_target - position.offset) / position.scale
    alternating_p_tape = torch.stack(
      (p_tape[0], p_tape[2], p_tape[0], p_tape[2])
    )
    decimation = int(env.cfg.decimation)
    dynamic_rows: dict[str, list[torch.Tensor]] = {
      name: []
      for name in ("qpos", "qvel", "qtarget", "force", "kp", "bias")
    }
    original_compute_substep = env.metrics_manager.compute_substep

    def capture_authority_substep() -> None:
      original_compute_substep()
      values = {
        "qpos": robot.data.joint_pos[:, position.target_ids],
        "qvel": robot.data.joint_vel[:, position.target_ids],
        "qtarget": robot.data.joint_pos_target[:, position.target_ids],
        "force": env.sim.data.actuator_force[:, control_ids],
        "kp": model.actuator_gainprm[:, control_ids, 0],
        "bias": model.actuator_biasprm[:, control_ids],
      }
      for name, value in values.items():
        dynamic_rows[name].append(value.detach().clone())

    env.metrics_manager.compute_substep = capture_authority_substep
    for p in alternating_p_tape:
      env.step(torch.cat((raw_position, p), dim=1))
    env.metrics_manager.compute_substep = original_compute_substep
    original_compute_substep = None

    if torch.device(env.device).type == "cuda":
      torch.cuda.synchronize(env.device)
    static = _stack_trace(static_rows)
    dynamic = _stack_trace(dynamic_rows)
    multiplier = torch.pow(torch.tensor(1.25, device=env.device), p_tape)
    expected_kp = multiplier * stiffness.nominal_kp
    expected_kd = torch.sqrt(multiplier) * stiffness.nominal_kd
    alternating_multiplier = torch.pow(
      torch.tensor(1.25, dtype=p_tape.dtype, device=env.device),
      alternating_p_tape,
    )
    alternating_kp = (
      alternating_multiplier * stiffness.nominal_kp
    ).repeat_interleave(decimation, dim=0)
    alternating_kd = (
      torch.sqrt(alternating_multiplier) * stiffness.nominal_kd
    ).repeat_interleave(decimation, dim=0)
    alternating_qtarget = physical_target.unsqueeze(0).expand(
      alternating_kp.shape[0], -1, -1
    )
    alternating_hold = (
      torch.equal(dynamic["kp"], alternating_kp)
      and torch.equal(dynamic["bias"][..., 1], -alternating_kp)
      and torch.equal(dynamic["bias"][..., 2], -alternating_kd)
      and torch.allclose(
        dynamic["qtarget"],
        alternating_qtarget,
        rtol=0.0,
        atol=1.0e-7,
      )
    )
    force_range = force_range_before[:, control_ids]
    force_limit = force_range.abs().amax(dim=-1)
    static_force = static["force"]
    dynamic_force = dynamic["force"]
    soft_force = torch.stack((static_force[0, 0], static_force[2, 1])).abs()
    nominal_force = static_force[1].abs()
    stiff_force = torch.stack((static_force[2, 0], static_force[0, 1])).abs()
    static_in_range = (
      (static_force >= force_range[None, :, :, 0])
      & (static_force <= force_range[None, :, :, 1])
    ).all()
    dynamic_in_range = (
      (dynamic_force >= force_range[None, :, :, 0])
      & (dynamic_force <= force_range[None, :, :, 1])
    ).all()
    selected_reset_isolated = reset_world_is_nominal and all(
      torch.equal(kept_before[name], kept_after[name]) for name in kept_before
    )
    native_limits_unchanged = torch.equal(
      force_range_before, model.actuator_forcerange
    ) and torch.equal(force_limited_before, model.actuator_forcelimited)
    targeted_force_limited = (
      force_limited_before[control_ids]
      if force_limited_before.ndim == 1
      else force_limited_before[:, control_ids]
    )
    finite_tensors = (*static.values(), *dynamic.values())
    return {
      "gain_map": (
        torch.equal(static["p"], p_tape)
        and torch.equal(static["kp"], expected_kp)
        and torch.equal(static["kd"], expected_kd)
        and torch.equal(static["bias_kp"], -expected_kp)
        and torch.equal(static["bias_kd"], -expected_kd)
      ),
      "finite": all(bool(torch.isfinite(value).all()) for value in finite_tensors),
      "force_order": bool((soft_force < nominal_force).all())
      and bool((nominal_force < stiff_force).all()),
      "nonsaturated": bool(
        (static_force.abs() < 0.1 * force_limit[None]).all()
      ),
      "force_limits": bool(static_in_range)
      and bool(dynamic_in_range)
      and bool((targeted_force_limited != 0).all())
      and native_limits_unchanged,
      "declared_error": bool(
        torch.allclose(
          static["error"],
          torch.full_like(static["error"], _AUTHORITY_POSITION_ERROR_RAD),
          rtol=0.0,
          atol=1.0e-6,
        )
      ),
      "selected_reset": selected_reset_isolated,
      "alternating_hold": bool(alternating_hold),
      "alternating_substeps": dynamic["force"].shape[0],
      "force_max": float(static_force.abs().max()),
      "force_limit_min": float(force_limit.min()),
    }
  finally:
    if original_compute_substep is not None:
      env.metrics_manager.compute_substep = original_compute_substep
    env.close()


def run_vic_qualification_checks(
  device: str = "cpu", num_envs: int = 2
) -> list[tuple[str, bool, str]]:
  """Prove FIC parity and bounded VIC authority through the live manager seam."""
  if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs != 2:
    raise ValueError("VIC qualification requires exactly two environments")
  results: list[tuple[str, bool, str]] = []

  def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))

  exact_names = ("qtarget", "gain_kp", "bias_kp", "bias_kd", "contact")
  physical_names = ("qpos", "qvel")
  cat_names = (
    "r_tt",
    "peak_qv",
    "impulse",
    "delivered",
    "vel_raw",
    "imp_raw",
    "vel_prob",
    "imp_prob",
    "cat_delta",
  )
  tolerance_names = physical_names + cat_names
  is_cuda = torch.device(device).type == "cuda"
  fic = _capture_nominal_trace(
    DIRECT_FICTT_TASK, device, num_envs=num_envs
  )
  cuda_tolerances: dict[str, float] | None = None
  cuda_repeat_deltas: dict[str, float] | None = None
  if is_cuda:
    fic_repeat = _capture_nominal_trace(
      DIRECT_FICTT_TASK, device, num_envs=num_envs
    )
    for name in exact_names:
      if name != "contact" and (
        not bool(torch.isfinite(fic[name]).all())
        or not bool(torch.isfinite(fic_repeat[name]).all())
      ):
        raise RuntimeError(
          f"same-arm CUDA repeatability field {name!r} contains a non-finite value"
        )
      if (
        fic[name].shape != fic_repeat[name].shape
        or not torch.equal(fic[name], fic_repeat[name])
      ):
        raise RuntimeError(
          f"same-arm CUDA repeatability field {name!r} must be exactly equal "
          "before freezing tolerances or observing VIC"
        )
    cuda_tolerances = _cuda_repeat_tolerances(
      fic, fic_repeat, tolerance_names
    )
    cuda_repeat_deltas = {
      name: _max_abs_difference(fic[name], fic_repeat[name])
      for name in tolerance_names
    }
  vic = _capture_nominal_trace(VIC_TT_TASK, device, num_envs=num_envs)
  contract = load_joint_position_contract(_JOINT_POSITION_CONTRACT_PATH)
  expected_steps = len(contract.source_target_tape_rad)
  fic_decimation = int(load_env_cfg(DIRECT_FICTT_TASK, play=True).decimation)
  vic_decimation = int(load_env_cfg(VIC_TT_TASK, play=True).decimation)
  if fic_decimation != vic_decimation:
    raise RuntimeError(
      "FIC-TT and VIC-TT must use the same control decimation for parity: "
      f"{fic_decimation} != {vic_decimation}"
    )
  decimation = fic_decimation
  fic_steps = fic["r_tt"].shape[0]
  vic_steps = vic["r_tt"].shape[0]
  substep_count_ok = (
    fic_steps == expected_steps
    and vic_steps == expected_steps
    and fic["qpos"].shape[0] == fic_steps * decimation
    and vic["qpos"].shape[0] == vic_steps * decimation
  )
  fic_expected = fic["expected_qtarget"].repeat_interleave(decimation, dim=0)
  vic_expected = vic["expected_qtarget"].repeat_interleave(decimation, dim=0)
  targets_are_paired = (
    fic["qtarget"].shape == fic_expected.shape
    and vic["qtarget"].shape == vic_expected.shape
    and torch.allclose(fic["qtarget"], fic_expected, rtol=0.0, atol=1.0e-7)
    and torch.allclose(vic["qtarget"], vic_expected, rtol=0.0, atol=1.0e-7)
  )
  check(
    "nominal tape captures every target and its configured physics substeps",
    substep_count_ok and targets_are_paired,
    f"targets={expected_steps}; decimation={decimation}; "
    f"fic_steps={fic_steps}; vic_steps={vic_steps}; "
    f"fic_substeps={fic['qpos'].shape[0]}; vic_substeps={vic['qpos'].shape[0]}",
  )

  cross_exact_names = ("qtarget", "gain_kp", "bias_kp", "bias_kd")
  exact_parity = all(
    fic[name].shape == vic[name].shape and torch.equal(fic[name], vic[name])
    for name in cross_exact_names
  )
  check(
    "p=0 applies exactly equal position targets and native gains",
    exact_parity,
    str({name: _max_abs_difference(fic[name], vic[name]) for name in cross_exact_names}),
  )

  physical_parity = (
    _trace_matches_tolerances(
      fic,
      vic,
      {name: cuda_tolerances[name] for name in physical_names},
    )
    if cuda_tolerances is not None
    else _allclose_trace(
      fic,
      vic,
      physical_names,
      atol=_CPU_PARITY_TOLERANCE,
      rtol=_CPU_PARITY_TOLERANCE,
    )
  )
  check(
    (
      "nominal CUDA physical traces match frozen same-arm tolerances"
      if is_cuda
      else "nominal CPU physical traces match FIC-TT within 1e-6"
    ),
    physical_parity,
    str(
      {name: _max_abs_difference(fic[name], vic[name]) for name in physical_names}
      | (
        {
          "repeat_deltas": cuda_repeat_deltas,
          "cuda_tolerances": cuda_tolerances,
        }
        if is_cuda
        else {}
      )
    ),
  )

  cat_parity = (
    _trace_matches_tolerances(
      fic,
      vic,
      {name: cuda_tolerances[name] for name in cat_names},
    )
    if cuda_tolerances is not None
    else _allclose_trace(
      fic,
      vic,
      cat_names,
      atol=_CPU_PARITY_TOLERANCE,
      rtol=_CPU_PARITY_TOLERANCE,
    )
  )
  contact_parity = fic["contact"].shape == vic["contact"].shape and torch.equal(
    fic["contact"], vic["contact"]
  )
  check(
    (
      "nominal CUDA RTT, contact, and CaT traces match frozen same-arm tolerances"
      if is_cuda
      else "nominal RTT, contact, and CaT traces match FIC-TT within 1e-6"
    ),
    contact_parity
    and cat_parity,
    str(
      {name: _max_abs_difference(fic[name], vic[name]) for name in cat_names}
      | {"contact": _max_abs_difference(fic["contact"], vic["contact"])}
      | (
        {
          "repeat_deltas": cuda_repeat_deltas,
          "cuda_tolerances": cuda_tolerances,
        }
        if is_cuda
        else {}
      )
    ),
  )
  check(
    "nominal parity tape exercises live hammer-nail contact",
    bool(fic["contact"].any()) and bool(vic["contact"].any()),
    f"fic_contact_substeps={int(fic['contact'].any(dim=1).sum())}; "
    f"vic_contact_substeps={int(vic['contact'].any(dim=1).sum())}",
  )

  authority = _run_authority_probe(device)
  check(
    "C=1.25 corners implement the declared native gain map",
    bool(authority["gain_map"]),
  )
  check(
    "declared nonsaturated 1 mrad error produces soft < nominal < stiff force",
    bool(authority["declared_error"])
    and bool(authority["nonsaturated"])
    and bool(authority["force_order"]),
    f"force_max={authority['force_max']:.6f}; "
    f"smallest_limit={authority['force_limit_min']:.6f}",
  )
  check(
    "alternating commands hold paired targets and gains within native force limits",
    bool(authority["finite"])
    and bool(authority["force_limits"])
    and bool(authority["alternating_hold"])
    and authority["alternating_substeps"] == 40,
    f"captured_substeps={authority['alternating_substeps']}",
  )
  check(
    "selected reset restores only the requested world's nominal gains",
    bool(authority["selected_reset"]),
  )
  return results


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
  action_width = contract["action_width"]
  is_vic = contract["vic"]
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
    command = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim),
      device=env.device,
    )
    command[:, :action.action_dim] = 0.25
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
      env.action_manager.total_action_dim == action_width
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
    if not is_vic:
      check(
        "there is no gain action",
        tuple(env.action_manager.active_terms) == ("joint_position",),
        str(env.action_manager.active_terms),
      )
    else:
      stiffness = env.action_manager.get_term("joint_stiffness")
      telemetry = stiffness.telemetry
      telemetry_tensors = (
        telemetry.control_ids,
        telemetry.nominal_kp,
        telemetry.nominal_kd,
        telemetry.p,
        telemetry.multiplier,
        telemetry.kp,
        telemetry.kd,
      )
      check(
        "VIC action is the exact ordered 6D position plus 6D stiffness pair",
        tuple(env.action_manager.active_terms)
        == ("joint_position", "joint_stiffness")
        and tuple(env.action_manager.action_term_dim) == (6, 6)
        and stiffness.action_dim == 6,
        f"terms={tuple(env.action_manager.active_terms)}; "
        f"dims={tuple(env.action_manager.action_term_dim)}",
      )
      check(
        "VIC live immutable controller telemetry is finite and canonical",
        telemetry.mapping_family == "author_v1_exponential"
        and telemetry.C == 1.25
        and telemetry.p_bounds == (-1.0, 1.0)
        and telemetry.joint_names == JOINT_NAMES
        and telemetry.control_ids.detach().cpu().tolist() == [0, 5, 1, 2, 3, 4]
        and all(value.device == requested_device for value in telemetry_tensors)
        and all(bool(torch.isfinite(value).all()) for value in telemetry_tensors),
        f"family={telemetry.mapping_family}; C={telemetry.C}; "
        f"control_ids={telemetry.control_ids.detach().cpu().tolist()}",
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
        torch.tensor(IMP_J_LIMIT, device=hook._imp_limit.device),
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
      metadata["action_type"]
      == (
        "joint_position_variable_impedance" if is_vic else "joint_position"
      )
      and metadata["action_dim"] == action_width
      and metadata["raw_policy_clip"] == raw_policy_clip == 1.0
      and metadata["delivered_impulse_i_ref_n_s"]
      == _FIC_CONTROLLED_DROP_I_REF_N_S
      and metadata["observation_widths"]
      == {"actor": observation_width, "critic": observation_width}
      and metadata["guidance_type"] == guidance
      and metadata["guidance_reward_key"]
      == ("r_imit" if guidance == "direct_reference" else "r_waypoint_progress")
      and metadata["r_tt_enabled"] is tt_enabled
      and metadata["r_tt_k_tt"] == (1.0 if tt_enabled else "not_applicable")
      and (
        not is_vic
        or (
          metadata["action_terms"] == ["joint_position", "joint_stiffness"]
          and metadata["action_term_dims"] == [6, 6]
          and metadata["position_action_dim"] == 6
          and metadata["action_observation_source"] == "joint_position"
          and metadata["action_rate_source"] == "joint_position"
          and metadata["r_tt_weight"] == -1.0
          and metadata["variable_impedance"]["C"] == 1.25
        )
      ),
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

  if args.task in (VIC_TT_TASK, "all"):
    print(f"\n[qualification] FIC-TT vs VIC-TT device={args.device} envs=2")
    results = run_vic_qualification_checks(args.device, num_envs=2)
    for name, passed, detail in results:
      print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
      if not passed:
        failed.append(f"VIC qualification: {name}")
    print(f"=== {sum(passed for _, passed, _ in results)}/{len(results)} checks passed ===")

  if failed:
    print("FAILED:", *failed, sep="\n  ")
    return 1
  print("\nSelected joint-policy live managers verified.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
