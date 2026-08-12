"""Native variable-impedance action for the Z1 arm."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING

import torch

from mjlab.actuator import BuiltinPositionActuator
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import requires_model_fields

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


VARIABLE_IMPEDANCE_MAPPING_FAMILY = "author_v1_exponential"
VARIABLE_IMPEDANCE_P_BOUNDS = (-1.0, 1.0)
_NATIVE_GAIN_FIELDS = ("actuator_gainprm", "actuator_biasprm")
_Z1_CANONICAL_ARM_JOINT_NAMES = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
)


@dataclass(frozen=True)
class VariableImpedanceGains:
  """Author-v1 gain-map outputs for one policy action tensor."""

  p: torch.Tensor
  multiplier: torch.Tensor
  kp: torch.Tensor
  kd: torch.Tensor


@dataclass(frozen=True)
class JointStiffnessTelemetry:
  """Read-only snapshot of the live stiffness-action contract and outputs."""

  mapping_family: str
  C: float
  p_bounds: tuple[float, float]
  joint_names: tuple[str, ...]
  control_ids: torch.Tensor
  nominal_kp: torch.Tensor
  nominal_kd: torch.Tensor
  p: torch.Tensor
  multiplier: torch.Tensor
  kp: torch.Tensor
  kd: torch.Tensor


def variable_impedance_gains(
  p: torch.Tensor,
  nominal_kp: torch.Tensor,
  nominal_kd: torch.Tensor,
  *,
  C: float,
) -> VariableImpedanceGains:
  """Map bounded stiffness coordinates to coupled native PD gains."""
  if isinstance(C, bool) or not isinstance(C, Real):
    raise ValueError(f"C must be a finite real number greater than one, got {C!r}")
  C = float(C)
  if not math.isfinite(C) or C <= 1.0:
    raise ValueError(f"C must be finite and greater than one, got {C!r}")
  for name, value in (
    ("p", p),
    ("nominal_kp", nominal_kp),
    ("nominal_kd", nominal_kd),
  ):
    if not isinstance(value, torch.Tensor):
      raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
      raise ValueError(f"{name} must have a floating dtype")
  if nominal_kp.ndim != 1 or nominal_kd.ndim != 1:
    raise ValueError("nominal_kp and nominal_kd must be one-dimensional")
  if nominal_kp.shape != nominal_kd.shape:
    raise ValueError("nominal_kp and nominal_kd must have the same shape")
  if p.ndim < 1 or p.shape[-1] != nominal_kp.shape[0]:
    raise ValueError(
      "p shape must end in the nominal gain width; "
      f"got p={tuple(p.shape)}, nominal={tuple(nominal_kp.shape)}"
    )
  if p.dtype != nominal_kp.dtype or p.dtype != nominal_kd.dtype:
    raise ValueError("p, nominal_kp, and nominal_kd must have the same dtype")
  if p.device != nominal_kp.device or p.device != nominal_kd.device:
    raise ValueError("p, nominal_kp, and nominal_kd must be on the same device")
  if not bool(torch.isfinite(p).all()):
    raise ValueError("p must contain only finite values")
  if not bool(torch.isfinite(nominal_kp).all()) or not bool((nominal_kp > 0).all()):
    raise ValueError("nominal_kp must contain only finite positive values")
  if not bool(torch.isfinite(nominal_kd).all()) or not bool((nominal_kd > 0).all()):
    raise ValueError("nominal_kd must contain only finite positive values")

  gains = _variable_impedance_gains_unchecked(p, nominal_kp, nominal_kd, C=C)
  if not bool(torch.isfinite(gains.kp).all()) or not bool(
    torch.isfinite(gains.kd).all()
  ):
    raise ValueError("gain map produced a non-finite output")
  return gains


def _variable_impedance_gains_unchecked(
  p: torch.Tensor,
  nominal_kp: torch.Tensor,
  nominal_kd: torch.Tensor,
  *,
  C: float,
) -> VariableImpedanceGains:
  """Tensor-only gain map for the action-manager hot path.

  Shape, base, and nominal-gain validation is performed once at construction;
  policy wrappers and the environment's numerical guards own non-finite policy
  outputs.  Keeping reductions out of this path avoids a CUDA host sync on every
  control step.
  """
  bounded_p = p.clamp(-1.0, 1.0)
  base = torch.as_tensor(C, dtype=p.dtype, device=p.device)
  multiplier = torch.pow(base, bounded_p)
  return VariableImpedanceGains(
    p=bounded_p,
    multiplier=multiplier,
    kp=multiplier * nominal_kp,
    kd=torch.sqrt(multiplier) * nominal_kd,
  )


@requires_model_fields(*_NATIVE_GAIN_FIELDS)
def expand_variable_impedance_model_fields(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
) -> None:
  """Declare the per-world native fields required by the stiffness action.

  The event manager expands declared fields before constructing the action
  manager.  The startup callback itself deliberately has no mutation to apply.
  """
  del env, env_ids


@dataclass(kw_only=True)
class JointStiffnessActionCfg(ActionTermCfg):
  """Configuration for per-joint native position-actuator gain commands."""

  joint_names: tuple[str, ...]
  C: float = 1.25

  def build(self, env: ManagerBasedRlEnv) -> JointStiffnessAction:
    return JointStiffnessAction(self, env)


class JointStiffnessAction(ActionTerm):
  """Apply the author-v1 variable-stiffness map to native MuJoCo PD fields."""

  cfg: JointStiffnessActionCfg

  def __init__(self, cfg: JointStiffnessActionCfg, env: ManagerBasedRlEnv):
    self._require_independent_model_fields(env)
    super().__init__(cfg=cfg, env=env)

    self._joint_names = tuple(cfg.joint_names)
    if self._joint_names != _Z1_CANONICAL_ARM_JOINT_NAMES:
      raise ValueError(
        "joint stiffness requires canonical Z1 arm joint_names "
        f"{_Z1_CANONICAL_ARM_JOINT_NAMES}; got {self._joint_names}"
      )

    control_id_by_joint: dict[str, int] = {}
    wrong_type: dict[str, str] = {}
    for actuator in self._entity.actuators:
      for joint_name, control_id in zip(
        actuator.target_names, actuator.global_ctrl_ids.tolist(), strict=True
      ):
        if joint_name not in self._joint_names:
          continue
        if not isinstance(actuator, BuiltinPositionActuator):
          wrong_type[joint_name] = type(actuator).__name__
          continue
        if joint_name in control_id_by_joint:
          raise ValueError(f"joint {joint_name!r} maps to multiple native actuators")
        control_id_by_joint[joint_name] = int(control_id)

    if wrong_type:
      details = ", ".join(
        f"{name} ({kind})" for name, kind in sorted(wrong_type.items())
      )
      raise ValueError(
        "joint stiffness requires BuiltinPositionActuator targets; got " + details
      )
    missing = [name for name in self._joint_names if name not in control_id_by_joint]
    if missing:
      raise ValueError(
        "joint stiffness could not resolve native actuator control IDs for "
        + ", ".join(missing)
      )

    model_dtype = env.sim.model.actuator_gainprm.dtype
    self._control_ids = torch.tensor(
      [control_id_by_joint[name] for name in self._joint_names],
      dtype=torch.long,
      device=self.device,
    )
    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    default_biasprm = env.sim.get_default_field("actuator_biasprm")
    self._nominal_kp = default_gainprm[self._control_ids, 0].to(
      dtype=model_dtype, device=self.device
    ).clone()
    self._nominal_kd = (-default_biasprm[self._control_ids, 2]).to(
      dtype=model_dtype, device=self.device
    ).clone()
    nominal_bias_kp = default_biasprm[self._control_ids, 1].to(
      dtype=model_dtype, device=self.device
    )
    if not torch.equal(nominal_bias_kp, -self._nominal_kp):
      raise ValueError(
        "native position actuator contract requires biasprm[..., 1] == -Kp"
      )

    self._action_dim = len(self._joint_names)
    self._raw_actions = torch.zeros(
      (self.num_envs, self.action_dim), dtype=model_dtype, device=self.device
    )
    nominal = variable_impedance_gains(
      self._raw_actions,
      self._nominal_kp,
      self._nominal_kd,
      C=cfg.C,
    )
    self._C = float(cfg.C)
    self._p = nominal.p.clone()
    self._multiplier = nominal.multiplier.clone()
    self._kp = nominal.kp.clone()
    self._kd = nominal.kd.clone()
    self._all_env_ids = torch.arange(
      self.num_envs, dtype=torch.long, device=self.device
    )

  @staticmethod
  def _require_independent_model_fields(env: ManagerBasedRlEnv) -> None:
    missing = [
      field for field in _NATIVE_GAIN_FIELDS if field not in env.sim.expanded_fields
    ]
    if missing:
      raise ValueError(
        "joint stiffness requires native model fields expanded per world before "
        f"action construction; missing {missing}"
      )
    for field in _NATIVE_GAIN_FIELDS:
      value = getattr(env.sim.model, field)
      if value.shape[0] != env.num_envs or (
        env.num_envs > 1 and value.stride(0) == 0
      ):
        raise ValueError(
          f"joint stiffness requires {field} expanded per world with independent "
          f"storage; got shape={tuple(value.shape)}, stride={value.stride()}"
        )

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def joint_names(self) -> tuple[str, ...]:
    return self._joint_names

  @property
  def control_ids(self) -> torch.Tensor:
    return self._control_ids.clone()

  @property
  def nominal_kp(self) -> torch.Tensor:
    return self._nominal_kp.clone()

  @property
  def nominal_kd(self) -> torch.Tensor:
    return self._nominal_kd.clone()

  @property
  def telemetry(self) -> JointStiffnessTelemetry:
    return JointStiffnessTelemetry(
      mapping_family=VARIABLE_IMPEDANCE_MAPPING_FAMILY,
      C=self._C,
      p_bounds=VARIABLE_IMPEDANCE_P_BOUNDS,
      joint_names=self._joint_names,
      control_ids=self._control_ids.clone(),
      nominal_kp=self._nominal_kp.clone(),
      nominal_kd=self._nominal_kd.clone(),
      p=self._p.clone(),
      multiplier=self._multiplier.clone(),
      kp=self._kp.clone(),
      kd=self._kd.clone(),
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    if not isinstance(actions, torch.Tensor) or actions.shape != self._raw_actions.shape:
      actual_shape = getattr(actions, "shape", None)
      raise ValueError(
        f"joint stiffness action shape must be {tuple(self._raw_actions.shape)}, "
        f"got {actual_shape}"
      )
    actions = actions.to(dtype=self._raw_actions.dtype, device=self.device)
    gains = _variable_impedance_gains_unchecked(
      actions,
      self._nominal_kp,
      self._nominal_kd,
      C=self._C,
    )
    self._raw_actions.copy_(actions)
    self._p.copy_(gains.p)
    self._multiplier.copy_(gains.multiplier)
    self._kp.copy_(gains.kp)
    self._kd.copy_(gains.kd)

  def apply_actions(self) -> None:
    self._write_native_gains(self._all_env_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = self._resolve_env_ids(env_ids)
    self._raw_actions[env_ids] = 0.0
    self._p[env_ids] = 0.0
    self._multiplier[env_ids] = 1.0
    self._kp[env_ids] = self._nominal_kp
    self._kd[env_ids] = self._nominal_kd
    self._write_native_gains(env_ids)

  def _resolve_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
      return self._all_env_ids
    if isinstance(env_ids, slice):
      return self._all_env_ids[env_ids]
    if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1:
      raise ValueError("env_ids must be a one-dimensional tensor, slice, or None")
    return env_ids.to(dtype=torch.long, device=self.device)

  def _write_native_gains(self, env_ids: torch.Tensor) -> None:
    rows = env_ids[:, None]
    control_ids = self._control_ids[None, :]
    model = self._env.sim.model
    model.actuator_gainprm[rows, control_ids, 0] = self._kp[env_ids]
    model.actuator_biasprm[rows, control_ids, 1] = -self._kp[env_ids]
    model.actuator_biasprm[rows, control_ids, 2] = -self._kd[env_ids]
