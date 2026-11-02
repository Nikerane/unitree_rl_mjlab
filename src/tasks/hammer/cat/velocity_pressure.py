"""Opt-in velocity CaT ceiling, clocked by completed vector-env transitions.

This changes a learning probability ceiling, never actions or physical constraints.
The metric runs after the common counter increments, so its transition index is
``common_step_counter - 1``. A checkpoint's counter instead indexes the NEXT step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


VELOCITY_PRESSURE_SCHEMA = "z1_cat_velocity_pressure_v1"
PRESSURE_STATE_SCHEMA = "z1_cat_velocity_pressure_state_v1"


def contract_digest(value: dict) -> str:
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  ).hexdigest()


def require_counter(value: object) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise ValueError("velocity pressure requires a nonnegative integer common_step_counter")
  return value


@dataclass(frozen=True)
class VelocityPressure:
  schema: str
  mode: str
  initial: float
  final: float
  ramp_steps: int
  total_steps: int

  @classmethod
  def from_config(cls, value: object) -> "VelocityPressure":
    keys = {"schema", "mode", "initial", "final", "ramp_steps", "total_steps"}
    if not isinstance(value, dict) or set(value) != keys:
      raise ValueError(f"velocity_pressure must contain exactly {sorted(keys)}")
    if value["schema"] != VELOCITY_PRESSURE_SCHEMA:
      raise ValueError("unsupported velocity_pressure schema")
    if value["mode"] not in ("constant", "reciprocal"):
      raise ValueError("velocity_pressure mode must be constant or reciprocal")
    for key in ("initial", "final"):
      v = value[key]
      if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"velocity_pressure {key} must be numeric")
      if not math.isfinite(v) or not 0.0 < v <= 1.0:
        raise ValueError(f"velocity_pressure {key} must be in (0, 1]")
    for key in ("ramp_steps", "total_steps"):
      if isinstance(value[key], bool) or not isinstance(value[key], int):
        raise ValueError(f"velocity_pressure {key} must be an integer")
    if not 2 <= value["ramp_steps"] <= value["total_steps"]:
      raise ValueError("velocity_pressure requires 2 <= ramp_steps <= total_steps")
    if value["initial"] > value["final"]:
      raise ValueError("velocity_pressure must be nondecreasing")
    if value["mode"] == "constant" and value["initial"] != value["final"]:
      raise ValueError("constant velocity_pressure requires initial == final")
    return cls(**{**value, "initial": float(value["initial"]), "final": float(value["final"])})

  def to_dict(self) -> dict:
    return asdict(self)

  def at(self, transition_index: int) -> float:
    require_counter(transition_index)
    if self.mode == "constant" or transition_index >= self.ramp_steps - 1:
      return self.final
    f = transition_index / (self.ramp_steps - 1)
    initial_lifetime = 1.0 / self.initial
    lifetime = initial_lifetime + (1.0 / self.final - initial_lifetime) * f
    return 1.0 / lifetime

  def describe(self, transition_index: int) -> dict:
    return {
      "transition_index": transition_index,
      "velocity_max_p": self.at(transition_index),
      "phase": (
        "constant" if self.mode == "constant"
        else "ramp" if transition_index < self.ramp_steps else "plateau"
      ),
    }


def pressure_hook(env):
  """Find the live manager-owned instance, not the original config class.

  mjlab 1.4 constructs term instances in ``metrics_manager.cfg``. Do not mutate
  cfg params to schedule: the existing hook caches ceilings at construction.
  """
  manager = getattr(env, "metrics_manager", None)
  cfg = getattr(manager, "cfg", None)
  term = cfg.get("cat_soft") if isinstance(cfg, dict) else None
  hook = getattr(term, "func", None)
  if getattr(hook, "velocity_pressure_enabled", False):
    return hook
  if term is not None and "velocity_pressure" in term.params:
    raise ValueError("velocity_pressure requires the live CatSoftHook metric instance")
  return None
