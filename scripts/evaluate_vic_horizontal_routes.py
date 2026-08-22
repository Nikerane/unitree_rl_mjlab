#!/usr/bin/env python3
"""Frozen no-learning evaluation for the Z1 horizontal-route pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from scripts import impulse_cat_activation_survey as survey


_TASK_PREFIX = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT-HorizontalRoutes-"
)
HORIZONTAL_ANNEALED_TASK = f"{_TASK_PREFIX}Annealed"
HORIZONTAL_PERSISTENT_TASK = f"{_TASK_PREFIX}Persistent"
TRAINING_CAPS_N_M_S = (0.369, 0.246, 0.738, 0.369, 0.246, 0.0164)
ROUTE_LABEL_BY_SIGN: Mapping[int, str] = MappingProxyType(
  {-1: "rminus", 0: "r0", 1: "rplus"}
)
POLICY_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
  {
    "horizontal_routes_annealed": MappingProxyType(
      {
        "task": HORIZONTAL_ANNEALED_TASK,
        "sha256": "2a434c50457455daab10b5ff32e1ee26e3b189b0092a315f3e881bb7f960156a",
      }
    ),
    "horizontal_routes_persistent": MappingProxyType(
      {
        "task": HORIZONTAL_PERSISTENT_TASK,
        "sha256": "feb22dd5e5740e96348387c846ef395143316354f7e3163b5684f24db58f4e9e",
      }
    ),
  }
)
TRAINING_CODE_REVISION = "1f0656012e9e5a03b0c2d404d9d4fc605786009b"
EXPECTED_ASSET_REVISION = survey.EXPECTED_EVALUATION_ASSET_REVISION


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _evaluation_revisions() -> dict[str, str]:
  return survey._evaluation_revisions()


def validate_evaluation_revision(repository: Path, revision: str) -> str:
  return survey.validate_evaluation_revision(
    repository, revision, approved_base_revision=TRAINING_CODE_REVISION
  )


def prepare_horizontal_measurement_config(env_cfg: Any, *, role: str) -> dict[str, object]:
  """Validate the trained treatment, then make impulse CaT measurement-only."""
  if role not in POLICY_SPECS:
    raise ValueError("unknown horizontal-route policy role")
  return survey._prepare_survey_measurement_config(
    env_cfg,
    survey.PROVISIONAL_CAPS_N_M_S,
    source_imp_max_p=0.2,
    source_imp_limit_n_m_s=TRAINING_CAPS_N_M_S,
  )


def validate_checkpoint(checkpoint: Path, *, role: str) -> tuple[Path, str]:
  if role not in POLICY_SPECS:
    raise ValueError("unknown horizontal-route policy role")
  checkpoint = Path(checkpoint)
  if checkpoint.name != "model_499.pt" or checkpoint.is_symlink() or not checkpoint.is_file():
    raise ValueError("checkpoint must be a regular non-symlink model_499.pt")
  resolved = checkpoint.resolve(strict=True)
  digest = _sha256(resolved)
  if digest != POLICY_SPECS[role]["sha256"]:
    raise RuntimeError("horizontal role/checkpoint SHA-256 mismatch")
  return resolved, digest


def _validate_route_trace(trace: Mapping[str, np.ndarray], *, route_sign: int) -> None:
  required = {
    "episode_id",
    "route_sign",
    "strike_phase",
    "hammer_head_pos_w",
    "assigned_reference_waypoint_w",
    "straight_reference_waypoint_w",
    "imitation_eligible",
    "delta_impulse",
  }
  if not required.issubset(trace):
    raise RuntimeError("horizontal route trace is missing required telemetry")
  scalar_shape = np.asarray(trace["episode_id"]).shape
  if len(scalar_shape) != 2:
    raise RuntimeError("horizontal route trace episode shape is invalid")
  if np.asarray(trace["route_sign"]).shape != scalar_shape or not np.array_equal(
    trace["route_sign"], np.full(scalar_shape, route_sign, dtype=np.asarray(trace["route_sign"]).dtype)
  ):
    raise RuntimeError("horizontal route trace does not retain the forced sign")
  for name in ("strike_phase", "imitation_eligible", "delta_impulse"):
    if np.asarray(trace[name]).shape != scalar_shape:
      raise RuntimeError(f"horizontal route trace {name} shape is invalid")
  for name in (
    "hammer_head_pos_w",
    "assigned_reference_waypoint_w",
    "straight_reference_waypoint_w",
  ):
    if np.asarray(trace[name]).shape != (*scalar_shape, 3):
      raise RuntimeError(f"horizontal route trace {name} shape is invalid")
  if np.asarray(trace["imitation_eligible"]).dtype != np.bool_:
    raise RuntimeError("horizontal route imitation gate is not boolean")
  if not np.array_equal(trace["delta_impulse"], np.zeros(scalar_shape)):
    raise RuntimeError("horizontal route evaluation did not keep impulse CaT log-only")
  for name, value in trace.items():
    array = np.asarray(value)
    if array.dtype.hasobject:
      raise RuntimeError(f"horizontal route trace {name} has object dtype")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
      raise RuntimeError(f"horizontal route trace {name} is non-finite")


def _population_payload(
  trace: dict[str, np.ndarray], protocol: dict[str, object], *, route_sign: int
) -> dict[str, object]:
  _validate_route_trace(trace, route_sign=route_sign)
  if protocol.get("forced_route_sign") != route_sign:
    raise RuntimeError("population protocol forced-route identity drifted")
  if protocol.get("actor_only_checkpoint_load") is not True:
    raise RuntimeError("population protocol must record actor-only checkpoint loading")
  training = protocol.get("training_config")
  if training != {
    "imp_max_p": 0.2,
    "imp_limit_n_m_s": list(TRAINING_CAPS_N_M_S),
  }:
    raise RuntimeError("population protocol training treatment identity drifted")
  cat = protocol.get("cat_replay")
  if not isinstance(cat, Mapping) or cat.get("imp_max_p_live") != 0.0:
    raise RuntimeError("population protocol must record live impulse p=0")
  return {"protocol": protocol}


def run_horizontal_route_evaluation(
  *,
  checkpoint: Path,
  role: str,
  route_sign: int,
  output_dir: Path,
  device: str,
) -> dict[str, object]:
  """Run one immutable actor on one forced route and four frozen populations."""
  if type(route_sign) is not int or route_sign not in ROUTE_LABEL_BY_SIGN:
    raise ValueError("route sign must be one of -1, 0, +1")
  checkpoint, checkpoint_sha = validate_checkpoint(checkpoint, role=role)
  output_dir = Path(output_dir).absolute()
  if output_dir.exists() or output_dir.is_symlink():
    raise FileExistsError(f"refusing to overwrite horizontal evaluation output: {output_dir}")
  revisions = _evaluation_revisions()
  revisions["code_revision"] = validate_evaluation_revision(
    survey._REPO_ROOT, revisions["code_revision"]
  )
  if revisions.get("asset_revision") != EXPECTED_ASSET_REVISION:
    raise RuntimeError("horizontal evaluation asset revision mismatch")
  output_dir.mkdir(parents=True)
  task_id = str(POLICY_SPECS[role]["task"])

  def run_population(*, seed: int, num_envs: int, steps: int | None, stochastic: bool):
    trace, protocol = survey._run_population(
      checkpoint=checkpoint,
      device=device,
      num_envs=num_envs,
      seed=seed,
      rng_seeds=survey.EvaluationRngSeeds.from_evaluation_seed(seed),
      steps=steps,
      stochastic=stochastic,
      task_id=task_id,
      source_imp_max_p=0.2,
      source_imp_limit_n_m_s=TRAINING_CAPS_N_M_S,
      forced_route_sign=route_sign,
    )
    return trace, _population_payload(trace, protocol, route_sign=route_sign)

  fixed_trace, fixed_payload = run_population(
    seed=survey.FIXED_SEED, num_envs=survey.FIXED_ENVS, steps=None, stochastic=False
  )
  np.savez_compressed(output_dir / "fixed_trace.npz", **fixed_trace)
  stochastic_payloads: dict[str, object] = {}
  for seed in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS:
    trace, population = run_population(
      seed=seed,
      num_envs=survey.TRAINING_LIKE_ENVS,
      steps=survey.TRAINING_LIKE_STEPS,
      stochastic=True,
    )
    trace_name = f"training_like_seed_{seed}_trace.npz"
    np.savez_compressed(output_dir / trace_name, **trace)
    stochastic_payloads[str(seed)] = {"trace": trace_name, **population}

  payload: dict[str, object] = {
    "schema_version": 1,
    "purpose": "frozen actor-only diagnostic of one forced horizontal strike route",
    "task": task_id,
    "checkpoint": {"role": role, "path": str(checkpoint), "sha256": checkpoint_sha},
    **revisions,
    "route": {"label": ROUTE_LABEL_BY_SIGN[route_sign], "sign": route_sign},
    "training_cap_identity_n_m_s": list(TRAINING_CAPS_N_M_S),
    "protocol": {
      "actor_only_checkpoint_load": True,
      "live_imp_max_p": 0.0,
      "fixed_seed": survey.FIXED_SEED,
      "stochastic_seeds": list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS),
      "stochastic_shape": [survey.TRAINING_LIKE_ENVS, survey.TRAINING_LIKE_STEPS],
      "route_sign_forced_after_sampler_on_every_reset": True,
      "primary_route_unit": "initial episode with eligible core samples",
      "controller_reads_are_independent": False,
    },
    "populations": {
      "fixed_mean": {"trace": "fixed_trace.npz", **fixed_payload},
      "training_like_sampled": stochastic_payloads,
    },
  }
  (output_dir / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
  )
  return payload


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-role", required=True, choices=tuple(POLICY_SPECS))
  parser.add_argument("--route-sign", required=True, type=int, choices=(-1, 0, 1))
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--device", default="cpu")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  run_horizontal_route_evaluation(
    checkpoint=args.checkpoint,
    role=args.checkpoint_role,
    route_sign=args.route_sign,
    output_dir=args.output_dir,
    device=args.device,
  )
  print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
  main()
