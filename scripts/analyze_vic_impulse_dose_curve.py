#!/usr/bin/env python3
"""Preregistered four-policy analysis for the Z1 impulse-CaT dose curve."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts import analyze_vic_impulse_diag90_500_evaluation as historical
from scripts import analyze_vic_impulse_p02_bridge_evaluation as bridge
from scripts import impulse_cat_activation_survey as survey


EXPECTED_ROLES = (
  "dose_p0_control",
  "dose_p01_target",
  "dose_p02_target",
  "dose_p03_target",
)
EXPECTED_CHECKPOINTS = MappingProxyType(
  {role: survey.EVALUATION_CHECKPOINTS[role] for role in EXPECTED_ROLES}
)
EXPECTED_IMP_MAX_P = MappingProxyType(
  {
    "dose_p0_control": 0.0,
    "dose_p01_target": 0.1,
    "dose_p02_target": 0.2,
    "dose_p03_target": 0.3,
  }
)
TRAINING_SEED = 2


def evaluate_gate_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
  """Delegate every dose gate to the approved bridge comparator."""
  return bridge.evaluate_gate_metrics(metrics)


def validate_checkpoint_identities(
  summaries: Mapping[str, Mapping[str, object]],
  checkpoint_paths: Mapping[str, Path],
) -> dict[str, str]:
  """Bind every summary to one explicit canonical frozen checkpoint path."""
  if tuple(summaries) != EXPECTED_ROLES or tuple(checkpoint_paths) != EXPECTED_ROLES:
    raise ValueError("checkpoint identity requires the exact ordered policy roles")
  canonical: dict[str, str] = {}
  for role in EXPECTED_ROLES:
    path = Path(checkpoint_paths[role])
    if path.name != "model_499.pt" or path.is_symlink() or not path.is_file():
      raise ValueError("checkpoint must be a regular non-symlink model_499.pt")
    canonical_path = str(path.resolve(strict=True))
    checkpoint = summaries[role].get("checkpoint")
    expected_checkpoint = {
      "role": role,
      "path": canonical_path,
      "sha256": EXPECTED_CHECKPOINTS[role],
    }
    if (
      not isinstance(checkpoint, Mapping)
      or set(checkpoint) != set(expected_checkpoint)
      or checkpoint != expected_checkpoint
    ):
      raise ValueError("summary checkpoint must contain exact role, path, and SHA")
    canonical[role] = canonical_path
  if len(set(canonical.values())) != len(EXPECTED_ROLES):
    raise ValueError("each policy role requires a distinct checkpoint path")
  return canonical


def compact_protocol_identity(
  summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
  """Extract one canonical matched protocol identity from the validated summaries."""
  if tuple(summaries) != EXPECTED_ROLES:
    raise ValueError("protocol identity requires the exact ordered policy roles")
  protocols = [summaries[role].get("protocol") for role in EXPECTED_ROLES]
  if not all(isinstance(protocol, Mapping) for protocol in protocols) or any(
    protocol != protocols[0] for protocol in protocols[1:]
  ):
    raise ValueError("all dose summaries require one matched evaluation protocol")
  protocol = protocols[0]
  if (
    protocol.get("live_imp_max_p") != 0.0
    or protocol.get("fixed_seed") != survey.FIXED_SEED
    or protocol.get("stochastic_seeds")
    != list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS)
  ):
    raise ValueError("dose summaries drifted from the canonical evaluation seeds or live dose")

  def population_identity(slot: str, *, seed: int) -> dict[str, object]:
    identities: list[dict[str, object]] = []
    for role in EXPECTED_ROLES:
      populations = summaries[role].get("populations")
      if not isinstance(populations, Mapping):
        raise ValueError("dose summary is missing population identity")
      if slot == "fixed_mean":
        population = populations.get(slot)
      else:
        sampled = populations.get("training_like_sampled")
        population = sampled.get(slot) if isinstance(sampled, Mapping) else None
      population_protocol = (
        population.get("protocol") if isinstance(population, Mapping) else None
      )
      if not isinstance(population_protocol, Mapping):
        raise ValueError("dose summary is missing population protocol identity")
      identity = {
        "seed": population_protocol.get("seed"),
        "initial_population_sha256": population_protocol.get(
          "initial_population_sha256"
        ),
        "rng_streams": population_protocol.get("rng_streams"),
      }
      identities.append(identity)
    if any(identity != identities[0] for identity in identities[1:]):
      raise ValueError("population or RNG identity is not matched across dose roles")
    identity = identities[0]
    if identity["seed"] != seed:
      raise ValueError("population identity seed drifted")
    return identity

  control_populations = summaries[EXPECTED_ROLES[0]].get("populations")
  fixed = (
    control_populations.get("fixed_mean")
    if isinstance(control_populations, Mapping)
    else None
  )
  provisional = fixed.get("provisional_caps") if isinstance(fixed, Mapping) else None
  if not isinstance(provisional, Mapping) or provisional.get("caps_n_m_s") != list(
    survey.PROVISIONAL_CAPS_N_M_S
  ):
    raise ValueError("protocol identity requires the exact provisional cap vector")
  identity: dict[str, object] = {
    "live_imp_max_p": 0.0,
    "provisional_caps_n_m_s": list(survey.PROVISIONAL_CAPS_N_M_S),
    "fixed_seed": survey.FIXED_SEED,
    "stochastic_seeds": list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS),
    "training_seed": TRAINING_SEED,
    "training_seed_source": "frozen seed-2 checkpoint role bindings",
    "populations": {
      "fixed_mean": population_identity("fixed_mean", seed=survey.FIXED_SEED),
      "training_like_by_seed": {
        str(seed): population_identity(str(seed), seed=seed)
        for seed in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS
      },
    },
  }
  encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
  identity["identity_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
  return identity


def assemble_analysis(
  *, pair_results: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
  """Assemble three independent control comparisons without cross-dose inference."""
  target_roles = EXPECTED_ROLES[1:]
  if tuple(pair_results) != target_roles:
    raise ValueError("analysis requires the exact ordered dose roles")
  seed_keys = tuple(str(seed) for seed in bridge.STOCHASTIC_SEEDS)
  dose_results: dict[str, object] = {}
  for role in target_roles:
    pair = pair_results[role]
    fixed = pair.get("fixed_mean_descriptive_only")
    stochastic = pair.get("training_like_by_seed")
    if (
      not isinstance(fixed, Mapping)
      or fixed.get("descriptive_only") is not True
      or not isinstance(stochastic, Mapping)
      or tuple(stochastic) != seed_keys
    ):
      raise ValueError("each dose requires the exact separate stochastic populations")
    passes: list[bool] = []
    for population in stochastic.values():
      verdict = population.get("verdict") if isinstance(population, Mapping) else None
      if not isinstance(verdict, Mapping) or type(verdict.get("pass")) is not bool:
        raise ValueError("each stochastic population requires a boolean bridge verdict")
      passes.append(verdict["pass"])
    dose_results[role] = {
      "imp_max_p_during_training": EXPECTED_IMP_MAX_P[role],
      "overall_verdict": {
        "pass": all(passes),
        "requires_every_stochastic_population": True,
      },
      "fixed_mean_descriptive_only": dict(fixed),
      "training_like_by_seed": dict(stochastic),
    }
  return {
    "schema_version": 1,
    "purpose": "preregistered native-observed Z1 impulse-CaT dose screen",
    "analysis_protocol": {
      "control_role": EXPECTED_ROLES[0],
      "dose_roles": list(target_roles),
      "fixed_mean_is_inferential": False,
      "stochastic_populations_reported_separately": True,
      "bootstrap_resamples": bridge.BOOTSTRAP_RESAMPLES,
      "bootstrap_seed": bridge.BOOTSTRAP_SEED,
      "resampling_unit": "whole paired environment ID",
    },
    "dose_results": dose_results,
    "claim_limits": {
      "training_seeds": 1,
      "native_observed_horizon_only": True,
      "soft_pressure_not_clamp": True,
      "provisional_project_caps_not_manufacturer_limits": True,
      "hardware_safety_claim_authorized": False,
      "complete_contact_claim_authorized": False,
      "optimal_dose_claim_authorized": False,
    },
  }


def analyze_evaluation_curve(
  leaves: Mapping[str, Path],
  *,
  checkpoint_paths: Mapping[str, Path],
  expected_code_revision: str,
) -> dict[str, object]:
  """Validate four frozen leaves and compare every dose with the same control."""
  if tuple(leaves) != EXPECTED_ROLES:
    raise ValueError("analysis requires the exact ordered policy roles")
  resolved = {
    role: Path(leaves[role]).resolve(strict=True) for role in EXPECTED_ROLES
  }
  if len(set(resolved.values())) != len(EXPECTED_ROLES):
    raise ValueError("each policy role requires a distinct evaluation leaf")

  manifests = {
    role: historical.validate_leaf_manifest(leaf)
    for role, leaf in resolved.items()
  }
  summaries: dict[str, dict[str, object]] = {}
  for role, leaf in resolved.items():
    summary = json.loads(
      (leaf / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    historical._require_finite_json(summary)
    if summary.get("schema_version") != 2:
      raise ValueError("dose evaluation requires summary schema version 2")
    summaries[role] = summary
  canonical_checkpoints = validate_checkpoint_identities(summaries, checkpoint_paths)

  control_role = EXPECTED_ROLES[0]
  control_summary = summaries[control_role]
  pair_results: dict[str, Mapping[str, object]] = {}
  for target_role in EXPECTED_ROLES[1:]:
    target_summary = summaries[target_role]
    survey.compare_policy_evaluations(
      control_summary,
      target_summary,
      expected_code_revision=expected_code_revision,
      expected_roles=(control_role, target_role),
    )
    control_leaf = resolved[control_role]
    target_leaf = resolved[target_role]
    fixed = bridge.evaluate_population_pair(
      historical._load_trace(control_leaf / "evaluation" / "fixed_trace.npz"),
      historical._load_trace(target_leaf / "evaluation" / "fixed_trace.npz"),
      control_summary["populations"]["fixed_mean"],
      target_summary["populations"]["fixed_mean"],
      inferential=False,
    )
    stochastic: dict[str, object] = {}
    for seed in bridge.STOCHASTIC_SEEDS:
      seed_key = str(seed)
      trace_name = f"training_like_seed_{seed}_trace.npz"
      stochastic[seed_key] = bridge.evaluate_population_pair(
        historical._load_trace(control_leaf / "evaluation" / trace_name),
        historical._load_trace(target_leaf / "evaluation" / trace_name),
        control_summary["populations"]["training_like_sampled"][seed_key],
        target_summary["populations"]["training_like_sampled"][seed_key],
        inferential=True,
      )
    pair_results[target_role] = {
      "fixed_mean_descriptive_only": fixed,
      "training_like_by_seed": stochastic,
    }

  payload = assemble_analysis(pair_results=pair_results)
  payload["protocol_identity"] = compact_protocol_identity(summaries)
  payload["provenance"] = {
    "expected_roles": list(EXPECTED_ROLES),
    "expected_checkpoint_sha256": dict(EXPECTED_CHECKPOINTS),
    "canonical_checkpoint_paths": canonical_checkpoints,
    "imp_max_p_during_training": dict(EXPECTED_IMP_MAX_P),
    "code_revision": expected_code_revision,
    "asset_revision": survey.EXPECTED_EVALUATION_ASSET_REVISION,
    "independently_rehashed_artifacts": manifests,
    "manifest_sha256": {
      role: historical._sha256(leaf / "SHA256SUMS")
      for role, leaf in resolved.items()
    },
    "identity_finiteness_native_claim_gate": True,
  }
  historical._require_finite_json(payload)
  return payload


def encode_analysis(payload: Mapping[str, object]) -> str:
  """Encode a finite analysis deterministically for immutable evidence."""
  historical._require_finite_json(payload)
  return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_analysis(
  output: Path,
  payload: Mapping[str, object],
  *,
  immutable_leaves: tuple[Path, ...],
) -> None:
  """Create one fresh output outside every immutable evaluator leaf."""
  output = Path(output)
  if output.exists() or output.is_symlink():
    raise FileExistsError(f"analysis output must be fresh: {output}")
  output_parent = output.parent.resolve(strict=True)
  resolved_output = output_parent / output.name
  for leaf in immutable_leaves:
    resolved_leaf = Path(leaf).resolve(strict=True)
    if resolved_output == resolved_leaf or resolved_leaf in resolved_output.parents:
      raise ValueError("analysis output must remain outside every immutable input leaf")
  with resolved_output.open("x", encoding="utf-8") as handle:
    handle.write(encode_analysis(payload))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dose-p0-control-leaf", required=True, type=Path)
  parser.add_argument("--dose-p01-target-leaf", required=True, type=Path)
  parser.add_argument("--dose-p02-target-leaf", required=True, type=Path)
  parser.add_argument("--dose-p03-target-leaf", required=True, type=Path)
  parser.add_argument("--dose-p0-control-checkpoint", required=True, type=Path)
  parser.add_argument("--dose-p01-target-checkpoint", required=True, type=Path)
  parser.add_argument("--dose-p02-target-checkpoint", required=True, type=Path)
  parser.add_argument("--dose-p03-target-checkpoint", required=True, type=Path)
  parser.add_argument("--expected-code-revision", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  leaves = {
    role: getattr(args, f"{role}_leaf")
    for role in EXPECTED_ROLES
  }
  checkpoint_paths = {
    role: getattr(args, f"{role}_checkpoint")
    for role in EXPECTED_ROLES
  }
  payload = analyze_evaluation_curve(
    leaves,
    checkpoint_paths=checkpoint_paths,
    expected_code_revision=args.expected_code_revision,
  )
  write_analysis(
    args.output,
    payload,
    immutable_leaves=tuple(leaves.values()),
  )
  print(args.output)


if __name__ == "__main__":
  main()
