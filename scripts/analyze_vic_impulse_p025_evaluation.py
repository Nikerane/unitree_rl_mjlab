#!/usr/bin/env python3
"""Matched native-horizon analysis for the exploratory Z1 impulse-CaT p=0.25 policy."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from types import MappingProxyType


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts import analyze_vic_impulse_diag90_500_evaluation as historical
from scripts import analyze_vic_impulse_dose_curve as dose
from scripts import analyze_vic_impulse_p02_bridge_evaluation as bridge
from scripts import impulse_cat_activation_survey as survey


EXPECTED_ROLES = ("dose_p0_control", "dose_p025_exploratory")
EXPECTED_CHECKPOINTS = MappingProxyType(
  {role: survey.EVALUATION_CHECKPOINTS[role] for role in EXPECTED_ROLES}
)
TARGET_IMP_MAX_P = 0.25


def _load_summary(leaf: Path) -> dict[str, object]:
  summary = json.loads(
    (leaf / "evaluation" / "summary.json").read_text(encoding="utf-8")
  )
  historical._require_finite_json(summary)
  if summary.get("schema_version") != 2:
    raise ValueError("p=0.25 evaluation requires summary schema version 2")
  return summary


def _validate_checkpoint_identities(
  summaries: Mapping[str, Mapping[str, object]],
  checkpoint_paths: Mapping[str, Path],
) -> dict[str, str]:
  if tuple(summaries) != EXPECTED_ROLES or tuple(checkpoint_paths) != EXPECTED_ROLES:
    raise ValueError("checkpoint identity requires the exact ordered pair roles")
  canonical: dict[str, str] = {}
  for role in EXPECTED_ROLES:
    path = Path(checkpoint_paths[role])
    if path.name != "model_499.pt" or path.is_symlink() or not path.is_file():
      raise ValueError("checkpoint must be a regular non-symlink model_499.pt")
    if historical._sha256(path) != EXPECTED_CHECKPOINTS[role]:
      raise ValueError(f"{role} checkpoint file SHA-256 does not match the frozen role")
    canonical_path = str(path.resolve(strict=True))
    expected = {
      "role": role,
      "path": canonical_path,
      "sha256": EXPECTED_CHECKPOINTS[role],
    }
    checkpoint = summaries[role].get("checkpoint")
    if (
      not isinstance(checkpoint, Mapping)
      or set(checkpoint) != set(expected)
      or checkpoint != expected
    ):
      raise ValueError("summary checkpoint must contain exact role, path, and SHA")
    canonical[role] = canonical_path
  if len(set(canonical.values())) != 2:
    raise ValueError("control and target require distinct checkpoint paths")
  return canonical


def _protocol_identity(
  summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
  protocols = [summaries[role].get("protocol") for role in EXPECTED_ROLES]
  if (
    not all(isinstance(protocol, Mapping) for protocol in protocols)
    or protocols[0] != protocols[1]
    or protocols[0].get("live_imp_max_p") != 0.0
    or protocols[0].get("fixed_seed") != survey.FIXED_SEED
    or protocols[0].get("stochastic_seeds")
    != list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS)
  ):
    raise ValueError("control and p=0.25 require one exact live-log-only protocol")

  identities: dict[str, object] = {}
  for label, seed in (
    ("fixed_mean", survey.FIXED_SEED),
    *((str(value), value) for value in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS),
  ):
    observed = []
    for role in EXPECTED_ROLES:
      populations = summaries[role].get("populations")
      if not isinstance(populations, Mapping):
        raise ValueError("evaluation population identity is missing")
      if label == "fixed_mean":
        population = populations.get("fixed_mean")
      else:
        sampled = populations.get("training_like_sampled")
        population = sampled.get(label) if isinstance(sampled, Mapping) else None
      protocol = population.get("protocol") if isinstance(population, Mapping) else None
      if not isinstance(protocol, Mapping):
        raise ValueError("evaluation population protocol is missing")
      observed.append(
        {
          "seed": protocol.get("seed"),
          "initial_population_sha256": protocol.get("initial_population_sha256"),
          "rng_streams": protocol.get("rng_streams"),
        }
      )
    if observed[0] != observed[1] or observed[0]["seed"] != seed:
      raise ValueError("evaluation population or RNG identity is not matched")
    identities[label] = observed[0]
  return {
    "live_imp_max_p": 0.0,
    "provisional_caps_n_m_s": list(survey.PROVISIONAL_CAPS_N_M_S),
    "fixed_seed": survey.FIXED_SEED,
    "stochastic_seeds": list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS),
    "populations": identities,
  }


def analyze_evaluation_pair(
  control_leaf: Path,
  target_leaf: Path,
  *,
  checkpoint_paths: Mapping[str, Path],
  expected_code_revision: str,
) -> dict[str, object]:
  """Validate and compare the exact p=0/p=0.25 frozen policy pair."""
  leaves = {
    EXPECTED_ROLES[0]: Path(control_leaf).resolve(strict=True),
    EXPECTED_ROLES[1]: Path(target_leaf).resolve(strict=True),
  }
  if len(set(leaves.values())) != 2:
    raise ValueError("control and target require distinct evaluation leaves")
  manifests = {
    role: historical.validate_leaf_manifest(leaf) for role, leaf in leaves.items()
  }
  summaries = {role: _load_summary(leaf) for role, leaf in leaves.items()}
  canonical_checkpoints = _validate_checkpoint_identities(summaries, checkpoint_paths)
  survey.compare_policy_evaluations(
    summaries[EXPECTED_ROLES[0]],
    summaries[EXPECTED_ROLES[1]],
    expected_code_revision=expected_code_revision,
    expected_roles=EXPECTED_ROLES,
  )

  fixed = bridge.evaluate_population_pair(
    historical._load_trace(leaves[EXPECTED_ROLES[0]] / "evaluation/fixed_trace.npz"),
    historical._load_trace(leaves[EXPECTED_ROLES[1]] / "evaluation/fixed_trace.npz"),
    summaries[EXPECTED_ROLES[0]]["populations"]["fixed_mean"],
    summaries[EXPECTED_ROLES[1]]["populations"]["fixed_mean"],
    inferential=False,
  )
  stochastic: dict[str, object] = {}
  for seed in bridge.STOCHASTIC_SEEDS:
    key = str(seed)
    filename = f"training_like_seed_{seed}_trace.npz"
    stochastic[key] = bridge.evaluate_population_pair(
      historical._load_trace(leaves[EXPECTED_ROLES[0]] / "evaluation" / filename),
      historical._load_trace(leaves[EXPECTED_ROLES[1]] / "evaluation" / filename),
      summaries[EXPECTED_ROLES[0]]["populations"]["training_like_sampled"][key],
      summaries[EXPECTED_ROLES[1]]["populations"]["training_like_sampled"][key],
      inferential=True,
    )
  passes = [bool(result["verdict"]["pass"]) for result in stochastic.values()]
  payload: dict[str, object] = {
    "schema_version": 1,
    "purpose": "exploratory native-observed Z1 impulse-CaT p=0.25 screen",
    "target_imp_max_p_during_training": TARGET_IMP_MAX_P,
    "overall_verdict": {
      "pass": all(passes),
      "requires_every_stochastic_population": True,
    },
    "fixed_mean_descriptive_only": fixed,
    "training_like_by_seed": stochastic,
    "protocol_identity": _protocol_identity(summaries),
    "provenance": {
      "expected_roles": list(EXPECTED_ROLES),
      "expected_checkpoint_sha256": dict(EXPECTED_CHECKPOINTS),
      "canonical_checkpoint_paths": canonical_checkpoints,
      "code_revision": expected_code_revision,
      "asset_revision": survey.EXPECTED_EVALUATION_ASSET_REVISION,
      "independently_rehashed_artifacts": manifests,
      "manifest_sha256": {
        role: historical._sha256(leaf / "SHA256SUMS")
        for role, leaf in leaves.items()
      },
      "identity_finiteness_native_claim_gate": True,
    },
    "claim_limits": {
      "training_seeds": 1,
      "exploratory_nonpreregistered_dose": True,
      "native_observed_horizon_only": True,
      "soft_pressure_not_clamp": True,
      "provisional_project_caps_not_manufacturer_limits": True,
      "hardware_safety_claim_authorized": False,
      "complete_contact_claim_authorized": False,
      "optimal_dose_claim_authorized": False,
    },
  }
  historical._require_finite_json(payload)
  return payload


def encode_analysis(payload: Mapping[str, object]) -> str:
  return dose.encode_analysis(payload)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--control-leaf", required=True, type=Path)
  parser.add_argument("--target-leaf", required=True, type=Path)
  parser.add_argument("--control-checkpoint", required=True, type=Path)
  parser.add_argument("--target-checkpoint", required=True, type=Path)
  parser.add_argument("--expected-code-revision", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  checkpoint_paths = {
    EXPECTED_ROLES[0]: args.control_checkpoint,
    EXPECTED_ROLES[1]: args.target_checkpoint,
  }
  payload = analyze_evaluation_pair(
    args.control_leaf,
    args.target_leaf,
    checkpoint_paths=checkpoint_paths,
    expected_code_revision=args.expected_code_revision,
  )
  dose.write_analysis(
    args.output,
    payload,
    immutable_leaves=(args.control_leaf, args.target_leaf),
  )
  print(args.output)


if __name__ == "__main__":
  main()
