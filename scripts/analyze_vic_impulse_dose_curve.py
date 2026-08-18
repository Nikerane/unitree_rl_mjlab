#!/usr/bin/env python3
"""Preregistered four-policy analysis for the Z1 impulse-CaT dose curve."""

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


def evaluate_gate_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
  """Delegate every dose gate to the approved bridge comparator."""
  return bridge.evaluate_gate_metrics(metrics)


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
  leaves: Mapping[str, Path], *, expected_code_revision: str
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
  payload["provenance"] = {
    "expected_roles": list(EXPECTED_ROLES),
    "expected_checkpoint_sha256": dict(EXPECTED_CHECKPOINTS),
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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dose-p0-control-leaf", required=True, type=Path)
  parser.add_argument("--dose-p01-target-leaf", required=True, type=Path)
  parser.add_argument("--dose-p02-target-leaf", required=True, type=Path)
  parser.add_argument("--dose-p03-target-leaf", required=True, type=Path)
  parser.add_argument("--expected-code-revision", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  leaves = {
    role: getattr(args, f"{role}_leaf")
    for role in EXPECTED_ROLES
  }
  payload = analyze_evaluation_curve(
    leaves, expected_code_revision=args.expected_code_revision
  )
  args.output.write_text(encode_analysis(payload), encoding="utf-8")
  print(args.output)


if __name__ == "__main__":
  main()
