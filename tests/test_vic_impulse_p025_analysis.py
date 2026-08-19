"""Preregistered pair-analysis contract for exploratory p=0.25."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import MappingProxyType

import pytest

from scripts import impulse_cat_activation_survey as survey
from tests import test_vic_impulse_dose_curve_analysis as fixtures


analysis = importlib.import_module("scripts.analyze_vic_impulse_p025_evaluation")


ROLES = ("dose_p0_control", "dose_p025_exploratory")
CODE_REVISION = "a" * 40


def _pair(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], dict[str, Path]]:
  checkpoints: dict[str, Path] = {}
  hashes: dict[str, str] = {}
  for role in ROLES:
    checkpoint = tmp_path / "checkpoints" / role / "model_499.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"{role} synthetic checkpoint\n".encode())
    checkpoints[role] = checkpoint
    hashes[role] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
  frozen = MappingProxyType(hashes)
  monkeypatch.setattr(analysis, "EXPECTED_CHECKPOINTS", frozen)
  monkeypatch.setattr(fixtures.analysis, "EXPECTED_CHECKPOINTS", frozen)
  combined = dict(survey.EVALUATION_CHECKPOINTS)
  combined.update(hashes)
  monkeypatch.setattr(survey, "EVALUATION_CHECKPOINTS", MappingProxyType(combined))
  leaves = {
    role: fixtures._write_leaf(tmp_path / "leaves", role, checkpoints[role])
    for role in ROLES
  }
  return leaves, checkpoints


def test_p025_analysis_binds_exact_roles_hashes_and_training_dose():
  assert analysis.EXPECTED_ROLES == ROLES
  assert dict(analysis.EXPECTED_CHECKPOINTS) == {
    "dose_p0_control": (
      "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3"
    ),
    "dose_p025_exploratory": (
      "5efc45c11c02dd400ad7d417bcdeefe9d271038ab43007f08a2820ceca0e744d"
    ),
  }
  assert analysis.TARGET_IMP_MAX_P == 0.25


def test_p025_pair_analysis_is_finite_separate_and_byte_deterministic(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  leaves, checkpoints = _pair(tmp_path, monkeypatch)

  first = analysis.analyze_evaluation_pair(
    leaves[ROLES[0]],
    leaves[ROLES[1]],
    checkpoint_paths=checkpoints,
    expected_code_revision=CODE_REVISION,
  )
  second = analysis.analyze_evaluation_pair(
    leaves[ROLES[0]],
    leaves[ROLES[1]],
    checkpoint_paths=checkpoints,
    expected_code_revision=CODE_REVISION,
  )

  assert first["overall_verdict"] == {
    "pass": True,
    "requires_every_stochastic_population": True,
  }
  assert first["target_imp_max_p_during_training"] == 0.25
  assert first["fixed_mean_descriptive_only"]["descriptive_only"] is True
  assert tuple(first["training_like_by_seed"]) == ("2", "2026081701", "2026081702")
  assert first["provenance"]["expected_roles"] == list(ROLES)
  assert first["claim_limits"]["optimal_dose_claim_authorized"] is False
  assert analysis.encode_analysis(first) == analysis.encode_analysis(second)
  json.loads(analysis.encode_analysis(first))


def test_p025_pair_rejects_checkpoint_mutation_and_role_drift(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  leaves, checkpoints = _pair(tmp_path, monkeypatch)
  checkpoints[ROLES[1]].write_bytes(b"mutated\n")
  with pytest.raises(ValueError, match="checkpoint file SHA-256"):
    analysis.analyze_evaluation_pair(
      leaves[ROLES[0]],
      leaves[ROLES[1]],
      checkpoint_paths=checkpoints,
      expected_code_revision=CODE_REVISION,
    )

  leaves, checkpoints = _pair(tmp_path / "role", monkeypatch)
  summary_path = leaves[ROLES[1]] / "evaluation" / "summary.json"
  summary = json.loads(summary_path.read_text())
  summary["checkpoint"]["role"] = "dose_p03_target"
  summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
  fixtures._refresh_manifest(leaves[ROLES[1]])
  with pytest.raises(ValueError, match="exact role, path, and SHA"):
    analysis.analyze_evaluation_pair(
      leaves[ROLES[0]],
      leaves[ROLES[1]],
      checkpoint_paths=checkpoints,
      expected_code_revision=CODE_REVISION,
    )


def test_p025_analysis_cli_writes_one_fresh_output(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
):
  leaves, checkpoints = _pair(tmp_path, monkeypatch)
  output = tmp_path / "analysis.json"
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "analyze_vic_impulse_p025_evaluation.py",
      "--control-leaf",
      str(leaves[ROLES[0]]),
      "--target-leaf",
      str(leaves[ROLES[1]]),
      "--control-checkpoint",
      str(checkpoints[ROLES[0]]),
      "--target-checkpoint",
      str(checkpoints[ROLES[1]]),
      "--expected-code-revision",
      CODE_REVISION,
      "--output",
      str(output),
    ],
  )

  analysis.main()

  assert capsys.readouterr().out.strip() == str(output)
  assert json.loads(output.read_text())["target_imp_max_p_during_training"] == 0.25
  with pytest.raises(FileExistsError, match="fresh"):
    analysis.main()
