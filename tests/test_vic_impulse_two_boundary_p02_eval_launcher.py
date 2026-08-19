"""Shell contract for the frozen two-boundary p=0.2 diagnostic evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.impulse_cat_activation_survey import (
  EVALUATION_CHECKPOINTS,
  TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S,
)
from tests import test_vic_impulse_dose_curve_eval_launcher as qualified


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_two_boundary_p02_eval.sbatch"
ARMS = (
  (
    0,
    "p02_uniform09",
    "UNIFORM09_CHECKPOINT",
    "a99593b263a74944d60ac412bb1da733a36a29a1cd9f4eeeaed89906372595df",
    "[0.738,1.476,0.738,0.738,0.738,0.738]",
  ),
  (
    1,
    "p02_joint_stress",
    "JOINT_STRESS_CHECKPOINT",
    "509cc26a2e521a935bcbc8c342040c95c7d2e3d7fc105a52d5d9450518bd2ec7",
    "[0.369,0.246,0.738,0.369,0.246,0.0164]",
  ),
)


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(qualified, "LAUNCHER", LAUNCHER)
  monkeypatch.setattr(qualified, "ARMS", tuple(row[:4] for row in ARMS))


def _env(
  tmp_path: Path, task_id: int, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
  _configure(monkeypatch)
  env = qualified._env(tmp_path, task_id)
  env.update(
    {
      "SLURM_ARRAY_TASK_COUNT": "2",
      "SLURM_ARRAY_TASK_MIN": "0",
      "SLURM_ARRAY_TASK_MAX": "1",
    }
  )
  return env


def test_two_boundary_eval_binds_exact_role_hash_cap_identity_and_artifacts(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  calls: list[list[str]] = []
  for task_id, role, checkpoint_variable, frozen_sha, caps in ARMS:
    assert EVALUATION_CHECKPOINTS[role] == frozen_sha
    assert list(TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S[role]) == pytest.approx(
      json.loads(caps)
    )
    env = _env(tmp_path / role, task_id, monkeypatch)
    result = qualified._run(env)
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Z1_VIC_IMPULSE_TWO_BOUNDARY_P02_EVAL_PASS role={role}" in result.stdout
    call = qualified._survey_call(env)
    assert qualified._value(call, "--checkpoint-role") == role
    assert qualified._value(call, "--checkpoint") == env[checkpoint_variable]
    assert qualified._value(call, "--training-caps-n-m-s") == caps
    assert qualified._value(call, "--device") == "cuda:0"
    output = Path(qualified._value(call, "--output-dir"))
    assert sorted(path.name for path in output.iterdir()) == sorted(qualified.OUTPUTS)
    assert (output.parent / "SHA256SUMS").read_text() == "".join(
      f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  evaluation/{name}\n"
      for name in qualified.OUTPUTS
    )
    calls.append(call)

  normalized = []
  for call in calls:
    copy = list(call)
    copy[0] = "<repo>/scripts/impulse_cat_activation_survey.py"
    for flag, value in (
      ("--checkpoint-role", "<role>"),
      ("--checkpoint", "<checkpoint>"),
      ("--training-caps-n-m-s", "<training-cap-identity>"),
      ("--output-dir", "<output>"),
    ):
      copy[copy.index(flag) + 1] = value
    normalized.append(copy)
  assert normalized[0] == normalized[1]


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("SLURM_ARRAY_TASK_COUNT", "3"),
    ("SLURM_ARRAY_TASK_MIN", "1"),
    ("SLURM_ARRAY_TASK_MAX", "2"),
    ("SLURM_ARRAY_TASK_STEP", "2"),
    ("SLURM_ARRAY_TASK_ID", "2"),
    ("SLURM_JOB_ID", "bad"),
    ("WORLD_SIZE", "1"),
    ("RESUME_PATH", "/tmp/model.pt"),
  ),
)
def test_two_boundary_eval_rejects_malformed_execution_context(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  variable: str,
  value: str,
):
  env = _env(tmp_path / variable, 0, monkeypatch)
  env[variable] = value
  assert qualified._run(env).returncode == 2


@pytest.mark.parametrize("task_id", (0, 1))
def test_two_boundary_eval_rejects_checkpoint_drift_nonzero_or_malformed_telemetry(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_id: int
):
  env = _env(tmp_path / f"hash-{task_id}", task_id, monkeypatch)
  checkpoint_variable = ARMS[task_id][2]
  env[f"{checkpoint_variable}_FROZEN_SHA"] = "0" * 64
  result = qualified._run(env)
  assert result.returncode == 2
  assert "role/checkpoint SHA-256 mismatch" in result.stdout

  for mode in ("nonzero_impulse", "nonfinite", "wrong_delta", "wrong_shape"):
    env = _env(tmp_path / f"{mode}-{task_id}", task_id, monkeypatch)
    env["BAD_OUTPUT_MODE"] = mode
    result = qualified._run(env)
    assert result.returncode == 2
    assert "Z1_VIC_IMPULSE_TWO_BOUNDARY_P02_EVAL_PASS" not in result.stdout
