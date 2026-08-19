"""Shell contract for the matched p=0 versus exploratory p=0.25 evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.impulse_cat_activation_survey import EVALUATION_CHECKPOINTS
from tests import test_vic_impulse_dose_curve_eval_launcher as qualified


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_p025_eval.sbatch"
ARMS = (
  (
    0,
    "dose_p0_control",
    "P0_CHECKPOINT",
    "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3",
  ),
  (
    1,
    "dose_p025_exploratory",
    "P025_CHECKPOINT",
    "5efc45c11c02dd400ad7d417bcdeefe9d271038ab43007f08a2820ceca0e744d",
  ),
)


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(qualified, "LAUNCHER", LAUNCHER)
  monkeypatch.setattr(qualified, "ARMS", ARMS)


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


def test_p025_eval_maps_exact_pair_and_reuses_matched_survey(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  calls: list[list[str]] = []
  for task_id, role, checkpoint_variable, frozen_sha in ARMS:
    assert EVALUATION_CHECKPOINTS[role] == frozen_sha
    env = _env(tmp_path / role, task_id, monkeypatch)
    result = qualified._run(env)

    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Z1_VIC_IMPULSE_P025_EVAL_PASS role={role}" in result.stdout
    call = qualified._survey_call(env)
    assert qualified._value(call, "--checkpoint-role") == role
    assert qualified._value(call, "--checkpoint") == env[checkpoint_variable]
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
    normalized_call = list(call)
    normalized_call[0] = "<repo>/scripts/impulse_cat_activation_survey.py"
    normalized_call[normalized_call.index("--checkpoint-role") + 1] = "<role>"
    normalized_call[normalized_call.index("--checkpoint") + 1] = "<checkpoint>"
    normalized_call[normalized_call.index("--output-dir") + 1] = "<output>"
    normalized.append(normalized_call)
  assert normalized[0] == normalized[1]


def test_p025_eval_is_exact_two_arm_no_learning_a100_array():
  source = LAUNCHER.read_text()
  assert "#SBATCH --array=0-1" in source
  assert "#SBATCH --gres=gpu:1" in source
  assert "#SBATCH --cpus-per-task=8" in source
  assert "#SBATCH --mem=40G" in source
  assert "#SBATCH --time=00:20:00" in source
  assert "#SBATCH --no-requeue" in source
  assert "NVIDIA A100-SXM4-40GB" in source
  assert "--checkpoint-role \"$ROLE\"" in source
  assert "--device cuda:0" in source
  assert "imp_max_p=0" in source


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("SLURM_ARRAY_TASK_COUNT", "4"),
    ("SLURM_ARRAY_TASK_MIN", "1"),
    ("SLURM_ARRAY_TASK_MAX", "2"),
    ("SLURM_ARRAY_TASK_STEP", "2"),
    ("SLURM_ARRAY_TASK_ID", "2"),
    ("SLURM_JOB_ID", "bad"),
    ("WORLD_SIZE", "1"),
    ("RESUME_PATH", "/tmp/model.pt"),
  ),
)
def test_p025_eval_rejects_wrong_array_or_execution_context(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  variable: str,
  value: str,
):
  env = _env(tmp_path / variable, 0, monkeypatch)
  env[variable] = value
  assert qualified._run(env).returncode == 2


@pytest.mark.parametrize("task_id", (0, 1))
def test_p025_eval_rejects_checkpoint_drift_and_invalid_outputs(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_id: int
):
  env = _env(tmp_path / f"hash-{task_id}", task_id, monkeypatch)
  variable = ARMS[task_id][2]
  env[f"{variable}_FROZEN_SHA"] = "0" * 64
  result = qualified._run(env)
  assert result.returncode == 2
  assert "role/checkpoint SHA-256 mismatch" in result.stdout

  env = _env(tmp_path / f"delta-{task_id}", task_id, monkeypatch)
  env["BAD_OUTPUT_MODE"] = "nonzero_impulse"
  result = qualified._run(env)
  assert result.returncode == 2
  assert "Z1_VIC_IMPULSE_P025_EVAL_PASS" not in result.stdout
