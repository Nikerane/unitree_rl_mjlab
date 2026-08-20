"""Shell contract for the J6 late-checkpoint frozen evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import impulse_cat_activation_survey as survey
from tests import test_vic_impulse_dose_curve_eval_launcher as qualified


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_joint_stress_checkpoint_eval.sbatch"
CAPS = "[0.369,0.246,0.738,0.369,0.246,0.0164]"
ARMS = (
  (0, "p02_joint_stress_iter300", "MODEL_300_CHECKPOINT", "model_300.pt"),
  (1, "p02_joint_stress_iter400", "MODEL_400_CHECKPOINT", "model_400.pt"),
  (2, "p02_joint_stress_iter450", "MODEL_450_CHECKPOINT", "model_450.pt"),
  (3, "p02_joint_stress_iter499", "MODEL_499_CHECKPOINT", "model_499.pt"),
)


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(qualified, "LAUNCHER", LAUNCHER)
  monkeypatch.setattr(
    qualified,
    "ARMS",
    tuple(
      (task_id, role, variable, survey.EVALUATION_CHECKPOINTS[role])
      for task_id, role, variable, _ in ARMS
    ),
  )


def _env(tmp_path: Path, task_id: int, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
  _configure(monkeypatch)
  env = qualified._env(tmp_path, task_id)
  for _, role, variable, basename in ARMS:
    old = Path(env[variable])
    new = old.with_name(basename)
    if new != old:
      old.rename(new)
    env[variable] = str(new)
    env[f"INITIAL_{variable}_SHA"] = hashlib.sha256(new.read_bytes()).hexdigest()
  generator = Path(env["HOME"]) / "generate_evaluation.py"
  source = generator.read_text()
  source = source.replace(
    "if mode == 'wrong_role': summary['checkpoint']['role'] = 'dose_p0_control'\n",
    f"summary['training_cap_identity_n_m_s'] = {CAPS}\n"
    "if mode == 'wrong_role': summary['checkpoint']['role'] = 'dose_p0_control'\n",
  )
  generator.write_text(source)
  return env


def test_launcher_evaluates_exact_four_checkpoints_with_one_matched_protocol(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  normalized_calls = []
  for task_id, role, variable, basename in ARMS:
    env = _env(tmp_path / role, task_id, monkeypatch)
    result = qualified._run(env)
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Z1_VIC_IMPULSE_JOINT_STRESS_CHECKPOINT_EVAL_PASS role={role}" in result.stdout
    call = qualified._survey_call(env)
    assert qualified._value(call, "--checkpoint-role") == role
    assert Path(qualified._value(call, "--checkpoint")).name == basename
    assert qualified._value(call, "--training-caps-n-m-s") == CAPS
    assert qualified._value(call, "--device") == "cuda:0"
    assert Path(qualified._value(call, "--checkpoint")) == Path(env[variable]).resolve()
    normalized = list(call)
    normalized[0] = "<survey>"
    for flag, replacement in (
      ("--checkpoint-role", "<role>"),
      ("--checkpoint", "<checkpoint>"),
      ("--output-dir", "<output>"),
    ):
      normalized[normalized.index(flag) + 1] = replacement
    normalized_calls.append(normalized)
  assert normalized_calls.count(normalized_calls[0]) == 4


@pytest.mark.parametrize("task_id", range(4))
def test_launcher_rejects_checkpoint_hash_or_basename_drift(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_id: int
):
  env = _env(tmp_path / f"hash-{task_id}", task_id, monkeypatch)
  _, _, variable, _ = ARMS[task_id]
  env[f"{variable}_FROZEN_SHA"] = "0" * 64
  result = qualified._run(env)
  assert result.returncode == 2
  assert "PASS" not in result.stdout

  env = _env(tmp_path / f"basename-{task_id}", task_id, monkeypatch)
  path = Path(env[variable])
  wrong = path.with_name("model_123.pt")
  path.rename(wrong)
  env[variable] = str(wrong)
  result = qualified._run(env)
  assert result.returncode == 2
  assert "PASS" not in result.stdout


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("SLURM_ARRAY_TASK_COUNT", "3"),
    ("SLURM_ARRAY_TASK_MIN", "1"),
    ("SLURM_ARRAY_TASK_MAX", "2"),
    ("SLURM_ARRAY_TASK_STEP", "2"),
    ("SLURM_ARRAY_TASK_ID", "4"),
    ("SLURM_JOB_ID", "bad"),
    ("WORLD_SIZE", "1"),
    ("RESUME_PATH", "/tmp/model.pt"),
  ),
)
def test_launcher_rejects_malformed_execution_context(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  variable: str,
  value: str,
):
  env = _env(tmp_path / variable, 0, monkeypatch)
  env[variable] = value
  assert qualified._run(env).returncode == 2


def test_launcher_rejects_wrong_cap_identity_or_nonfinite_trace(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  env = _env(tmp_path / "caps", 0, monkeypatch)
  env["BAD_OUTPUT_MODE"] = "nonfinite"
  result = qualified._run(env)
  assert result.returncode == 2
  assert "PASS" not in result.stdout
