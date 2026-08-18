"""Shell contract for the exploratory provisional-cap impulse-CaT p=0.25 run."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import test_vic_impulse_dose_curve_train_launcher as qualified


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_p025_train.sbatch"
ROLE = "dose_p025_exploratory"


def _env(tmp_path: Path) -> dict[str, str]:
  env = qualified._env(tmp_path, 0)
  for name in (
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_TASK_MIN",
    "SLURM_ARRAY_TASK_MAX",
    "SLURM_ARRAY_TASK_STEP",
  ):
    env.pop(name)
  return env


def _run(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], *args: str):
  monkeypatch.setattr(qualified, "LAUNCHER", LAUNCHER)
  return qualified._run(env, *args)


def test_p025_launcher_runs_only_the_exact_exploratory_treatment(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  env = _env(tmp_path)

  result = _run(monkeypatch, env)

  assert result.returncode == 0, result.stderr + result.stdout
  call = qualified._training_call(env)
  assert call[1] == qualified.TASK
  assert qualified._value(call, "--agent.seed") == "2"
  assert qualified._value(call, "--agent.num-steps-per-env") == "24"
  assert qualified._value(call, "--agent.max-iterations") == "500"
  assert qualified._value(call, "--agent.save-interval") == "50"
  assert qualified._value(call, "--env.scene.num-envs") == "4096"
  assert qualified._value(call, "--env.metrics.cat-soft.params.imp-limit") == qualified.CAPS
  assert qualified._value(call, "--env.metrics.cat-soft.params.imp-max-p") == "0.25"
  assert ROLE in qualified._value(call, "--agent.run-name")
  assert f"Z1_VIC_IMPULSE_P025_TRAIN_PASS role={ROLE} imp_max_p=0.25" in result.stdout


def test_p025_launcher_is_one_nonarray_a100_job_with_no_requeue():
  source = LAUNCHER.read_text()

  assert "#SBATCH --job-name=z1-vic-imp-p025" in source
  assert "#SBATCH --gres=gpu:1" in source
  assert "#SBATCH --cpus-per-task=8" in source
  assert "#SBATCH --mem=40G" in source
  assert "#SBATCH --time=01:30:00" in source
  assert "#SBATCH --no-requeue" in source
  assert "#SBATCH --array" not in source
  assert "%A" not in source and "%a" not in source
  assert 'readonly IMP_MAX_P="0.25"' in source
  assert 'readonly ROLE="dose_p025_exploratory"' in source
  assert "EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)" in source


@pytest.mark.parametrize(
  "variable",
  (
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_TASK_MIN",
    "SLURM_ARRAY_TASK_MAX",
    "SLURM_ARRAY_TASK_STEP",
  ),
)
def test_p025_launcher_rejects_array_context(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
):
  env = _env(tmp_path)
  env[variable] = "1"

  result = _run(monkeypatch, env)

  assert result.returncode == 2
  assert f"{variable} must be empty" in result.stdout


def test_p025_launcher_rejects_arguments_dirty_code_and_collision(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  env = _env(tmp_path / "args")
  assert _run(monkeypatch, env, "unexpected").returncode == 2

  env = _env(tmp_path / "dirty")
  (Path(env["RUN_ROOT"]) / "dirty").write_text("no\n")
  assert _run(monkeypatch, env).returncode == 2

  env = _env(tmp_path / "collision")
  assert _run(monkeypatch, env).returncode == 0
  assert _run(monkeypatch, env).returncode == 2


@pytest.mark.parametrize("mode", ("missing", "wrong_iter", "nonfinite", "extra"))
def test_p025_launcher_rejects_invalid_checkpoint_set_or_content(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
):
  env = _env(tmp_path)
  env["BAD_CHECKPOINT_MODE"] = mode

  result = _run(monkeypatch, env)

  assert result.returncode == 2
  assert "checkpoint" in result.stdout
  assert "Z1_VIC_IMPULSE_P025_TRAIN_PASS" not in result.stdout


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("RUNTIME_GPU", "NVIDIA H100 80GB HBM3"),
    ("RUNTIME_MJLAB_VERSION", "1.4.1"),
    ("RUNTIME_MUJOCO_VERSION", "3.8.2"),
    ("RUNTIME_MUJOCO_WARP_VERSION", "3.8.2"),
  ),
)
def test_p025_launcher_rejects_runtime_identity_drift(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  variable: str,
  value: str,
):
  env = _env(tmp_path)
  env[variable] = value

  result = _run(monkeypatch, env)

  assert result.returncode == 2
  assert "runtime package versions do not match" in result.stdout


@pytest.mark.parametrize(
  "mode", ("code_dirty", "asset_dirty", "code_revision", "asset_revision")
)
def test_p025_launcher_rejects_postflight_provenance_drift(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
):
  env = _env(tmp_path)
  env["POSTFLIGHT_MUTATION"] = mode

  result = _run(monkeypatch, env)

  assert result.returncode == 2
  assert "provenance drifted" in result.stdout
  assert "Z1_VIC_IMPULSE_P025_TRAIN_PASS" not in result.stdout


@pytest.mark.parametrize("mode", ("fail", "empty", "invalid"))
def test_p025_launcher_rejects_failed_or_malformed_checkpoint_hash(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
):
  env = _env(tmp_path)
  env["BAD_SHA256_MODE"] = mode

  result = _run(monkeypatch, env)

  assert result.returncode == 2
  assert "checkpoint SHA-256 validation failed" in result.stdout
  assert "Z1_VIC_IMPULSE_P025_TRAIN_PASS" not in result.stdout
