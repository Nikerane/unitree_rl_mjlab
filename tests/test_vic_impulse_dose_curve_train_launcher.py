"""Shell contract for the two-target provisional-cap impulse-CaT dose curve."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_dose_curve_train.sbatch"
TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT"
)
CAPS = "[0.82,1.64,0.82,0.82,0.82,0.82]"
EXPECTED_MODELS = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499)
ARMS = ((0, "dose_p01_target", "0.1"), (1, "dose_p03_target", "0.3"))


def _git(path: Path, *args: str) -> str:
  result = subprocess.run(
    [
      "git",
      "-c",
      "user.email=dose-curve-test@example.com",
      "-c",
      "user.name=Dose Curve Test",
      *args,
    ],
    cwd=path,
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _repo(path: Path) -> str:
  path.mkdir(parents=True)
  (path / "marker").write_text("fixture\n")
  _git(path, "init", "-q")
  _git(path, "add", "marker")
  _git(path, "commit", "-qm", "fixture")
  return _git(path, "rev-parse", "HEAD")


def _env(tmp_path: Path, task_id: int) -> dict[str, str]:
  home = tmp_path / "home"
  code = tmp_path / "repos/code"
  assets = tmp_path / "repos/safe_impact_manipulation"
  fake_bin = home / "bin"
  fake_bin.mkdir(parents=True)
  real_sha256sum = shutil.which("sha256sum")
  assert real_sha256sum is not None
  sha256sum = fake_bin / "sha256sum"
  sha256sum.write_text(
    "#!/bin/sh\n"
    "case \"${BAD_SHA256_MODE:-}\" in\n"
    "  fail) exit 9 ;;\n"
    "  empty) exit 0 ;;\n"
    "  invalid) printf 'not-a-digest  %s\\n' \"$1\"; exit 0 ;;\n"
    f"  *) exec \"{real_sha256sum}\" \"$@\" ;;\n"
    "esac\n"
  )
  sha256sum.chmod(0o755)
  env = {
    "PATH": f"{fake_bin}:{os.environ['PATH']}",
    "HOME": str(home),
    "RUN_ROOT": str(code),
    "ASSET_REPO": str(assets),
    "EXPECTED_CODE_REVISION": _repo(code),
    "EXPECTED_ASSET_REVISION": _repo(assets),
    "SLURM_ARRAY_JOB_ID": "54321",
    "SLURM_ARRAY_TASK_ID": str(task_id),
    "SLURM_ARRAY_TASK_COUNT": "2",
    "SLURM_ARRAY_TASK_MIN": "0",
    "SLURM_ARRAY_TASK_MAX": "1",
    "SLURM_ARRAY_TASK_STEP": "1",
    "SLURM_JOB_ID": f"54321_{task_id}",
  }
  python = home / "repos/unitree_rl_mjlab/.venv/bin/python"
  python.parent.mkdir(parents=True)
  python.write_text(
    "#!/bin/sh\n"
    "{ printf 'CALL'; for arg in \"$@\"; do printf '\\t%s' \"$arg\"; done; printf '\\n'; } >> \"$HOME/calls\"\n"
    "case \"${1:-}\" in\n"
    "  *scripts/train.py)\n"
    "    [ -z \"${BAD_RUNTIME_MODE:-}\" ] || exit 8\n"
    "    dir=\"$PWD/logs/rsl_rl/z1_hammer/fixture_${RUN_NAME}\"\n"
    "    mkdir -p \"$dir\"\n"
    f"    \"{sys.executable}\" - \"$dir\" <<'PY'\n"
    "import os\n"
    "from pathlib import Path\n"
    "import sys\n"
    "import torch\n"
    "root = Path(sys.argv[1])\n"
    f"iterations = {EXPECTED_MODELS!r}\n"
    "mode = os.environ.get('BAD_CHECKPOINT_MODE', '')\n"
    "for iteration in iterations:\n"
    "  if iteration == 450 and mode == 'missing':\n"
    "    continue\n"
    "  stored_iteration = 498 if iteration == 499 and mode == 'wrong_iter' else iteration\n"
    "  value = torch.tensor([float('nan') if iteration == 499 and mode == 'nonfinite' else 1.0])\n"
    "  torch.save({'iter': stored_iteration, 'nested': {'state': value}}, root / f'model_{iteration}.pt')\n"
    "if mode == 'extra':\n"
    "  torch.save({'iter': 10, 'nested': {'state': torch.tensor([1.0])}}, root / 'model_10.pt')\n"
    "if os.environ.get('POSTFLIGHT_DIRTY'):\n"
    "  Path(os.environ['RUN_ROOT'], 'postflight-dirty').write_text('no\\n')\n"
    "PY\n"
    "    ;;\n"
    "  -)\n"
    "    [ -z \"${BAD_RUNTIME_MODE:-}\" ] || exit 8\n"
    f"    if [ \"$#\" -gt 1 ]; then exec \"{sys.executable}\" \"$@\"; fi\n"
    "    ;;\n"
    "esac\n"
  )
  python.chmod(0o755)
  return env


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    ["bash", str(LAUNCHER), *args],
    env=env,
    capture_output=True,
    text=True,
    timeout=30,
  )


def _training_call(env: dict[str, str]) -> list[str]:
  calls = [
    line.split("\t")[1:]
    for line in (Path(env["HOME"]) / "calls").read_text().splitlines()
  ]
  return next(call for call in calls if call and call[0].endswith("scripts/train.py"))


def _value(call: list[str], flag: str) -> str:
  return call[call.index(flag) + 1]


def test_dose_curve_arms_run_only_the_predeclared_roles_and_doses(tmp_path: Path):
  """Breaks if an array role selects the wrong impulse-CaT training dose."""
  calls = []
  for task_id, role, dose in ARMS:
    env = _env(tmp_path / role, task_id)
    result = _run(env)
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"role={role} imp_max_p={dose}" in result.stdout
    assert f"Z1_VIC_IMPULSE_DOSE_CURVE_TRAIN_PASS role={role}" in result.stdout
    checkpoint_lines = [
      line
      for line in result.stdout.splitlines()
      if line.startswith("Z1_VIC_IMPULSE_DOSE_CURVE_TRAIN_CHECKPOINT ")
    ]
    assert len(checkpoint_lines) == len(EXPECTED_MODELS)
    assert all(re.search(r"sha256=[0-9a-f]{64}$", line) for line in checkpoint_lines)
    calls.append(_training_call(env))

  for call, (_, role, dose) in zip(calls, ARMS, strict=True):
    assert call[1] == TASK
    assert _value(call, "--agent.seed") == "2"
    assert _value(call, "--agent.num-steps-per-env") == "24"
    assert _value(call, "--agent.max-iterations") == "500"
    assert _value(call, "--agent.save-interval") == "50"
    assert _value(call, "--env.scene.num-envs") == "4096"
    assert _value(call, "--env.metrics.cat-soft.params.imp-limit") == CAPS
    assert _value(call, "--env.metrics.cat-soft.params.imp-max-p") == dose
    assert role in _value(call, "--agent.run-name")

  normalized = []
  for call in calls:
    copy = list(call)
    copy[0] = "<repo>/scripts/train.py"
    copy[copy.index("--agent.run-name") + 1] = "<role-and-job-specific-run-name>"
    copy[copy.index("--env.metrics.cat-soft.params.imp-max-p") + 1] = "<dose>"
    normalized.append(copy)
  assert normalized[0] == normalized[1]


def test_dose_curve_launcher_freezes_the_two_arm_one_shot_resource_contract():
  """Breaks if Slurm can allocate a different job shape or requeue an arm."""
  source = LAUNCHER.read_text()

  assert "#SBATCH --array=0-1" in source
  assert "#SBATCH --gres=gpu:1" in source
  assert "#SBATCH --cpus-per-task=8" in source
  assert "#SBATCH --mem=40G" in source
  assert "#SBATCH --time=01:30:00" in source
  assert "#SBATCH --no-requeue" in source
  assert "%A_%a" in source
  assert "scontrol requeue" not in source
  assert "EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)" in source
  assert 'gpu != "NVIDIA A100-SXM4-40GB"' in source
  assert '"mjlab": "1.4.0"' in source
  assert '"mujoco": "3.8.1"' in source
  assert '"mujoco-warp": "3.8.1"' in source


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("PYTHONOPTIMIZE", "1"),
    ("SLURM_ARRAY_TASK_COUNT", "3"),
    ("SLURM_ARRAY_TASK_MIN", "1"),
    ("SLURM_ARRAY_TASK_MAX", "2"),
    ("SLURM_ARRAY_TASK_STEP", "2"),
    ("SLURM_ARRAY_JOB_ID", "not-numeric"),
    ("SLURM_ARRAY_TASK_ID", "2"),
    ("WORLD_SIZE", "1"),
    ("RANK", "0"),
    ("LOCAL_RANK", "0"),
    ("MASTER_ADDR", "localhost"),
    ("MASTER_PORT", "29500"),
    ("RESUME_PATH", "/tmp/model.pt"),
  ),
)
def test_dose_curve_launcher_rejects_wrong_array_or_unsupported_execution_context(
  tmp_path: Path, variable: str, value: str
):
  """Breaks if a malformed array, resume, or distributed job can train a dose arm."""
  env = _env(tmp_path / variable, 0)
  env[variable] = value

  result = _run(env)

  assert result.returncode == 2


def test_dose_curve_launcher_rejects_arguments_provenance_and_leaf_collisions(tmp_path: Path):
  """Breaks if a run can bypass fresh, canonical, clean provenance leaves."""
  env = _env(tmp_path / "args", 0)
  assert _run(env, "unexpected").returncode == 2

  env = _env(tmp_path / "dirty-code", 0)
  (Path(env["RUN_ROOT"]) / "dirty").write_text("no\n")
  assert _run(env).returncode == 2

  env = _env(tmp_path / "dirty-assets", 0)
  (Path(env["ASSET_REPO"]) / "dirty").write_text("no\n")
  assert _run(env).returncode == 2

  env = _env(tmp_path / "noncanonical", 0)
  other_assets = tmp_path / "other-assets"
  _repo(other_assets)
  env["ASSET_REPO"] = str(other_assets)
  assert _run(env).returncode == 2

  env = _env(tmp_path / "collision", 0)
  assert _run(env).returncode == 0
  assert _run(env).returncode == 2


@pytest.mark.parametrize("mode", ("missing", "wrong_iter", "nonfinite", "extra"))
def test_dose_curve_launcher_rejects_incomplete_extra_or_nonfinite_checkpoints(
  tmp_path: Path, mode: str
):
  """Breaks if Task 3 could consume a non-final, malformed checkpoint leaf."""
  env = _env(tmp_path / mode, 1)
  env["BAD_CHECKPOINT_MODE"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "checkpoint" in result.stdout
  assert "Z1_VIC_IMPULSE_DOSE_CURVE_TRAIN_PASS" not in result.stdout


@pytest.mark.parametrize("mode", ("fail", "empty", "invalid"))
def test_dose_curve_launcher_rejects_failed_or_invalid_lowercase_checkpoint_hashes(
  tmp_path: Path, mode: str
):
  """Breaks if malformed hashes could be emitted as final-checkpoint provenance."""
  env = _env(tmp_path / mode, 0)
  env["BAD_SHA256_MODE"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "checkpoint SHA-256 validation failed" in result.stdout
  assert "Z1_VIC_IMPULSE_DOSE_CURVE_TRAIN_PASS" not in result.stdout


@pytest.mark.parametrize("mode", ("gpu", "runtime"))
def test_dose_curve_launcher_rejects_wrong_gpu_or_runtime(tmp_path: Path, mode: str):
  """Breaks if an arm can train without the qualified A100/package runtime."""
  env = _env(tmp_path / mode, 0)
  env["BAD_RUNTIME_MODE"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "CUDA identity or runtime package versions do not match" in result.stdout


def test_dose_curve_launcher_rejects_postflight_provenance_drift(tmp_path: Path):
  """Breaks if code provenance can change after a training arm begins."""
  env = _env(tmp_path, 0)
  env["POSTFLIGHT_DIRTY"] = "1"

  result = _run(env)

  assert result.returncode == 2
  assert "postflight code provenance drifted" in result.stdout
