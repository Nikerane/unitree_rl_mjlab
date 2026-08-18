"""Shell contract for the one-target provisional-cap impulse-CaT p=0.2 bridge."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_p02_bridge.sbatch"
TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT"
)
CAPS = "[0.82,1.64,0.82,0.82,0.82,0.82]"
EXPECTED_MODELS = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499)


def _git(path: Path, *args: str) -> str:
  result = subprocess.run(
    [
      "git",
      "-c",
      "user.email=p02-bridge-test@example.com",
      "-c",
      "user.name=P02 Bridge Test",
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


def _env(tmp_path: Path) -> dict[str, str]:
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
    "SLURM_JOB_ID": "54321",
  }
  python = home / "repos/unitree_rl_mjlab/.venv/bin/python"
  python.parent.mkdir(parents=True)
  python.write_text(
    "#!/bin/sh\n"
    "{ printf 'CALL'; for arg in \"$@\"; do printf '\\t%s' \"$arg\"; done; printf '\\n'; } >> \"$HOME/calls\"\n"
    "case \"${1:-}\" in\n"
    "  *scripts/train.py)\n"
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
    "  stored_iteration = 498 if iteration == 499 and mode == 'wrong_iter' else iteration\n"
    "  value = torch.tensor([float('nan') if iteration == 499 and mode == 'nonfinite' else 1.0])\n"
    "  torch.save({'iter': stored_iteration, 'state': value}, root / f'model_{iteration}.pt')\n"
    "if mode == 'extra':\n"
    "  torch.save({'iter': 10, 'state': torch.tensor([1.0])}, root / 'model_10.pt')\n"
    "PY\n"
    "    ;;\n"
    "  -)\n"
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


def test_bridge_launcher_runs_exactly_one_predeclared_target(tmp_path: Path):
  env = _env(tmp_path)

  result = _run(env)

  assert result.returncode == 0, result.stderr + result.stdout
  call = _training_call(env)
  assert call[1] == TASK
  assert _value(call, "--agent.seed") == "2"
  assert _value(call, "--agent.num-steps-per-env") == "24"
  assert _value(call, "--agent.max-iterations") == "500"
  assert _value(call, "--agent.save-interval") == "50"
  assert _value(call, "--env.scene.num-envs") == "4096"
  assert _value(call, "--env.metrics.cat-soft.params.imp-limit") == CAPS
  assert _value(call, "--env.metrics.cat-soft.params.imp-max-p") == "0.2"
  assert "Z1_VIC_IMPULSE_P02_BRIDGE_PASS" in result.stdout
  assert "imp_max_p=0.2" in result.stdout
  assert f"caps={CAPS}" in result.stdout


def test_bridge_launcher_is_one_non_array_fixed_budget_job():
  source = LAUNCHER.read_text()

  assert "#SBATCH --job-name=z1-vic-imp-p02-bridge" in source
  assert "#SBATCH --gres=gpu:1" in source
  assert "#SBATCH --cpus-per-task=8" in source
  assert "#SBATCH --mem=40G" in source
  assert "#SBATCH --time=01:30:00" in source
  assert "#SBATCH --no-requeue" in source
  assert "#SBATCH --array" not in source
  assert "%a" not in source and "%A" not in source
  assert "scontrol requeue" not in source
  assert "EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)" in source


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("PYTHONOPTIMIZE", "1"),
    ("SLURM_ARRAY_JOB_ID", "54321"),
    ("SLURM_ARRAY_TASK_ID", "0"),
    ("SLURM_ARRAY_TASK_COUNT", "1"),
    ("SLURM_ARRAY_TASK_MIN", "0"),
    ("SLURM_ARRAY_TASK_MAX", "0"),
    ("SLURM_ARRAY_TASK_STEP", "1"),
    ("WORLD_SIZE", "1"),
    ("RANK", "0"),
    ("LOCAL_RANK", "0"),
    ("MASTER_ADDR", "localhost"),
    ("MASTER_PORT", "29500"),
    ("RESUME_PATH", "/tmp/model.pt"),
  ),
)
def test_bridge_launcher_rejects_array_distributed_resume_or_optimization_env(
  tmp_path: Path, variable: str, value: str
):
  env = _env(tmp_path)
  env[variable] = value

  result = _run(env)

  assert result.returncode == 2
  assert f"{variable} must be empty" in result.stdout


def test_bridge_launcher_rejects_arguments_dirty_code_and_collision(tmp_path: Path):
  env = _env(tmp_path / "args")
  assert _run(env, "unexpected").returncode == 2

  env = _env(tmp_path / "dirty")
  (Path(env["RUN_ROOT"]) / "dirty").write_text("no\n")
  assert _run(env).returncode == 2

  env = _env(tmp_path / "collision")
  assert _run(env).returncode == 0
  assert _run(env).returncode == 2


@pytest.mark.parametrize("mode", ("wrong_iter", "nonfinite", "extra"))
def test_bridge_launcher_rejects_invalid_checkpoint_set_or_content(
  tmp_path: Path, mode: str
):
  env = _env(tmp_path)
  env["BAD_CHECKPOINT_MODE"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "checkpoint" in result.stdout


@pytest.mark.parametrize("mode", ("fail", "empty", "invalid"))
def test_bridge_launcher_rejects_failed_or_malformed_checkpoint_sha256(
  tmp_path: Path, mode: str
):
  env = _env(tmp_path)
  env["BAD_SHA256_MODE"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "checkpoint SHA-256 validation failed" in result.stdout
  assert "Z1_VIC_IMPULSE_P02_BRIDGE_PASS" not in result.stdout
