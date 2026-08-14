"""Shell contract for the matched 90%-boundary impulse-CaT Vega canary."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from scripts.impulse_cat_activation_survey import DIAGNOSTIC_LIMITS_N_M_S


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_diag90_canary.sbatch"
CAPS = "[" + ",".join(f"{value:g}" for value in DIAGNOSTIC_LIMITS_N_M_S) + "]"


def _git(path: Path, *args: str) -> str:
  result = subprocess.run(
    [
      "git",
      "-c",
      "user.email=diag90-test@example.com",
      "-c",
      "user.name=Diag90 Test",
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
  env = {
    "PATH": os.environ["PATH"],
    "HOME": str(home),
    "RUN_ROOT": str(code),
    "ASSET_REPO": str(assets),
    "EXPECTED_CODE_REVISION": _repo(code),
    "EXPECTED_ASSET_REVISION": _repo(assets),
    "SLURM_ARRAY_JOB_ID": "12345",
    "SLURM_ARRAY_TASK_ID": str(task_id),
    "SLURM_ARRAY_TASK_COUNT": "2",
    "SLURM_ARRAY_TASK_MIN": "0",
    "SLURM_ARRAY_TASK_MAX": "1",
    "SLURM_ARRAY_TASK_STEP": "1",
    "SLURM_JOB_ID": f"12345_{task_id}",
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
    "    printf zero > \"$dir/model_0.pt\"\n"
    "    printf final > \"$dir/model_49.pt\"\n"
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
  calls = [line.split("\t")[1:] for line in (Path(env["HOME"]) / "calls").read_text().splitlines()]
  return next(call for call in calls if call and call[0].endswith("scripts/train.py"))


def _value(call: list[str], flag: str) -> str:
  return call[call.index(flag) + 1]


def test_array_arms_are_exactly_matched_except_impulse_dose_and_run_name(tmp_path: Path):
  calls = []
  for task_id in (0, 1):
    env = _env(tmp_path / str(task_id), task_id)
    result = _run(env)
    assert result.returncode == 0, result.stderr + result.stdout
    calls.append(_training_call(env))

  assert _value(calls[0], "--env.metrics.cat-soft.params.imp-max-p") == "0.0"
  assert _value(calls[1], "--env.metrics.cat-soft.params.imp-max-p") == "0.5"
  for call in calls:
    assert _value(call, "--env.metrics.cat-soft.params.imp-limit") == CAPS
    assert _value(call, "--agent.seed") == "2"
    assert _value(call, "--agent.num-steps-per-env") == "24"
    assert _value(call, "--agent.max-iterations") == "50"
    assert _value(call, "--env.scene.num-envs") == "4096"

  normalized = []
  for call in calls:
    copy = list(call)
    copy[0] = "<repo>/scripts/train.py"
    copy[copy.index("--agent.run-name") + 1] = "<run>"
    copy[copy.index("--env.metrics.cat-soft.params.imp-max-p") + 1] = "<dose>"
    normalized.append(copy)
  assert normalized[0] == normalized[1]


def test_launcher_fails_closed_on_wrong_array_shape_arguments_or_dirty_code(tmp_path: Path):
  env = _env(tmp_path / "shape", 0)
  env["SLURM_ARRAY_TASK_COUNT"] = "3"
  assert _run(env).returncode == 2

  env = _env(tmp_path / "args", 0)
  assert _run(env, "unexpected").returncode == 2

  env = _env(tmp_path / "dirty", 0)
  (Path(env["RUN_ROOT"]) / "dirty").write_text("no\n")
  assert _run(env).returncode == 2


def test_launcher_disables_requeue_and_freezes_short_budget():
  source = LAUNCHER.read_text()
  assert "#SBATCH --array=0-1" in source
  assert "#SBATCH --no-requeue" in source
  assert "--agent.max-iterations 50" in source
  assert "--agent.save-interval 50" in source
  assert "scontrol requeue" not in source
