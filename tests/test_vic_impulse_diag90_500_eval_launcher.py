"""Shell contract for frozen matched diagnostic impulse-CaT evaluation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.impulse_cat_activation_survey import EVALUATION_CHECKPOINTS


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_diag90_500_eval.sbatch"
EXPECTED_OUTPUTS = (
  "fixed_trace.npz",
  "training_like_seed_2_trace.npz",
  "training_like_seed_2026081701_trace.npz",
  "training_like_seed_2026081702_trace.npz",
  "summary.json",
)


def _git(path: Path, *args: str) -> str:
  result = subprocess.run(
    [
      "git",
      "-c",
      "user.email=diag90-eval-test@example.com",
      "-c",
      "user.name=Diag90 Eval Test",
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


def _checkpoint(path: Path, content: bytes) -> tuple[Path, str]:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(content)
  return path, hashlib.sha256(content).hexdigest()


def _env(tmp_path: Path, task_id: int) -> dict[str, str]:
  home = tmp_path / "home"
  code = tmp_path / "repos/code"
  assets = tmp_path / "repos/safe_impact_manipulation"
  control, control_sha = _checkpoint(
    tmp_path / "checkpoints/control/model_499.pt", b"control frozen checkpoint\n"
  )
  target, target_sha = _checkpoint(
    tmp_path / "checkpoints/target/model_499.pt", b"target frozen checkpoint\n"
  )
  assert control_sha == hashlib.sha256(b"control frozen checkpoint\n").hexdigest()
  assert target_sha == hashlib.sha256(b"target frozen checkpoint\n").hexdigest()
  tool_bin = tmp_path / "bin"
  tool_bin.mkdir()
  real_git = shutil.which("git")
  real_sha256sum = shutil.which("sha256sum")
  assert real_git is not None
  assert real_sha256sum is not None
  (tool_bin / "git").write_text(
    "#!/bin/sh\n"
    "case \" $* \" in\n"
    "  *' merge-base '*) exit 0 ;;\n"
    "  *' cat-file -e 030942f34ac4a79131c1b70206d3c4acd58da79a^{commit}'*) exit 0 ;;\n"
    "  *'safe_impact_manipulation rev-parse HEAD'*) printf '%s\\n' b58ccd2f81fd246f27c1e8d88cf86484cd888703 ;;\n"
    "  *) exec \"$REAL_GIT\" \"$@\" ;;\n"
    "esac\n"
  )
  (tool_bin / "sha256sum").write_text(
    "#!/bin/sh\n"
    "case \"${1:-}\" in\n"
    "  \"$CONTROL_CHECKPOINT\")\n"
    "    actual=$(\"$REAL_SHA256SUM\" \"$1\" | cut -d ' ' -f 1)\n"
    "    [ \"$actual\" = \"$INITIAL_CONTROL_FILE_SHA\" ] && printf '%s  %s\\n' \"$CONTROL_SHA\" \"$1\" || exec \"$REAL_SHA256SUM\" \"$@\"\n"
    "    ;;\n"
    "  \"$TARGET_CHECKPOINT\")\n"
    "    actual=$(\"$REAL_SHA256SUM\" \"$1\" | cut -d ' ' -f 1)\n"
    "    [ \"$actual\" = \"$INITIAL_TARGET_FILE_SHA\" ] && printf '%s  %s\\n' \"$TARGET_SHA\" \"$1\" || exec \"$REAL_SHA256SUM\" \"$@\"\n"
    "    ;;\n"
    "  *) exec \"$REAL_SHA256SUM\" \"$@\" ;;\n"
    "esac\n"
  )
  (tool_bin / "git").chmod(0o755)
  (tool_bin / "sha256sum").chmod(0o755)
  env = {
    "PATH": str(tool_bin) + os.pathsep + os.environ["PATH"],
    "HOME": str(home),
    "RUN_ROOT": str(code),
    "ASSET_REPO": str(assets),
    "EXPECTED_CODE_REVISION": _repo(code),
    "EXPECTED_ASSET_REVISION": "b58ccd2f81fd246f27c1e8d88cf86484cd888703",
    "CONTROL_CHECKPOINT": str(control),
    "TARGET_CHECKPOINT": str(target),
    "CONTROL_SHA": EVALUATION_CHECKPOINTS["diag90_control"],
    "TARGET_SHA": EVALUATION_CHECKPOINTS["diag90_target"],
    "INITIAL_CONTROL_FILE_SHA": control_sha,
    "INITIAL_TARGET_FILE_SHA": target_sha,
    "REAL_GIT": real_git,
    "REAL_SHA256SUM": real_sha256sum,
    "SLURM_ARRAY_JOB_ID": "67890",
    "SLURM_ARRAY_TASK_ID": str(task_id),
    "SLURM_ARRAY_TASK_COUNT": "2",
    "SLURM_ARRAY_TASK_MIN": "0",
    "SLURM_ARRAY_TASK_MAX": "1",
    "SLURM_ARRAY_TASK_STEP": "1",
  }
  _repo(assets)
  python = home / "repos/unitree_rl_mjlab/.venv/bin/python"
  python.parent.mkdir(parents=True)
  python.write_text(
    "#!/bin/sh\n"
    "{ printf 'CALL'; for arg in \"$@\"; do printf '\\t%s' \"$arg\"; done; printf '\\n'; } >> \"$HOME/calls\"\n"
    "if [ \"${1:-}\" = \"-\" ]; then exit 0; fi\n"
    "case \"${1:-}\" in\n"
    "  *scripts/impulse_cat_activation_survey.py)\n"
    "    output=\"\"\n"
    "    previous=\"\"\n"
    "    for arg in \"$@\"; do\n"
    "      if [ \"$previous\" = \"--output-dir\" ]; then output=\"$arg\"; fi\n"
    "      previous=\"$arg\"\n"
    "    done\n"
    "    [ -n \"$output\" ] && [ ! -e \"$output\" ] || exit 71\n"
    "    mkdir \"$output\"\n"
    "    for name in fixed_trace.npz training_like_seed_2_trace.npz training_like_seed_2026081701_trace.npz training_like_seed_2026081702_trace.npz summary.json; do\n"
    "      printf 'fake %s\\n' \"$name\" > \"$output/$name\"\n"
    "    done\n"
    "    if [ -n \"${EXTRA_HIDDEN_OUTPUT:-}\" ]; then printf extra > \"$output/.extra\"; fi\n"
    "    if [ -n \"${MUTATE_CHECKPOINT:-}\" ]; then printf mutation > \"$MUTATE_CHECKPOINT\"; fi\n"
    "    ;;\n"
    "  *) exit 72 ;;\n"
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


def _evaluator_call(env: dict[str, str]) -> list[str]:
  calls = [line.split("\t")[1:] for line in (Path(env["HOME"]) / "calls").read_text().splitlines()]
  return next(
    call
    for call in calls
    if call and call[0].endswith("scripts/impulse_cat_activation_survey.py")
  )


def _value(call: list[str], flag: str) -> str:
  return call[call.index(flag) + 1]


def test_launcher_maps_each_array_arm_to_exact_frozen_role_checkpoint_and_outputs(
  tmp_path: Path,
):
  calls: list[list[str]] = []
  for task_id, role, checkpoint_key in (
    (0, "diag90_control", "CONTROL_CHECKPOINT"),
    (1, "diag90_target", "TARGET_CHECKPOINT"),
  ):
    env = _env(tmp_path / str(task_id), task_id)
    result = _run(env)
    assert result.returncode == 0, result.stderr + result.stdout
    call = _evaluator_call(env)
    calls.append(call)
    assert _value(call, "--checkpoint-role") == role
    assert _value(call, "--checkpoint") == env[checkpoint_key]
    assert Path(_value(call, "--checkpoint")).name == "model_499.pt"
    assert _value(call, "--device") == "cuda:0"
    output = Path(_value(call, "--output-dir"))
    assert output.name == "evaluation"
    assert output.parent.name.endswith(f"_{task_id}_{role}")
    assert sorted(path.name for path in output.iterdir()) == sorted(EXPECTED_OUTPUTS)
    manifest = output.parent / "SHA256SUMS"
    assert manifest.is_file()
    expected_manifest = "".join(
      f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  evaluation/{name}\n"
      for name in EXPECTED_OUTPUTS
    )
    assert manifest.read_text() == expected_manifest
    assert EVALUATION_CHECKPOINTS[role] == env[
      "CONTROL_SHA" if role == "diag90_control" else "TARGET_SHA"
    ]

  normalized = []
  for call in calls:
    copy = list(call)
    copy[0] = "<repo>/scripts/impulse_cat_activation_survey.py"
    copy[copy.index("--checkpoint-role") + 1] = "<role>"
    copy[copy.index("--checkpoint") + 1] = "<checkpoint>"
    copy[copy.index("--output-dir") + 1] = "<output>"
    normalized.append(copy)
  assert normalized[0] == normalized[1]


def test_launcher_refuses_collisions_dirty_repositories_and_invalid_checkpoints(
  tmp_path: Path,
):
  env = _env(tmp_path / "collision", 0)
  result_root = (
    Path(env["HOME"])
    / "campaigns/z1-vic-impulse-diag90-500/evaluation"
    / env["EXPECTED_CODE_REVISION"]
    / env["EXPECTED_ASSET_REVISION"]
  )
  attempt = result_root / "67890_0_diag90_control"
  attempt.mkdir(parents=True)
  result = _run(env)
  assert result.returncode == 2
  assert "attempt directory already exists" in result.stdout

  env = _env(tmp_path / "dirty", 0)
  (Path(env["RUN_ROOT"]) / "dirty").write_text("no\n")
  assert _run(env).returncode == 2

  env = _env(tmp_path / "symlink", 0)
  checkpoint = Path(env["CONTROL_CHECKPOINT"])
  link = checkpoint.parent / "model_499_link.pt"
  link.symlink_to(checkpoint)
  checkpoint.unlink()
  link.rename(checkpoint)
  result = _run(env)
  assert result.returncode == 2
  assert "regular non-symlink model_499.pt" in result.stdout


def test_launcher_refuses_wrong_frozen_hash_and_postflight_checkpoint_drift(
  tmp_path: Path,
):
  env = _env(tmp_path / "wrong-hash", 0)
  env["CONTROL_SHA"] = "0" * 64
  result = _run(env)
  assert result.returncode == 2
  assert "role/checkpoint SHA-256 mismatch" in result.stdout

  env = _env(tmp_path / "drift", 0)
  env["MUTATE_CHECKPOINT"] = env["CONTROL_CHECKPOINT"]
  result = _run(env)
  assert result.returncode == 2
  assert "checkpoint SHA-256 drifted during evaluation" in result.stdout


def test_launcher_refuses_hidden_extra_evaluator_artifact(tmp_path: Path):
  env = _env(tmp_path, 0)
  env["EXTRA_HIDDEN_OUTPUT"] = "1"

  result = _run(env)

  assert result.returncode == 2
  assert "evaluator output set does not contain exactly five artifacts" in result.stdout


def test_launcher_freezes_resources_runtime_guards_and_zero_retry_contract():
  source = LAUNCHER.read_text()
  assert "#SBATCH --array=0-1" in source
  assert "#SBATCH --time=00:20:00" in source
  assert "#SBATCH --gres=gpu:1" in source
  assert "#SBATCH --no-requeue" in source
  assert "NVIDIA A100-SXM4-40GB" in source
  assert '"mjlab": "1.4.0"' in source
  assert '"mujoco": "3.8.1"' in source
  assert '"mujoco-warp": "3.8.1"' in source
  assert "merge-base --is-ancestor" in source
  assert "scontrol requeue" not in source


@pytest.mark.parametrize("variable", ("CONTROL_CHECKPOINT", "TARGET_CHECKPOINT"))
def test_launcher_requires_both_checkpoint_variables(tmp_path: Path, variable: str):
  env = _env(tmp_path / variable, 0)
  env.pop(variable)
  result = _run(env)
  assert result.returncode == 2
  assert f"{variable} is required" in result.stdout
