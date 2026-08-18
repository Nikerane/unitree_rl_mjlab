"""Shell contract for the four-policy frozen impulse-CaT dose evaluation."""

from __future__ import annotations

import hashlib
from itertools import count
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.impulse_cat_activation_survey import EVALUATION_CHECKPOINTS


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_impulse_dose_curve_eval.sbatch"
ASSET_SHA = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
OUTPUTS = (
  "fixed_trace.npz",
  "training_like_seed_2_trace.npz",
  "training_like_seed_2026081701_trace.npz",
  "training_like_seed_2026081702_trace.npz",
  "summary.json",
)
ARMS = (
  (0, "dose_p0_control", "P0_CHECKPOINT", "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3"),
  (1, "dose_p01_target", "P01_CHECKPOINT", "92f1d97c8ff1476cb26c0b648478a0bc3c522e4e1eb7087389fb8e0d6bf73f86"),
  (2, "dose_p02_target", "P02_CHECKPOINT", "57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de"),
  (3, "dose_p03_target", "P03_CHECKPOINT", "4c0a665fffc077d488630a28b258c1593050f969b4f6227147c3594097dc4efd"),
)
ARRAY_JOB_IDS = count(720000)
ELEMENT_JOB_IDS = count(1720000)


def _git(path: Path, *args: str) -> str:
  result = subprocess.run(
    [
      "git", "-c", "user.email=dose-eval-test@example.com",
      "-c", "user.name=Dose Eval Test", *args,
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
  real_git = shutil.which("git")
  real_sha256sum = shutil.which("sha256sum")
  assert real_git is not None
  assert real_sha256sum is not None
  (fake_bin / "git").write_text(
    "#!/bin/sh\n"
    "case \" $* \" in\n"
    "  *' merge-base '*) exit 0 ;;\n"
    "  *' cat-file -e 7d0761e3dd156012a22378fd0b269e4c50ea5ae6^{commit}'*) exit 0 ;;\n"
    f"  *'safe_impact_manipulation rev-parse HEAD'*) if grep -q drift \"$2/marker\" 2>/dev/null; then exec \"$REAL_GIT\" \"$@\"; else printf '%s\\n' {ASSET_SHA}; fi ;;\n"
    "  *) exec \"$REAL_GIT\" \"$@\" ;;\n"
    "esac\n"
  )
  (fake_bin / "git").chmod(0o755)

  checkpoints: dict[str, Path] = {}
  file_hashes: dict[str, str] = {}
  for _, role, variable, _ in ARMS:
    checkpoint = tmp_path / "checkpoints" / role / "model_499.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"{role} fixture checkpoint\n".encode())
    checkpoints[variable] = checkpoint
    file_hashes[variable] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

  sha_lines = ["#!/bin/sh", "first=${1:-}"]
  sha_lines += [
    "if [ \"${BAD_MANIFEST_SHA_MODE:-}\" = fail ] && case \"$first\" in evaluation/*) true;; *) false;; esac; then exit 9; fi",
    "if [ \"${BAD_MANIFEST_SHA_MODE:-}\" = invalid ] && case \"$first\" in evaluation/*) true;; *) false;; esac; then",
    "  for path in \"$@\"; do printf 'NOT-A-SHA  %s\\n' \"$path\"; done",
    "  exit 0",
    "fi",
  ]
  for _, _, variable, frozen_sha in ARMS:
    sha_lines += [
      f'if [ "$first" = "${{{variable}}}" ]; then',
      '  case "${BAD_CHECKPOINT_SHA_MODE:-}" in fail) exit 9;; invalid) printf "NOT-A-SHA  %s\\n" "$first"; exit 0;; esac',
      '  actual=$("$REAL_SHA256SUM" "$first" | cut -d " " -f 1)',
      f'  [ "$actual" = "${{INITIAL_{variable}_SHA}}" ] && printf "%s  %s\\n" "${{{variable}_FROZEN_SHA}}" "$first" || exec "$REAL_SHA256SUM" "$@"',
      "  exit 0",
      "fi",
    ]
  sha_lines.append('exec "$REAL_SHA256SUM" "$@"')
  (fake_bin / "sha256sum").write_text("\n".join(sha_lines) + "\n")
  (fake_bin / "sha256sum").chmod(0o755)

  runtime_stubs = home / "runtime-stubs"
  runtime_stubs.mkdir()
  (runtime_stubs / "sitecustomize.py").write_text(
    "import os, sys, types\n"
    "from importlib import metadata\n"
    "versions = {'mjlab': os.environ.get('RUNTIME_MJLAB_VERSION', '1.4.0'), "
    "'mujoco': os.environ.get('RUNTIME_MUJOCO_VERSION', '3.8.1'), "
    "'mujoco-warp': os.environ.get('RUNTIME_MUJOCO_WARP_VERSION', '3.8.1')}\n"
    "metadata.version = versions.__getitem__\n"
    "torch = types.ModuleType('torch')\n"
    "torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, "
    "get_device_name=lambda index: os.environ.get('RUNTIME_GPU', 'NVIDIA A100-SXM4-40GB'))\n"
    "sys.modules['torch'] = torch\n"
  )

  generator = home / "generate_evaluation.py"
  generator.write_text(
    "import json, os, sys\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "output, role, checkpoint_sha, code_sha, asset_sha = sys.argv[1:]\n"
    "root = Path(output); root.mkdir()\n"
    "mode = os.environ.get('BAD_OUTPUT_MODE', '')\n"
    "velocity = np.array([[0.2, 0.4]], dtype=np.float64)\n"
    "impulse = np.zeros((1, 2), dtype=np.float64)\n"
    "delta = np.maximum(velocity, impulse)\n"
    "if mode == 'nonfinite': velocity[0, 0] = np.nan\n"
    "if mode == 'nonzero_impulse': impulse[0, 0] = 0.1\n"
    "if mode == 'wrong_delta': delta[0, 0] = 0.9\n"
    "if mode == 'wrong_shape': impulse = np.zeros((1, 1), dtype=np.float64)\n"
    "for name in ('fixed_trace.npz', 'training_like_seed_2_trace.npz', "
    "'training_like_seed_2026081701_trace.npz', 'training_like_seed_2026081702_trace.npz'):\n"
    "  if mode == 'missing' and name == 'fixed_trace.npz': continue\n"
    "  np.savez(root / name, delta_velocity=velocity, delta_impulse=impulse, delta=delta, "
    "lambda_per_joint=np.zeros((1, 2, 6)), substep_rolling_per_joint=np.zeros((10, 2, 6)))\n"
    "summary = {'schema_version': 2, 'task': "
    "'Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT', "
    "'checkpoint': {'role': role, 'sha256': checkpoint_sha}, 'code_revision': code_sha, "
    "'asset_revision': asset_sha, 'protocol': {'live_imp_max_p': 0.0, "
    "'fixed_seed': 2026081202, 'stochastic_seeds': [2, 2026081701, 2026081702]}}\n"
    "def population_protocol(seed, num_envs, steps, mode):\n"
    "  return {'seed': seed, 'num_envs': num_envs, 'control_steps': steps, "
    "'policy_mode': mode, 'auto_reset': mode == 'sampled', "
    "'initial_population_sha256': ('320efae8c21b3303c5dc3f18ae7df00e887653abf26aa8fb413803cf78d094cb' if mode == 'mean' else 'a' * 64), "
    "'rng_streams': {'reset': seed + 10000019, 'observation': seed + 20000033, 'action': seed + 30000041}}\n"
    "summary['populations'] = {'fixed_mean': {'protocol': population_protocol(2026081202, 64, 8, 'mean')}, "
    "'training_like_sampled': {str(seed): {'protocol': population_protocol(seed, 4096, 24, 'sampled')} "
    "for seed in (2, 2026081701, 2026081702)}}\n"
    "if mode == 'wrong_role': summary['checkpoint']['role'] = 'dose_p0_control'\n"
    "if mode == 'wrong_summary_hash': summary['checkpoint']['sha256'] = '0' * 64\n"
    "if mode == 'wrong_code': summary['code_revision'] = '0' * 40\n"
    "if mode == 'wrong_asset': summary['asset_revision'] = '0' * 40\n"
    "if mode == 'wrong_live_p': summary['protocol']['live_imp_max_p'] = 0.3\n"
    "if mode == 'wrong_seeds': summary['protocol']['stochastic_seeds'] = [2]\n"
    "if mode == 'wrong_population': summary['populations']['training_like_sampled']['2']['protocol']['num_envs'] = 2\n"
    "if mode == 'wrong_rng': summary['populations']['training_like_sampled']['2']['protocol']['rng_streams']['action'] += 1\n"
    "(root / 'summary.json').write_text(json.dumps(summary, allow_nan=False))\n"
    "if mode == 'extra': (root / '.extra').write_text('extra')\n"
    "if mode == 'symlink':\n"
    "  path = root / 'fixed_trace.npz'; target = root / 'real_fixed_trace.npz'; path.rename(target); path.symlink_to(target)\n"
  )

  python = home / "repos/unitree_rl_mjlab/.venv/bin/python"
  python.parent.mkdir(parents=True)
  python.write_text(
    "#!/bin/sh\n"
    "{ printf 'CALL'; for arg in \"$@\"; do printf '\\t%s' \"$arg\"; done; printf '\\n'; } >> \"$HOME/calls\"\n"
    "case \"${1:-}\" in\n"
    "  -) PYTHONPATH=\"$HOME/runtime-stubs:${PYTHONPATH:-}\" exec \"$REAL_PYTHON\" \"$@\" ;;\n"
    "  *scripts/impulse_cat_activation_survey.py)\n"
    "    role=''; output=''; previous=''\n"
    "    for arg in \"$@\"; do\n"
    "      [ \"$previous\" = --checkpoint-role ] && role=\"$arg\"\n"
    "      [ \"$previous\" = --output-dir ] && output=\"$arg\"\n"
    "      previous=\"$arg\"\n"
    "    done\n"
    "    case \"$role\" in\n"
    "      dose_p0_control) digest=$P0_CHECKPOINT_FROZEN_SHA ;;\n"
    "      dose_p01_target) digest=$P01_CHECKPOINT_FROZEN_SHA ;;\n"
    "      dose_p02_target) digest=$P02_CHECKPOINT_FROZEN_SHA ;;\n"
    "      dose_p03_target) digest=$P03_CHECKPOINT_FROZEN_SHA ;;\n"
    "      *) exit 73 ;;\n"
    "    esac\n"
    "    \"$REAL_PYTHON\" \"$HOME/generate_evaluation.py\" \"$output\" \"$role\" \"$digest\" \"$EXPECTED_CODE_REVISION\" \"$EXPECTED_ASSET_REVISION\"\n"
    "    case \"${POSTFLIGHT_MUTATION:-}\" in\n"
    "      checkpoint) printf mutation > \"${MUTATE_CHECKPOINT}\" ;;\n"
    "      code_dirty) printf no > \"$RUN_ROOT/postflight-dirty\" ;;\n"
    "      asset_dirty) printf no > \"$ASSET_REPO/postflight-dirty\" ;;\n"
    "      code_revision) printf drift >> \"$RUN_ROOT/marker\"; git -C \"$RUN_ROOT\" add marker; git -C \"$RUN_ROOT\" -c user.email=dose-eval-test@example.com -c user.name='Dose Eval Test' commit -qm drift ;;\n"
    "      asset_revision) printf drift >> \"$ASSET_REPO/marker\"; git -C \"$ASSET_REPO\" add marker; git -C \"$ASSET_REPO\" -c user.email=dose-eval-test@example.com -c user.name='Dose Eval Test' commit -qm drift ;;\n"
    "    esac\n"
    "    ;;\n"
    "  *) exit 74 ;;\n"
    "esac\n"
  )
  python.chmod(0o755)

  env = {
    "PATH": f"{fake_bin}:{os.environ['PATH']}",
    "HOME": str(home),
    "RUN_ROOT": str(code),
    "ASSET_REPO": str(assets),
    "EXPECTED_CODE_REVISION": _repo(code),
    "EXPECTED_ASSET_REVISION": ASSET_SHA,
    "REAL_SHA256SUM": real_sha256sum,
    "REAL_GIT": real_git,
    "REAL_PYTHON": sys.executable,
    "SLURM_ARRAY_JOB_ID": str(next(ARRAY_JOB_IDS)),
    "SLURM_JOB_ID": str(next(ELEMENT_JOB_IDS)),
    "SLURM_ARRAY_TASK_ID": str(task_id),
    "SLURM_ARRAY_TASK_COUNT": "4",
    "SLURM_ARRAY_TASK_MIN": "0",
    "SLURM_ARRAY_TASK_MAX": "3",
    "SLURM_ARRAY_TASK_STEP": "1",
  }
  _repo(assets)
  for _, _, variable, frozen_sha in ARMS:
    env[variable] = str(checkpoints[variable])
    env[f"INITIAL_{variable}_SHA"] = file_hashes[variable]
    env[f"{variable}_FROZEN_SHA"] = frozen_sha
  return env


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    ["bash", str(LAUNCHER), *args],
    env=env,
    capture_output=True,
    text=True,
    timeout=30,
  )


def _survey_call(env: dict[str, str]) -> list[str]:
  calls = [
    line.split("\t")[1:]
    for line in (Path(env["HOME"]) / "calls").read_text().splitlines()
  ]
  return next(call for call in calls if call and call[0].endswith("impulse_cat_activation_survey.py"))


def _value(call: list[str], flag: str) -> str:
  return call[call.index(flag) + 1]


def test_dose_eval_maps_all_four_immutable_roles_hashes_and_matched_inputs(tmp_path: Path):
  calls = []
  for task_id, role, checkpoint_variable, frozen_sha in ARMS:
    assert EVALUATION_CHECKPOINTS[role] == frozen_sha
    env = _env(tmp_path / role, task_id)
    result = _run(env)
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Z1_VIC_IMPULSE_DOSE_CURVE_EVAL_PASS role={role}" in result.stdout
    call = _survey_call(env)
    assert _value(call, "--checkpoint-role") == role
    assert _value(call, "--checkpoint") == env[checkpoint_variable]
    assert _value(call, "--device") == "cuda:0"
    output = Path(_value(call, "--output-dir"))
    assert output.parent.name.endswith(f"_{task_id}_{role}")
    assert sorted(path.name for path in output.iterdir()) == sorted(OUTPUTS)
    manifest = output.parent / "SHA256SUMS"
    assert manifest.read_text() == "".join(
      f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  evaluation/{name}\n"
      for name in OUTPUTS
    )
    calls.append(call)

  normalized = []
  for call in calls:
    copy = list(call)
    copy[0] = "<repo>/scripts/impulse_cat_activation_survey.py"
    copy[copy.index("--checkpoint-role") + 1] = "<role>"
    copy[copy.index("--checkpoint") + 1] = "<checkpoint>"
    copy[copy.index("--output-dir") + 1] = "<output-dir>"
    normalized.append(copy)
  assert normalized[0] == normalized[1] == normalized[2] == normalized[3]


def test_dose_eval_launcher_freezes_four_a100_arms_and_existing_survey_protocol():
  source = LAUNCHER.read_text()
  assert "#SBATCH --array=0-3" in source
  assert "#SBATCH --gres=gpu:1" in source
  assert "#SBATCH --cpus-per-task=8" in source
  assert "#SBATCH --mem=40G" in source
  assert "#SBATCH --time=00:20:00" in source
  assert "#SBATCH --no-requeue" in source
  assert "NVIDIA A100-SXM4-40GB" in source
  assert "scontrol requeue" not in source
  assert "--checkpoint-role \"$ROLE\"" in source
  assert "--device cuda:0" in source


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("PYTHONOPTIMIZE", "1"),
    ("SLURM_ARRAY_TASK_COUNT", "3"),
    ("SLURM_ARRAY_TASK_MIN", "1"),
    ("SLURM_ARRAY_TASK_MAX", "4"),
    ("SLURM_ARRAY_TASK_STEP", "2"),
    ("SLURM_ARRAY_JOB_ID", "not-numeric"),
    ("SLURM_ARRAY_TASK_ID", "4"),
    ("SLURM_JOB_ID", "1720000_0"),
    ("WORLD_SIZE", "1"),
    ("RANK", "0"),
    ("LOCAL_RANK", "0"),
    ("MASTER_ADDR", "localhost"),
    ("MASTER_PORT", "29500"),
    ("RESUME_PATH", "/tmp/model.pt"),
  ),
)
def test_dose_eval_rejects_malformed_array_or_execution_context(
  tmp_path: Path, variable: str, value: str
):
  env = _env(tmp_path / variable, 0)
  env[variable] = value
  assert _run(env).returncode == 2


def test_dose_eval_accepts_a_real_numeric_element_job_id(tmp_path: Path):
  env = _env(tmp_path, 0)
  assert env["SLURM_JOB_ID"] != env["SLURM_ARRAY_JOB_ID"]
  result = _run(env)
  assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize("_,role,checkpoint_variable,frozen_sha", ARMS)
def test_dose_eval_rejects_each_role_checkpoint_hash_mismatch(
  tmp_path: Path, _: int, role: str, checkpoint_variable: str, frozen_sha: str
):
  del frozen_sha
  task_id = next(row[0] for row in ARMS if row[1] == role)
  env = _env(tmp_path / role, task_id)
  env[f"{checkpoint_variable}_FROZEN_SHA"] = "0" * 64
  result = _run(env)
  assert result.returncode == 2
  assert "role/checkpoint SHA-256 mismatch" in result.stdout


@pytest.mark.parametrize("variable", tuple(row[2] for row in ARMS))
def test_dose_eval_requires_all_four_checkpoint_variables(tmp_path: Path, variable: str):
  env = _env(tmp_path / variable, 0)
  env.pop(variable)
  result = _run(env)
  assert result.returncode == 2
  assert f"{variable} is required" in result.stdout


def test_dose_eval_rejects_arguments_dirty_noncanonical_or_colliding_inputs(tmp_path: Path):
  env = _env(tmp_path / "args", 0)
  assert _run(env, "unexpected").returncode == 2

  env = _env(tmp_path / "dirty-code", 0)
  (Path(env["RUN_ROOT"]) / "dirty").write_text("no")
  assert _run(env).returncode == 2

  env = _env(tmp_path / "dirty-assets", 0)
  (Path(env["ASSET_REPO"]) / "dirty").write_text("no")
  assert _run(env).returncode == 2

  env = _env(tmp_path / "noncanonical", 0)
  other = tmp_path / "other-assets"
  _repo(other)
  env["ASSET_REPO"] = str(other)
  assert _run(env).returncode == 2

  env = _env(tmp_path / "collision", 0)
  assert _run(env).returncode == 0
  result = _run(env)
  assert result.returncode == 2
  assert "attempt directory already exists" in result.stdout


@pytest.mark.parametrize("task_id", range(4))
def test_dose_eval_rejects_symlink_or_mutated_checkpoint(tmp_path: Path, task_id: int):
  variable = ARMS[task_id][2]
  env = _env(tmp_path / f"symlink-{task_id}", task_id)
  checkpoint = Path(env[variable])
  target = checkpoint.parent / "real_model_499.pt"
  checkpoint.rename(target)
  checkpoint.symlink_to(target)
  result = _run(env)
  assert result.returncode == 2
  assert "regular non-symlink model_499.pt" in result.stdout

  env = _env(tmp_path / f"mutation-{task_id}", task_id)
  env["POSTFLIGHT_MUTATION"] = "checkpoint"
  env["MUTATE_CHECKPOINT"] = env[variable]
  result = _run(env)
  assert result.returncode == 2
  assert "checkpoint SHA-256 drifted" in result.stdout


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("RUNTIME_GPU", "NVIDIA H100 80GB HBM3"),
    ("RUNTIME_MJLAB_VERSION", "1.4.1"),
    ("RUNTIME_MUJOCO_VERSION", "3.8.2"),
    ("RUNTIME_MUJOCO_WARP_VERSION", "3.8.2"),
  ),
)
def test_dose_eval_executes_and_rejects_wrong_gpu_or_runtime(
  tmp_path: Path, variable: str, value: str
):
  env = _env(tmp_path / variable, 0)
  env[variable] = value
  result = _run(env)
  assert result.returncode == 2
  assert "CUDA identity or runtime package versions do not match" in result.stdout


@pytest.mark.parametrize(
  ("mutation", "message"),
  (
    ("code_dirty", "postflight code provenance drifted"),
    ("asset_dirty", "postflight asset provenance drifted"),
    ("code_revision", "postflight code provenance drifted"),
    ("asset_revision", "postflight asset provenance drifted"),
  ),
)
def test_dose_eval_rejects_postflight_provenance_drift(
  tmp_path: Path, mutation: str, message: str
):
  env = _env(tmp_path / mutation, 0)
  env["POSTFLIGHT_MUTATION"] = mutation
  result = _run(env)
  assert result.returncode == 2
  assert message in result.stdout


@pytest.mark.parametrize(
  "mode",
  (
    "missing", "extra", "symlink", "nonfinite", "nonzero_impulse", "wrong_delta",
    "wrong_shape", "wrong_role", "wrong_summary_hash", "wrong_code", "wrong_asset",
    "wrong_live_p", "wrong_seeds", "wrong_population", "wrong_rng",
  ),
)
def test_dose_eval_rejects_missing_nonfinite_or_wrong_outputs(tmp_path: Path, mode: str):
  env = _env(tmp_path / mode, 1)
  env["BAD_OUTPUT_MODE"] = mode
  result = _run(env)
  assert result.returncode == 2
  assert "Z1_VIC_IMPULSE_DOSE_CURVE_EVAL_PASS" not in result.stdout


@pytest.mark.parametrize("variable,mode", (("BAD_CHECKPOINT_SHA_MODE", "fail"), ("BAD_CHECKPOINT_SHA_MODE", "invalid"), ("BAD_MANIFEST_SHA_MODE", "fail"), ("BAD_MANIFEST_SHA_MODE", "invalid")))
def test_dose_eval_rejects_failed_or_invalid_hash_generation(
  tmp_path: Path, variable: str, mode: str
):
  env = _env(tmp_path / f"{variable}-{mode}", 0)
  env[variable] = mode
  result = _run(env)
  assert result.returncode == 2
  assert "Z1_VIC_IMPULSE_DOSE_CURVE_EVAL_PASS" not in result.stdout
