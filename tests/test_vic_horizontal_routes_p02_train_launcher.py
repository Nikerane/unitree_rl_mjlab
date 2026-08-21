"""Shell contract for the guarded horizontal-route VIC impulse-CaT pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from itertools import count

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_horizontal_routes_p02_train.sbatch"
TASK_PREFIX = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT-HorizontalRoutes-"
)
CAPS = "[0.369,0.246,0.738,0.369,0.246,0.0164]"
EXPECTED_MODELS = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499)
ANNEAL_STAGES = [
  {"step": 0, "weight": 0.10},
  {"step": 1200, "weight": 0.08},
  {"step": 2400, "weight": 0.06},
  {"step": 3600, "weight": 0.04},
  {"step": 4800, "weight": 0.02},
  {"step": 6000, "weight": 0.00},
]
ARMS = (
  (0, "horizontal_routes_annealed", f"{TASK_PREFIX}Annealed", 0.1, ANNEAL_STAGES),
  (1, "horizontal_routes_persistent", f"{TASK_PREFIX}Persistent", 0.2, None),
)
ARRAY_JOB_IDS = count(980000)
ELEMENT_JOB_IDS = count(1980000)


def _git(path: Path, *args: str) -> str:
  result = subprocess.run(
    [
      "git",
      "-c",
      "user.email=horizontal-routes-test@example.com",
      "-c",
      "user.name=Horizontal Routes Test",
      *args,
    ],
    cwd=path,
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _repo(path: Path) -> str:
  path.mkdir(parents=True, exist_ok=True)
  (path / "marker").write_text("fixture\n")
  _git(path, "init", "-q")
  _git(path, "add", ".")
  _git(path, "commit", "-qm", "fixture")
  return _git(path, "rev-parse", "HEAD")


def _env(tmp_path: Path, task_id: int) -> dict[str, str]:
  home = tmp_path / "home"
  code = tmp_path / "repos/code"
  assets = tmp_path / "repos/safe_impact_manipulation"
  fake_bin = home / "bin"
  fake_bin.mkdir(parents=True)
  (code / "src/tasks").mkdir(parents=True)
  (code / "src/__init__.py").write_text("")
  (code / "src/tasks/__init__.py").write_text("")
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
    "SLURM_ARRAY_JOB_ID": str(next(ARRAY_JOB_IDS)),
    "SLURM_ARRAY_TASK_ID": str(task_id),
    "SLURM_ARRAY_TASK_COUNT": "2",
    "SLURM_ARRAY_TASK_MIN": "0",
    "SLURM_ARRAY_TASK_MAX": "1",
    "SLURM_ARRAY_TASK_STEP": "1",
    "SLURM_JOB_ID": str(next(ELEMENT_JOB_IDS)),
  }
  python = home / "repos/unitree_rl_mjlab/.venv/bin/python"
  python.parent.mkdir(parents=True)
  runtime_stubs = home / "runtime-stubs"
  runtime_stubs.mkdir()
  (runtime_stubs / "sitecustomize.py").write_text(
    "import os\n"
    "import sys\n"
    "import types\n"
    "from importlib import metadata\n"
    "versions = {\n"
    "  'mjlab': os.environ.get('RUNTIME_MJLAB_VERSION', '1.4.0'),\n"
    "  'mujoco': os.environ.get('RUNTIME_MUJOCO_VERSION', '3.8.1'),\n"
    "  'mujoco-warp': os.environ.get('RUNTIME_MUJOCO_WARP_VERSION', '3.8.1'),\n"
    "}\n"
    "metadata.version = versions.__getitem__\n"
    "torch = types.ModuleType('torch')\n"
    "torch.cuda = types.SimpleNamespace(\n"
    "  is_available=lambda: True,\n"
    "  device_count=lambda: 1,\n"
    "  get_device_name=lambda index: os.environ.get('RUNTIME_GPU', 'NVIDIA A100-SXM4-40GB'),\n"
    ")\n"
    "sys.modules['torch'] = torch\n"
    "class Term:\n"
    "  def __init__(self, *, func=None, mode=None, params=None, weight=None):\n"
    "    self.func = func\n"
    "    self.mode = mode\n"
    "    self.params = {} if params is None else params\n"
    "    self.weight = weight\n"
    "class Group:\n"
    "  def __init__(self, terms): self.terms = terms\n"
    "class Cfg: pass\n"
    "def sample_strike_route_signs(): pass\n"
    "sample_strike_route_signs.__module__ = 'src.tasks.hammer.mdp.references'\n"
    "calls = [0]\n"
    "def load_env_cfg(task):\n"
    "  calls[0] += 1\n"
    "  with open(os.path.join(os.environ['HOME'], 'preflight-loads'), 'a') as f:\n"
    "    f.write(task + '\\n')\n"
    "  mode = os.environ.get('PREFLIGHT_MODE', '')\n"
    "  persistent = task.endswith('-Persistent')\n"
    "  cfg = Cfg()\n"
    "  cfg.actions = {'joint_position': object(), 'joint_stiffness': object()}\n"
    "  amplitude = 0.021 if mode == 'route_amplitude' else 0.020\n"
    "  sampler = (lambda: None) if mode == 'route_sign_sampler' else sample_strike_route_signs\n"
    "  cfg.events = {'sample_strike_route_signs': Term(func=sampler, mode='reset', params={'horizontal_detour_m': amplitude})}\n"
    "  obs = {\n"
    "    'strike_phase': Term(params={'horizontal_detour_m': amplitude}),\n"
    "    'strike_ref_error': Term(params={'horizontal_detour_m': amplitude}),\n"
    "  }\n"
    "  cfg.observations = {'actor': Group(obs.copy()), 'critic': Group(obs.copy())}\n"
    "  weight = 0.2 if persistent else 0.1\n"
    "  if mode == 'reward_weight': weight = 0.9\n"
    "  cfg.rewards = {\n"
    "    'r_imit': Term(weight=weight, params={'sigma': 0.05, 'horizontal_detour_m': amplitude}),\n"
    "    'delivered_impulse': Term(weight=4.0),\n"
    "    'impact_progress': Term(weight=8.0),\n"
    "  }\n"
    "  stages = None if persistent else [\n"
    "    {'step': 0, 'weight': 0.10}, {'step': 1200, 'weight': 0.08},\n"
    "    {'step': 2400, 'weight': 0.06}, {'step': 3600, 'weight': 0.04},\n"
    "    {'step': 4800, 'weight': 0.02}, {'step': 6000, 'weight': 0.00},\n"
    "  ]\n"
    "  if mode == 'reward_schedule': stages = [{'step': 0, 'weight': weight}]\n"
    "  cfg.curriculum = {} if stages is None else {'r_imit_anneal': Term(params={'stages': stages})}\n"
    "  caps = [0.369, 0.246, 0.738, 0.369, 0.246, 0.0164]\n"
    "  if mode == 'impulse_caps': caps[-1] = 0.02\n"
    "  cfg.metrics = {'cat_soft': Term(params={\n"
    "    'use_vel': mode != 'velocity_disabled', 'use_impulse': True,\n"
    "    'imp_limit': caps, 'imp_max_p': 0.2,\n"
    "  })}\n"
    "  if mode == 'config_drift' and calls[0] == 2:\n"
    "    cfg.rewards['impact_progress'].weight = 7.0\n"
    "  return cfg\n"
    "mjlab = types.ModuleType('mjlab')\n"
    "tasks = types.ModuleType('mjlab.tasks')\n"
    "registry = types.ModuleType('mjlab.tasks.registry')\n"
    "registry.load_env_cfg = load_env_cfg\n"
    "mjlab.tasks = tasks\n"
    "tasks.registry = registry\n"
    "sys.modules['mjlab'] = mjlab\n"
    "sys.modules['mjlab.tasks'] = tasks\n"
    "sys.modules['mjlab.tasks.registry'] = registry\n"
    "src = types.ModuleType('src')\n"
    "src.__path__ = []\n"
    "src_tasks = types.ModuleType('src.tasks')\n"
    "src.tasks = src_tasks\n"
    "sys.modules['src'] = src\n"
    "sys.modules['src.tasks'] = src_tasks\n"
  )
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
    "  if iteration == 450 and mode == 'missing': continue\n"
    "  stored = 498 if iteration == 499 and mode == 'wrong_iter' else iteration\n"
    "  value = torch.tensor([float('nan') if iteration == 499 and mode == 'nonfinite' else 1.0])\n"
    "  torch.save({'iter': stored, 'nested': {'state': value}}, root / f'model_{iteration}.pt')\n"
    "if mode == 'extra':\n"
    "  torch.save({'iter': 10, 'state': torch.tensor([1.0])}, root / 'model_10.pt')\n"
    "PY\n"
    "    case \"${POSTFLIGHT_MUTATION:-}\" in\n"
    "      code_dirty) printf 'no\\n' > \"$RUN_ROOT/postflight-dirty\" ;;\n"
    "      asset_dirty) printf 'no\\n' > \"$ASSET_REPO/postflight-dirty\" ;;\n"
    "      code_revision)\n"
    "        printf 'drift\\n' >> \"$RUN_ROOT/marker\"\n"
    "        git -C \"$RUN_ROOT\" add marker\n"
    "        git -C \"$RUN_ROOT\" -c user.email=horizontal-routes-test@example.com -c user.name='Horizontal Routes Test' commit -qm drift\n"
    "        ;;\n"
    "      asset_revision)\n"
    "        printf 'drift\\n' >> \"$ASSET_REPO/marker\"\n"
    "        git -C \"$ASSET_REPO\" add marker\n"
    "        git -C \"$ASSET_REPO\" -c user.email=horizontal-routes-test@example.com -c user.name='Horizontal Routes Test' commit -qm drift\n"
    "        ;;\n"
    "    esac\n"
    "    ;;\n"
    "  -)\n"
    f"    if [ \"$#\" -eq 1 ]; then PYTHONPATH=\"{runtime_stubs}:${{PYTHONPATH:-}}\" exec \"{sys.executable}\" \"$@\"; fi\n"
    f"    exec \"{sys.executable}\" \"$@\"\n"
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


def _config_preflight_program() -> str:
  source = LAUNCHER.read_text()
  match = re.search(
    r'"\$PY" - <<\'PY\' \|\| fail "horizontal-route VIC-TT serialized config preflight failed"\n'
    r"(?P<program>.*?)\nPY\n\ncd \"\$ATTEMPT_DIR\"",
    source,
    flags=re.DOTALL,
  )
  assert match is not None
  return match.group("program")


def _expected_identity(task: str, reward_weight: float, stages: list[dict] | None) -> dict:
  return {
    "actions": ["joint_position", "joint_stiffness"],
    "impulse_cat": {
      "caps": [0.369, 0.246, 0.738, 0.369, 0.246, 0.0164],
      "max_p": 0.2,
      "use_impulse": True,
      "use_velocity": True,
    },
    "maximize_rewards": {"delivered_impulse": 4.0, "impact_progress": 8.0},
    "route": {
      "amplitude_m": 0.02,
      "event_mode": "reset",
      "sign_sampler": "src.tasks.hammer.mdp.references.sample_strike_route_signs",
    },
    "schedule": {
      "r_imit_sigma": 0.05,
      "r_imit_stages": stages,
      "r_imit_weight": reward_weight,
      "r_imit_zero_iteration": 250 if stages is not None else None,
    },
    "task": task,
  }


def test_horizontal_route_arms_run_only_the_two_frozen_serialized_configs(
  tmp_path: Path,
) -> None:
  """Breaks if either index trains the wrong task, route, schedule, dose, or budget."""
  calls = []
  for task_id, role, task, reward_weight, stages in ARMS:
    env = _env(tmp_path / role, task_id)
    result = _run(env)

    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Z1_VIC_HORIZONTAL_ROUTES_P02_TRAIN_PASS role={role} task={task}" in result.stdout
    config_line = next(
      line for line in result.stdout.splitlines()
      if line.startswith("Z1_VIC_HORIZONTAL_ROUTES_P02_TRAIN_CONFIG ")
    )
    serialized = config_line.split(" serialized=", 1)[1]
    assert json.loads(serialized) == _expected_identity(task, reward_weight, stages)
    assert re.search(r" identity_sha256=[0-9a-f]{64} serialized=", config_line)
    checkpoint_lines = [
      line for line in result.stdout.splitlines()
      if line.startswith("Z1_VIC_HORIZONTAL_ROUTES_P02_TRAIN_CHECKPOINT ")
    ]
    assert len(checkpoint_lines) == len(EXPECTED_MODELS)
    assert all(re.search(r"sha256=[0-9a-f]{64}$", line) for line in checkpoint_lines)
    assert (Path(env["HOME"]) / "preflight-loads").read_text().splitlines() == [task, task]
    calls.append(_training_call(env))

  for call, (_, role, task, _, _) in zip(calls, ARMS, strict=True):
    assert call[1] == task
    assert _value(call, "--agent.seed") == "2"
    assert _value(call, "--agent.num-steps-per-env") == "24"
    assert _value(call, "--agent.max-iterations") == "500"
    assert _value(call, "--agent.save-interval") == "50"
    assert _value(call, "--env.scene.num-envs") == "4096"
    assert _value(call, "--env.metrics.cat-soft.params.imp-limit") == CAPS
    assert _value(call, "--env.metrics.cat-soft.params.imp-max-p") == "0.2"
    assert role in _value(call, "--agent.run-name")
    assert "--resume" not in call

  normalized = []
  for call in calls:
    copy = list(call)
    copy[0] = "<repo>/scripts/train.py"
    copy[1] = "<arm-task>"
    copy[copy.index("--agent.run-name") + 1] = "<arm-and-job-specific-run-name>"
    normalized.append(copy)
  assert normalized[0] == normalized[1]


@pytest.mark.parametrize(
  ("role", "task"),
  ((role, task) for _, role, task, _, _ in ARMS),
)
def test_horizontal_route_preflight_accepts_only_the_real_registered_arm_configs(
  tmp_path: Path, role: str, task: str
) -> None:
  """Breaks if the embedded serializer disagrees with either live Task 1 config."""
  env = dict(os.environ)
  env.update(
    {
      "PYTHONPATH": str(ROOT),
      "TASK": task,
      "ROLE": role,
      "CAPS": CAPS,
      "IMP_MAX_P": "0.2",
      "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
  )

  result = subprocess.run(
    [sys.executable, "-"],
    input=_config_preflight_program(),
    cwd=ROOT,
    env=env,
    capture_output=True,
    text=True,
    timeout=30,
  )

  assert result.returncode == 0, result.stderr + result.stdout
  assert "Z1_VIC_HORIZONTAL_ROUTES_P02_TRAIN_CONFIG" in result.stdout


def test_horizontal_route_launcher_freezes_resources_names_and_no_retry_contract() -> None:
  """Breaks if submission can use another campaign, resource shape, or retry path."""
  source = LAUNCHER.read_text()
  campaign = "z1-vic-horizontal-routes-p02-train"

  assert "#SBATCH --array=0-1" in source
  assert "#SBATCH --gres=gpu:1" in source
  assert "#SBATCH --cpus-per-task=8" in source
  assert "#SBATCH --mem=40G" in source
  assert "#SBATCH --time=01:30:00" in source
  assert "#SBATCH --exclude=gn03,gn34" in source
  assert "#SBATCH --no-requeue" in source
  assert f"#SBATCH --job-name={campaign}" in source
  assert f"campaigns/{campaign}/slurm/{campaign}-%A_%a.out" in source
  assert f"campaigns/{campaign}/slurm/{campaign}-%A_%a.err" in source
  assert f'CAMPAIGN="{campaign}"' in source
  assert "%A_%a" in source
  assert "scontrol requeue" not in source
  assert "sbatch " not in source
  assert "--resume" not in source
  assert "EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)" in source
  assert not re.search(r"mkdir[^\n]*campaigns/[^\n]*/slurm", source)


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
    ("SLURM_JOB_ID", "not-numeric"),
    ("WORLD_SIZE", "1"),
    ("RANK", "0"),
    ("LOCAL_RANK", "0"),
    ("MASTER_ADDR", "localhost"),
    ("MASTER_PORT", "29500"),
    ("RESUME_PATH", "/tmp/model.pt"),
    ("TASK", f"{TASK_PREFIX}Annealed"),
    ("ROLE", "horizontal_routes_annealed"),
    ("CAPS", CAPS),
    ("IMP_MAX_P", "0.2"),
    ("RUN_NAME", "injected"),
    ("EXPECTED_CODE_REVISION", "not-a-revision"),
    ("EXPECTED_ASSET_REVISION", "not-a-revision"),
  ),
)
def test_horizontal_route_launcher_rejects_other_index_env_or_execution_context(
  tmp_path: Path, variable: str, value: str
) -> None:
  """Breaks if an index or inherited training/resume/distributed override can run."""
  env = _env(tmp_path / variable.lower(), 0)
  env[variable] = value

  result = _run(env)

  assert result.returncode == 2


def test_horizontal_route_launcher_rejects_args_provenance_and_collisions(
  tmp_path: Path,
) -> None:
  """Breaks if a run can bypass no-argument, clean, canonical, or fresh-leaf guards."""
  env = _env(tmp_path / "args", 0)
  assert _run(env, "unexpected").returncode == 2

  env = _env(tmp_path / "dirty-code", 0)
  (Path(env["RUN_ROOT"]) / "dirty").write_text("no\n")
  assert _run(env).returncode == 2

  env = _env(tmp_path / "dirty-assets", 0)
  (Path(env["ASSET_REPO"]) / "dirty").write_text("no\n")
  assert _run(env).returncode == 2

  env = _env(tmp_path / "wrong-code-revision", 0)
  env["EXPECTED_CODE_REVISION"] = "0" * 40
  assert _run(env).returncode == 2

  env = _env(tmp_path / "wrong-asset-revision", 0)
  env["EXPECTED_ASSET_REVISION"] = "0" * 40
  assert _run(env).returncode == 2

  env = _env(tmp_path / "noncanonical", 0)
  other_assets = tmp_path / "other-assets"
  _repo(other_assets)
  env["ASSET_REPO"] = str(other_assets)
  assert _run(env).returncode == 2

  env = _env(tmp_path / "collision", 1)
  assert _run(env).returncode == 0
  assert _run(env).returncode == 2


@pytest.mark.parametrize(
  "mode",
  (
    "route_amplitude",
    "route_sign_sampler",
    "reward_weight",
    "reward_schedule",
    "impulse_caps",
    "velocity_disabled",
    "config_drift",
  ),
)
@pytest.mark.parametrize("task_id", (0, 1))
def test_horizontal_route_launcher_rejects_serialized_config_drift(
  tmp_path: Path, mode: str, task_id: int
) -> None:
  """Breaks if either arm can train with a different route, schedule, cap, or CaT identity."""
  env = _env(tmp_path / f"{task_id}-{mode}", task_id)
  env["PREFLIGHT_MODE"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "serialized config preflight failed" in result.stdout
  assert "Z1_VIC_HORIZONTAL_ROUTES_P02_TRAIN_PASS" not in result.stdout


@pytest.mark.parametrize("mode", ("missing", "wrong_iter", "nonfinite", "extra"))
def test_horizontal_route_launcher_rejects_invalid_checkpoint_set_or_content(
  tmp_path: Path, mode: str
) -> None:
  """Breaks if a partial, mislabelled, extra, or nonfinite checkpoint leaf is accepted."""
  env = _env(tmp_path / mode, 1)
  env["BAD_CHECKPOINT_MODE"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "checkpoint" in result.stdout
  assert "Z1_VIC_HORIZONTAL_ROUTES_P02_TRAIN_PASS" not in result.stdout


@pytest.mark.parametrize(
  ("variable", "value"),
  (
    ("RUNTIME_GPU", "NVIDIA H100 80GB HBM3"),
    ("RUNTIME_MJLAB_VERSION", "1.4.1"),
    ("RUNTIME_MUJOCO_VERSION", "3.8.2"),
    ("RUNTIME_MUJOCO_WARP_VERSION", "3.8.2"),
  ),
)
def test_horizontal_route_launcher_rejects_runtime_identity_drift(
  tmp_path: Path, variable: str, value: str
) -> None:
  """Breaks if a non-A100 or unqualified runtime can produce a checkpoint."""
  env = _env(tmp_path / variable.lower(), 0)
  env[variable] = value

  result = _run(env)

  assert result.returncode == 2
  assert "CUDA identity or runtime package versions do not match" in result.stdout


@pytest.mark.parametrize(
  "mode", ("code_dirty", "asset_dirty", "code_revision", "asset_revision")
)
def test_horizontal_route_launcher_rejects_postflight_provenance_drift(
  tmp_path: Path, mode: str
) -> None:
  """Breaks if code or assets can change while either arm is training."""
  env = _env(tmp_path / mode, 0)
  env["POSTFLIGHT_MUTATION"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "provenance drifted" in result.stdout
  assert "Z1_VIC_HORIZONTAL_ROUTES_P02_TRAIN_PASS" not in result.stdout


@pytest.mark.parametrize("mode", ("fail", "empty", "invalid"))
def test_horizontal_route_launcher_rejects_failed_or_malformed_checkpoint_hash(
  tmp_path: Path, mode: str
) -> None:
  """Breaks if a failed, absent, or non-lowercase digest can be emitted as provenance."""
  env = _env(tmp_path / mode, 0)
  env["BAD_SHA256_MODE"] = mode

  result = _run(env)

  assert result.returncode == 2
  assert "checkpoint SHA-256 validation failed" in result.stdout
  assert "Z1_VIC_HORIZONTAL_ROUTES_P02_TRAIN_PASS" not in result.stdout
