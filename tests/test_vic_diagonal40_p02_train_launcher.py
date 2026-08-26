"""Shell contract for the bounded 40-mm diagonal-start VIC canary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from tests import test_vic_horizontal_routes_p02_train_launcher as qualified


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_diagonal40_p02_train.sbatch"
TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT-DiagonalStarts40mm-Persistent"
)
ROLE = "diagonal40_persistent_p02"
CAPS = "[0.369,0.246,0.738,0.369,0.246,0.0164]"
ROWS = [
  [0.0, 1.487947605122, -0.319221074008, -1.190426531114, -0.0013, 1.5544],
  [0.0, 1.606, -0.4301, -1.1976, -0.0013, 1.5544],
  [0.0, 1.709237021603, -0.555950051788, -1.174986969814, -0.0013, 1.5544],
]


def _env(tmp_path: Path) -> dict[str, str]:
  """Reuse the qualified launcher harness, replacing only its registered config."""
  env = qualified._env(tmp_path, 0)
  for name in (
    "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_TASK_MIN", "SLURM_ARRAY_TASK_MAX", "SLURM_ARRAY_TASK_STEP",
  ):
    env.pop(name)
  stub = Path(env["HOME"]) / "runtime-stubs/sitecustomize.py"
  stub.write_text(stub.read_text() + f'''\n
def sample_strike_route_signs(): pass
sample_strike_route_signs.__module__ = "src.tasks.hammer.mdp.references"
def reset_joints_by_offset(): pass
reset_joints_by_offset.__module__ = "mjlab.envs.mdp.events"
def reset_joints_to_strike_route_starts(): pass
reset_joints_to_strike_route_starts.__module__ = "src.tasks.hammer.mdp.references"
def expand_variable_impedance_model_fields(): pass
expand_variable_impedance_model_fields.__module__ = "src.tasks.hammer.mdp.variable_impedance"
ROWS = {ROWS!r}
def diagonal40_cfg(task):
  if task != {TASK!r}: raise RuntimeError("wrong task")
  with open(os.path.join(os.environ["HOME"], "preflight-loads"), "a") as f: f.write(task + "\\n")
  mode = os.environ.get("PREFLIGHT_MODE", "")
  route_rows = ROWS if mode != "pose_rows" else ROWS[:2]
  cat_params = {{"use_vel": mode != "velocity_disabled", "use_impulse": True,
                "imp_limit": [0.369, 0.246, 0.738, 0.369, 0.246, 0.0164], "imp_max_p": 0.2}}
  if mode == "impulse_caps": cat_params["imp_limit"][-1] = 0.02
  if mode == "missing_identity_field": cat_params.pop("use_vel")
  cfg = Cfg()
  cfg.actions = {{"joint_position": object(), "joint_stiffness": object()}}
  cfg.events = {{
    "sample_strike_route_signs": Term(func=sample_strike_route_signs, mode="reset", params={{"horizontal_detour_m": 0.0, "followthrough_mode": "strike_axis"}}),
    "reset_robot_joints": Term(func=reset_joints_by_offset, mode="reset", params={{"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": {{"name": "robot", "joint_names": [".*"], "site_names": None, "preserve_order": False}}}}),
    "reset_strike_route_joints": Term(func=reset_joints_to_strike_route_starts, mode="reset", params={{"route_joint_positions": route_rows, "asset_cfg": {{"name": "robot", "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], "site_names": None, "preserve_order": False}}}}),
    "reset_nail": Term(func=reset_joints_by_offset, mode="reset", params={{"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": {{"name": "nail_block", "joint_names": ["nail_slide"], "site_names": None, "preserve_order": False}}}}),
    "expand_variable_impedance_model_fields": Term(func=expand_variable_impedance_model_fields, mode="startup", params={{}}),
  }}
  names = ["joint_pos", "joint_vel", "ee_pos", "ee_vel", "head_pos", "head_vel", "nail_top_pos", "nail_depth", "strike_phase", "strike_ref_error", "actions"]
  terms = {{name: Term(params={{"followthrough_mode": "strike_axis"}} if name in ("strike_phase", "strike_ref_error") else {{}}) for name in names}}
  cfg.observations = {{"actor": Group(terms.copy()), "critic": Group(terms.copy())}}
  cfg.rewards = {{"r_imit": Term(weight=0.2, params={{"sigma": 0.05, "followthrough_mode": "strike_axis"}}), "delivered_impulse": Term(weight=4.0), "impact_progress": Term(weight=8.0)}}
  cfg.curriculum = {{}}
  cfg.metrics = {{"cat_soft": Term(params=cat_params)}}
  return cfg
calls = [0]
def load_env_cfg(task):
  calls[0] += 1
  cfg = diagonal40_cfg(task)
  if os.environ.get("PREFLIGHT_MODE") == "repeated_drift" and calls[0] == 2: cfg.rewards["impact_progress"].weight = 7.0
  return cfg
registry.load_env_cfg = load_env_cfg
''')
  return env


def _run(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], *args: str):
  monkeypatch.setattr(qualified, "LAUNCHER", LAUNCHER)
  return qualified._run(env, *args)


def _training_call(env: dict[str, str]) -> list[str]:
  return qualified._training_call(env)


def _value(call: list[str], flag: str) -> str:
  return qualified._value(call, flag)


def _config_preflight_program() -> str:
  source = LAUNCHER.read_text()
  start = '"$PY" - <<\'PY\' || fail "diagonal40 serialized config preflight failed"\n'
  _, found, remainder = source.partition(start)
  assert found
  program, found, _ = remainder.partition("\nPY\ncd \"$ATTEMPT_DIR\"")
  assert found
  return program


def test_diagonal40_launcher_runs_only_the_frozen_treatment(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Breaks if the one job can train another task, budget, or CaT treatment."""
  env = _env(tmp_path)
  result = _run(monkeypatch, env)

  assert result.returncode == 0, result.stderr + result.stdout
  assert f"Z1_VIC_DIAGONAL40_P02_TRAIN_PASS role={ROLE} task={TASK}" in result.stdout
  config_line = next(line for line in result.stdout.splitlines() if "_CONFIG " in line)
  identity = json.loads(config_line.split(" serialized=", 1)[1])
  assert identity["task"] == TASK
  assert identity["route_joint_positions_rad"] == ROWS
  assert identity["event_names"][:3] == ["sample_strike_route_signs", "reset_robot_joints", "reset_strike_route_joints"]
  assert identity["actions"] == ["joint_position", "joint_stiffness"]
  assert identity["schedule"] == {"r_imit_weight": 0.2, "r_imit_sigma": 0.05, "r_imit_anneal": None}
  assert re.search(r" identity_sha256=[0-9a-f]{64} serialized=", config_line)
  assert (Path(env["HOME"]) / "preflight-loads").read_text().splitlines() == [TASK, TASK]
  call = _training_call(env)
  assert call[1] == TASK
  assert _value(call, "--agent.seed") == "2"
  assert _value(call, "--agent.num-steps-per-env") == "24"
  assert _value(call, "--agent.max-iterations") == "500"
  assert _value(call, "--agent.save-interval") == "50"
  assert _value(call, "--env.scene.num-envs") == "4096"
  assert _value(call, "--env.metrics.cat-soft.params.imp-limit") == CAPS
  assert _value(call, "--env.metrics.cat-soft.params.imp-max-p") == "0.2"
  assert ROLE in _value(call, "--agent.run-name")
  checkpoint_lines = [line for line in result.stdout.splitlines() if "_CHECKPOINT " in line]
  assert len(checkpoint_lines) == 11
  assert all(re.search(r"sha256=[0-9a-f]{64}$", line) for line in checkpoint_lines)


def test_diagonal40_preflight_accepts_the_registered_config(tmp_path: Path) -> None:
  """The serialized identity is bound to the real registered 40-mm task."""
  env = dict(os.environ, PYTHONPATH=str(ROOT), TASK=TASK, ROLE=ROLE, CAPS=CAPS, IMP_MAX_P="0.2", MPLCONFIGDIR=str(tmp_path))
  result = subprocess.run([sys.executable, "-"], input=_config_preflight_program(), cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
  assert result.returncode == 0, result.stderr + result.stdout
  assert "Z1_VIC_DIAGONAL40_P02_TRAIN_CONFIG" in result.stdout


def test_diagonal40_launcher_freezes_resource_and_no_retry_contract() -> None:
  source = LAUNCHER.read_text()
  campaign = "z1-vic-diagonal40-p02-train"
  assert f"#SBATCH --job-name={campaign}" in source
  for directive in ("#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=40G", "#SBATCH --time=01:30:00", "#SBATCH --exclude=gn03,gn34", "#SBATCH --no-requeue"):
    assert directive in source
  assert "#SBATCH --array" not in source
  assert "scontrol requeue" not in source and "sbatch " not in source and "--resume" not in source
  assert "EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)" in source


@pytest.mark.parametrize("variable", ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "RESUME_PATH", "TASK", "ROLE", "CAPS", "IMP_MAX_P", "RUN_NAME"))
def test_diagonal40_launcher_rejects_inherited_execution_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
  env = _env(tmp_path)
  env[variable] = "injected"
  assert _run(monkeypatch, env).returncode == 2


def test_diagonal40_launcher_rejects_args_provenance_and_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  env = _env(tmp_path / "args")
  assert _run(monkeypatch, env, "unexpected").returncode == 2
  env = _env(tmp_path / "dirty")
  (Path(env["RUN_ROOT"]) / "dirty").write_text("no\\n")
  assert _run(monkeypatch, env).returncode == 2
  env = _env(tmp_path / "revision")
  env["EXPECTED_CODE_REVISION"] = "0" * 40
  assert _run(monkeypatch, env).returncode == 2
  env = _env(tmp_path / "collision")
  assert _run(monkeypatch, env).returncode == 0
  assert _run(monkeypatch, env).returncode == 2


@pytest.mark.parametrize("mode", ("pose_rows", "impulse_caps", "velocity_disabled", "missing_identity_field", "repeated_drift"))
def test_diagonal40_launcher_rejects_config_identity_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
  env = _env(tmp_path)
  env["PREFLIGHT_MODE"] = mode
  result = _run(monkeypatch, env)
  assert result.returncode == 2
  assert "serialized config preflight failed" in result.stdout


@pytest.mark.parametrize("mode", ("missing", "wrong_iter", "nonfinite", "extra"))
def test_diagonal40_launcher_rejects_checkpoint_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
  env = _env(tmp_path)
  env["BAD_CHECKPOINT_MODE"] = mode
  result = _run(monkeypatch, env)
  assert result.returncode == 2
  assert "checkpoint" in result.stdout


@pytest.mark.parametrize(("variable", "value"), (("RUNTIME_GPU", "NVIDIA H100 80GB HBM3"), ("RUNTIME_MJLAB_VERSION", "1.4.1"), ("RUNTIME_MUJOCO_VERSION", "3.8.2"), ("RUNTIME_MUJOCO_WARP_VERSION", "3.8.2")))
def test_diagonal40_launcher_rejects_runtime_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, value: str) -> None:
  env = _env(tmp_path)
  env[variable] = value
  result = _run(monkeypatch, env)
  assert result.returncode == 2
  assert "CUDA identity or runtime package versions do not match" in result.stdout


@pytest.mark.parametrize("mode", ("code_dirty", "asset_dirty", "code_revision", "asset_revision"))
def test_diagonal40_launcher_rejects_postflight_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
  env = _env(tmp_path)
  env["POSTFLIGHT_MUTATION"] = mode
  result = _run(monkeypatch, env)
  assert result.returncode == 2
  assert "provenance drifted" in result.stdout


@pytest.mark.parametrize("mode", ("fail", "empty", "invalid", "uppercase"))
def test_diagonal40_launcher_rejects_bad_checkpoint_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
  env = _env(tmp_path)
  env["BAD_SHA256_MODE"] = mode
  result = _run(monkeypatch, env)
  assert result.returncode == 2
  assert "checkpoint SHA-256 validation failed" in result.stdout
