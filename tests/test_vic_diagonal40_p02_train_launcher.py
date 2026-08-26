"""Shell contract for the bounded 40-mm diagonal-start VIC canary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
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
def joint_pos_rel(): pass
joint_pos_rel.__module__ = "mjlab.envs.mdp.observations"
def joint_vel_rel(): pass
joint_vel_rel.__module__ = "mjlab.envs.mdp.observations"
def ee_pos_b(): pass
ee_pos_b.__module__ = "src.tasks.hammer.mdp.observations"
def ee_vel_b(): pass
ee_vel_b.__module__ = "src.tasks.hammer.mdp.observations"
def hammer_head_pos_b(): pass
hammer_head_pos_b.__module__ = "src.tasks.hammer.mdp.observations"
def hammer_head_vel_b(): pass
hammer_head_vel_b.__module__ = "src.tasks.hammer.mdp.observations"
def nail_top_pos_w(): pass
nail_top_pos_w.__module__ = "src.tasks.hammer.mdp.observations"
def nail_depth(): pass
nail_depth.__module__ = "src.tasks.hammer.mdp.observations"
def strike_phase(): pass
strike_phase.__module__ = "src.tasks.hammer.mdp.observations"
def strike_ref_error(): pass
strike_ref_error.__module__ = "src.tasks.hammer.mdp.observations"
def last_action(): pass
last_action.__module__ = "mjlab.envs.mdp.observations"
def ImitationPriorTerm(): pass
ImitationPriorTerm.__module__ = "src.tasks.hammer.mdp.rewards"
class JointPositionActionCfg:
  def __init__(self):
    self.entity_name = "robot"
    self.clip = {{"joint1": (-2.61799, 2.61799), "joint2": (0.0, 2.96706), "joint3": (-2.87979, 0.0), "joint4": (-1.51844, 1.51844), "joint5": (-1.3439, 1.3439), "joint6": (-2.79253, 2.79253)}}
    self.transmission_type = "joint"
    self.actuator_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    self.scale = {{"joint1": 0.05570376467774623, "joint2": 0.5878274488449098, "joint3": 0.14464383006095885, "joint4": 0.22881677627563476, "joint5": 0.05571571884909646, "joint6": 0.05516456842422486}}
    self.offset = 0.0
    self.preserve_order = True
    self.use_default_offset = True
JointPositionActionCfg.__module__ = "mjlab.envs.mdp.actions.actions"
class JointStiffnessActionCfg:
  def __init__(self):
    self.entity_name = "robot"; self.clip = None
    self.joint_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    self.C = 1.25
JointStiffnessActionCfg.__module__ = "src.tasks.hammer.mdp.variable_impedance"
class UniformNoiseCfg:
  def __init__(self, n_min, n_max):
    self.operation = "add"; self.n_min = n_min; self.n_max = n_max
UniformNoiseCfg.__module__ = "mjlab.utils.noise.noise_cfg"
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
  cfg.actions = {{"joint_position": JointPositionActionCfg(), "joint_stiffness": JointStiffnessActionCfg()}}
  if mode == "action_config": cfg.actions["joint_stiffness"].C = 2.0
  cfg.events = {{
    "sample_strike_route_signs": Term(func=sample_strike_route_signs, mode="reset", params={{"horizontal_detour_m": 0.0, "followthrough_mode": "strike_axis"}}),
    "reset_robot_joints": Term(func=reset_joints_by_offset, mode="reset", params={{"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": {{"name": "robot", "joint_names": [".*"], "site_names": None, "preserve_order": False}}}}),
    "reset_strike_route_joints": Term(func=reset_joints_to_strike_route_starts, mode="reset", params={{"route_joint_positions": route_rows, "asset_cfg": {{"name": "robot", "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], "site_names": None, "preserve_order": False}}}}),
    "reset_nail": Term(func=reset_joints_by_offset, mode="reset", params={{"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": {{"name": "nail_block", "joint_names": ["nail_slide"], "site_names": None, "preserve_order": False}}}}),
    "expand_variable_impedance_model_fields": Term(func=expand_variable_impedance_model_fields, mode="startup", params={{}}),
  }}
  asset = lambda name, joints=None, sites=None: {{"name": name, "joint_names": joints, "site_names": sites, "preserve_order": False}}
  def observation_term(func, params=None, noise=None):
    result = Term(func=func, params=params)
    result.noise = noise; result.clip = None; result.scale = None
    result.delay_min_lag = 0; result.delay_max_lag = 0; result.delay_per_env = True
    result.delay_hold_prob = 0.0; result.delay_update_period = 0; result.delay_per_env_phase = True
    result.history_length = 0; result.flatten_history_dim = True
    return result
  n = UniformNoiseCfg
  terms = {{
    "joint_pos": observation_term(joint_pos_rel, noise=n(-0.01, 0.01)), "joint_vel": observation_term(joint_vel_rel, noise=n(-1.5, 1.5)),
    "ee_pos": observation_term(ee_pos_b, {{"asset_cfg": asset("robot", sites=["ee_center_site"])}}, n(-0.005, 0.005)),
    "ee_vel": observation_term(ee_vel_b, {{"asset_cfg": asset("robot", sites=["ee_center_site"])}}, n(-0.01, 0.01)),
    "head_pos": observation_term(hammer_head_pos_b, {{"asset_cfg": asset("robot", sites=["hammer_head_site"])}}, n(-0.005, 0.005)),
    "head_vel": observation_term(hammer_head_vel_b, {{"asset_cfg": asset("robot", sites=["hammer_head_site"])}}, n(-0.01, 0.01)),
    "nail_top_pos": observation_term(nail_top_pos_w, {{"asset_cfg": asset("nail_block", sites=["nail_top"])}}, n(-0.002, 0.002)),
    "nail_depth": observation_term(nail_depth, {{"asset_cfg": asset("nail_block", joints=["nail_slide"])}}, n(-0.001, 0.001)),
    "strike_phase": observation_term(strike_phase, {{"robot_cfg": asset("robot", sites=["hammer_head_site"]), "nail_cfg": asset("nail_block", sites=["nail_top"]), "followthrough_mode": "strike_axis"}}),
    "strike_ref_error": observation_term(strike_ref_error, {{"robot_cfg": asset("robot", sites=["hammer_head_site"]), "nail_cfg": asset("nail_block", sites=["nail_top"]), "followthrough_mode": "strike_axis"}}),
    "actions": observation_term(last_action, {{"action_name": "joint_position"}}),
  }}
  if mode == "observation_function": terms["strike_phase"].func = lambda: None
  actor, critic = Group(terms.copy()), Group(terms.copy())
  for group, corrupt in ((actor, True), (critic, False)):
    group.concatenate_terms = True; group.concatenate_dim = -1; group.enable_corruption = corrupt
    group.history_length = None; group.flatten_history_dim = True
    group.nan_policy = "disabled"; group.nan_check_per_term = True
  if mode == "observation_order": actor.terms = dict(reversed(tuple(actor.terms.items())))
  if mode == "observation_noise": actor.terms["joint_pos"].noise.n_max = 0.02
  if mode == "observation_corruption": actor.enable_corruption = False
  cfg.observations = {{"actor": actor, "critic": critic}}
  cfg.rewards = {{"r_imit": Term(func=ImitationPriorTerm, weight=0.2, params={{"sensor_name": "hammer_nail_contact", "robot_cfg": asset("robot", sites=["hammer_head_site"]), "nail_cfg": asset("nail_block", sites=["nail_top"]), "sigma": 0.05, "followthrough_mode": "strike_axis"}}), "delivered_impulse": Term(weight=4.0), "impact_progress": Term(weight=8.0)}}
  if mode == "imitation_reference": cfg.rewards["r_imit"].params["followthrough_mode"] = "vertical"
  if mode == "imitation_detour": cfg.rewards["r_imit"].params["horizontal_detour_m"] = 0.02
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
  real_git = shutil.which("git")
  assert real_git is not None
  fake_git = Path(env["HOME"]) / "bin/git"
  fake_git.write_text(
    "#!/bin/sh\n"
    "if [ \"${3:-}\" = status ]; then\n"
    "  count_file=\"$HOME/git-status-count\"\n"
    "  count=$(cat \"$count_file\" 2>/dev/null || printf 0)\n"
    "  count=$((count + 1))\n"
    "  printf '%s\\n' \"$count\" > \"$count_file\"\n"
    "  case \"${GIT_STATUS_FAILURE:-}:$count\" in\n"
    "    initial_code:1|initial_asset:2|postflight_code:3|postflight_asset:4) exit 9 ;;\n"
    "  esac\n"
    "fi\n"
    f"exec {real_git!s} \"$@\"\n"
  )
  fake_git.chmod(0o755)
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
  assert identity["action_configs"]["joint_stiffness"]["params"]["C"] == 1.25
  assert identity["imitation_reference"]["function"] == "src.tasks.hammer.mdp.rewards.ImitationPriorTerm"
  assert identity["imitation_reference"]["params"]["followthrough_mode"] == "strike_axis"
  actor_observations = identity["observations"]["actor"]
  assert [term["name"] for term in actor_observations["terms"]] == [
    "joint_pos", "joint_vel", "ee_pos", "ee_vel", "head_pos", "head_vel",
    "nail_top_pos", "nail_depth", "strike_phase", "strike_ref_error", "actions",
  ]
  assert actor_observations["settings"]["enable_corruption"] is True
  assert identity["observations"]["critic"]["settings"]["enable_corruption"] is False
  assert actor_observations["terms"][0]["noise"] == {
    "class": "mjlab.utils.noise.noise_cfg.UniformNoiseCfg",
    "params": {"n_max": 0.01, "n_min": -0.01, "operation": "add"},
  }
  assert actor_observations["terms"][8]["function"] == "src.tasks.hammer.mdp.observations.strike_phase"
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
  env = _env(tmp_path / "asset-dirty")
  (Path(env["ASSET_REPO"]) / "dirty").write_text("no\\n")
  assert _run(monkeypatch, env).returncode == 2
  env = _env(tmp_path / "revision")
  env["EXPECTED_CODE_REVISION"] = "0" * 40
  assert _run(monkeypatch, env).returncode == 2
  env = _env(tmp_path / "asset-revision")
  env["EXPECTED_ASSET_REVISION"] = "0" * 40
  assert _run(monkeypatch, env).returncode == 2
  env = _env(tmp_path / "noncanonical")
  other_assets = tmp_path / "other-assets"
  qualified._repo(other_assets)
  env["ASSET_REPO"] = str(other_assets)
  assert _run(monkeypatch, env).returncode == 2
  env = _env(tmp_path / "collision")
  assert _run(monkeypatch, env).returncode == 0
  assert _run(monkeypatch, env).returncode == 2


@pytest.mark.parametrize("mode", ("pose_rows", "impulse_caps", "velocity_disabled", "missing_identity_field", "repeated_drift", "imitation_reference", "imitation_detour", "observation_function", "observation_order", "observation_noise", "observation_corruption", "action_config"))
def test_diagonal40_launcher_rejects_config_identity_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
  env = _env(tmp_path)
  env["PREFLIGHT_MODE"] = mode
  result = _run(monkeypatch, env)
  assert result.returncode == 2
  assert "serialized config preflight failed" in result.stdout


@pytest.mark.parametrize(
  ("mode", "message"),
  (
    ("initial_code", "code repository status check failed"),
    ("initial_asset", "asset repository status check failed"),
    ("postflight_code", "postflight code status check failed"),
    ("postflight_asset", "postflight asset status check failed"),
  ),
)
def test_diagonal40_launcher_rejects_git_status_command_failures(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
  """A failed status subprocess must not collapse to an apparently clean tree."""
  env = _env(tmp_path)
  env["GIT_STATUS_FAILURE"] = mode
  result = _run(monkeypatch, env)
  assert result.returncode == 2
  assert message in result.stdout


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
