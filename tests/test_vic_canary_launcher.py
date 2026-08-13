"""Public shell contract for the single VIC-TT Vega engineering canary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/slurm/vega_vic_canary.sbatch"
TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-VariableImpedance-TT"
)
ITERATIONS = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499)


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=vic-canary-test@example.com",
            "-c",
            "user.name=VIC Canary Test",
            *args,
        ],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )


def _git_repo(path: Path, marker: str) -> str:
    path.mkdir(parents=True)
    (path / "marker.txt").write_text(marker)
    _git(path, "init", "-q")
    _git(path, "add", "marker.txt")
    _git(path, "commit", "-q", "-m", "fixture")
    return _git(path, "rev-parse", "HEAD").stdout.strip()


def _base_env(tmp_path: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "RUN_ROOT": str(tmp_path / "repos/code"),
        "ASSET_REPO": str(tmp_path / "repos/safe_impact_manipulation"),
        "SLURM_JOB_ID": "12345",
    }
    env["EXPECTED_CODE_REVISION"] = _git_repo(Path(env["RUN_ROOT"]), "code")
    env["EXPECTED_ASSET_REVISION"] = _git_repo(Path(env["ASSET_REPO"]), "assets")
    return env


def _install_fake_python(env: dict[str, str], *, mode: str = "success") -> None:
    py = Path(env["HOME"]) / "repos/unitree_rl_mjlab/.venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(
        "#!/bin/sh\n"
        "{ printf 'CALL'; for arg in \"$@\"; do printf '\\t%s' \"$arg\"; done; printf '\\n'; } >> \"$HOME/python_calls.log\"\n"
        "mode=${FAKE_PY_MODE:-success}\n"
        "if [ \"${1:-}\" = - ]; then\n"
        "  case \"$#\" in\n"
        "    1) [ \"$mode\" != runtime-fail ] || exit 9 ;;\n"
        "    4)\n"
        "      [ \"$mode\" != postflight-fail ] || exit 9\n"
        "      [ \"$mode\" != postflight-missing-output ] || exit 0\n"
        "      [ \"$mode\" != preexisting-telemetry ] || printf 'owner data' > \"$3\"\n"
        "      printf '{}\\n' > \"$3\"\n"
        "      ;;\n"
        "    3) [ \"$mode\" != strict-reload-fail ] || exit 9 ;;\n"
        "    *) exit 29 ;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        "case \"${1:-}\" in\n"
        "  *smoke_joint_position_fixed.py) [ \"$mode\" != live-smoke-fail ] || exit 11 ;;\n"
        "  *smoke_cat_soft.py) [ \"$mode\" != cat-smoke-fail ] || exit 12 ;;\n"
        "  *scripts/train.py)\n"
        "    [ \"$mode\" != train-fail ] || exit 13\n"
        "    case \"$mode\" in\n"
        "      no-run-dir) ;;\n"
        "      multiple-run-dirs) mkdir -p \"$PWD/logs/rsl_rl/z1_hammer/a_$RUN_NAME\" \"$PWD/logs/rsl_rl/z1_hammer/b_$RUN_NAME\" ;;\n"
        "      run-path-file) mkdir -p \"$PWD/logs/rsl_rl/z1_hammer\"; printf x > \"$PWD/logs/rsl_rl/z1_hammer/a_$RUN_NAME\" ;;\n"
        "      *)\n"
        "        dir=\"$PWD/logs/rsl_rl/z1_hammer/fixture_$RUN_NAME\"\n"
        "        mkdir -p \"$dir\"\n"
        "        for iteration in 0 50 100 150 200 250 300 350 400 450 499; do printf 'checkpoint-%s\\n' \"$iteration\" > \"$dir/model_${iteration}.pt\"; done\n"
        "        [ \"$mode\" != missing-checkpoint ] || rm \"$dir/model_250.pt\"\n"
        "        [ \"$mode\" != extra-checkpoint ] || printf extra > \"$dir/model_500.pt\"\n"
        "        [ \"$mode\" != preexisting-telemetry ] || printf 'owner data' > \"$PWD/victt_seed2_telemetry.json\"\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$mode\" = mutate-code-postflight ]; then printf dirty > \"$RUN_ROOT/postflight-drift.txt\"; fi\n"
        "if [ \"$mode\" = mutate-assets-postflight ]; then printf dirty > \"$ASSET_REPO/postflight-drift.txt\"; fi\n"
        "if [ \"$mode\" = mutate-code-head ]; then printf drift >> \"$RUN_ROOT/marker.txt\"; git -C \"$RUN_ROOT\" add marker.txt; git -C \"$RUN_ROOT\" -c user.email=test@example.com -c user.name=test commit -qm drift; fi\n"
        "exit 0\n"
    )
    py.chmod(0o755)
    env["FAKE_PY_MODE"] = mode


def _prepared_env(tmp_path: Path, *, mode: str = "success") -> dict[str, str]:
    env = _base_env(tmp_path)
    _install_fake_python(env, mode=mode)
    return env


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _calls(env: dict[str, str]) -> list[list[str]]:
    path = Path(env["HOME"]) / "python_calls.log"
    if not path.exists():
        return []
    return [line.split("\t")[1:] for line in path.read_text().splitlines()]


def _attempt(env: dict[str, str]) -> Path:
    return (
        Path(env["HOME"])
        / "campaigns/z1-vic-prototype/runs"
        / env["EXPECTED_CODE_REVISION"]
        / env["EXPECTED_ASSET_REVISION"]
        / "12345_victt_seed2"
    )


def _install_failing_git_status(
    tmp_path: Path, env: dict[str, str], *, repository: Path
) -> None:
    real_git = shutil.which("git")
    assert real_git is not None
    fake_bin = tmp_path / "fake-git-bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = -C ] && [ \"${2:-}\" = \"$FAKE_GIT_STATUS_FAIL_PATH\" ] && [ \"${3:-}\" = status ]; then exit 42; fi\n"
        "exec \"$REAL_GIT\" \"$@\"\n"
    )
    git.chmod(0o755)
    env["REAL_GIT"] = real_git
    env["FAKE_GIT_STATUS_FAIL_PATH"] = str(repository.resolve())
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"


def _embedded_python_source(invocation: str) -> str:
    source = LAUNCHER.read_text()
    start = source.index(invocation) + len(invocation)
    end = source.index("\nPY\n", start)
    return source[start:end] + "\n"


def _runtime_source() -> str:
    return _embedded_python_source(
        '"$PY" - <<\'PY\' || fail "CUDA identity or runtime package versions do not match"\n'
    )


def _postflight_source() -> str:
    return _embedded_python_source(
        '"$PY" - "$RUN_DIR" "$TELEMETRY_JSON" "$TASK" <<\'PY\' \\\n'
        '  || fail "checkpoint or TensorBoard telemetry validation failed"\n'
    )


def _strict_reload_source() -> str:
    return _embedded_python_source(
        '"$PY" - "$TASK" "$FINAL_CHECKPOINT" <<\'PY\' \\\n'
        '  || fail "model_499.pt failed strict VIC runner reload"\n'
    )


def _valid_telemetry() -> dict[str, object]:
    summary = {
        "mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "std": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "minimum": [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        "p05": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        "median": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "p95": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "maximum": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "lower_bound_occupancy": [0.0] * 6,
        "upper_bound_occupancy": [0.0] * 6,
    }
    return {
        "schema_version": 1,
        "source": "cat_rollout_storage",
        "action_terms": ["joint_position", "joint_stiffness"],
        "joint_names": [f"joint{index}" for index in range(1, 7)],
        "gain_action_indices": [6, 7, 8, 9, 10, 11],
        "raw_action_clip": 1.0,
        "rollout_steps_per_env": 24,
        "sample_count": 24 * 4096,
        "temporal_provenance": {
            "rollout_generated_by": "pre_update_behavior_policy",
            "checkpoint_weights": "post_update",
        },
        "deterministic_gaussian_mean": {
            "raw": dict(summary), "clipped": dict(summary),
        },
        "sampled_action": {"raw": dict(summary), "clipped": dict(summary)},
        "gaussian_exploration_std": [0.25] * 6,
    }


def _valid_events() -> dict[str, list[dict[str, object]]]:
    curriculum = []
    for step in range(500):
        if step < 50:
            value = 0.1
        elif step < 100:
            value = 0.08
        elif step < 150:
            value = 0.06
        elif step < 200:
            value = 0.04
        elif step < 250:
            value = 0.02
        else:
            value = 0.0
        curriculum.append({"step": step, "value": value})
    return {
        "Curriculum/r_imit_anneal/weight": curriculum,
        "Episode_Metrics/impossible_success": [
            {"step": step, "value": 0.0} for step in range(500)
        ],
        "Loss/value_function": [{"step": 0, "value": 1.0}],
    }


def _install_postflight_stubs(tmp_path: Path) -> Path:
    root = tmp_path / "postflight-stubs"
    package = root / "tensorboard/backend/event_processing"
    package.mkdir(parents=True)
    for init in (
        root / "tensorboard/__init__.py",
        root / "tensorboard/backend/__init__.py",
        package / "__init__.py",
    ):
        init.write_text("")
    (root / "torch.py").write_text(
        "import json, os, re\n"
        "class Tensor:\n"
        "    def __init__(self, finite=True): self.finite = finite\n"
        "class Result:\n"
        "    def __init__(self, value): self.value = value\n"
        "    def all(self): return self.value\n"
        "def isfinite(value): return Result(value.finite)\n"
        "def load(path, map_location, weights_only):\n"
        "    mode = os.environ.get('FAKE_POST_MODE', 'success')\n"
        "    iteration = int(re.search(r'model_(\\d+)\\.pt$', str(path)).group(1))\n"
        "    state = {'iter': iteration, 'model_state_dict': {'weight': Tensor(mode != 'nonfinite')}}\n"
        "    if mode == 'tensorless': state = {'iter': iteration}\n"
        "    if mode == 'bad-iteration' and iteration == 499: state['iter'] = 498\n"
        "    missing = ((mode == 'missing-telemetry-0' and iteration == 0) or (mode == 'missing-telemetry-499' and iteration == 499))\n"
        "    if iteration in (0, 499) and not missing:\n"
        "        state['infos'] = {'vic_rollout_telemetry': json.loads(os.environ['FAKE_TELEMETRY'])}\n"
        "    return state\n"
    )
    (package / "event_accumulator.py").write_text(
        "import json, os\n"
        "from types import SimpleNamespace\n"
        "class EventAccumulator:\n"
        "    def __init__(self, path, size_guidance):\n"
        "        if size_guidance != {'scalars': 0}: raise RuntimeError('bad size guidance')\n"
        "        self.events = json.loads(os.environ['FAKE_EVENTS'])\n"
        "    def Reload(self):\n"
        "        if os.environ.get('FAKE_POST_MODE') == 'corrupt-events': raise RuntimeError('corrupt events')\n"
        "        return self\n"
        "    def Tags(self): return {'scalars': list(self.events)}\n"
        "    def Scalars(self, tag): return [SimpleNamespace(**row) for row in self.events[tag]]\n"
    )
    return root


def _run_postflight(
    tmp_path: Path,
    *,
    telemetry: dict[str, object] | None = None,
    events: dict[str, list[dict[str, object]]] | None = None,
    mode: str = "success",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for iteration in ITERATIONS:
        (run_dir / f"model_{iteration}.pt").write_text("fixture")
    output = tmp_path / "telemetry.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_install_postflight_stubs(tmp_path))
    env["FAKE_TELEMETRY"] = json.dumps(telemetry or _valid_telemetry())
    env["FAKE_EVENTS"] = json.dumps(events or _valid_events())
    env["FAKE_POST_MODE"] = mode
    result = subprocess.run(
        [sys.executable, "-O", "-", str(run_dir), str(output), TASK],
        input=_postflight_source(),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, output


def _install_runtime_stubs(
    tmp_path: Path, *, cuda=True, count=1, name="NVIDIA A100-SXM4-40GB", mjlab="1.4.0"
) -> Path:
    root = tmp_path / "runtime-stubs"
    root.mkdir()
    (root / "torch.py").write_text(
        "class Cuda:\n"
        f"    def is_available(self): return {cuda!r}\n"
        f"    def device_count(self): return {count!r}\n"
        f"    def get_device_name(self, index): return {name!r}\n"
        "cuda = Cuda()\n"
    )
    for package, version in (("mjlab", mjlab), ("mujoco", "3.8.1"), ("mujoco-warp", "3.8.1")):
        metadata = root / f"{package.replace('-', '_')}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
        )
    return root


def _install_strict_reload_stubs(tmp_path: Path, *, mode: str) -> Path:
    root = tmp_path / "reload-stubs"
    for package in (
        "mjlab", "mjlab/envs", "mjlab/rl", "mjlab/tasks", "src", "src/tasks",
        "src/tasks/hammer", "src/tasks/hammer/config", "src/tasks/hammer/config/z1",
    ):
        current = root / package
        current.mkdir(parents=True, exist_ok=True)
        (current / "__init__.py").write_text("")
    (root / "mjlab/envs/__init__.py").write_text(
        "class ManagerBasedRlEnv:\n"
        "    def __init__(self, **kwargs): pass\n"
    )
    (root / "mjlab/rl/vecenv_wrapper.py").write_text(
        "class RslRlVecEnvWrapper:\n"
        "    def __init__(self, env, clip_actions): pass\n"
        "    def close(self): pass\n"
    )
    (root / "mjlab/tasks/registry.py").write_text(
        "import os\n"
        "from dataclasses import dataclass\n"
        "from types import SimpleNamespace\n"
        "def load_env_cfg(task, play=False): return SimpleNamespace(scene=SimpleNamespace(num_envs=0))\n"
        "def load_rl_cfg(task):\n"
        "    @dataclass\n"
        "    class C:\n"
        "        clip_actions: float = 1.0\n"
        "    return C()\n"
        "class Runner:\n"
        "    def __init__(self, *args, **kwargs): pass\n"
        "    def load(self, checkpoint, load_cfg, strict, map_location):\n"
        "        if load_cfg is not None: raise RuntimeError('checkpoint reload was not full')\n"
        "        if strict is not True or map_location != 'cuda:0': raise RuntimeError('strict reload arguments drifted')\n"
        "        if os.environ['FAKE_RELOAD_MODE'] == 'malformed-critic': raise RuntimeError('critic state mismatch')\n"
        "        if os.environ['FAKE_RELOAD_MODE'] == 'malformed-optimizer': raise RuntimeError('optimizer state mismatch')\n"
        "def load_runner_cls(task): return Runner\n"
    )
    return root


def test_launcher_is_a_single_non_array_a100_job() -> None:
    source = LAUNCHER.read_text()

    assert source.startswith("#!/usr/bin/env bash\n")
    assert "#SBATCH --job-name=z1-victt-seed2" in source
    assert "#SBATCH --array" not in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --time=01:30:00" in source
    assert "#SBATCH --no-requeue" in source
    assert (
        "#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/"
        "z1-vic-prototype/slurm/z1-victt-seed2-%j.out"
    ) in source
    assert (
        "#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/"
        "z1-vic-prototype/slurm/z1-victt-seed2-%j.err"
    ) in source


def test_launcher_runs_exact_smokes_training_and_hashed_postflight(
    tmp_path: Path,
) -> None:
    env = _prepared_env(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stdout + result.stderr
    attempt = _attempt(env)
    run_name = "victt_seed2_job12345"
    run_dir = attempt / "logs/rsl_rl/z1_hammer" / f"fixture_{run_name}"
    telemetry = attempt / "victt_seed2_telemetry.json"
    calls = _calls(env)
    assert calls == [
        ["-"],
        [
            str(Path(env["RUN_ROOT"]) / "scripts/smoke_joint_position_fixed.py"),
            "--task", TASK, "--device", "cuda:0",
        ],
        [
            str(Path(env["RUN_ROOT"]) / "scripts/smoke_cat_soft.py"),
            "--task", TASK, "--device", "cuda:0", "--iters", "1",
        ],
        [
            str(Path(env["RUN_ROOT"]) / "scripts/train.py"), TASK,
            "--gpu-ids", "[0]", "--agent.logger", "tensorboard",
            "--agent.run-name", run_name, "--agent.seed", "2",
            "--agent.max-iterations", "500", "--agent.save-interval", "50",
            "--env.scene.num-envs", "4096",
        ],
        ["-", str(run_dir), str(telemetry), TASK],
        ["-", TASK, str(run_dir / "model_499.pt")],
    ]
    assert telemetry.read_text() == "{}\n"
    for iteration in ITERATIONS:
        checkpoint = run_dir / f"model_{iteration}.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert (
            f"Z1_VIC_CANARY_CHECKPOINT_SHA256 basename=model_{iteration}.pt "
            f"sha256={digest}"
        ) in result.stdout
    final_digest = hashlib.sha256((run_dir / "model_499.pt").read_bytes()).hexdigest()
    assert f"Z1_VIC_CANARY_FINAL_CHECKPOINT_SHA256={final_digest}" in result.stdout
    assert f"Z1_VIC_CANARY_CODE_REVISION={env['EXPECTED_CODE_REVISION']}" in result.stdout
    assert f"Z1_VIC_CANARY_ASSET_REVISION={env['EXPECTED_ASSET_REVISION']}" in result.stdout
    assert f"Z1_VIC_CANARY_TELEMETRY={telemetry}" in result.stdout
    assert "VIC_CANARY_DONE seed=2 task=" + TASK in result.stdout


def test_launcher_contains_only_frozen_operational_training_overrides() -> None:
    source = LAUNCHER.read_text()

    for prohibited in (
        "--env.rewards", "--env.actions", "--env.events", "--env.terminations",
        "--env.curriculum", "--env.randomization", "--env.metrics", "--resume",
    ):
        assert prohibited not in source
    assert 'smoke_joint_position_fixed.py" \\\n+  --task "$TASK" --device cuda:0'.replace("\n+", "\n") in source
    assert 'smoke_cat_soft.py" \\\n+  --task "$TASK" --device cuda:0 --iters 1'.replace("\n+", "\n") in source


def test_launcher_rejects_cli_override_before_python(tmp_path: Path) -> None:
    env = _prepared_env(tmp_path)
    result = _run(env, "--agent.seed=9")
    assert result.returncode == 2
    assert "launcher accepts no arguments" in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize("value", ("1", "0", " "))
def test_launcher_rejects_python_optimization_before_python(
    tmp_path: Path, value: str
) -> None:
    env = _prepared_env(tmp_path)
    env["PYTHONOPTIMIZE"] = value
    result = _run(env)
    assert result.returncode == 2
    assert "PYTHONOPTIMIZE must be empty" in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize(
    "name",
    (
        "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN", "SLURM_ARRAY_TASK_MAX", "SLURM_ARRAY_TASK_STEP",
    ),
)
def test_launcher_rejects_every_array_context_before_python(
    tmp_path: Path, name: str
) -> None:
    env = _prepared_env(tmp_path)
    env[name] = "1"
    result = _run(env)
    assert result.returncode == 2
    assert f"array context is forbidden: {name}" in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize("value", (None, "", "abc", "123_4", "-2"))
def test_launcher_requires_numeric_job_id(tmp_path: Path, value: str | None) -> None:
    env = _prepared_env(tmp_path)
    if value is None:
        env.pop("SLURM_JOB_ID")
    else:
        env["SLURM_JOB_ID"] = value
    result = _run(env)
    assert result.returncode == 2
    assert "SLURM_JOB_ID must be numeric" in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("EXPECTED_CODE_REVISION", "a" * 39),
        ("EXPECTED_CODE_REVISION", "G" * 40),
        ("EXPECTED_ASSET_REVISION", "b" * 39),
        ("EXPECTED_ASSET_REVISION", "z" * 40),
    ),
)
def test_launcher_rejects_malformed_revision(
    tmp_path: Path, name: str, value: str
) -> None:
    env = _prepared_env(tmp_path)
    env[name] = value
    result = _run(env)
    assert result.returncode == 2
    assert f"{name} must be a 40-hex revision" in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize(
    ("name", "message"),
    (
        ("EXPECTED_CODE_REVISION", "EXPECTED_CODE_REVISION must be a 40-hex revision"),
        ("EXPECTED_ASSET_REVISION", "EXPECTED_ASSET_REVISION must be a 40-hex revision"),
        ("RUN_ROOT", "RUN_ROOT is required"),
        ("ASSET_REPO", "ASSET_REPO is required"),
    ),
)
def test_launcher_rejects_missing_required_input_before_python(
    tmp_path: Path, name: str, message: str
) -> None:
    env = _prepared_env(tmp_path)
    env.pop(name)
    result = _run(env)
    assert result.returncode == 2
    assert message in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize(
    ("name", "message"),
    (("RUN_ROOT", "code repository is unavailable"), ("ASSET_REPO", "asset repository is unavailable")),
)
def test_launcher_rejects_unavailable_repository_root_before_python(
    tmp_path: Path, name: str, message: str
) -> None:
    env = _prepared_env(tmp_path)
    env[name] = str(tmp_path / "does-not-exist")
    result = _run(env)
    assert result.returncode == 2
    assert message in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize(
    ("repo_var", "message"),
    (("RUN_ROOT", "code repository must be clean"), ("ASSET_REPO", "asset repository must be clean")),
)
def test_launcher_rejects_dirty_repository(
    tmp_path: Path, repo_var: str, message: str
) -> None:
    env = _prepared_env(tmp_path)
    (Path(env[repo_var]) / "owner.txt").write_text("untracked")
    result = _run(env)
    assert result.returncode == 2
    assert message in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize(
    ("repo_var", "label"), (("RUN_ROOT", "code"), ("ASSET_REPO", "asset"))
)
def test_launcher_rejects_git_status_failure(
    tmp_path: Path, repo_var: str, label: str
) -> None:
    env = _prepared_env(tmp_path)
    _install_failing_git_status(tmp_path, env, repository=Path(env[repo_var]))
    result = _run(env)
    assert result.returncode == 2
    assert f"{label} repository status check failed" in result.stdout
    assert _calls(env) == []


@pytest.mark.parametrize(
    ("repo_var", "revision_var", "message"),
    (
        ("RUN_ROOT", "EXPECTED_CODE_REVISION", "code revision mismatch"),
        ("ASSET_REPO", "EXPECTED_ASSET_REVISION", "asset revision mismatch"),
    ),
)
def test_launcher_rejects_clean_wrong_revision(
    tmp_path: Path, repo_var: str, revision_var: str, message: str
) -> None:
    env = _prepared_env(tmp_path)
    env[revision_var] = "c" * 40
    result = _run(env)
    assert result.returncode == 2
    assert message in result.stdout
    assert _calls(env) == []


def test_launcher_rejects_noncanonical_asset_sibling(tmp_path: Path) -> None:
    env = _prepared_env(tmp_path)
    other = tmp_path / "other-assets"
    env["ASSET_REPO"] = str(other)
    env["EXPECTED_ASSET_REVISION"] = _git_repo(other, "other")
    result = _run(env)
    assert result.returncode == 2
    assert "ASSET_REPO must be the canonical sibling" in result.stdout
    assert _calls(env) == []


def test_launcher_never_reuses_attempt_leaf(tmp_path: Path) -> None:
    env = _prepared_env(tmp_path)
    attempt = _attempt(env)
    attempt.mkdir(parents=True)
    sentinel = attempt / "owner-data"
    sentinel.write_text("preserve")
    result = _run(env)
    assert result.returncode == 2
    assert "attempt directory already exists" in result.stdout
    assert sentinel.read_text() == "preserve"
    assert _calls(env) == []


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("runtime-fail", "CUDA identity or runtime package versions do not match"),
        ("live-smoke-fail", ""), ("cat-smoke-fail", ""), ("train-fail", ""),
        ("no-run-dir", "expected exactly one matching training run directory"),
        ("multiple-run-dirs", "expected exactly one matching training run directory"),
        ("run-path-file", "expected exactly one matching training run directory"),
        ("missing-checkpoint", "checkpoint set does not match the exact canary schedule"),
        ("extra-checkpoint", "checkpoint set does not match the exact canary schedule"),
        ("postflight-fail", "checkpoint or TensorBoard telemetry validation failed"),
        ("postflight-missing-output", "canonical telemetry extractor did not create a non-empty JSON result"),
        ("strict-reload-fail", "model_499.pt failed strict VIC runner reload"),
    ),
)
def test_launcher_stops_on_failed_qualification_or_postflight(
    tmp_path: Path, mode: str, message: str
) -> None:
    env = _prepared_env(tmp_path, mode=mode)
    result = _run(env)
    assert result.returncode != 0
    if message:
        assert message in result.stdout
    assert "VIC_CANARY_DONE" not in result.stdout
    calls = _calls(env)
    if mode == "live-smoke-fail":
        assert not any(call and call[0].endswith("scripts/train.py") for call in calls)
    if mode == "cat-smoke-fail":
        assert not any(call and call[0].endswith("scripts/train.py") for call in calls)


def test_launcher_preserves_preexisting_canonical_telemetry(tmp_path: Path) -> None:
    env = _prepared_env(tmp_path, mode="preexisting-telemetry")
    result = _run(env)
    telemetry = _attempt(env) / "victt_seed2_telemetry.json"
    assert result.returncode == 2
    assert "canonical telemetry output already exists" in result.stdout
    assert telemetry.read_text() == "owner data"
    assert "VIC_CANARY_DONE" not in result.stdout


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("mutate-code-postflight", "postflight code repository must be clean"),
        ("mutate-assets-postflight", "postflight asset repository must be clean"),
        ("mutate-code-head", "postflight code revision mismatch"),
    ),
)
def test_launcher_rechecks_repository_cleanliness_before_done(
    tmp_path: Path, mode: str, message: str
) -> None:
    env = _prepared_env(tmp_path, mode=mode)
    result = _run(env)
    assert result.returncode == 2
    assert message in result.stdout
    assert "VIC_CANARY_DONE" not in result.stdout


def test_launcher_hashes_every_checkpoint_and_checks_every_digest() -> None:
    source = LAUNCHER.read_text()
    assert 'for CHECKPOINT in "${EXPECTED_CHECKPOINTS[@]}"; do' in source
    assert 'sha256sum "$CHECKPOINT"' in source
    assert '[[ "$CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]]' in source
    assert '[[ "$TELEMETRY_SHA256" =~ ^[0-9a-f]{64}$ ]]' in source
    assert source.index("FINAL_CODE_REVISION=") > source.index("CHECKPOINT_SHA256=")
    assert source.index("VIC_CANARY_DONE") > source.index("FINAL_ASSET_STATUS=")


@pytest.mark.parametrize("basename", ("model_250.pt", "model_499.pt"))
def test_launcher_rejects_invalid_hash_output_before_done(
    tmp_path: Path, basename: str
) -> None:
    env = _prepared_env(tmp_path)
    real_sha = shutil.which("sha256sum")
    assert real_sha is not None
    fake_bin = tmp_path / "fake-sha-bin"
    fake_bin.mkdir()
    fake_sha = fake_bin / "sha256sum"
    fake_sha.write_text(
        "#!/bin/sh\n"
        "case \"${1:-}\" in *\"$INVALID_HASH_BASENAME\") printf 'invalid  %s\\n' \"$1\" ;; "
        "*) exec \"$REAL_SHA256SUM\" \"$@\" ;; esac\n"
    )
    fake_sha.chmod(0o755)
    env["REAL_SHA256SUM"] = real_sha
    env["INVALID_HASH_BASENAME"] = basename
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    result = _run(env)
    assert result.returncode == 2
    assert "invalid checkpoint SHA-256 result" in result.stdout
    assert "Z1_VIC_CANARY_FINAL_CHECKPOINT_SHA256=" not in result.stdout
    assert "VIC_CANARY_DONE" not in result.stdout


@pytest.mark.parametrize("name", ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
def test_launcher_rejects_distributed_context_before_python(
    tmp_path: Path, name: str
) -> None:
    env = _prepared_env(tmp_path)
    env[name] = "0"
    result = _run(env)
    assert result.returncode == 2
    assert f"distributed context is forbidden: {name}" in result.stdout
    assert _calls(env) == []


def test_launcher_owns_temporary_files_and_cleans_scratch_only_on_success(
    tmp_path: Path,
) -> None:
    success = _prepared_env(tmp_path / "success")
    result = _run(success)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (_attempt(success) / "scratch").exists()

    failed = _prepared_env(tmp_path / "failed", mode="cat-smoke-fail")
    result = _run(failed)
    assert result.returncode != 0
    assert (_attempt(failed) / "scratch").is_dir()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"cuda": False}, "CUDA unavailable"),
        ({"count": 2}, "CUDA device count mismatch"),
        ({"name": "NVIDIA H100 80GB HBM3"}, "unexpected GPU"),
        ({"mjlab": "1.4.1"}, "runtime package versions do not match"),
    ),
)
def test_runtime_guard_rejects_invalid_identity_under_optimization(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_install_runtime_stubs(tmp_path, **kwargs))
    result = subprocess.run(
        [sys.executable, "-O", "-"], input=_runtime_source(), env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_postflight_accepts_exact_24_step_4096_world_record_under_optimization(
    tmp_path: Path,
) -> None:
    result, output = _run_postflight(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = __import__("json").loads(output.read_text())
    assert payload["checkpoint_iterations"] == list(ITERATIONS)
    assert payload["final_gain_classification"] == {
        "all_zero": False,
        "nonzero_spread": True,
        "bound_pinned_joints": [],
    }
    assert payload["r_imit_curriculum"]["observed"][499] == {
        "iteration": 499, "value": 0.0,
    }


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("tensorless", "contains no tensors"),
        ("nonfinite", "contains a non-finite tensor"),
        ("bad-iteration", "iteration identity drifted"),
        ("missing-telemetry-0", "model_0.pt infos are missing"),
        ("missing-telemetry-499", "model_499.pt infos are missing"),
        ("corrupt-events", "corrupt events"),
    ),
)
def test_postflight_rejects_bad_checkpoint_or_event_state_under_optimization(
    tmp_path: Path, mode: str, message: str
) -> None:
    result, output = _run_postflight(tmp_path, mode=mode)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


def _replace_telemetry(path: str, value: object) -> dict[str, object]:
    telemetry = _valid_telemetry()
    cursor: object = telemetry
    parts = path.split(".")
    for part in parts[:-1]:
        assert isinstance(cursor, dict)
        cursor = cursor[part]
    assert isinstance(cursor, dict)
    cursor[parts[-1]] = value
    return telemetry


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        ("rollout_steps_per_env", 23, "rollout lifecycle drifted"),
        ("sample_count", 24, "sample count drifted"),
        ("temporal_provenance", {}, "temporal provenance drifted"),
        ("gaussian_exploration_std", [0.0] * 6, "must be positive"),
        ("deterministic_gaussian_mean.clipped.std", [-0.1] * 6, "std must be nonnegative"),
        ("deterministic_gaussian_mean.clipped.p05", [0.9] * 6, "summary order drifted"),
        ("deterministic_gaussian_mean.clipped.mean", [1.1] * 6, "above its upper bound"),
        ("deterministic_gaussian_mean.clipped.std", [0.0] * 6, "has no spread"),
        ("deterministic_gaussian_mean.clipped.lower_bound_occupancy", [1.0] + [0.0] * 5, "bound pinned"),
    ),
)
def test_postflight_rejects_malformed_or_trivial_gain_telemetry_under_optimization(
    tmp_path: Path, path: str, value: object, message: str
) -> None:
    telemetry = _replace_telemetry(path, value)
    result, output = _run_postflight(tmp_path, telemetry=telemetry)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        ("missing-curriculum", "missing TensorBoard scalar"),
        ("missing-impossible", "missing TensorBoard scalar"),
        ("nonfinite-unrelated", "non-finite TensorBoard scalar"),
        ("early-collapse", "curriculum timing drifted"),
        ("missing-anchor", "curriculum timing drifted"),
        ("nonzero-final", "curriculum"),
        ("impossible", "impossible_success is nonzero"),
        ("empty-stream", "empty TensorBoard scalar stream"),
    ),
)
def test_postflight_rejects_bad_tensorboard_semantics_under_optimization(
    tmp_path: Path, mutate: str, message: str
) -> None:
    events = _valid_events()
    if mutate == "missing-curriculum":
        events.pop("Curriculum/r_imit_anneal/weight")
    elif mutate == "missing-impossible":
        events.pop("Episode_Metrics/impossible_success")
    elif mutate == "nonfinite-unrelated":
        events["Loss/value_function"][0]["value"] = float("nan")
    elif mutate == "early-collapse":
        for row in events["Curriculum/r_imit_anneal/weight"][25:50]:
            row["value"] = 0.08
    elif mutate == "missing-anchor":
        events["Curriculum/r_imit_anneal/weight"] = [
            row for row in events["Curriculum/r_imit_anneal/weight"]
            if row["step"] != 100
        ]
    elif mutate == "nonzero-final":
        events["Curriculum/r_imit_anneal/weight"][-1]["value"] = 0.02
    elif mutate == "impossible":
        events["Episode_Metrics/impossible_success"][0]["value"] = 1.0
    elif mutate == "empty-stream":
        events["Loss/value_function"] = []
    result, output = _run_postflight(tmp_path, events=events)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    (("malformed-critic", "critic state mismatch"), ("malformed-optimizer", "optimizer state mismatch")),
)
def test_full_strict_reload_rejects_malformed_training_state_under_optimization(
    tmp_path: Path, mode: str, message: str
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_install_strict_reload_stubs(tmp_path, mode=mode))
    env["FAKE_RELOAD_MODE"] = mode
    checkpoint = tmp_path / "model_499.pt"
    checkpoint.write_text("fixture")
    result = subprocess.run(
        [sys.executable, "-O", "-", TASK, str(checkpoint)],
        input=_strict_reload_source(), env=env,
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_strict_reload_uses_full_checkpoint_contract() -> None:
    source = _strict_reload_source()
    assert (
        'runner.load(checkpoint, load_cfg=None, strict=True, map_location="cuda:0")'
        in source
    )


def test_embedded_production_guards_never_use_assert() -> None:
    assert all(
        re.search(r"(?m)^\s*assert\b", source) is None
        for source in (_runtime_source(), _postflight_source(), _strict_reload_source())
    )
