"""Public shell contracts for the direct-reference FIC Vega launchers."""

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
LAUNCHER = REPO_ROOT / "scripts/slurm/vega_fic_direct_reference.sbatch"
SMOKE_LAUNCHER = (
    REPO_ROOT / "scripts/slurm/vega_fic_direct_reference_smoke.sbatch"
)
FIC0_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-Fixed"
)


def _base_env(
    tmp_path: Path, *, array_index: str = "0", shape: str = "canary"
) -> dict[str, str]:
    shapes = {
        "canary": {"COUNT": "2", "MIN": "0", "MAX": "1", "STEP": "1"},
        "expansion": {"COUNT": "4", "MIN": "2", "MAX": "5", "STEP": "1"},
    }
    values = shapes[shape]
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "RUN_ROOT": str(tmp_path / "repos/code"),
        "ASSET_REPO": str(tmp_path / "repos/safe_impact_manipulation"),
        "EXPECTED_CODE_REVISION": "a" * 40,
        "EXPECTED_ASSET_REVISION": "b" * 40,
        "SLURM_ARRAY_TASK_ID": array_index,
        "SLURM_ARRAY_JOB_ID": "12345",
        "SLURM_ARRAY_TASK_COUNT": values["COUNT"],
        "SLURM_ARRAY_TASK_MIN": values["MIN"],
        "SLURM_ARRAY_TASK_MAX": values["MAX"],
        "SLURM_ARRAY_TASK_STEP": values["STEP"],
        "SLURM_JOB_ID": f"12345_{array_index}",
    }


def _run(
    env: dict[str, str],
    launcher: Path = LAUNCHER,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(launcher), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=fic-direct-test@example.com",
            "-c",
            "user.name=FIC Direct Test",
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


def _clean_repo_env(
    tmp_path: Path, *, array_index: str = "0", shape: str = "canary"
) -> dict[str, str]:
    env = _base_env(tmp_path, array_index=array_index, shape=shape)
    code = Path(env["RUN_ROOT"])
    assets = Path(env["ASSET_REPO"])
    env["EXPECTED_CODE_REVISION"] = _git_repo(code, "code")
    env["EXPECTED_ASSET_REVISION"] = _git_repo(assets, "assets")
    return env


def _install_fake_python(env: dict[str, str]) -> None:
    py = Path(env["HOME"]) / "repos/unitree_rl_mjlab/.venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(
        "#!/bin/sh\n"
        "{ printf 'CALL'; for arg in \"$@\"; do printf '\\t%s' \"$arg\"; done; "
        "printf '\\n'; } >> \"$HOME/python_calls.log\"\n"
        "mode=${FAKE_PY_MODE:-success}\n"
        "if [ \"${1:-}\" = - ]; then\n"
        "  case \"$#\" in\n"
        "    1)\n"
        "      [ \"$mode\" != runtime-fail ] || exit 9\n"
        "      printf 'Z1_DIRECT_CUDA_GPU=NVIDIA A100-SXM4-40GB\\n'\n"
        "      printf 'Z1_DIRECT_RUNTIME=mjlab=1.4.0 mujoco=3.8.1 "
        "mujoco-warp=3.8.1\\n'\n"
        "      ;;\n"
        "    2) [ \"$mode\" != checkpoint-fail ] || exit 9 ;;\n"
        "    6)\n"
        "      [ \"$mode\" != curriculum-fail ] || exit 9\n"
        "      [ \"$mode\" != curriculum-missing-output ] || exit 0\n"
        "      printf '%b' \"{\\\"arm\\\":\\\"$5\\\","
        "\\\"expected_control_step_stages\\\":[[0,0.1],[1200,0.08],"
        "[2400,0.06],[3600,0.04],[4800,0.02],[6000,0.0]],"
        "\\\"final_weight\\\":0.0,\\\"observed\\\":["
        "{\\\"iteration\\\":0,\\\"weight\\\":0.1},"
        "{\\\"iteration\\\":1,\\\"weight\\\":0.09},"
        "{\\\"iteration\\\":2,\\\"weight\\\":0.08},"
        "{\\\"iteration\\\":3,\\\"weight\\\":0.06},"
        "{\\\"iteration\\\":4,\\\"weight\\\":0.04},"
        "{\\\"iteration\\\":5,\\\"weight\\\":0.02},"
        "{\\\"iteration\\\":6,\\\"weight\\\":0.0}],"
        "\\\"scalar\\\":\\\"Curriculum/r_imit_anneal/weight\\\","
        "\\\"schema_version\\\":1,\\\"task\\\":\\\"$4\\\","
        "\\\"training_seed\\\":$6}\\n\" > \"$3\"\n"
        "      ;;\n"
        "    *) exit 29 ;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        "case \"${1:-}\" in\n"
        "  *scripts/train.py)\n"
        "    case \"$mode\" in\n"
        "      no-run-dir) ;;\n"
        "      multiple-run-dirs)\n"
        "        for suffix in a b; do\n"
        "          dir=\"$PWD/logs/rsl_rl/z1_hammer/${suffix}_$RUN_NAME\"\n"
        "          mkdir -p \"$dir\"\n"
        "          printf 'fake checkpoint\\n' > \"$dir/model_499.pt\"\n"
        "        done\n"
        "        ;;\n"
        "      missing-checkpoint)\n"
        "        mkdir -p \"$PWD/logs/rsl_rl/z1_hammer/fixture_$RUN_NAME\"\n"
        "        ;;\n"
        "      run-path-file)\n"
        "        mkdir -p \"$PWD/logs/rsl_rl/z1_hammer\"\n"
        "        printf 'not a directory\\n' > "
        "\"$PWD/logs/rsl_rl/z1_hammer/fixture_$RUN_NAME\"\n"
        "        ;;\n"
        "      *)\n"
        "        dir=\"$PWD/logs/rsl_rl/z1_hammer/fixture_$RUN_NAME\"\n"
        "        mkdir -p \"$dir\"\n"
        "        printf 'fake checkpoint\\n' > \"$dir/model_499.pt\"\n"
        "        [ \"$mode\" != preexisting-evaluation ] || "
        "printf 'owner data' > \"$PWD/${FIC_SHORT}_seed${FIC_SEED}.json\"\n"
        "        [ \"$mode\" != preexisting-curriculum ] || "
        "printf 'owner data' > "
        "\"$PWD/${FIC_SHORT}_seed${FIC_SEED}_curriculum.json\"\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "  *evaluate_fic_pilot.py)\n"
        "    [ \"$mode\" != evaluator-fail ] || exit 17\n"
        "    [ \"$mode\" != evaluator-missing-output ] || exit 0\n"
        "    while [ $# -gt 0 ]; do\n"
        "      if [ \"$1\" = --output ]; then shift; "
        "printf '{}\\n' > \"$1\"; break; fi\n"
        "      shift\n"
        "    done\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n"
    )
    py.chmod(0o755)


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
        "if [ \"${1:-}\" = -C ] "
        "&& [ \"${2:-}\" = \"$FAKE_GIT_STATUS_FAIL_PATH\" ] "
        "&& [ \"${3:-}\" = status ]; then\n"
        "  printf 'simulated git status failure for %s\\n' \"$2\" >&2\n"
        "  exit 42\n"
        "fi\n"
        "exec \"$REAL_GIT\" \"$@\"\n"
    )
    git.chmod(0o755)
    env["REAL_GIT"] = real_git
    env["FAKE_GIT_STATUS_FAIL_PATH"] = str(repository.resolve())
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"


def _prepared_fake_env(
    tmp_path: Path,
    *,
    array_index: str = "0",
    shape: str = "canary",
    fake_mode: str = "success",
) -> dict[str, str]:
    env = _clean_repo_env(tmp_path, array_index=array_index, shape=shape)
    evaluator = Path(env["RUN_ROOT"]) / "evaluation/joint_position/evaluate_fic_pilot.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# evaluator fixture\n")
    _git(Path(env["RUN_ROOT"]), "add", str(evaluator.relative_to(env["RUN_ROOT"])))
    _git(Path(env["RUN_ROOT"]), "commit", "-q", "-m", "evaluator fixture")
    env["EXPECTED_CODE_REVISION"] = _git(
        Path(env["RUN_ROOT"]), "rev-parse", "HEAD"
    ).stdout.strip()
    env["FAKE_PY_MODE"] = fake_mode
    _install_fake_python(env)
    return env


def _calls(env: dict[str, str]) -> list[list[str]]:
    return [
        line.split("\t")[1:]
        for line in (Path(env["HOME"]) / "python_calls.log").read_text().splitlines()
    ]


def _embedded_python_source(launcher: Path, invocation: str) -> str:
    source = launcher.read_text()
    start = source.index(invocation) + len(invocation)
    end = source.index("\nPY\n", start)
    return source[start:end] + "\n"


def _runtime_guard_source(launcher: Path) -> str:
    return _embedded_python_source(
        launcher,
        '"$PY" - <<\'PY\' || fail '
        '"CUDA identity or runtime package versions do not match"\n',
    )


def _checkpoint_guard_source() -> str:
    return _embedded_python_source(
        LAUNCHER,
        '"$PY" - "$CHECKPOINT" <<\'PY\' \\\n'
        '  || fail "model_499.pt contains no finite checkpoint state"\n',
    )


def _curriculum_extractor_source() -> str:
    invocation = (
        '"$PY" - "$RUN_DIR" "$CURRICULUM_JSON" "$TASK" "$SHORT" "$SEED" '
        "<<'PY' \\\n  || fail \"curriculum telemetry validation failed\"\n"
    )
    return _embedded_python_source(LAUNCHER, invocation)


def _install_runtime_stubs(
    tmp_path: Path,
    *,
    cuda_available: bool = True,
    device_count: int = 1,
    gpu_name: str = "NVIDIA A100-SXM4-40GB",
    mjlab_version: str = "1.4.0",
) -> Path:
    root = tmp_path / "runtime-stubs"
    root.mkdir()
    (root / "torch.py").write_text(
        "class Cuda:\n"
        f"    @staticmethod\n    def is_available(): return {cuda_available!r}\n"
        f"    @staticmethod\n    def device_count(): return {device_count!r}\n"
        f"    @staticmethod\n    def get_device_name(index): return {gpu_name!r}\n"
        "cuda = Cuda()\n"
    )
    for package, version in (
        ("mjlab", mjlab_version),
        ("mujoco", "3.8.1"),
        ("mujoco-warp", "3.8.1"),
    ):
        normalized = package.replace("-", "_")
        metadata = root / f"{normalized}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
        )
    return root


def _install_checkpoint_torch_stub(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoint-stubs"
    root.mkdir()
    (root / "torch.py").write_text(
        "import os\n"
        "class Tensor:\n"
        "    def __init__(self, finite): self.finite = finite\n"
        "class FiniteResult:\n"
        "    def __init__(self, finite): self.finite = finite\n"
        "    def all(self): return self.finite\n"
        "def load(path, map_location, weights_only):\n"
        "    if os.environ['FAKE_CHECKPOINT_MODE'] == 'tensorless': return {}\n"
        "    return {'state': Tensor(False)}\n"
        "def isfinite(value): return FiniteResult(value.finite)\n"
    )
    return root


def _install_fake_tensorboard(tmp_path: Path) -> Path:
    root = tmp_path / "fake-packages"
    package = root / "tensorboard/backend/event_processing"
    package.mkdir(parents=True)
    for init in (
        root / "tensorboard/__init__.py",
        root / "tensorboard/backend/__init__.py",
        package / "__init__.py",
    ):
        init.write_text("")
    (package / "event_accumulator.py").write_text(
        "import json\n"
        "import os\n"
        "from types import SimpleNamespace\n"
        "\n"
        "class EventAccumulator:\n"
        "    def __init__(self, path, size_guidance):\n"
        "        assert size_guidance == {'scalars': 0}\n"
        "        self.rows = json.loads(os.environ.get('FAKE_EVENTS', '[]'))\n"
        "\n"
        "    def Reload(self):\n"
        "        return self\n"
        "\n"
        "    def Tags(self):\n"
        "        if os.environ.get('FAKE_EVENT_MODE') == 'missing-tag':\n"
        "            return {'scalars': []}\n"
        "        return {'scalars': ['Curriculum/r_imit_anneal/weight']}\n"
        "\n"
        "    def Scalars(self, tag):\n"
        "        assert tag == 'Curriculum/r_imit_anneal/weight'\n"
        "        return [SimpleNamespace(**row) for row in self.rows]\n"
    )
    return root


def _run_curriculum_extractor(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    mode: str = "success",
    preexisting: bool = False,
    optimize: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    output = tmp_path / "curriculum.json"
    if preexisting:
        output.write_text("owner data")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_install_fake_tensorboard(tmp_path))
    env["FAKE_EVENTS"] = json.dumps(rows)
    env["FAKE_EVENT_MODE"] = mode
    command = [sys.executable]
    if optimize:
        command.append("-O")
    command.extend(["-", str(run_dir), str(output), FIC0_TASK, "fic0", "2"])
    result = subprocess.run(
        command,
        input=_curriculum_extractor_source(),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, output


def test_production_launcher_has_collision_safe_scheduler_contract() -> None:
    source = LAUNCHER.read_text()

    assert source.startswith("#!/usr/bin/env bash\n")
    assert "#SBATCH --array=0-1" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --time=01:30:00" in source
    assert (
        "#SBATCH --output=/ceph/hpc/home/eunikhilr/"
        "z1-fic-direct-reference-%A_%a.out"
    ) in source
    assert (
        "#SBATCH --error=/ceph/hpc/home/eunikhilr/"
        "z1-fic-direct-reference-%A_%a.err"
    ) in source
    assert "set -euo pipefail" in source
    assert '^[0-9a-f]{40}$' in source
    assert (
        'z1-fic-direct-reference-500/$EXPECTED_CODE_REVISION/'
        '$EXPECTED_ASSET_REVISION'
    ) in source
    assert '"NVIDIA A100-SXM4-40GB"' in source
    assert "device_count = torch.cuda.device_count()" in source
    assert (
        'require(device_count == 1, f"CUDA device count mismatch: {device_count}")'
        in source
    )
    assert (
        'expected = {"mjlab": "1.4.0", "mujoco": "3.8.1", '
        '"mujoco-warp": "3.8.1"}'
    ) in source
    assert "model_499.pt" in source
    assert "torch.isfinite" in source
    assert "evaluate_fic_pilot.py" in source
    assert "--training-seed \"$SEED\"" in source
    assert 'EventAccumulator(str(run_dir), size_guidance={"scalars": 0})' in source
    assert 'scalar = "Curriculum/r_imit_anneal/weight"' in source
    assert "math.isfinite(weight)" in source
    assert "previous_weight >= weight - tolerance" in source
    assert "-tolerance <= weight <= 0.1 + tolerance" in source
    assert "for stage in stages:" in source
    assert '"observed": observed' in source
    assert '"final_weight": 0.0' in source
    assert 'output.open("x", encoding="utf-8")' in source
    assert 'sort_keys=True, separators=(",", ":"), allow_nan=False' in source
    assert "CHECKPOINT_SHA256" in source
    assert "EVALUATION_SHA256" in source
    assert "CURRICULUM_SHA256" in source
    for prohibited in (
        "--env.rewards",
        "--env.actions",
        "--env.events",
        "--env.terminations",
        "--env.curriculum",
        "--env.randomization",
        "--env.metrics",
        "--env.scene.robot.actuators",
    ):
        assert prohibited not in source


def test_smoke_launcher_has_single_attempt_scheduler_contract() -> None:
    source = SMOKE_LAUNCHER.read_text()

    assert source.startswith("#!/usr/bin/env bash\n")
    assert "#SBATCH --array" not in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --time=00:30:00" in source
    assert "set -euo pipefail" in source
    assert '^[0-9a-f]{40}$' in source
    assert (
        'z1-fic-direct-reference-smoke/$EXPECTED_CODE_REVISION/'
        '$EXPECTED_ASSET_REVISION'
    ) in source
    assert '"NVIDIA A100-SXM4-40GB"' in source
    assert "device_count = torch.cuda.device_count()" in source
    assert (
        'require(device_count == 1, f"CUDA device count mismatch: {device_count}")'
        in source
    )
    assert (
        'expected = {"mjlab": "1.4.0", "mujoco": "3.8.1", '
        '"mujoco-warp": "3.8.1"}'
    ) in source
    assert "scripts/train.py" not in source
    assert "evaluate_fic_pilot.py" not in source


@pytest.mark.parametrize(
    ("array_index", "shape", "task", "short", "seed"),
    (
        ("0", "canary", FIC0_TASK, "fic0", "2"),
        ("1", "canary", f"{FIC0_TASK}-TT", "fictt", "2"),
        ("2", "expansion", FIC0_TASK, "fic0", "3"),
        ("3", "expansion", f"{FIC0_TASK}-TT", "fictt", "3"),
        ("4", "expansion", FIC0_TASK, "fic0", "4"),
        ("5", "expansion", f"{FIC0_TASK}-TT", "fictt", "4"),
    ),
)
def test_launcher_runs_exact_frozen_map_and_publishes_hashed_results(
    tmp_path: Path,
    array_index: str,
    shape: str,
    task: str,
    short: str,
    seed: str,
) -> None:
    env = _prepared_fake_env(tmp_path, array_index=array_index, shape=shape)

    result = _run(env)

    assert result.returncode == 0, result.stdout + result.stderr
    attempt = (
        Path(env["HOME"])
        / "z1-fic-direct-reference-500"
        / env["EXPECTED_CODE_REVISION"]
        / env["EXPECTED_ASSET_REVISION"]
        / f"12345_{array_index}_{short}_seed{seed}"
    )
    run_name = f"fic_direct_reference_{short}_seed{seed}_job12345_{array_index}"
    calls = _calls(env)
    train = next(call for call in calls if call and call[0].endswith("scripts/train.py"))
    assert train == [
        str(Path(env["RUN_ROOT"]) / "scripts/train.py"),
        task,
        "--gpu-ids",
        "[0]",
        "--agent.logger",
        "tensorboard",
        "--agent.run-name",
        run_name,
        "--agent.seed",
        seed,
        "--agent.max-iterations",
        "500",
        "--agent.save-interval",
        "50",
        "--env.scene.num-envs",
        "4096",
    ]
    checkpoint = next((attempt / "logs/rsl_rl/z1_hammer").glob("*/model_499.pt"))
    evaluation = attempt / f"{short}_seed{seed}.json"
    curriculum = attempt / f"{short}_seed{seed}_curriculum.json"
    evaluate = next(
        call for call in calls if call and call[0].endswith("evaluate_fic_pilot.py")
    )
    assert evaluate == [
        str(Path(env["RUN_ROOT"]) / "evaluation/joint_position/evaluate_fic_pilot.py"),
        task,
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(evaluation),
        "--device",
        "cuda:0",
        "--training-seed",
        seed,
    ]
    payload = json.loads(curriculum.read_text())
    assert payload["task"] == task
    assert payload["arm"] == short
    assert payload["training_seed"] == int(seed)
    assert [row["weight"] for row in payload["observed"]] == [
        0.1,
        0.09,
        0.08,
        0.06,
        0.04,
        0.02,
        0.0,
    ]
    for label, path in (
        ("CHECKPOINT", checkpoint),
        ("EVALUATION", evaluation),
        ("CURRICULUM", curriculum),
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert f"Z1_DIRECT_{label}_SHA256={digest}" in result.stdout
    assert f"Z1_DIRECT_CODE_REVISION={env['EXPECTED_CODE_REVISION']}" in result.stdout
    assert f"Z1_DIRECT_ASSET_REVISION={env['EXPECTED_ASSET_REVISION']}" in result.stdout
    assert f"FIC_DIRECT_REFERENCE_DONE arm={short} seed={seed} task={task}" in result.stdout


def test_smoke_launcher_runs_all_live_smokes_and_both_real_catppo_updates(
    tmp_path: Path,
) -> None:
    env = _prepared_fake_env(tmp_path)
    env["SLURM_JOB_ID"] = "789"

    result = _run(env, SMOKE_LAUNCHER)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = _calls(env)
    assert calls == [
        ["-"],
        [
            str(Path(env["RUN_ROOT"]) / "scripts/smoke_joint_position_fixed.py"),
            "--task",
            "all",
            "--device",
            "cuda:0",
        ],
        [
            str(Path(env["RUN_ROOT"]) / "scripts/smoke_cat_soft.py"),
            "--task",
            FIC0_TASK,
            "--device",
            "cuda:0",
            "--iters",
            "1",
        ],
        [
            str(Path(env["RUN_ROOT"]) / "scripts/smoke_cat_soft.py"),
            "--task",
            f"{FIC0_TASK}-TT",
            "--device",
            "cuda:0",
            "--iters",
            "1",
        ],
    ]
    attempt = (
        Path(env["HOME"])
        / "z1-fic-direct-reference-smoke"
        / env["EXPECTED_CODE_REVISION"]
        / env["EXPECTED_ASSET_REVISION"]
        / "789_smoke"
    )
    assert attempt.is_dir()
    assert f"Z1_DIRECT_CODE_REVISION={env['EXPECTED_CODE_REVISION']}" in result.stdout
    assert f"Z1_DIRECT_ASSET_REVISION={env['EXPECTED_ASSET_REVISION']}" in result.stdout
    assert "Z1_DIRECT_CUDA_GPU=NVIDIA A100-SXM4-40GB" in result.stdout
    assert "FIC_DIRECT_REFERENCE_SMOKE_DONE" in result.stdout


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
def test_launchers_reject_all_cli_overrides(
    tmp_path: Path, launcher: Path
) -> None:
    env = _prepared_fake_env(tmp_path)
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"

    result = _run(env, launcher, "--env.rewards.r_imit.weight=9")

    assert result.returncode == 2
    assert "launcher accepts no arguments" in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
@pytest.mark.parametrize("optimization", ("1", "0", " "))
def test_launchers_reject_inherited_python_optimization_before_python(
    tmp_path: Path, launcher: Path, optimization: str
) -> None:
    env = _prepared_fake_env(tmp_path)
    env["PYTHONOPTIMIZE"] = optimization
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"

    result = _run(env, launcher)

    assert result.returncode == 2
    assert "PYTHONOPTIMIZE must be empty" in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("SLURM_ARRAY_TASK_COUNT", "6"),
        ("SLURM_ARRAY_TASK_MIN", "1"),
        ("SLURM_ARRAY_TASK_MAX", "2"),
        ("SLURM_ARRAY_TASK_STEP", "2"),
    ),
)
def test_launcher_rejects_every_unapproved_observable_array_shape(
    tmp_path: Path, name: str, value: str
) -> None:
    env = _base_env(tmp_path)
    env[name] = value

    result = _run(env)

    assert result.returncode == 2
    assert "Z1_DIRECT_FAIL: array must be exactly 0-1 or 2-5" in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize(
    ("array_index", "shape"),
    (("2", "canary"), ("0", "expansion"), ("6", "expansion")),
)
def test_launcher_rejects_index_outside_the_selected_exact_map(
    tmp_path: Path, array_index: str, shape: str
) -> None:
    env = _base_env(tmp_path, array_index=array_index, shape=shape)

    result = _run(env)

    assert result.returncode == 2
    assert (
        "Z1_DIRECT_FAIL: array index is not valid for the approved array shape"
        in result.stdout
    )
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("EXPECTED_CODE_REVISION", "a" * 39),
        ("EXPECTED_CODE_REVISION", "g" * 40),
        ("EXPECTED_ASSET_REVISION", "b" * 39),
        ("EXPECTED_ASSET_REVISION", "z" * 40),
    ),
)
def test_launchers_reject_malformed_revisions_before_python(
    tmp_path: Path, launcher: Path, name: str, value: str
) -> None:
    env = _base_env(tmp_path)
    env[name] = value
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"

    result = _run(env, launcher)

    assert result.returncode == 2
    assert f"{name} must be a 40-hex revision" in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
@pytest.mark.parametrize(
    ("repo_var", "message"),
    (
        ("RUN_ROOT", "code repository must be clean"),
        ("ASSET_REPO", "asset repository must be clean"),
    ),
)
def test_launchers_reject_dirty_repositories_before_python(
    tmp_path: Path, launcher: Path, repo_var: str, message: str
) -> None:
    env = _clean_repo_env(tmp_path)
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"
    (Path(env[repo_var]) / "owner-untracked.txt").write_text("do not use")

    result = _run(env, launcher)

    assert result.returncode == 2
    assert message in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
@pytest.mark.parametrize(
    ("repo_var", "repository_label"),
    (("RUN_ROOT", "code"), ("ASSET_REPO", "asset")),
)
def test_launchers_reject_git_status_command_failure_before_python(
    tmp_path: Path, launcher: Path, repo_var: str, repository_label: str
) -> None:
    env = _prepared_fake_env(tmp_path)
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"
    repository = Path(env[repo_var]).resolve()
    _install_failing_git_status(tmp_path, env, repository=repository)

    result = _run(env, launcher)

    assert result.returncode == 2
    assert f"{repository_label} repository status check failed" in result.stdout
    assert f"simulated git status failure for {repository}" in result.stderr
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
@pytest.mark.parametrize(
    ("repo_var", "revision_var", "message"),
    (
        ("RUN_ROOT", "EXPECTED_CODE_REVISION", "code revision mismatch"),
        ("ASSET_REPO", "EXPECTED_ASSET_REVISION", "asset revision mismatch"),
    ),
)
def test_launchers_reject_clean_but_mismatched_revisions_before_python(
    tmp_path: Path,
    launcher: Path,
    repo_var: str,
    revision_var: str,
    message: str,
) -> None:
    env = _clean_repo_env(tmp_path)
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"
    other = tmp_path / f"other-{repo_var.lower()}"
    env[revision_var] = _git_repo(other, repo_var)

    result = _run(env, launcher)

    assert result.returncode == 2
    assert message in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
def test_launchers_reject_noncanonical_asset_repository_before_python(
    tmp_path: Path, launcher: Path
) -> None:
    env = _clean_repo_env(tmp_path)
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"
    other_assets = tmp_path / "other-assets"
    env["ASSET_REPO"] = str(other_assets)
    env["EXPECTED_ASSET_REVISION"] = _git_repo(other_assets, "other assets")

    result = _run(env, launcher)

    assert result.returncode == 2
    assert "ASSET_REPO must be the canonical sibling" in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
def test_launchers_reject_wrong_cuda_or_runtime_packages(
    tmp_path: Path, launcher: Path
) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode="runtime-fail")
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"

    result = _run(env, launcher)

    assert result.returncode == 2
    assert "CUDA identity or runtime package versions do not match" in result.stdout
    assert not any(
        call and (
            call[0].endswith("scripts/train.py")
            or call[0].endswith("scripts/smoke_cat_soft.py")
        )
        for call in _calls(env)
    )


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
def test_launchers_never_reuse_an_attempt_directory(
    tmp_path: Path, launcher: Path
) -> None:
    env = _prepared_fake_env(tmp_path)
    if launcher == SMOKE_LAUNCHER:
        env["SLURM_JOB_ID"] = "789"
        leaf = "789_smoke"
        root_name = "z1-fic-direct-reference-smoke"
    else:
        leaf = "12345_0_fic0_seed2"
        root_name = "z1-fic-direct-reference-500"
    attempt = (
        Path(env["HOME"])
        / root_name
        / env["EXPECTED_CODE_REVISION"]
        / env["EXPECTED_ASSET_REVISION"]
        / leaf
    )
    attempt.mkdir(parents=True)
    sentinel = attempt / "owner-data.txt"
    sentinel.write_text("do not overwrite")

    result = _run(env, launcher)

    assert result.returncode == 2
    assert "attempt directory already exists" in result.stdout
    assert sentinel.read_text() == "do not overwrite"
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize(
    "fake_mode", ("no-run-dir", "multiple-run-dirs", "run-path-file")
)
def test_launcher_rejects_missing_or_multiple_training_runs(
    tmp_path: Path, fake_mode: str
) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode=fake_mode)

    result = _run(env)

    assert result.returncode == 2
    assert "expected exactly one matching training run directory" in result.stdout
    assert not any(
        call and call[0].endswith("evaluate_fic_pilot.py") for call in _calls(env)
    )
    assert "FIC_DIRECT_REFERENCE_DONE" not in result.stdout


def test_launcher_requires_model_499(tmp_path: Path) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode="missing-checkpoint")

    result = _run(env)

    assert result.returncode == 2
    assert "Z1_DIRECT_FAIL: model_499.pt is missing" in result.stdout
    assert not any(
        call and call[0].endswith("evaluate_fic_pilot.py") for call in _calls(env)
    )


def test_launcher_rejects_nonfinite_or_tensorless_checkpoint(tmp_path: Path) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode="checkpoint-fail")

    result = _run(env)

    assert result.returncode == 2
    assert "model_499.pt contains no finite checkpoint state" in result.stdout
    assert not any(
        call and call[0].endswith("evaluate_fic_pilot.py") for call in _calls(env)
    )
    assert "FIC_DIRECT_REFERENCE_DONE" not in result.stdout


@pytest.mark.parametrize(
    ("fake_mode", "returncode", "message"),
    (
        ("evaluator-fail", 17, ""),
        (
            "evaluator-missing-output",
            2,
            "evaluator did not create a non-empty JSON result",
        ),
        ("preexisting-evaluation", 2, "evaluation output already exists"),
        ("curriculum-fail", 2, "curriculum telemetry validation failed"),
        (
            "curriculum-missing-output",
            2,
            "curriculum extractor did not create a non-empty JSON result",
        ),
        ("preexisting-curriculum", 2, "curriculum output already exists"),
    ),
)
def test_launcher_requires_fresh_successful_evaluator_and_curriculum_outputs(
    tmp_path: Path, fake_mode: str, returncode: int, message: str
) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode=fake_mode)

    result = _run(env)

    assert result.returncode == returncode
    if message:
        assert message in result.stdout
    assert "FIC_DIRECT_REFERENCE_DONE" not in result.stdout


@pytest.mark.parametrize("launcher", (LAUNCHER, SMOKE_LAUNCHER))
@pytest.mark.parametrize(
    ("stub_kwargs", "message"),
    (
        ({"cuda_available": False}, "CUDA unavailable"),
        ({"device_count": 2}, "CUDA device count mismatch"),
        ({"gpu_name": "NVIDIA H100 80GB HBM3"}, "unexpected GPU"),
        ({"mjlab_version": "1.4.1"}, "runtime package versions do not match"),
    ),
)
def test_embedded_runtime_guards_reject_invalid_inputs_under_optimization(
    tmp_path: Path,
    launcher: Path,
    stub_kwargs: dict[str, object],
    message: str,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_install_runtime_stubs(tmp_path, **stub_kwargs))

    result = subprocess.run(
        [sys.executable, "-O", "-"],
        input=_runtime_guard_source(launcher),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("tensorless", "checkpoint contains no tensors"),
        ("nonfinite", "non-finite checkpoint tensor"),
    ),
)
def test_embedded_checkpoint_guard_rejects_invalid_state_under_optimization(
    tmp_path: Path, mode: str, message: str
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_install_checkpoint_torch_stub(tmp_path))
    env["FAKE_CHECKPOINT_MODE"] = mode
    checkpoint = tmp_path / "model_499.pt"
    checkpoint.write_text("fixture")

    result = subprocess.run(
        [sys.executable, "-O", "-", str(checkpoint)],
        input=_checkpoint_guard_source(),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_embedded_python_guards_do_not_use_optimization_sensitive_asserts() -> None:
    sources = (
        _runtime_guard_source(LAUNCHER),
        _runtime_guard_source(SMOKE_LAUNCHER),
        _checkpoint_guard_source(),
        _curriculum_extractor_source(),
    )

    assert all(re.search(r"(?m)^\s*assert\b", source) is None for source in sources)


def test_curriculum_extractor_accepts_convex_averages_and_preserves_raw_samples(
    tmp_path: Path,
) -> None:
    weights = [0.1, 0.09, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02, 0.01, 0.0]
    rows = [
        {"step": 4 + 7 * index, "value": weight}
        for index, weight in enumerate(weights)
    ]

    result, output = _run_curriculum_extractor(tmp_path, rows, optimize=True)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text())
    assert payload == {
        "schema_version": 1,
        "task": FIC0_TASK,
        "arm": "fic0",
        "training_seed": 2,
        "scalar": "Curriculum/r_imit_anneal/weight",
        "expected_control_step_stages": [
            [0, 0.1],
            [1200, 0.08],
            [2400, 0.06],
            [3600, 0.04],
            [4800, 0.02],
            [6000, 0.0],
        ],
        "observed": [
            {"iteration": 4 + 7 * index, "weight": weight}
            for index, weight in enumerate(weights)
        ],
        "final_weight": 0.0,
    }
    expected_bytes = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    assert output.read_text() == expected_bytes


@pytest.mark.parametrize(
    ("rows", "mode", "message"),
    (
        (
            [
                {"step": index, "value": value}
                for index, value in enumerate(
                    [0.1, float("nan"), 0.08, 0.06, 0.04, 0.02, 0.0]
                )
            ],
            "success",
            "non-finite curriculum weight",
        ),
        (
            [
                {"step": index, "value": value}
                for index, value in enumerate(
                    [0.1, 0.08, 0.09, 0.06, 0.04, 0.02, 0.0]
                )
            ],
            "success",
            "curriculum weights increase",
        ),
        (
            [
                {"step": index, "value": value}
                for index, value in enumerate(
                    [0.10001, 0.1, 0.08, 0.06, 0.04, 0.02, 0.0]
                )
            ],
            "success",
            "curriculum weight outside [0,0.1]",
        ),
        (
            [
                {"step": index, "value": value}
                for index, value in enumerate([0.1, 0.08, 0.06, 0.02, 0.0])
            ],
            "success",
            "canonical curriculum plateau was not observed: 0.04",
        ),
        (
            [
                {"step": step, "value": value}
                for step, value in zip(
                    [0, 1, 2, 2, 4, 5],
                    [0.1, 0.08, 0.06, 0.04, 0.02, 0.0],
                    strict=True,
                )
            ],
            "success",
            "curriculum event steps are not strictly increasing",
        ),
        (
            [
                {"step": step, "value": value}
                for step, value in zip(
                    [0, 1, 2.5, 3, 4, 5],
                    [0.1, 0.08, 0.06, 0.04, 0.02, 0.0],
                    strict=True,
                )
            ],
            "success",
            "curriculum event step is not an integer",
        ),
        (
            [
                {"step": step, "value": value}
                for step, value in zip(
                    [-1, 0, 1, 2, 3, 4],
                    [0.1, 0.08, 0.06, 0.04, 0.02, 0.0],
                    strict=True,
                )
            ],
            "success",
            "curriculum event step is negative",
        ),
        (
            [
                {"step": index, "value": value}
                for index, value in enumerate([0.1, 0.08, 0.06, 0.04, 0.02, 0.0])
            ],
            "missing-tag",
            "missing TensorBoard scalar",
        ),
        ([], "success", "empty curriculum scalar stream"),
    ),
)
def test_curriculum_extractor_rejects_invalid_telemetry_under_optimization(
    tmp_path: Path,
    rows: list[dict[str, object]],
    mode: str,
    message: str,
) -> None:
    result, output = _run_curriculum_extractor(
        tmp_path, rows, mode=mode, optimize=True
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


def test_curriculum_extractor_create_new_semantics_preserve_existing_output(
    tmp_path: Path,
) -> None:
    rows = [
        {"step": index, "value": value}
        for index, value in enumerate([0.1, 0.08, 0.06, 0.04, 0.02, 0.0])
    ]

    result, output = _run_curriculum_extractor(tmp_path, rows, preexisting=True)

    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert output.read_text() == "owner data"
