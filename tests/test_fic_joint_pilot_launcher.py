"""Public shell contract for the matched two-arm FIC pilot launcher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/slurm/vega_fic_joint_pilot.sbatch"


def _base_env(tmp_path: Path, *, array_index: str = "0") -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "RUN_ROOT": str(tmp_path / "repos/code"),
        "ASSET_REPO": str(tmp_path / "repos/safe_impact_manipulation"),
        "EXPECTED_CODE_REVISION": "a" * 40,
        "EXPECTED_ASSET_REVISION": "b" * 40,
        "SLURM_ARRAY_TASK_ID": array_index,
        "SLURM_ARRAY_JOB_ID": "12345",
        "SLURM_ARRAY_TASK_COUNT": "2",
        "SLURM_ARRAY_TASK_MIN": "0",
        "SLURM_ARRAY_TASK_MAX": "1",
        "SLURM_ARRAY_TASK_STEP": "1",
        "SLURM_JOB_ID": f"12345_{array_index}",
    }


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
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
            "user.email=fic-pilot-test@example.com",
            "-c",
            "user.name=FIC Pilot Test",
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


def _clean_repo_env(tmp_path: Path, *, array_index: str = "0") -> dict[str, str]:
    env = _base_env(tmp_path, array_index=array_index)
    code = Path(env["RUN_ROOT"])
    assets = Path(env["ASSET_REPO"])
    env["EXPECTED_CODE_REVISION"] = _git_repo(code, "code")
    env["EXPECTED_ASSET_REVISION"] = _git_repo(assets, "assets")
    return env


def _install_fake_commands(tmp_path: Path, env: dict[str, str]) -> None:
    py = Path(env["HOME"]) / "repos/unitree_rl_mjlab/.venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$HOME/python_calls.log\"\n"
        "if [ \"${FAKE_PY_MODE:-success}\" = checkpoint-fail ] "
        "&& [ \"${1:-}\" = - ] && [ -n \"${2:-}\" ]; then\n"
        "  exit 9\n"
        "fi\n"
        "case \"$*\" in\n"
        "  *scripts/train.py*)\n"
        "    case \"${FAKE_PY_MODE:-success}\" in\n"
        "      no-run-dir) ;;\n"
        "      multiple-run-dirs)\n"
        "        for suffix in a b; do\n"
        "          dir=\"$PWD/logs/rsl_rl/z1_hammer/${suffix}_$RUN_NAME\"\n"
        "          mkdir -p \"$dir\"\n"
        "          printf 'fake checkpoint\\n' > \"$dir/model_199.pt\"\n"
        "        done\n"
        "        ;;\n"
        "      *)\n"
        "        dir=\"$PWD/logs/rsl_rl/z1_hammer/fixture_$RUN_NAME\"\n"
        "        mkdir -p \"$dir\"\n"
        "        printf 'fake checkpoint\\n' > \"$dir/model_199.pt\"\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "  *evaluate_fic_pilot.py*)\n"
        "    if [ \"${FAKE_PY_MODE:-success}\" = evaluator-fail ]; then exit 17; fi\n"
        "    if [ \"${FAKE_PY_MODE:-success}\" = evaluator-missing-output ]; then exit 0; fi\n"
        "    while [ $# -gt 0 ]; do\n"
        "      if [ \"$1\" = --output ]; then shift; printf '{}' > \"$1\"; break; fi\n"
        "      shift\n"
        "    done\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n"
    )
    py.chmod(0o755)


def _prepared_fake_env(
    tmp_path: Path, *, array_index: str = "0", fake_mode: str
) -> dict[str, str]:
    env = _clean_repo_env(tmp_path, array_index=array_index)
    _install_fake_commands(tmp_path, env)
    evaluator = Path(env["RUN_ROOT"]) / "evaluation/joint_position/evaluate_fic_pilot.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# evaluator fixture\n")
    _git(Path(env["RUN_ROOT"]), "add", "evaluation/joint_position/evaluate_fic_pilot.py")
    _git(Path(env["RUN_ROOT"]), "commit", "-q", "-m", "evaluator fixture")
    env["EXPECTED_CODE_REVISION"] = _git(
        Path(env["RUN_ROOT"]), "rev-parse", "HEAD"
    ).stdout.strip()
    env["FAKE_PY_MODE"] = fake_mode
    return env


def test_fic_pilot_launcher_has_frozen_two_arm_contract() -> None:
    source = LAUNCHER.read_text()

    assert "#SBATCH --array=0-1" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --output=/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot-%A_%a.out" in source
    assert "#SBATCH --error=/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot-%A_%a.err" in source
    assert (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4-JointPosition-Fixed"
    ) in source
    assert 'SHORT="fic0"' in source
    assert 'SHORT="fictt"' in source
    assert "--agent.seed 2" in source
    assert "--agent.max-iterations 200" in source
    assert "--agent.save-interval 50" in source
    assert "--env.scene.num-envs 4096" in source
    assert "--env.metrics" not in source
    assert "--env.rewards" not in source
    assert "--env.events" not in source
    assert "--env.terminations" not in source
    assert "model_199.pt" in source
    assert '"mjlab": "1.4.0"' in source
    assert '"mujoco": "3.8.1"' in source
    assert '"mujoco-warp": "3.8.1"' in source
    assert '"NVIDIA A100-SXM4-40GB"' in source
    assert "torch.isfinite" in source
    assert "evaluate_fic_pilot.py" in source
    assert "CHECKPOINT_SHA256" in source
    assert "EVALUATION_SHA256" in source


def test_fic_pilot_launcher_rejects_out_of_range_array_before_python(
    tmp_path: Path,
) -> None:
    env = _base_env(tmp_path, array_index="2")

    result = _run(env)

    assert result.returncode == 2
    assert "FIC_PILOT_FAIL: array index must be exactly 0 or 1" in result.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("SLURM_ARRAY_TASK_COUNT", "1"),
        ("SLURM_ARRAY_TASK_MIN", "1"),
        ("SLURM_ARRAY_TASK_MAX", "2"),
        ("SLURM_ARRAY_TASK_STEP", "2"),
    ),
)
def test_fic_pilot_launcher_rejects_array_shape_override_before_python(
    tmp_path: Path, name: str, value: str
) -> None:
    env = _base_env(tmp_path)
    env[name] = value

    result = _run(env)

    assert result.returncode == 2
    assert "FIC_PILOT_FAIL: array must be exactly 0-1" in result.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("EXPECTED_CODE_REVISION", "a" * 39),
        ("EXPECTED_CODE_REVISION", "g" * 40),
        ("EXPECTED_ASSET_REVISION", "b" * 39),
        ("EXPECTED_ASSET_REVISION", "z" * 40),
    ),
)
def test_fic_pilot_launcher_rejects_malformed_revision_before_python(
    tmp_path: Path, name: str, value: str
) -> None:
    env = _base_env(tmp_path)
    env[name] = value

    result = _run(env)

    assert result.returncode == 2
    assert f"FIC_PILOT_FAIL: {name} must be a 40-hex revision" in result.stdout


@pytest.mark.parametrize(
    ("repo_var", "message"),
    (
        ("RUN_ROOT", "code repository must be clean"),
        ("ASSET_REPO", "asset repository must be clean"),
    ),
)
def test_fic_pilot_launcher_rejects_dirty_repository_before_python(
    tmp_path: Path, repo_var: str, message: str
) -> None:
    env = _clean_repo_env(tmp_path)
    (Path(env[repo_var]) / "untracked.txt").write_text("dirty")

    result = _run(env)

    assert result.returncode == 2
    assert f"FIC_PILOT_FAIL: {message}" in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize(
    ("repo_var", "revision_var", "message"),
    (
        ("RUN_ROOT", "EXPECTED_CODE_REVISION", "code revision mismatch"),
        ("ASSET_REPO", "EXPECTED_ASSET_REVISION", "asset revision mismatch"),
    ),
)
def test_fic_pilot_launcher_rejects_wrong_clean_revision_before_python(
    tmp_path: Path, repo_var: str, revision_var: str, message: str
) -> None:
    env = _clean_repo_env(tmp_path)
    other_repo = tmp_path / f"other-{repo_var.lower()}"
    env[revision_var] = _git_repo(other_repo, f"other {repo_var}")

    result = _run(env)

    assert result.returncode == 2
    assert f"FIC_PILOT_FAIL: {message}" in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


def test_fic_pilot_launcher_rejects_noncanonical_asset_repository_before_python(
    tmp_path: Path,
) -> None:
    env = _clean_repo_env(tmp_path)
    other_assets = tmp_path / "other-assets"
    env["ASSET_REPO"] = str(other_assets)
    env["EXPECTED_ASSET_REVISION"] = _git_repo(other_assets, "other assets")

    result = _run(env)

    assert result.returncode == 2
    assert "FIC_PILOT_FAIL: ASSET_REPO must be the canonical sibling" in result.stdout
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


def test_fic_pilot_launcher_rejects_reused_attempt_before_python(
    tmp_path: Path,
) -> None:
    env = _clean_repo_env(tmp_path)
    attempt = (
        Path(env["HOME"])
        / "z1-fic-joint-pilot"
        / env["EXPECTED_CODE_REVISION"]
        / "12345_0_fic0_seed2"
    )
    attempt.mkdir(parents=True)
    sentinel = attempt / "owner-data.txt"
    sentinel.write_text("do not overwrite")

    result = _run(env)

    assert result.returncode == 2
    assert "FIC_PILOT_FAIL: attempt directory already exists" in result.stdout
    assert sentinel.read_text() == "do not overwrite"
    assert not (Path(env["HOME"]) / "python_calls.log").exists()


@pytest.mark.parametrize(
    ("array_index", "task_suffix", "short"),
    (("0", "JointPosition-Fixed", "fic0"), ("1", "JointPosition-Fixed-TT", "fictt")),
)
def test_fic_pilot_launcher_runs_exact_frozen_arm_and_evaluator(
    tmp_path: Path, array_index: str, task_suffix: str, short: str
) -> None:
    env = _prepared_fake_env(tmp_path, array_index=array_index, fake_mode="success")

    result = _run(env)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = (Path(env["HOME"]) / "python_calls.log").read_text().splitlines()
    train = next(line for line in calls if "scripts/train.py" in line)
    evaluate = next(line for line in calls if "evaluate_fic_pilot.py" in line)
    assert task_suffix in train
    assert "--agent.seed 2" in train
    assert "--agent.max-iterations 200" in train
    assert "--agent.save-interval 50" in train
    assert "--env.scene.num-envs 4096" in train
    assert all(
        token not in train
        for token in (
            "--env.rewards",
            "--env.metrics",
            "--env.events",
            "--env.terminations",
            "curriculum",
            "randomization",
            "gain",
        )
    )
    assert task_suffix in evaluate
    assert "--device cuda:0" in evaluate
    attempt = (
        Path(env["HOME"])
        / "z1-fic-joint-pilot"
        / env["EXPECTED_CODE_REVISION"]
        / f"12345_{array_index}_{short}_seed2"
    )
    evaluation = attempt / f"{short}.json"
    assert evaluation.is_file()
    checkpoint = next(
        (attempt / "logs/rsl_rl/z1_hammer").glob("*/model_199.pt")
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evaluation_sha256 = hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert f"FIC_PILOT_CHECKPOINT_SHA256={checkpoint_sha256}" in result.stdout
    assert f"FIC_PILOT_EVALUATION_SHA256={evaluation_sha256}" in result.stdout
    assert "FIC_PILOT_DONE" in result.stdout


@pytest.mark.parametrize("fake_mode", ("no-run-dir", "multiple-run-dirs"))
def test_fic_pilot_launcher_rejects_wrong_run_directory_count(
    tmp_path: Path, fake_mode: str
) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode=fake_mode)

    result = _run(env)

    assert result.returncode == 2
    assert (
        "FIC_PILOT_FAIL: expected exactly one matching training run directory"
        in result.stdout
    )
    assert "evaluate_fic_pilot.py" not in (
        Path(env["HOME"]) / "python_calls.log"
    ).read_text()
    assert "FIC_PILOT_DONE" not in result.stdout


def test_fic_pilot_launcher_rejects_failed_checkpoint_validation(
    tmp_path: Path,
) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode="checkpoint-fail")

    result = _run(env)

    assert result.returncode == 2
    assert (
        "FIC_PILOT_FAIL: model_199.pt contains no finite checkpoint state"
        in result.stdout
    )
    assert "evaluate_fic_pilot.py" not in (
        Path(env["HOME"]) / "python_calls.log"
    ).read_text()
    assert "FIC_PILOT_DONE" not in result.stdout


def test_fic_pilot_launcher_propagates_evaluator_failure(
    tmp_path: Path,
) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode="evaluator-fail")

    result = _run(env)

    assert result.returncode == 17
    assert "evaluate_fic_pilot.py" in (
        Path(env["HOME"]) / "python_calls.log"
    ).read_text()
    assert "FIC_PILOT_DONE" not in result.stdout


def test_fic_pilot_launcher_rejects_missing_evaluator_output(
    tmp_path: Path,
) -> None:
    env = _prepared_fake_env(tmp_path, fake_mode="evaluator-missing-output")

    result = _run(env)

    assert result.returncode == 2
    assert (
        "FIC_PILOT_FAIL: evaluator did not create a non-empty JSON result"
        in result.stdout
    )
    assert "FIC_PILOT_DONE" not in result.stdout
