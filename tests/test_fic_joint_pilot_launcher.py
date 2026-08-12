"""Retirement contract for the superseded waypoint-FIC Vega launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/slurm/vega_fic_joint_pilot.sbatch"
SUCCESSOR = "scripts/slurm/vega_fic_direct_reference.sbatch"
FROZEN_REVISION = "0b270e26fac03cea5b2fedac5a34b60276dd9a4d"


def _poisoned_env(tmp_path: Path, *, array_index: str) -> dict[str, str]:
    """Record the first legacy setup or Python action if retirement is bypassed."""
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    module = fake_bin / "module"
    module.write_text(
        "#!/bin/sh\n"
        "printf 'module %s\\n' \"$*\" >> \"$HOME/legacy_actions.log\"\n"
    )
    module.chmod(0o755)

    python = home / "repos/unitree_rl_mjlab/.venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\n"
        "printf 'python %s\\n' \"$*\" >> \"$HOME/legacy_actions.log\"\n"
        "exit 99\n"
    )
    python.chmod(0o755)
    return {
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "HOME": str(home),
        "SLURM_ARRAY_TASK_ID": array_index,
        "SLURM_ARRAY_JOB_ID": "12345",
        "SLURM_ARRAY_TASK_COUNT": "2",
        "SLURM_ARRAY_TASK_MIN": "0",
        "SLURM_ARRAY_TASK_MAX": "1",
        "SLURM_ARRAY_TASK_STEP": "1",
        "SLURM_JOB_ID": f"12345_{array_index}",
    }


@pytest.mark.parametrize("array_index", ("0", "1"))
def test_retired_fic_launcher_exits_before_setup_training_or_evaluation(
    tmp_path: Path, array_index: str
) -> None:
    env = _poisoned_env(tmp_path, array_index=array_index)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "RETIRED" in result.stderr
    assert SUCCESSOR in result.stderr
    assert (
        f"banked results remain reproducible from frozen revision {FROZEN_REVISION}"
        in result.stderr
    )
    assert not (Path(env["HOME"]) / "legacy_actions.log").exists()
    assert not (Path(env["HOME"]) / "z1-fic-joint-pilot").exists()
