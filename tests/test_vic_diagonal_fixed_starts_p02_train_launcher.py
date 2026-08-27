"""Public contract for the five fixed-start VIC specialist launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_diagonal_fixed_starts_p02_train.sbatch"
TASK_PREFIX = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-VariableImpedance-TT-DiagonalFixedStart"
)
SPECIALISTS = (
    ("M40mm", "fixed_m40mm_p02", "-40", "-1"),
    ("M20mm", "fixed_m20mm_p02", "-20", "-1"),
    ("0mm", "fixed_0mm_p02", "0", "0"),
    ("P20mm", "fixed_p20mm_p02", "20", "1"),
    ("P40mm", "fixed_p40mm_p02", "40", "1"),
)


def _text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def _config_preflight_source() -> str:
    text = _text()
    marker = '"$PY" - <<\'PY\' || fail "fixed-start serialized config preflight failed"\n'
    start = text.index(marker) + len(marker)
    end = text.index("\nPY\n\ncd ", start)
    return text[start:end]


def test_fixed_start_launcher_freezes_one_guarded_five_policy_campaign() -> None:
    text = _text()
    for required in (
        "#SBATCH --array=0-4",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=40G",
        "#SBATCH --time=01:30:00",
        "#SBATCH --no-requeue",
        "%A_%a",
        "gn13",
        "OFFSETS_MM=(-40 -20 0 20 40)",
        "FIXED_SIGNS=(-1 -1 0 1 1)",
        "--agent.seed 2",
        "--agent.num-steps-per-env 24",
        "--agent.max-iterations 500",
        "--agent.save-interval 50",
        "--env.scene.num-envs 4096",
        'readonly IMP_MAX_P="0.2"',
        'readonly CAPS="[0.369,0.246,0.738,0.369,0.246,0.0164]"',
        "EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)",
        "fixed_route_sign",
        "use_default_offset",
        "joint_action.offset",
        "mapping drifted from the qualified VIC-TT contract",
        "r_imit",
        "imp_limit",
        "imp_max_p",
        "EXPECTED_CODE_REVISION",
        "EXPECTED_ASSET_REVISION",
        "status --porcelain=v1 --untracked-files=all",
        "A100-SXM4-40GB",
        "torch.isfinite",
        "sha256sum",
    ):
        assert required in text
    for suffix, role, _, _ in SPECIALISTS:
        assert f'"{TASK_PREFIX}{suffix}-Persistent"' in text
        assert f'"{role}"' in text
    assert "--resume" not in text
    assert "--requeue" not in text
    assert subprocess.run(["bash", "-n", str(LAUNCHER)], check=False).returncode == 0


def test_fixed_start_launcher_preflight_accepts_all_five_registered_tasks() -> None:
    source = _config_preflight_source()
    for suffix, role, offset_mm, fixed_sign in SPECIALISTS:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(ROOT),
                "TASK": f"{TASK_PREFIX}{suffix}-Persistent",
                "ROLE": role,
                "OFFSET_MM": offset_mm,
                "FIXED_SIGN": fixed_sign,
                "CAPS": "[0.369,0.246,0.738,0.369,0.246,0.0164]",
                "IMP_MAX_P": "0.2",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert f"role={role}" in completed.stdout
