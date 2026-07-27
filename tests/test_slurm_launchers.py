"""Fail-closed tests for the fq4x8 branch of the Vega Slurm launchers.

Drives scripts/slurm/vega_train.sbatch and scripts/slurm/vega_eval.sbatch with
`bash` directly -- never `sbatch`/`srun`/`salloc`/ssh -- and asserts on exit
codes plus FQ4X8_FAIL/EVAL_FAIL messages. Every guard under test fires before
any GPU/training/evaluation work: a "valid" invocation only proves the fq4x8
guard let it through, then is expected to fail later for an unrelated,
pre-existing, non-fq4x8 reason (no CUDA/A100 on this machine) -- it never
reaches real training or evaluation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "slurm" / "vega_train.sbatch"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "slurm" / "vega_eval.sbatch"

FQ_ARMS = (
    ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear", "f8"),
    ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0", "f0"),
    ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0", "d0"),
    ("Unitree-Z1-Hammer-CaT-Impulse-Event-Quality", "fq"),
)
FQ_SEEDS = tuple(range(8, 16))


def _git(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )


def _git_repo(path: Path) -> str:
    """Init a one-commit repo (clean, so the launcher's dirty-check passes)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitignore").write_text("logs/\n.venv/\n")
    _git(path, "init", "-q")
    _git(path, "add", ".gitignore")
    _git(path, "commit", "-q", "-m", "init")
    return _git(path, "rev-parse", "HEAD").stdout.strip()


def _base_env(home: Path) -> dict:
    # Deliberately NOT os.environ: a full-env test would let a stray
    # IMPACT_W/DELIVERED_W in the developer's own shell mask the guard.
    return {"PATH": os.environ.get("PATH", ""), "HOME": str(home)}


def _run(script: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def campaign_env(tmp_path):
    repo_root = tmp_path / "repo_root"
    sibling = tmp_path / "safe_impact_manipulation"
    home = tmp_path / "home"
    home.mkdir()
    return {
        "repo_root": repo_root,
        "sibling": sibling,
        "home": home,
        "code_rev": _git_repo(repo_root),
        "asset_rev": _git_repo(sibling),
    }


def _apply_overrides(env: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _train_env(campaign_env, **overrides) -> dict:
    env = _base_env(campaign_env["home"])
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(campaign_env["repo_root"]),
            "SLURM_ARRAY_TASK_ID": "0",
            "CAMPAIGN": "fq4x8",
            "SEEDS": "8 9 10 11 12 13 14 15",
            "SINGLE_TASK": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
            "SINGLE_SHORT": "f8",
            "ITERS": "500",
            "EXPECTED_CODE_REVISION": campaign_env["code_rev"],
            "EXPECTED_ASSET_REVISION": campaign_env["asset_rev"],
        }
    )
    return _apply_overrides(env, overrides)


# --------------------------------------------------------------------------
# Training branch
# --------------------------------------------------------------------------


def test_train_requests_exactly_one_gpu_per_run():
    assert "#SBATCH --gres=gpu:1" in TRAIN_SCRIPT.read_text()


@pytest.mark.parametrize("task, short", FQ_ARMS)
def test_train_accepts_each_frozen_arm(campaign_env, task, short):
    env = _train_env(campaign_env, SINGLE_TASK=task, SINGLE_SHORT=short)
    result = _run(TRAIN_SCRIPT, env)
    out = result.stdout + result.stderr
    assert "FQ4X8_FAIL" not in out
    assert "### TRAIN task=" in out
    # The guard passed; it must now fail for an unrelated, pre-existing reason
    # (no CUDA/A100 on this machine) -- never reaching real training.
    assert result.returncode == 1
    assert "### NO GPU/A100" in out


def test_train_rejects_wrong_task(campaign_env):
    env = _train_env(
        campaign_env, SINGLE_TASK="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Bogus"
    )
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: SINGLE_TASK/SINGLE_SHORT is not a frozen fq4x8 arm" in result.stdout


def test_train_rejects_wrong_short(campaign_env):
    env = _train_env(campaign_env, SINGLE_SHORT="zz")
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: SINGLE_TASK/SINGLE_SHORT is not a frozen fq4x8 arm" in result.stdout


def test_train_rejects_wrong_seed_set(campaign_env):
    env = _train_env(campaign_env, SEEDS="0 1 2 3 4 5 6 7")
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: seeds must be exactly 8..15" in result.stdout


def test_train_rejects_wrong_iteration_count(campaign_env):
    env = _train_env(campaign_env, ITERS="1000")
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: ITERS must be 500" in result.stdout


@pytest.mark.parametrize("override_var", ["IMPACT_W", "DELIVERED_W", "NAIL_DRIVEN_W"])
def test_train_rejects_each_leaked_reward_override(campaign_env, override_var):
    env = _train_env(campaign_env, **{override_var: "8"})
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert f"FQ4X8_FAIL: {override_var} must be unset" in result.stdout


def test_train_rejects_missing_code_revision(campaign_env):
    env = _train_env(campaign_env, EXPECTED_CODE_REVISION=None)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision" in result.stdout


def test_train_rejects_dirty_code_revision(campaign_env):
    env = _train_env(
        campaign_env, EXPECTED_CODE_REVISION=campaign_env["code_rev"] + "-dirty"
    )
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision" in result.stdout


def test_train_rejects_short_code_revision(campaign_env):
    env = _train_env(campaign_env, EXPECTED_CODE_REVISION=campaign_env["code_rev"][:10])
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision" in result.stdout


def test_train_rejects_nonhex_code_revision(campaign_env):
    env = _train_env(campaign_env, EXPECTED_CODE_REVISION="g" * 40)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision" in result.stdout


def test_train_rejects_bad_asset_revision(campaign_env):
    env = _train_env(campaign_env, EXPECTED_ASSET_REVISION="z" * 40)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: EXPECTED_ASSET_REVISION must be a clean 40-hex revision" in result.stdout


def test_train_fsr4x8_still_requires_impact_w_8(campaign_env):
    """Regression: fsr4x8 keeps its own (opposite) rule -- IMPACT_W=8 required."""
    env = _base_env(campaign_env["home"])
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(campaign_env["repo_root"]),
            "SLURM_ARRAY_TASK_ID": "0",
            "CAMPAIGN": "fsr4x8",
            "SEEDS": "0 1 2 3 4 5 6 7",
            "SINGLE_TASK": "Unitree-Z1-Hammer-CaT-Impulse",
            "SINGLE_SHORT": "c",
            "ITERS": "500",
            "DELIVERED_W": "2",
            "EXPECTED_CODE_REVISION": campaign_env["code_rev"],
            "EXPECTED_ASSET_REVISION": campaign_env["asset_rev"],
            # IMPACT_W deliberately omitted -> fsr4x8 must still reject.
        }
    )
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FSR4X8_FAIL: IMPACT_W must be 8" in result.stdout


# --------------------------------------------------------------------------
# CAMPAIGN typo guard (case-insensitive resemblance to a known campaign)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_campaign, expected_prefix",
    [
        ("Fq4x8", "FQ4X8_FAIL"),
        ("FQ4X8", "FQ4X8_FAIL"),
        ("fsr4X8", "FSR4X8_FAIL"),
    ],
)
def test_train_rejects_campaign_typo(campaign_env, bad_campaign, expected_prefix):
    """A CAMPAIGN value that case-insensitively matches a known campaign but
    isn't an exact match must be rejected, not silently fall through to the
    unguarded default path (with leaked overrides, wrong seeds, no revision
    enforcement)."""
    env = _train_env(campaign_env, CAMPAIGN=bad_campaign, SEEDS="0 1 2", IMPACT_W="999")
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert f"{expected_prefix}: CAMPAIGN '{bad_campaign}' looks like" in result.stdout


def test_train_unrelated_campaign_unaffected_by_typo_guard(campaign_env):
    """Regression: a genuinely different campaign name must behave exactly as
    before -- the new typo guard must not reject it."""
    env = _train_env(campaign_env, CAMPAIGN="ab1")
    result = _run(TRAIN_SCRIPT, env)
    out = result.stdout + result.stderr
    assert "FQ4X8_FAIL" not in out
    assert "FSR4X8_FAIL" not in out
    assert result.returncode == 1
    assert "### NO GPU/A100" in out


# --------------------------------------------------------------------------
# Evaluation branch
# --------------------------------------------------------------------------


def _valid_manifest_rows() -> list[tuple]:
    rows = []
    for task, short in FQ_ARMS:
        for seed in FQ_SEEDS:
            run_name = f"fq4x8_{short}_seed{seed}"
            rows.append(
                (short, task, seed, run_name, f"/fake/{run_name}/model_499.pt", "a" * 64)
            )
    return rows


def _write_manifest(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(str(field) for field in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def eval_env_data(campaign_env):
    eval_root = campaign_env["home"] / "unitree_rl_mjlab_eval"
    manifest_path = eval_root / "fq4x8" / "accepted_training_checkpoints.tsv"
    _write_manifest(manifest_path, _valid_manifest_rows())
    campaign_env["eval_root"] = eval_root
    campaign_env["manifest_path"] = manifest_path
    return campaign_env


def _eval_env(env_data, **overrides) -> dict:
    env = _base_env(env_data["home"])
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(env_data["repo_root"]),
            "CAMPAIGN": "fq4x8",
            "EVAL_ATTEMPT": "attempt1",
            "EVAL_ROOT": str(env_data["eval_root"]),
            "ACCEPTED_MANIFEST": str(env_data["manifest_path"]),
            "EXPECTED_CODE_REVISION": env_data["code_rev"],
            "EXPECTED_ASSET_REVISION": env_data["asset_rev"],
        }
    )
    return _apply_overrides(env, overrides)


def test_eval_accepts_valid_32_row_manifest(eval_env_data):
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    out = result.stdout + result.stderr
    assert "EVAL_FAIL" not in out
    assert "### EVAL_PROVENANCE" in out
    # The guard passed; it must now fail for an unrelated, pre-existing reason
    # (no CUDA on this machine) -- never reaching real evaluation.
    assert result.returncode == 1
    assert "### NO CUDA" in out


def test_eval_rejects_wrong_manifest_row_count(eval_env_data):
    rows = _valid_manifest_rows()[:-1]  # 31 rows
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: accepted-training manifest must contain exactly 32 rows, found 31"
        in result.stdout
    )


def test_eval_rejects_task_short_mapping_violation(eval_env_data):
    rows = _valid_manifest_rows()
    short, _task, seed, run_name, ckpt, sha = rows[0]
    rows[0] = (short, "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-BOGUS", seed, run_name, ckpt, sha)
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "outside the frozen fq4x8 task/short mapping" in result.stdout


def test_eval_rejects_missing_manifest(eval_env_data):
    eval_env_data["manifest_path"].unlink()
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: accepted-training manifest is missing" in result.stdout


def test_eval_rejects_duplicated_fq4x8_segment_via_eval_root(eval_env_data):
    env = _eval_env(eval_env_data, EVAL_ROOT=str(eval_env_data["eval_root"] / "fq4x8"))
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EVAL_ROOT must be exactly" in result.stdout


def test_eval_rejects_duplicated_fq4x8_segment_via_manifest(eval_env_data):
    duplicated = (
        eval_env_data["eval_root"] / "fq4x8" / "fq4x8" / "accepted_training_checkpoints.tsv"
    )
    env = _eval_env(eval_env_data, ACCEPTED_MANIFEST=str(duplicated))
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: ACCEPTED_MANIFEST must be exactly" in result.stdout


def test_eval_rejects_missing_code_revision(eval_env_data):
    env = _eval_env(eval_env_data, EXPECTED_CODE_REVISION=None)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision" in result.stdout


def test_eval_rejects_dirty_code_revision(eval_env_data):
    env = _eval_env(
        eval_env_data, EXPECTED_CODE_REVISION=eval_env_data["code_rev"] + "-dirty"
    )
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision" in result.stdout


def test_eval_rejects_short_code_revision(eval_env_data):
    env = _eval_env(eval_env_data, EXPECTED_CODE_REVISION=eval_env_data["code_rev"][:10])
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision" in result.stdout


def test_eval_rejects_nonhex_code_revision(eval_env_data):
    env = _eval_env(eval_env_data, EXPECTED_CODE_REVISION="g" * 40)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision" in result.stdout


def test_eval_fsr4x8_unaffected_by_fq4x8_guard(campaign_env):
    """Regression: fsr4x8 still hits the pre-existing CUDA guard first, and the
    new fq4x8-only early block never fires for it."""
    env = _base_env(campaign_env["home"])
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(campaign_env["repo_root"]),
            "CAMPAIGN": "fsr4x8",
        }
    )
    result = _run(EVAL_SCRIPT, env)
    out = result.stdout + result.stderr
    assert "### NO CUDA" in out
    assert "EVAL_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex" not in out


# --------------------------------------------------------------------------
# CAMPAIGN typo guard (case-insensitive resemblance to a known campaign)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_campaign", ["Fq4x8", "FQ4X8", "fsr4X8"])
def test_eval_rejects_campaign_typo(eval_env_data, bad_campaign):
    """A CAMPAIGN value that case-insensitively matches a known campaign but
    isn't an exact match must be rejected before any GPU/manifest work, not
    silently fall through to the unguarded default (glob-based) path."""
    env = _eval_env(eval_env_data, CAMPAIGN=bad_campaign)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert f"EVAL_FAIL: CAMPAIGN '{bad_campaign}' looks like" in result.stdout


def test_eval_unrelated_campaign_unaffected_by_typo_guard(campaign_env):
    """Regression: a genuinely different campaign name must behave exactly as
    before -- the new typo guard must not reject it."""
    env = _base_env(campaign_env["home"])
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(campaign_env["repo_root"]),
            "CAMPAIGN": "ab1",
        }
    )
    result = _run(EVAL_SCRIPT, env)
    out = result.stdout + result.stderr
    assert "EVAL_FAIL: CAMPAIGN" not in out
    assert "### NO CUDA" in out


# --------------------------------------------------------------------------
# Accepted-training manifest row shape (field count + training_seed range)
# --------------------------------------------------------------------------


def test_eval_rejects_manifest_row_with_too_few_fields(eval_env_data):
    """A row missing checkpoint_path/checkpoint_sha256 (fewer than 6 tab
    fields) must fail closed even though it still preserves the 32-row count
    and the per-arm count (columns 1-2 are untouched)."""
    rows = _valid_manifest_rows()
    short, task, seed, run_name, _ckpt, _sha = rows[0]
    rows[0] = (short, task, seed, run_name)  # missing checkpoint_path + checkpoint_sha256
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: accepted-training manifest has 1 row(s) without exactly 6 tab-separated fields"
        in result.stdout
    )


def test_eval_rejects_training_seed_out_of_range(eval_env_data):
    rows = _valid_manifest_rows()
    short, task, _seed, run_name, ckpt, sha = rows[0]
    rows[0] = (short, task, 7, run_name, ckpt, sha)  # valid range is 8..15
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: accepted-training manifest has 1 row(s) with training_seed outside 8..15"
        in result.stdout
    )


def test_eval_rejects_duplicate_training_seed_within_arm(eval_env_data):
    """8 rows, all seeds in range, but not the exact set 8..15 (a duplicate
    displaces one of the required seeds) must still fail closed."""
    rows = _valid_manifest_rows()
    short, task, _seed, run_name, ckpt, sha = rows[7]  # f8 arm, seed 15
    rows[7] = (short, task, 8, run_name, ckpt, sha)  # duplicate seed 8; seed 15 now missing
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: accepted-training manifest has duplicate training_seed values "
        "for f8/Unitree-Z1-Hammer-CaT-Impulse-Event-Linear" in result.stdout
    )
