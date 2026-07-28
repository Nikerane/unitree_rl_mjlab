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

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import evaluation.analysis.first_strike_campaign as legacy
import evaluation.analysis.first_strike_quality_campaign as quality
import evaluation.analysis.fq4x8_manifests as manifests

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


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


def _git(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )


def _git_repo(path: Path) -> str:
    """Init a one-commit repo (clean, so the launcher's dirty-check passes).

    The commit content is seeded from ``path`` itself so two independently
    created repos never collide on commit hash (git commit hashes have
    second-resolution timestamps by default -- an identical tree/author/
    message committed within the same wall-clock second produces an
    IDENTICAL hash, which would silently defeat any "wrong revision"
    equality test).
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitignore").write_text(f"logs/\n.venv/\n# {path.name}\n")
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
    assert "FQ4X8_FAIL: ITERS must be exactly 500" in result.stdout


def test_train_rejects_missing_iters(campaign_env):
    """MEDIUM-3: an omitted ITERS must fail closed, not silently default to
    500 -- ITERS=500 is a frozen contract for this campaign, not a default."""
    env = _train_env(campaign_env, ITERS=None)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: ITERS must be exactly 500" in result.stdout


def test_train_rejects_empty_iters(campaign_env):
    """MEDIUM-3: an exported-but-empty ITERS must also fail closed --
    "${ITERS:-500}" treats empty the same as unset, which is the hole."""
    env = _train_env(campaign_env, ITERS="")
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: ITERS must be exactly 500" in result.stdout


@pytest.mark.parametrize("override_var", ["IMPACT_W", "DELIVERED_W", "NAIL_DRIVEN_W"])
def test_train_rejects_each_leaked_reward_override(campaign_env, override_var):
    env = _train_env(campaign_env, **{override_var: "8"})
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert f"FQ4X8_FAIL: {override_var} must be unset" in result.stdout


@pytest.mark.parametrize("override_var", ["IMPACT_W", "DELIVERED_W", "NAIL_DRIVEN_W"])
def test_train_rejects_each_exported_empty_reward_override(campaign_env, override_var):
    """LOW-5: "must be unset" was implemented as "must be empty" (`-z
    ${VAR:-}`), so an exported-but-empty override (e.g. a leaked --export=ALL
    setting IMPACT_W="") passed. `${VAR+x}` distinguishes "unset" from
    "set to empty"."""
    env = _train_env(campaign_env, **{override_var: ""})
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


def test_train_rejects_missing_asset_revision(campaign_env):
    env = _train_env(campaign_env, EXPECTED_ASSET_REVISION=None)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: EXPECTED_ASSET_REVISION must be a clean 40-hex revision" in result.stdout


def test_train_rejects_short_asset_revision(campaign_env):
    env = _train_env(campaign_env, EXPECTED_ASSET_REVISION=campaign_env["asset_rev"][:10])
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: EXPECTED_ASSET_REVISION must be a clean 40-hex revision" in result.stdout


def test_train_rejects_wrong_code_revision_equality(campaign_env):
    """Well-formed 40-hex, but simply the wrong commit -- distinct from the
    format checks above."""
    other = _git_repo(campaign_env["home"].parent / "other-code-repo")
    env = _train_env(campaign_env, EXPECTED_CODE_REVISION=other)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: code revision does not equal expected code revision" in result.stdout


def test_train_rejects_wrong_asset_revision_equality(campaign_env):
    other = _git_repo(campaign_env["home"].parent / "other-asset-repo")
    env = _train_env(campaign_env, EXPECTED_ASSET_REVISION=other)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "FQ4X8_FAIL: asset revision does not equal expected asset revision" in result.stdout


def test_train_rejects_actual_dirty_code_repo(campaign_env):
    (campaign_env["repo_root"] / "untracked.txt").write_text("dirty")
    env = _train_env(campaign_env)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "code provenance is unknown or dirty" in result.stdout


def test_train_rejects_actual_dirty_asset_repo(campaign_env):
    (campaign_env["sibling"] / "untracked.txt").write_text("dirty")
    env = _train_env(campaign_env)
    result = _run(TRAIN_SCRIPT, env)
    assert result.returncode == 2
    assert "asset provenance is unknown or dirty" in result.stdout


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
# fq3x8 training branch
# --------------------------------------------------------------------------


FQ3X8_ARMS = (
    ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear", "f8"),
    ("Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded", "b8"),
    ("Unitree-Z1-Hammer-CaT-Impulse-Event-Quality", "fq"),
)


def _fq3x8_train_env(campaign_env, **overrides) -> dict:
    env = _base_env(campaign_env["home"])
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(campaign_env["repo_root"]),
            "SLURM_ARRAY_TASK_ID": "0",
            "CAMPAIGN": "fq3x8",
            "SEEDS": "16 17 18 19 20 21 22 23",
            "SINGLE_TASK": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
            "SINGLE_SHORT": "f8",
            "ITERS": "500",
            "EXPECTED_CODE_REVISION": campaign_env["code_rev"],
            "EXPECTED_ASSET_REVISION": campaign_env["asset_rev"],
        }
    )
    return _apply_overrides(env, overrides)


@pytest.mark.parametrize("task, short", FQ3X8_ARMS)
def test_fq3x8_train_accepts_only_each_frozen_arm(campaign_env, task, short):
    """Changing the frozen task/short matrix must reject before GPU work."""
    result = _run(
        TRAIN_SCRIPT, _fq3x8_train_env(campaign_env, SINGLE_TASK=task, SINGLE_SHORT=short)
    )
    out = result.stdout + result.stderr
    assert "FQ3X8_FAIL" not in out
    assert "### TRAIN task=" in out
    assert result.returncode == 1
    assert "### NO GPU/A100" in out


@pytest.mark.parametrize(
    "overrides",
    [
        {"SINGLE_TASK": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Bogus"},
        {"SINGLE_SHORT": "wrong"},
    ],
)
def test_fq3x8_train_rejects_task_short_matrix_mutations(campaign_env, overrides):
    """A broadened or mismatched frozen matrix must fail closed."""
    result = _run(TRAIN_SCRIPT, _fq3x8_train_env(campaign_env, **overrides))
    assert result.returncode == 2
    assert "FQ3X8_FAIL: SINGLE_TASK/SINGLE_SHORT is not a frozen fq3x8 arm" in result.stdout


@pytest.mark.parametrize(
    "task, short",
    [
        ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear", "b8"),
        ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear", "fq"),
        ("Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded", "f8"),
        ("Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded", "fq"),
        ("Unitree-Z1-Hammer-CaT-Impulse-Event-Quality", "f8"),
        ("Unitree-Z1-Hammer-CaT-Impulse-Event-Quality", "b8"),
    ],
)
def test_fq3x8_train_rejects_every_cross_pair_of_frozen_values(campaign_env, task, short):
    """Valid task and short values are invalid when paired with another arm."""
    result = _run(
        TRAIN_SCRIPT, _fq3x8_train_env(campaign_env, SINGLE_TASK=task, SINGLE_SHORT=short)
    )
    assert result.returncode == 2
    assert "FQ3X8_FAIL: SINGLE_TASK/SINGLE_SHORT is not a frozen fq3x8 arm" in result.stdout
    assert "### NO GPU/A100" not in result.stdout


@pytest.mark.parametrize(
    "overrides, expected_message",
    [
        ({"CAMPAIGN": "Fq3x8"}, "FQ3X8_FAIL: CAMPAIGN 'Fq3x8' looks like fq3x8"),
        ({"SEEDS": "8 9 10 11 12 13 14 15"}, "FQ3X8_FAIL: seeds must be exactly 16..23"),
        ({"ITERS": "499"}, "FQ3X8_FAIL: ITERS must be exactly 500"),
        ({"ITERS": None}, "FQ3X8_FAIL: ITERS must be exactly 500"),
        ({"ITERS": ""}, "FQ3X8_FAIL: ITERS must be exactly 500"),
    ],
)
def test_fq3x8_train_rejects_campaign_seed_iteration_mutations(
    campaign_env, overrides, expected_message
):
    """Exact campaign spelling, seed list, and explicit iteration count are frozen."""
    result = _run(TRAIN_SCRIPT, _fq3x8_train_env(campaign_env, **overrides))
    assert result.returncode == 2
    assert expected_message in result.stdout


@pytest.mark.parametrize("override_var", ["IMPACT_W", "DELIVERED_W", "NAIL_DRIVEN_W"])
@pytest.mark.parametrize("leaked_value", ["8", ""])
def test_fq3x8_train_rejects_set_reward_override(campaign_env, override_var, leaked_value):
    """Set-to-empty is still a leaked reward override, not an unset variable."""
    result = _run(
        TRAIN_SCRIPT, _fq3x8_train_env(campaign_env, **{override_var: leaked_value})
    )
    assert result.returncode == 2
    assert f"FQ3X8_FAIL: {override_var} must be unset" in result.stdout


@pytest.mark.parametrize(
    "revision_var, bad_revision",
    [
        ("EXPECTED_CODE_REVISION", None),
        ("EXPECTED_CODE_REVISION", "g" * 40),
        ("EXPECTED_CODE_REVISION", "a" * 39),
        ("EXPECTED_ASSET_REVISION", None),
        ("EXPECTED_ASSET_REVISION", "z" * 40),
        ("EXPECTED_ASSET_REVISION", "b" * 39),
    ],
)
def test_fq3x8_train_rejects_noncanonical_expected_revision(
    campaign_env, revision_var, bad_revision
):
    """Both expected revisions must be supplied as clean, 40-character hex IDs."""
    result = _run(
        TRAIN_SCRIPT, _fq3x8_train_env(campaign_env, **{revision_var: bad_revision})
    )
    assert result.returncode == 2
    assert f"FQ3X8_FAIL: {revision_var} must be a clean 40-hex revision" in result.stdout


@pytest.mark.parametrize(
    "revision_var, other_repo",
    [
        ("EXPECTED_CODE_REVISION", "other-fq3x8-code-repo"),
        ("EXPECTED_ASSET_REVISION", "other-fq3x8-asset-repo"),
    ],
)
def test_fq3x8_train_rejects_wrong_well_formed_revision(
    campaign_env, revision_var, other_repo
):
    """A syntactically valid revision must still equal the clean target repository."""
    other = _git_repo(campaign_env["home"].parent / other_repo)
    result = _run(TRAIN_SCRIPT, _fq3x8_train_env(campaign_env, **{revision_var: other}))
    assert result.returncode == 2
    expected = "code" if revision_var == "EXPECTED_CODE_REVISION" else "asset"
    assert f"FQ3X8_FAIL: {expected} revision does not equal expected {expected} revision" in result.stdout


@pytest.mark.parametrize("campaign", ["fq4x8", "fsr4x8"])
def test_fq3x8_train_guard_leaves_historical_campaigns_unchanged(campaign_env, campaign):
    """The fq3x8 branch must not reject pre-existing guarded campaigns."""
    if campaign == "fq4x8":
        env = _train_env(campaign_env)
    else:
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
                "IMPACT_W": "8",
                "DELIVERED_W": "2",
                "EXPECTED_CODE_REVISION": campaign_env["code_rev"],
                "EXPECTED_ASSET_REVISION": campaign_env["asset_rev"],
            }
        )
    result = _run(TRAIN_SCRIPT, env)
    out = result.stdout + result.stderr
    assert "FQ3X8_FAIL" not in out
    assert result.returncode == 1
    assert "### NO GPU/A100" in out


# --------------------------------------------------------------------------
# Evaluation branch
#
# The accepted-training manifest is now the canonical 21-column, headered TSV
# (evaluation.analysis.fq4x8_manifests.TRAINING_FIELDS), validated by invoking
# the real `python -m evaluation.analysis.fq4x8_manifests validate-accepted-
# manifest` CLI from inside the launcher. To exercise that real validation
# (not just stop at "no CUDA"), eval_env_data installs a `.venv/bin/python`
# wrapper that forwards ONLY that one invocation to the real interpreter
# (against the real source tree, via FQ_SOURCE_ROOT/FQ_REAL_PYTHON) and fails
# every other invocation -- so the pre-existing CUDA-guard fallback ("### NO
# CUDA", returncode 1) still fires for every accepted manifest, exactly as
# before, and the launcher never reaches real evaluation.
# --------------------------------------------------------------------------


def _hex(label: str, length: int = 64) -> str:
    return hashlib.sha256(label.encode()).hexdigest()[:length]


def _training_row(arm: str, seed: int, *, code_rev: str, asset_rev: str) -> dict:
    return {
        "campaign": manifests.CAMPAIGN_NAME,
        "disposition": "accepted",
        "arm": arm,
        "short": manifests.SHORT[arm],
        "task": manifests.TASKS[arm],
        "training_seed": seed,
        "checkpoint_path": f"/fake/checkpoints/{arm}/seed{seed}/model_499.pt",
        "checkpoint_sha256": _hex(f"checkpoint:{arm}:{seed}"),
        "training_attempt": "attempt1",
        "retry_history": "attempt1:accepted",
        "code_revision": code_rev,
        "asset_revision": asset_rev,
        "campaign_config_sha256": _hex("campaign-config"),
        "treatment_config_sha256": manifests.EXPECTED_TREATMENT_CONFIG_SHA256[arm],
        "reader_sha256": manifests.EXPECTED_READER_SHA256[arm],
        "normalizer_sha256": manifests.EXPECTED_NORMALIZER_SHA256[arm],
        "treatment_reward_sha256": quality.EXPECTED_TREATMENT_REWARD_SHA256[arm],
        "fixed_action_signature_sha256": legacy.EXPECTED_FIXED_ACTION_SIGNATURE_SHA256,
        "fixed_impedance_signature_sha256": legacy.EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256,
        "cap_signature_sha256": manifests.EXPECTED_CAP_SIGNATURE_SHA256,
        "clean_state": True,
    }


def valid_training_rows(*, code_rev: str, asset_rev: str) -> list[dict]:
    return [
        _training_row(arm, seed, code_rev=code_rev, asset_rev=asset_rev)
        for arm in manifests.LABELS
        for seed in sorted(manifests.SEEDS)
    ]


def _mutate(rows: list[dict], index: int, **overrides) -> list[dict]:
    import copy

    rows = copy.deepcopy(rows)
    rows[index] = {**rows[index], **overrides}
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifests.serialize_training_manifest(rows))


FQ_PYTHON_WRAPPER = """#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "evaluation.analysis.fq4x8_manifests" ]; then
  PYTHONPATH="$FQ_SOURCE_ROOT" exec "$FQ_REAL_PYTHON" "$@"
fi
exit 1
"""


@pytest.fixture
def eval_env_data(campaign_env):
    eval_root = campaign_env["home"] / "unitree_rl_mjlab_eval"
    manifest_path = eval_root / "fq4x8" / "accepted_training_checkpoints.tsv"
    rows = valid_training_rows(code_rev=campaign_env["code_rev"], asset_rev=campaign_env["asset_rev"])
    _write_manifest(manifest_path, rows)
    _write_executable(campaign_env["repo_root"] / ".venv" / "bin" / "python", FQ_PYTHON_WRAPPER)
    campaign_env["eval_root"] = eval_root
    campaign_env["manifest_path"] = manifest_path
    campaign_env["rows"] = rows
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
            "EXPECTED_TRAINING_CODE_REVISION": env_data["code_rev"],
            "EXPECTED_ASSET_REVISION": env_data["asset_rev"],
            "FQ_SOURCE_ROOT": str(REPO_ROOT),
            "FQ_REAL_PYTHON": sys.executable,
        }
    )
    return _apply_overrides(env, overrides)


def _assert_manifest_content_rejected(result: subprocess.CompletedProcess, stderr_substring: str) -> None:
    assert result.returncode == 2
    assert "EVAL_FAIL: accepted-training manifest validation failed" in result.stdout
    assert "MANIFEST_FAIL" in result.stderr
    assert stderr_substring in result.stderr


def test_eval_accepts_valid_32_row_manifest(eval_env_data):
    """The central positive case: a fully valid canonical 32-row manifest
    must be ACCEPTED by the real Python validator, then fall through to the
    pre-existing CUDA guard (never reaching real evaluation)."""
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    out = result.stdout + result.stderr
    assert "EVAL_FAIL" not in out
    assert "MANIFEST_FAIL" not in out
    assert "### EVAL_PROVENANCE" in out
    # The guard passed; it must now fail for an unrelated, pre-existing reason
    # (no CUDA on this machine) -- never reaching real evaluation.
    assert result.returncode == 1
    assert "### NO CUDA" in out


def test_eval_rejects_wrong_manifest_row_count(eval_env_data):
    rows = valid_training_rows(
        code_rev=eval_env_data["code_rev"], asset_rev=eval_env_data["asset_rev"]
    )[:-1]  # 31 rows
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "32 rows")


def test_eval_rejects_task_short_mapping_violation(eval_env_data):
    rows = _mutate(eval_env_data["rows"], 0, task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-BOGUS")
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "task")


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


def test_eval_rejects_symlinked_eval_root_with_duplicated_fq4x8_component(eval_env_data):
    """MEDIUM-4: EVAL_ROOT is checked literally against the fixed expected
    path, but a symlink at that exact path can resolve to a physical
    directory that already ends in "fq4x8" -- the literal check alone cannot
    see through the symlink, so the appended campaign segment would be
    duplicated (.../fq4x8/fq4x8/attempt1)."""
    import shutil

    physical_target = eval_env_data["home"].parent / "physical-eval-root" / "fq4x8"
    physical_target.mkdir(parents=True)
    symlinked_root = eval_env_data["home"] / "unitree_rl_mjlab_eval"
    shutil.rmtree(symlinked_root)  # eval_env_data pre-creates this as a real directory
    symlinked_root.symlink_to(physical_target)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: resolved EVAL_ROOT must not already contain an fq4x8 path component"
        in result.stdout
    )


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


def test_eval_rejects_missing_training_code_revision(eval_env_data):
    env = _eval_env(eval_env_data, EXPECTED_TRAINING_CODE_REVISION=None)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: EXPECTED_TRAINING_CODE_REVISION must be a clean 40-hex revision"
        in result.stdout
    )


def test_eval_rejects_dirty_training_code_revision(eval_env_data):
    env = _eval_env(
        eval_env_data,
        EXPECTED_TRAINING_CODE_REVISION=eval_env_data["code_rev"] + "-dirty",
    )
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: EXPECTED_TRAINING_CODE_REVISION must be a clean 40-hex revision"
        in result.stdout
    )


def test_eval_rejects_short_training_code_revision(eval_env_data):
    env = _eval_env(
        eval_env_data,
        EXPECTED_TRAINING_CODE_REVISION=eval_env_data["code_rev"][:10],
    )
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: EXPECTED_TRAINING_CODE_REVISION must be a clean 40-hex revision"
        in result.stdout
    )


def test_eval_rejects_nonhex_training_code_revision(eval_env_data):
    env = _eval_env(eval_env_data, EXPECTED_TRAINING_CODE_REVISION="g" * 40)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert (
        "EVAL_FAIL: EXPECTED_TRAINING_CODE_REVISION must be a clean 40-hex revision"
        in result.stdout
    )


def test_eval_rejects_missing_asset_revision(eval_env_data):
    env = _eval_env(eval_env_data, EXPECTED_ASSET_REVISION=None)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EXPECTED_ASSET_REVISION must be a clean 40-hex revision" in result.stdout


def test_eval_rejects_dirty_asset_revision(eval_env_data):
    env = _eval_env(
        eval_env_data, EXPECTED_ASSET_REVISION=eval_env_data["asset_rev"] + "-dirty"
    )
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EXPECTED_ASSET_REVISION must be a clean 40-hex revision" in result.stdout


def test_eval_rejects_short_asset_revision(eval_env_data):
    env = _eval_env(eval_env_data, EXPECTED_ASSET_REVISION=eval_env_data["asset_rev"][:10])
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EXPECTED_ASSET_REVISION must be a clean 40-hex revision" in result.stdout


def test_eval_rejects_wrong_code_revision_equality(eval_env_data):
    """Well-formed 40-hex, but simply the wrong commit -- distinct from the
    format checks above."""
    other = _git_repo(eval_env_data["home"].parent / "other-code-repo")
    env = _eval_env(eval_env_data, EXPECTED_CODE_REVISION=other)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: code revision does not equal expected code revision" in result.stdout


def test_eval_rejects_wrong_asset_revision_equality(eval_env_data):
    other = _git_repo(eval_env_data["home"].parent / "other-asset-repo")
    env = _eval_env(eval_env_data, EXPECTED_ASSET_REVISION=other)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: asset revision does not equal expected asset revision" in result.stdout


def test_eval_accepts_training_manifest_whose_code_revision_differs_from_eval_checkout(
    eval_env_data,
):
    """Provenance over-constraint fix (root regression test): the training
    manifest's code_revision is bound to EXPECTED_TRAINING_CODE_REVISION, not
    to the evaluation checkout's own EXPECTED_CODE_REVISION -- a persistence/
    provenance-only fix can land in the evaluation checkout after training
    froze. The asset revision is unaffected and still bound to the single
    EXPECTED_ASSET_REVISION for both training and evaluation."""
    frozen_training_revision = "9" * 40
    rows = valid_training_rows(
        code_rev=frozen_training_revision, asset_rev=eval_env_data["asset_rev"]
    )
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(
        eval_env_data,
        EXPECTED_TRAINING_CODE_REVISION=frozen_training_revision,
    )
    result = _run(EVAL_SCRIPT, env)
    out = result.stdout + result.stderr
    assert "EVAL_FAIL" not in out
    assert "MANIFEST_FAIL" not in out
    assert "### EVAL_PROVENANCE" in out
    # The guard passed; it must now fail for an unrelated, pre-existing
    # reason (no CUDA on this machine) -- never reaching real evaluation.
    assert result.returncode == 1
    assert "### NO CUDA" in out


def test_eval_rejects_actual_dirty_code_repo(eval_env_data):
    (eval_env_data["repo_root"] / "untracked.txt").write_text("dirty")
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: code provenance is unknown or dirty" in result.stdout


def test_eval_rejects_actual_dirty_asset_repo(eval_env_data):
    (eval_env_data["sibling"] / "untracked.txt").write_text("dirty")
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: asset provenance is unknown or dirty" in result.stdout


@pytest.mark.parametrize("bad_attempt", ["", ".", "..", "foo/bar", "attempt one"])
def test_eval_rejects_unsafe_eval_attempt(eval_env_data, bad_attempt):
    env = _eval_env(eval_env_data, EVAL_ATTEMPT=bad_attempt if bad_attempt else None)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: EVAL_ATTEMPT must be a nonempty path-safe attempt label" in result.stdout


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
# Accepted-training manifest content (HIGH-1: the frozen 32-row identity)
#
# The old hand-rolled awk validator only checked row count, per-arm count,
# and the task/short mapping -- it did not enforce seed uniqueness, checkpoint
# identity, or any of the frozen reward/action/impedance/cap signatures. Every
# check below now runs through the real Python validator
# (evaluation.analysis.fq4x8_manifests.validate_training_manifest).
# --------------------------------------------------------------------------


def test_eval_rejects_manifest_row_with_wrong_column_count(eval_env_data):
    """A row with fewer columns than the canonical 21-field header (e.g.
    checkpoint_path/checkpoint_sha256 truncated off) must fail closed."""
    text = manifests.serialize_training_manifest(eval_env_data["rows"])
    lines = text.rstrip("\n").split("\n")
    header, first_row, *rest = lines
    truncated = "\t".join(first_row.split("\t")[:-2])
    eval_env_data["manifest_path"].write_text(
        "\n".join([header, truncated, *rest]) + "\n"
    )
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "expected 21 columns")


def test_eval_rejects_training_seed_out_of_range(eval_env_data):
    rows = _mutate(eval_env_data["rows"], 0, training_seed=7)
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "8..15")


def test_eval_rejects_missing_seed(eval_env_data):
    """32 rows, right shape, but one arm is missing seed 11 (displaced by a
    duplicate of a D0/seed8 row). Seeds are a closed 8-element set per arm, so
    any "missing seed, still 32 rows" shape necessarily introduces a
    duplicate (arm, seed) somewhere else -- the duplicate check and the exact-
    seed-set check are the same underlying violation here; either is a
    legitimate rejection."""
    rows = [
        row
        for row in eval_env_data["rows"]
        if not (row["arm"] == "F8" and row["training_seed"] == 11)
    ]
    rows.append(
        _training_row(
            "D0", 8, code_rev=eval_env_data["code_rev"], asset_rev=eval_env_data["asset_rev"]
        )
    )
    assert len(rows) == 32
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: accepted-training manifest validation failed" in result.stdout
    assert "MANIFEST_FAIL" in result.stderr


def test_eval_rejects_uneven_per_arm_row_count(eval_env_data):
    """32 rows total, but F8 holds 9 (one duplicated) and F0 holds 7."""
    rows = [
        row
        for row in eval_env_data["rows"]
        if not (row["arm"] == "F0" and row["training_seed"] == 8)
    ]
    rows.append(
        _training_row(
            "F8", 8, code_rev=eval_env_data["code_rev"], asset_rev=eval_env_data["asset_rev"]
        )
    )
    assert len(rows) == 32
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: accepted-training manifest validation failed" in result.stdout
    assert "MANIFEST_FAIL" in result.stderr


def test_eval_rejects_duplicate_arm_seed(eval_env_data):
    """Row 1 becomes a byte-identical copy of row 0 -- 32 rows, but only 31
    distinct (arm, seed) identities."""
    rows = _mutate(eval_env_data["rows"], 1, **eval_env_data["rows"][0])
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "duplicate")


def test_eval_rejects_eight_copies_of_seed_8_per_arm(eval_env_data):
    """Reproduces the exact independent-review HIGH-1 attack: eight duplicate
    seed-8 rows per arm (32 rows total, right count, wrong content). This
    manifest passed the old hand-rolled awk validator and reached the CUDA
    guard."""
    rows = [
        _training_row(
            arm, 8, code_rev=eval_env_data["code_rev"], asset_rev=eval_env_data["asset_rev"]
        )
        for arm in manifests.LABELS
        for _ in manifests.SEEDS
    ]
    assert len(rows) == 32
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "duplicate")


def test_eval_rejects_the_original_review_attack_manifest(eval_env_data):
    """The literal review reproduction: eight duplicate seed-8 rows per arm,
    every checkpoint hash malformed, every training_attempt 'bogus' -- 32
    rows, right shape, wrong content throughout."""
    rows = []
    for arm in manifests.LABELS:
        for _ in manifests.SEEDS:
            row = _training_row(
                arm, 8, code_rev=eval_env_data["code_rev"], asset_rev=eval_env_data["asset_rev"]
            )
            row["training_attempt"] = "bogus"
            row["retry_history"] = "bogus:accepted"
            row["checkpoint_sha256"] = "not-a-sha"
            rows.append(row)
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    assert result.returncode == 2
    assert "EVAL_FAIL: accepted-training manifest validation failed" in result.stdout
    assert "MANIFEST_FAIL" in result.stderr


def test_eval_rejects_duplicate_checkpoint_path(eval_env_data):
    rows = _mutate(
        eval_env_data["rows"], 1, checkpoint_path=eval_env_data["rows"][0]["checkpoint_path"]
    )
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "checkpoint_path is duplicated")


def test_eval_rejects_non_model_499_checkpoint_basename(eval_env_data):
    rows = _mutate(
        eval_env_data["rows"], 0, checkpoint_path="/fake/checkpoints/F8/seed8/model_250.pt"
    )
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "model_499.pt")


def test_eval_rejects_malformed_checkpoint_sha256(eval_env_data):
    rows = _mutate(eval_env_data["rows"], 0, checkpoint_sha256="z" * 64)
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "checkpoint_sha256")


def test_eval_rejects_wrong_campaign_identity(eval_env_data):
    rows = _mutate(eval_env_data["rows"], 0, campaign="fsr4x8")
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "campaign")


def test_eval_rejects_non_accepted_disposition(eval_env_data):
    rows = _mutate(eval_env_data["rows"], 0, disposition="pending")
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "disposition")


@pytest.mark.parametrize(
    "field",
    [
        "treatment_config_sha256",
        "reader_sha256",
        "normalizer_sha256",
        "treatment_reward_sha256",
        "fixed_action_signature_sha256",
        "fixed_impedance_signature_sha256",
        "cap_signature_sha256",
    ],
)
def test_eval_rejects_each_wrong_frozen_signature(eval_env_data, field):
    rows = _mutate(eval_env_data["rows"], 0, **{field: _hex(f"wrong-{field}")})
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, field)


def test_eval_rejects_code_revision_mismatch_within_manifest_row(eval_env_data):
    """Every row's code_revision must equal the actual accepted revision --
    not just be well-formed 40-hex."""
    rows = _mutate(eval_env_data["rows"], 0, code_revision=_hex("other-code-revision", 40))
    _write_manifest(eval_env_data["manifest_path"], rows)
    env = _eval_env(eval_env_data)
    result = _run(EVAL_SCRIPT, env)
    _assert_manifest_content_rejected(result, "code_revision")
