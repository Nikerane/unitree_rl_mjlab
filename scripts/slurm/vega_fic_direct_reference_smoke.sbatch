#!/usr/bin/env bash
#SBATCH --job-name=z1-fic-direct-smoke
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/z1-fic-direct-smoke-%j.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/z1-fic-direct-smoke-%j.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "Z1_DIRECT_SMOKE_FAIL: $*"
  exit 2
}

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"

for REVISION_NAME in EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION; do
  REVISION_VALUE="${!REVISION_NAME:-}"
  [[ "$REVISION_VALUE" =~ ^[0-9a-f]{40}$ ]] \
    || fail "$REVISION_NAME must be a 40-hex revision"
done

[[ -n "${RUN_ROOT:-}" ]] || fail "RUN_ROOT is required"
[[ -n "${ASSET_REPO:-}" ]] || fail "ASSET_REPO is required"
RESOLVED_RUN_ROOT=$(cd "$RUN_ROOT" 2>/dev/null && pwd -P) \
  || fail "code repository is unavailable"
CANONICAL_ASSET_REPO=$(cd "$RESOLVED_RUN_ROOT/../safe_impact_manipulation" 2>/dev/null && pwd -P) \
  || fail "canonical sibling asset repository is unavailable"
RESOLVED_ASSET_REPO=$(cd "$ASSET_REPO" 2>/dev/null && pwd -P) \
  || fail "asset repository is unavailable"
[[ "$RESOLVED_ASSET_REPO" = "$CANONICAL_ASSET_REPO" ]] \
  || fail "ASSET_REPO must be the canonical sibling"
RUN_ROOT="$RESOLVED_RUN_ROOT"
ASSET_REPO="$RESOLVED_ASSET_REPO"

CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) \
  || fail "code repository is unavailable"
[[ "$CODE_REVISION" = "$EXPECTED_CODE_REVISION" ]] \
  || fail "code revision mismatch"
[[ -z "$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all 2>/dev/null)" ]] \
  || fail "code repository must be clean"

ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) \
  || fail "asset repository is unavailable"
[[ "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] \
  || fail "asset revision mismatch"
[[ -z "$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all 2>/dev/null)" ]] \
  || fail "asset repository must be clean"

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] \
  || fail "SLURM_JOB_ID must be numeric"

PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"
[[ -x "$PY" ]] || fail "Vega Python environment is unavailable"

RESULT_ROOT="$HOME/z1-fic-direct-reference-smoke/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
ATTEMPT_DIR="$RESULT_ROOT/${SLURM_JOB_ID}_smoke"
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null \
  || fail "attempt directory already exists"

export PYTHONPATH="$RUN_ROOT"
export WARP_CACHE_PATH="$ATTEMPT_DIR/warp-cache"
mkdir "$WARP_CACHE_PATH"

"$PY" - <<'PY' || fail "CUDA identity or runtime package versions do not match"
from importlib.metadata import version

import torch


expected = {"mjlab": "1.4.0", "mujoco": "3.8.1", "mujoco-warp": "3.8.1"}
actual = {package: version(package) for package in expected}
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0)
assert gpu_name == "NVIDIA A100-SXM4-40GB", gpu_name
assert actual == expected, actual
print(f"Z1_DIRECT_CUDA_GPU={gpu_name}")
print("Z1_DIRECT_RUNTIME=" + " ".join(f"{key}={value}" for key, value in actual.items()))
PY

printf 'Z1_DIRECT_CODE_REVISION=%s\n' "$CODE_REVISION"
printf 'Z1_DIRECT_ASSET_REVISION=%s\n' "$ASSET_REVISION"

cd "$ATTEMPT_DIR"
FIC0_TASK="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed"
"$PY" "$RUN_ROOT/scripts/smoke_joint_position_fixed.py" \
  --task all --device cuda:0
"$PY" "$RUN_ROOT/scripts/smoke_cat_soft.py" \
  --task "$FIC0_TASK" --device cuda:0 --iters 1
"$PY" "$RUN_ROOT/scripts/smoke_cat_soft.py" \
  --task "${FIC0_TASK}-TT" --device cuda:0 --iters 1

printf 'FIC_DIRECT_REFERENCE_SMOKE_DONE\n'
