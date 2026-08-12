#!/usr/bin/env bash
#SBATCH --job-name=z1-fic-pilot
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot-%A_%a.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "FIC_PILOT_FAIL: $*"
  exit 2
}

[[ "${SLURM_ARRAY_TASK_COUNT:-}" = "2" \
  && "${SLURM_ARRAY_TASK_MIN:-}" = "0" \
  && "${SLURM_ARRAY_TASK_MAX:-}" = "1" \
  && "${SLURM_ARRAY_TASK_STEP:-}" = "1" ]] \
  || fail "array must be exactly 0-1"

FIC0_TASK="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed"
case "${SLURM_ARRAY_TASK_ID:?must run as a two-element Slurm array}" in
  0)
    TASK="$FIC0_TASK"
    SHORT="fic0"
    ;;
  1)
    TASK="${FIC0_TASK}-TT"
    SHORT="fictt"
    ;;
  *)
    fail "array index must be exactly 0 or 1"
    ;;
esac

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

CODE_REVISION=$(git -C "$RESOLVED_RUN_ROOT" rev-parse HEAD 2>/dev/null) \
  || fail "code repository is unavailable"
[[ "$CODE_REVISION" = "$EXPECTED_CODE_REVISION" ]] \
  || fail "code revision mismatch"
[[ -z "$(git -C "$RESOLVED_RUN_ROOT" status --porcelain=v1 --untracked-files=all 2>/dev/null)" ]] \
  || fail "code repository must be clean"

ASSET_REVISION=$(git -C "$RESOLVED_ASSET_REPO" rev-parse HEAD 2>/dev/null) \
  || fail "asset repository is unavailable"
[[ "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] \
  || fail "asset revision mismatch"
[[ -z "$(git -C "$RESOLVED_ASSET_REPO" status --porcelain=v1 --untracked-files=all 2>/dev/null)" ]] \
  || fail "asset repository must be clean"

[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] \
  || fail "SLURM_ARRAY_JOB_ID must be numeric"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "SLURM_JOB_ID is required"

RESULT_ROOT="$HOME/z1-fic-joint-pilot/$EXPECTED_CODE_REVISION"
ATTEMPT_DIR="$RESULT_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SHORT}_seed2"
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null \
  || fail "attempt directory already exists"

RUN_NAME="fic_pilot_${SHORT}_seed2_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"
[[ -x "$PY" ]] || fail "Vega Python environment is unavailable"

export PYTHONPATH="$RUN_ROOT"
export WARP_CACHE_PATH="$ATTEMPT_DIR/warp-cache"
export RUN_NAME
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
print(f"FIC_PILOT_GPU={gpu_name}")
print(f"FIC_PILOT_VERSIONS={actual}")
assert actual == expected, actual
PY

cd "$ATTEMPT_DIR"

"$PY" "$RUN_ROOT/scripts/train.py" "$TASK" \
  --gpu-ids '[0]' \
  --agent.logger tensorboard \
  --agent.run-name "$RUN_NAME" \
  --agent.seed 2 \
  --agent.max-iterations 200 \
  --agent.save-interval 50 \
  --env.scene.num-envs 4096

shopt -s nullglob
RUN_DIRS=("$ATTEMPT_DIR"/logs/rsl_rl/z1_hammer/*_"$RUN_NAME")
[[ "${#RUN_DIRS[@]}" -eq 1 ]] \
  || fail "expected exactly one matching training run directory"
CHECKPOINT="${RUN_DIRS[0]}/model_199.pt"
[[ -f "$CHECKPOINT" ]] || fail "model_199.pt is missing"

"$PY" - "$CHECKPOINT" <<'PY' || fail "model_199.pt contains no finite checkpoint state"
from collections.abc import Mapping
from pathlib import Path
import sys

import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


checkpoint = Path(sys.argv[1])
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
values = list(tensors(state))
assert values, "checkpoint contains no tensors"
assert all(bool(torch.isfinite(value).all()) for value in values), "non-finite checkpoint tensor"
PY

EVALUATOR="$RUN_ROOT/evaluation/joint_position/evaluate_fic_pilot.py"
[[ -f "$EVALUATOR" ]] || fail "FIC pilot evaluator is missing"
EVALUATION_JSON="$ATTEMPT_DIR/$SHORT.json"
[[ ! -e "$EVALUATION_JSON" ]] || fail "evaluation output already exists"
"$PY" "$EVALUATOR" "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output "$EVALUATION_JSON" \
  --device cuda:0
[[ -s "$EVALUATION_JSON" ]] || fail "evaluator did not create a non-empty JSON result"

CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)
EVALUATION_SHA256=$(sha256sum "$EVALUATION_JSON" | cut -d ' ' -f 1)
printf 'FIC_PILOT_CHECKPOINT=%s\n' "$CHECKPOINT"
printf 'FIC_PILOT_CHECKPOINT_SHA256=%s\n' "$CHECKPOINT_SHA256"
printf 'FIC_PILOT_EVALUATION=%s\n' "$EVALUATION_JSON"
printf 'FIC_PILOT_EVALUATION_SHA256=%s\n' "$EVALUATION_SHA256"
printf 'FIC_PILOT_DONE arm=%s seed=2 task=%s\n' "$SHORT" "$TASK"
