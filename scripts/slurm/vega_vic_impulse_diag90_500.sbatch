#!/usr/bin/env bash
# Matched 500-iteration engineering pair at the explicitly diagnostic 0.9 impulse boundary.
# This is not a hardware-limit experiment and cannot by itself establish learned safety.
#SBATCH --job-name=z1-vic-imp-d90-500
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-diag90-500/slurm/z1-vic-impulse-diag90-500-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-diag90-500/slurm/z1-vic-impulse-diag90-500-%A_%a.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "Z1_VIC_IMPULSE_DIAG90_500_FAIL: $*"
  exit 2
}

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE must be empty"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" = 2 ]] || fail "array task count must be exactly 2"
[[ "${SLURM_ARRAY_TASK_MIN:-}" = 0 ]] || fail "array minimum must be exactly 0"
[[ "${SLURM_ARRAY_TASK_MAX:-}" = 1 ]] || fail "array maximum must be exactly 1"
[[ "${SLURM_ARRAY_TASK_STEP:-}" = 1 ]] || fail "array step must be exactly 1"
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "array job id must be numeric"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]] || fail "array task id must be 0 or 1"
for VARIABLE in WORLD_SIZE RANK LOCAL_RANK RESUME_PATH; do
  [[ -z "${!VARIABLE:-}" ]] || fail "$VARIABLE must be empty"
done
for VARIABLE in EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION; do
  [[ "${!VARIABLE:-}" =~ ^[0-9a-f]{40}$ ]] || fail "$VARIABLE must be a 40-hex revision"
done
[[ -n "${RUN_ROOT:-}" ]] || fail "RUN_ROOT is required"
[[ -n "${ASSET_REPO:-}" ]] || fail "ASSET_REPO is required"

RUN_ROOT=$(cd "$RUN_ROOT" 2>/dev/null && pwd -P) || fail "code repository is unavailable"
CANONICAL_ASSET_REPO=$(cd "$RUN_ROOT/../safe_impact_manipulation" 2>/dev/null && pwd -P) \
  || fail "canonical sibling asset repository is unavailable"
ASSET_REPO=$(cd "$ASSET_REPO" 2>/dev/null && pwd -P) \
  || fail "asset repository is unavailable"
[[ "$ASSET_REPO" = "$CANONICAL_ASSET_REPO" ]] || fail "ASSET_REPO must be the canonical sibling"

CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) \
  || fail "code repository revision is unavailable"
[[ "$CODE_REVISION" = "$EXPECTED_CODE_REVISION" ]] || fail "code revision mismatch"
CODE_STATUS=$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all) \
  || fail "code repository status check failed"
[[ -z "$CODE_STATUS" ]] || fail "code repository must be clean"
ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) \
  || fail "asset repository revision is unavailable"
[[ "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] || fail "asset revision mismatch"
ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all) \
  || fail "asset repository status check failed"
[[ -z "$ASSET_STATUS" ]] || fail "asset repository must be clean"

readonly TASK="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT"
readonly CAPS="[0.738,1.476,0.738,0.738,0.738,0.738]"
case "$SLURM_ARRAY_TASK_ID" in
  0) readonly ARM="imp0_control"; readonly IMP_MAX_P="0.0" ;;
  1) readonly ARM="imp05_target"; readonly IMP_MAX_P="0.5" ;;
esac

readonly PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"
[[ -x "$PY" ]] || fail "Vega Python environment is unavailable"
readonly RESULT_ROOT="$HOME/campaigns/z1-vic-impulse-diag90-500/runs/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
readonly ATTEMPT_DIR="$RESULT_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_victt_seed2_diag90_500_${ARM}"
readonly NODE_TMPDIR="/tmp/z1-vic-impulse-diag90-500-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -m 700 -- "$NODE_TMPDIR" 2>/dev/null || fail "node-local temporary directory already exists"
cleanup_node_tmpdir() { rm -rf -- "$NODE_TMPDIR"; }
trap cleanup_node_tmpdir EXIT
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null || fail "attempt directory already exists"

export PYTHONPATH="$RUN_ROOT"
export WARP_CACHE_PATH="$ATTEMPT_DIR/warp-cache"
export TMPDIR="$NODE_TMPDIR"
export TEMP="$NODE_TMPDIR"
export TMP="$NODE_TMPDIR"
export RUN_NAME="victt_seed2_diag90_500_${ARM}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir "$WARP_CACHE_PATH"

"$PY" - <<'PY' || fail "CUDA identity or runtime package versions do not match"
from importlib.metadata import version
import torch

expected = {"mjlab": "1.4.0", "mujoco": "3.8.1", "mujoco-warp": "3.8.1"}
actual = {name: version(name) for name in expected}
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
  raise RuntimeError("exactly one CUDA device is required")
gpu = torch.cuda.get_device_name(0)
if gpu != "NVIDIA A100-SXM4-40GB":
  raise RuntimeError(f"unexpected GPU: {gpu}")
if actual != expected:
  raise RuntimeError(f"runtime package versions do not match: {actual!r}")
print(f"Z1_VIC_IMPULSE_DIAG90_500_GPU={gpu}")
print(f"Z1_VIC_IMPULSE_DIAG90_500_RUNTIME={actual}")
PY

cd "$ATTEMPT_DIR"
"$PY" "$RUN_ROOT/scripts/train.py" "$TASK" \
  --gpu-ids '[0]' \
  --agent.logger tensorboard \
  --agent.run-name "$RUN_NAME" \
  --agent.seed 2 \
  --agent.num-steps-per-env 24 \
  --agent.max-iterations 500 \
  --agent.save-interval 50 \
  --env.scene.num-envs 4096 \
  --env.metrics.cat-soft.params.imp-limit "$CAPS" \
  --env.metrics.cat-soft.params.imp-max-p "$IMP_MAX_P"

shopt -s nullglob
RUN_DIRS=("$ATTEMPT_DIR"/logs/rsl_rl/z1_hammer/*_"$RUN_NAME")
[[ "${#RUN_DIRS[@]}" -eq 1 && -d "${RUN_DIRS[0]}" ]] \
  || fail "expected exactly one matching training run directory"
RUN_DIR="${RUN_DIRS[0]}"
readonly EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)
EXPECTED_CHECKPOINTS=()
for ITERATION in "${EXPECTED_ITERATIONS[@]}"; do
  EXPECTED_CHECKPOINTS+=("$RUN_DIR/model_${ITERATION}.pt")
done
ACTUAL_CHECKPOINTS=("$RUN_DIR"/model_*.pt)
[[ "${#ACTUAL_CHECKPOINTS[@]}" -eq "${#EXPECTED_CHECKPOINTS[@]}" ]] \
  || fail "checkpoint set does not match the exact 500-iteration schedule"
for CHECKPOINT in "${EXPECTED_CHECKPOINTS[@]}"; do
  [[ -f "$CHECKPOINT" ]] || fail "checkpoint set does not match the exact 500-iteration schedule"
done

"$PY" - "${EXPECTED_CHECKPOINTS[@]}" <<'PY' \
  || fail "checkpoint identity or finiteness validation failed"
from collections.abc import Mapping
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

expected_iterations = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499)
if len(sys.argv[1:]) != len(expected_iterations):
  raise RuntimeError("checkpoint argument count mismatch")
for expected_iteration, path in zip(expected_iterations, sys.argv[1:], strict=True):
  state = torch.load(path, map_location="cpu", weights_only=False)
  if type(state.get("iter")) is not int or state["iter"] != expected_iteration:
    raise RuntimeError(f"checkpoint iteration mismatch: {path}")
  values = list(tensors(state))
  if not values or not all(bool(torch.isfinite(value).all()) for value in values):
    raise RuntimeError(f"checkpoint contains no finite tensor state: {path}")
PY

FINAL_CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) \
  || fail "postflight code revision is unavailable"
FINAL_CODE_STATUS=$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all) \
  || fail "postflight code status check failed"
FINAL_ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) \
  || fail "postflight asset revision is unavailable"
FINAL_ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all) \
  || fail "postflight asset status check failed"
[[ "$FINAL_CODE_REVISION" = "$EXPECTED_CODE_REVISION" && -z "$FINAL_CODE_STATUS" ]] \
  || fail "postflight code provenance drifted"
[[ "$FINAL_ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" && -z "$FINAL_ASSET_STATUS" ]] \
  || fail "postflight asset provenance drifted"

for CHECKPOINT in "${EXPECTED_CHECKPOINTS[@]}"; do
  printf 'Z1_VIC_IMPULSE_DIAG90_500_CHECKPOINT arm=%s imp_max_p=%s basename=%s sha256=%s\n' \
    "$ARM" "$IMP_MAX_P" "$(basename "$CHECKPOINT")" "$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)"
done
printf 'Z1_VIC_IMPULSE_DIAG90_500_PASS arm=%s imp_max_p=%s caps=%s run_dir=%s\n' \
  "$ARM" "$IMP_MAX_P" "$CAPS" "$RUN_DIR"
