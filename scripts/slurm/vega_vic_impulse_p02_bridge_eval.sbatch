#!/usr/bin/env bash
# Frozen no-learning evaluation of the matched p=0 versus p=0.2 impulse-CaT bridge pair.
# Evaluation runs with live imp_max_p=0 and does not establish hardware safety.
#SBATCH --job-name=z1-vic-imp-p02-eval
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-p02-bridge/slurm/z1-vic-impulse-p02-eval-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-p02-bridge/slurm/z1-vic-impulse-p02-eval-%A_%a.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "Z1_VIC_IMPULSE_P02_BRIDGE_EVAL_FAIL: $*"
  exit 2
}

readonly APPROVED_EVALUATOR_BASE_REVISION=7d0761e3dd156012a22378fd0b269e4c50ea5ae6
readonly PINNED_ASSET_REVISION=b58ccd2f81fd246f27c1e8d88cf86484cd888703
readonly CONTROL_CHECKPOINT_SHA256=f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3
readonly TARGET_CHECKPOINT_SHA256=57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de
readonly PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"

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
for VARIABLE in RUN_ROOT ASSET_REPO CONTROL_CHECKPOINT TARGET_CHECKPOINT; do
  [[ -n "${!VARIABLE:-}" ]] || fail "$VARIABLE is required"
done
[[ "$EXPECTED_ASSET_REVISION" = "$PINNED_ASSET_REVISION" ]] \
  || fail "asset revision must match the frozen evaluation asset revision"
[[ -x "$PY" ]] || fail "Vega Python environment is unavailable"

RUN_ROOT=$(cd "$RUN_ROOT" 2>/dev/null && pwd -P) || fail "code repository is unavailable"
CANONICAL_ASSET_REPO=$(cd "$RUN_ROOT/../safe_impact_manipulation" 2>/dev/null && pwd -P) \
  || fail "canonical sibling asset repository is unavailable"
ASSET_REPO=$(cd "$ASSET_REPO" 2>/dev/null && pwd -P) \
  || fail "asset repository is unavailable"
[[ "$ASSET_REPO" = "$CANONICAL_ASSET_REPO" ]] || fail "ASSET_REPO must be the canonical sibling"

CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) \
  || fail "code repository revision is unavailable"
[[ "$CODE_REVISION" = "$EXPECTED_CODE_REVISION" ]] || fail "code revision mismatch"
git -C "$RUN_ROOT" cat-file -e "$EXPECTED_CODE_REVISION^{commit}" 2>/dev/null \
  || fail "expected code revision is not a Git commit"
git -C "$RUN_ROOT" cat-file -e "$APPROVED_EVALUATOR_BASE_REVISION^{commit}" 2>/dev/null \
  || fail "approved evaluator base revision is unavailable"
git -C "$RUN_ROOT" merge-base --is-ancestor "$APPROVED_EVALUATOR_BASE_REVISION" \
  "$EXPECTED_CODE_REVISION" 2>/dev/null \
  || fail "code revision is not descended from the approved evaluator base"
CODE_STATUS=$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all) \
  || fail "code repository status check failed"
[[ -z "$CODE_STATUS" ]] || fail "code repository must be clean"
ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) \
  || fail "asset repository revision is unavailable"
[[ "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] || fail "asset revision mismatch"
ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all) \
  || fail "asset repository status check failed"
[[ -z "$ASSET_STATUS" ]] || fail "asset repository must be clean"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    readonly ROLE=bridge_p0_control
    readonly CHECKPOINT="$CONTROL_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$CONTROL_CHECKPOINT_SHA256"
    ;;
  1)
    readonly ROLE=bridge_p02_target
    readonly CHECKPOINT="$TARGET_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$TARGET_CHECKPOINT_SHA256"
    ;;
esac
[[ -f "$CHECKPOINT" && ! -L "$CHECKPOINT" && "$(basename "$CHECKPOINT")" = model_499.pt ]] \
  || fail "checkpoint must be a regular non-symlink model_499.pt"
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)
[[ "$CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint SHA-256 is invalid"
[[ "$CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256" ]] \
  || fail "role/checkpoint SHA-256 mismatch"

readonly RESULT_ROOT="$HOME/campaigns/z1-vic-impulse-p02-bridge/evaluation/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
readonly ATTEMPT_DIR="$RESULT_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${ROLE}"
readonly EVALUATION_DIR="$ATTEMPT_DIR/evaluation"
readonly NODE_TMPDIR="/tmp/z1-vic-impulse-p02-eval-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -m 700 -- "$NODE_TMPDIR" 2>/dev/null || fail "node-local temporary directory already exists"
cleanup_node_tmpdir() { rm -rf -- "$NODE_TMPDIR"; }
trap cleanup_node_tmpdir EXIT
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null || fail "attempt directory already exists"
[[ ! -e "$EVALUATION_DIR" && ! -L "$EVALUATION_DIR" ]] \
  || fail "evaluation output directory already exists"

export PYTHONPATH="$RUN_ROOT"
export WARP_CACHE_PATH="$NODE_TMPDIR/warp-cache"
export TMPDIR="$NODE_TMPDIR"
export TEMP="$NODE_TMPDIR"
export TMP="$NODE_TMPDIR"
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
print(f"Z1_VIC_IMPULSE_P02_BRIDGE_EVAL_GPU={gpu}")
print(f"Z1_VIC_IMPULSE_P02_BRIDGE_EVAL_RUNTIME={actual}")
PY

"$PY" "$RUN_ROOT/scripts/impulse_cat_activation_survey.py" \
  --checkpoint-role "$ROLE" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EVALUATION_DIR" \
  --device cuda:0

readonly EXPECTED_OUTPUTS=(
  fixed_trace.npz
  training_like_seed_2_trace.npz
  training_like_seed_2026081701_trace.npz
  training_like_seed_2026081702_trace.npz
  summary.json
)
[[ -d "$EVALUATION_DIR" && ! -L "$EVALUATION_DIR" ]] \
  || fail "evaluator did not create a regular output directory"
shopt -s nullglob dotglob
ACTUAL_OUTPUTS=("$EVALUATION_DIR"/*)
[[ "${#ACTUAL_OUTPUTS[@]}" -eq "${#EXPECTED_OUTPUTS[@]}" ]] \
  || fail "evaluator output set does not contain exactly five artifacts"
for OUTPUT in "${EXPECTED_OUTPUTS[@]}"; do
  [[ -f "$EVALUATION_DIR/$OUTPUT" && ! -L "$EVALUATION_DIR/$OUTPUT" ]] \
    || fail "expected evaluation artifact is missing or not a regular file: $OUTPUT"
done
for OUTPUT in "${ACTUAL_OUTPUTS[@]}"; do
  [[ -f "$OUTPUT" && ! -L "$OUTPUT" ]] \
    || fail "evaluator output set contains a non-regular artifact"
done

FINAL_CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)
[[ "$FINAL_CHECKPOINT_SHA256" = "$CHECKPOINT_SHA256" ]] \
  || fail "checkpoint SHA-256 drifted during evaluation"
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

(
  cd "$ATTEMPT_DIR"
  sha256sum \
    evaluation/fixed_trace.npz \
    evaluation/training_like_seed_2_trace.npz \
    evaluation/training_like_seed_2026081701_trace.npz \
    evaluation/training_like_seed_2026081702_trace.npz \
    evaluation/summary.json > SHA256SUMS
)
[[ "$(wc -l < "$ATTEMPT_DIR/SHA256SUMS" | tr -d ' ')" = 5 ]] \
  || fail "SHA256 manifest does not contain five artifacts"
printf 'Z1_VIC_IMPULSE_P02_BRIDGE_EVAL_PASS role=%s checkpoint_sha256=%s evaluation_dir=%s\n' \
  "$ROLE" "$CHECKPOINT_SHA256" "$EVALUATION_DIR"
