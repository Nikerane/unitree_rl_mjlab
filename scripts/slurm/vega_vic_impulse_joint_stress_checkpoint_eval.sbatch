#!/usr/bin/env bash
# Frozen no-learning evaluation of late checkpoints from the seed-2 p=.2 joint-stress policy.
# Every arm evaluates with live impulse CaT log-only against identical populations.
#SBATCH --job-name=z1-vic-imp-j6-checkpoints
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-j6-checkpoint-evaluation/slurm/z1-vic-imp-j6-checkpoints-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-j6-checkpoint-evaluation/slurm/z1-vic-imp-j6-checkpoints-%A_%a.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "Z1_VIC_IMPULSE_JOINT_STRESS_CHECKPOINT_EVAL_FAIL: $*"
  exit 2
}

readonly APPROVED_EVALUATOR_BASE_REVISION=7d0761e3dd156012a22378fd0b269e4c50ea5ae6
readonly PINNED_ASSET_REVISION=b58ccd2f81fd246f27c1e8d88cf86484cd888703
readonly MODEL_300_SHA256=38678071034a79de96b45bcd0ece5b18b80aa6aebd6af1aaf4215f084619cc8d
readonly MODEL_400_SHA256=6ff5ae3e9b197b6255617c53823c935e7348387d0728e6409dac403e905c67ad
readonly MODEL_450_SHA256=a80b4e7b3279e3283581e207d32ce6d6810827978449c92e4289d8a2b50fdfb3
readonly MODEL_499_SHA256=509cc26a2e521a935bcbc8c342040c95c7d2e3d7fc105a52d5d9450518bd2ec7
readonly TRAINING_CAPS="[0.369,0.246,0.738,0.369,0.246,0.0164]"
readonly PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE must be empty"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" = 4 ]] || fail "array task count must be exactly 4"
[[ "${SLURM_ARRAY_TASK_MIN:-}" = 0 ]] || fail "array minimum must be exactly 0"
[[ "${SLURM_ARRAY_TASK_MAX:-}" = 3 ]] || fail "array maximum must be exactly 3"
[[ "${SLURM_ARRAY_TASK_STEP:-}" = 1 ]] || fail "array step must be exactly 1"
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "array job id must be numeric"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || fail "array task id must be 0 through 3"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "SLURM_JOB_ID must be numeric"
for VARIABLE in WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT RESUME_PATH; do
  [[ -z "${!VARIABLE:-}" ]] || fail "$VARIABLE must be empty"
done
for VARIABLE in EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION; do
  [[ "${!VARIABLE:-}" =~ ^[0-9a-f]{40}$ ]] || fail "$VARIABLE must be a 40-hex revision"
done
for VARIABLE in RUN_ROOT ASSET_REPO MODEL_300_CHECKPOINT MODEL_400_CHECKPOINT MODEL_450_CHECKPOINT MODEL_499_CHECKPOINT; do
  [[ -n "${!VARIABLE:-}" ]] || fail "$VARIABLE is required"
done
[[ "$EXPECTED_ASSET_REVISION" = "$PINNED_ASSET_REVISION" ]] \
  || fail "asset revision must match the frozen evaluation asset revision"
[[ -x "$PY" ]] || fail "Vega Python environment is unavailable"

RUN_ROOT=$(cd "$RUN_ROOT" 2>/dev/null && pwd -P) || fail "code repository is unavailable"
CANONICAL_ASSET_REPO=$(cd "$RUN_ROOT/../safe_impact_manipulation" 2>/dev/null && pwd -P) \
  || fail "canonical sibling asset repository is unavailable"
ASSET_REPO=$(cd "$ASSET_REPO" 2>/dev/null && pwd -P) || fail "asset repository is unavailable"
[[ "$ASSET_REPO" = "$CANONICAL_ASSET_REPO" ]] || fail "ASSET_REPO must be the canonical sibling"

CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) || fail "code revision unavailable"
[[ "$CODE_REVISION" = "$EXPECTED_CODE_REVISION" ]] || fail "code revision mismatch"
git -C "$RUN_ROOT" cat-file -e "$EXPECTED_CODE_REVISION^{commit}" 2>/dev/null \
  || fail "expected code revision is not a Git commit"
git -C "$RUN_ROOT" cat-file -e "$APPROVED_EVALUATOR_BASE_REVISION^{commit}" 2>/dev/null \
  || fail "approved evaluator base revision is unavailable"
git -C "$RUN_ROOT" merge-base --is-ancestor "$APPROVED_EVALUATOR_BASE_REVISION" \
  "$EXPECTED_CODE_REVISION" 2>/dev/null || fail "code revision is not descended from approved evaluator"
[[ -z "$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "code repository must be clean"
ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) || fail "asset revision unavailable"
[[ "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] || fail "asset revision mismatch"
[[ -z "$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "asset repository must be clean"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    readonly ROLE=p02_joint_stress_iter300
    readonly CHECKPOINT="$MODEL_300_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$MODEL_300_SHA256"
    readonly EXPECTED_BASENAME=model_300.pt
    ;;
  1)
    readonly ROLE=p02_joint_stress_iter400
    readonly CHECKPOINT="$MODEL_400_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$MODEL_400_SHA256"
    readonly EXPECTED_BASENAME=model_400.pt
    ;;
  2)
    readonly ROLE=p02_joint_stress_iter450
    readonly CHECKPOINT="$MODEL_450_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$MODEL_450_SHA256"
    readonly EXPECTED_BASENAME=model_450.pt
    ;;
  3)
    readonly ROLE=p02_joint_stress_iter499
    readonly CHECKPOINT="$MODEL_499_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$MODEL_499_SHA256"
    readonly EXPECTED_BASENAME=model_499.pt
    ;;
esac
[[ -f "$CHECKPOINT" && ! -L "$CHECKPOINT" && "$(basename "$CHECKPOINT")" = "$EXPECTED_BASENAME" ]] \
  || fail "checkpoint must be the exact regular non-symlink iteration basename"
CHECKPOINT_DIRECTORY=$(cd "$(dirname "$CHECKPOINT")" 2>/dev/null && pwd -P) \
  || fail "checkpoint directory is unavailable"
readonly RESOLVED_CHECKPOINT="$CHECKPOINT_DIRECTORY/$EXPECTED_BASENAME"
CHECKPOINT_SHA256=$(sha256sum "$RESOLVED_CHECKPOINT" 2>/dev/null | cut -d ' ' -f 1) \
  || fail "checkpoint SHA-256 generation failed"
[[ "$CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint SHA-256 is invalid"
[[ "$CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256" ]] || fail "role/checkpoint SHA-256 mismatch"

readonly RESULT_ROOT="$HOME/campaigns/z1-vic-impulse-j6-checkpoint-evaluation/evaluation/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
readonly ATTEMPT_DIR="$RESULT_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${ROLE}"
readonly EVALUATION_DIR="$ATTEMPT_DIR/evaluation"
readonly NODE_TMPDIR="/tmp/z1-vic-imp-j6-checkpoint-eval-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -m 700 -- "$NODE_TMPDIR" 2>/dev/null || fail "node temporary directory already exists"
cleanup_node_tmpdir() { rm -rf -- "$NODE_TMPDIR"; }
trap cleanup_node_tmpdir EXIT
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null || fail "attempt directory already exists"
[[ ! -e "$EVALUATION_DIR" && ! -L "$EVALUATION_DIR" ]] || fail "evaluation output exists"

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
print(f"Z1_VIC_IMPULSE_JOINT_STRESS_CHECKPOINT_EVAL_GPU={gpu}")
print(f"Z1_VIC_IMPULSE_JOINT_STRESS_CHECKPOINT_EVAL_RUNTIME={actual}")
PY

"$PY" "$RUN_ROOT/scripts/impulse_cat_activation_survey.py" \
  --checkpoint-role "$ROLE" \
  --checkpoint "$RESOLVED_CHECKPOINT" \
  --output-dir "$EVALUATION_DIR" \
  --training-caps-n-m-s "$TRAINING_CAPS" \
  --device cuda:0

readonly EXPECTED_OUTPUTS=(
  fixed_trace.npz
  training_like_seed_2_trace.npz
  training_like_seed_2026081701_trace.npz
  training_like_seed_2026081702_trace.npz
  summary.json
)
[[ -d "$EVALUATION_DIR" && ! -L "$EVALUATION_DIR" ]] || fail "evaluator output directory invalid"
shopt -s nullglob dotglob
ACTUAL_OUTPUTS=("$EVALUATION_DIR"/*)
[[ "${#ACTUAL_OUTPUTS[@]}" -eq "${#EXPECTED_OUTPUTS[@]}" ]] || fail "output set must contain five files"
for OUTPUT in "${EXPECTED_OUTPUTS[@]}"; do
  [[ -f "$EVALUATION_DIR/$OUTPUT" && ! -L "$EVALUATION_DIR/$OUTPUT" ]] \
    || fail "expected evaluation artifact missing: $OUTPUT"
done

"$PY" - "$EVALUATION_DIR" "$ROLE" "$TRAINING_CAPS" "$RESOLVED_CHECKPOINT" \
  "$CHECKPOINT_SHA256" "$EXPECTED_CODE_REVISION" "$EXPECTED_ASSET_REVISION" <<'PY' \
  || fail "evaluation traces or provenance failed validation"
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
role, training_caps, checkpoint_path, checkpoint_sha, code_revision, asset_revision = sys.argv[2:]
for name in (
  "fixed_trace.npz",
  "training_like_seed_2_trace.npz",
  "training_like_seed_2026081701_trace.npz",
  "training_like_seed_2026081702_trace.npz",
):
  with np.load(root / name, allow_pickle=False) as trace:
    required = {"delta_velocity", "delta_impulse", "delta", "lambda_per_joint", "substep_rolling_per_joint"}
    if not required.issubset(trace.files):
      raise RuntimeError(f"{name} is missing telemetry")
    for field in trace.files:
      value = trace[field]
      if value.dtype.hasobject or (np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()):
        raise RuntimeError(f"{name}:{field} is invalid")
    velocity, impulse, combined = trace["delta_velocity"], trace["delta_impulse"], trace["delta"]
    if velocity.ndim != 2 or velocity.shape != impulse.shape or velocity.shape != combined.shape:
      raise RuntimeError(f"{name} scalar shapes invalid")
    if trace["lambda_per_joint"].shape != (*velocity.shape, 6):
      raise RuntimeError(f"{name} Lambda shape invalid")
    if not np.array_equal(impulse, np.zeros_like(impulse)):
      raise RuntimeError(f"{name} live impulse was not log-only")
    if not np.array_equal(combined, np.maximum(velocity, impulse)):
      raise RuntimeError(f"{name} violates exact max soft-OR")

summary = json.loads((root / "summary.json").read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
expected_task = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT"
if summary.get("schema_version") != 2 or summary.get("task") != expected_task:
  raise RuntimeError("summary schema or task mismatch")
if summary.get("checkpoint") != {"role": role, "path": checkpoint_path, "sha256": checkpoint_sha}:
  raise RuntimeError("summary checkpoint provenance mismatch")
if summary.get("training_cap_identity_n_m_s") != json.loads(training_caps):
  raise RuntimeError("summary training-cap identity mismatch")
if summary.get("code_revision") != code_revision or summary.get("asset_revision") != asset_revision:
  raise RuntimeError("summary repository provenance mismatch")
protocol = summary.get("protocol", {})
if protocol.get("live_imp_max_p") != 0.0 or protocol.get("fixed_seed") != 2026081202:
  raise RuntimeError("summary live-CaT or fixed population mismatch")
if protocol.get("stochastic_seeds") != [2, 2026081701, 2026081702]:
  raise RuntimeError("summary stochastic population mismatch")
sampled = summary.get("populations", {}).get("training_like_sampled", {})
if tuple(sampled) != ("2", "2026081701", "2026081702"):
  raise RuntimeError("summary stochastic population keys mismatch")
for seed, population in sampled.items():
  p = population.get("protocol", {})
  expected_seed = int(seed)
  if p.get("seed") != expected_seed or p.get("num_envs") != 4096 or p.get("control_steps") != 24:
    raise RuntimeError("sampled population shape mismatch")
  if p.get("rng_streams") != {
    "reset": expected_seed + 10_000_019,
    "observation": expected_seed + 20_000_033,
    "action": expected_seed + 30_000_041,
  }:
    raise RuntimeError("sampled population RNG mismatch")
PY

(
  cd "$ATTEMPT_DIR"
  sha256sum "${EXPECTED_OUTPUTS[@]/#/evaluation/}" > SHA256SUMS
) || fail "evaluation artifact hashing failed"
[[ "$(wc -l < "$ATTEMPT_DIR/SHA256SUMS" | tr -d ' ')" = 5 ]] || fail "manifest row count mismatch"
POST_CHECKPOINT_SHA256=$(sha256sum "$RESOLVED_CHECKPOINT" 2>/dev/null | cut -d ' ' -f 1) \
  || fail "postflight checkpoint hashing failed"
[[ "$POST_CHECKPOINT_SHA256" = "$CHECKPOINT_SHA256" ]] || fail "checkpoint changed during evaluation"
[[ "$(git -C "$RUN_ROOT" rev-parse HEAD)" = "$EXPECTED_CODE_REVISION" ]] || fail "code revision changed"
[[ -z "$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || fail "code became dirty"
[[ "$(git -C "$ASSET_REPO" rev-parse HEAD)" = "$EXPECTED_ASSET_REVISION" ]] || fail "asset revision changed"
[[ -z "$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all)" ]] || fail "assets became dirty"

printf 'Z1_VIC_IMPULSE_JOINT_STRESS_CHECKPOINT_EVAL_CHECKPOINT role=%s basename=%s sha256=%s\n' \
  "$ROLE" "$EXPECTED_BASENAME" "$CHECKPOINT_SHA256"
printf 'Z1_VIC_IMPULSE_JOINT_STRESS_CHECKPOINT_EVAL_PASS role=%s caps=%s evaluation_dir=%s\n' \
  "$ROLE" "$TRAINING_CAPS" "$EVALUATION_DIR"
