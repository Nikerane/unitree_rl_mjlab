#!/usr/bin/env bash
# Frozen no-learning evaluation of the p=0, 0.1, 0.2, and 0.3 impulse-CaT dose policies.
# Every arm evaluates with live imp_max_p=0; this is not a clamp or hardware-safety result.
#SBATCH --job-name=z1-vic-imp-dose-eval
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-dose-curve/slurm/z1-vic-imp-dose-eval-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-dose-curve/slurm/z1-vic-imp-dose-eval-%A_%a.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "Z1_VIC_IMPULSE_DOSE_CURVE_EVAL_FAIL: $*"
  exit 2
}

readonly APPROVED_EVALUATOR_BASE_REVISION=7d0761e3dd156012a22378fd0b269e4c50ea5ae6
readonly PINNED_ASSET_REVISION=b58ccd2f81fd246f27c1e8d88cf86484cd888703
readonly P0_CHECKPOINT_SHA256=f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3
readonly P01_CHECKPOINT_SHA256=92f1d97c8ff1476cb26c0b648478a0bc3c522e4e1eb7087389fb8e0d6bf73f86
readonly P02_CHECKPOINT_SHA256=57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de
readonly P03_CHECKPOINT_SHA256=4c0a665fffc077d488630a28b258c1593050f969b4f6227147c3594097dc4efd
readonly PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE must be empty"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" = 4 ]] || fail "array task count must be exactly 4"
[[ "${SLURM_ARRAY_TASK_MIN:-}" = 0 ]] || fail "array minimum must be exactly 0"
[[ "${SLURM_ARRAY_TASK_MAX:-}" = 3 ]] || fail "array maximum must be exactly 3"
[[ "${SLURM_ARRAY_TASK_STEP:-}" = 1 ]] || fail "array step must be exactly 1"
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "array job id must be numeric"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || fail "array task id must be 0, 1, 2, or 3"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "SLURM_JOB_ID must be numeric"
for VARIABLE in WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT RESUME_PATH; do
  [[ -z "${!VARIABLE:-}" ]] || fail "$VARIABLE must be empty"
done
for VARIABLE in EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION; do
  [[ "${!VARIABLE:-}" =~ ^[0-9a-f]{40}$ ]] || fail "$VARIABLE must be a 40-hex revision"
done
for VARIABLE in RUN_ROOT ASSET_REPO P0_CHECKPOINT P01_CHECKPOINT P02_CHECKPOINT P03_CHECKPOINT; do
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
    readonly ROLE=dose_p0_control
    readonly CHECKPOINT="$P0_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$P0_CHECKPOINT_SHA256"
    ;;
  1)
    readonly ROLE=dose_p01_target
    readonly CHECKPOINT="$P01_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$P01_CHECKPOINT_SHA256"
    ;;
  2)
    readonly ROLE=dose_p02_target
    readonly CHECKPOINT="$P02_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$P02_CHECKPOINT_SHA256"
    ;;
  3)
    readonly ROLE=dose_p03_target
    readonly CHECKPOINT="$P03_CHECKPOINT"
    readonly EXPECTED_CHECKPOINT_SHA256="$P03_CHECKPOINT_SHA256"
    ;;
esac
[[ -f "$CHECKPOINT" && ! -L "$CHECKPOINT" && "$(basename "$CHECKPOINT")" = model_499.pt ]] \
  || fail "checkpoint must be a regular non-symlink model_499.pt"
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" 2>/dev/null | cut -d ' ' -f 1) \
  || fail "checkpoint SHA-256 generation failed"
[[ "$CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint SHA-256 is invalid"
[[ "$CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256" ]] \
  || fail "role/checkpoint SHA-256 mismatch"

readonly RESULT_ROOT="$HOME/campaigns/z1-vic-impulse-dose-curve/evaluation/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
readonly ATTEMPT_DIR="$RESULT_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${ROLE}"
readonly EVALUATION_DIR="$ATTEMPT_DIR/evaluation"
readonly NODE_TMPDIR="/tmp/z1-vic-imp-dose-eval-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
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
print(f"Z1_VIC_IMPULSE_DOSE_CURVE_EVAL_GPU={gpu}")
print(f"Z1_VIC_IMPULSE_DOSE_CURVE_EVAL_RUNTIME={actual}")
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

"$PY" - "$EVALUATION_DIR" "$ROLE" "$CHECKPOINT_SHA256" \
  "$EXPECTED_CODE_REVISION" "$EXPECTED_ASSET_REVISION" <<'PY' \
  || fail "evaluation traces or provenance failed validation"
import json
from pathlib import Path
import re
import sys

import numpy as np

root = Path(sys.argv[1])
role, checkpoint_sha, code_revision, asset_revision = sys.argv[2:]
trace_names = (
  "fixed_trace.npz",
  "training_like_seed_2_trace.npz",
  "training_like_seed_2026081701_trace.npz",
  "training_like_seed_2026081702_trace.npz",
)
for name in trace_names:
  with np.load(root / name, allow_pickle=False) as trace:
    required = {
      "delta_velocity", "delta_impulse", "delta", "lambda_per_joint",
      "substep_rolling_per_joint",
    }
    if not required.issubset(trace.files):
      raise RuntimeError(f"{name} is missing required telemetry")
    for field in trace.files:
      value = trace[field]
      if value.dtype.hasobject:
        raise RuntimeError(f"{name}:{field} has object dtype")
      if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise RuntimeError(f"{name}:{field} is non-finite")
    velocity = trace["delta_velocity"]
    impulse = trace["delta_impulse"]
    combined = trace["delta"]
    lam = trace["lambda_per_joint"]
    rolling = trace["substep_rolling_per_joint"]
    if velocity.ndim != 2 or not (velocity.shape == impulse.shape == combined.shape):
      raise RuntimeError(f"{name} has invalid scalar telemetry shapes")
    if lam.shape != (*velocity.shape, 6):
      raise RuntimeError(f"{name} has invalid Lambda shape")
    if rolling.ndim != 3 or rolling.shape[1:] != (velocity.shape[1], 6):
      raise RuntimeError(f"{name} has invalid substep rolling shape")
    if not np.array_equal(impulse, np.zeros_like(impulse)):
      raise RuntimeError(f"{name} did not evaluate with live impulse log-only")
    if not np.array_equal(combined, np.maximum(velocity, impulse)):
      raise RuntimeError(f"{name} violates exact max soft-OR")

def reject_constant(value):
  raise ValueError(f"non-finite JSON constant: {value}")

summary = json.loads((root / "summary.json").read_text(), parse_constant=reject_constant)
expected_task = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT"
)
if summary.get("schema_version") != 2 or summary.get("task") != expected_task:
  raise RuntimeError("summary schema or task mismatch")
if summary.get("checkpoint") != {"role": role, "sha256": checkpoint_sha}:
  raise RuntimeError("summary checkpoint provenance mismatch")
if summary.get("code_revision") != code_revision or summary.get("asset_revision") != asset_revision:
  raise RuntimeError("summary repository provenance mismatch")
protocol = summary.get("protocol")
if not isinstance(protocol, dict):
  raise RuntimeError("summary protocol is missing")
if protocol.get("live_imp_max_p") != 0.0:
  raise RuntimeError("summary must record live imp_max_p=0")
if protocol.get("fixed_seed") != 2026081202:
  raise RuntimeError("summary fixed population seed mismatch")
if protocol.get("stochastic_seeds") != [2, 2026081701, 2026081702]:
  raise RuntimeError("summary stochastic populations mismatch")
populations = summary.get("populations")
if not isinstance(populations, dict):
  raise RuntimeError("summary populations are missing")
fixed = populations.get("fixed_mean")
sampled = populations.get("training_like_sampled")
if not isinstance(fixed, dict) or not isinstance(sampled, dict):
  raise RuntimeError("summary population structure is invalid")
if tuple(sampled) != ("2", "2026081701", "2026081702"):
  raise RuntimeError("summary stochastic population keys mismatch")

def validate_population(value, *, seed, num_envs, control_steps, mode):
  population_protocol = value.get("protocol") if isinstance(value, dict) else None
  if not isinstance(population_protocol, dict):
    raise RuntimeError("population protocol is missing")
  expected = {
    "seed": seed,
    "num_envs": num_envs,
    "policy_mode": mode,
    "auto_reset": mode == "sampled",
  }
  if any(population_protocol.get(field) != expected_value for field, expected_value in expected.items()):
    raise RuntimeError("population shape or mode provenance mismatch")
  observed_steps = population_protocol.get("control_steps")
  if control_steps is None:
    if type(observed_steps) is not int or observed_steps <= 0:
      raise RuntimeError("fixed population control steps are invalid")
  elif observed_steps != control_steps:
    raise RuntimeError("sampled population control steps mismatch")
  expected_rng = {
    "reset": seed + 10_000_019,
    "observation": seed + 20_000_033,
    "action": seed + 30_000_041,
  }
  if population_protocol.get("rng_streams") != expected_rng:
    raise RuntimeError("population RNG streams mismatch")
  population_sha = population_protocol.get("initial_population_sha256")
  if not isinstance(population_sha, str) or re.fullmatch(r"[0-9a-f]{64}", population_sha) is None:
    raise RuntimeError("population identity hash is invalid")
  return population_sha

fixed_sha = validate_population(
  fixed, seed=2026081202, num_envs=64, control_steps=None, mode="mean"
)
if fixed_sha != "320efae8c21b3303c5dc3f18ae7df00e887653abf26aa8fb413803cf78d094cb":
  raise RuntimeError("fixed population identity mismatch")
for seed in (2, 2026081701, 2026081702):
  validate_population(
    sampled[str(seed)], seed=seed, num_envs=4096, control_steps=24, mode="sampled"
  )
PY

FINAL_CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" 2>/dev/null | cut -d ' ' -f 1) \
  || fail "postflight checkpoint SHA-256 generation failed"
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

readonly MANIFEST_TMP="$NODE_TMPDIR/SHA256SUMS"
(
  cd "$ATTEMPT_DIR"
  sha256sum \
    evaluation/fixed_trace.npz \
    evaluation/training_like_seed_2_trace.npz \
    evaluation/training_like_seed_2026081701_trace.npz \
    evaluation/training_like_seed_2026081702_trace.npz \
    evaluation/summary.json > "$MANIFEST_TMP"
) || fail "SHA256 manifest generation failed"
"$PY" - "$MANIFEST_TMP" <<'PY' || fail "SHA256 manifest validation failed"
from pathlib import Path
import re
import sys

expected = (
  "evaluation/fixed_trace.npz",
  "evaluation/training_like_seed_2_trace.npz",
  "evaluation/training_like_seed_2026081701_trace.npz",
  "evaluation/training_like_seed_2026081702_trace.npz",
  "evaluation/summary.json",
)
lines = Path(sys.argv[1]).read_text().splitlines()
if len(lines) != len(expected):
  raise RuntimeError("manifest must contain exactly five rows")
for line, path in zip(lines, expected, strict=True):
  if re.fullmatch(rf"[0-9a-f]{{64}}  {re.escape(path)}", line) is None:
    raise RuntimeError("manifest row is invalid or out of order")
PY
mv "$MANIFEST_TMP" "$ATTEMPT_DIR/SHA256SUMS"
printf 'Z1_VIC_IMPULSE_DOSE_CURVE_EVAL_PASS role=%s checkpoint_sha256=%s evaluation_dir=%s\n' \
  "$ROLE" "$CHECKPOINT_SHA256" "$EVALUATION_DIR"
