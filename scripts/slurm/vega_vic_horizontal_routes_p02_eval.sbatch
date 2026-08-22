#!/usr/bin/env bash
# Frozen actor-only R-/R0/R+ evaluation for the two horizontal-route policies.
# Live impulse CaT is log-only (p=0); custom training caps are applied offline.
#SBATCH --job-name=z1-vic-horizontal-routes-p02-eval
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-horizontal-routes-p02-evaluation/slurm/z1-vic-horizontal-routes-p02-eval-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-horizontal-routes-p02-evaluation/slurm/z1-vic-horizontal-routes-p02-eval-%A_%a.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "Z1_VIC_HORIZONTAL_ROUTES_P02_EVAL_FAIL: $*"
  exit 2
}

readonly APPROVED_EVALUATOR_BASE_REVISION=1f0656012e9e5a03b0c2d404d9d4fc605786009b
readonly PINNED_ASSET_REVISION=b58ccd2f81fd246f27c1e8d88cf86484cd888703
readonly ANNEALED_CHECKPOINT_SHA256=2a434c50457455daab10b5ff32e1ee26e3b189b0092a315f3e881bb7f960156a
readonly PERSISTENT_CHECKPOINT_SHA256=feb22dd5e5740e96348387c846ef395143316354f7e3163b5684f24db58f4e9e
readonly CAPS='[0.369,0.246,0.738,0.369,0.246,0.0164]'
readonly PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE must be empty"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" = 6 ]] || fail "array task count must be exactly 6"
[[ "${SLURM_ARRAY_TASK_MIN:-}" = 0 ]] || fail "array minimum must be exactly 0"
[[ "${SLURM_ARRAY_TASK_MAX:-}" = 5 ]] || fail "array maximum must be exactly 5"
[[ "${SLURM_ARRAY_TASK_STEP:-}" = 1 ]] || fail "array step must be exactly 1"
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "array job id must be numeric"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-5]$ ]] || fail "array task id must be 0 through 5"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "SLURM_JOB_ID must be numeric"
for VARIABLE in WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT RESUME_PATH; do
  [[ -z "${!VARIABLE:-}" ]] || fail "$VARIABLE must be empty"
done
for VARIABLE in EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION; do
  [[ "${!VARIABLE:-}" =~ ^[0-9a-f]{40}$ ]] || fail "$VARIABLE must be a 40-hex revision"
done
for VARIABLE in RUN_ROOT ASSET_REPO ANNEALED_CHECKPOINT PERSISTENT_CHECKPOINT; do
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
  "$EXPECTED_CODE_REVISION" 2>/dev/null || fail "code revision is not descended from training"
[[ -z "$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "code repository must be clean"
ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) || fail "asset revision unavailable"
[[ "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] || fail "asset revision mismatch"
[[ -z "$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "asset repository must be clean"

case "$SLURM_ARRAY_TASK_ID" in
  0) readonly ROLE=horizontal_routes_annealed ROUTE_SIGN=-1 ROUTE_LABEL=rminus CHECKPOINT="$ANNEALED_CHECKPOINT" EXPECTED_CHECKPOINT_SHA256="$ANNEALED_CHECKPOINT_SHA256" ;;
  1) readonly ROLE=horizontal_routes_annealed ROUTE_SIGN=0 ROUTE_LABEL=r0 CHECKPOINT="$ANNEALED_CHECKPOINT" EXPECTED_CHECKPOINT_SHA256="$ANNEALED_CHECKPOINT_SHA256" ;;
  2) readonly ROLE=horizontal_routes_annealed ROUTE_SIGN=1 ROUTE_LABEL=rplus CHECKPOINT="$ANNEALED_CHECKPOINT" EXPECTED_CHECKPOINT_SHA256="$ANNEALED_CHECKPOINT_SHA256" ;;
  3) readonly ROLE=horizontal_routes_persistent ROUTE_SIGN=-1 ROUTE_LABEL=rminus CHECKPOINT="$PERSISTENT_CHECKPOINT" EXPECTED_CHECKPOINT_SHA256="$PERSISTENT_CHECKPOINT_SHA256" ;;
  4) readonly ROLE=horizontal_routes_persistent ROUTE_SIGN=0 ROUTE_LABEL=r0 CHECKPOINT="$PERSISTENT_CHECKPOINT" EXPECTED_CHECKPOINT_SHA256="$PERSISTENT_CHECKPOINT_SHA256" ;;
  5) readonly ROLE=horizontal_routes_persistent ROUTE_SIGN=1 ROUTE_LABEL=rplus CHECKPOINT="$PERSISTENT_CHECKPOINT" EXPECTED_CHECKPOINT_SHA256="$PERSISTENT_CHECKPOINT_SHA256" ;;
esac
[[ -f "$CHECKPOINT" && ! -L "$CHECKPOINT" && "$(basename "$CHECKPOINT")" = model_499.pt ]] \
  || fail "checkpoint must be a regular non-symlink model_499.pt"
CHECKPOINT_DIRECTORY=$(cd "$(dirname "$CHECKPOINT")" 2>/dev/null && pwd -P) \
  || fail "checkpoint directory is unavailable"
readonly RESOLVED_CHECKPOINT="$CHECKPOINT_DIRECTORY/model_499.pt"
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" 2>/dev/null | cut -d ' ' -f 1) \
  || fail "checkpoint SHA-256 generation failed"
[[ "$CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint SHA-256 is invalid"
[[ "$CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256" ]] \
  || fail "role/checkpoint SHA-256 mismatch"

readonly RESULT_ROOT="$HOME/campaigns/z1-vic-horizontal-routes-p02-evaluation/evaluation/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
readonly ATTEMPT_DIR="$RESULT_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${ROLE}_${ROUTE_LABEL}"
readonly EVALUATION_DIR="$ATTEMPT_DIR/evaluation"
readonly NODE_TMPDIR="/tmp/z1-vic-horizontal-routes-p02-eval-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -m 700 -- "$NODE_TMPDIR" 2>/dev/null || fail "node-local temporary directory already exists"
cleanup_node_tmpdir() { rm -rf -- "$NODE_TMPDIR"; }
trap cleanup_node_tmpdir EXIT
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null || fail "attempt directory already exists"
export PYTHONPATH="$RUN_ROOT"
export WARP_CACHE_PATH="$NODE_TMPDIR/warp-cache"
export TMPDIR="$NODE_TMPDIR" TEMP="$NODE_TMPDIR" TMP="$NODE_TMPDIR"
mkdir "$WARP_CACHE_PATH"

"$PY" - <<'PY' || fail "CUDA identity or runtime package versions do not match"
from importlib.metadata import version
import torch
expected = {"mjlab": "1.4.0", "mujoco": "3.8.1", "mujoco-warp": "3.8.1"}
actual = {name: version(name) for name in expected}
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
  raise RuntimeError("exactly one CUDA device is required")
if torch.cuda.get_device_name(0) != "NVIDIA A100-SXM4-40GB" or actual != expected:
  raise RuntimeError("GPU or runtime drift")
PY

"$PY" "$RUN_ROOT/scripts/evaluate_vic_horizontal_routes.py" \
  --checkpoint-role "$ROLE" --route-sign "$ROUTE_SIGN" \
  --checkpoint "$RESOLVED_CHECKPOINT" --output-dir "$EVALUATION_DIR" --device cuda:0

readonly EXPECTED_OUTPUTS=(
  fixed_trace.npz training_like_seed_2_trace.npz
  training_like_seed_2026081701_trace.npz training_like_seed_2026081702_trace.npz
  summary.json
)
[[ -d "$EVALUATION_DIR" && ! -L "$EVALUATION_DIR" ]] || fail "evaluator output missing"
shopt -s nullglob dotglob
ACTUAL_OUTPUTS=("$EVALUATION_DIR"/*)
[[ "${#ACTUAL_OUTPUTS[@]}" -eq 5 ]] || fail "evaluator output set must contain five artifacts"
for OUTPUT in "${EXPECTED_OUTPUTS[@]}"; do
  [[ -f "$EVALUATION_DIR/$OUTPUT" && ! -L "$EVALUATION_DIR/$OUTPUT" ]] \
    || fail "expected evaluation artifact is missing or non-regular: $OUTPUT"
done

"$PY" - "$EVALUATION_DIR" "$ROLE" "$ROUTE_SIGN" "$ROUTE_LABEL" "$RESOLVED_CHECKPOINT" \
  "$CHECKPOINT_SHA256" "$EXPECTED_CODE_REVISION" "$EXPECTED_ASSET_REVISION" "$CAPS" <<'PY' \
  || fail "route traces or provenance failed validation"
import json, re, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
role, sign_text, label, checkpoint, digest, code, asset, caps_text = sys.argv[2:]
sign = int(sign_text); caps = json.loads(caps_text)
names = ("fixed_trace.npz", "training_like_seed_2_trace.npz", "training_like_seed_2026081701_trace.npz", "training_like_seed_2026081702_trace.npz")
for name in names:
  with np.load(root / name, allow_pickle=False) as trace:
    required = {"episode_id","route_sign","strike_phase","hammer_head_pos_w","assigned_reference_waypoint_w","straight_reference_waypoint_w","imitation_eligible","delta_impulse","delta_velocity","delta","lambda_per_joint","substep_rolling_per_joint"}
    if not required.issubset(trace.files): raise RuntimeError(f"{name} missing telemetry")
    scalar = trace["episode_id"].shape
    if len(scalar) != 2 or trace["route_sign"].shape != scalar: raise RuntimeError("route scalar shape")
    if not np.array_equal(trace["route_sign"], np.full(scalar, sign)): raise RuntimeError("forced route drift")
    if trace["imitation_eligible"].dtype != np.bool_: raise RuntimeError("imitation gate dtype")
    if not np.array_equal(trace["delta_impulse"], np.zeros(scalar)): raise RuntimeError("live impulse not log-only")
    for field in trace.files:
      value = trace[field]
      if value.dtype.hasobject or (np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()):
        raise RuntimeError(f"{name}:{field} invalid")
def reject(value): raise ValueError(value)
summary = json.loads((root / "summary.json").read_text(), parse_constant=reject)
expected_task = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-HorizontalRoutes-" + ("Annealed" if role.endswith("annealed") else "Persistent")
if summary.get("schema_version") != 1 or summary.get("task") != expected_task: raise RuntimeError("schema/task")
if summary.get("checkpoint") != {"role":role,"path":checkpoint,"sha256":digest}: raise RuntimeError("checkpoint provenance")
if summary.get("route") != {"label":label,"sign":sign}: raise RuntimeError("route provenance")
if summary.get("training_cap_identity_n_m_s") != caps: raise RuntimeError("cap provenance")
if summary.get("code_revision") != code or summary.get("asset_revision") != asset: raise RuntimeError("Git provenance")
protocol = summary.get("protocol", {})
if protocol.get("live_imp_max_p") != 0.0 or protocol.get("actor_only_checkpoint_load") is not True: raise RuntimeError("evaluation mode")
if protocol.get("stochastic_seeds") != [2,2026081701,2026081702]: raise RuntimeError("seeds")
pop = summary.get("populations", {}); sampled = pop.get("training_like_sampled", {})
if tuple(sampled) != ("2","2026081701","2026081702"): raise RuntimeError("population keys")
for value in (pop.get("fixed_mean"), *sampled.values()):
  p = value.get("protocol", {})
  if p.get("forced_route_sign") != sign or p.get("actor_only_checkpoint_load") is not True: raise RuntimeError("population route/load")
  if p.get("training_config") != {"imp_max_p":0.2,"imp_limit_n_m_s":caps}: raise RuntimeError("training config")
  if p.get("cat_replay",{}).get("imp_max_p_live") != 0.0: raise RuntimeError("population live p")
PY

FINAL_CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" 2>/dev/null | cut -d ' ' -f 1) \
  || fail "postflight checkpoint SHA-256 generation failed"
[[ "$FINAL_CHECKPOINT_SHA256" = "$CHECKPOINT_SHA256" ]] || fail "checkpoint SHA-256 drifted"
[[ "$(git -C "$RUN_ROOT" rev-parse HEAD)" = "$EXPECTED_CODE_REVISION" \
   && -z "$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "postflight code provenance drifted"
[[ "$(git -C "$ASSET_REPO" rev-parse HEAD)" = "$EXPECTED_ASSET_REVISION" \
   && -z "$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all)" ]] \
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
import re, sys
expected = ("evaluation/fixed_trace.npz","evaluation/training_like_seed_2_trace.npz","evaluation/training_like_seed_2026081701_trace.npz","evaluation/training_like_seed_2026081702_trace.npz","evaluation/summary.json")
lines = Path(sys.argv[1]).read_text().splitlines()
if len(lines) != 5: raise RuntimeError("manifest row count")
for line, path in zip(lines, expected, strict=True):
  if re.fullmatch(rf"[0-9a-f]{{64}}  {re.escape(path)}", line) is None: raise RuntimeError("manifest row")
PY
mv "$MANIFEST_TMP" "$ATTEMPT_DIR/SHA256SUMS"
printf 'Z1_VIC_HORIZONTAL_ROUTES_P02_EVAL_PASS role=%s route=%s checkpoint_sha256=%s evaluation_dir=%s\n' \
  "$ROLE" "$ROUTE_LABEL" "$CHECKPOINT_SHA256" "$EVALUATION_DIR"
