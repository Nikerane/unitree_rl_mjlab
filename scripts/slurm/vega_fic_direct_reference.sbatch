#!/usr/bin/env bash
#SBATCH --job-name=z1-fic-direct-reference
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/z1-fic-direct-reference-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/z1-fic-direct-reference-%A_%a.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "Z1_DIRECT_FAIL: $*"
  exit 2
}

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"

if [[ "${SLURM_ARRAY_TASK_COUNT:-}" = "2" \
  && "${SLURM_ARRAY_TASK_MIN:-}" = "0" \
  && "${SLURM_ARRAY_TASK_MAX:-}" = "1" \
  && "${SLURM_ARRAY_TASK_STEP:-}" = "1" ]]; then
  ARRAY_SHAPE="canary"
elif [[ "${SLURM_ARRAY_TASK_COUNT:-}" = "4" \
  && "${SLURM_ARRAY_TASK_MIN:-}" = "2" \
  && "${SLURM_ARRAY_TASK_MAX:-}" = "5" \
  && "${SLURM_ARRAY_TASK_STEP:-}" = "1" ]]; then
  ARRAY_SHAPE="expansion"
else
  fail "array must be exactly 0-1 or 2-5"
fi

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] \
  || fail "SLURM_ARRAY_TASK_ID is required"
FIC0_TASK="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed"
case "$ARRAY_SHAPE:$SLURM_ARRAY_TASK_ID" in
  canary:0)
    TASK="$FIC0_TASK"
    SHORT="fic0"
    SEED="2"
    ;;
  canary:1)
    TASK="${FIC0_TASK}-TT"
    SHORT="fictt"
    SEED="2"
    ;;
  expansion:2)
    TASK="$FIC0_TASK"
    SHORT="fic0"
    SEED="3"
    ;;
  expansion:3)
    TASK="${FIC0_TASK}-TT"
    SHORT="fictt"
    SEED="3"
    ;;
  expansion:4)
    TASK="$FIC0_TASK"
    SHORT="fic0"
    SEED="4"
    ;;
  expansion:5)
    TASK="${FIC0_TASK}-TT"
    SHORT="fictt"
    SEED="4"
    ;;
  *)
    fail "array index is not valid for the approved array shape"
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

[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] \
  || fail "SLURM_ARRAY_JOB_ID must be numeric"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "SLURM_JOB_ID is required"

PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"
[[ -x "$PY" ]] || fail "Vega Python environment is unavailable"

RESULT_ROOT="$HOME/z1-fic-direct-reference-500/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
ATTEMPT_DIR="$RESULT_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SHORT}_seed${SEED}"
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null \
  || fail "attempt directory already exists"

RUN_NAME="fic_direct_reference_${SHORT}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$RUN_ROOT"
export WARP_CACHE_PATH="$ATTEMPT_DIR/warp-cache"
export RUN_NAME
export FIC_SHORT="$SHORT"
export FIC_SEED="$SEED"
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

cd "$ATTEMPT_DIR"

"$PY" "$RUN_ROOT/scripts/train.py" "$TASK" \
  --gpu-ids '[0]' --agent.logger tensorboard --agent.run-name "$RUN_NAME" \
  --agent.seed "$SEED" --agent.max-iterations 500 --agent.save-interval 50 \
  --env.scene.num-envs 4096

shopt -s nullglob
RUN_DIRS=("$ATTEMPT_DIR"/logs/rsl_rl/z1_hammer/*_"$RUN_NAME")
[[ "${#RUN_DIRS[@]}" -eq 1 ]] \
  || fail "expected exactly one matching training run directory"
RUN_DIR="${RUN_DIRS[0]}"
[[ -d "$RUN_DIR" ]] \
  || fail "expected exactly one matching training run directory"
CHECKPOINT="$RUN_DIR/model_499.pt"
[[ -f "$CHECKPOINT" ]] || fail "model_499.pt is missing"

"$PY" - "$CHECKPOINT" <<'PY' \
  || fail "model_499.pt contains no finite checkpoint state"
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
assert all(bool(torch.isfinite(value).all()) for value in values), (
    "non-finite checkpoint tensor"
)
PY

EVALUATOR="$RUN_ROOT/evaluation/joint_position/evaluate_fic_pilot.py"
[[ -f "$EVALUATOR" ]] || fail "direct-reference FIC evaluator is missing"
EVALUATION_JSON="$ATTEMPT_DIR/${SHORT}_seed${SEED}.json"
[[ ! -e "$EVALUATION_JSON" ]] || fail "evaluation output already exists"
"$PY" "$EVALUATOR" "$TASK" \
  --checkpoint "$CHECKPOINT" --output "$EVALUATION_JSON" \
  --device cuda:0 --training-seed "$SEED"
[[ -s "$EVALUATION_JSON" ]] \
  || fail "evaluator did not create a non-empty JSON result"

CURRICULUM_JSON="$ATTEMPT_DIR/${SHORT}_seed${SEED}_curriculum.json"
[[ ! -e "$CURRICULUM_JSON" ]] || fail "curriculum output already exists"
"$PY" - "$RUN_DIR" "$CURRICULUM_JSON" "$TASK" "$SHORT" "$SEED" <<'PY' \
  || fail "curriculum telemetry validation failed"
import json
import math
from numbers import Integral
from pathlib import Path
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


run_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
task = sys.argv[3]
arm = sys.argv[4]
training_seed = int(sys.argv[5])
scalar = "Curriculum/r_imit_anneal/weight"
stages = (0.10, 0.08, 0.06, 0.04, 0.02, 0.00)
tolerance = 1e-6

accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
accumulator.Reload()
assert scalar in accumulator.Tags().get("scalars", ()), (
    f"missing TensorBoard scalar: {scalar}"
)
events = accumulator.Scalars(scalar)
assert events, "empty curriculum scalar stream"

observed = []
previous_step = -1
previous_weight = math.inf
for event in events:
    assert isinstance(event.step, Integral), "curriculum event step is not an integer"
    iteration = int(event.step)
    assert iteration >= 0, "curriculum event step is negative"
    assert iteration > previous_step, "curriculum event steps are not strictly increasing"
    weight = float(event.value)
    assert math.isfinite(weight), "non-finite curriculum weight"
    assert -tolerance <= weight <= 0.1 + tolerance, (
        f"curriculum weight outside [0,0.1]: {weight}"
    )
    assert previous_weight >= weight - tolerance, (
        f"curriculum weights increase: {previous_weight} -> {weight}"
    )
    observed.append({"iteration": iteration, "weight": weight})
    previous_step = iteration
    previous_weight = weight

for stage in stages:
    assert any(abs(row["weight"] - stage) <= tolerance for row in observed), (
        f"canonical curriculum plateau was not observed: {stage}"
    )
assert abs(observed[-1]["weight"]) <= tolerance, (
    "final curriculum sample is not zero"
)

payload = {
    "schema_version": 1,
    "task": task,
    "arm": arm,
    "training_seed": training_seed,
    "scalar": scalar,
    "expected_control_step_stages": [
        [0, 0.1],
        [1200, 0.08],
        [2400, 0.06],
        [3600, 0.04],
        [4800, 0.02],
        [6000, 0.0],
    ],
    "observed": observed,
    "final_weight": 0.0,
}
encoded = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), allow_nan=False
) + "\n"
with output.open("x", encoding="utf-8") as stream:
    stream.write(encoded)
PY
[[ -s "$CURRICULUM_JSON" ]] \
  || fail "curriculum extractor did not create a non-empty JSON result"

CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)
EVALUATION_SHA256=$(sha256sum "$EVALUATION_JSON" | cut -d ' ' -f 1)
CURRICULUM_SHA256=$(sha256sum "$CURRICULUM_JSON" | cut -d ' ' -f 1)
for DIGEST in "$CHECKPOINT_SHA256" "$EVALUATION_SHA256" "$CURRICULUM_SHA256"; do
  [[ "$DIGEST" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256 result"
done

printf 'Z1_DIRECT_CODE_REVISION=%s\n' "$CODE_REVISION"
printf 'Z1_DIRECT_ASSET_REVISION=%s\n' "$ASSET_REVISION"
printf 'Z1_DIRECT_CHECKPOINT=%s\n' "$CHECKPOINT"
printf 'Z1_DIRECT_CHECKPOINT_SHA256=%s\n' "$CHECKPOINT_SHA256"
printf 'Z1_DIRECT_EVALUATION=%s\n' "$EVALUATION_JSON"
printf 'Z1_DIRECT_EVALUATION_SHA256=%s\n' "$EVALUATION_SHA256"
printf 'Z1_DIRECT_CURRICULUM=%s\n' "$CURRICULUM_JSON"
printf 'Z1_DIRECT_CURRICULUM_SHA256=%s\n' "$CURRICULUM_SHA256"
printf 'FIC_DIRECT_REFERENCE_DONE arm=%s seed=%s task=%s\n' "$SHORT" "$SEED" "$TASK"
