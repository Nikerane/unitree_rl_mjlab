#!/usr/bin/env bash
#SBATCH --job-name=z1-victt-seed2
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-prototype/slurm/z1-victt-seed2-%j.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-prototype/slurm/z1-victt-seed2-%j.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl

fail() {
  echo "Z1_VIC_CANARY_FAIL: $*"
  exit 2
}

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE must be empty"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "SLURM_JOB_ID must be numeric"
for ARRAY_VARIABLE in \
  SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP; do
  [[ -z "${!ARRAY_VARIABLE:-}" ]] || fail "array context is forbidden: $ARRAY_VARIABLE"
done
for DISTRIBUTED_VARIABLE in WORLD_SIZE RANK LOCAL_RANK; do
  [[ -z "${!DISTRIBUTED_VARIABLE:-}" ]] \
    || fail "distributed context is forbidden: $DISTRIBUTED_VARIABLE"
done

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
CODE_STATUS=$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all) \
  || fail "code repository status check failed"
[[ -z "$CODE_STATUS" ]] || fail "code repository must be clean"

ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) \
  || fail "asset repository is unavailable"
[[ "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] \
  || fail "asset revision mismatch"
ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all) \
  || fail "asset repository status check failed"
[[ -z "$ASSET_STATUS" ]] || fail "asset repository must be clean"

PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"
[[ -x "$PY" ]] || fail "Vega Python environment is unavailable"

RESULT_ROOT="$HOME/campaigns/z1-vic-prototype/runs/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
ATTEMPT_DIR="$RESULT_ROOT/${SLURM_JOB_ID}_victt_seed2"
RUN_NAME="victt_seed2_job${SLURM_JOB_ID}"
readonly NODE_TMPDIR="/tmp/z1-vic-canary-${SLURM_JOB_ID}"
mkdir -m 700 -- "$NODE_TMPDIR" 2>/dev/null \
  || fail "node-local temporary directory already exists or could not be created"
cleanup_node_tmpdir() {
  rm -rf -- "$NODE_TMPDIR"
}
trap cleanup_node_tmpdir EXIT
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null || fail "attempt directory already exists"

export PYTHONPATH="$RUN_ROOT"
export WARP_CACHE_PATH="$ATTEMPT_DIR/warp-cache"
export RUN_NAME
mkdir "$WARP_CACHE_PATH"
export TMPDIR="$NODE_TMPDIR"
export TEMP="$NODE_TMPDIR"
export TMP="$NODE_TMPDIR"

"$PY" - <<'PY' || fail "CUDA identity or runtime package versions do not match"
from importlib.metadata import version

import torch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


expected = {"mjlab": "1.4.0", "mujoco": "3.8.1", "mujoco-warp": "3.8.1"}
actual = {package: version(package) for package in expected}
require(torch.cuda.is_available(), "CUDA unavailable")
device_count = torch.cuda.device_count()
require(device_count == 1, f"CUDA device count mismatch: {device_count}")
gpu_name = torch.cuda.get_device_name(0)
require(gpu_name == "NVIDIA A100-SXM4-40GB", f"unexpected GPU: {gpu_name}")
require(actual == expected, f"runtime package versions do not match: {actual!r}")
print(f"Z1_VIC_CANARY_CUDA_GPU={gpu_name}")
print("Z1_VIC_CANARY_RUNTIME=" + " ".join(f"{key}={value}" for key, value in actual.items()))
PY

TASK="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT"
cd "$ATTEMPT_DIR"

"$PY" "$RUN_ROOT/scripts/smoke_joint_position_fixed.py" \
  --task "$TASK" --device cuda:0
"$PY" "$RUN_ROOT/scripts/smoke_cat_soft.py" \
  --task "$TASK" --device cuda:0 --iters 1
"$PY" "$RUN_ROOT/scripts/train.py" "$TASK" \
  --gpu-ids '[0]' --agent.logger tensorboard --agent.run-name "$RUN_NAME" \
  --agent.seed 2 --agent.max-iterations 500 --agent.save-interval 50 \
  --env.scene.num-envs 4096

shopt -s nullglob
RUN_DIRS=("$ATTEMPT_DIR"/logs/rsl_rl/z1_hammer/*_"$RUN_NAME")
[[ "${#RUN_DIRS[@]}" -eq 1 ]] \
  || fail "expected exactly one matching training run directory"
RUN_DIR="${RUN_DIRS[0]}"
[[ -d "$RUN_DIR" ]] \
  || fail "expected exactly one matching training run directory"

EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)
EXPECTED_CHECKPOINTS=()
for ITERATION in "${EXPECTED_ITERATIONS[@]}"; do
  EXPECTED_CHECKPOINTS+=("$RUN_DIR/model_${ITERATION}.pt")
done
ACTUAL_CHECKPOINTS=("$RUN_DIR"/model_*.pt)
[[ "${#ACTUAL_CHECKPOINTS[@]}" -eq "${#EXPECTED_CHECKPOINTS[@]}" ]] \
  || fail "checkpoint set does not match the exact canary schedule"
for CHECKPOINT in "${EXPECTED_CHECKPOINTS[@]}"; do
  [[ -f "$CHECKPOINT" ]] \
    || fail "checkpoint set does not match the exact canary schedule"
done

TELEMETRY_JSON="$ATTEMPT_DIR/victt_seed2_telemetry.json"
[[ ! -e "$TELEMETRY_JSON" ]] || fail "canonical telemetry output already exists"
"$PY" - "$RUN_DIR" "$TELEMETRY_JSON" "$TASK" <<'PY' \
  || fail "checkpoint or TensorBoard telemetry validation failed"
from collections.abc import Mapping
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def require_finite_json(value, path="telemetry"):
    if isinstance(value, Mapping):
        for key, item in value.items():
            require(isinstance(key, str), f"{path} contains a non-string key")
            require_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    elif isinstance(value, Real):
        require(math.isfinite(float(value)), f"{path} contains a non-finite value")
    else:
        raise RuntimeError(f"{path} contains a non-JSON value")


def require_vector(value, name, *, lower=None, upper=None, positive=False):
    require(isinstance(value, list) and len(value) == 6, f"{name} must contain six values")
    result = []
    for item in value:
        require(isinstance(item, Real) and not isinstance(item, bool), f"{name} is non-numeric")
        number = float(item)
        require(math.isfinite(number), f"{name} is non-finite")
        if lower is not None:
            require(number >= lower, f"{name} is below its lower bound")
        if upper is not None:
            require(number <= upper, f"{name} is above its upper bound")
        if positive:
            require(number > 0.0, f"{name} must be positive")
        result.append(number)
    return result


JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
SUMMARY_KEYS = {
    "mean", "std", "minimum", "p05", "median", "p95", "maximum",
    "lower_bound_occupancy", "upper_bound_occupancy",
}
TELEMETRY_KEYS = {
    "schema_version", "source", "action_terms", "joint_names",
    "gain_action_indices", "raw_action_clip", "rollout_steps_per_env",
    "sample_count", "temporal_provenance", "deterministic_gaussian_mean",
    "sampled_action", "gaussian_exploration_std",
}


def validate_telemetry(value, *, expected_samples):
    require(isinstance(value, Mapping), "VIC rollout telemetry is missing")
    require(set(value) == TELEMETRY_KEYS, "VIC rollout telemetry schema drifted")
    require(value["schema_version"] == 1, "VIC telemetry schema version drifted")
    require(value["source"] == "cat_rollout_storage", "VIC telemetry source drifted")
    require(value["action_terms"] == ["joint_position", "joint_stiffness"], "VIC action signature drifted")
    require(value["joint_names"] == JOINTS, "VIC joint order drifted")
    require(value["gain_action_indices"] == [6, 7, 8, 9, 10, 11], "VIC gain indices drifted")
    require(value["raw_action_clip"] == 1.0, "VIC raw action clip drifted")
    require(value["rollout_steps_per_env"] == 24, "VIC rollout lifecycle drifted")
    require(value["sample_count"] == expected_samples and expected_samples > 0, "VIC telemetry sample count drifted")
    require(value["temporal_provenance"] == {
        "rollout_generated_by": "pre_update_behavior_policy",
        "checkpoint_weights": "post_update",
    }, "VIC temporal provenance drifted")
    for source in ("deterministic_gaussian_mean", "sampled_action"):
        forms = value[source]
        require(isinstance(forms, Mapping) and set(forms) == {"raw", "clipped"}, f"{source} forms drifted")
        for form in ("raw", "clipped"):
            summary = forms[form]
            require(isinstance(summary, Mapping) and set(summary) == SUMMARY_KEYS, f"{source}.{form} schema drifted")
            vectors = {}
            for key in SUMMARY_KEYS:
                lower, upper = (0.0, 1.0) if key.endswith("occupancy") else (None, None)
                vectors[key] = require_vector(summary[key], f"{source}.{form}.{key}", lower=lower, upper=upper)
            require(all(value >= 0.0 for value in vectors["std"]), f"{source}.{form}.std must be nonnegative")
            for values in zip(
                vectors["minimum"], vectors["p05"], vectors["median"],
                vectors["p95"], vectors["maximum"], strict=True,
            ):
                require(
                    values[0] <= values[1] <= values[2] <= values[3] <= values[4],
                    f"{source}.{form} summary order drifted",
                )
            clipped = form == "clipped"
            if clipped:
                for key in ("mean", "minimum", "p05", "median", "p95", "maximum"):
                    require_vector(summary[key], f"{source}.clipped.{key}", lower=-1.0, upper=1.0)
    require_vector(value["gaussian_exploration_std"], "gaussian_exploration_std", positive=True)
    require_finite_json(value)
    json.dumps(value, allow_nan=False, sort_keys=True)
    return dict(value)


run_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
task = sys.argv[3]
iterations = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499]
states = {}
records = {}
for iteration in iterations:
    checkpoint = run_dir / f"model_{iteration}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(state, Mapping), f"model_{iteration}.pt is not a mapping")
    values = list(tensors(state))
    require(values, f"model_{iteration}.pt contains no tensors")
    require(all(bool(torch.isfinite(value).all()) for value in values), f"model_{iteration}.pt contains a non-finite tensor")
    require(type(state.get("iter")) is int and state["iter"] == iteration, f"model_{iteration}.pt iteration identity drifted")
    states[iteration] = state
for iteration in (0, 499):
    infos = states[iteration].get("infos")
    require(isinstance(infos, Mapping), f"model_{iteration}.pt infos are missing")
    expected_samples = 24 * 4096
    records[f"model_{iteration}"] = validate_telemetry(
        infos.get("vic_rollout_telemetry"), expected_samples=expected_samples
    )

final = records["model_499"]["deterministic_gaussian_mean"]["clipped"]
means = [float(value) for value in final["mean"]]
stds = [float(value) for value in final["std"]]
lower_occupancy = [float(value) for value in final["lower_bound_occupancy"]]
upper_occupancy = [float(value) for value in final["upper_bound_occupancy"]]
all_zero = all(value == 0.0 for value in means) and all(value == 0.0 for value in stds)
nonzero_spread = any(value > 0.0 for value in stds)
bound_pinned_joints = [
    joint for joint, lower, upper in zip(JOINTS, lower_occupancy, upper_occupancy, strict=True)
    if lower == 1.0 or upper == 1.0
]
require(not all_zero, "final deterministic clipped gain telemetry is all zero")
require(nonzero_spread, "final deterministic clipped gain telemetry has no spread")
require(not bound_pinned_joints, f"final deterministic gain is bound pinned: {bound_pinned_joints}")

accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
accumulator.Reload()
tags = accumulator.Tags().get("scalars", ())
require(tags, "TensorBoard contains no scalar streams")
all_events = {}
for tag in tags:
    events = accumulator.Scalars(tag)
    require(events, f"empty TensorBoard scalar stream: {tag}")
    rows = []
    for event in events:
        require(isinstance(event.step, Integral), f"non-integer TensorBoard step: {tag}")
        step = int(event.step)
        value = float(event.value)
        require(step >= 0, f"negative TensorBoard step: {tag}")
        require(math.isfinite(value), f"non-finite TensorBoard scalar: {tag}")
        rows.append({"iteration": step, "value": value})
    all_events[tag] = rows

curriculum_tag = "Curriculum/r_imit_anneal/weight"
impossible_tag = "Episode_Metrics/impossible_success"
require(curriculum_tag in all_events, f"missing TensorBoard scalar: {curriculum_tag}")
require(impossible_tag in all_events, f"missing TensorBoard scalar: {impossible_tag}")
curriculum = all_events[curriculum_tag]
previous_step = -1
previous_weight = math.inf
tolerance = 1e-6
for row in curriculum:
    require(row["iteration"] > previous_step, "curriculum event steps are not strictly increasing")
    require(-tolerance <= row["value"] <= 0.1 + tolerance, "curriculum weight outside [0,0.1]")
    require(previous_weight >= row["value"] - tolerance, "curriculum weights increase")
    previous_step = row["iteration"]
    previous_weight = row["value"]
stages = (0.10, 0.08, 0.06, 0.04, 0.02, 0.00)
anchors = {0: 0.10, 50: 0.08, 100: 0.06, 150: 0.04, 200: 0.02, 250: 0.00, 499: 0.00}
by_iteration = {row["iteration"]: row["value"] for row in curriculum}
for iteration, expected in anchors.items():
    require(
        iteration in by_iteration and abs(by_iteration[iteration] - expected) <= tolerance,
        f"curriculum timing drifted at iteration {iteration}",
    )
boundaries = {49: (0.08, 0.10), 99: (0.06, 0.08), 149: (0.04, 0.06), 199: (0.02, 0.04), 249: (0.00, 0.02)}
for row in curriculum:
    iteration = row["iteration"]
    weight = row["value"]
    if iteration in boundaries:
        lower, upper = boundaries[iteration]
        require(lower - tolerance <= weight <= upper + tolerance, f"curriculum timing drifted at iteration {iteration}")
    else:
        stage_index = min(iteration // 50, 5)
        require(abs(weight - stages[stage_index]) <= tolerance, f"curriculum timing drifted at iteration {iteration}")
for stage in stages:
    require(any(abs(row["value"] - stage) <= tolerance for row in curriculum), f"canonical curriculum plateau was not observed: {stage}")
require(abs(curriculum[-1]["value"]) <= tolerance, "final curriculum sample is not zero")
impossible = all_events[impossible_tag]
require(all(row["value"] == 0.0 for row in impossible), "impossible_success is nonzero")

payload = {
    "schema_version": 1,
    "task": task,
    "training_seed": 2,
    "checkpoint_iterations": iterations,
    "vic_rollout_telemetry": records,
    "final_gain_classification": {
        "all_zero": all_zero,
        "nonzero_spread": nonzero_spread,
        "bound_pinned_joints": bound_pinned_joints,
    },
    "r_imit_curriculum": {
        "scalar": curriculum_tag,
        "observed": curriculum,
        "final_weight": 0.0,
    },
    "impossible_success": {
        "scalar": impossible_tag,
        "observed": impossible,
        "all_zero": True,
    },
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
with output.open("x", encoding="utf-8") as stream:
    stream.write(encoded)
PY
[[ -s "$TELEMETRY_JSON" ]] \
  || fail "canonical telemetry extractor did not create a non-empty JSON result"

FINAL_CHECKPOINT="$RUN_DIR/model_499.pt"
"$PY" - "$TASK" "$FINAL_CHECKPOINT" <<'PY' \
  || fail "model_499.pt failed strict VIC runner reload"
from dataclasses import asdict
import sys

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.hammer.config.z1  # noqa: F401


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


task, checkpoint = sys.argv[1:]
env_cfg = load_env_cfg(task, play=False)
agent_cfg = load_rl_cfg(task)
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
try:
    runner_cls = load_runner_cls(task)
    require(runner_cls is not None, "VIC task has no registered runner")
    runner = runner_cls(wrapped, asdict(agent_cfg), device="cuda:0")
    runner.load(checkpoint, load_cfg=None, strict=True, map_location="cuda:0")
finally:
    wrapped.close()
PY

TELEMETRY_SHA256=$(sha256sum "$TELEMETRY_JSON" | cut -d ' ' -f 1)
[[ "$TELEMETRY_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "invalid telemetry SHA-256 result"
FINAL_CHECKPOINT_SHA256=""
for CHECKPOINT in "${EXPECTED_CHECKPOINTS[@]}"; do
  CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)
  [[ "$CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || fail "invalid checkpoint SHA-256 result"
  if [[ "$CHECKPOINT" = "$FINAL_CHECKPOINT" ]]; then
    FINAL_CHECKPOINT_SHA256="$CHECKPOINT_SHA256"
  fi
  printf 'Z1_VIC_CANARY_CHECKPOINT_SHA256 basename=%s sha256=%s\n' \
    "$(basename "$CHECKPOINT")" "$CHECKPOINT_SHA256"
done
[[ "$FINAL_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail "final checkpoint SHA-256 identity is missing"

FINAL_CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) \
  || fail "postflight code repository is unavailable"
[[ "$FINAL_CODE_REVISION" = "$EXPECTED_CODE_REVISION" ]] \
  || fail "postflight code revision mismatch"
FINAL_CODE_STATUS=$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all) \
  || fail "postflight code repository status check failed"
[[ -z "$FINAL_CODE_STATUS" ]] || fail "postflight code repository must be clean"
FINAL_ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) \
  || fail "postflight asset repository is unavailable"
[[ "$FINAL_ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] \
  || fail "postflight asset revision mismatch"
FINAL_ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all) \
  || fail "postflight asset repository status check failed"
[[ -z "$FINAL_ASSET_STATUS" ]] || fail "postflight asset repository must be clean"

printf 'Z1_VIC_CANARY_CODE_REVISION=%s\n' "$CODE_REVISION"
printf 'Z1_VIC_CANARY_ASSET_REVISION=%s\n' "$ASSET_REVISION"
printf 'Z1_VIC_CANARY_FINAL_CHECKPOINT=%s\n' "$FINAL_CHECKPOINT"
printf 'Z1_VIC_CANARY_FINAL_CHECKPOINT_SHA256=%s\n' "$FINAL_CHECKPOINT_SHA256"
printf 'Z1_VIC_CANARY_TELEMETRY=%s\n' "$TELEMETRY_JSON"
printf 'Z1_VIC_CANARY_TELEMETRY_SHA256=%s\n' "$TELEMETRY_SHA256"
printf 'VIC_CANARY_DONE seed=2 task=%s\n' "$TASK"
