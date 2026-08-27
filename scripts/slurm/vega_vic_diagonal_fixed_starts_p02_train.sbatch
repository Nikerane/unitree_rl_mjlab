#!/usr/bin/env bash
# Five matched fixed-start specialists; this launcher never retries, requeues, resumes, or evaluates.
#SBATCH --job-name=z1-vic-diagonal-fixed-starts-p02-train
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --exclude=gn03,gn13,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-diagonal-fixed-starts-p02-train/slurm/z1-vic-diagonal-fixed-starts-p02-train-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-diagonal-fixed-starts-p02-train/slurm/z1-vic-diagonal-fixed-starts-p02-train-%A_%a.err

set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet MUJOCO_GL=egl

fail(){ echo "Z1_VIC_DIAGONAL_FIXED_STARTS_P02_TRAIN_FAIL: $*"; exit 2; }
[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
for V in PYTHONOPTIMIZE WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT RESUME_PATH TASK ROLE OFFSET_MM FIXED_SIGN CAPS IMP_MAX_P RUN_NAME; do
  [[ -z "${!V:-}" ]] || fail "$V must be empty"
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "SLURM_JOB_ID must be numeric"
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "SLURM_ARRAY_JOB_ID must be numeric"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-4]$ ]] || fail "SLURM_ARRAY_TASK_ID must be 0..4"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" = 5 && "${SLURM_ARRAY_TASK_MIN:-}" = 0 && "${SLURM_ARRAY_TASK_MAX:-}" = 4 && "${SLURM_ARRAY_TASK_STEP:-}" = 1 ]] || fail "array identity must be exactly 0-4"
for V in EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION; do
  [[ "${!V:-}" =~ ^[0-9a-f]{40}$ ]] || fail "$V must be a 40-hex revision"
done
[[ -n "${RUN_ROOT:-}" && -n "${ASSET_REPO:-}" ]] || fail "RUN_ROOT and ASSET_REPO are required"

RUN_ROOT=$(cd "$RUN_ROOT" 2>/dev/null && pwd -P) || fail "code repository is unavailable"
CANONICAL_ASSET_REPO=$(cd "$RUN_ROOT/../safe_impact_manipulation" 2>/dev/null && pwd -P) || fail "canonical sibling asset repository is unavailable"
ASSET_REPO=$(cd "$ASSET_REPO" 2>/dev/null && pwd -P) || fail "asset repository is unavailable"
[[ "$ASSET_REPO" = "$CANONICAL_ASSET_REPO" ]] || fail "ASSET_REPO must be the canonical sibling"
CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) || fail "code repository revision is unavailable"
[[ "$CODE_REVISION" = "$EXPECTED_CODE_REVISION" ]] || fail "code revision mismatch"
CODE_STATUS=$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all) || fail "code repository status check failed"
[[ -z "$CODE_STATUS" ]] || fail "code repository must be clean"
ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) || fail "asset repository revision is unavailable"
[[ "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] || fail "asset revision mismatch"
ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all) || fail "asset repository status check failed"
[[ -z "$ASSET_STATUS" ]] || fail "asset repository must be clean"

TASKS=(
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-DiagonalFixedStartM40mm-Persistent"
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-DiagonalFixedStartM20mm-Persistent"
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-DiagonalFixedStart0mm-Persistent"
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-DiagonalFixedStartP20mm-Persistent"
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-DiagonalFixedStartP40mm-Persistent"
)
ROLES=("fixed_m40mm_p02" "fixed_m20mm_p02" "fixed_0mm_p02" "fixed_p20mm_p02" "fixed_p40mm_p02")
OFFSETS_MM=(-40 -20 0 20 40)
FIXED_SIGNS=(-1 -1 0 1 1)
readonly ARRAY_INDEX="$SLURM_ARRAY_TASK_ID"
readonly TASK="${TASKS[$ARRAY_INDEX]}" ROLE="${ROLES[$ARRAY_INDEX]}"
readonly OFFSET_MM="${OFFSETS_MM[$ARRAY_INDEX]}" FIXED_SIGN="${FIXED_SIGNS[$ARRAY_INDEX]}"
readonly CAPS="[0.369,0.246,0.738,0.369,0.246,0.0164]"
readonly IMP_MAX_P="0.2"
readonly PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"
[[ -x "$PY" ]] || fail "Vega Python environment is unavailable"

readonly CAMPAIGN="z1-vic-diagonal-fixed-starts-p02-train"
readonly RESULT_ROOT="$HOME/campaigns/$CAMPAIGN/runs/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
readonly JOB_KEY="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly ATTEMPT_DIR="$RESULT_ROOT/${JOB_KEY}_${ROLE}"
readonly NODE_TMPDIR="/tmp/z1-vic-diagonal-fixed-${JOB_KEY}"
mkdir -m 700 -- "$NODE_TMPDIR" 2>/dev/null || fail "node-local temporary directory already exists"
trap 'rm -rf -- "$NODE_TMPDIR"' EXIT
mkdir -p "$RESULT_ROOT"
mkdir "$ATTEMPT_DIR" 2>/dev/null || fail "attempt directory already exists"
export PYTHONPATH="$RUN_ROOT" WARP_CACHE_PATH="$ATTEMPT_DIR/warp-cache"
export TMPDIR="$NODE_TMPDIR" TEMP="$NODE_TMPDIR" TMP="$NODE_TMPDIR"
export RUN_NAME="victt_seed2_diagonal_fixed_${ROLE}_job${JOB_KEY}"
export TASK ROLE OFFSET_MM FIXED_SIGN CAPS IMP_MAX_P
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
PY

"$PY" - <<'PY' || fail "fixed-start serialized config preflight failed"
import hashlib
import json
import os
import mjlab.tasks
import src.tasks
from mjlab.tasks.registry import load_env_cfg

rows20 = (
  (0.0, 1.549360913, -0.372460482, -1.198600431, -0.0013, 1.5544),
  (0.0, 1.606, -0.4301, -1.1976, -0.0013, 1.5544),
  (0.0, 1.658994499, -0.491428010, -1.189266489, -0.0013, 1.5544),
)
rows40 = (
  (0.0, 1.487947605122, -0.319221074008, -1.190426531114, -0.0013, 1.5544),
  (0.0, 1.606, -0.4301, -1.1976, -0.0013, 1.5544),
  (0.0, 1.709237021603, -0.555950051788, -1.174986969814, -0.0013, 1.5544),
)
expected = {
  -40: (-1, rows40),
  -20: (-1, rows20),
  0: (0, rows40),
  20: (1, rows20),
  40: (1, rows40),
}
task = os.environ["TASK"]
offset_mm = int(os.environ["OFFSET_MM"])
fixed_sign = int(os.environ["FIXED_SIGN"])
sign, rows = expected[offset_mm]
if sign != fixed_sign:
  raise RuntimeError("fixed start mapping drifted")
cfg = load_env_cfg(task)
qualified_cfg = load_env_cfg(
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT"
)
sampler = cfg.events["sample_strike_route_signs"]
if sampler.params != {"horizontal_detour_m": 0.0, "followthrough_mode": "strike_axis", "fixed_route_sign": sign}:
  raise RuntimeError("fixed route sampler drifted")
route_reset = cfg.events["reset_strike_route_joints"]
if tuple(tuple(row) for row in route_reset.params["route_joint_positions"]) != rows:
  raise RuntimeError("fixed route pose table drifted")
joint_action = cfg.actions["joint_position"]
expected_pose = rows[sign + 1]
joint_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
robot_default = cfg.scene.entities["robot"].init_state.joint_pos
if not joint_action.use_default_offset or joint_action.offset != 0.0:
  raise RuntimeError("joint_action.offset must retain default-offset semantics")
if tuple(robot_default[name] for name in joint_names) != rows20[1]:
  raise RuntimeError("common robot default pose drifted")
if tuple(cfg.actions) != ("joint_position", "joint_stiffness"):
  raise RuntimeError("action layout drifted")
for action_name in cfg.actions:
  action = cfg.actions[action_name]
  qualified_action = qualified_cfg.actions[action_name]
  if type(action) is not type(qualified_action) or vars(action) != vars(qualified_action):
    raise RuntimeError(f"{action_name} mapping drifted from the qualified VIC-TT contract")
r_imit = cfg.rewards["r_imit"]
if r_imit.weight != 0.2 or r_imit.params["sigma"] != 0.05 or r_imit.params["followthrough_mode"] != "strike_axis" or "r_imit_anneal" in cfg.curriculum:
  raise RuntimeError("r_imit treatment drifted")
cat = cfg.metrics["cat_soft"].params
caps = json.loads(os.environ["CAPS"])
if cat["imp_limit"] != caps or cat["imp_max_p"] != 0.2 or not cat["use_impulse"] or not cat["use_vel"]:
  raise RuntimeError("CaT treatment drifted")
if cfg.rewards["delivered_impulse"].weight != 4.0 or cfg.rewards["impact_progress"].weight != 8.0:
  raise RuntimeError("impact objective drifted")
obs_names = tuple(cfg.observations["actor"].terms)
if obs_names != ("joint_pos", "joint_vel", "ee_pos", "ee_vel", "head_pos", "head_vel", "nail_top_pos", "nail_depth", "strike_phase", "strike_ref_error", "actions"):
  raise RuntimeError("actor observation layout drifted")
summary = {"task": task, "role": os.environ["ROLE"], "offset_mm": offset_mm, "fixed_sign": sign, "pose": expected_pose, "caps": caps, "imp_max_p": 0.2, "r_imit": {"weight": 0.2, "sigma": 0.05}, "action_offset": joint_action.offset, "default_pose": rows20[1], "observations": obs_names}
encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
print(f"Z1_VIC_DIAGONAL_FIXED_STARTS_P02_TRAIN_CONFIG role={os.environ['ROLE']} identity_sha256={hashlib.sha256(encoded.encode()).hexdigest()} serialized={encoded}")
PY

cd "$ATTEMPT_DIR"
"$PY" "$RUN_ROOT/scripts/train.py" "$TASK" --gpu-ids '[0]' --agent.logger tensorboard --agent.run-name "$RUN_NAME" --agent.seed 2 --agent.num-steps-per-env 24 --agent.max-iterations 500 --agent.save-interval 50 --env.scene.num-envs 4096 --env.metrics.cat-soft.params.imp-limit "$CAPS" --env.metrics.cat-soft.params.imp-max-p "$IMP_MAX_P"

shopt -s nullglob
RUN_DIRS=("$ATTEMPT_DIR"/logs/rsl_rl/z1_hammer/*_"$RUN_NAME")
[[ "${#RUN_DIRS[@]}" -eq 1 && -d "${RUN_DIRS[0]}" ]] || fail "expected exactly one matching training run directory"
RUN_DIR="${RUN_DIRS[0]}"
readonly EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499)
EXPECTED_CHECKPOINTS=()
for I in "${EXPECTED_ITERATIONS[@]}"; do EXPECTED_CHECKPOINTS+=("$RUN_DIR/model_$I.pt"); done
ACTUAL_CHECKPOINTS=("$RUN_DIR"/model_*.pt)
[[ "${#ACTUAL_CHECKPOINTS[@]}" -eq 11 ]] || fail "checkpoint set does not match the exact 500-iteration schedule"
for C in "${EXPECTED_CHECKPOINTS[@]}"; do [[ -f "$C" ]] || fail "checkpoint set does not match the exact 500-iteration schedule"; done
"$PY" - "${EXPECTED_CHECKPOINTS[@]}" <<'PY' || fail "checkpoint identity or finiteness validation failed"
from collections.abc import Mapping
import sys
import torch
def tensors(value):
  if isinstance(value, torch.Tensor): yield value
  elif isinstance(value, Mapping):
    for item in value.values(): yield from tensors(item)
  elif isinstance(value, (list, tuple)):
    for item in value: yield from tensors(item)
for iteration, path in zip((0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499), sys.argv[1:], strict=True):
  state = torch.load(path, map_location="cpu", weights_only=False)
  values = list(tensors(state))
  if type(state.get("iter")) is not int or state["iter"] != iteration or not values or not all(bool(torch.isfinite(value).all()) for value in values):
    raise RuntimeError("invalid checkpoint")
PY

FINAL_CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) || fail "postflight code revision is unavailable"
FINAL_ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) || fail "postflight asset revision is unavailable"
FINAL_CODE_STATUS=$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all) || fail "postflight code status check failed"
FINAL_ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all) || fail "postflight asset status check failed"
[[ "$FINAL_CODE_REVISION" = "$EXPECTED_CODE_REVISION" && -z "$FINAL_CODE_STATUS" ]] || fail "postflight code provenance drifted"
[[ "$FINAL_ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" && -z "$FINAL_ASSET_STATUS" ]] || fail "postflight asset provenance drifted"
for C in "${EXPECTED_CHECKPOINTS[@]}"; do
  LINE=$(sha256sum "$C") || fail "checkpoint SHA-256 validation failed"
  HASH="${LINE%%[[:space:]]*}"
  [[ "$HASH" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint SHA-256 validation failed"
  printf 'Z1_VIC_DIAGONAL_FIXED_STARTS_P02_TRAIN_CHECKPOINT role=%s task=%s basename=%s sha256=%s\n' "$ROLE" "$TASK" "$(basename "$C")" "$HASH"
done
printf 'Z1_VIC_DIAGONAL_FIXED_STARTS_P02_TRAIN_PASS role=%s task=%s offset_mm=%s imp_max_p=%s caps=%s run_dir=%s\n' "$ROLE" "$TASK" "$OFFSET_MM" "$IMP_MAX_P" "$CAPS" "$RUN_DIR"
