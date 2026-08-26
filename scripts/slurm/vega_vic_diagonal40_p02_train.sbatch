#!/usr/bin/env bash
# One bounded engineering canary; it never submits, retries, requeues, evaluates, or trains another task.
#SBATCH --job-name=z1-vic-diagonal40-p02-train
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-diagonal40-p02-train/slurm/z1-vic-diagonal40-p02-train-%j.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/campaigns/z1-vic-diagonal40-p02-train/slurm/z1-vic-diagonal40-p02-train-%j.err
set -euo pipefail
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet MUJOCO_GL=egl
fail(){ echo "Z1_VIC_DIAGONAL40_P02_TRAIN_FAIL: $*"; exit 2; }
[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
for V in PYTHONOPTIMIZE SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT RESUME_PATH TASK ROLE CAPS IMP_MAX_P RUN_NAME; do [[ -z "${!V:-}" ]] || fail "$V must be empty"; done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "SLURM_JOB_ID must be numeric"
for V in EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION; do [[ "${!V:-}" =~ ^[0-9a-f]{40}$ ]] || fail "$V must be a 40-hex revision"; done
[[ -n "${RUN_ROOT:-}" && -n "${ASSET_REPO:-}" ]] || fail "RUN_ROOT and ASSET_REPO are required"
RUN_ROOT=$(cd "$RUN_ROOT" 2>/dev/null && pwd -P) || fail "code repository is unavailable"
CANONICAL_ASSET_REPO=$(cd "$RUN_ROOT/../safe_impact_manipulation" 2>/dev/null && pwd -P) || fail "canonical sibling asset repository is unavailable"
ASSET_REPO=$(cd "$ASSET_REPO" 2>/dev/null && pwd -P) || fail "asset repository is unavailable"
[[ "$ASSET_REPO" = "$CANONICAL_ASSET_REPO" ]] || fail "ASSET_REPO must be the canonical sibling"
CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) || fail "code repository revision is unavailable"
[[ "$CODE_REVISION" =~ ^[0-9a-f]{40}$ && "$CODE_REVISION" = "$EXPECTED_CODE_REVISION" ]] || fail "code revision mismatch"
[[ -z "$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || fail "code repository must be clean"
ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) || fail "asset repository revision is unavailable"
[[ "$ASSET_REVISION" =~ ^[0-9a-f]{40}$ && "$ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" ]] || fail "asset revision mismatch"
[[ -z "$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all)" ]] || fail "asset repository must be clean"
readonly TASK="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-DiagonalStarts40mm-Persistent" ROLE="diagonal40_persistent_p02" CAPS="[0.369,0.246,0.738,0.369,0.246,0.0164]" IMP_MAX_P="0.2"
readonly PY="$HOME/repos/unitree_rl_mjlab/.venv/bin/python"; [[ -x "$PY" ]] || fail "Vega Python environment is unavailable"
readonly CAMPAIGN="z1-vic-diagonal40-p02-train"
readonly RESULT_ROOT="$HOME/campaigns/$CAMPAIGN/runs/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION"
readonly ATTEMPT_DIR="$RESULT_ROOT/${SLURM_JOB_ID}_${ROLE}"
readonly NODE_TMPDIR="/tmp/z1-vic-diagonal40-p02-${SLURM_JOB_ID}"
mkdir -m 700 -- "$NODE_TMPDIR" 2>/dev/null || fail "node-local temporary directory already exists"; trap 'rm -rf -- "$NODE_TMPDIR"' EXIT
mkdir -p "$RESULT_ROOT"; mkdir "$ATTEMPT_DIR" 2>/dev/null || fail "attempt directory already exists"
export PYTHONPATH="$RUN_ROOT" WARP_CACHE_PATH="$ATTEMPT_DIR/warp-cache" TMPDIR="$NODE_TMPDIR" TEMP="$NODE_TMPDIR" TMP="$NODE_TMPDIR" RUN_NAME="victt_seed2_diagonal40_p02_${ROLE}_job${SLURM_JOB_ID}" TASK ROLE CAPS IMP_MAX_P
mkdir "$WARP_CACHE_PATH"
"$PY" - <<'PY' || fail "CUDA identity or runtime package versions do not match"
from importlib.metadata import version
import torch
expected={"mjlab":"1.4.0","mujoco":"3.8.1","mujoco-warp":"3.8.1"}; actual={n:version(n) for n in expected}
if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("exactly one CUDA device is required")
gpu=torch.cuda.get_device_name(0)
if gpu!="NVIDIA A100-SXM4-40GB": raise RuntimeError(f"unexpected GPU: {gpu}")
if actual!=expected: raise RuntimeError(f"runtime package versions do not match: {actual!r}")
PY
"$PY" - <<'PY' || fail "diagonal40 serialized config preflight failed"
import hashlib,json,os
import mjlab.tasks,src.tasks
from mjlab.tasks.registry import load_env_cfg
TASK="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-DiagonalStarts40mm-Persistent"
ROWS=[[0.0,1.487947605122,-0.319221074008,-1.190426531114,-0.0013,1.5544],[0.0,1.606,-0.4301,-1.1976,-0.0013,1.5544],[0.0,1.709237021603,-0.555950051788,-1.174986969814,-0.0013,1.5544]]; CAPS=[.369,.246,.738,.369,.246,.0164]
def fn(f): return f"{f.__module__}.{f.__qualname__}"
def norm(x):
 if x is None or type(x) in (bool,int,float,str): return x
 if isinstance(x,(list,tuple)): return [norm(i) for i in x]
 if isinstance(x,dict): return {str(k):norm(v) for k,v in sorted(x.items())}
 return {"name":x.name,"joint_names":norm(x.joint_names),"site_names":norm(x.site_names),"preserve_order":bool(x.preserve_order)}
def identity(c):
 cat=c.metrics["cat_soft"].params
 return {"task":TASK,"events":[{"name":n,"function":fn(e.func),"mode":e.mode,"params":{k:norm(v) for k,v in sorted(e.params.items())}} for n,e in c.events.items()],"event_names":list(c.events),"route_joint_positions_rad":norm(c.events["reset_strike_route_joints"].params["route_joint_positions"]),"observations":{g:[{"name":n,"followthrough_mode":t.params.get("followthrough_mode")} for n,t in c.observations[g].terms.items()] for g in ("actor","critic")},"actions":list(c.actions),"schedule":{"r_imit_weight":float(c.rewards["r_imit"].weight),"r_imit_sigma":float(c.rewards["r_imit"].params["sigma"]),"r_imit_anneal":None if not c.curriculum else c.curriculum.get("r_imit_anneal")},"impulse_cat":{"caps":norm(cat["imp_limit"]),"max_p":float(cat["imp_max_p"]),"use_impulse":cat["use_impulse"],"use_velocity":cat["use_vel"]},"maximize_rewards":{"delivered_impulse":float(c.rewards["delivered_impulse"].weight),"impact_progress":float(c.rewards["impact_progress"].weight)}}
if os.environ["TASK"]!=TASK or os.environ["ROLE"]!="diagonal40_persistent_p02" or json.loads(os.environ["CAPS"])!=CAPS or os.environ["IMP_MAX_P"]!="0.2": raise RuntimeError("frozen treatment drifted")
asset=lambda n,j:{"name":n,"joint_names":j,"site_names":None,"preserve_order":False}
names=("joint_pos","joint_vel","ee_pos","ee_vel","head_pos","head_vel","nail_top_pos","nail_depth","strike_phase","strike_ref_error","actions")
expected={"task":TASK,"events":[{"name":"sample_strike_route_signs","function":"src.tasks.hammer.mdp.references.sample_strike_route_signs","mode":"reset","params":{"followthrough_mode":"strike_axis","horizontal_detour_m":0.0}},{"name":"reset_robot_joints","function":"mjlab.envs.mdp.events.reset_joints_by_offset","mode":"reset","params":{"asset_cfg":asset("robot",[".*"]),"position_range":[0.,0.],"velocity_range":[0.,0.]}},{"name":"reset_strike_route_joints","function":"src.tasks.hammer.mdp.references.reset_joints_to_strike_route_starts","mode":"reset","params":{"asset_cfg":asset("robot",["joint1","joint2","joint3","joint4","joint5","joint6"]),"route_joint_positions":ROWS}},{"name":"reset_nail","function":"mjlab.envs.mdp.events.reset_joints_by_offset","mode":"reset","params":{"asset_cfg":asset("nail_block",["nail_slide"]),"position_range":[0.,0.],"velocity_range":[0.,0.]}},{"name":"expand_variable_impedance_model_fields","function":"src.tasks.hammer.mdp.variable_impedance.expand_variable_impedance_model_fields","mode":"startup","params":{}}],"event_names":["sample_strike_route_signs","reset_robot_joints","reset_strike_route_joints","reset_nail","expand_variable_impedance_model_fields"],"route_joint_positions_rad":ROWS,"observations":{g:[{"name":n,"followthrough_mode":"strike_axis" if n in ("strike_phase","strike_ref_error") else None} for n in names] for g in ("actor","critic")},"actions":["joint_position","joint_stiffness"],"schedule":{"r_imit_weight":.2,"r_imit_sigma":.05,"r_imit_anneal":None},"impulse_cat":{"caps":CAPS,"max_p":.2,"use_impulse":True,"use_velocity":True},"maximize_rewards":{"delivered_impulse":4.,"impact_progress":8.}}
first,second=identity(load_env_cfg(TASK)),identity(load_env_cfg(TASK)); encoded=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)
if encoded(first)!=encoded(second): raise RuntimeError("repeated config loads drifted")
if encoded(first)!=encoded(expected): raise RuntimeError(f"serialized diagonal40 config identity drifted: {encoded(first)}")
s=encoded(first); print(f"Z1_VIC_DIAGONAL40_P02_TRAIN_CONFIG role=diagonal40_persistent_p02 identity_sha256={hashlib.sha256(s.encode()).hexdigest()} serialized={s}")
PY
cd "$ATTEMPT_DIR"
"$PY" "$RUN_ROOT/scripts/train.py" "$TASK" --gpu-ids '[0]' --agent.logger tensorboard --agent.run-name "$RUN_NAME" --agent.seed 2 --agent.num-steps-per-env 24 --agent.max-iterations 500 --agent.save-interval 50 --env.scene.num-envs 4096 --env.metrics.cat-soft.params.imp-limit "$CAPS" --env.metrics.cat-soft.params.imp-max-p "$IMP_MAX_P"
shopt -s nullglob; RUN_DIRS=("$ATTEMPT_DIR"/logs/rsl_rl/z1_hammer/*_"$RUN_NAME"); [[ "${#RUN_DIRS[@]}" -eq 1 && -d "${RUN_DIRS[0]}" ]] || fail "expected exactly one matching training run directory"; RUN_DIR="${RUN_DIRS[0]}"
readonly EXPECTED_ITERATIONS=(0 50 100 150 200 250 300 350 400 450 499); EXPECTED_CHECKPOINTS=(); for I in "${EXPECTED_ITERATIONS[@]}"; do EXPECTED_CHECKPOINTS+=("$RUN_DIR/model_$I.pt"); done; ACTUAL_CHECKPOINTS=("$RUN_DIR"/model_*.pt); [[ "${#ACTUAL_CHECKPOINTS[@]}" -eq 11 ]] || fail "checkpoint set does not match the exact 500-iteration schedule"; for C in "${EXPECTED_CHECKPOINTS[@]}"; do [[ -f "$C" ]] || fail "checkpoint set does not match the exact 500-iteration schedule"; done
"$PY" - "${EXPECTED_CHECKPOINTS[@]}" <<'PY' || fail "checkpoint identity or finiteness validation failed"
from collections.abc import Mapping
import sys,torch
def ts(x):
 if isinstance(x,torch.Tensor): yield x
 elif isinstance(x,Mapping):
  for v in x.values(): yield from ts(v)
 elif isinstance(x,(list,tuple)):
  for v in x: yield from ts(v)
for i,p in zip((0,50,100,150,200,250,300,350,400,450,499),sys.argv[1:],strict=True):
 s=torch.load(p,map_location="cpu",weights_only=False); v=list(ts(s))
 if type(s.get("iter")) is not int or s["iter"]!=i or not v or not all(bool(torch.isfinite(x).all()) for x in v): raise RuntimeError("invalid checkpoint")
PY
FINAL_CODE_REVISION=$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null) || fail "postflight code revision is unavailable"; FINAL_ASSET_REVISION=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null) || fail "postflight asset revision is unavailable"
[[ "$FINAL_CODE_REVISION" =~ ^[0-9a-f]{40}$ && "$FINAL_CODE_REVISION" = "$EXPECTED_CODE_REVISION" && -z "$(git -C "$RUN_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || fail "postflight code provenance drifted"
[[ "$FINAL_ASSET_REVISION" =~ ^[0-9a-f]{40}$ && "$FINAL_ASSET_REVISION" = "$EXPECTED_ASSET_REVISION" && -z "$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all)" ]] || fail "postflight asset provenance drifted"
for C in "${EXPECTED_CHECKPOINTS[@]}"; do L=$(sha256sum "$C") || fail "checkpoint SHA-256 validation failed"; H="${L%%[[:space:]]*}"; [[ "$H" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint SHA-256 validation failed"; printf 'Z1_VIC_DIAGONAL40_P02_TRAIN_CHECKPOINT role=%s task=%s basename=%s sha256=%s\n' "$ROLE" "$TASK" "$(basename "$C")" "$H"; done
printf 'Z1_VIC_DIAGONAL40_P02_TRAIN_PASS role=%s task=%s imp_max_p=%s caps=%s run_dir=%s\n' "$ROLE" "$TASK" "$IMP_MAX_P" "$CAPS" "$RUN_DIR"
