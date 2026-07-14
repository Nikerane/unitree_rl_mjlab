#!/usr/bin/env bash
# V1+ training: one GPU per array task; task id = seed shard.
# V1 (A-BASE, Path-A): sbatch --array=0-2 scripts/slurm/train_array.sbatch
# Override iters/envs/run-name via env vars when submitting, e.g.:
#   ITERS=5000 NENVS=4096 RUN=a_base sbatch --array=0-2 scripts/slurm/train_array.sbatch
# EXTRA passes verbatim CLI overrides (velocity-bound ablation arms), e.g.:
#   RUN=c_a1 EXTRA="--env.actions.ik-hammer-head.delta-pos-scale 0.10" sbatch --array=0-2 ...
#SBATCH --job-name=z1-train
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ITERS="${ITERS:-5000}"
NENVS="${NENVS:-4096}"
RUN="${RUN:-a_base}"
TASK="${TASK:-Unitree-Z1-Hammer}"   # arm select: Unitree-Z1-Hammer (A-BASE) | -Track (A-TRACK)
EXTRA="${EXTRA:-}"   # optional verbatim CLI overrides (ablation arms); empty by default
SEED="${SLURM_ARRAY_TASK_ID:-0}"

echo "[train] run=$RUN seed=$SEED iters=$ITERS envs=$NENVS task=$TASK extra='$EXTRA' node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

.venv/bin/python scripts/train.py "$TASK" \
  --gpu-ids '[0]' \
  --agent.logger tensorboard \
  --agent.run-name "${RUN}_seed${SEED}" \
  --agent.seed "$SEED" \
  --agent.max-iterations "$ITERS" \
  --env.scene.num-envs "$NENVS" \
  $EXTRA

echo "TRAIN_DONE run=$RUN seed=$SEED"
