#!/usr/bin/env bash
# V0 sanity: prove the GPU training pipeline works end-to-end (short).
#SBATCH --job-name=z1-sanity
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "[sanity] node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# 50 iters, 2048 envs — finishes in a couple minutes if warp+CUDA are healthy.
.venv/bin/python scripts/train.py Unitree-Z1-Hammer \
  --gpu-ids '[0]' \
  --agent.logger tensorboard \
  --agent.run-name sanity \
  --agent.max-iterations 50 \
  --agent.save-interval 50 \
  --env.scene.num-envs 2048

echo "[sanity] checkpoints written:"
ls -t logs/rsl_rl/z1_hammer/*sanity*/model_*.pt 2>/dev/null | head -3
echo "GPU_TRAIN_OK"
