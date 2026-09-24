#!/usr/bin/env bash
#SBATCH --job-name=z1-pd500-eval-check
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
module load git
readonly CAMPAIGN="${1:?campaign required}"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$CAMPAIGN/source:/cephhome/eunikhilr/campaigns/z1-speed-preflight-20260921.O9UyYJ/test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export WARP_CACHE_PATH=/cephhome/eunikhilr/campaigns/z1-quintic-m20-pilot/runs/3ca89a2-seed2-250iter/warp-cache
cd "$CAMPAIGN/source"
/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python "$CAMPAIGN/evaluate.py" --checkpoint "$CAMPAIGN/smoke/model_1.pt" --output "$CAMPAIGN/evaluator-smoke-v3" --num-envs 2 --device cpu --max-steps 4
