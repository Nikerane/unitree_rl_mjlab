#!/usr/bin/env bash
#SBATCH --job-name=z1-pd500-regression
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
set -euo pipefail
module load git
readonly CAMPAIGN="${1:?campaign required}"
export PYTHONPATH="$CAMPAIGN/source:/cephhome/eunikhilr/campaigns/z1-speed-preflight-20260921.O9UyYJ/test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MUJOCO_GL=egl
export WARP_CACHE_PATH=/cephhome/eunikhilr/campaigns/z1-quintic-m20-pilot/runs/3ca89a2-seed2-250iter/warp-cache
cd "$CAMPAIGN/source"
sha256sum -c "$CAMPAIGN/source.sha256"
/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python -m pytest -q
