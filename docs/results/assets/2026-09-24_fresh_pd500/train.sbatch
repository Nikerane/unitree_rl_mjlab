#!/usr/bin/env bash
#SBATCH --job-name=z1-pd500-fresh250
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
module load git
readonly CAMPAIGN="${1:?campaign required}"
readonly SOURCE="$CAMPAIGN/source"
readonly PY=/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python
export PYTHONPATH="$SOURCE:/cephhome/eunikhilr/campaigns/z1-speed-preflight-20260921.O9UyYJ/test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MUJOCO_GL=egl
export WARP_CACHE_PATH=/cephhome/eunikhilr/campaigns/z1-quintic-m20-pilot/runs/3ca89a2-seed2-250iter/warp-cache
cd "$SOURCE"
sha256sum -c "$CAMPAIGN/source.sha256"
test -f "$CAMPAIGN/smoke/completion.json"
"$PY" "$CAMPAIGN/pilot.py" --phase train --output "$CAMPAIGN/train250" > "$CAMPAIGN/train250.log" 2>&1
test -f "$CAMPAIGN/train250/completion.json"
"$PY" "$CAMPAIGN/evaluate.py" --checkpoint "$CAMPAIGN/train250/model_249.pt" --output "$CAMPAIGN/evaluation" > "$CAMPAIGN/evaluation.log" 2>&1
