#!/usr/bin/env bash
#SBATCH --job-name=z1-paired-impact
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%1
#SBATCH --no-requeue
set -euo pipefail

readonly CAMPAIGN="${1:?unique evaluation campaign directory required}"
readonly TRAINING_CAMPAIGN="${2:?frozen six-run training campaign directory required}"
readonly SOURCE="$TRAINING_CAMPAIGN/source"
readonly PY=/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python
readonly TASK_INDEX="${SLURM_ARRAY_TASK_ID:?array index required}"

export PYTHONPATH="$SOURCE:$TRAINING_CAMPAIGN/test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MUJOCO_GL=egl
export WARP_CACHE_PATH="$CAMPAIGN/cache/${SLURM_JOB_ID}_${TASK_INDEX}"
mkdir -p "$WARP_CACHE_PATH" "$CAMPAIGN/logs"
cd "$SOURCE"

sha256sum -c "$TRAINING_CAMPAIGN/source.sha256" > "$WARP_CACHE_PATH/source-check.log"
(cd "$CAMPAIGN/overlay" && sha256sum -c overlay.sha256) > "$WARP_CACHE_PATH/overlay-check.log"
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
"$PY" "$CAMPAIGN/overlay/vega_paired_impact_speed_eval.py" \
    --index "$TASK_INDEX" \
    --training-campaign "$TRAINING_CAMPAIGN" \
    --output "$CAMPAIGN" \
    --evaluator-script "$CAMPAIGN/overlay/evaluate_impact_speed.py"
