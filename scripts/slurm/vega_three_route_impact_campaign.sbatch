#!/usr/bin/env bash
#SBATCH --job-name=z1-three-route-speed
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
module load git
readonly CAMPAIGN="${1:?isolated campaign directory required}"
readonly PHASE="${2:?preflight, train, or audit required}"
readonly PY=/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python
export PYTHONPATH="$CAMPAIGN/source:$CAMPAIGN/test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MUJOCO_GL=egl
export WARP_CACHE_PATH="$CAMPAIGN/cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}"
mkdir -p "$WARP_CACHE_PATH"
cd "$CAMPAIGN/source"
sha256sum -c "$CAMPAIGN/source.sha256" > "$WARP_CACHE_PATH/source-check.log"
(cd "$CAMPAIGN/safe_impact_manipulation/hammer_z1_env/assets" && sha256sum -c "$CAMPAIGN/source/docs/results/2026-09-24-free-time-spline-external-assets.sha256")
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
if [[ "$PHASE" == preflight ]]; then
    "$PY" -m pytest -q tests/test_vega_three_route_impact_campaign.py tests/test_impact_speed_evaluation.py -k 'not short_three_route_policy_mode_steps_latch_speed_once_per_episode' > "$CAMPAIGN/task-2-tests.log" 2>&1
    "$PY" scripts/vega_three_route_impact_campaign.py --phase preflight --output "$CAMPAIGN/preflight"
elif [[ "$PHASE" == train ]]; then
    test -f "$CAMPAIGN/preflight/PASS"
    "$PY" scripts/vega_three_route_impact_campaign.py --phase train --index "${SLURM_ARRAY_TASK_ID:?array index required}" --output "$CAMPAIGN/runs/run-$SLURM_ARRAY_TASK_ID"
elif [[ "$PHASE" == audit ]]; then
    "$PY" scripts/vega_three_route_impact_campaign.py --phase audit --output "$CAMPAIGN"
else
    exit 2
fi
