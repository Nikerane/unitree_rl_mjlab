#!/usr/bin/env bash
#SBATCH --job-name=z1-pd500-strike
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=cpu
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
module load git
readonly SOURCE=/cephhome/eunikhilr/repos/unitree_rl_mjlab-impact-speed-cc46d99
readonly ASSETS=/cephhome/eunikhilr/repos/safe_impact_manipulation
readonly PRIOR=/cephhome/eunikhilr/campaigns/z1-wrist-calibration-20260922.7DPuER
readonly RUNROOT="${1:?output root required}"
test "$(git -C "$SOURCE" rev-parse HEAD)" = cc46d9998a3c9cd5d8813e89b57e44248353e03e
test "$(git -C "$ASSETS" rev-parse HEAD)" = b58ccd2f81fd246f27c1e8d88cf86484cd888703
git -C "$SOURCE" diff --exit-code
git -C "$SOURCE" diff --cached --exit-code
git -C "$ASSETS" diff --exit-code
git -C "$ASSETS" diff --cached --exit-code
arm=$((1 + 2 * (SLURM_ARRAY_TASK_ID % 2)))
observer=()
if (( SLURM_ARRAY_TASK_ID >= 2 )); then observer=(--observe-contact); fi
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH="$SOURCE:/cephhome/eunikhilr/campaigns/z1-speed-preflight-20260921.O9UyYJ/test-deps"
export WARP_CACHE_PATH=/cephhome/eunikhilr/campaigns/z1-quintic-m20-pilot/runs/3ca89a2-seed2-250iter/warp-cache
cd "$SOURCE"
/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python "$RUNROOT/replay_sampled.py" \
  --prior-tools "$PRIOR/cpu-reaction-telemetry-20260923" \
  --capture "$PRIOR/engineering-validation/contact-$arm-capture" \
  --kp-scale .5 --arm-kd 6.4 --device cpu "${observer[@]}" --output "$RUNROOT/arm-${SLURM_ARRAY_TASK_ID}"
