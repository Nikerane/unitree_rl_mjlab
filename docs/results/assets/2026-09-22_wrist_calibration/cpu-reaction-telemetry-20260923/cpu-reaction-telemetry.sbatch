#!/usr/bin/env bash
#SBATCH --job-name=z1-cpu-reaction
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=1,3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
module load git
readonly SOURCE=/cephhome/eunikhilr/repos/unitree_rl_mjlab-impact-speed-cc46d99
readonly ASSETS=/cephhome/eunikhilr/repos/safe_impact_manipulation
readonly VALIDATION=/cephhome/eunikhilr/campaigns/z1-wrist-calibration-20260922.7DPuER/engineering-validation
readonly RUNROOT="${1:?fresh output/tool root required}"
readonly CPU_REPEAT="${2:?passed CPU-repeat root required}"
readonly CPU_TIMESTEP="${3:?CPU-timestep root required}"
readonly DT="${4:?requested timestep (.00025 or .000125) required}"
case "$DT" in
  .00025) readonly LEVEL=dt-0p25ms ;;
  .000125) readonly LEVEL=dt-0p125ms ;;
  *) printf 'Unsupported timestep: %s\n' "$DT" >&2; exit 2 ;;
esac
test "$(git -C "$SOURCE" rev-parse HEAD)" = cc46d9998a3c9cd5d8813e89b57e44248353e03e
test "$(git -C "$ASSETS" rev-parse HEAD)" = b58ccd2f81fd246f27c1e8d88cf86484cd888703
git -C "$SOURCE" diff --exit-code
git -C "$SOURCE" diff --cached --exit-code
git -C "$ASSETS" diff --exit-code
git -C "$ASSETS" diff --cached --exit-code
test ! -e "$RUNROOT/arm-${SLURM_ARRAY_TASK_ID}/$LEVEL"
printf 'Node: %s; CPU telemetry %s (reserved GPU unused)\n' "$(hostname)" "$DT"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$SOURCE:/cephhome/eunikhilr/campaigns/z1-speed-preflight-20260921.O9UyYJ/test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export WARP_CACHE_PATH=/cephhome/eunikhilr/campaigns/z1-quintic-m20-pilot/runs/3ca89a2-seed2-250iter/warp-cache
cd "$SOURCE"
/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python "$RUNROOT/cpu_reaction_telemetry.py" \
  --validation-root "$VALIDATION" --coarse-reference-root "$CPU_REPEAT" \
  --fine-reference-root "$CPU_TIMESTEP" --arm "$SLURM_ARRAY_TASK_ID" --dt "$DT" \
  --output "$RUNROOT/arm-${SLURM_ARRAY_TASK_ID}"
