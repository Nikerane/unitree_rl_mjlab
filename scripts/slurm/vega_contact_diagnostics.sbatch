#!/usr/bin/env bash
#SBATCH --job-name=z1-contact-diagnostics
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
module load git
readonly DIAGNOSTIC="${1:?new diagnostic directory required}"
readonly CHECKPOINT_CAMPAIGN="${2:?completed checkpoint campaign required}"
readonly PY=/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python
export PYTHONPATH="$DIAGNOSTIC/source:$DIAGNOSTIC/test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MUJOCO_GL=egl
export WARP_CACHE_PATH="$DIAGNOSTIC/cache/$SLURM_JOB_ID"
mkdir -p "$WARP_CACHE_PATH"
cd "$DIAGNOSTIC/source"
sha256sum -c "$DIAGNOSTIC/source.sha256" > "$DIAGNOSTIC/source-check.log"
(cd "$DIAGNOSTIC/safe_impact_manipulation/hammer_z1_env/assets" && sha256sum -c "$DIAGNOSTIC/source/docs/results/2026-09-24-free-time-spline-external-assets.sha256")
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_contact_trace.py > "$DIAGNOSTIC/observer-tests.log" 2>&1
for index in 0 1 2 3 4 5; do
    gain_mode=fixed
    if (( index % 2 )); then gain_mode=variable; fi
    checkpoint="$CHECKPOINT_CAMPAIGN/runs/run-$index/model_49.pt"
    "$PY" -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); c=json.loads((p.parent/"completion.json").read_text()); assert c["iterations"]==50 and c["reload"]=="passed" and c["evaluation"]=="completed"; assert hashlib.sha256(p.read_bytes()).hexdigest()==c["checkpoint_sha256"]' "$checkpoint"
    "$PY" scripts/evaluate_impact_speed.py \
        --task Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed \
        --checkpoint "$checkpoint" --speed-bounds 1.5 2.2 --speed-cost .8493676 \
        --seed 20260924 --num-envs 32 --mode external --goals 1.5 1.85 2.2 \
        --output "$DIAGNOSTIC/results/run-$index" --device cuda:0 \
        --sampled-pd500 --sampled-pd500-gain-mode "$gain_mode" \
        --measurement-protocol common-window --trace-contact
done
