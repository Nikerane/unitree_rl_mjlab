#!/usr/bin/env bash
# Staged prior-vs-none training array on Vega (Codex's 1-GPU-per-task pattern).
# One array task = one (arm, seed), interleaved track/none. 500 iters (post-anneal treatment,
# per Codex), 4096 envs, log-only (imp_max_p=0), substep_rows OFF, unique campaign run names.
# Submit:
#   validity : CAMPAIGN=nf1 SEEDS="0 1 2"       sbatch --array=0-5  --export=ALL,CAMPAIGN,SEEDS scripts/slurm/vega_train.sbatch
#   expansion: CAMPAIGN=nf1 SEEDS="3 4 5 6 7"   sbatch --array=0-9  --export=ALL,CAMPAIGN,SEEDS scripts/slurm/vega_train.sbatch
#SBATCH --job-name=z1-train
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
# Known-bad nodes that kill jobs during SLURM setup (FAILED, ExitCode 0:53, NO log — the in-script A100
# guard cannot catch a pre-exec failure). Remove once repaired. A submit-time --exclude OVERRIDES this,
# so pass the FULL list (gn03,gn34,...) if adding nodes at submit.
#SBATCH --exclude=gn03,gn34
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/repos/unitree_rl_mjlab}"
mkdir -p logs
module load git 2>/dev/null || true       # git on compute node (for provenance logging)
export GIT_PYTHON_REFRESH=quiet            # don't crash if GitPython can't validate git
export PYTHONPATH=.
export WARP_CACHE_PATH="$PWD/.warp_cache"
PY=.venv/bin/python
CAMPAIGN="${CAMPAIGN:-nf}"
read -ra SEEDARR <<< "${SEEDS:-0 1 2}"
# nail_driven=0.5 is now the BAKED default (farm-cut). Optional per-run reward-weight overrides:
#   DELIVERED_W=0  -> F2 arm (delivered-impulse reward off, tests its contribution to the strike).
#   NAIL_DRIVEN_W  -> legacy escape hatch; normally leave empty (baked 0.5 is authoritative).
EXTRA_FLAGS=""
[ -n "${NAIL_DRIVEN_W:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --env.rewards.nail-driven.weight ${NAIL_DRIVEN_W}"
[ -n "${DELIVERED_W:-}" ]   && EXTRA_FLAGS="$EXTRA_FLAGS --env.rewards.delivered-impulse.weight ${DELIVERED_W}"
# DEPTH_GATE=True|False -> Phase-2 depth-gate sweep (False = pay delivered impulse into a seated nail).
[ -n "${DEPTH_GATE:-}" ]    && EXTRA_FLAGS="$EXTRA_FLAGS --env.rewards.delivered-impulse.params.depth-gate ${DEPTH_GATE}"
# IMPACT_W -> Phase-2b: strengthen the ante-impact velocity term (impact_progress).
[ -n "${IMPACT_W:-}" ]      && EXTRA_FLAGS="$EXTRA_FLAGS --env.rewards.impact-progress.weight ${IMPACT_W}"
ARM_TASK=(Unitree-Z1-Hammer-CaT-Impulse-Track Unitree-Z1-Hammer-CaT-Impulse)
ARM_SHORT=(track none)
t=${SLURM_ARRAY_TASK_ID:?must run as an array}
# SINGLE_TASK mode (Option B / any one-arm campaign): array index == seed index, one fixed task.
# Default (paired) mode: index -> (seed = SEEDS[t/2], arm = track/none by t%2).
if [ -n "${SINGLE_TASK:-}" ]; then
  seed=${SEEDARR[$t]}; task="$SINGLE_TASK"; short="${SINGLE_SHORT:-single}"
else
  seed=${SEEDARR[$((t/2))]}; ai=$((t%2)); task=${ARM_TASK[$ai]}; short=${ARM_SHORT[$ai]}
fi
RUN="${CAMPAIGN}_${short}_seed${seed}"
echo "### TRAIN task=$t arm=$short seed=$seed run=$RUN host=$(hostname) $(date '+%F %T')"
$PY -c "import torch; assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0), 'no A100'; print('A100 ok:', torch.cuda.get_device_name(0))" || { echo "### NO GPU/A100 on $(hostname) — requeue+exit"; scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null; exit 1; }
$PY scripts/train.py "$task" --gpu-ids [0] --agent.logger tensorboard \
  --agent.run-name "$RUN" --agent.seed "$seed" --agent.max-iterations 500 --agent.save-interval 50 \
  --env.scene.num-envs 4096 --env.metrics.cat-soft.params.imp-max-p 0 \
  --env.metrics.substep-impulse-rows.params.enabled False $EXTRA_FLAGS
rc=$?
echo "### TRAIN_DONE $RUN rc=$rc $(date '+%F %T')"
exit $rc
