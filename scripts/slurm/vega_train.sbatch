#!/usr/bin/env bash
# One GPU per (arm, seed). The preregistered fsr4x8 campaign deliberately uses
# the existing SINGLE_TASK/IMPACT_W/DELIVERED_W hooks; there is no campaign
# framework hidden in this launcher. Prepare (do not submit before the clean
# provenance + four-task CUDA smoke gates):
#   CAMPAIGN=fsr4x8 SEEDS="0 1 2 3 4 5 6 7" SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse SINGLE_SHORT=c IMPACT_W=8 DELIVERED_W=2 ITERS=500 \
#     EXPECTED_CODE_REVISION=<40hex> EXPECTED_ASSET_REVISION=<40hex> \
#     sbatch --array=0-7 --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION scripts/slurm/vega_train.sbatch
# Repeat with task/short:
#   Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy / dprime
#   Unitree-Z1-Hammer-CaT-Impulse-Event-Linear       / f
#   Unitree-Z1-Hammer-CaT-Impulse-Event              / e
#
# fq4x8 (first-contact quality campaign): reward weights are baked into the
# REGISTERED TASK, not CLI overrides -- IMPACT_W/DELIVERED_W/NAIL_DRIVEN_W must
# be unset (a leaked --export=ALL from an fsr4x8 shell must not silently
# mis-train). Seeds 8..15, ITERS=500, one of exactly four task/short pairs:
#   CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear SINGLE_SHORT=f8 ITERS=500 \
#     EXPECTED_CODE_REVISION=<40hex> EXPECTED_ASSET_REVISION=<40hex> \
#     sbatch --array=0-7 --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION scripts/slurm/vega_train.sbatch
# Repeat with task/short:
#   Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0    / f0
#   Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0    / d0
#   Unitree-Z1-Hammer-CaT-Impulse-Event-Quality      / fq
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
REPO_ROOT=$(pwd -P)
CANONICAL_ASSET_REPO=$(
  cd "$REPO_ROOT/../safe_impact_manipulation" 2>/dev/null && pwd -P
) || { echo "### TRAIN_FAIL: canonical sibling asset repository is missing"; exit 2; }
if [ -n "${ASSET_REPO:-}" ]; then
  RESOLVED_ASSET_REPO=$(
    cd "$ASSET_REPO" 2>/dev/null && pwd -P
  ) || { echo "### TRAIN_FAIL: ASSET_REPO cannot be resolved"; exit 2; }
  [ "$RESOLVED_ASSET_REPO" = "$CANONICAL_ASSET_REPO" ] \
    || { echo "### TRAIN_FAIL: ASSET_REPO must be the canonical sibling repository"; exit 2; }
fi
ASSET_REPO="$CANONICAL_ASSET_REPO"
mkdir -p logs
module load git 2>/dev/null || true       # git on compute node (for provenance logging)
export GIT_PYTHON_REFRESH=quiet            # don't crash if GitPython can't validate git
export PYTHONPATH=.
export WARP_CACHE_PATH="$HOME/.cache/unitree_rl_mjlab/warp"
mkdir -p "$WARP_CACHE_PATH"
PY=.venv/bin/python
CAMPAIGN="${CAMPAIGN:-nf}"
# Reject typos of a known campaign name (e.g. "Fq4x8") that would otherwise
# fall through to the unguarded default path below with leaked overrides,
# wrong seeds, and no revision-equality enforcement. Exact matches proceed to
# their dedicated guard blocks; unrelated campaign names are untouched.
CAMPAIGN_LC=$(printf '%s' "$CAMPAIGN" | tr '[:upper:]' '[:lower:]')
if [ "$CAMPAIGN_LC" = "fq4x8" ] && [ "$CAMPAIGN" != "fq4x8" ]; then
  echo "FQ4X8_FAIL: CAMPAIGN '$CAMPAIGN' looks like fq4x8 but is not an exact, case-sensitive match"
  exit 2
fi
if [ "$CAMPAIGN_LC" = "fsr4x8" ] && [ "$CAMPAIGN" != "fsr4x8" ]; then
  echo "FSR4X8_FAIL: CAMPAIGN '$CAMPAIGN' looks like fsr4x8 but is not an exact, case-sensitive match"
  exit 2
fi
read -ra SEEDARR <<< "${SEEDS:-0 1 2}"
# nail_driven=0.5 is now the BAKED default (farm-cut). Optional per-run reward-weight overrides:
#   DELIVERED_W=0  -> F2 arm (delivered-impulse reward off, tests its contribution to the strike).
#   NAIL_DRIVEN_W  -> legacy escape hatch; normally leave empty (baked 0.5 is authoritative).
EXTRA_FLAGS=""
[ -n "${NAIL_DRIVEN_W:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --env.rewards.nail-driven.weight ${NAIL_DRIVEN_W}"
[ -n "${DELIVERED_W:-}" ]   && EXTRA_FLAGS="$EXTRA_FLAGS --env.rewards.delivered-impulse.weight ${DELIVERED_W}"
# IMPACT_W -> Phase-2b: strengthen the ante-impact velocity term (impact_progress).
[ -n "${IMPACT_W:-}" ]      && EXTRA_FLAGS="$EXTRA_FLAGS --env.rewards.impact-progress.weight ${IMPACT_W}"
# ITERS -> longer/shorter training (default 500); pair a bigger --time at submit for long runs.
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

# Fail closed on any drift from the frozen 4x8 matrix. Other historical
# campaigns retain the launcher's existing behavior.
if [ "$CAMPAIGN" = "fsr4x8" ]; then
  [ "${SEEDS:-}" = "0 1 2 3 4 5 6 7" ] || { echo "FSR4X8_FAIL: seeds must be exactly 0..7"; exit 2; }
  [ "${IMPACT_W:-}" = "8" ] || { echo "FSR4X8_FAIL: IMPACT_W must be 8"; exit 2; }
  [ "${DELIVERED_W:-}" = "2" ] || { echo "FSR4X8_FAIL: DELIVERED_W must be 2"; exit 2; }
  [ "${ITERS:-500}" = "500" ] || { echo "FSR4X8_FAIL: ITERS must be 500"; exit 2; }
  [ -z "${NAIL_DRIVEN_W:-}" ] || { echo "FSR4X8_FAIL: NAIL_DRIVEN_W must be unset"; exit 2; }
  [ -n "${EXPECTED_CODE_REVISION:-}" ] \
    || { echo "FSR4X8_FAIL: EXPECTED_CODE_REVISION is required"; exit 2; }
  [ -n "${EXPECTED_ASSET_REVISION:-}" ] \
    || { echo "FSR4X8_FAIL: EXPECTED_ASSET_REVISION is required"; exit 2; }
  case "$task:$short" in
    Unitree-Z1-Hammer-CaT-Impulse:c|\
    Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy:dprime|\
    Unitree-Z1-Hammer-CaT-Impulse-Event-Linear:f|\
    Unitree-Z1-Hammer-CaT-Impulse-Event:e) ;;
    *) echo "FSR4X8_FAIL: SINGLE_TASK/SINGLE_SHORT is not a frozen arm"; exit 2 ;;
  esac
fi

# fq4x8 INVERTS the fsr4x8 rule: reward weights come from the registered task,
# not CLI overrides, so every override hook must be unset (reject inherited
# --export=ALL leakage). Fail closed before any provenance/GPU work.
if [ "$CAMPAIGN" = "fq4x8" ]; then
  [ "${SEEDS:-}" = "8 9 10 11 12 13 14 15" ] \
    || { echo "FQ4X8_FAIL: seeds must be exactly 8..15"; exit 2; }
  # Exact presence + value: "${ITERS:-500}" = "500" would silently accept both an unset AND an
  # explicitly empty ITERS (the ":-" operator treats empty the same as unset). ITERS=500 is a
  # frozen contract for this campaign, not a default -- an omitted/empty override must fail closed.
  [ "${ITERS+x}" = x ] && [ "$ITERS" = 500 ] \
    || { echo "FQ4X8_FAIL: ITERS must be exactly 500 (set, not omitted or empty)"; exit 2; }
  # "must be unset" means unset, not merely empty: "-z ${VAR:-}" accepts an exported-but-empty
  # override (e.g. IMPACT_W="" leaked via --export=ALL), silently passing a value that would later
  # be treated as absent. "${VAR+x}" only expands when VAR is actually set (even to "").
  [ -z "${IMPACT_W+x}" ] \
    || { echo "FQ4X8_FAIL: IMPACT_W must be unset (weights come from the registered task)"; exit 2; }
  [ -z "${DELIVERED_W+x}" ] \
    || { echo "FQ4X8_FAIL: DELIVERED_W must be unset (weights come from the registered task)"; exit 2; }
  [ -z "${NAIL_DRIVEN_W+x}" ] \
    || { echo "FQ4X8_FAIL: NAIL_DRIVEN_W must be unset (weights come from the registered task)"; exit 2; }
  [[ "${EXPECTED_CODE_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "FQ4X8_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision"; exit 2; }
  [[ "${EXPECTED_ASSET_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "FQ4X8_FAIL: EXPECTED_ASSET_REVISION must be a clean 40-hex revision"; exit 2; }
  case "$task:$short" in
    Unitree-Z1-Hammer-CaT-Impulse-Event-Linear:f8|\
    Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0:f0|\
    Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0:d0|\
    Unitree-Z1-Hammer-CaT-Impulse-Event-Quality:fq) ;;
    *) echo "FQ4X8_FAIL: SINGLE_TASK/SINGLE_SHORT is not a frozen fq4x8 arm"; exit 2 ;;
  esac
fi
RUN="${CAMPAIGN}_${short}_seed${seed}"
# Provenance: record the exact code state that trains (git hash + '-dirty' if the tree has uncommitted
# tracked edits). Deploy via git pull, NOT scp, so this hash is trustworthy (2026-07-19 provenance fix).
GITHASH=$(git rev-parse HEAD 2>/dev/null || echo unknown)
GITSTATUS=$(git status --porcelain=v1 --untracked-files=all 2>/dev/null || echo provenance-unavailable)
[ -n "$GITSTATUS" ] && GITHASH="${GITHASH}-dirty"
echo "### TRAIN task=$t arm=$short seed=$seed run=$RUN git=$GITHASH host=$(hostname) $(date '+%F %T')"
[ "$GITHASH" != "unknown" ] && [ "${GITHASH%-dirty}" = "$GITHASH" ] \
  || { echo "### TRAIN_FAIL: code provenance is unknown or dirty"; exit 2; }
ASSET_HASH=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null || echo unknown)
ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all 2>/dev/null || echo provenance-unavailable)
[ "$ASSET_HASH" != "unknown" ] && [ -z "$ASSET_STATUS" ] \
  || { echo "### TRAIN_FAIL: asset provenance is unknown or dirty"; exit 2; }
echo "### TRAIN_ASSET git=$ASSET_HASH repo=$ASSET_REPO"
if [ "$CAMPAIGN" = "fsr4x8" ]; then
  [ "$GITHASH" = "$EXPECTED_CODE_REVISION" ] \
    || { echo "### TRAIN_FAIL: code revision does not equal expected code revision"; exit 2; }
  [ "$ASSET_HASH" = "$EXPECTED_ASSET_REVISION" ] \
    || { echo "### TRAIN_FAIL: asset revision does not equal expected asset revision"; exit 2; }
fi
if [ "$CAMPAIGN" = "fq4x8" ]; then
  [ "$GITHASH" = "$EXPECTED_CODE_REVISION" ] \
    || { echo "FQ4X8_FAIL: code revision does not equal expected code revision"; exit 2; }
  [ "$ASSET_HASH" = "$EXPECTED_ASSET_REVISION" ] \
    || { echo "FQ4X8_FAIL: asset revision does not equal expected asset revision"; exit 2; }
fi
$PY -c "import torch; assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0), 'no A100'; print('A100 ok:', torch.cuda.get_device_name(0))" || { echo "### NO GPU/A100 on $(hostname) — requeue+exit"; scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null; exit 1; }
$PY scripts/train.py "$task" --gpu-ids [0] --agent.logger tensorboard \
  --agent.run-name "$RUN" --agent.seed "$seed" --agent.max-iterations "${ITERS:-500}" --agent.save-interval 50 \
  --env.scene.num-envs 4096 --env.metrics.cat-soft.params.imp-max-p 0 \
  --env.metrics.substep-impulse-rows.params.enabled False $EXTRA_FLAGS
rc=$?
echo "### TRAIN_DONE $RUN rc=$rc $(date '+%F %T')"
exit $rc
