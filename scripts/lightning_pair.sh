#!/usr/bin/env bash
# Lightning AI pair protocol -- the canonical prior-vs-none impulse pilot (checked in per
# adversarial review F5: the C3 campaign has its launch lines in docs/VEGA_TRAINING_PLAN.md;
# this is the equivalent single-GPU manifest for the -CaT-Impulse twin pair).
#
#   arm A: Unitree-Z1-Hammer-CaT-Impulse-Track  (weak annealed r_imit prior)
#   arm B: Unitree-Z1-Hammer-CaT-Impulse        (identical, prior off)
#
# Both are LOG-ONLY measurement arms: they maximize delivered impulse while logging the
# per-joint reaction impulse Lambda_j. imp_max_p=0 is the cfg default AND pinned explicitly
# below (belt-and-braces per review R2-F5) -- NEVER raise it here; enforcement is blocked on
# Khadiv decision (e) + the calibration gates.
#
# Registered rl_cfg defaults are 5000 iters / seed 42 (shared across arms; per-campaign
# protocol is CLI-passed by repo convention) -- the overrides below ARE the protocol.
# 500 iters is deliberate: the strike converges by ~iter 25 (V1); seed 0 single-seed = PILOT.
# Final checkpoint: model_499.pt (rsl_rl names the last save current_learning_iteration-1).
#
# Usage (from the repo root, GPU box):    bash scripts/lightning_pair.sh
# The eval step (R2-F5 hardened) runs automatically after training:
#   - PYBIN passthrough: eval_impulse.sh defaults to .venv/bin/python (Vega layout), which
#     does not exist on a conda/pip box -- we pass the ACTIVE interpreter explicitly.
#   - Isolated OUT + FRESH=1: the production /tmp/eval_impulse dir SKIPs by label alone, so
#     pilot rows could silently reuse stale results; a pair-specific dir + fresh wipe cannot.
#   - CKPTS built via printf (no embedded indentation -- leading spaces corrupt the label).
set -euo pipefail
cd "$(dirname "$0")/.."

ITERS="${ITERS:-500}"
SEED="${SEED:-0}"
NENVS="${NENVS:-4096}"
SAVE_EVERY="${SAVE_EVERY:-100}"
PYBIN="${PYBIN:-$(command -v python)}"

for arm in track none; do
  if [ "$arm" = "track" ]; then TASK=Unitree-Z1-Hammer-CaT-Impulse-Track; else TASK=Unitree-Z1-Hammer-CaT-Impulse; fi
  RUN="pair_${arm}_seed${SEED}"
  echo "[pair] training $RUN  task=$TASK  iters=$ITERS  envs=$NENVS"
  "$PYBIN" scripts/train.py "$TASK" \
    --gpu-ids '[0]' \
    --agent.logger tensorboard \
    --agent.run-name "$RUN" \
    --agent.seed "$SEED" \
    --agent.max-iterations "$ITERS" \
    --agent.save-interval "$SAVE_EVERY" \
    --env.scene.num-envs "$NENVS" \
    --env.metrics.cat-soft.params.imp-max-p 0
done

echo "[pair] training done -- running the pinned Lambda eval (CPU-capable)."
LAST=$((ITERS - 1))
C1=$(ls -t logs/rsl_rl/z1_hammer/*"pair_track_seed${SEED}"/model_${LAST}.pt | head -1)
C2=$(ls -t logs/rsl_rl/z1_hammer/*"pair_none_seed${SEED}"/model_${LAST}.pt | head -1)
CKPTS=$(printf '%s\n%s' "pair_track_seed${SEED}:${C1}" "pair_none_seed${SEED}:${C2}")
OUT="/tmp/eval_impulse_pair_seed${SEED}" FRESH=1 PY="$PYBIN" CKPTS="$CKPTS" \
  bash scripts/eval_impulse.sh
echo "[pair] done. Summary: /tmp/eval_impulse_pair_seed${SEED}/summary.csv"
