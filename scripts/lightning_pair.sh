#!/usr/bin/env bash
# Lightning AI pair protocol -- the canonical prior-vs-none impulse pilot (checked in per
# adversarial review F5: the C3 campaign has its launch lines in docs/VEGA_TRAINING_PLAN.md;
# this is the equivalent single-GPU manifest for the -CaT-Impulse twin pair).
#
#   arm A: Unitree-Z1-Hammer-CaT-Impulse-Track  (weak annealed r_imit prior)
#   arm B: Unitree-Z1-Hammer-CaT-Impulse        (identical, prior off)
#
# Both are LOG-ONLY measurement arms (imp_max_p=0 -> delta==0): they maximize delivered
# impulse while logging per-joint reaction impulse Lambda_j. NEVER pass an imp-max-p
# override here -- enforcement is blocked on Khadiv decision (e) + the calibration gates.
#
# Registered rl_cfg defaults are 5000 iters / seed 42 (shared across arms; per-campaign
# protocol is CLI-passed by repo convention) -- the overrides below ARE the protocol.
# 500 iters is deliberate: the strike converges by ~iter 25 (V1); seed 0 single-seed = PILOT.
# Final checkpoint: model_499.pt (rsl_rl names the last save current_learning_iteration-1).
#
# Usage (from the repo root, GPU box):    bash scripts/lightning_pair.sh
# Then eval (CPU is fine) -- the explicit CKPTS override is REQUIRED: eval_impulse.sh's
# production default globs model_4999.pt (C3 campaign) and will not find pilot checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

ITERS="${ITERS:-500}"
SEED="${SEED:-0}"
NENVS="${NENVS:-4096}"
SAVE_EVERY="${SAVE_EVERY:-100}"

for arm in track none; do
  if [ "$arm" = "track" ]; then TASK=Unitree-Z1-Hammer-CaT-Impulse-Track; else TASK=Unitree-Z1-Hammer-CaT-Impulse; fi
  RUN="pair_${arm}_seed${SEED}"
  echo "[pair] training $RUN  task=$TASK  iters=$ITERS  envs=$NENVS"
  python scripts/train.py "$TASK" \
    --gpu-ids '[0]' \
    --agent.logger tensorboard \
    --agent.run-name "$RUN" \
    --agent.seed "$SEED" \
    --agent.max-iterations "$ITERS" \
    --agent.save-interval "$SAVE_EVERY" \
    --env.scene.num-envs "$NENVS"
done

echo "[pair] training done. Eval (CPU ok):"
C1=$(ls -t logs/rsl_rl/z1_hammer/*pair_track_seed${SEED}/model_$((ITERS - 1)).pt | head -1)
C2=$(ls -t logs/rsl_rl/z1_hammer/*pair_none_seed${SEED}/model_$((ITERS - 1)).pt | head -1)
echo "  OUT=/tmp/eval_impulse CKPTS=\"pair_track_seed${SEED}:$C1"
echo "  pair_none_seed${SEED}:$C2\" bash scripts/eval_impulse.sh"
