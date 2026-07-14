#!/usr/bin/env bash
# Peak-|q̇| safety eval across the velocity-enforcement comparison matrix.
# Each policy is rolled out in the SAME neutral base env (Unitree-Z1-Hammer): the constraint
# mechanisms (soft/hard CaT, hard-term) alter only learning, not physics/obs/actions, so this
# isolates the learned policy's behavior. Reports peak |arm joint vel| vs the 3.1415 rad/s Z1 limit.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/repos/unitree_rl_mjlab}"

ENVS="${ENVS:-256}"; STEPS="${STEPS:-120}"; DEV="${DEV:-cuda:0}"

NAMES=(softcat_seed0 softcat_seed1 softcat_seed2 a_base_seed0 c_a3_cat_seed0 c_hardterm_seed0)
CKS=(
  logs/rsl_rl/z1_hammer/2026-06-18_07-58-20_c_softcat_seed0/model_499.pt
  logs/rsl_rl/z1_hammer/2026-06-18_07-58-20_c_softcat_seed1/model_499.pt
  logs/rsl_rl/z1_hammer/2026-06-18_07-58-20_c_softcat_seed2/model_499.pt
  logs/rsl_rl/z1_hammer/2026-06-17_10-50-13_a_base_seed0/model_499.pt
  logs/rsl_rl/z1_hammer/2026-06-17_17-39-50_c_a3_cat_seed0/model_499.pt
  logs/rsl_rl/z1_hammer/2026-06-17_21-37-10_c_hardterm_seed0/model_499.pt
)

for i in "${!NAMES[@]}"; do
  echo "########## ${NAMES[$i]} :: ${CKS[$i]} ##########"
  if [ ! -f "${CKS[$i]}" ]; then echo "MISSING CHECKPOINT"; echo; continue; fi
  .venv/bin/python scripts/diag_policy_trace.py --task Unitree-Z1-Hammer \
    --ckpt "${CKS[$i]}" --num-envs "$ENVS" --nsteps "$STEPS" --device "$DEV" \
    2>&1 | grep -E "success rate|peak v_axial @ contact|arm joint vel|EXCEEDS|honest"
  echo
done
echo EVAL_PEAK_QV_DONE
