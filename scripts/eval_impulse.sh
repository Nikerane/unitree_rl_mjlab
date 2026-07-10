#!/usr/bin/env bash
# C3 checkpoint eval -- per-joint impulse Λ peaks/p95 + delivered impulse summary (the C4 analogue).
# Thin driver: pins the protocol below, loops checkpoints, and calls scripts/eval_impulse.py once per
# checkpoint (rollout + one summary.csv row appended). Companion to the June `eval_peak_qv.sh`
# (frozen, historical joint-velocity protocol) -- read that script for the shared conventions; do NOT
# edit it. `diag_impulse_trace.py --ckpt` produces one force-propagation figure per seed-0 checkpoint.
#
# PINNED PROTOCOL (task-11-brief.md Step 2; keep this header AND scripts/eval_impulse.py's docstring
# in sync -- do not drift without updating both AND the eventual results record):
#   - eval env:        Unitree-Z1-Hammer-CaT-Impulse, PLAY cfg (deterministic reset, no obs noise)
#   - instrumentation: imp_max_p FORCED to 0 at eval time regardless of the checkpoint's training-time
#     value (log-only hook = pure instrumentation; the obs/action space is IDENTICAL across all four
#     hammer arms, so the SAME env scores any arm's checkpoint -- eval_peak_qv.sh's same-env
#     cross-arm protocol, extended to the impulse metrics). Equivalent scripts/train.py-style override:
#         --env.metrics.cat-soft.params.imp-max-p 0
#   - checkpoint:      final model_4999.pt per run (production default below)
#   - envs/steps:      256 envs x >=2 episodes/env (>=512 episodes/policy) -- eval_impulse.py's
#     --episode-len-s finitizes the play-cfg horizon so this holds even for a 0%-success checkpoint
#   - action mode:     mean-action rollout (primary, the CSV row) + one sampled-action repeat
#     (robustness check, condensed "_sampled" columns on the SAME row)
#   - eval seed:       42 (fixed; the sampled-action repeat uses seed+1, still deterministic)
#   - provenance:      host + UTC timestamp + checkpoint path + repo git hash recorded per CSV row
#
# Rsync (NOT run by this script -- run manually before the production pass):
#   rsync -av vega:<remote-repo-path>/logs/rsl_rl/ logs/rsl_rl/
#
# Production default targets the 12 C3 checkpoints (4 arms x 3 seeds: c3_imp, c3_imp0, c3_track,
# c3_catsoft -- docs/VEGA_TRAINING_PLAN.md "C3 campaign"), globbed as they arrive from rsync.
#
# Override CKPTS to point at a different/smaller checkpoint set (e.g. local smoke checkpoints, one
# "name:path" pair per line -- name should end "_seedN" so the seed0 ones get a trace figure below).
# Keep NSTEPS >= 2 x EPLEN x 50 (control steps/s) or a never-succeeding smoke checkpoint completes
# ZERO episodes and the row degenerates to n_episodes=0:
#   CKPTS="smoke_impcfg_seed0:logs/rsl_rl/z1_hammer/2026-07-10_10-33-23/model_4.pt
# smoke_impcfg2_seed1:logs/rsl_rl/z1_hammer/2026-07-10_11-58-15/model_1.pt" \
#     NENVS=16 NSTEPS=110 EPLEN=1.0 DEV=cpu PY=~/miniconda3/envs/unitree_mjlab/bin/python \
#     scripts/eval_impulse.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/repos/unitree_rl_mjlab}"

TASK="${TASK:-Unitree-Z1-Hammer-CaT-Impulse}"
NENVS="${NENVS:-256}"
NSTEPS="${NSTEPS:-400}"        # >=2x EPLEN's control-step cap -- guarantees >=2 episodes/env worst-case
EPLEN="${EPLEN:-4.0}"          # finite eval episode horizon (s); see eval_impulse.py --episode-len-s
DEV="${DEV:-cuda:0}"
SEED="${SEED:-42}"
OUT="${OUT:-/tmp/eval_impulse}"
PY="${PY:-.venv/bin/python}"   # Vega provisions .venv; override for local runs (see smoke example above)

rm -rf "$OUT"
mkdir -p "$OUT/traces"

if [ -n "${CKPTS:-}" ]; then
  # Override mode: "name:path" pairs, one per line (smoke / ad hoc checkpoint sets).
  PAIRS=()
  while IFS= read -r line; do
    [ -n "$line" ] && PAIRS+=("$line")
  done <<< "$CKPTS"
else
  # Production default: the 12 C3 checkpoints (4 arms x 3 seeds), globbed as they arrive from rsync.
  PAIRS=()
  for p in logs/rsl_rl/z1_hammer/*c3_*_seed*/model_4999.pt; do
    [ -f "$p" ] || continue
    run_dir=$(basename "$(dirname "$p")")
    # Strip the leading "YYYY-MM-DD_HH-MM-SS_" timestamp prefix train_array.sbatch adds; keep the
    # "c3_<arm>_seed<N>" run-name (docs/VEGA_TRAINING_PLAN.md "C3 campaign").
    name=$(echo "$run_dir" | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_//')
    PAIRS+=("${name:-$run_dir}:$p")
  done
fi

if [ "${#PAIRS[@]}" -eq 0 ]; then
  echo "eval_impulse.sh: no checkpoints matched (production glob logs/rsl_rl/z1_hammer/*c3_*_seed*/model_4999.pt,"
  echo "  or the CKPTS override). Nothing to do -- rsync the C3 logs back first (see header) or set CKPTS."
  exit 1
fi

# J_limit is READ from the shipped config (never hardcoded here) so the trace-figure cap lines can
# never drift from the same IMP_J_LIMIT the eval/reward code actually uses.
J_LIMIT_CSV=$("$PY" -c "from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT; print(','.join(str(x) for x in IMP_J_LIMIT))")

FAILS=0
for pair in "${PAIRS[@]}"; do
  name="${pair%%:*}"
  ckpt="${pair#*:}"
  echo "########## $name :: $ckpt ##########"
  if [ ! -f "$ckpt" ]; then echo "MISSING CHECKPOINT"; FAILS=$((FAILS + 1)); echo; continue; fi

  "$PY" scripts/eval_impulse.py \
    --task "$TASK" --ckpt "$ckpt" --name "$name" \
    --num-envs "$NENVS" --nsteps "$NSTEPS" --episode-len-s "$EPLEN" \
    --device "$DEV" --seed "$SEED" --out "$OUT" \
    || { echo "EVAL FAILED: $name"; FAILS=$((FAILS + 1)); }

  # One force-propagation trace figure per ARM (seed0 checkpoint only) -- brief Step 3.
  case "$name" in
    *seed0)
      "$PY" scripts/diag_impulse_trace.py --ckpt "$ckpt" --task "$TASK" --play \
        --num-envs 1 --nsteps 60 --device "$DEV" \
        --j-limit "$J_LIMIT_CSV" --out "$OUT/traces/$name" \
        || { echo "TRACE FAILED: $name"; FAILS=$((FAILS + 1)); }
      ;;
  esac
  echo
done
echo "EVAL_IMPULSE_DONE (failures: $FAILS)"
exit "$((FAILS > 0 ? 1 : 0))"
