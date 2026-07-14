#!/usr/bin/env bash
# C3 checkpoint eval -- per-joint impulse Λ peaks/p95 + delivered impulse summary (the C4 analogue).
# Thin driver: pins the protocol below, loops checkpoints, and calls scripts/eval_impulse.py once per
# checkpoint (rollout + one summary.csv row appended). Protocol lineage: the June joint-velocity
# eval (same-env cross-arm convention) is preserved at docs/archive/tooling/eval_peak_qv.sh
# (frozen historical record, archived 2026-07-14). `diag_impulse_trace.py --ckpt` produces one
# force-propagation figure per seed-0 checkpoint.
#
# PINNED PROTOCOL (task-11-brief.md Step 2; keep this header AND scripts/eval_impulse.py's docstring
# in sync -- do not drift without updating both AND the eventual results record):
#   - eval env:        Unitree-Z1-Hammer-CaT-Impulse, PLAY cfg (deterministic reset, no obs noise)
#   - instrumentation: imp_max_p FORCED to 0 at eval time regardless of the checkpoint's training-time
#     value (log-only hook = pure instrumentation; the obs/action space is IDENTICAL across all four
#     hammer arms, so the SAME env scores any arm's checkpoint -- the archived eval_peak_qv.sh
#     same-env cross-arm protocol, extended to the impulse metrics). Equivalent train.py override:
#         --env.metrics.cat-soft.params.imp-max-p 0
#   - checkpoint:      final model_4999.pt per run (production default below)
#   - envs/steps:      256 envs x >=2 episodes/env (>=512 episodes/policy) -- eval_impulse.py's
#     --episode-len-s finitizes the play-cfg horizon so this holds even for a 0%-success checkpoint
#   - action mode:     mean-action rollout (primary, the CSV row) + one sampled-action repeat
#     (robustness check, condensed "_sampled" columns on the SAME row)
#   - eval seed:       42 (fixed; the sampled-action repeat uses seed+1). Bit-reproducible on CPU
#     only; on CUDA, mujoco_warp is non-deterministic run-to-run (atomic-reduction ordering), so the
#     seeds pin the PROTOCOL -- CUDA results are statistical, not bitwise.
#   - provenance:      host + UTC timestamp + checkpoint path + repo git hash recorded per CSV row
#   - invariants:      every row carries impossible_success_n / lambda_dead_n (Tier-1 safety
#     net, 2026-07-14: success-with-zero-impulse and cross-path disagreement are dead-instrument
#     states, never physics); a nonzero count exits 2 and this driver counts the row as a FAIL
#
# Checkpoints must already be under logs/rsl_rl/ (on Lightning they are written in place by
# scripts/lightning_pair.sh; for a remote box, rsync them back manually first).
#
# Production default targets the 12 C3 checkpoints (4 arms x 3 seeds: c3_imp, c3_imp0, c3_track,
# c3_catsoft -- docs/VEGA_TRAINING_PLAN.md "C3 campaign"), globbed as they arrive.
#
# Re-runs are NON-DESTRUCTIVE by default: existing $OUT is kept, and any checkpoint whose name
# already has a summary.csv row is SKIPped ("SKIP <name> (row exists)") -- so re-running after
# late-arriving checkpoints only evaluates the new ones. Set FRESH=1 to wipe $OUT and re-evaluate
# everything (also the recovery path when eval_impulse.py rejects a stale-schema summary.csv).
#
# Override CKPTS to point at a different/smaller checkpoint set (e.g. local smoke checkpoints, one
# "name:path" pair per line -- name should end "_seedN" so the seed0 ones get a trace figure below).
# Keep NSTEPS >= 2 x EPLEN x 50 (control steps/s) or a never-succeeding smoke checkpoint completes
# ZERO episodes and eval_impulse.py fails the row loudly (n_episodes=0):
#   FRESH=1 CKPTS="smoke_impcfg_seed0:logs/rsl_rl/z1_hammer/2026-07-10_10-33-23/model_4.pt
# smoke_impcfg2_seed1:logs/rsl_rl/z1_hammer/2026-07-10_11-58-15/model_1.pt" \
#     NENVS=16 NSTEPS=110 EPLEN=1.0 DEV=cpu PY=~/miniconda3/envs/unitree_mjlab/bin/python \
#     scripts/eval_impulse.sh
set -uo pipefail
# Repo root: SLURM submit dir (Vega) or SCRIPT-RELATIVE (2026-07-14 fix: the old
# $HOME/repos/unitree_rl_mjlab fallback broke on any box with a different layout,
# e.g. Lightning /teamspace/studios/...; it printed "No such file" and relied on
# the caller already being in the right cwd).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

TASK="${TASK:-Unitree-Z1-Hammer-CaT-Impulse}"
NENVS="${NENVS:-256}"
NSTEPS="${NSTEPS:-400}"        # >=2x EPLEN's control-step cap -- guarantees >=2 episodes/env worst-case
EPLEN="${EPLEN:-4.0}"          # finite eval episode horizon (s); see eval_impulse.py --episode-len-s
DEV="${DEV:-cuda:0}"
SEED="${SEED:-42}"
OUT="${OUT:-/tmp/eval_impulse}"
# Interpreter: default to the active python (conda/Lightning). The old default (.venv/bin/python)
# was the retired Vega layout and silently pointed at a nonexistent file everywhere else.
PY="${PY:-$(command -v python)}"
FRESH="${FRESH:-0}"            # FRESH=1: wipe $OUT first. Default: keep output, skip existing rows.

if [ "$FRESH" = "1" ]; then rm -rf "$OUT"; fi
mkdir -p "$OUT/traces"
SUMMARY_CSV="$OUT/summary.csv"

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
    # Strip the leading "YYYY-MM-DD_HH-MM-SS_" timestamp prefix rsl_rl adds; keep the
    # "<arm>_seed<N>" run-name (the RUN-NAME CONTRACT, canonical in scripts/compare_runs.py).
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
# never drift from the same IMP_J_LIMIT the eval/reward code actually uses. The substitution is
# CHECKED: a broken $PY/import must abort loudly, not silently drop the cap lines from every figure.
if ! J_LIMIT_CSV=$("$PY" -c "from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT; print(','.join(str(x) for x in IMP_J_LIMIT))") \
    || [ -z "$J_LIMIT_CSV" ]; then
  echo "eval_impulse.sh: ERROR -- failed to read IMP_J_LIMIT via \$PY ($PY); every trace figure would"
  echo "  silently lose its J_limit cap lines. Fix the interpreter/env (PY=...) and re-run."
  exit 1
fi

FAILS=0
for pair in "${PAIRS[@]}"; do
  case "$pair" in
    *:*) ;;
    *) echo "MALFORMED ENTRY '$pair' (expected name:path)"; FAILS=$((FAILS + 1)); echo; continue ;;
  esac
  name="${pair%%:*}"
  ckpt="${pair#*:}"
  echo "########## $name :: $ckpt ##########"
  # Non-destructive re-run default: a name that already has a summary.csv row is done -- skip it
  # (FRESH=1 wipes $OUT up front, so every name re-runs).
  if [ -f "$SUMMARY_CSV" ] && awk -F, -v n="$name" 'NR>1 && $1==n{found=1} END{exit !found}' "$SUMMARY_CSV"; then
    echo "SKIP $name (row exists)"; echo; continue
  fi
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
