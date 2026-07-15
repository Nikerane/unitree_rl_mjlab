#!/usr/bin/env bash
# Core+3-seeds campaign (2026-07-14): the impulse prior-vs-none pair × 3 seeds = 6 log-only runs,
# turning the n=1 pilot into a cross-arm result WITH confidence intervals. Self-protecting and
# GPU-adaptive — it fans one run per GPU by default (clean, zero contention), and leans on the
# shipped safety net (preflight instrument gate + Tier-1 eval invariants) rather than a new
# concurrency subsystem, so single-GPU packing can be attempted without risking silent corruption.
#
#   arms:  Unitree-Z1-Hammer-CaT-Impulse-Track (motion prior)  |  -CaT-Impulse (no prior)
#   seeds: 0 1 2       all LOG-ONLY (imp_max_p=0); enforcement excluded (blocked on Khadiv (e)).
#
# Usage (from anywhere; runs from the repo root):
#   bash scripts/lightning_campaign.sh                 # 1 run per GPU (default)
#   CONCURRENCY=3 bash scripts/lightning_campaign.sh   # pack a single GPU 3-deep (guarded, see below)
#   DRYRUN=1 bash scripts/lightning_campaign.sh         # print the plan + commands, run nothing
# Knobs: ITERS(500) NENVS(4096) SAVE_EVERY(100) SEEDS("0 1 2") EVAL_DEV(cpu) EVAL_NENVS(256) PYBIN(python).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ITERS="${ITERS:-500}"
NENVS="${NENVS:-4096}"
SAVE_EVERY="${SAVE_EVERY:-100}"
SEEDS="${SEEDS:-0 1 2}"
PYBIN="${PYBIN:-$(command -v python)}"
EVAL_DEV="${EVAL_DEV:-cpu}"   # eval on the PROVEN-live CPU path by default (the C0-gate reference);
                              # the Tier-1 invariants would catch a CUDA eval-readout regression anyway.
DRYRUN="${DRYRUN:-0}"

NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
[ "${NGPU:-0}" -ge 1 ] || { echo "[campaign] no GPU detected (nvidia-smi -L) — aborting."; exit 1; }
CONCURRENCY="${CONCURRENCY:-$NGPU}"   # default: one run per GPU (no contention)
echo "[campaign] GPUs=$NGPU  concurrency=$CONCURRENCY  iters=$ITERS  envs=$NENVS  seeds=[$SEEDS]  eval_dev=$EVAL_DEV"

# --- job matrix: arm × seed -------------------------------------------------------------------
JOBS=()
for arm in track none; do for s in $SEEDS; do JOBS+=("$arm:$s"); done; done
echo "[campaign] ${#JOBS[@]} runs: ${JOBS[*]}"

task_of() { [ "$1" = track ] && echo "Unitree-Z1-Hammer-CaT-Impulse-Track" || echo "Unitree-Z1-Hammer-CaT-Impulse"; }

# One training run in the background, pinned to a physical GPU. Args: arm seed gpu iters run-name.
launch() {
  local arm="$1" s="$2" gpu="$3" iters="$4" name="$5"
  local task; task=$(task_of "$arm")
  local cmd="$PYBIN scripts/train.py $task --gpu-ids [$gpu] --agent.logger tensorboard \
--agent.run-name $name --agent.seed $s --agent.max-iterations $iters --agent.save-interval $SAVE_EVERY \
--env.scene.num-envs $NENVS --env.metrics.cat-soft.params.imp-max-p 0"
  echo "[campaign] gpu$gpu <- $name (${iters} iters)"
  if [ "$DRYRUN" = 1 ]; then echo "  DRYRUN: $cmd"; ( sleep 0.1 ) & return; fi
  eval "$cmd" > "logs/campaign_${name}.log" 2>&1 &
}

# Run a list of "arm:seed" jobs in chunks of CONCURRENCY; GPU = within-chunk index % NGPU.
# Equal-length runs, so chunking is ~as efficient as a rolling queue and far simpler. Returns
# nonzero if ANY run failed (its log has the error).
run_matrix() {
  local iters="$1"; shift
  local -a jobs=("$@") pids=() names=(); local rc=0 k=0
  for job in "${jobs[@]}"; do
    local arm="${job%%:*}" s="${job##*:}" gpu=$(( k % NGPU )) name="camp_${job%%:*}_seed${job##*:}"
    launch "$arm" "$s" "$gpu" "$iters" "$name"; pids+=($!); names+=("$name"); k=$((k+1))
    if [ "$(( k % CONCURRENCY ))" -eq 0 ]; then
      for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[campaign] FAILED: ${names[$i]} (see logs/campaign_${names[$i]}.log)"; rc=1; }; done
      pids=(); names=()
    fi
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[campaign] FAILED: ${names[$i]}"; rc=1; }; done
  return $rc
}

# --- eval a set of run-names → one summary.csv (Tier-1 invariants gate every row) --------------
eval_runs() {
  local out="$1"; shift
  if [ "$DRYRUN" = 1 ]; then echo "[campaign] DRYRUN would eval [$*] on $EVAL_DEV -> $out/summary.csv"; return 0; fi
  local ckpts=""
  for name in "$@"; do
    local ck; ck=$(ls -t logs/rsl_rl/z1_hammer/*"${name}"/model_*.pt 2>/dev/null | head -1)
    [ -n "$ck" ] && ckpts+="${name}:${ck}"$'\n'
  done
  [ -n "$ckpts" ] || { echo "[campaign] no checkpoints to eval"; return 1; }
  # Eval at a SMALL env count regardless of the training NENVS: eval_impulse.sh reads $NENVS, and a
  # `NENVS=4096 bash lightning_campaign.sh` invocation leaks that into the CPU eval -> hours per run.
  OUT="$out" FRESH=1 PY="$PYBIN" DEV="$EVAL_DEV" NENVS="${EVAL_NENVS:-256}" CKPTS="$ckpts" bash scripts/eval_impulse.sh
}

# === 1. Preflight instrument gate (single process) ===========================================
echo "[campaign] preflight instrumentation gate on cuda:0"
if [ "$DRYRUN" != 1 ] && ! "$PYBIN" scripts/diag_cuda_substep_probe.py --device cuda:0 --gate; then
  echo "[campaign] PREFLIGHT FAILED — impulse instrument is NOT live on the GPU. Refusing to spend"
  echo "[campaign] credits. Run the diagnostic probe (no --gate) and report its VERDICT."
  exit 1
fi

# === 2. Single-GPU packing guard: if concurrency exceeds the GPU count, prove the shipped safety
#        net catches nothing wrong under contention BEFORE committing the full batch. Reuses the
#        eval invariants: two concurrent 50-iter runs, then eval — abort if either invariant fires.
if [ "$CONCURRENCY" -gt "$NGPU" ] && [ "$DRYRUN" != 1 ]; then
  echo "[campaign] CONCURRENCY=$CONCURRENCY > $NGPU GPU(s): running a 2-run concurrency smoke first"
  run_matrix 50 "track:90" "none:91" || { echo "[campaign] concurrency smoke run FAILED"; exit 1; }
  eval_runs /tmp/eval_campaign_smoke camp_track_seed90 camp_none_seed91 || {
    echo "[campaign] CONCURRENCY SMOKE FAILED the Tier-1 invariants — single-GPU packing corrupts the"
    echo "[campaign] measurement on this box. Re-run with CONCURRENCY=$NGPU (one job per GPU)."; exit 1; }
  echo "[campaign] concurrency smoke PASSED — packing is safe on this box."
fi

# === 3. Train the 6-run matrix ================================================================
echo "[campaign] training ${#JOBS[@]} runs (chunks of $CONCURRENCY)"
run_matrix "$ITERS" "${JOBS[@]}" || echo "[campaign] WARNING: some runs failed — evaluating the rest."

# === 4. Eval all → one CSV ====================================================================
NAMES=(); for job in "${JOBS[@]}"; do NAMES+=("camp_${job%%:*}_seed${job##*:}"); done
eval_runs /tmp/eval_impulse_campaign "${NAMES[@]}"
echo "[campaign] done. Summary: /tmp/eval_impulse_campaign/summary.csv"
echo "[campaign] CHECK the impossible_success_n and lambda_dead_n columns — both MUST be 0 for every row."
