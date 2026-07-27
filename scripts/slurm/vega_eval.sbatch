#!/usr/bin/env bash
# Evaluate the preregistered fsr4x8 model_499 checkpoints on one GPU.
# This job consumes a hash-checked 32-row accepted-attempt manifest, then
# evaluates each checkpoint in its own registered task. It never globs/selects
# "latest" and never routes D-prime/F/E through the C task default.
# Prepare only; submit after the clean provenance + four-task CUDA smoke gates:
#   CAMPAIGN=fsr4x8 EVAL_ATTEMPT=attempt1 EVAL_ROOT="$HOME/unitree_rl_mjlab_eval" \
#     ACCEPTED_MANIFEST="$HOME/unitree_rl_mjlab_eval/fsr4x8/accepted_attempts.tsv" \
#     EXPECTED_CODE_REVISION=<40hex> EXPECTED_ASSET_REVISION=<40hex> \
#     sbatch --export=ALL,CAMPAIGN,EVAL_ATTEMPT,EVAL_ROOT,ACCEPTED_MANIFEST,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION scripts/slurm/vega_eval.sbatch
#
# fq4x8 (first-contact quality campaign): consumes the canonical 21-column,
# headered accepted-training manifest (evaluation.analysis.fq4x8_manifests.
# TRAINING_FIELDS -- campaign/disposition/arm/short/task/training_seed/
# checkpoint_path/checkpoint_sha256/... through cap_signature_sha256/
# clean_state), snapshotted first and validated ONLY as that read-only
# snapshot via `python -m evaluation.analysis.fq4x8_manifests
# validate-accepted-manifest`, which fails closed on every drift from the
# frozen FROZEN_QUALITY_CAMPAIGN_MATRIX identity (arm/short/task/seed
# uniqueness, checkpoint basename+hash+uniqueness, reader/normalizer/reward/
# action/impedance/cap signature hashes, clean asset revision, and every
# manifest row's training code revision against EXPECTED_TRAINING_CODE_REVISION
# -- deliberately NOT EXPECTED_CODE_REVISION: training and evaluation code
# revisions are independently pinned, since a persistence/provenance-only fix
# can land in the evaluation checkout after training froze). EVAL_ROOT and
# ACCEPTED_MANIFEST must be the exact paths below -- no other layout is
# accepted, and the fq4x8 segment is appended exactly once:
#   CAMPAIGN=fq4x8 EVAL_ATTEMPT=attempt1 EVAL_ROOT="$HOME/unitree_rl_mjlab_eval" \
#     ACCEPTED_MANIFEST="$HOME/unitree_rl_mjlab_eval/fq4x8/accepted_training_checkpoints.tsv" \
#     EXPECTED_CODE_REVISION=<40hex clean evaluation checkout> \
#     EXPECTED_TRAINING_CODE_REVISION=<40hex frozen training checkout> \
#     EXPECTED_ASSET_REVISION=<40hex> \
#     sbatch --export=ALL,CAMPAIGN,EVAL_ATTEMPT,EVAL_ROOT,ACCEPTED_MANIFEST,EXPECTED_CODE_REVISION,EXPECTED_TRAINING_CODE_REVISION,EXPECTED_ASSET_REVISION scripts/slurm/vega_eval.sbatch
#SBATCH --job-name=z1-eval
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
# Thirty-two checkpoints each run a diagnostic mean rollout plus the balanced
# 256x2 sampled rollout; reserve four hours rather than inheriting the old
# one-hour small-campaign limit.
#SBATCH --time=04:00:00
# Known-bad nodes that kill jobs during SLURM setup (State FAILED, ExitCode 0:53, NO log — the in-script
# GPU guard below cannot catch a pre-exec failure). Remove a node once ops repair it. NOTE: a submit-time
# --exclude OVERRIDES this line, so pass the FULL list (gn03,gn34,...) if you add nodes at submit.
#SBATCH --exclude=gn03,gn34
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/repos/unitree_rl_mjlab}"
REPO_ROOT=$(pwd -P)
CANONICAL_ASSET_REPO=$(
  cd "$REPO_ROOT/../safe_impact_manipulation" 2>/dev/null && pwd -P
) || { echo "EVAL_FAIL: canonical sibling asset repository is missing"; exit 2; }
if [ -n "${ASSET_REPO:-}" ]; then
  RESOLVED_ASSET_REPO=$(
    cd "$ASSET_REPO" 2>/dev/null && pwd -P
  ) || { echo "EVAL_FAIL: ASSET_REPO cannot be resolved"; exit 2; }
  [ "$RESOLVED_ASSET_REPO" = "$CANONICAL_ASSET_REPO" ] \
    || { echo "EVAL_FAIL: ASSET_REPO must be the canonical sibling repository"; exit 2; }
fi
ASSET_REPO="$CANONICAL_ASSET_REPO"
mkdir -p logs
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export PYTHONPATH=.
export WARP_CACHE_PATH="$HOME/.cache/unitree_rl_mjlab/warp"
mkdir -p "$WARP_CACHE_PATH"
PY=.venv/bin/python
CKGLOB="${CKGLOB:-}"
CAMPAIGN="${CAMPAIGN:-ab1}"
# Reject typos of a known campaign name (e.g. "Fq4x8") before any GPU or
# manifest work -- otherwise they fall through to the unguarded default
# (glob-based) path. Exact matches proceed to their dedicated guard blocks;
# unrelated campaign names are untouched.
CAMPAIGN_LC=$(printf '%s' "$CAMPAIGN" | tr '[:upper:]' '[:lower:]')
if [ "$CAMPAIGN_LC" = "fq4x8" ] && [ "$CAMPAIGN" != "fq4x8" ]; then
  echo "EVAL_FAIL: CAMPAIGN '$CAMPAIGN' looks like fq4x8 but is not an exact, case-sensitive match"
  exit 2
fi
if [ "$CAMPAIGN_LC" = "fsr4x8" ] && [ "$CAMPAIGN" != "fsr4x8" ]; then
  echo "EVAL_FAIL: CAMPAIGN '$CAMPAIGN' looks like fsr4x8 but is not an exact, case-sensitive match"
  exit 2
fi
EVAL_OUTPUT="$PWD/eval/${CAMPAIGN}"
# Space-separated dir globs (campaign-agnostic). Default = the A/B parking-fix campaign.
CKGLOBS="${CKGLOBS:-${CKGLOB:-*a05_* *b1_*}}"

# fq4x8 fail-closed input/manifest validation. This MUST run before the GPU guard below (no
# GPU/sbatch/srun/network work happens here) so a bad submission never wastes a node allocation.
# Mirrors the proven fsr4x8 pattern (below): the accepted-training manifest is snapshotted FIRST
# into the claimed, read-only attempt output -- closing the validate-then-copy race on the mutable
# source path -- and ONLY the snapshot is ever validated, via the dedicated
# evaluation.analysis.fq4x8_manifests Python CLI (the single source of truth for the frozen
# FROZEN_QUALITY_CAMPAIGN_MATRIX identity; no hand-rolled awk reinterpretation of the manifest
# schema lives here). Does not touch the fsr4x8 branch below or its behavior.
if [ "$CAMPAIGN" = "fq4x8" ]; then
  [[ "${EXPECTED_CODE_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "EVAL_FAIL: EXPECTED_CODE_REVISION must be a clean 40-hex revision"; exit 2; }
  # Deliberately separate from EXPECTED_CODE_REVISION: the frozen manifest's
  # training checkpoints were trained at a revision that need not equal the
  # (possibly newer, e.g. carrying a persistence/provenance-only fix) clean
  # evaluation checkout above.
  [[ "${EXPECTED_TRAINING_CODE_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "EVAL_FAIL: EXPECTED_TRAINING_CODE_REVISION must be a clean 40-hex revision"; exit 2; }
  [[ "${EXPECTED_ASSET_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "EVAL_FAIL: EXPECTED_ASSET_REVISION must be a clean 40-hex revision"; exit 2; }
  case "${EVAL_ATTEMPT:-}" in
    ""|"."|".."|*[!A-Za-z0-9._-]*)
      echo "EVAL_FAIL: EVAL_ATTEMPT must be a nonempty path-safe attempt label"
      exit 2
      ;;
  esac
  # EVAL_ROOT is a fixed, durable, external root (not "any absolute path outside the repos" like
  # fsr4x8) -- fixing it is what guarantees the fq4x8 segment appended below is never duplicated.
  [ "${EVAL_ROOT:-}" = "$HOME/unitree_rl_mjlab_eval" ] \
    || { echo "EVAL_FAIL: EVAL_ROOT must be exactly \$HOME/unitree_rl_mjlab_eval"; exit 2; }
  case "$EVAL_ROOT" in
    "$REPO_ROOT"|"$REPO_ROOT"/*)
      echo "EVAL_FAIL: EVAL_ROOT must be outside the code repository"
      exit 2
      ;;
  esac
  case "$EVAL_ROOT" in
    "$ASSET_REPO"|"$ASSET_REPO"/*)
      echo "EVAL_FAIL: EVAL_ROOT must be outside the asset repository"
      exit 2
      ;;
  esac
  mkdir -p "$EVAL_ROOT" \
    || { echo "EVAL_FAIL: cannot create EVAL_ROOT=$EVAL_ROOT"; exit 2; }
  FQ_EVAL_ROOT=$(cd "$EVAL_ROOT" && pwd -P)
  case "$FQ_EVAL_ROOT" in
    "$REPO_ROOT"|"$REPO_ROOT"/*)
      echo "EVAL_FAIL: resolved EVAL_ROOT must be outside the code repository"
      exit 2
      ;;
  esac
  case "$FQ_EVAL_ROOT" in
    "$ASSET_REPO"|"$ASSET_REPO"/*)
      echo "EVAL_FAIL: resolved EVAL_ROOT must be outside the asset repository"
      exit 2
      ;;
  esac
  # A symlinked EVAL_ROOT can resolve to a physical directory that already ends in an "fq4x8"
  # path component (e.g. $HOME/unitree_rl_mjlab_eval -> .../physical/fq4x8) -- the literal-path
  # check above only sees the unresolved symlink, so it cannot catch this. Without this check the
  # fq4x8 segment appended below would be duplicated (.../fq4x8/fq4x8/attempt1).
  case "$FQ_EVAL_ROOT" in
    */fq4x8|*/fq4x8/*|fq4x8)
      echo "EVAL_FAIL: resolved EVAL_ROOT must not already contain an fq4x8 path component"
      exit 2
      ;;
  esac
  [ "${ACCEPTED_MANIFEST:-}" = "$HOME/unitree_rl_mjlab_eval/fq4x8/accepted_training_checkpoints.tsv" ] \
    || { echo "EVAL_FAIL: ACCEPTED_MANIFEST must be exactly \$EVAL_ROOT/fq4x8/accepted_training_checkpoints.tsv"; exit 2; }
  [ -f "$ACCEPTED_MANIFEST" ] \
    || { echo "EVAL_FAIL: accepted-training manifest is missing"; exit 2; }
  [ ! -L "$ACCEPTED_MANIFEST" ] \
    || { echo "EVAL_FAIL: accepted-training manifest may not be a symlink"; exit 2; }

  FQ_CODE_HASH=$(git rev-parse HEAD 2>/dev/null || echo unknown)
  FQ_CODE_STATUS=$(git status --porcelain=v1 --untracked-files=all 2>/dev/null || echo provenance-unavailable)
  FQ_ASSET_HASH=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null || echo unknown)
  FQ_ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all 2>/dev/null || echo provenance-unavailable)
  [ "$FQ_CODE_HASH" != "unknown" ] && [ -z "$FQ_CODE_STATUS" ] \
    || { echo "EVAL_FAIL: code provenance is unknown or dirty"; exit 2; }
  [ "$FQ_ASSET_HASH" != "unknown" ] && [ -z "$FQ_ASSET_STATUS" ] \
    || { echo "EVAL_FAIL: asset provenance is unknown or dirty"; exit 2; }
  [ "$FQ_CODE_HASH" = "$EXPECTED_CODE_REVISION" ] \
    || { echo "EVAL_FAIL: code revision does not equal expected code revision"; exit 2; }
  [ "$FQ_ASSET_HASH" = "$EXPECTED_ASSET_REVISION" ] \
    || { echo "EVAL_FAIL: asset revision does not equal expected asset revision"; exit 2; }
  echo "### EVAL_PROVENANCE code=$FQ_CODE_HASH asset=$FQ_ASSET_HASH"

  # Snapshot FIRST into the claimed, read-only attempt output (closes the validate-then-copy race
  # on the mutable source manifest), THEN validate ONLY that snapshot -- exactly the fsr4x8 pattern
  # below. FQ_EVAL_OUTPUT/FQ_VALIDATED_ROWS/ACCEPTED_MANIFEST_SHA256 are consumed after the GPU
  # guard by the fq4x8 execution branch; no GPU/sbatch/srun/network work has happened yet.
  FQ_EVAL_OUTPUT="$FQ_EVAL_ROOT/${CAMPAIGN}/${EVAL_ATTEMPT}"
  mkdir -p "$FQ_EVAL_ROOT/${CAMPAIGN}" \
    || { echo "EVAL_FAIL: cannot create campaign output root"; exit 2; }
  mkdir "$FQ_EVAL_OUTPUT" \
    || { echo "EVAL_FAIL: attempt output already exists or cannot be claimed: $FQ_EVAL_OUTPUT"; exit 2; }
  FQ_MANIFEST_SOURCE="$ACCEPTED_MANIFEST"
  ACCEPTED_MANIFEST="$FQ_EVAL_OUTPUT/accepted_training_checkpoints.tsv"
  cp "$FQ_MANIFEST_SOURCE" "$ACCEPTED_MANIFEST" \
    || { echo "EVAL_FAIL: cannot snapshot accepted-training manifest"; exit 2; }
  chmod a-w "$ACCEPTED_MANIFEST" \
    || { echo "EVAL_FAIL: cannot make accepted manifest snapshot read-only"; exit 2; }

  # FQ4X8 ACCEPTED-MANIFEST CONTRACT: the canonical 21-column, headered TSV defined by
  # evaluation.analysis.fq4x8_manifests.TRAINING_FIELDS. Every row is checked against the frozen
  # FROZEN_QUALITY_CAMPAIGN_MATRIX identity by that module -- not reinterpreted here.
  # --expected-code-revision here validates the manifest's TRAINING code revision, so it is bound
  # to EXPECTED_TRAINING_CODE_REVISION, NOT the evaluation checkout's own $FQ_CODE_HASH (already
  # independently verified against EXPECTED_CODE_REVISION above).
  FQ_VALIDATED_ROWS=$(
    "$PY" -m evaluation.analysis.fq4x8_manifests \
      validate-accepted-manifest \
      --manifest "$ACCEPTED_MANIFEST" \
      --expected-code-revision "$EXPECTED_TRAINING_CODE_REVISION" \
      --expected-asset-revision "$FQ_ASSET_HASH"
  ) || { echo "EVAL_FAIL: accepted-training manifest validation failed"; exit 2; }
  [ "$(printf '%s\n' "$FQ_VALIDATED_ROWS" | awk 'NF{n++} END{print n+0}')" -eq 32 ] \
    || { echo "EVAL_FAIL: accepted-training manifest did not emit 32 rows"; exit 2; }
  ACCEPTED_MANIFEST_SHA256=$(
    printf '%s\n' "$FQ_VALIDATED_ROWS" | awk -F '\t' 'NR==1{print $1}'
  )
  printf '%s  %s\n' "$ACCEPTED_MANIFEST_SHA256" accepted_training_checkpoints.tsv \
    > "$FQ_EVAL_OUTPUT/accepted_training_checkpoints.tsv.sha256"
fi

# GPU guard (some Vega nodes, e.g. gn03, expose no allocatable GPU -> fail-fast + requeue instead of
# a confusing COMPLETED-but-crashed run).
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "### NO CUDA on $(hostname) — requeue+exit"; scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null; exit 1; }

if [ "$CAMPAIGN" = "fsr4x8" ]; then
  [ -n "${EXPECTED_CODE_REVISION:-}" ] \
    || { echo "EVAL_FAIL: EXPECTED_CODE_REVISION is required"; exit 2; }
  [ -n "${EXPECTED_ASSET_REVISION:-}" ] \
    || { echo "EVAL_FAIL: EXPECTED_ASSET_REVISION is required"; exit 2; }
  case "${EVAL_ATTEMPT:-}" in
    ""|"."|".."|*[!A-Za-z0-9._-]*)
      echo "EVAL_FAIL: EVAL_ATTEMPT must be a nonempty path-safe attempt label"
      exit 2
      ;;
  esac
  case "${EVAL_ROOT:-}" in
    /*) ;;
    *)
      echo "EVAL_FAIL: EVAL_ROOT must be an absolute durable path outside the code repository"
      exit 2
      ;;
  esac
  case "$EVAL_ROOT" in
    "$REPO_ROOT"|"$REPO_ROOT"/*)
      echo "EVAL_FAIL: EVAL_ROOT must be outside the code repository"
      exit 2
      ;;
  esac
  case "$EVAL_ROOT" in
    "$ASSET_REPO"|"$ASSET_REPO"/*)
      echo "EVAL_FAIL: EVAL_ROOT must be outside the asset repository"
      exit 2
      ;;
  esac
  mkdir -p "$EVAL_ROOT" \
    || { echo "EVAL_FAIL: cannot create EVAL_ROOT=$EVAL_ROOT"; exit 2; }
  EVAL_ROOT=$(cd "$EVAL_ROOT" && pwd -P)
  case "$EVAL_ROOT" in
    "$REPO_ROOT"|"$REPO_ROOT"/*)
      echo "EVAL_FAIL: resolved EVAL_ROOT must be outside the code repository"
      exit 2
      ;;
  esac
  case "$EVAL_ROOT" in
    "$ASSET_REPO"|"$ASSET_REPO"/*)
      echo "EVAL_FAIL: resolved EVAL_ROOT must be outside the asset repository"
      exit 2
      ;;
  esac
  EVAL_OUTPUT="$EVAL_ROOT/${CAMPAIGN}/${EVAL_ATTEMPT}"
  mkdir -p "$EVAL_ROOT/${CAMPAIGN}" \
    || { echo "EVAL_FAIL: cannot create campaign output root"; exit 2; }
  mkdir "$EVAL_OUTPUT" \
    || { echo "EVAL_FAIL: attempt output already exists or cannot be claimed: $EVAL_OUTPUT"; exit 2; }
  CODE_HASH=$(git rev-parse HEAD 2>/dev/null || echo unknown)
  CODE_STATUS=$(git status --porcelain=v1 --untracked-files=all 2>/dev/null || echo provenance-unavailable)
  ASSET_HASH=$(git -C "$ASSET_REPO" rev-parse HEAD 2>/dev/null || echo unknown)
  ASSET_STATUS=$(git -C "$ASSET_REPO" status --porcelain=v1 --untracked-files=all 2>/dev/null || echo provenance-unavailable)
  [ "$CODE_HASH" != "unknown" ] && [ -z "$CODE_STATUS" ] \
    || { echo "EVAL_FAIL: code provenance is unknown or dirty"; exit 2; }
  [ "$ASSET_HASH" != "unknown" ] && [ -z "$ASSET_STATUS" ] \
    || { echo "EVAL_FAIL: asset provenance is unknown or dirty"; exit 2; }
  [ "$CODE_HASH" = "$EXPECTED_CODE_REVISION" ] \
    || { echo "EVAL_FAIL: code revision does not equal expected code revision"; exit 2; }
  [ "$ASSET_HASH" = "$EXPECTED_ASSET_REVISION" ] \
    || { echo "EVAL_FAIL: asset revision does not equal expected asset revision"; exit 2; }
  echo "### EVAL_PROVENANCE code=$CODE_HASH asset=$ASSET_HASH"

  case "${ACCEPTED_MANIFEST:-}" in
    /*) ;;
    *)
      echo "EVAL_FAIL: accepted-attempt manifest must be an absolute path"
      exit 2
      ;;
  esac
  [ -f "$ACCEPTED_MANIFEST" ] \
    || { echo "EVAL_FAIL: accepted-attempt manifest is missing"; exit 2; }
  [ ! -L "$ACCEPTED_MANIFEST" ] \
    || { echo "EVAL_FAIL: accepted-attempt manifest may not be a symlink"; exit 2; }
  case "$ACCEPTED_MANIFEST" in
    "$REPO_ROOT"|"$REPO_ROOT"/*|"$ASSET_REPO"|"$ASSET_REPO"/*)
      echo "EVAL_FAIL: filled accepted-attempt manifest must be outside both repositories"
      exit 2
      ;;
  esac
  ACCEPTED_MANIFEST=$(
    cd "$(dirname "$ACCEPTED_MANIFEST")" && printf '%s/%s\n' "$(pwd -P)" "$(basename "$ACCEPTED_MANIFEST")"
  ) || { echo "EVAL_FAIL: cannot resolve accepted-attempt manifest"; exit 2; }
  case "$ACCEPTED_MANIFEST" in
    "$REPO_ROOT"|"$REPO_ROOT"/*|"$ASSET_REPO"|"$ASSET_REPO"/*)
      echo "EVAL_FAIL: resolved accepted-attempt manifest must be outside both repositories"
      exit 2
      ;;
  esac
  MANIFEST_SOURCE="$ACCEPTED_MANIFEST"
  ACCEPTED_MANIFEST="$EVAL_OUTPUT/accepted_attempts.tsv"
  cp "$MANIFEST_SOURCE" "$ACCEPTED_MANIFEST" \
    || { echo "EVAL_FAIL: cannot snapshot accepted-attempt manifest"; exit 2; }
  chmod a-w "$ACCEPTED_MANIFEST" \
    || { echo "EVAL_FAIL: cannot make accepted manifest snapshot read-only"; exit 2; }
  VALIDATED_ROWS=$(
    "$PY" -m evaluation.analysis.first_strike_campaign \
      validate-accepted-manifest \
      --manifest "$ACCEPTED_MANIFEST" \
      --expected-code-revision "$CODE_HASH" \
      --expected-asset-revision "$ASSET_HASH"
  ) || { echo "EVAL_FAIL: accepted-attempt manifest validation failed"; exit 2; }
  [ "$(printf '%s\n' "$VALIDATED_ROWS" | awk 'NF{n++} END{print n+0}')" -eq 32 ] \
    || { echo "EVAL_FAIL: accepted-attempt manifest did not emit 32 rows"; exit 2; }
  ACCEPTED_MANIFEST_SHA256=$(
    printf '%s\n' "$VALIDATED_ROWS" | awk -F '\t' 'NR==1{print $1}'
  )
  printf '%s  %s\n' "$ACCEPTED_MANIFEST_SHA256" accepted_attempts.tsv \
    > "$EVAL_OUTPUT/accepted_attempts.tsv.sha256"

  eval_fsr_arm() {
    short=$1
    task=$2
    fresh=$3
    arm_ckpts=""
    while IFS=$'\t' read -r manifest_sha row_short seed row_task run_name checkpoint checkpoint_sha; do
      [ "$row_short" = "$short" ] || continue
      [ "$manifest_sha" = "$ACCEPTED_MANIFEST_SHA256" ] \
        || { echo "EVAL_FAIL: manifest digest changed within validated rows"; return 2; }
      [ "$row_task" = "$task" ] \
        || { echo "EVAL_FAIL: manifest task mismatch for $run_name"; return 2; }
      arm_ckpts="${arm_ckpts}${run_name}:${checkpoint}:${checkpoint_sha}"$'\n'
    done <<< "$VALIDATED_ROWS"
    [ "$(printf '%s' "$arm_ckpts" | awk 'NF{n++} END{print n+0}')" -eq 8 ] \
      || { echo "EVAL_FAIL: accepted manifest did not yield 8 rows for $short"; return 2; }
    echo "### EVAL arm=$short task=$task checkpoints:"
    printf '%s' "$arm_ckpts"
    FRESH="$fresh" DEV=cuda:0 PY="$PY" NENVS=256 NSTEPS=400 EPLEN=4.0 \
      SEED=2026072900 TASK="$task" OUT="$EVAL_OUTPUT" RUN_LEGACY_DIAG=0 \
      ACCEPTED_MANIFEST_SHA256="$ACCEPTED_MANIFEST_SHA256" \
      TRAINING_CODE_REVISION="$CODE_HASH" \
      TRAINING_ASSET_REVISION="$ASSET_HASH" \
      CKPTS="$arm_ckpts" scripts/eval_impulse.sh
  }

  eval_fsr_arm c Unitree-Z1-Hammer-CaT-Impulse 0
  rc=$?
  if [ "$rc" -eq 0 ]; then
    eval_fsr_arm dprime Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy 0 &&
    eval_fsr_arm f Unitree-Z1-Hammer-CaT-Impulse-Event-Linear 0 &&
    eval_fsr_arm e Unitree-Z1-Hammer-CaT-Impulse-Event 0
    rc=$?
  fi
elif [ "$CAMPAIGN" = "fq4x8" ]; then
  # All argument/manifest validation, the read-only snapshot, and its cryptographic digest
  # (ACCEPTED_MANIFEST_SHA256) already happened above, before the GPU guard.
  EVAL_ROOT="$FQ_EVAL_ROOT"
  EVAL_OUTPUT="$FQ_EVAL_OUTPUT"
  CODE_HASH="$FQ_CODE_HASH"
  ASSET_HASH="$FQ_ASSET_HASH"

  eval_fq_arm() {
    short=$1
    task=$2
    arm_ckpts=""
    while IFS=$'\t' read -r manifest_sha row_short seed row_task run_name checkpoint checkpoint_sha; do
      [ "$row_short" = "$short" ] || continue
      [ "$manifest_sha" = "$ACCEPTED_MANIFEST_SHA256" ] \
        || { echo "EVAL_FAIL: manifest digest changed within validated rows"; return 2; }
      [ "$row_task" = "$task" ] \
        || { echo "EVAL_FAIL: manifest task mismatch for $run_name"; return 2; }
      arm_ckpts="${arm_ckpts}${run_name}:${checkpoint}:${checkpoint_sha}"$'\n'
    done <<< "$FQ_VALIDATED_ROWS"
    [ "$(printf '%s' "$arm_ckpts" | awk 'NF{n++} END{print n+0}')" -eq 8 ] \
      || { echo "EVAL_FAIL: accepted manifest did not yield 8 rows for $short"; return 2; }
    echo "### EVAL arm=$short task=$task checkpoints:"
    printf '%s' "$arm_ckpts"
    FRESH=0 DEV=cuda:0 PY="$PY" NENVS=256 NSTEPS=400 EPLEN=4.0 \
      SEED=2026072900 TASK="$task" OUT="$EVAL_OUTPUT" RUN_LEGACY_DIAG=0 \
      ACCEPTED_MANIFEST_SHA256="$ACCEPTED_MANIFEST_SHA256" \
      TRAINING_CODE_REVISION="$EXPECTED_TRAINING_CODE_REVISION" \
      TRAINING_ASSET_REVISION="$ASSET_HASH" \
      CKPTS="$arm_ckpts" scripts/eval_impulse.sh
  }

  eval_fq_arm f8 Unitree-Z1-Hammer-CaT-Impulse-Event-Linear
  rc=$?
  if [ "$rc" -eq 0 ]; then
    eval_fq_arm f0 Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0 &&
    eval_fq_arm d0 Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0 &&
    eval_fq_arm fq Unitree-Z1-Hammer-CaT-Impulse-Event-Quality
    rc=$?
  fi
else
  # Historical/ad-hoc mode: preserve the existing campaign-agnostic globs.
  CKPTS=""
  for pat in $CKGLOBS; do
    for ck in logs/rsl_rl/z1_hammer/$pat/model_499.pt; do
      [ -f "$ck" ] || continue
      name=$(basename "$(dirname "$ck")" | sed -E 's/^[0-9][0-9-]*_[0-9][0-9-]*_//')
      CKPTS="${CKPTS}${name}:${ck}"$'\n'
    done
  done
  echo "### EVAL checkpoints:"; printf '%s' "$CKPTS"
  [ -n "$CKPTS" ] || { echo "EVAL_FAIL: no model_499 found for globs $CKGLOBS"; exit 1; }
  FRESH="${FRESH:-1}" DEV=cuda:0 PY="$PY" NENVS=256 NSTEPS=400 EPLEN=4.0 SEED=42 \
    OUT="$EVAL_OUTPUT" CKPTS="$CKPTS" scripts/eval_impulse.sh
  rc=$?
fi
echo "### EVAL driver rc=$rc"
echo "### SUMMARY (key cols):"
EVAL_OUTPUT="$EVAL_OUTPUT" $PY - <<'PYEOF'
import csv, os
out = os.environ["EVAL_OUTPUT"]
f = os.path.join(out, "summary.csv")
if not os.path.exists(f): print(f"(no summary.csv under {out})"); raise SystemExit
rows = list(csv.DictReader(open(f)))
def g(r,k): 
    v=r.get(k,""); 
    try: return f"{float(v):.3f}"
    except: return v
keys=["name","treatment","first_strike_useful_speed_mean_sampled",
      "first_strike_success_rate_sampled","overall_success_rate_sampled",
      "qvel_violation_rate_sampled","impossible_success_n","lambda_dead_n"]
print(" | ".join(keys))
for r in rows:
    print(" | ".join(g(r,k) for k in keys))
PYEOF
echo "### EVAL_DONE rc=$rc"
exit $rc
