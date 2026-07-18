#!/usr/bin/env bash
# Eval the nf1 validity checkpoints (model_499) on 1 GPU via scripts/eval_impulse.sh.
# Log-only (imp_max_p forced 0), Tier-1 self-certifying (impossible_success_n/lambda_dead_n).
# Submit: sbatch scripts/slurm/vega_eval.sbatch   (optionally CAMPAIGN=nf1 CKGLOB='*nf1_*')
#SBATCH --job-name=z1-eval
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
# Known-bad nodes that kill jobs during SLURM setup (State FAILED, ExitCode 0:53, NO log — the in-script
# GPU guard below cannot catch a pre-exec failure). Remove a node once ops repair it. NOTE: a submit-time
# --exclude OVERRIDES this line, so pass the FULL list (gn03,gn34,...) if you add nodes at submit.
#SBATCH --exclude=gn03,gn34
#SBATCH --requeue
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/repos/unitree_rl_mjlab}"
mkdir -p logs
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet
export PYTHONPATH=.
export WARP_CACHE_PATH="$PWD/.warp_cache"
PY=.venv/bin/python
CKGLOB="${CKGLOB:-}"
CAMPAIGN="${CAMPAIGN:-ab1}"
# Space-separated dir globs (campaign-agnostic). Default = the A/B parking-fix campaign.
CKGLOBS="${CKGLOBS:-${CKGLOB:-*a05_* *b1_*}}"

# Build "name:path" pairs from every model_499.pt under the matched run dirs.
CKPTS=""
for pat in $CKGLOBS; do
  for ck in logs/rsl_rl/z1_hammer/$pat/model_499.pt; do
    [ -f "$ck" ] || continue
    # strip the "2026-07-16_21-14-22_" timestamp prefix -> a05_none_s0 / b1_noterm_s0
    name=$(basename "$(dirname "$ck")" | sed -E 's/^[0-9][0-9-]*_[0-9][0-9-]*_//')
    CKPTS="${CKPTS}${name}:${ck}"$'\n'
  done
done
echo "### EVAL checkpoints:"; printf '%s' "$CKPTS"
[ -n "$CKPTS" ] || { echo "EVAL_FAIL: no model_499 found for globs $CKGLOBS"; exit 1; }

# GPU guard (some Vega nodes, e.g. gn03, expose no allocatable GPU -> fail-fast + requeue instead of
# a confusing COMPLETED-but-crashed run).
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "### NO CUDA on $(hostname) — requeue+exit"; scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null; exit 1; }

FRESH="${FRESH:-1}" DEV=cuda:0 PY="$PY" NENVS=256 NSTEPS=400 EPLEN=4.0 SEED=42 \
  OUT="$PWD/eval/${CAMPAIGN}" CKPTS="$CKPTS" scripts/eval_impulse.sh
rc=$?
echo "### EVAL driver rc=$rc"
echo "### SUMMARY (key cols):"
CAMPAIGN="$CAMPAIGN" $PY - <<'PYEOF'
import csv, os
camp = os.environ.get("CAMPAIGN", "")
f = os.path.join(os.environ.get("PWD","."), "eval", camp, "summary.csv")
if not os.path.exists(f): print(f"(no summary.csv for campaign {camp})"); raise SystemExit
rows = list(csv.DictReader(open(f)))
def g(r,k): 
    v=r.get(k,""); 
    try: return f"{float(v):.3f}"
    except: return v
keys=["name","success","success_sampled","worst_ratio_max","worst_ratio_mean",
      "delivered_mean","nail_depth_mean","n_episodes","impossible_success_n","lambda_dead_n"]
print(" | ".join(keys))
for r in rows:
    print(" | ".join(g(r,k) for k in keys))
PYEOF
echo "### EVAL_DONE rc=$rc"
exit $rc
