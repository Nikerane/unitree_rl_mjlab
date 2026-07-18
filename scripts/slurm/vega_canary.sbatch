#!/usr/bin/env bash
# Production-path canary (Codex Phase 2): assert A100, instrument gate, then a 50-iter
# track+none run through the REAL training path (log-only, substep_rows OFF), and verify
# the production TB metrics are live (delivered_total>0, impossible_success==0) — not just
# the monkey-patched probe. Fail-closed: any FAIL prints CANARY_FAIL and exits nonzero.
#SBATCH --job-name=z1-canary
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
# Known-bad nodes that kill jobs during SLURM setup (FAILED, ExitCode 0:53, NO log). Remove once
# repaired. A submit-time --exclude OVERRIDES this — pass the FULL list (gn03,gn34,...) if adding.
#SBATCH --exclude=gn03,gn34
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/repos/unitree_rl_mjlab}"
mkdir -p logs
module load git 2>/dev/null || true       # git on compute node (for provenance logging)
export GIT_PYTHON_REFRESH=quiet            # don't crash if GitPython can't validate git
export PYTHONPATH=.
export WARP_CACHE_PATH="$PWD/.warp_cache"
PY=.venv/bin/python
echo "### CANARY START $(date '+%F %T') host=$(hostname)"

echo "### [1/4] GPU assertion (A100)"
$PY - <<'PYEOF' || { echo "CANARY_FAIL: gpu-assert"; exit 1; }
import os, torch
assert torch.cuda.is_available(), "no CUDA"
n = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(n)]
print("GPUs:", n, names, "| CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), "| torch", torch.__version__)
assert all("A100" in x for x in names), "not all A100"
PYEOF

echo "### [2/4] instrument gate on cuda:0"
$PY scripts/diag_cuda_substep_probe.py --device cuda:0 --gate || { echo "CANARY_FAIL: instrument-gate"; exit 1; }

echo "### [3/4] 50-iter production canary (log-only, substep_rows OFF)"
run_arm () {  # $1=name $2=task
  $PY scripts/train.py "$2" --gpu-ids [0] --agent.logger tensorboard \
     --agent.run-name "$1" --agent.seed 99 --agent.max-iterations 50 --agent.save-interval 50 \
     --env.scene.num-envs 1024 --env.metrics.cat-soft.params.imp-max-p 0 \
     --env.metrics.substep-impulse-rows.params.enabled False \
     || { echo "CANARY_FAIL: train $1"; exit 1; }
}
run_arm canary_track Unitree-Z1-Hammer-CaT-Impulse-Track
run_arm canary_none  Unitree-Z1-Hammer-CaT-Impulse

echo "### [4/4] production-path metric check (TB scalars, not the probe)"
$PY - <<'PYEOF' || { echo "CANARY_FAIL: metrics"; exit 1; }
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
runs = sorted(glob.glob("logs/rsl_rl/z1_hammer/*canary_*"), key=os.path.getmtime)[-2:]
assert runs, "no canary runs found"
def find(tags, sub): return next((t for t in tags if sub in t), None)
ok = True
for r in runs:
    ev = sorted(glob.glob(os.path.join(r, "events.out.tfevents*")))
    if not ev: print("NO EVENTS", os.path.basename(r)); ok = False; continue
    ea = EventAccumulator(ev[-1]); ea.Reload(); tags = ea.Tags().get("scalars", [])
    def last(sub):
        t = find(tags, sub); return ea.Scalars(t)[-1].value if t else None
    dt, cs, im = last("delivered_total"), last("contact_seen"), last("impossible_success")
    print(f"{os.path.basename(r):40s} delivered_total={dt} contact_seen={cs} impossible_success={im}")
    if dt is None or dt <= 0: ok = False            # delivered_total must be positive
    if im is None or im > 0: ok = False             # missing OR nonzero impossible_success = FAIL (fail-closed)
    if cs is None or cs <= 0: ok = False            # contact_seen must be positive (validate, not just print)
print("CANARY_METRICS:", "PASS" if ok else "FAIL")
assert ok
PYEOF
rc=$?
echo "### CANARY done rc=$rc $(date '+%F %T')"
exit $rc
