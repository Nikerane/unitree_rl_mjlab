#!/bin/bash
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
stage=${SLURM_SUBMIT_DIR:?}
export GIT_PYTHON_GIT_EXECUTABLE=/cvmfs/sling.si/modules/el7/software/git/2.45.1-GCCcore-13.3.0/bin/git
export PATH="${GIT_PYTHON_GIT_EXECUTABLE%/*}:$PATH"
git_lib_root=/cvmfs/sling.si/modules/el7/software
export LD_LIBRARY_PATH="$git_lib_root/libiconv/1.17-GCCcore-13.3.0/lib:$git_lib_root/zlib/1.3.1-GCCcore-13.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$GIT_PYTHON_GIT_EXECUTABLE" --version
node_tmp=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/u22-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$node_tmp"' EXIT
export TMPDIR="$node_tmp" PYTHONPYCACHEPREFIX="$node_tmp/pycache"
export PYTHONPATH="$node_tmp/source" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MUJOCO_GL=egl
export Z1_STUDY_CACHE="$node_tmp/warp-cache"

cd "$node_tmp"
runtime_archive="/tmp/u22-python-${SLURM_JOB_ID}.tar.gz"
deadline=$((SECONDS + 900))
while [[ ! -f "$runtime_archive.ready" ]]; do
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for Slurm runtime transfer" >&2
        exit 1
    fi
    sleep 2
done
echo "${U22_RUNTIME_SHA256:?}  $runtime_archive" | sha256sum -c -
cp "/tmp/u22-source-${SLURM_JOB_ID}.tar.gz" u22-direct-git-inputs.tar.gz
echo "${U22_SOURCE_SHA256:?}  u22-direct-git-inputs.tar.gz" | sha256sum -c -
tar -xzf u22-direct-git-inputs.tar.gz
sha256sum -c source.sha256
unlink "/tmp/u22-source-${SLURM_JOB_ID}.tar.gz"
tar -xzf "$runtime_archive"
unlink "$runtime_archive"
unlink "$runtime_archive.ready"
py=("$node_tmp/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12" -S)
export PYTHONHOME="$node_tmp/cpython-3.12.13-linux-x86_64-gnu"
export PYTHONPATH="$node_tmp/source:$node_tmp/site-packages"
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME="$node_tmp/cache" CUDA_CACHE_PATH="$node_tmp/cuda-cache"
"${py[@]}" -B -c 'import sys,os,torch,warp,mujoco,mjlab; from importlib.metadata import version; print("RUNTIME_PROBE", sys.executable, sys.path, torch.cuda.is_available(), os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True); assert torch.cuda.is_available(); assert all("/ceph" not in p for p in sys.path); print("LOCAL_RUNTIME", sys.executable, {k:version(k) for k in ("torch","warp-lang","mujoco","mujoco-warp","mjlab")}, flush=True)'
cd source
if [[ ${1:?} == gates ]]; then
    for gate in validate_rewards verify_contact_sensor verify_reward_setup; do
        "${py[@]}" -B -c "import os,runpy,warp; warp.config.use_precompiled_headers=False; warp.config.kernel_cache_dir=os.environ['Z1_STUDY_CACHE']; runpy.run_path('docs/research/reward-design/$gate.py',run_name='__main__')" > "$stage/$gate.log" 2>&1
    done
    grep -q 'ALL PHASES PASSED' "$stage/validate_rewards.log"
    PYTHONPATH="$PYTHONPATH:$node_tmp/test-deps" "${py[@]}" -m pytest -p no:cacheprovider -q tests/test_{impact_progress_reward,impulse_bound,impulse_constraint,delivered_impulse_reward,cat_soft_hook}.py tests/test_timed_reference_phase.py > "$stage/pytest.log" 2>&1
    touch "$stage/gates-passed"
    exit
fi
test -f "$stage/gates-passed"
arms=(control preview)
coefficients=(10 100 200)
id=${SLURM_ARRAY_TASK_ID:?}
arm=${arms[id / 3 % 2]}
k=${coefficients[id / 6]}
seed=$((id % 3))
run="$stage/k$k/$arm-$seed"
runner=("${py[@]}" -B "$node_tmp/z1_u22_timed_pair.py" --arm "$arm" --seed-index "$seed" --coefficient "$k" --reference U22 --recorded-phase)
case "$1" in
    qualify)
        if [[ -f "$run/qualified" ]]; then exit 0; fi
        if [[ ! -f "$run/smoke/qualified.json" ]]; then
            "${runner[@]}" --smoke --output "$run/smoke"
        fi
        "${runner[@]}" --worlds 32 --evaluate "$run/smoke/model_0.pt" --output "$run/smoke-eval"
        "${py[@]}" -B "$node_tmp/check_phase.py" "$run" "$arm" "$seed"
        touch "$run/qualified"
        ;;
    train)
        if [[ ${2:-} == k10-only ]]; then
            [[ $k == 10 ]]
            coefficients=(10)
        fi
        for coefficient in "${coefficients[@]}"; do
            for index in 0 1 2; do
                control="$stage/k$coefficient/control-$index"
                preview="$stage/k$coefficient/preview-$index"
                test -f "$control/qualified"
                test -f "$preview/qualified"
                cmp "$control/smoke/shared-initial-weights.json" "$preview/smoke/shared-initial-weights.json"
            done
        done
        "${runner[@]}" --output "$run/production"
        "${runner[@]}" --worlds 32 --evaluate "$run/production/model_199.pt" --output "$run/eval"
        ;;
    *) exit 2 ;;
esac
