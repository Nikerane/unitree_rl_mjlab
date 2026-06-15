#!/usr/bin/env bash
# One-time Vega bring-up (run on the LOGIN node — it has internet).
# Clones both repos as siblings and builds a uv venv with pinned, known-good deps.
# Idempotent-ish: re-running re-syncs git and re-installs.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/repos}"
BRANCH="${BRANCH:-hammer-z1}"
mkdir -p "$REPO_ROOT"
cd "$REPO_ROOT"

# 1. uv (fetches its own CPython — system python is 3.6.8, unusable)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

# 2. Repos as siblings (asset path resolves via parents[5]/safe_impact_manipulation)
clone_or_pull() {  # $1=name $2=url $3=branch
  if [ -d "$1/.git" ]; then git -C "$1" fetch --quiet origin && git -C "$1" checkout "$3" && git -C "$1" pull --ff-only --quiet
  else git clone --branch "$3" "https://github.com/Nikerane/$2.git" "$1"; fi
}
clone_or_pull unitree_rl_mjlab unitree_rl_mjlab "$BRANCH"
clone_or_pull safe_impact_manipulation safe_impact_manipulation main

# 3. Venv + pinned deps (versions = known-good local env, 2026-06)
cd "$REPO_ROOT/unitree_rl_mjlab"
uv venv --python 3.12 .venv
# torch first (CUDA build); if the default index lacks a CUDA wheel, re-run with
#   UV_TORCH_INDEX=https://download.pytorch.org/whl/cu124  (set before calling)
TORCH_ARGS=()
[ -n "${UV_TORCH_INDEX:-}" ] && TORCH_ARGS=(--index-url "$UV_TORCH_INDEX")
uv pip install "${TORCH_ARGS[@]}" "torch==2.12.0" || uv pip install "${TORCH_ARGS[@]}" torch
uv pip install "mjlab==1.4.0" "mujoco==3.8.1" "mujoco-warp==3.8.1"
uv pip install -e .

# 4. Smoke import (CPU, no GPU needed) — confirms the asset path resolves
.venv/bin/python -c "
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
cfg = z1_hammer_env_cfg(); print('[setup] env cfg built OK; deps + asset path resolve')
import torch; print('[setup] torch', torch.__version__, 'cuda_build', torch.version.cuda)
"
echo "[setup] done. Next: sbatch scripts/slurm/sanity.sbatch"
