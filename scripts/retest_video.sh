#!/usr/bin/env bash
# Retest mjlab offscreen / training video pipeline for Unitree-Z1-Hammer.
# Usage: ./scripts/retest_video.sh [checkpoint.pt]
set -euo pipefail

cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASK="Unitree-Z1-Hammer"
CKPT="${1:-}"

echo "=== 0) Environment ==="
echo "DISPLAY=${DISPLAY:-unset}  MUJOCO_GL=${MUJOCO_GL}  PWD=$(pwd)"

echo
echo "=== 1) Offscreen rgb_array (no viewer) ==="
PYTHON="${PYTHON:-python}"
"$PYTHON" - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
import numpy as np
import torch
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task = "Unitree-Z1-Hammer"
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 1
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
env.reset()
for _ in range(5):
    env.step(torch.zeros(1, env.action_space.shape[-1], device=device))
frame = np.asarray(env.render())
env.close()
nz = float((frame > 10).mean())
print(f"frame {frame.shape} mean={frame.mean():.1f} nz={nz:.3f}")
if frame.max() < 5 or nz < 0.01:
    raise SystemExit("BLANK/BLACK frame — check MUJOCO_GL, camera, EGL")
print("OK: scene visible in offscreen render")
PY

echo
echo "=== 2) Interactive viewer (zero agent) ==="
echo "Run manually (opens UI):"
echo "  python scripts/play.py ${TASK} --agent zero --viewer viser"
echo "  python scripts/play.py ${TASK} --agent zero --viewer native"

echo
echo "=== 3) Play video (trained agent only; dummy+video disabled in play.py) ==="
if [[ -z "${CKPT}" ]]; then
  echo "Skip: pass checkpoint, e.g.:"
  echo "  ./scripts/retest_video.sh logs/rsl_rl/z1_hammer/<run>/model_0.pt"
else
  "$PYTHON" scripts/play.py "${TASK}" \
    --checkpoint-file "${CKPT}" \
    --video True \
    --video-length 100 \
    --num-envs 1
  echo "Check: $(dirname "${CKPT}")/videos/play/"
fi

echo
echo "=== 4) Train video (small run) ==="
echo "  export MUJOCO_GL=egl"
echo "  python scripts/train.py ${TASK} --env.scene.num-envs 16 --video True"
echo "Videos: logs/rsl_rl/z1_hammer/<timestamp>/videos/train/"
