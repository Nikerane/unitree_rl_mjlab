#!/usr/bin/env bash
#SBATCH --job-name=z1-pd500-smoke
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
module load git
readonly CAMPAIGN="${1:?campaign required}"
readonly SOURCE="$CAMPAIGN/source"
readonly PY=/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python
test "$(git -C "$SOURCE" rev-parse HEAD)" = cc46d9998a3c9cd5d8813e89b57e44248353e03e
test "$(git -C /cephhome/eunikhilr/repos/safe_impact_manipulation rev-parse HEAD)" = b58ccd2f81fd246f27c1e8d88cf86484cd888703
export PYTHONPATH="$SOURCE:/cephhome/eunikhilr/campaigns/z1-speed-preflight-20260921.O9UyYJ/test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MUJOCO_GL=egl
export WARP_CACHE_PATH=/cephhome/eunikhilr/campaigns/z1-quintic-m20-pilot/runs/3ca89a2-seed2-250iter/warp-cache
cd "$SOURCE"
sha256sum -c "$CAMPAIGN/source.sha256"
"$PY" docs/research/reward-design/validate_rewards.py > "$CAMPAIGN/reward-gates.log" 2>&1
"$PY" docs/research/reward-design/verify_contact_sensor.py > "$CAMPAIGN/contact-gate.log" 2>&1
"$PY" docs/research/reward-design/verify_reward_setup.py > "$CAMPAIGN/setup-gate.log" 2>&1
"$PY" -m pytest -q tests/test_sampled_pd500.py tests/test_impact_progress_reward.py tests/test_impulse_bound.py tests/test_impulse_constraint.py tests/test_delivered_impulse_reward.py tests/test_cat_soft_hook.py tests/test_impact_speed_policy.py tests/test_impact_speed_contract.py > "$CAMPAIGN/tests.log" 2>&1
"$PY" "$CAMPAIGN/pilot.py" --phase smoke --output "$CAMPAIGN/smoke" > "$CAMPAIGN/smoke.log" 2>&1
