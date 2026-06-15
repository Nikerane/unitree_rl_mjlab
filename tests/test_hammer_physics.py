"""Physics and reward correctness checks for the hammer-nail task.

Run with:
    cd unitree_rl_mjlab
    python tests/test_hammer_physics.py
"""

import sys
sys.path.insert(0, "src")
import src  # noqa: F401

import torch
import pytest
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.rewards import NailDepthDeltaTerm

# Device-agnostic (was hardcoded cuda:0 → errored on CPU-only machines so these
# never ran on the Mac). Uses GPU when available.
_DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def env_wrapped():
    cfg = z1_hammer_env_cfg()
    cfg.scene.num_envs = 4
    raw = ManagerBasedRlEnv(cfg=cfg, device=_DEV)
    wrapped = RslRlVecEnvWrapper(raw)
    wrapped.reset()
    yield wrapped, raw
    raw.close()


# ---------------------------------------------------------------------------
# NailDepthDeltaTerm — settling artefact
# ---------------------------------------------------------------------------

def test_nail_depth_delta_no_spike_on_reset(env_wrapped):
    """nail_depth_delta must not produce reward on step 1 from any reset transient.

    Since the 2026-06-15 gravcomp fix the nail no longer creeps under gravity
    (it holds at ~0), so this is doubly safe: the _max_depth dead-zone
    (_SETTLE_OFFSET) is now belt-and-suspenders rather than load-bearing.
    """
    wrapped, raw = env_wrapped
    wrapped.reset()

    # Step once with zero actions — nothing should move the nail.
    _, rewards, _, _ = wrapped.step(torch.zeros(4, 3, device=_DEV))

    # Observable behaviour (robust to manager internals): the nail_depth_delta
    # term contributes ~0 on a no-op step, and the nail itself hasn't moved.
    rm = raw.reward_manager
    idx = list(rm.active_terms).index("nail_depth_delta")
    delta_reward = rm._step_reward[:, idx]
    assert (delta_reward.abs() < 1e-6).all(), (
        f"nail_depth_delta fired on a zero-action step: {delta_reward}"
    )
    depth = raw.scene["nail_block"].data.joint_pos[:, 0]
    assert (depth.abs() < 5e-4).all(), (
        f"Nail moved {depth.abs().max().item()*1000:.3f} mm under zero action "
        "— gravity-creep bug (gravcomp on the nail body missing?)."
    )


def test_nail_depth_delta_rewards_real_driving(env_wrapped):
    """Nail must be driven well past SETTLE_OFFSET (4 mm) during a downward strike.

    We check peak nail depth rather than computing delta outside the step, because
    after nail_driven termination fires the env resets inside wrapped.step() —
    reading _max_depth after the call always sees the post-reset 0.004 value.
    """
    wrapped, raw = env_wrapped
    wrapped.reset()

    nail = raw.scene["nail_block"]
    down = torch.tensor([[0., 0., -1.]] * 4, device=_DEV)
    peak_depth = 0.0

    for _ in range(20):
        wrapped.step(down)
        d = nail.data.joint_pos[:, 0].max().item()
        if d > peak_depth:
            peak_depth = d

    assert peak_depth > NailDepthDeltaTerm._SETTLE_OFFSET * 5, (
        f"Nail only reached {peak_depth*1000:.1f} mm — expected > "
        f"{NailDepthDeltaTerm._SETTLE_OFFSET*5*1000:.0f} mm. "
        "Arm may not be contacting the nail."
    )


# ---------------------------------------------------------------------------
# ContactSensor — sanity check (prerequisite for future impact_velocity_bonus)
# ---------------------------------------------------------------------------

def test_contact_sensor_wired(env_wrapped):
    """hammer_nail_contact sensor must be present in the scene."""
    _, raw = env_wrapped
    assert "hammer_nail_contact" in raw.scene.sensors, (
        "hammer_nail_contact sensor not found — impact_velocity_bonus cannot be added"
    )


def test_contact_sensor_no_contact_at_reset(env_wrapped):
    """Sensor must report no contact immediately after reset (arm is 12 cm above nail)."""
    wrapped, raw = env_wrapped
    wrapped.reset()
    sensor = raw.scene.sensors["hammer_nail_contact"]
    # found is shape (B, 1) float — nonzero means contact
    assert not (sensor.data.found > 0).any(), (
        f"Unexpected contact at reset: {sensor.data.found}. "
        "Arm may be initialised overlapping the nail."
    )


def test_contact_sensor_detects_contact_on_strike(env_wrapped):
    """Sensor must fire within 15 steps of sustained max downward action."""
    wrapped, raw = env_wrapped
    wrapped.reset()
    sensor = raw.scene.sensors["hammer_nail_contact"]
    down = torch.tensor([[0., 0., -1.]] * 4, device=_DEV)

    contact_ever_detected = False
    for _ in range(15):
        wrapped.step(down)
        if (sensor.data.found > 0).any():
            contact_ever_detected = True
            break

    assert contact_ever_detected, (
        "ContactSensor never detected hammer-nail contact across 15 downward steps. "
        "Nail may be out of reach — check nail_block_scene.xml body positions."
    )


if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        capture_output=False
    )
    sys.exit(result.returncode)
