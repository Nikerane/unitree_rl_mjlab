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


@pytest.fixture(scope="module")
def env_wrapped():
    cfg = z1_hammer_env_cfg()
    cfg.scene.num_envs = 4
    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(raw)
    wrapped.reset()
    yield wrapped, raw
    raw.close()


# ---------------------------------------------------------------------------
# NailDepthDeltaTerm — settling artefact
# ---------------------------------------------------------------------------

def test_nail_depth_delta_no_spike_on_reset(env_wrapped):
    """nail_depth_delta must not produce reward on step 1 from gravity settling.

    The nail drifts ~3.5 mm downward due to a MuJoCo frictionloss boundary
    artefact. _max_depth is initialised at 4 mm so this drift never triggers
    a positive delta.
    """
    wrapped, raw = env_wrapped
    wrapped.reset()

    # Step once with zero actions — only gravity settling acts on the nail
    _, rewards, _, _ = wrapped.step(torch.zeros(4, 3, device="cuda:0"))

    # Get the nail_depth_delta contribution from the reward manager
    delta_term = raw.reward_manager.get_term("nail_depth_delta")
    # Re-evaluate the term directly to isolate it
    from src.tasks.hammer.mdp.rewards import NailDepthDeltaTerm
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    nail_cfg = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
    nail_cfg.resolve(raw.scene)

    # After reset, _max_depth should be _SETTLE_OFFSET
    term_obj = None
    for name, term in raw.reward_manager._terms.items():
        if isinstance(term, NailDepthDeltaTerm):
            term_obj = term
            break

    assert term_obj is not None, "NailDepthDeltaTerm not found in reward manager"
    assert (term_obj._max_depth == NailDepthDeltaTerm._SETTLE_OFFSET).all(), (
        f"_max_depth should be {NailDepthDeltaTerm._SETTLE_OFFSET} after reset, "
        f"got {term_obj._max_depth}"
    )

    # nail depth at step 1 should be below SETTLE_OFFSET (gravity settling < 4mm)
    nail = raw.scene["nail_block"]
    depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
    assert (depth < NailDepthDeltaTerm._SETTLE_OFFSET).all(), (
        f"Nail depth {depth.max().item():.5f} exceeded SETTLE_OFFSET "
        f"{NailDepthDeltaTerm._SETTLE_OFFSET} — settling artefact larger than expected"
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
    down = torch.tensor([[0., 0., -1.]] * 4, device="cuda:0")
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
    down = torch.tensor([[0., 0., -1.]] * 4, device="cuda:0")

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
