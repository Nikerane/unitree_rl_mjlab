"""Shared pytest fixtures and configuration for the unitree_rl_mjlab test suite."""

import pytest


# ---------------------------------------------------------------------------
# Integration fixtures — require the full mjlab + Warp stack.
# These are module-scoped to avoid recompiling Warp kernels per test.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env_cpu():
    """Create a Z1 hammer ManagerBasedRlEnv on CPU; close after the module."""
    from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    cfg = z1_hammer_env_cfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


@pytest.fixture(scope="module")
def obs_after_reset(env_cpu):
    """Reset the env once per module and return the observation dict."""
    obs, _ = env_cpu.reset()
    return obs
