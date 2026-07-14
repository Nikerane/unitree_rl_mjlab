"""Verify nail physics: stays at 0 under gravity, moves when struck downward."""

import sys
sys.path.insert(0, "src")
import src  # noqa: F401 — loads compat shim

import torch
import pytest
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg

# Full env-stack physics rollouts (among the slowest tests in the suite) — keep them out of the
# documented fast path (`pytest -m "not integration"`).
pytestmark = pytest.mark.integration

# Device-agnostic (was hardcoded cuda:0, which errored on CPU-only machines and
# meant these behavioural tests never ran on the Mac). Runs on GPU when present.
_DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def env():
    cfg = z1_hammer_env_cfg()
    cfg.scene.num_envs = 1
    _env = ManagerBasedRlEnv(cfg=cfg, device=_DEV)
    wrapped = RslRlVecEnvWrapper(_env)
    wrapped.reset()
    yield wrapped, _env
    _env.close()


def nail_depth(raw_env):
    return raw_env.scene["nail_block"].data.joint_pos[:, 0].item()


def test_nail_stable_at_rest(env):
    """With zero actions the nail must stay put (within 0.5 mm of qpos=0).

    Regression test for the gravity-creep bug (2026-06-15): MuJoCo joint
    frictionloss does NOT statically hold the nail, so without gravcomp="1" on
    the nail body it free-creeps downward at ~9.6 mm/s under its own weight and
    would self-reach the success depth with no hammer. gravcomp fixes it at the
    source; the nail now holds at ~0. Tight 0.5 mm tolerance guards the fix.
    """
    wrapped, raw_env = env
    wrapped.reset()
    for _ in range(40):
        wrapped.step(torch.zeros(1, 3, device=_DEV))
    depth = nail_depth(raw_env)
    assert abs(depth) < 5e-4, (
        f"Nail drifted to {depth*1000:.3f} mm under zero action — gravcomp on the "
        "nail body may be missing (gravity-creep bug)."
    )


def test_nail_does_not_go_negative(env):
    """Nail qpos must never go below -1 mm (nail pushed upward = physics bug)."""
    wrapped, raw_env = env
    wrapped.reset()
    min_depth = 0.0
    for _ in range(30):
        wrapped.step(torch.zeros(1, 3, device=_DEV))
        min_depth = min(min_depth, nail_depth(raw_env))
    assert min_depth > -1e-3, f"Nail went negative: {min_depth:.5f} m (shaft-block collision not fixed)"


def test_nail_driven_by_downward_action(env):
    """Sustained strike toward the nail must drive it > 5 mm within 60 steps.

    Drives the IK head along the actual head->nail_top gap direction (recomputed
    each step, closed-loop) instead of a blind world [0, 0, -1]. The old blind
    action silently assumed the hammer face pointed straight down in the world
    frame; after the grasp-#10 remount the face-to-nail line is not world-vertical,
    so [0, 0, -1] under-drove the nail and made this test flaky. Aiming at the nail
    exercises a real strike regardless of how the hammer is mounted.
    """
    wrapped, raw_env = env
    wrapped.reset()
    robot = raw_env.scene["robot"]
    nail = raw_env.scene["nail_block"]
    head_ids, _ = robot.find_sites("hammer_head_site")
    nail_ids, _ = nail.find_sites("nail_top")
    max_depth = 0.0
    for _ in range(60):
        head = robot.data.site_pos_w[:, head_ids].squeeze(1)   # (B, 3)
        ntop = nail.data.site_pos_w[:, nail_ids].squeeze(1)     # (B, 3)
        gap = ntop - head
        action = gap / gap.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # unit dir
        wrapped.step(action.to(_DEV))
        max_depth = max(max_depth, nail_depth(raw_env))
    assert max_depth > 5e-3, (
        f"Nail only reached {max_depth*1000:.2f} mm after 60 strike steps "
        f"(expected > 5 mm). frictionloss or contact may still be blocking."
    )
