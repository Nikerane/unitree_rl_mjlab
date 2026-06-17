"""Unit tests for the weak-annealed tracking prior (ImitationPriorTerm, plan T2).

    r_imit = exp(-||p_head - p*(phi)||^2 / sigma^2) * 1[pre-contact]

Runs against a tiny stub env (no MuJoCo/Warp). The SingleStrikeReference is pure
torch, so get_strike_reference() attaches a real instance to the stub env.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.rewards import ImitationPriorTerm

_ROBOT_CFG = SimpleNamespace(name="robot", site_ids=[0])
_NAIL_CFG = SimpleNamespace(name="nail_block", site_ids=[0])
_PARAMS = dict(
    sensor_name="hammer_nail_contact",
    robot_cfg=_ROBOT_CFG,
    nail_cfg=_NAIL_CFG,
    sigma=0.05,
)


class _StubSensor:
    def __init__(self, num_envs: int):
        self.data = SimpleNamespace(found=torch.zeros(num_envs, 1))

    def set_found(self, value: bool) -> None:
        self.data.found.fill_(float(value))


def _make_env(num_envs: int = 1):
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3)))
    nail = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3)))
    sensor = _StubSensor(num_envs)
    scene = {"robot": robot, "nail_block": nail, "hammer_nail_contact": sensor}
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )
    return env, robot, nail, sensor


def _set(robot, nail, *, head, nail_top=(0.0, 0.0, 0.10)):
    robot.data.site_pos_w[:, 0, :] = torch.tensor(head)
    nail.data.site_pos_w[:, 0, :] = torch.tensor(nail_top)


def test_at_reference_anchor_is_one():
    """First call after reset: head == anchor == waypoint(0) -> Gaussian=1, pre-contact -> 1."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0          # fresh episode -> reference anchors here
    r = term(env, **_PARAMS)
    assert float(r) == pytest.approx(1.0, abs=1e-4)


def test_far_from_reference_decays_to_zero():
    """Head far from the waypoint on a later step -> Gaussian ~ 0."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    term(env, **_PARAMS)                   # anchor
    _set(robot, nail, head=(1.0, 1.0, 1.0))  # far away
    env.episode_length_buf[:] = 1          # no re-anchor
    sensor.set_found(False)
    r = term(env, **_PARAMS)
    assert float(r) < 0.01


def test_ante_impact_latch_zeroes_from_first_contact():
    """From the first contact onward the term is 0, regardless of distance."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    assert float(term(env, **_PARAMS)) == pytest.approx(1.0, abs=1e-4)  # pre-contact
    # Contact begins; even sitting on the reference, the term must be 0.
    env.episode_length_buf[:] = 1
    sensor.set_found(True)
    assert float(term(env, **_PARAMS)) == 0.0
    # Contact released, but the latch persists for the rest of the episode.
    env.episode_length_buf[:] = 2
    sensor.set_found(False)
    assert float(term(env, **_PARAMS)) == 0.0


def test_latch_resets_per_episode():
    """reset() clears the contact latch so a new episode tracks again."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    term(env, **_PARAMS)
    env.episode_length_buf[:] = 1
    sensor.set_found(True)
    assert float(term(env, **_PARAMS)) == 0.0  # latched
    term.reset(None)
    sensor.set_found(False)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0              # re-anchor
    assert float(term(env, **_PARAMS)) == pytest.approx(1.0, abs=1e-4)


def test_returns_per_env_shape():
    env, robot, nail, sensor = _make_env(num_envs=3)
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    r = term(env, **_PARAMS)
    assert tuple(r.shape) == (3,)
