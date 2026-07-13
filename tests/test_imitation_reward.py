"""Unit tests for the weak-annealed tracking prior (ImitationPriorTerm, plan T2).

    r_imit = exp(-||p_head - p*(phi)||^2 / sigma^2) * 1[pre-contact]

Runs against a tiny stub env (no MuJoCo/Warp). The SingleStrikeReference is pure
torch, so get_strike_reference() attaches a real instance to the stub env.

PURITY CONTRACT (2026-07-13, adversarial review F1): the reward term is a pure
READER of the shared reference — it peeks the phase committed by the last
observation pass and never calls update(). The tests therefore drive the
obs-path update explicitly via _obs_update() (in the real env the obs terms
strike_phase/strike_ref_error do this every control step), then call the term.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.references import get_strike_reference
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
    def __init__(self, num_envs: int, air_time: bool = False):
        self.data = SimpleNamespace(found=torch.zeros(num_envs, 1))
        if air_time:
            # Mirrors mjlab ContactSensor track_air_time=True fields ([B, P],
            # per-episode-reset) that the hardened latch ORs in (F2A fix).
            self.data.current_contact_time = torch.zeros(num_envs, 1)
            self.data.last_contact_time = torch.zeros(num_envs, 1)

    def set_found(self, value: bool) -> None:
        self.data.found.fill_(float(value))


def _make_env(num_envs: int = 1, air_time: bool = False):
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3)))
    nail = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3)))
    sensor = _StubSensor(num_envs, air_time=air_time)
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


def _obs_update(env, robot, nail):
    """Simulate the observation pass that commits the phase (strike_phase et al.)."""
    ref = get_strike_reference(env)
    head = robot.data.site_pos_w[:, 0, :]
    nail_top = nail.data.site_pos_w[:, 0, :]
    return ref.update(head, nail_top, env.episode_length_buf)


def test_at_reference_anchor_is_one():
    """First step after reset: obs pass anchors, head == waypoint(0) -> Gaussian=1."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0          # fresh episode -> obs pass anchors here
    _obs_update(env, robot, nail)
    r = term(env, **_PARAMS)
    assert float(r) == pytest.approx(1.0, abs=1e-4)


def test_far_from_reference_decays_to_zero():
    """Head far from the waypoint on a later step -> Gaussian ~ 0."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    _obs_update(env, robot, nail)          # anchor
    term(env, **_PARAMS)
    _set(robot, nail, head=(1.0, 1.0, 1.0))  # far away
    env.episode_length_buf[:] = 1          # no re-anchor
    _obs_update(env, robot, nail)
    sensor.set_found(False)
    r = term(env, **_PARAMS)
    assert float(r) < 0.01


def test_ante_impact_latch_zeroes_from_first_contact():
    """From the first contact onward the term is 0, regardless of distance."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    _obs_update(env, robot, nail)
    assert float(term(env, **_PARAMS)) == pytest.approx(1.0, abs=1e-4)  # pre-contact
    # Contact begins; even sitting on the reference, the term must be 0.
    env.episode_length_buf[:] = 1
    _obs_update(env, robot, nail)
    sensor.set_found(True)
    assert float(term(env, **_PARAMS)) == 0.0
    # Contact released, but the latch persists for the rest of the episode.
    env.episode_length_buf[:] = 2
    _obs_update(env, robot, nail)
    sensor.set_found(False)
    assert float(term(env, **_PARAMS)) == 0.0


def test_within_interval_touch_release_closes_gate():
    """F2A fix: a contact that began AND released inside one control interval is
    invisible to `found` at reward time, but the substep-tracked air-time fields
    latch it — the gate must close on last_contact_time > 0 alone."""
    env, robot, nail, sensor = _make_env(air_time=True)
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    _obs_update(env, robot, nail)
    assert float(term(env, **_PARAMS)) == pytest.approx(1.0, abs=1e-4)
    # Touch-and-release entirely within the interval: found stays 0, but the
    # sensor's first-detach latch recorded the completed contact.
    env.episode_length_buf[:] = 1
    _obs_update(env, robot, nail)
    sensor.data.found.fill_(0.0)
    sensor.data.last_contact_time.fill_(0.004)  # 2 substeps of contact, then release
    assert float(term(env, **_PARAMS)) == 0.0
    # And the latch persists after the air-time fields go quiet again.
    env.episode_length_buf[:] = 2
    _obs_update(env, robot, nail)
    sensor.data.last_contact_time.fill_(0.0)
    assert float(term(env, **_PARAMS)) == 0.0


def test_reward_call_never_mutates_phase():
    """F1/R2-F1 fix: the reward term is a pure reader — repeated calls with moving
    (even rebounding) kinematics must not advance, re-anchor, or latch anything on
    the shared reference (preview() computes from current kinematics, writes nothing)."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.30))
    env.episode_length_buf[:] = 0
    _obs_update(env, robot, nail)          # anchor; phi = 0
    ref = get_strike_reference(env)
    phi_before = ref._phi.clone()
    anchored_before = ref._anchored.clone()
    s0_before = ref._s0.clone()
    # Deep-descent kinematics + later step index: update() would advance phi here.
    _set(robot, nail, head=(0.0, 0.0, 0.105))
    env.episode_length_buf[:] = 7
    for _ in range(3):
        term(env, **_PARAMS)
    assert torch.equal(ref._phi, phi_before)
    assert torch.equal(ref._anchored, anchored_before)
    assert torch.equal(ref._s0, s0_before)
    # The returned phase is a defensive copy: mutating it must not leak back.
    peeked = ref.peek()
    peeked += 123.0
    assert torch.equal(ref._phi, phi_before)


def test_latch_resets_per_episode():
    """reset() clears the contact latch so a new episode tracks again."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    _obs_update(env, robot, nail)
    term(env, **_PARAMS)
    env.episode_length_buf[:] = 1
    _obs_update(env, robot, nail)
    sensor.set_found(True)
    assert float(term(env, **_PARAMS)) == 0.0  # latched
    term.reset(None)
    sensor.set_found(False)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0              # re-anchor via the obs pass
    _obs_update(env, robot, nail)
    assert float(term(env, **_PARAMS)) == pytest.approx(1.0, abs=1e-4)


def test_returns_per_env_shape():
    env, robot, nail, sensor = _make_env(num_envs=3)
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    _obs_update(env, robot, nail)
    r = term(env, **_PARAMS)
    assert tuple(r.shape) == (3,)


def test_per_env_latch_independence():
    """One env's contact latch must not zero another env's reward."""
    env, robot, nail, sensor = _make_env(num_envs=2)
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    _obs_update(env, robot, nail)
    term(env, **_PARAMS)  # anchor both envs
    # env 0 contacts, env 1 does not; both stay near the reference.
    env.episode_length_buf[:] = 1
    _obs_update(env, robot, nail)
    sensor.data.found[0, 0] = 1.0
    sensor.data.found[1, 0] = 0.0
    r = term(env, **_PARAMS)
    assert float(r[0]) == 0.0   # env 0 latched off
    assert float(r[1]) > 0.0    # env 1 still tracking
