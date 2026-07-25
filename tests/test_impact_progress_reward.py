"""Unit tests for the `impact_progress` reward term (ImpactProgressTerm).

Design source: docs/archive/IMPACT_PROGRESS_IMPL_SPEC.md

The term implements the double-gated momentum reward

    r = (v_axial / v_expected) * 1[first_contact] * 1[depth advanced > eps]

These tests run in isolation against a tiny stub env (no MuJoCo / Warp): the
three gates are independent tensor values, so scrape-vs-strike is fully
deterministic. Pure Python + torch → no marker (runs under `-m "not integration"`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
from src.tasks.hammer.mdp.rewards import (
    FirstStrikeImpactRewardTerm,
    FirstStrikeLegacyImpactRewardTerm,
    ImpactProgressTerm,
)


# --- Stub env -------------------------------------------------------------
# ImpactProgressTerm reads only: env.num_envs, env.device, env.step_dt,
# env.scene[name], robot.data.site_pos_w, nail.data.joint_pos, and
# sensor.compute_first_contact(dt). All are trivially fakeable.


class _StubSensor:
    def __init__(self, num_envs: int):
        self._fc = torch.zeros(num_envs, 1, dtype=torch.bool)

    def set_first_contact(self, value: bool) -> None:
        self._fc.fill_(bool(value))

    def compute_first_contact(self, dt):  # noqa: ARG002 - dt unused by stub
        return self._fc.clone()


# Resolved-SceneEntityCfg stand-ins (the real SceneEntityCfg exposes the same
# `.name` / `.site_ids` / `.joint_ids` after `.resolve(scene)`).
_ROBOT_CFG = SimpleNamespace(name="robot", site_ids=[0])
_NAIL_CFG = SimpleNamespace(name="nail_block", joint_ids=[0])
_PARAMS = dict(
    sensor_name="hammer_nail_contact",
    robot_cfg=_ROBOT_CFG,
    nail_cfg=_NAIL_CFG,
    axis=(0.0, 0.0, -1.0),
    eps=5e-4,
    v_expected=1.0,
)


def _make_stub_env(num_envs: int = 1, step_dt: float = 0.02):
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3)))
    nail = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(num_envs, 1)))
    sensor = _StubSensor(num_envs)
    scene = {"robot": robot, "nail_block": nail, "hammer_nail_contact": sensor}
    env = SimpleNamespace(num_envs=num_envs, device="cpu", step_dt=step_dt, scene=scene)
    return env, robot, nail, sensor


def _step(term, env, robot, nail, sensor, *, head_z, depth, first_contact, head_x=0.0):
    """Set the per-step stub state, then evaluate the term once."""
    robot.data.site_pos_w[:, 0, 0] = head_x
    robot.data.site_pos_w[:, 0, 2] = head_z
    nail.data.joint_pos[:, 0] = depth
    sensor.set_first_contact(first_contact)
    return term(env, **_PARAMS)


# --- Tests ----------------------------------------------------------------


def test_productive_strike_rewards_axial_speed():
    """Fresh contact + downward speed + real depth advance → positive, scaled reward."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    # Warm-up call establishes prev head position (velocity is 0 on the first step).
    r0 = _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.0, first_contact=False)
    assert float(r0) == 0.0
    # Head drops 1 cm in one 0.02 s step → v_axial = 0.5 m/s; nail advances 1 cm; fresh contact.
    r1 = _step(term, env, robot, nail, sensor, head_z=0.04, depth=0.01, first_contact=True)
    assert float(r1) == pytest.approx(0.5)  # (0.5 / 1.0) * 1 * 1


def test_scrape_without_depth_advance_is_zero():
    """Fast contact that does not move the nail earns nothing (depth gate)."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.0, first_contact=False)
    r = _step(term, env, robot, nail, sensor, head_z=0.04, depth=0.0, first_contact=True)
    assert float(r) == 0.0


def test_contact_at_peak_depth_is_zero():
    """A contact at or below the max depth so far makes no new progress → zero."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.0, first_contact=False)
    _step(term, env, robot, nail, sensor, head_z=0.04, depth=0.02, first_contact=True)  # peak = 2 cm
    r = _step(term, env, robot, nail, sensor, head_z=0.03, depth=0.02, first_contact=True)
    assert float(r) == 0.0


def test_no_first_contact_is_zero():
    """Downward speed + depth advance but no fresh contact (e.g. resting) → zero."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.0, first_contact=False)
    r = _step(term, env, robot, nail, sensor, head_z=0.04, depth=0.01, first_contact=False)
    assert float(r) == 0.0


def test_horizontal_motion_has_no_axial_speed():
    """Purely horizontal head motion → v_axial = 0 even with contact + depth advance."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    _step(term, env, robot, nail, sensor, head_x=0.0, head_z=0.05, depth=0.0, first_contact=False)
    r = _step(term, env, robot, nail, sensor, head_x=0.02, head_z=0.05, depth=0.01, first_contact=True)
    assert float(r) == 0.0


def test_upward_motion_is_clamped_to_zero():
    """Retracting (upward) head motion has negative axial speed → clamp_min(0) = 0."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    _step(term, env, robot, nail, sensor, head_z=0.04, depth=0.0, first_contact=False)
    r = _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.01, first_contact=True)
    assert float(r) == 0.0


def test_depth_advance_below_epsilon_is_zero():
    """A sub-epsilon depth advance (< 5e-4 m) does not trip the progress gate."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.0, first_contact=False)
    r = _step(term, env, robot, nail, sensor, head_z=0.04, depth=1e-4, first_contact=True)
    assert float(r) == 0.0


def test_first_step_after_reset_has_no_velocity():
    """On the very first call (no prior head pos) velocity is 0, so reward is 0."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    r = _step(term, env, robot, nail, sensor, head_z=0.04, depth=0.01, first_contact=True)
    assert float(r) == 0.0


def test_reset_clears_depth_history():
    """reset() must zero the max-depth tracker so a new episode is scored afresh."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.0, first_contact=False)
    _step(term, env, robot, nail, sensor, head_z=0.04, depth=0.02, first_contact=True)  # peak = 2 cm
    term.reset(None)
    # A fresh strike to 0.5 cm would be 0 if the 2 cm peak persisted; reset → it's rewarded.
    _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.0, first_contact=False)
    r = _step(term, env, robot, nail, sensor, head_z=0.04, depth=0.005, first_contact=True)
    assert float(r) == pytest.approx(0.5)


def test_v_expected_normalisation_scales_reward():
    """Dividing by a larger v_expected shrinks the bonus proportionally."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    params = {**_PARAMS, "v_expected": 2.0}
    robot.data.site_pos_w[:, 0, 2] = 0.05
    sensor.set_first_contact(False)
    term(env, **params)  # warm-up
    robot.data.site_pos_w[:, 0, 2] = 0.04  # v_axial = 0.5 m/s
    nail.data.joint_pos[:, 0] = 0.01
    sensor.set_first_contact(True)
    r = term(env, **params)
    assert float(r) == pytest.approx(0.25)  # 0.5 / 2.0


def test_returns_per_env_shape():
    """Reward is shape (num_envs,)."""
    env, robot, nail, sensor = _make_stub_env(num_envs=3)
    term = ImpactProgressTerm(cfg=None, env=env)
    r = _step(term, env, robot, nail, sensor, head_z=0.05, depth=0.0, first_contact=False)
    assert tuple(r.shape) == (3,)


def test_bad_v_expected_raises():
    """F6 (2026-07-19 audit): a CLI sweep passing a zero/NaN normalizer must fail loud, not divide by 0."""
    env, robot, nail, sensor = _make_stub_env()
    term = ImpactProgressTerm(cfg=None, env=env)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="v_expected"):
            term(env, **{**_PARAMS, "v_expected": bad})


# --- Event-correct one-shot reward ----------------------------------------


def _event_env(
    *,
    finalized=True,
    productive=True,
    v_precontact=1.0,
    num_envs=1,
    attach_tracker=True,
):
    env = SimpleNamespace(num_envs=num_envs, device="cpu")
    tracker = SimpleNamespace(
        finalized=torch.full((num_envs,), bool(finalized), dtype=torch.bool),
        productive=torch.full((num_envs,), bool(productive), dtype=torch.bool),
        v_precontact=torch.full((num_envs,), float(v_precontact)),
        delivered=torch.zeros(num_envs),
    )
    if attach_tracker:
        setattr(env, _ENV_FIRST_STRIKE_ATTR, tracker)
    return env, tracker


def test_event_reward_waits_until_tracker_finalizes():
    """A productive snapshot is unreadable until its shared event is final."""
    env, tracker = _event_env(finalized=False, v_precontact=2.0, attach_tracker=False)
    term = FirstStrikeImpactRewardTerm(cfg=None, env=env)
    # Reward terms are constructed before metrics, so tracker lookup must be lazy.
    setattr(env, _ENV_FIRST_STRIKE_ATTR, tracker)

    assert torch.equal(term(env, v_expected=1.0), torch.tensor([0.0]))
    tracker.finalized.fill_(True)
    assert torch.equal(term(env, v_expected=1.0), torch.tensor([2.0]))


def test_event_reward_pays_once_only():
    """A finalized productive snapshot cannot be collected twice."""
    env, _ = _event_env(v_precontact=1.5)
    term = FirstStrikeImpactRewardTerm(cfg=None, env=env)

    assert torch.equal(term(env, v_expected=1.0), torch.tensor([1.5]))
    assert torch.equal(term(env, v_expected=1.0), torch.tensor([0.0]))


def test_event_reward_rejects_unproductive_snapshot():
    """Fast scrape/tap snapshots with no real nail progress earn zero."""
    env, _ = _event_env(productive=False, v_precontact=100.0)
    term = FirstStrikeImpactRewardTerm(cfg=None, env=env)

    assert torch.equal(term(env, v_expected=1.0), torch.tensor([0.0]))


def test_event_impact_uses_latched_precontact_speed():
    """Impact payout comes directly from the 500 Hz tracker's onset snapshot."""
    env, _ = _event_env(v_precontact=1.25)
    term = FirstStrikeImpactRewardTerm(cfg=None, env=env)

    assert torch.equal(term(env, v_expected=0.5), torch.tensor([2.5]))


def test_event_reward_reset_rearms_paid_latch():
    """Resetting selected envs re-enables only those one-shot payouts."""
    env, tracker = _event_env(num_envs=2)
    tracker.v_precontact[:] = torch.tensor([1.0, 2.0])
    term = FirstStrikeImpactRewardTerm(cfg=None, env=env)
    assert torch.equal(term(env, v_expected=1.0), torch.tensor([1.0, 2.0]))

    term.reset(torch.tensor([0]))

    assert torch.equal(term(env, v_expected=1.0), torch.tensor([1.0, 0.0]))


def _legacy_tracker_env(num_envs=1):
    env = SimpleNamespace(num_envs=num_envs, device="cpu")
    tracker = SimpleNamespace(
        started=torch.zeros(num_envs, dtype=torch.bool),
        finalized=torch.zeros(num_envs, dtype=torch.bool),
        productive=torch.zeros(num_envs, dtype=torch.bool),
    )
    setattr(env, _ENV_FIRST_STRIKE_ATTR, tracker)
    return env, tracker


def _patch_impact_raw(monkeypatch, values):
    sequence = iter(torch.as_tensor(value, dtype=torch.float32) for value in values)
    monkeypatch.setattr(
        ImpactProgressTerm,
        "__call__",
        lambda self, env, **params: next(sequence),
    )


def test_legacy_impact_waits_then_pays_first_positive_pulse_once(monkeypatch):
    env, tracker = _legacy_tracker_env()
    _patch_impact_raw(monkeypatch, [[0.0], [1.25], [1.75], [0.0]])
    term = FirstStrikeLegacyImpactRewardTerm(cfg=None, env=env)
    outputs = []
    for started, finalized, productive in (
        (False, False, False),
        (True, False, False),
        (True, True, True),
        (True, True, True),
    ):
        tracker.started.fill_(started)
        tracker.finalized.fill_(finalized)
        tracker.productive.fill_(productive)
        outputs.append(float(term(env)))
    assert outputs == [0.0, 0.0, 1.25, 0.0]


def test_legacy_impact_ignores_second_positive_pulse_inside_window(monkeypatch):
    env, tracker = _legacy_tracker_env()
    _patch_impact_raw(monkeypatch, [[1.25], [1.75], [9.0]])
    term = FirstStrikeLegacyImpactRewardTerm(cfg=None, env=env)
    tracker.started.fill_(True)
    assert float(term(env)) == 0.0
    assert float(term(env)) == 0.0
    tracker.finalized.fill_(True)
    tracker.productive.fill_(True)
    assert float(term(env)) == pytest.approx(1.25)


def test_legacy_wrappers_include_finalization_boundary_values(monkeypatch):
    env, tracker = _legacy_tracker_env()
    _patch_impact_raw(monkeypatch, [[0.0], [2.0]])
    term = FirstStrikeLegacyImpactRewardTerm(cfg=None, env=env)
    tracker.started.fill_(True)
    assert float(term(env)) == 0.0
    tracker.finalized.fill_(True)
    tracker.productive.fill_(True)
    assert float(term(env)) == pytest.approx(2.0)


def test_legacy_wrappers_consume_unproductive_finalization_without_payout(monkeypatch):
    env, tracker = _legacy_tracker_env()
    _patch_impact_raw(monkeypatch, [[1.0], [5.0]])
    term = FirstStrikeLegacyImpactRewardTerm(cfg=None, env=env)
    tracker.started.fill_(True)
    tracker.finalized.fill_(True)
    assert float(term(env)) == 0.0
    tracker.productive.fill_(True)
    assert float(term(env)) == 0.0


def test_legacy_wrappers_ignore_delayed_recontact(monkeypatch):
    env, tracker = _legacy_tracker_env()
    _patch_impact_raw(monkeypatch, [[1.0], [7.0]])
    term = FirstStrikeLegacyImpactRewardTerm(cfg=None, env=env)
    tracker.started.fill_(True)
    tracker.finalized.fill_(True)
    tracker.productive.fill_(True)
    assert float(term(env)) == pytest.approx(1.0)
    assert float(term(env)) == 0.0


def test_legacy_wrappers_reset_selected_environments_only(monkeypatch):
    env, tracker = _legacy_tracker_env(num_envs=2)
    _patch_impact_raw(monkeypatch, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    term = FirstStrikeLegacyImpactRewardTerm(cfg=None, env=env)
    tracker.started.fill_(True)
    tracker.finalized.fill_(True)
    tracker.productive.fill_(True)
    assert torch.equal(term(env), torch.tensor([1.0, 2.0]))

    term.reset(torch.tensor([0]))
    tracker.finalized[:] = torch.tensor([False, True])
    assert torch.equal(term(env), torch.zeros(2))
    tracker.finalized[:] = True
    assert torch.equal(term(env), torch.tensor([3.0, 0.0]))


def test_legacy_wrapper_inner_outputs_match_standalone_legacy_terms():
    standalone_env, robot_s, nail_s, sensor_s = _make_stub_env()
    wrapped_env, robot_w, nail_w, sensor_w = _make_stub_env()
    tracker = SimpleNamespace(
        started=torch.tensor([False]),
        finalized=torch.tensor([False]),
        productive=torch.tensor([False]),
    )
    setattr(wrapped_env, _ENV_FIRST_STRIKE_ATTR, tracker)
    standalone = ImpactProgressTerm(cfg=None, env=standalone_env)
    wrapped = FirstStrikeLegacyImpactRewardTerm(cfg=None, env=wrapped_env)

    _step(
        standalone, standalone_env, robot_s, nail_s, sensor_s,
        head_z=0.05, depth=0.0, first_contact=False,
    )
    _step(
        wrapped, wrapped_env, robot_w, nail_w, sensor_w,
        head_z=0.05, depth=0.0, first_contact=False,
    )
    tracker.started.fill_(True)
    legacy_raw = _step(
        standalone, standalone_env, robot_s, nail_s, sensor_s,
        head_z=0.04, depth=0.01, first_contact=True,
    )
    assert float(_step(
        wrapped, wrapped_env, robot_w, nail_w, sensor_w,
        head_z=0.04, depth=0.01, first_contact=True,
    )) == 0.0
    tracker.finalized.fill_(True)
    tracker.productive.fill_(True)
    payout = _step(
        wrapped, wrapped_env, robot_w, nail_w, sensor_w,
        head_z=0.04, depth=0.01, first_contact=False,
    )
    assert torch.equal(payout, legacy_raw)


def test_event_impact_requires_shared_tracker():
    env, _ = _event_env(attach_tracker=False)
    term = FirstStrikeImpactRewardTerm(cfg=None, env=env)

    with pytest.raises(RuntimeError, match="FirstStrikeEventTracker"):
        term(env, v_expected=1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_event_impact_rejects_bad_normalizer(bad):
    env, _ = _event_env()
    term = FirstStrikeImpactRewardTerm(cfg=None, env=env)

    with pytest.raises(ValueError, match="v_expected"):
        term(env, v_expected=bad)
