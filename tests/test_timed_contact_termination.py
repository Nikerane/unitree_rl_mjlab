"""The impulse branch reuses the native first-contact clock and miss timeout."""
from types import SimpleNamespace

import pytest
import torch

from scripts.z1_safety_factorial import build_config
from scripts.z1_u22_timed_pair import configure_termination, reference_exhausted
from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete


def config():
    return build_config('B', 0)[1].env


def test_contact_mode_keeps_existing_timer_and_five_second_cap():
    cfg = config()
    native = cfg.terminations['nail_driven']
    timeout = cfg.terminations['time_out']
    assert configure_termination(cfg, 21, 'contact') == 250
    assert 'reference_exhausted' not in cfg.terminations
    assert cfg.terminations['nail_driven'] is native
    assert native.func is horizon_window_complete and native.params == {'window_s': .2}
    assert native.time_out is False and cfg.terminations['time_out'] is timeout
    assert cfg.episode_length_s == 5.


def test_reference_mode_can_round_trip_without_changing_native_terms():
    cfg = config()
    assert configure_termination(cfg, 21, 'reference') == 21
    term = cfg.terminations['reference_exhausted']
    assert term.func is reference_exhausted and term.time_out is True
    assert configure_termination(cfg, 21, 'contact') == 250
    assert 'reference_exhausted' not in cfg.terminations


def clock(started, onset, time):
    return SimpleNamespace(
        physics_dt=.00025, cfg=SimpleNamespace(episode_length_s=5.),
        _hammer_first_strike=SimpleNamespace(
            started=torch.tensor(started), first_contact_time_s=torch.tensor(onset),
            _substep_count=torch.tensor([round(t/.00025) for t in time])),
    )


def test_existing_timer_ends_at_first_policy_boundary_after_contact_plus_200ms():
    env = clock([True, True, True, False], [.413, .413, .4, 0.], [.6, .62, .6, 5.])
    torch.testing.assert_close(horizon_window_complete(env, .2), torch.tensor([False, True, True, False]))


def test_miss_ends_at_existing_timeout_instead_of_waiting_for_contact():
    cfg = config()
    configure_termination(cfg, 21, 'contact')
    env = clock([False, False], [0., 0.], [4.98, 5.])
    env.episode_length_buf = torch.tensor([249, 250])
    env.max_episode_length = 250
    torch.testing.assert_close(cfg.terminations['time_out'].func(env), torch.tensor([False, True]))
    assert not horizon_window_complete(env, .2).any()


def test_reference_cutoff_still_comes_from_selected_recording():
    cfg = config()
    assert configure_termination(cfg, 19, 'reference') == 19
    assert cfg.terminations['reference_exhausted'].params == {'clip_steps': 19}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='termination'):
        configure_termination(config(), 21, 'unknown')
