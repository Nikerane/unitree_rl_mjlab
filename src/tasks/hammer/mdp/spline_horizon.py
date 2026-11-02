"""Opt-in horizon boundaries in integer physics ticks; legacy timers stay frozen."""

import torch


def horizon_window_complete(env, window_s=.05):
    """First policy boundary with the admitted number of elapsed physics ticks.

    FirstStrikeEventTracker keeps the current integer tick but stores onset as
    float32 tick*dt. For dt in {.002, .00025} and onset within 20 seconds,
    timestamp rounding is far below half a tick (even 20s float32 spacing is
    under 2 microseconds, versus the smallest half tick of 125 microseconds).
    Dividing in float64 and rounding therefore recovers the exact onset tick.
    This relies on the existing tracker timestamp construction, not wall time.
    """
    dt = env.physics_dt
    if dt not in (.002, .00025):
        raise ValueError("horizon integer timer requires .002 or .00025 physics dt")
    if window_s not in (.05, .2):
        raise ValueError("horizon integer timer requires .05 or .2 second window")
    if not 0. < env.cfg.episode_length_s <= 20.:
        raise ValueError("horizon integer timer requires episode duration in (0, 20] seconds")
    tracker = env._hammer_first_strike
    onset_tick = torch.round(tracker.first_contact_time_s.to(torch.float64) / dt).to(torch.int64)
    elapsed_ticks = tracker._substep_count - onset_tick
    return tracker.started & (elapsed_ticks >= round(window_s / dt))
