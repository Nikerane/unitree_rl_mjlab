"""Evaluation-only terminal-speed interventions for paired policy responses."""

from __future__ import annotations

import math

import torch


def override_latched_goal(env, goal: float):
    state = env.action_manager.get_term("strike_speed").state
    if bool(env._hammer_first_strike.started.any()):
        raise ValueError("cannot intervene after contact has started")
    if bool(state.pending.any()):
        raise ValueError("intervention requires an already latched goal")
    if not math.isfinite(goal) or not state.bounds[0] <= goal <= state.bounds[1]:
        raise ValueError("goal outside speed bounds")
    state.goal_mps.fill_(goal)
    # Recompute public groups without advancing observation history or policy time.
    return {group: env.observation_manager.compute_group(group, update_history=False)
            for group in ("actor", "critic")}


def set_goal_before_first_action(env, goal: float, observations):
    """Set a paired target while preserving the exact noisy reset observation."""
    state = env.action_manager.get_term("strike_speed").state
    if bool(env._hammer_first_strike.started.any()):
        raise ValueError("cannot set a pre-action goal after contact has started")
    if not bool(state.pending.bool().all()):
        raise ValueError("pre-action goal requires an unconsumed policy selector")
    if not math.isfinite(goal) or not state.bounds[0] <= goal <= state.bounds[1]:
        raise ValueError("goal outside speed bounds")
    lo, hi = state.bounds
    normalized_goal = 2 * (goal - lo) / (hi - lo) - 1
    tail = torch.stack((torch.full_like(state.goal_mps, normalized_goal),
                        torch.zeros_like(state.pending)), dim=-1)
    updated = {}
    for group in ("actor", "critic"):
        value = observations.get(group)
        if not isinstance(value, torch.Tensor) or value.shape != (env.num_envs, 42):
            raise ValueError("pre-action goal requires finite 42D actor/critic reset observations")
        if not torch.isfinite(value).all():
            raise ValueError("pre-action goal requires finite 42D actor/critic reset observations")
        patched = value.clone()
        patched[:, 40:42] = tail.to(device=value.device, dtype=value.dtype)
        updated[group] = patched
    state.goal_mps.fill_(goal)
    state.pending.zero_()
    return updated
