"""Safety-net correctness (2026-07-14, zero-Λ incident): the physical-impossibility
invariants and live sentinels must fire on dead instrumentation and stay silent on
real physics — a false "live" defeats the entire net. Pure torch / stub env, no MuJoCo.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.tasks.hammer.mdp.impulse_bound import (
    _ENV_SUBSTEP_IMPULSE_ATTR,
    contact_seen,
    impossible_success,
)
from scripts.eval_impulse import _invariant_violations


def _rec(succ, delivered, lam_worst):
    """Build a rollout record; lam is a list of (6,) tensors with the given worst-joint value."""
    return {
        "succ": succ,
        "delivered": delivered,
        "lam": [torch.tensor([0.0, lw, 0.0, 0.0, 0.0, 0.0]) for lw in lam_worst],
    }


def test_invariants_silent_on_healthy_rollout():
    # Every success carries positive impulse on both paths -> zero violations.
    rec = _rec(succ=[True, True, False], delivered=[0.6, 0.5, 0.0], lam_worst=[0.3, 0.25, 0.0])
    assert _invariant_violations(rec) == (0, 0)


def test_impossible_success_fires_on_dead_instrument():
    # The exact incident: success with zero recorded impulse on both paths.
    rec = _rec(succ=[True, True], delivered=[0.0, 0.0], lam_worst=[0.0, 0.0])
    impossible, lam_dead = _invariant_violations(rec)
    assert impossible == 2


def test_lambda_dead_fires_on_cross_path_disagreement():
    # Object-side delivered > 0 but robot-side Λ == 0 (impossible direction), even w/o success.
    rec = _rec(succ=[False, False], delivered=[0.4, 0.4], lam_worst=[0.0, 0.0])
    impossible, lam_dead = _invariant_violations(rec)
    assert impossible == 0 and lam_dead == 2


def test_lateral_graze_not_flagged():
    # Λ > 0 with delivered == 0 is a legitimate lateral graze (no axial delivery) -> not counted.
    rec = _rec(succ=[False], delivered=[0.0], lam_worst=[0.2])
    assert _invariant_violations(rec) == (0, 0)


def _stub_env(*, contact_time, lam_worst, terminated, depth=0.0):
    sensor = SimpleNamespace(data=SimpleNamespace(
        current_contact_time=torch.tensor([[contact_time]]),
        last_contact_time=torch.zeros(1, 1),
    ))
    # nail_block entity: impossible_success reads its joint_pos depth (NoTerm depth-keyed alarm path).
    nail = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[depth]])))
    acc = SimpleNamespace(_episode_peak_perjoint=torch.tensor([[0.0, lam_worst, 0.0, 0.0, 0.0, 0.0]]))
    env = SimpleNamespace(
        num_envs=1, device="cpu",
        scene={"hammer_nail_contact": sensor, "nail_block": nail},
        reset_terminated=torch.tensor([terminated]),
    )
    setattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, acc)
    return env


def test_contact_seen_tracks_contact():
    assert float(contact_seen(_stub_env(contact_time=0.004, lam_worst=0.3, terminated=True),
                              "hammer_nail_contact")) == 1.0
    assert float(contact_seen(_stub_env(contact_time=0.0, lam_worst=0.0, terminated=False),
                              "hammer_nail_contact")) == 0.0


def test_impossible_success_metric():
    # Terminated on success with Λ == 0 -> alarm fires.
    assert float(impossible_success(_stub_env(contact_time=0.0, lam_worst=0.0, terminated=True))) == 1.0
    # Success with real Λ -> silent.
    assert float(impossible_success(_stub_env(contact_time=0.004, lam_worst=0.3, terminated=True))) == 0.0
    # Timeout (not terminated) with Λ == 0 -> silent (undertrained, not dead).
    assert float(impossible_success(_stub_env(contact_time=0.0, lam_worst=0.0, terminated=False))) == 0.0
    # NoTerm arm: depth-success (>=0.030) but NOT terminated, Λ == 0 -> alarm STILL fires (dead qfrc).
    assert float(impossible_success(_stub_env(contact_time=0.0, lam_worst=0.0, terminated=False, depth=0.031))) == 1.0
    # NoTerm depth-success with real Λ -> silent.
    assert float(impossible_success(_stub_env(contact_time=0.004, lam_worst=0.3, terminated=False, depth=0.031))) == 0.0
    # NoTerm sub-threshold depth (parked <0.030) with Λ == 0 -> silent (not a success, undertrained).
    assert float(impossible_success(_stub_env(contact_time=0.0, lam_worst=0.0, terminated=False, depth=0.028))) == 0.0
