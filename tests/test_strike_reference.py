"""Unit tests for the single-strike reference (plan stage T1).

Pure torch — no MuJoCo/Warp. Covers anchoring, the two-segment phase law
(time-indexed wind-up, projection-indexed descent), the monotone latch,
re-anchoring on reset, playback continuity, batching, and the two observation
terms against a stub env (same pattern as test_impact_progress_reward.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.references import SingleStrikeReference, get_strike_reference


B = 3
HEAD0 = torch.tensor([[0.50, 0.00, 0.12]]).repeat(B, 1)
NAIL = torch.tensor([[0.50, 0.00, 0.102]]).repeat(B, 1)


def _ref(**kw) -> SingleStrikeReference:
    return SingleStrikeReference(B, "cpu", **kw)


def _steps(v: int) -> torch.Tensor:
    return torch.full((B,), v, dtype=torch.long)


# --- anchoring -------------------------------------------------------------


def test_anchor_on_step_zero_phi_is_zero():
    ref = _ref()
    phi = ref.update(HEAD0, NAIL, _steps(0))
    assert torch.allclose(phi, torch.zeros(B))


def test_waypoint_endpoints():
    ref = _ref(approach_height=0.10, overshoot=0.005)
    ref.update(HEAD0, NAIL, _steps(0))
    assert torch.allclose(ref.waypoint(torch.zeros(B)), HEAD0, atol=1e-6)
    apex = NAIL.clone()
    apex[:, 2] += 0.10
    assert torch.allclose(ref.waypoint(torch.full((B,), 0.5)), apex, atol=1e-6)
    target = NAIL.clone()
    target[:, 2] -= 0.005
    assert torch.allclose(ref.waypoint(torch.ones(B)), target, atol=1e-6)


# --- wind-up phase (time-indexed) -------------------------------------------


def test_windup_phase_grows_with_step_count():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    phi_mid = ref.update(HEAD0, NAIL, _steps(max(n_w // 2, 1)))
    assert 0.0 < phi_mid[0] < 0.5
    # Wind-up phase never exceeds 0.5 even for huge step counts pre-descent...
    # (step >= n_windup switches to the descent law, tested below).


def test_descent_phase_is_projection_indexed():
    ref = _ref(approach_height=0.10, overshoot=0.005)
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    apex = NAIL.clone()
    apex[:, 2] += 0.10
    # Head at the apex when descent begins -> phi == 0.5.
    phi = ref.update(apex, NAIL, _steps(n_w))
    assert torch.allclose(phi, torch.full((B,), 0.5), atol=1e-6)
    # Head halfway down the strike axis -> phi == 0.75.
    target = NAIL.clone()
    target[:, 2] -= 0.005
    halfway = (apex + target) / 2
    phi = ref.update(halfway, NAIL, _steps(n_w + 1))
    assert torch.allclose(phi, torch.full((B,), 0.75), atol=1e-3)
    # Head at the strike target -> phi == 1.
    phi = ref.update(target, NAIL, _steps(n_w + 2))
    assert torch.allclose(phi, torch.ones(B), atol=1e-3)


def test_descent_phase_clamped_for_overshoot_below_target():
    """Head below the strike target must clamp to phi = 1, not exceed it.

    Probe point is anchored to the reference's OWN target so the test stays a
    clamp-law test under any overshoot default (the 2026-07-13 follow-through fix
    moved the default from 0.035 to 0.15). It sits WITHIN axis_tol below the
    target: past the clamped s=1, the excess below-target distance projects into
    the perp term, so a probe deeper than axis_tol reads off-axis by design and
    would test the axis gate, not the clamp."""
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    below = NAIL.clone()
    below[:, 2] -= ref.overshoot + 0.8 * ref.axis_tol
    phi = ref.update(below, NAIL, _steps(n_w + 1))
    assert torch.allclose(phi, torch.ones(B), atol=1e-6)


# --- monotone latch ----------------------------------------------------------


def test_phase_latch_survives_bounce():
    """A post-impact bounce (head moving back up) must not rewind the phase."""
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    target = NAIL.clone()
    target[:, 2] -= 0.005
    phi_deep = ref.update(target, NAIL, _steps(n_w + 3))
    bounced = NAIL.clone()
    bounced[:, 2] += 0.06  # head thrown back up above the nail
    phi_after = ref.update(bounced, NAIL, _steps(n_w + 4))
    assert torch.all(phi_after >= phi_deep - 1e-6)


def test_lateral_motion_does_not_advance_descent_phase():
    """A head far off the strike axis must NOT read as strike progress
    (review finding: altitude-only phase aliased lateral wandering as phi≈0.9
    and the latch made it irreversible)."""
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    off_axis = NAIL.clone()
    off_axis[:, 0] += 0.14  # 14 cm lateral of the nail, at nail height
    phi = ref.update(off_axis, NAIL, _steps(n_w + 1))
    assert torch.all(phi <= 0.5 + 1e-6), f"off-axis head advanced phase: {phi}"
    # Returning to the axis resumes normal descent indexing.
    apex = NAIL.clone()
    apex[:, 2] += ref.approach_height
    target = NAIL.clone()
    target[:, 2] -= ref.overshoot
    halfway = (apex + target) / 2
    phi = ref.update(halfway, NAIL, _steps(n_w + 2))
    assert torch.allclose(phi, torch.full((B,), 0.75), atol=1e-3)


def test_update_is_idempotent_within_a_step():
    ref = _ref()
    phi1 = ref.update(HEAD0, NAIL, _steps(0))
    phi2 = ref.update(HEAD0, NAIL, _steps(0))
    assert torch.allclose(phi1, phi2)


# --- re-anchoring -------------------------------------------------------------


def test_reanchor_on_episode_reset_follows_new_nail():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    target = NAIL.clone()
    target[:, 2] -= 0.005
    ref.update(target, NAIL, _steps(n_w + 2))  # drive phase to ~1
    new_nail = NAIL.clone()
    new_nail[:, 0] += 0.03  # nail moved 3 cm in x (reset randomization)
    phi = ref.update(HEAD0, new_nail, _steps(0))  # fresh episode
    assert torch.allclose(phi, torch.zeros(B))
    apex = new_nail.clone()
    apex[:, 2] += ref.approach_height
    assert torch.allclose(ref.waypoint(torch.full((B,), 0.5)), apex, atol=1e-6)


def test_explicit_reset_clears_latch():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    target = NAIL.clone()
    target[:, 2] -= 0.005
    ref.update(target, NAIL, _steps(n_w + 2))
    ref.reset()
    phi = ref.update(HEAD0, NAIL, _steps(5))  # mid-episode step but unanchored
    assert torch.all(phi < 0.5)  # re-anchored fresh, wind-up law applies


def test_per_env_independence():
    """Different nail positions per env produce different apexes."""
    nails = NAIL.clone()
    nails[1, 0] += 0.05
    ref = _ref()
    ref.update(HEAD0, nails, _steps(0))
    wp = ref.waypoint(torch.full((B,), 0.5))
    assert not torch.allclose(wp[0], wp[1])
    assert torch.allclose(wp[0], wp[2])


# --- playback ------------------------------------------------------------------


def test_playback_continuity_and_endpoint():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    n = ref.playback_length()
    assert n >= 2
    max_step = max(ref.windup_speed, ref.descent_speed) + 1e-6
    prev = ref.playback_target(0)
    for k in range(1, n + 1):
        cur = ref.playback_target(k)
        assert float((cur - prev).norm(dim=-1).max()) <= max_step
        prev = cur
    target = NAIL.clone()
    target[:, 2] -= ref.overshoot
    assert torch.allclose(ref.playback_target(n), target, atol=1e-6)


def test_playback_passes_through_apex():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    apex = NAIL.clone()
    apex[:, 2] += ref.approach_height
    assert torch.allclose(ref.playback_target(n_w), apex, atol=1e-6)


# --- shapes ----------------------------------------------------------------------


def test_shapes():
    ref = _ref()
    phi = ref.update(HEAD0, NAIL, _steps(0))
    assert tuple(phi.shape) == (B,)
    assert tuple(ref.waypoint(phi).shape) == (B, 3)
    assert tuple(ref.playback_target(1).shape) == (B, 3)


# --- observation terms against a stub env ------------------------------------------


_ROBOT_CFG = SimpleNamespace(name="robot", site_ids=[0])
_NAIL_CFG = SimpleNamespace(name="nail_block", site_ids=[0])


def _stub_env(num_envs: int = B):
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=HEAD0.clone().unsqueeze(1)))
    nail = SimpleNamespace(data=SimpleNamespace(site_pos_w=NAIL.clone().unsqueeze(1)))
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"robot": robot, "nail_block": nail},
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )


def test_strike_phase_obs_shape_and_anchor():
    from src.tasks.hammer.mdp.observations import strike_phase

    env = _stub_env()
    phi = strike_phase(env, robot_cfg=_ROBOT_CFG, nail_cfg=_NAIL_CFG)
    assert tuple(phi.shape) == (B, 1)
    assert torch.allclose(phi, torch.zeros(B, 1))


def test_strike_ref_error_points_to_apex_during_windup():
    from src.tasks.hammer.mdp.observations import strike_ref_error

    env = _stub_env()
    err = strike_ref_error(env, robot_cfg=_ROBOT_CFG, nail_cfg=_NAIL_CFG)
    assert tuple(err.shape) == (B, 3)
    # At phi = 0 the waypoint is head0 itself -> error ~ 0.
    assert float(err.norm(dim=-1).max()) < 1e-5
    # Advance one wind-up step without moving the head: waypoint moves toward
    # the apex (upward), so the error gains a positive z component.
    env.episode_length_buf += 1
    err = strike_ref_error(env, robot_cfg=_ROBOT_CFG, nail_cfg=_NAIL_CFG)
    assert float(err[:, 2].min()) > 0.0


def test_get_strike_reference_is_cached_per_env():
    env = _stub_env()
    r1 = get_strike_reference(env)
    r2 = get_strike_reference(env)
    assert r1 is r2


# --- 2026-07-13 adversarial-review fixes (F1 peek purity, F3 apex clearance) ---


def test_peek_is_pure_and_matches_latch():
    """peek() returns the latched phase without anchoring or advancing (F1)."""
    ref = _ref()
    # Before any update: unanchored, peek returns zeros and does NOT anchor.
    assert torch.allclose(ref.peek(), torch.zeros(B))
    assert not bool(ref._anchored.any())
    # After anchoring + descent progress, peek == the latch, and repeated
    # peeks with no update in between never move it.
    ref.update(HEAD0, NAIL, _steps(0))
    n_w = int(ref._n_windup[0].item())
    deep = NAIL.clone()
    deep[:, 2] -= 0.005
    phi = ref.update(deep, NAIL, _steps(n_w + 2))
    for _ in range(3):
        assert torch.equal(ref.peek(), phi)
    # Defensive copy: mutating the peeked tensor must not leak into the latch.
    p = ref.peek()
    p += 1.0
    assert torch.equal(ref.peek(), phi)


def test_apex_clearance_floors_windup_at_near_apex_reset():
    """F3 fix: when the head anchors AT/ABOVE nail_top+approach_height (the L6
    near-nail reset), the apex is floored at head0+min_windup_clearance so the
    wind-up cannot degenerate to a 1-step nudge."""
    # L6-like: head 2 mm below the nominal apex (0.252) — the degenerate case.
    head_l6 = NAIL.clone()
    head_l6[:, 2] = 0.102 + 0.148  # z = 0.250 vs nail_top+0.15 = 0.252
    ref = _ref()
    ref.update(head_l6, NAIL, _steps(0))
    apex_z = ref._apex[:, 2]
    assert torch.allclose(apex_z, head_l6[:, 2] + ref.min_windup_clearance)
    assert int(ref._n_windup[0].item()) >= 2  # real lift, not a 1-step nudge
    # waypoint(0.5) is the floored apex.
    wp_apex = ref.waypoint(torch.full((B,), 0.5))
    assert torch.allclose(wp_apex[:, 2], apex_z, atol=1e-6)
    # Heads well BELOW the nominal apex keep the classic anchor untouched.
    ref2 = _ref()
    ref2.update(HEAD0, NAIL, _steps(0))  # head z=0.12 << 0.252
    assert torch.allclose(ref2._apex[:, 2], NAIL[:, 2] + ref2.approach_height)
