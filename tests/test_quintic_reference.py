"""Accepted quintic geometry, spatial progress and shared rendering contract."""
import torch

from src.tasks.hammer.mdp.references import SingleStrikeReference


def test_quintic_pilot_config_shares_reference_contract():
    import src.tasks
    from mjlab.tasks.registry import load_env_cfg
    cfg = load_env_cfg('Unitree-Z1-Hammer-Quintic-M20-Pilot', play=True)
    consumers = [cfg.events['sample_strike_route_signs'].params, cfg.rewards['r_imit'].params]
    for group in cfg.observations.values():
        consumers.extend(group.terms[n].params for n in ('strike_phase', 'strike_ref_error'))
    for params in consumers:
        assert params['path_shape'] == 'quintic'
        assert params['overshoot'] == .02
        assert params['followthrough_mode'] == 'vertical'
    assert cfg.rewards['r_imit'].weight == .2
    assert cfg.metrics['cat_soft'].params['imp_max_p'] == .2


def test_quintic_passes_nail_before_vertical_followthrough():
    ref = SingleStrikeReference(1, 'cpu', overshoot=.02, path_shape='quintic')
    start = torch.tensor([[.48, 0., .25]])
    nail = torch.tensor([[.50, 0., .10]])
    ref.update(start, nail, torch.tensor([0]))
    # At one quarter of the 150 mm descent, h(1/4)=0.103515625.
    phi = torch.tensor([.0375/.17])
    expected = torch.tensor([[.4820703125, 0., .2125]])
    torch.testing.assert_close(ref.waypoint(phi), expected)
    torch.testing.assert_close(ref.waypoint(torch.tensor([.15/.17])), nail)
    torch.testing.assert_close(ref.waypoint(torch.ones(1)), torch.tensor([[.5, 0., .08]]))
    # A true on-curve point must recover its phase, not straight-chord projection.
    torch.testing.assert_close(ref.preview(expected, torch.tensor([1])), phi)
    torch.testing.assert_close(ref.peek(), torch.zeros(1))


def test_quintic_overlay_and_subset_reset():
    ref = SingleStrikeReference(2, 'cpu', overshoot=.02, path_shape='quintic')
    head = torch.tensor([[.48, 0., .25], [.52, 0., .25]])
    nail = torch.tensor([[.5, 0., .1], [.5, 0., .1]])
    ref.update(head, nail, torch.zeros(2, dtype=torch.long))
    points = ref.reference_polyline(0, 17)
    for i, phi in enumerate(torch.linspace(0, 1, 17)):
        torch.testing.assert_close(points[i], ref.waypoint(phi.repeat(2))[0])
    middle = ref.waypoint(torch.full((2,), .5))
    ref.update(middle, nail, torch.ones(2, dtype=torch.long))
    ref.reset(torch.tensor([0]))
    phase = ref.update(head, nail, torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(phase, torch.tensor([0., .5]))


def test_video_overlay_preserves_quintic_bend():
    from scripts.render_policy import _reference_polyline_m
    ref = SingleStrikeReference(1, 'cpu', overshoot=.02, path_shape='quintic')
    ref.update(torch.tensor([[.48, 0., .25]]), torch.tensor([[.5, 0., .1]]), torch.tensor([0]))
    line = _reference_polyline_m(ref, 'cpu')
    assert len(line) > 2
    assert abs(float(line[16, 0]) - .482870) < .0001
