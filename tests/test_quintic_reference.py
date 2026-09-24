"""Accepted quintic geometry, spatial progress and shared rendering contract."""
import pytest
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


def test_current_height_diagnostic_is_sample_rate_independent_and_pure():
    ref = SingleStrikeReference(1, "cpu", overshoot=.02, path_shape="quintic")
    start = torch.tensor([[.48, 0., .25]])
    nail = torch.tensor([[.50, 0., .10]])
    ref.update(start, nail, torch.tensor([0]))
    for count in (11, 51):
        for point in ref.reference_polyline(num_points=count):
            phase, waypoint, error_m = ref.diagnose_current_height(point.unsqueeze(0))
            torch.testing.assert_close(waypoint, point.unsqueeze(0), atol=1e-6, rtol=0)
            torch.testing.assert_close(error_m, torch.zeros(1), atol=1e-6, rtol=0)
            assert 0 <= phase.item() <= 1
    assert ref.peek().item() == 0


def test_current_height_diagnostic_reports_off_curve_error_and_rebound():
    ref = SingleStrikeReference(1, "cpu", overshoot=.02, path_shape="quintic")
    start = torch.tensor([[.48, 0., .25]])
    nail = torch.tensor([[.50, 0., .10]])
    ref.update(start, nail, torch.tensor([0]))
    low = ref.waypoint(torch.tensor([.75]))
    high = ref.waypoint(torch.tensor([.25]))
    ref.update(low, nail, torch.tensor([1]))
    low_phase, _, _ = ref.diagnose_current_height(low)
    high_phase, _, _ = ref.diagnose_current_height(high)
    assert low_phase.item() == pytest.approx(.75)
    assert high_phase.item() == pytest.approx(.25)
    off_curve = high + torch.tensor([[.0, .03, .0]])
    phase, waypoint, error_m = ref.diagnose_current_height(off_curve)
    assert phase.item() == pytest.approx(.25)
    torch.testing.assert_close(waypoint, high)
    assert error_m.item() == pytest.approx(.03)
    assert ref.peek().item() == pytest.approx(.75)


def test_paired_same_head_and_nail_expose_start_geometry_in_reference_error():
    ref = SingleStrikeReference(3, "cpu", overshoot=.02, path_shape="quintic")
    starts = torch.tensor([[.48, 0., .25], [.50, 0., .25], [.52, 0., .25]])
    nail = torch.tensor([[.50, 0., .10]]).expand(3, -1)
    ref.update(starts, nail, torch.zeros(3, dtype=torch.long))
    paired_head = torch.tensor([[.50, 0., .20]]).expand(3, -1)
    phase = ref.update(paired_head, nail, torch.ones(3, dtype=torch.long))
    error = ref.waypoint(phase) - paired_head
    torch.testing.assert_close(phase, phase[:1].expand_as(phase))
    assert error[0, 0] < error[1, 0] < error[2, 0]
    assert ref.peek().shape == (3,)
