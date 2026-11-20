"""Recorded-time phase follows the released DeepMimic clock convention."""

import pytest
import torch

from scripts.z1_u22_timed_pair import recorded_time_phase


def test_phase_wraps_and_accepts_negative_offsets():
    elapsed = torch.tensor([0., .25, .5, .75, 1., 1.25, -.25])
    expected = torch.tensor([0., .25, .5, .75, 0., .25, .75])
    torch.testing.assert_close(recorded_time_phase(elapsed, 1., 1.), expected)


@pytest.mark.parametrize("duration, cutoff", [(.38, 19), (.41, 21)])
def test_phase_uses_selected_recording_duration(duration, cutoff):
    steps = torch.arange(cutoff + 1)
    phase = recorded_time_phase(steps, .02, duration)
    expected = torch.remainder(steps * (.02 / duration), 1.)
    torch.testing.assert_close(phase, expected)
    assert phase[0] == 0.
    assert ((phase >= 0.) & (phase < 1.)).all()


@pytest.mark.parametrize("duration", [0., -1.])
def test_phase_requires_a_positive_recording_duration(duration):
    with pytest.raises(ValueError, match="positive"):
        recorded_time_phase(torch.tensor([0]), .02, duration)
