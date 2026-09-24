import importlib.util
from pathlib import Path
import numpy as np
import pytest


def test_first_episode_peak_includes_terminal_step_excludes_resets():
    path = Path(__file__).with_name("evaluate.py")
    assert path.exists(), "first-episode evaluator missing"
    spec = importlib.util.spec_from_file_location("pilot_evaluate",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    peaks = np.array([[[1,2],[2,1]],[[3,4],[4,3]],[[99,99],[6,5]]])
    done = np.array([[False,False],[True,False],[False,True]])
    np.testing.assert_array_equal(module.first_episode_peaks(peaks,done),[[3,4],[6,5]])


def test_smoke_step_cap_does_not_shorten_default_evaluation():
    path = Path(__file__).with_name("evaluate.py")
    spec = importlib.util.spec_from_file_location("pilot_evaluate",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module,"control_steps"), "explicit smoke horizon missing"
    assert module.control_steps(1000,None) == 1001
    assert module.control_steps(1000,4) == 4
    assert module.control_steps(1000,2000) == 1001
    with pytest.raises(ValueError):
        module.control_steps(1000,0)
