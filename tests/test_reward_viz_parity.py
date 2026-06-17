"""Drift guard for the Reward Explorer viz (viz/reward_explorer/reward_formulas.py).

Two checks, both fast (no warp env):
  1. The viz's default params/weights match the LIVE env config (read from z1_hammer_env_cfg).
  2. The numpy formulas compute the same math as src/tasks/hammer/mdp/rewards.py (reference points).

If someone retunes a reward in the code, (1) fails until the viz is updated.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "viz" / "reward_explorer"))
import reward_formulas as rf  # noqa: E402

from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg  # noqa: E402
from src.tasks.hammer.mdp.rewards import NailDepthDeltaTerm  # noqa: E402


@pytest.fixture(scope="module")
def rewards():
    # Cheap: builds the cfg dataclasses only (no env / no warp).
    return z1_hammer_env_cfg(imitation=True).rewards


class TestVizParamsMatchEnv:
    def test_all_weights_match(self, rewards):
        for name, w in rf.WEIGHTS.items():
            assert rewards[name].weight == pytest.approx(w), f"weight drift: {name}"

    def test_nail_driven_params(self, rewards):
        p = rewards["nail_driven"].params
        assert p["goal_depth"] == pytest.approx(rf.GOAL_DEPTH)
        assert p["std"] == pytest.approx(rf.NAIL_DRIVEN_STD)

    def test_approach_std(self, rewards):
        assert rewards["approach"].params["std"] == pytest.approx(rf.APPROACH_STD)

    def test_completion_threshold(self, rewards):
        assert rewards["completion"].params["success_depth"] == pytest.approx(rf.SUCCESS_THRESHOLD)

    def test_r_imit_sigma(self, rewards):
        assert rewards["r_imit"].params["sigma"] == pytest.approx(rf.R_IMIT_SIGMA)

    def test_impact_v_expected(self, rewards):
        assert rewards["impact_progress"].params["v_expected"] == pytest.approx(rf.IMPACT_V_EXPECTED)

    def test_settle_offset(self):
        assert NailDepthDeltaTerm._SETTLE_OFFSET == pytest.approx(rf.SETTLE_OFFSET)


class TestFormulaCorrectness:
    def test_nail_driven_peaks_at_goal(self):
        assert rf.nail_driven(rf.GOAL_DEPTH) == pytest.approx(1.0)
        assert rf.nail_driven(rf.GOAL_DEPTH - rf.NAIL_DRIVEN_STD) == pytest.approx(math.exp(-1))

    def test_gaussian_distance(self):
        assert rf.gaussian_distance(0.0, 0.05) == pytest.approx(1.0)
        assert rf.gaussian_distance(0.05, 0.05) == pytest.approx(math.exp(-1))

    def test_completion_step(self):
        assert float(rf.completion(rf.SUCCESS_THRESHOLD - 1e-4)) == 0.0
        assert float(rf.completion(rf.SUCCESS_THRESHOLD + 1e-4)) == 1.0

    def test_r_imit_gated_off_after_contact(self):
        assert rf.r_imit(0.0, pre_contact=True) == pytest.approx(1.0)
        assert float(rf.r_imit(0.0, pre_contact=False)) == 0.0

    def test_impact_progress_double_gate(self):
        assert rf.impact_progress(0.5, 1.0, 1.0) == pytest.approx(0.5)
        assert float(rf.impact_progress(0.5, 0.0, 1.0)) == 0.0  # no first contact
        assert float(rf.impact_progress(0.5, 1.0, 0.0)) == 0.0  # no depth advance
        assert float(rf.impact_progress(-0.5, 1.0, 1.0)) == 0.0  # upward clamped

    def test_nail_depth_delta_ratchet(self):
        # settle=0.004: step1 0.010-0.004=0.006; hold->0; 0.020-0.010=0.010; bounce->0.
        out = rf.nail_depth_delta([0.0, 0.010, 0.010, 0.020, 0.015], settle=0.004)
        assert list(out) == pytest.approx([0.0, 0.006, 0.0, 0.010, 0.0])
