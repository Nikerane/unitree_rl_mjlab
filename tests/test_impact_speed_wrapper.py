"""The experimental wrapper rejects raw nonfinite actions before clipping."""

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.rl.impact_speed_wrapper import (
    ImpactSpeedVecEnvWrapper,
    configure_impact_speed_experiment,
    require_impact_speed_experiment_cfg,
    wrap_impact_speed_env,
)


class _FakeEnv:
    def __init__(self, terms=("joint_position", "joint_stiffness", "strike_speed")):
        self.unwrapped = self
        self.action_manager = SimpleNamespace(
            active_terms=list(terms),
            total_action_dim=13 if len(terms) == 3 else 12,
        )
        self.num_envs = 1
        self.device = "cpu"
        self.max_episode_length = 100
        self.cfg = SimpleNamespace(is_finite_horizon=True)
        self.render_mode = None
        self.step_calls = 0
        self.last_action = None

    def reset(self):
        return {"actor": torch.zeros(1, 42)}, {}

    def step(self, actions):
        self.step_calls += 1
        self.last_action = actions.clone()
        return (
            {"actor": torch.zeros(1, 42)},
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )


@pytest.fixture
def guarded_wrapper():
    return wrap_impact_speed_env(_FakeEnv(), clip_actions=1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_raw_nonfinite_never_reaches_environment(bad, guarded_wrapper):
    action = torch.zeros(1, 13)
    action[0, 12] = bad
    with pytest.raises(ValueError, match="finite"):
        guarded_wrapper.step(action)
    assert guarded_wrapper.env.step_calls == 0


def test_raw_nonfinite_first_twelve_also_rejected(guarded_wrapper):
    action = torch.zeros(1, 13)
    action[0, 3] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        guarded_wrapper.step(action)
    assert guarded_wrapper.env.step_calls == 0


def test_wrapper_clips_copy_without_mutating_raw_ppo_sample(guarded_wrapper):
    raw = torch.tensor([[2.0] * 12 + [-2.0]])
    before = raw.clone()
    guarded_wrapper.step(raw)
    torch.testing.assert_close(raw, before)
    torch.testing.assert_close(guarded_wrapper.env.last_action, raw.clamp(-1, 1))


def test_first_decision_executes_all_thirteen_actions_without_hold(guarded_wrapper):
    raw = torch.arange(13, dtype=torch.float32).unsqueeze(0) / 13
    guarded_wrapper.step(raw)
    assert guarded_wrapper.env.step_calls == 1
    torch.testing.assert_close(guarded_wrapper.env.last_action, raw)


def test_only_exact_experimental_signature_gets_guard():
    from mjlab.rl import RslRlVecEnvWrapper

    assert isinstance(wrap_impact_speed_env(_FakeEnv(), 1.0), ImpactSpeedVecEnvWrapper)
    legacy = wrap_impact_speed_env(_FakeEnv(("joint_position", "joint_stiffness")), 1.0)
    assert type(legacy) is RslRlVecEnvWrapper


def test_experimental_configuration_needs_reviewed_bounds_and_negative_cost():
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg

    cfg = _impact_speed_env_cfg(play=False)
    with pytest.raises(ValueError, match="bounds"):
        require_impact_speed_experiment_cfg(cfg)
    cfg.actions["strike_speed"].bounds = (0.2, 0.8)
    with pytest.raises(ValueError, match="negative"):
        require_impact_speed_experiment_cfg(cfg)
    cfg.rewards["speed_goal"].weight = -0.4
    require_impact_speed_experiment_cfg(cfg)


def test_cli_calibration_is_explicit_and_controls_mode():
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg

    cfg = _impact_speed_env_cfg(play=False)
    with pytest.raises(ValueError, match="bounds"):
        configure_impact_speed_experiment(cfg, bounds=None, cost=None, mode="external")
    with pytest.raises(ValueError, match="negative"):
        configure_impact_speed_experiment(cfg, bounds=(0.2, 0.8), cost=None, mode="external")
    configure_impact_speed_experiment(cfg, bounds=(0.2, 0.8), cost=0.4, mode="external")
    assert cfg.actions["strike_speed"].bounds == (0.2, 0.8)
    assert cfg.actions["strike_speed"].mode == "external"
    assert cfg.rewards["speed_goal"].weight == -0.4


def test_nested_action_mode_survives_unspecified_cli_mode():
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg

    cfg = _impact_speed_env_cfg(play=False)
    cfg.actions["strike_speed"].mode = "external"
    configure_impact_speed_experiment(cfg, bounds=(0.2, 0.8), cost=0.4)
    assert cfg.actions["strike_speed"].mode == "external"


def test_malformed_experimental_signature_cannot_bypass_calibration_guard():
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg

    cfg = _impact_speed_env_cfg(play=False)
    cfg.actions["extra"] = cfg.actions["strike_speed"]
    with pytest.raises(ValueError, match="signature"):
        configure_impact_speed_experiment(cfg, bounds=None, cost=None)
