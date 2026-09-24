"""Qualification transfer is an explicit, guarded training entry point."""

from dataclasses import replace

import pytest

import scripts.train as train
import scripts.play as play
from scripts.train import TrainConfig
from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_controller
from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
from src.tasks.hammer.config.z1.rl_cfg import z1_impact_speed_ppo_runner_cfg


TASK_ID = "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed"


def _cfg(checkpoint):
    env = _impact_speed_env_cfg(play=True)
    env.actions["strike_speed"].bounds = (0.2, 0.8)
    env.rewards["speed_goal"].weight = -0.4
    return TrainConfig(
        env=env, agent=z1_impact_speed_ppo_runner_cfg(),
        qualification_checkpoint=str(checkpoint),
    )


def test_qualification_checkpoint_route_accepts_only_fresh_autonomous_experiment(tmp_path):
    checkpoint = tmp_path / "qualified.pt"
    checkpoint.touch()
    cfg = _cfg(checkpoint)
    assert train.qualification_checkpoint_path(TASK_ID, cfg) == checkpoint.resolve()
    assert train.qualification_checkpoint_path(TASK_ID, replace(cfg, qualification_checkpoint=None)) is None
    cfg.agent.resume = True
    with pytest.raises(ValueError, match="resume"):
        train.qualification_checkpoint_path(TASK_ID, cfg)
    cfg.agent.resume = False
    with pytest.raises(ValueError, match="experimental"):
        train.qualification_checkpoint_path("Other-Task", cfg)
    cfg.env.actions["strike_speed"].mode = "external"
    with pytest.raises(ValueError, match="policy"):
        train.qualification_checkpoint_path(TASK_ID, cfg)
    cfg.env.actions["strike_speed"].mode = "policy"
    with pytest.raises(FileNotFoundError):
        train.qualification_checkpoint_path(TASK_ID, replace(cfg, qualification_checkpoint=str(tmp_path / "absent.pt")))


def test_variable_mode_requires_sampled_control_and_is_available_in_all_entry_points():
    assert TrainConfig.__dataclass_fields__["sampled_pd500_gain_mode"].default == "fixed"
    assert play.PlayConfig.__dataclass_fields__["sampled_pd500_gain_mode"].default == "fixed"
    cfg = _cfg(None)
    with pytest.raises(ValueError, match="sampled"):
        configure_impact_speed_controller(cfg.env, sampled_pd500=False, gain_mode="variable")
    configure_impact_speed_controller(cfg.env, sampled_pd500=True, gain_mode="variable")
    assert cfg.env.sim.mujoco.timestep == .00025
    assert cfg.env.decimation == 80


def test_play_installs_variable_controller_before_viewer(monkeypatch):
    seen = []
    class Viewer:
        def __init__(self, wrapped, policy):
            controller = wrapped.unwrapped._sampled_pd500
            seen.append((controller.gain_mode, wrapped.unwrapped.cfg.sim.mujoco.timestep))
        def run(self):
            pass
    monkeypatch.setattr(play, "NativeMujocoViewer", Viewer)
    play.run_play(TASK_ID, play.PlayConfig(
        agent="zero", device="cpu", num_envs=1, viewer="native",
        speed_bounds=(.2, .8), speed_cost=.4,
        sampled_pd500=True, sampled_pd500_gain_mode="variable",
    ))
    assert seen == [("variable", .00025)]
