"""Qualification transfer is an explicit, guarded training entry point."""

from dataclasses import replace

import pytest

import scripts.train as train
from scripts.train import TrainConfig
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
