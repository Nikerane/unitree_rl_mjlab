"""Opt-in eight-cell credit timing and critic-event-information screen.

Every cell retains the complete 200 ms horizon and the underlying D treatment;
only the accepted-unit payment deadline and optional critic inputs differ.
"""

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.spline_horizon_cfg import (
    TASKS as HORIZON_TASKS,
    spline_horizon_env_cfg,
)
from src.tasks.hammer.config.z1.spline_learning_cfg import spline_credit_rl_cfg


DEADLINES_MS = (50, 100, 150, 200)
CRITIC_MODES = ("current", "event")
TASKS = {
    (deadline, mode): HORIZON_TASKS["extended200"] + f"-Credit{deadline}-Critic{mode.title()}"
    for deadline in DEADLINES_MS
    for mode in CRITIC_MODES
}


def _require_critic_mode(critic_mode: str) -> None:
    if type(critic_mode) is not str or critic_mode not in CRITIC_MODES:
        raise ValueError("critic mode must be current or event")


def credit_critic_env_cfg(*, play: bool, credit_deadline_ms: int, critic_mode: str):
    """Change only payment timing and, in event mode, four critic-only terms."""
    if type(credit_deadline_ms) is not int or credit_deadline_ms not in DEADLINES_MS:
        raise ValueError("credit deadline must be an integer in (50, 100, 150, 200) ms")
    _require_critic_mode(critic_mode)
    from src.tasks.hammer.mdp.spline_credit_deadline import (
        CREDIT_EVENT_FIELDS, DeadlineSplineStrikeReward, credit_event_observation,
    )

    cfg = spline_horizon_env_cfg(play=play, arm="extended200")
    reward = cfg.rewards["accepted_spline_strike"]
    reward.func = DeadlineSplineStrikeReward
    reward.params.pop("onset_fraction")
    reward.params["credit_deadline_ms"] = credit_deadline_ms
    if critic_mode == "event":
        for field in CREDIT_EVENT_FIELDS:
            cfg.observations["critic"].terms[field] = ObservationTermCfg(
                func=credit_event_observation,
                params={"field": field, "credit_deadline_ms": credit_deadline_ms},
            )
    return cfg


def credit_critic_rl_cfg(*, critic_mode: str):
    """Keep the N actor and PPO unchanged; select only the event critic class."""
    _require_critic_mode(critic_mode)
    cfg = spline_credit_rl_cfg("D")
    if critic_mode == "event":
        cfg.critic.class_name = "src.tasks.hammer.rl.spline_credit_critic:CreditEventCritic"
    return cfg


def register_credit_critic_tasks():
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

    for (deadline, mode), task in TASKS.items():
        register_mjlab_task(
            task_id=task,
            env_cfg=credit_critic_env_cfg(
                play=False, credit_deadline_ms=deadline, critic_mode=mode,
            ),
            play_env_cfg=credit_critic_env_cfg(
                play=True, credit_deadline_ms=deadline, critic_mode=mode,
            ),
            rl_cfg=credit_critic_rl_cfg(critic_mode=mode),
            runner_cls=HammerOnPolicyRunner,
        )
