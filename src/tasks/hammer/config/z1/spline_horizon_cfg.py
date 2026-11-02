"""Opt-in contact-relative horizon comparison on the unchanged deferred-credit D.

Both arms retain normal policy-step termination and any-native-reset credit.
The extended arm changes training exposure and deferred reward timing together;
it does not add settling, stopping, recovery targets or a new safety threshold.
"""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.spline_learning_cfg import (
    spline_credit_env_cfg, spline_credit_rl_cfg,
)
from src.tasks.hammer.mdp.spline_learning_variants import TASKS as LEARNING_TASKS


WINDOWS_S = {"native50": .05, "extended200": .2}
TASKS = {arm: LEARNING_TASKS["D"] + "-Horizon-" + arm for arm in WINDOWS_S}


def spline_horizon_env_cfg(*, play: bool, arm: str):
    if arm not in WINDOWS_S:
        raise ValueError("horizon arm must be native50 or extended200")
    cfg = spline_credit_env_cfg(play=play, arm="D")
    from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
    cfg.terminations["nail_driven"].func = horizon_window_complete
    cfg.terminations["nail_driven"].params = {"window_s": WINDOWS_S[arm]}
    return cfg


def register_spline_horizon_tasks():
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

    for arm, task in TASKS.items():
        register_mjlab_task(
            task_id=task,
            env_cfg=spline_horizon_env_cfg(play=False, arm=arm),
            play_env_cfg=spline_horizon_env_cfg(play=True, arm=arm),
            rl_cfg=spline_credit_rl_cfg("D"),
            runner_cls=HammerOnPolicyRunner,
        )
