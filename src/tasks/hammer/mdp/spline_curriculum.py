"""Bounded geometric curriculum: pure reference readers, episode-owned credit."""

import torch
from mjlab.managers.manager_base import ManagerTermBase

TASK = "Unitree-Z1-Hammer-Quintic-ThreeStarts-SplineSpeedCurriculum"
DISABLED_REWARDS = ("approach", "nail_driven", "nail_depth_delta", "impact_progress",
                    "completion", "delivered_impulse", "r_imit")


def _geometry(env, robot_cfg):
    head = env.scene[robot_cfg.name].data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    ref = env._strike_reference
    _, _, error = ref.diagnose_current_height(head)
    return ref.approach_height_coordinate(head), error


class SplinePathDeviationCost(ManagerTermBase):
    """Squared current-height path error, capped at four, until accepted onset."""

    def __init__(self, cfg, env):
        super().__init__(env)

    def __call__(self, env, robot_cfg, width_m=.01, clamp_cost=4.):
        _, error = _geometry(env, robot_cfg)
        cost = (error / width_m).square().clamp_max(clamp_cost)
        return torch.where(env._hammer_first_strike.started, 0., cost)


class SplineProgressReward(ManagerTermBase):
    """At most one unit per episode for new precontact, on-path height progress.

    Return event units / step_dt because the reward manager multiplies by dt.
    The shared reference anchors/phase are never changed by these readers.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self._credited = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids):
        self._credited[slice(None) if env_ids is None else env_ids] = 0.

    def __call__(self, env, robot_cfg, width_m=.01):
        progress, error = _geometry(env, robot_cfg)
        eligible = (~env._hammer_first_strike.started) & (error <= width_m)
        delta = torch.where(eligible, (progress - self._credited).clamp_min(0.), 0.)
        self._credited += delta
        return delta / env.step_dt


class AcceptedSplineStrikeReward(ManagerTermBase):
    """Pay accepted downward first contact near the precontact spline endpoint once.

    Snapshot position is taken at physics onset, so rebound before reward time
    cannot relabel the event. A rejected first contact cannot earn a later bonus.
    Endpoint tolerance is the same 10 mm width as path eligibility.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self._consumed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids):
        self._consumed[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env, robot_cfg, width_m=.01):
        del robot_cfg
        return self._accepted_onset(env, width_m).float() / env.step_dt

    def _accepted_onset(self, env, width_m):
        """Consume exactly the original first-contact eligibility decision."""
        tracker, ref = env._hammer_first_strike, env._strike_reference
        event = tracker.started & ~self._consumed
        _, _, error = ref.diagnose_current_height(tracker.onset_head_w)
        # The nail waypoint is frozen at reset; postcontact nail motion is irrelevant.
        endpoint_error = (tracker.onset_head_w - ref._nail).norm(dim=-1)
        accepted = event & (tracker.v_precontact > 0.) & (error <= width_m) & (endpoint_error <= width_m)
        self._consumed |= event
        return accepted


def accepted_strike_window_complete(env, window_s=.05):
    """Retain the complete 50 ms after accepted onset, even after early nail drive."""
    tracker = env._hammer_first_strike
    elapsed = tracker.elapsed_since_contact_s
    return tracker.started & (elapsed >= window_s - 1e-7)
