"""Opt-in overnight spline treatments; the completed pilot remains unchanged."""

import torch
from mjlab.managers.manager_base import ManagerTermBase

from src.tasks.hammer.mdp import observations
from src.tasks.hammer.mdp.spline_curriculum import TASK, SplineProgressReward

TASKS = {"B": TASK, "O": TASK + "-Observation", "P": TASK + "-VisitedBins"}


def _current_height_geometry(env, robot_cfg, nail_cfg, **reference_params):
    # Keep the existing anchoring/latch writer for every other shared consumer.
    observations.strike_phase(env, robot_cfg, nail_cfg, **reference_params)
    head = env.scene[robot_cfg.name].data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    phase, waypoint, _ = env._strike_reference.diagnose_current_height(head)
    return phase, waypoint - head


def current_height_phase(env, robot_cfg, nail_cfg, horizontal_detour_m=0.,
                         followthrough_mode="vertical", path_shape="straight", overshoot=.15):
    """Current-height full-spline phase, including backward motion on rebound."""
    phase, _ = _current_height_geometry(env, robot_cfg, nail_cfg,
        horizontal_detour_m=horizontal_detour_m, followthrough_mode=followthrough_mode,
        path_shape=path_shape, overshoot=overshoot)
    return phase.unsqueeze(-1)


def current_height_error(env, robot_cfg, nail_cfg, horizontal_detour_m=0.,
                         followthrough_mode="vertical", path_shape="straight", overshoot=.15):
    """Current-height spline waypoint minus head, using the reward's geometry."""
    _, error = _current_height_geometry(env, robot_cfg, nail_cfg,
        horizontal_detour_m=horizontal_detour_m, followthrough_mode=followthrough_mode,
        path_shape=path_shape, overshoot=overshoot)
    return error


class SplineBinVisits:
    """Episode-owned credit for actual precontact visits, independent of evaluation.

    A visit to the starting bin earns one occupancy bonus of 1/16. Hovering and
    repeated visits earn nothing further. No sample implies no credit: intervals
    between samples are never interpolated, and out-of-domain heights are ignored.
    """

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.visited = torch.zeros(num_envs, 16, device=device, dtype=torch.bool)
        self._pending = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._worlds = torch.arange(num_envs, device=device)

    def reset(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self.visited[ids] = False
        self._pending[ids] = 0

    def sample(self, *, approach_height, error_m, started):
        eligible = (~started & torch.isfinite(approach_height) & torch.isfinite(error_m)
                    & (approach_height >= 0.) & (approach_height <= 1.)
                    & (error_m >= 0.) & (error_m <= .01))
        # Sanitization only makes indexing safe. Eligibility still rejects NaN,
        # infinities and out-of-domain samples instead of inventing endpoint visits.
        bins = (torch.nan_to_num(approach_height).clamp(0., 1.) * 16).long().clamp_max(15)
        seen = self.visited[self._worlds, bins]
        new = eligible & ~seen
        self.visited[self._worlds, bins] = seen | new
        self._pending += new.long()
        return new.float() / 16

    def drain(self):
        credit = self._pending.float() / 16
        self._pending.zero_()
        return credit


class VisitedBinsProgressReward(ManagerTermBase):
    """Flush precontact visits once at reward cadence, including an onset step.

    This term owns reset. A first contact later in the control step does not erase
    previously queued credit; the sampler prevents any onset/postcontact earning.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.collector = SplineBinVisits(self.num_envs, self.device)
        env._spline_bin_visits = self.collector

    def reset(self, env_ids):
        self.collector.reset(env_ids)

    def __call__(self, env):
        return self.collector.drain() / env.step_dt


class SplineBinVisitMetric(ManagerTermBase):
    """Read each real physics sample after the first-strike tracker updates.

    Geometry matches diagnose_current_height, without its host-synchronizing
    diagnostic guards in the physics loop. Anchored quintic references are checked
    in the task contract; no reference state is mutated here.
    """

    def __init__(self, cfg, env):
        super().__init__(env)

    def __call__(self, env, robot_cfg):
        head = env.scene[robot_cfg.name].data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
        ref = env._strike_reference
        phase = ((ref._head0[:, 2] - head[:, 2]) /
                 (ref._head0[:, 2] - ref._target[:, 2])).clamp(0., 1.)
        error = (head - ref.waypoint(phase)).norm(dim=-1)
        height = ((ref._head0[:, 2] - head[:, 2]) /
                  (ref._head0[:, 2] - ref._nail[:, 2]))
        return env._spline_bin_visits.sample(approach_height=height, error_m=error,
            started=env._hammer_first_strike.started | ~ref._anchored)


def variant_metadata(env):
    """Validate actual O/P plumbing; add no fields to legacy B contracts."""
    original = (observations.strike_phase, observations.strike_ref_error)
    current = (current_height_phase, current_height_error)
    pairs = [tuple(env.observation_manager.get_term_cfg(group, name).func
                   for name in ("strike_phase", "strike_ref_error"))
             for group in ("actor", "critic")]
    if pairs[0] != pairs[1] or pairs[0] not in (original, current):
        raise ValueError("spline variant requires matching actor/critic observation semantics")
    progress = env.reward_manager.get_term_cfg("spline_progress")
    visit_reward = type(progress.func) is VisitedBinsProgressReward
    legacy_reward = type(progress.func) is SplineProgressReward
    manager = env.metrics_manager
    has_metric = "spline_bin_visits" in manager.active_terms
    if (not (visit_reward or legacy_reward) or visit_reward != has_metric
            or (visit_reward and pairs[0] != original)):
        raise ValueError("spline variant must isolate observations or visited-bin progress")
    if legacy_reward and pairs[0] == original:
        return None
    def identity(func):
        return f"{func.__module__}.{func.__qualname__}"
    metadata = {"schema_version": "spline_speed_variant_v1",
                "variant": "P" if visit_reward else "O",
                "observation_geometry": "monotone_axis_gated" if visit_reward else "current_height",
                "observation_functions": [identity(func) for func in pairs[0]]}
    if visit_reward:
        index = manager.active_terms.index("spline_bin_visits")
        first_index = manager.active_terms.index("first_strike")
        metric = manager._term_cfgs[index]
        first = manager._term_cfgs[first_index]
        robot = metric.params.get("robot_cfg")
        if (type(metric.func) is not SplineBinVisitMetric or not metric.per_substep
                or not first.per_substep or index <= first_index
                or index not in manager._substep_term_indices
                or first_index not in manager._substep_term_indices
                or progress.params or progress.weight != 1.
                or set(metric.params) != {"robot_cfg"}
                or robot.name != "robot" or tuple(robot.site_names) != ("hammer_head_site",)
                or progress.func.collector is not getattr(env, "_spline_bin_visits", None)):
            raise ValueError("spline P variant requires ordered physics-substep bin collection")
        metadata["progress"] = {
            "collector_impl": identity(SplineBinVisitMetric),
            "sampling": "physics_substep_after_first_strike_before_reward",
            "physics_dt_s": env.physics_dt, "bins": 16, "tube_width_m": .01,
            "domain": "unclamped_start_to_nail_height_inclusive_0_to_1",
            "credit": "one_sixteenth_once_per_actually_observed_bin_no_interpolation",
            "initial_bin": "one_time_occupancy_credit_on_actual_sample",
            "contact": "exclude_onset_and_postcontact_flush_prior_pending_credit",
            "maximum_episode_credit": 1.,
        }
    return metadata
