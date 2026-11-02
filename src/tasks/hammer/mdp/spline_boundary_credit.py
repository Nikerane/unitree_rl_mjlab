"""Split or defer the existing accepted-contact unit until the native boundary.

These are reward-timing treatments with the unchanged N observations. Eligibility
and unpaid credit are episode history, not actor/critic inputs: S/D are partially
observable, and do not claim a fully Markov observation contract.
"""

import torch

from src.tasks.hammer.mdp.spline_curriculum import AcceptedSplineStrikeReward


class BoundarySplineStrikeReward(AcceptedSplineStrikeReward):
    """Conserve one raw accepted unit, paying its balance once before native reset.

    mjlab computes termination/reset_buf after final physics, then rewards, then
    CaT metrics, then resets. Returning an event rate here makes the balance part
    of normal reward-manager positive-term accounting, including final-step CaT.
    Timeouts retain their existing bootstrap flags. No contact dwell, timer or
    evaluator cutoff can pay credit; an external reset discards unpaid credit.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.onset_fraction = cfg.params["onset_fraction"]
        if self.onset_fraction not in (0., .2):
            raise ValueError("onset_fraction must be 0.0 (D) or 0.2 (S)")
        self.unpaid_credit = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.unpaid_credit[slice(None) if env_ids is None else env_ids] = 0.

    def __call__(self, env, robot_cfg, onset_fraction, width_m=.01):
        del robot_cfg
        if onset_fraction != self.onset_fraction:
            raise ValueError("onset_fraction differs from the constructed reward treatment")
        accepted = self._accepted_onset(env, width_m).float()
        self.unpaid_credit += accepted * (1. - self.onset_fraction)
        balance = torch.where(env.reset_buf, self.unpaid_credit, 0.)
        self.unpaid_credit.masked_fill_(env.reset_buf, 0.)
        return (accepted * self.onset_fraction + balance) / env.step_dt
