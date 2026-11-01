"""Opt-in native physics depth observer with one policy-boundary receipt.

The installed mjlab metrics API uses ManagerTermBase with per_substep=True.
Only this owner resets episode state; reward functions are stateless readers.
"""
from types import MappingProxyType
import torch
from mjlab.managers.manager_base import ManagerTermBase
from src.tasks.hammer.mdp.depth_credit031 import DepthCreditLedger031
from src.tasks.hammer.mdp.spline_credit_deadline import elapsed_contact_ticks

ARM_FLAGS = MappingProxyType({'A0': (False, False), 'A1': (True, False),
                              'A2': (False, True), 'A3': (True, True)})


def arm_flags(arm):
    if type(arm) is not str or arm not in ARM_FLAGS:
        raise ValueError('depth031 arm must be A0, A1, A2 or A3')
    return ARM_FLAGS[arm]


class NativeDepthCreditObserver031(ManagerTermBase):
    """Observe integrated nail qpos once per physics substep, without host reads."""
    def __init__(self, cfg, env):
        super().__init__(env)
        if hasattr(env, '_z1_depth_credit031'):
            raise ValueError('depth031 permits exactly one observer owner')
        self._arm = cfg.params['arm']
        progress, completion = arm_flags(self._arm)
        self.nail_cfg = cfg.params['nail_cfg']
        self.ledger = DepthCreditLedger031(env.num_envs, env.device,
            progress_enabled=progress, completion_enabled=completion)
        self.sample_count = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.first_attainment_tick = torch.full_like(self.sample_count, -1)
        self._previous_potential = torch.zeros(env.num_envs, device=env.device)
        self._completion_paid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._counter = None
        self._snapshot = None
        self.call_index = 0
        env._z1_depth_credit031 = self
        self.reset(None)

    @property
    def arm(self):
        return self._arm

    def _depth(self):
        # Read integrated qpos; no forward/sensor/physical step is required.
        return self._env.scene[self.nail_cfg.name].data.joint_pos[:, self.nail_cfg.joint_ids].squeeze(1)

    @property
    def last_snapshot(self):
        return None if self._snapshot is None else {k: v.clone() for k,v in self._snapshot.items()}

    @torch.no_grad()
    def __call__(self, env, *, arm, nail_cfg):
        del nail_cfg
        if env is not self._env or arm != self.arm:
            raise ValueError('depth031 observer identity differs')
        depth = self._depth()
        self.sample_count.add_(1)
        crossing = (depth >= .030) & (self.first_attainment_tick < 0)
        self.first_attainment_tick.copy_(torch.where(crossing, self.sample_count, self.first_attainment_tick))
        self.ledger.observe(depth)
        return depth.clamp(0., .032)

    @torch.no_grad()
    def collect(self):
        env = self._env
        if self._counter == env.common_step_counter:
            return self.last_snapshot
        if type(env.common_step_counter) is not int:
            raise ValueError('depth031 requires integer policy counter')
        elapsed = elapsed_contact_ticks(env)
        if env.physics_dt != .00025 or env.step_dt != .02:
            raise ValueError('depth031 requires 4kHz physics and 20ms policy clock')
        horizon = env._hammer_first_strike.started & (elapsed >= 800)
        snap = self.ledger.collect(horizon_complete=horizon, native_reset=env.reset_buf)
        snap.update(previous_progress_potential=self._previous_potential.clone(),
                    completion_paid_before=self._completion_paid.clone(),
                    completion_paid_after=(snap['completion_paid_total'] > 0),
                    physics_sample_count=self.sample_count.clone(),
                    first_attainment_tick=self.first_attainment_tick.clone(), elapsed_contact_ticks=elapsed.clone())
        self._previous_potential.copy_(snap['progress_potential'])
        self._completion_paid.copy_(snap['completion_paid_after'])
        self.call_index += 1
        snap['call_index'] = torch.full_like(self.sample_count, self.call_index)
        self._snapshot = snap
        self._counter = env.common_step_counter
        return self.last_snapshot

    @torch.no_grad()
    def reset(self, env_ids=None):
        # Scene/events have reset before metrics. Never read another monitor.
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        ids = slice(None) if env_ids is None else env_ids
        initial = self._depth()[ids]
        self.ledger.reset(env_ids, initial_depth_m=initial)
        initial = initial.clamp(0., .032).float()
        potential = ((initial - .004) / .026).clamp(0., 1.)
        self._previous_potential[ids] = torch.where(initial >= .030, 1., potential)
        self._completion_paid[ids] = False
        self.sample_count[ids] = 0
        self.first_attainment_tick[ids] = torch.where(initial >= .030, 0, -1)
        # Preserve both cache counter and last pre-reset snapshot. A reader in
        # this same policy step must never recollect a partially reset batch.


def depth_progress_reward031(env):
    return env._z1_depth_credit031.collect()['progress_paid'] / env.step_dt


def depth_completion_reward031(env):
    return env._z1_depth_credit031.collect()['completion_paid'] / env.step_dt


def depth_collect_metric031(env):
    """Separate non-substep collector ensures A0 also closes every boundary."""
    return env._z1_depth_credit031.collect()['peak_depth_m']
