"""Opt-in fixed-gain pilot: 500 Hz torque hold, 4 kHz physics, 50 Hz policy.

The 13-output policy interface is retained; stiffness outputs have no physical
effect in this fixed-impedance experiment. Native checkpoint routes are untouched.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch


ARM_KP = (256., 384., 384., 256., 192., 128.)
ARM_KD = (6.4,) * 6


def configure_sampled_pd500(cfg) -> None:
    if tuple(cfg.actions) != ("joint_position", "joint_stiffness", "strike_speed"):
        raise ValueError("sampled PD pilot requires the impact-speed action interface")
    cfg.sim.mujoco.timestep = .00025
    cfg.decimation = 80
    cfg.metrics["first_strike"].params["window_substeps"] = 200
    cfg.metrics["substep_impulse"].params["event_window_substeps"] = 200
    cfg.metrics["substep_delivered"].params.update(
        event_window_substeps=200, rearm_gap_substeps=200)


class SampledPD500:
    """Own the torque hold at the simulation-step seam, without host sync."""

    def __init__(self, env):
        if env.cfg.sim.mujoco.timestep != .00025 or env.cfg.decimation != 80:
            raise ValueError("sampled PD pilot requires .00025 physics / decimation 80")
        self.env = env
        self.sim = env.sim
        model = self.sim.mj_model
        if (model.nu != 7
            or not np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            or not np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
            or not np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
            or not np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
            or not np.all(model.actuator_forcelimited)
            or not np.array_equal(model.actuator_gear, np.tile([1.,0,0,0,0,0], (7,1)))):
            raise ValueError("sampled PD requires seven unit-gear, affine joint actuators")
        self.gains = env.action_manager.get_term("joint_stiffness")
        self.ids = self.gains.control_ids
        self.kp = self.sim.model.actuator_gainprm[:, :, 0].clone()
        self.kd = -self.sim.model.actuator_biasprm[:, :, 2].clone()
        self.arm_kp = self.kp.new_tensor(ARM_KP)
        self.arm_kd = self.kd.new_tensor(ARM_KD)
        self.kp[:, self.ids] = self.arm_kp
        self.kd[:, self.ids] = self.arm_kd
        self.limits = self.kp.new_tensor(model.actuator_forcerange.copy())
        expected = self.limits.new_tensor([30.,60.,30.,30.,30.,30.])
        if not torch.equal(self.limits[self.ids], torch.stack((-expected, expected), -1)):
            raise ValueError("sampled PD must preserve canonical effort limits")
        joints = model.actuator_trnid[:, 0]
        self.qids = torch.as_tensor(model.jnt_qposadr[joints].copy(), device=env.device).long()
        self.vids = torch.as_tensor(model.jnt_dofadr[joints].copy(), device=env.device).long()
        self.original_step = self.sim.step
        self.original_process = self.gains.process_actions
        self.original_reset = self.gains.reset
        self.original_env_reset = env.reset
        self.count = self.updates = 0
        self.torque = None
        self.validate_physics()

    def validate_physics(self):
        model = self.sim.mj_model
        ids = self.gains.control_ids
        joints = model.actuator_trnid[:, 0]
        qids = torch.as_tensor(model.jnt_qposadr[joints].copy(), device=self.env.device).long()
        vids = torch.as_tensor(model.jnt_dofadr[joints].copy(), device=self.env.device).long()
        expected_kp = torch.full_like(self.kp, 100.)
        expected_kd = torch.full_like(self.kd, 20.)
        expected_kp[:, ids] = expected_kp.new_tensor(ARM_KP)
        expected_kd[:, ids] = expected_kd.new_tensor(ARM_KD)
        if not (torch.equal(self.ids, ids) and torch.equal(self.qids, qids)
                and torch.equal(self.vids, vids)):
            raise ValueError("sampled controller actuator mapping drift")
        if not (torch.equal(self.kp, expected_kp) and torch.equal(self.kd, expected_kd)
                and torch.equal(self.arm_kp, self.kp.new_tensor(ARM_KP))
                and torch.equal(self.arm_kd, self.kd.new_tensor(ARM_KD))
                and torch.equal(self.limits, self.limits.new_tensor(model.actuator_forcerange.copy()))):
            raise ValueError("sampled controller gains/limits drift")
        windows = (("first_strike", "window_substeps", "_hammer_first_strike", "_window"),
                   ("substep_impulse", "event_window_substeps", "_hammer_substep_impulse", "_window"),
                   ("substep_delivered", "event_window_substeps", "_hammer_substep_delivered", "_window"),
                   ("substep_delivered", "rearm_gap_substeps", "_hammer_substep_delivered", "_rearm_gap"))
        for term, param, attr, field in windows:
            if (self.env.cfg.metrics[term].params[param] != 200
                    or getattr(getattr(self.env, attr), field) != 200):
                raise ValueError("sampled controller requires 50ms event windows")

    def fixed_gains(self):
        self.gains._kp[:] = self.arm_kp
        self.gains._kd[:] = self.arm_kd

    def process_actions(self, actions):
        self.original_process(actions)
        self.fixed_gains()

    def reset_gains(self, env_ids=None):
        self.original_reset(env_ids)
        self.fixed_gains()
        self.gains._write_native_gains(self.gains._resolve_env_ids(env_ids))

    def reset_env(self, *args, **kwargs):
        self.count = self.updates = 0
        self.torque = None
        return self.original_env_reset(*args, **kwargs)

    def step(self):
        if self.count % 8 == 0:
            requested = (self.kp * (self.sim.data.ctrl[:] - self.sim.data.qpos[:, self.qids])
                         - self.kd * self.sim.data.qvel[:, self.vids])
            self.torque = torch.maximum(torch.minimum(requested, self.limits[:, 1]), self.limits[:, 0])
            self.updates += 1
        # Native action application rewrites PD fields every physics step.
        # Clear ALL feedback immediately before integration: torque only.
        self.sim.model.actuator_gainprm.zero_()
        self.sim.model.actuator_biasprm.zero_()
        self.sim.model.actuator_biasprm[:, :, 0].copy_(self.torque)
        self.original_step()
        self.count += 1

    def metadata(self):
        self.validate_physics()
        if (self.sim.step != self.step or self.gains.process_actions != self.process_actions
            or self.gains.reset != self.reset_gains or self.env.reset != self.reset_env):
            raise ValueError("sampled PD controller hooks are not installed")
        if self.env.cfg.sim.mujoco.timestep != .00025 or self.env.cfg.decimation != 80:
            raise ValueError("sampled PD timing changed after installation")
        return dict(version="fixed_pd500_v1", pd_dt_s=.002, physics_dt_s=.00025,
                    policy_dt_s=.02, kp=list(ARM_KP), kd=list(ARM_KD),
                    control_ids=self.ids.tolist(), effort_limits_nm=[30.,60.,30.,30.,30.,30.],
                    gripper_kp=100., gripper_kd=20., ignored_action_indices=list(range(6,12)),
                    event_window_s=.05, delivered_rearm_gap_s=.05,
                    torque_hold="zero_order_hold_without_native_feedback")


def install_sampled_pd500(env) -> SampledPD500:
    if hasattr(env, "_sampled_pd500"):
        raise ValueError("sampled PD already installed")
    controller = SampledPD500(env)
    env._sampled_pd500 = controller
    env.sim.step = controller.step
    controller.gains.process_actions = controller.process_actions
    controller.gains.reset = controller.reset_gains
    env.reset = controller.reset_env
    controller.fixed_gains()
    return controller
