"""Opt-in sampled joint impedance: 500 Hz torque hold, 4 kHz physics, 50 Hz policy."""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from .variable_impedance import _variable_impedance_gains_unchecked, variable_impedance_gains


ARM_KP = (256., 384., 384., 256., 192., 128.)
ARM_KD = (6.4,) * 6


def _check_gain_mode(gain_mode):
    if gain_mode not in ("fixed", "variable"):
        raise ValueError(f"gain_mode must be 'fixed' or 'variable', got {gain_mode!r}")


def configure_sampled_pd500(cfg, *, gain_mode="fixed") -> None:
    _check_gain_mode(gain_mode)
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

    def __init__(self, env, *, gain_mode="fixed"):
        _check_gain_mode(gain_mode)
        self.gain_mode = gain_mode
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
        self.C = self.gains._C
        self.kp = self.sim.model.actuator_gainprm[:, :, 0].clone()
        self.kd = -self.sim.model.actuator_biasprm[:, :, 2].clone()
        self.arm_kp = self.kp.new_tensor(ARM_KP)
        self.arm_kd = self.kd.new_tensor(ARM_KD)
        if gain_mode == "variable":
            if self.C != 1.25:
                raise ValueError("sampled variable impedance requires C=1.25")
            variable_impedance_gains(
                torch.zeros((env.num_envs, 6), dtype=self.kp.dtype, device=env.device),
                self.arm_kp, self.arm_kd, C=self.C,
            )
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
        self.requested_torque = None
        self.saturated = None
        self.applied_kp = None
        self.applied_kd = None
        self.validate_physics()

    def validate_physics(self):
        model = self.sim.mj_model
        if (model.nu != 7
            or not np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            or not np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
            or not np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
            or not np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
            or not np.all(model.actuator_forcelimited)
            or not np.array_equal(model.actuator_gear, np.tile([1.,0,0,0,0,0], (7,1)))):
            raise ValueError("sampled controller physics actuator mechanics drift")
        # Validate the arrays the integrator actually reads, not just its host
        # template. Some bridge fields are shared; others carry a world axis.
        # Gain/bias parameters are controller-owned and rewritten on each step.
        for name in ("actuator_trntype", "actuator_dyntype", "actuator_gaintype",
                     "actuator_biastype", "actuator_forcelimited", "actuator_gear",
                     "actuator_trnid", "actuator_forcerange", "jnt_qposadr", "jnt_dofadr"):
            expected = torch.as_tensor(getattr(model, name).copy())
            actual = getattr(self.sim.model, name).detach().cpu()
            if actual.ndim == expected.ndim + 1:
                if actual.shape[0] not in (1, self.env.num_envs):
                    raise ValueError(f"sampled runtime {name} has unexpected world shape")
                if not torch.equal(actual, actual[:1].expand_as(actual)):
                    raise ValueError(f"sampled runtime {name} is heterogeneous across worlds")
                actual = actual[0]
            if not torch.equal(actual, expected):
                raise ValueError(f"sampled runtime {name} actuator mechanics/mapping/limits drift")
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
        if self.gain_mode == "fixed":
            self.fixed_gains()

    def reset_gains(self, env_ids=None):
        self.original_reset(env_ids)
        if self.gain_mode == "fixed":
            self.fixed_gains()
            self.gains._write_native_gains(self.gains._resolve_env_ids(env_ids))
        else:
            ids = self.gains._resolve_env_ids(env_ids)
            for snapshot in (self.torque, self.requested_torque, self.saturated,
                             self.applied_kp, self.applied_kd):
                if snapshot is not None:
                    snapshot[ids] = 0

    def reset_env(self, *args, **kwargs):
        self.count = self.updates = 0
        self.torque = None
        self.requested_torque = None
        self.saturated = None
        self.applied_kp = None
        self.applied_kd = None
        return self.original_env_reset(*args, **kwargs)

    def step(self):
        if self.count % 8 == 0:
            sampled_kp = self.kp.clone()
            sampled_kd = self.kd.clone()
            if self.gain_mode == "variable":
                mapped = _variable_impedance_gains_unchecked(
                    self.gains._p, self.arm_kp, self.arm_kd, C=self.C,
                )
                sampled_kp[:, self.ids] = mapped.kp
                sampled_kd[:, self.ids] = mapped.kd
            self.applied_kp = sampled_kp[:, self.ids].clone()
            self.applied_kd = sampled_kd[:, self.ids].clone()
            requested = (sampled_kp * (self.sim.data.ctrl[:] - self.sim.data.qpos[:, self.qids])
                         - sampled_kd * self.sim.data.qvel[:, self.vids])
            self.requested_torque = requested.clone()
            self.torque = torch.maximum(torch.minimum(requested, self.limits[:, 1]), self.limits[:, 0])
            self.saturated = requested.ne(self.torque)
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
        if self.gain_mode == "variable" and (
            self.C != 1.25 or self.gains._C != self.C
            or self.gains.cfg.C != self.C
            or tuple(self.gains.cfg.joint_names) != ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
            or not torch.equal(self.gains.control_ids, self.ids)
        ):
            raise ValueError("sampled variable impedance C, joint order, or mapping changed")
        if (self.sim.step != self.step or self.gains.process_actions != self.process_actions
            or self.gains.reset != self.reset_gains or self.env.reset != self.reset_env):
            raise ValueError("sampled PD controller hooks are not installed")
        live_dt = self.sim.model.opt.timestep.detach().cpu()
        if (self.env.cfg.sim.mujoco.timestep != .00025 or self.env.cfg.decimation != 80
            or live_dt.ndim != 1 or live_dt.shape[0] not in (1, self.env.num_envs)
            or not torch.all(live_dt == .00025)):
            raise ValueError("sampled PD execution timestep or configured timing changed after installation")
        metadata = dict(version="fixed_pd500_v1", pd_dt_s=.002, physics_dt_s=.00025,
                    policy_dt_s=.02, kp=list(ARM_KP), kd=list(ARM_KD),
                    control_ids=self.ids.tolist(), effort_limits_nm=[30.,60.,30.,30.,30.,30.],
                    gripper_kp=100., gripper_kd=20., ignored_action_indices=list(range(6,12)),
                    event_window_s=.05, delivered_rearm_gap_s=.05,
                    torque_hold="zero_order_hold_without_native_feedback")
        if self.gain_mode == "variable":
            metadata["version"] = "variable_impedance_pd500_v1"
            metadata["gain_mode"] = "variable"
            metadata["mapping_family"] = "author_v1_exponential"
            metadata["C"] = 1.25
            metadata["p_bounds"] = [-1., 1.]
            metadata.pop("ignored_action_indices")
        return metadata


def install_sampled_pd500(env, *, gain_mode="fixed") -> SampledPD500:
    if hasattr(env, "_sampled_pd500"):
        raise ValueError("sampled PD already installed")
    controller = SampledPD500(env, gain_mode=gain_mode)
    env._sampled_pd500 = controller
    env.sim.step = controller.step
    controller.gains.process_actions = controller.process_actions
    controller.gains.reset = controller.reset_gains
    env.reset = controller.reset_env
    if gain_mode == "fixed":
        controller.fixed_gains()
    return controller
