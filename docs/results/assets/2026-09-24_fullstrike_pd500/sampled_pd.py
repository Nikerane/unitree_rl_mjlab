"""Diagnostic 500Hz sampled PD with explicit torque zero-order hold."""
import math
import torch


class SampledPD:
    def __init__(self, physics_dt):
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError('positive finite physics timestep required')
        self.stride=round(.002/physics_dt)
        if self.stride < 1 or not math.isclose(self.stride*physics_dt,.002,rel_tol=0,abs_tol=1e-12):
            raise ValueError('physics timestep must divide the fixed2ms PD interval')
        self.torque=None
        self.requested=None
        self.updates=0

    def update(self, step, *, q, v, target, kp, kd, limits):
        if step % self.stride == 0:
            self.requested=kp*(target-q)-kd*v
            self.torque=torch.maximum(torch.minimum(self.requested,limits[:,1]),limits[:,0]).clone()
            self.updates+=1
        if self.torque is None:
            raise ValueError('first sample must be a PD update tick')
        return self.torque


def write_held_torque(model, torque):
    """Use existing affine native actuators as constant-force slots.

    Caller verifies unit-gear joint transmissions, no actuator dynamics and
    fixed/affine types. Zero native position AND velocity feedback so implicit
    integration cannot evaluate hidden PD inside the held interval.
    """
    model.actuator_gainprm.zero_()
    model.actuator_biasprm.zero_()
    model.actuator_biasprm[:,:,0].copy_(torque)
