"""Evaluation-only sampled torque step wrapper."""
import torch
from sampled_pd import SampledPD, write_held_torque


class HeldTorqueStep:
    def __init__(self,sim,qids,vids,kp,kd,limits,dt):
        self.sim=sim; self.original=sim.step
        self.qids=qids; self.vids=vids
        self.kp=kp; self.kd=kd; self.limits=limits
        self.pd=SampledPD(dt); self.count=0; self.max_error=0.

    def __call__(self):
        torque=self.pd.update(self.count,q=self.sim.data.qpos[:,self.qids],
            v=self.sim.data.qvel[:,self.vids],target=self.sim.data.ctrl,
            kp=self.kp,kd=self.kd,limits=self.limits)
        write_held_torque(self.sim.model,torque)
        self.original()
        actual=self.sim.data.actuator_force[:]
        torch.testing.assert_close(actual,torque,rtol=0,atol=1e-5)
        self.max_error=max(self.max_error,float((actual-torque).abs().max()))
        self.count+=1
