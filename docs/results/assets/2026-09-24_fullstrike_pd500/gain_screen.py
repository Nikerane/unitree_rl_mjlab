"""Small, isolated helpers for the approved no-contact gain screen."""
import math
import numpy as np
import torch


def validate_arm_columns(resolved, native):
    torch.testing.assert_close(resolved.long(),native.long())


def apply_gains(kp, kd, arm_ids, scale, damping):
    if not all(math.isfinite(x) and x > 0 for x in (scale,damping)):
        raise ValueError("gain scale and damping must be finite and positive")
    arm_kp=torch.tensor([512,768,768,512,384,256],dtype=kp.dtype,device=kp.device)*scale
    kp[:,arm_ids]=arm_kp
    kd[:,arm_ids]=damping
    return arm_kp


def max_saturated_reversals(torque, limits):
    """Input is controller ticks, worlds, canonical joints, not physics steps."""
    saturated=(torque<=limits[None,None,:,0]+1e-4)|(torque>=limits[None,None,:,1]-1e-4)
    reverse=saturated[1:] & saturated[:-1] & (torque[1:]*torque[:-1]<0)
    run=np.zeros(torque.shape[1:],dtype=int)
    longest=run.copy()
    for tick in reverse:
        run=np.where(tick,run+1,0)
        longest=np.maximum(longest,run)
    return longest
