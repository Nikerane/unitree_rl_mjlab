import numpy as np
import pytest
import torch
import gain_screen as subject


def test_native_mapping_scales_arm_but_preserves_gripper():
    # Catches canonical-joint/native-actuator order confusion and gripper changes.
    kp=torch.full((2,7),100.); kd=torch.full((2,7),20.)
    subject.apply_gains(kp,kd,torch.tensor([0,5,1,2,3,4]),.5,6.4)
    torch.testing.assert_close(kp,torch.tensor([[256,384,256,192,128,384,100.]]).repeat(2,1))
    torch.testing.assert_close(kd,torch.tensor([[6.4,6.4,6.4,6.4,6.4,6.4,20.]]).repeat(2,1))


@pytest.mark.parametrize('scale,damping',[(0,6.4),(float('nan'),6.4),(1,-1),(1,float('inf'))])
def test_invalid_gains_rejected(scale,damping):
    with pytest.raises(ValueError):
        subject.apply_gains(torch.ones(1,7),torch.ones(1,7),torch.tensor([0,5,1,2,3,4]),scale,damping)


def test_chatter_counts_only_consecutive_saturated_reversals():
    # Columns: sustained chatter, isolated reversal, tiny unsaturated reversals,
    # and chatter broken by an unsaturated sample.
    torque=np.array([[30,30,1,30],[-30,-30,-1,-30],[30,-30,1,0],[-30,-30,-1,30],[30,-30,1,-30.]])[:,None,:]
    limits=np.tile([-30.,30.],(4,1))
    np.testing.assert_array_equal(subject.max_saturated_reversals(torque,limits),[[4,1,0,1]])


def test_chatter_respects_each_joint_force_limit():
    torque=np.array([[60,30],[-60,-30],[60,30],[-60,-30.]])[:,None,:]
    np.testing.assert_array_equal(subject.max_saturated_reversals(torque,np.array([[-60,60],[-60,60]])),[[3,0]])
