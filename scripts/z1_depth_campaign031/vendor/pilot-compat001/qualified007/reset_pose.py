"""Evaluation-only reset override; never changes source, cfg table or dynamics."""
import math

class ResetPoseContext:
    def __init__(self,env,pose,trained_table):
        self.env=env
        self.pose=tuple(float(x) for x in pose)
        self.table=tuple(tuple(float(x) for x in row) for row in trained_table)
        if len(self.pose)!=6 or not all(math.isfinite(x) for x in self.pose):raise ValueError('invalid effective reset pose')
        if len(self.table)!=3 or any(len(row)!=6 or not all(math.isfinite(x) for x in row) for row in self.table):raise ValueError('invalid trained reset table')
        self.active=False

    def __enter__(self):
        if self.active:raise RuntimeError('reset context already active')
        self.cfg=self.env.event_manager.get_term_cfg('reset_strike_route_joints')
        if tuple(tuple(row) for row in self.cfg.params['route_joint_positions'])!=self.table:raise ValueError('trained cfg table differs')
        self.original=self.cfg.func
        def reset(env,env_ids,*,route_joint_positions,asset_cfg):
            if env is not self.env or tuple(tuple(row) for row in route_joint_positions)!=self.table:
                raise ValueError('trained reset parameters changed')
            if not bool((env._strike_reference.route_signs()[env_ids]==0).all()):raise ValueError('intermediate start requires central route plumbing')
            table=(self.table[0],self.pose,self.table[2])
            return self.original(env,env_ids,route_joint_positions=table,asset_cfg=asset_cfg)
        self.wrapper=reset;self.cfg.func=reset;self.active=True
        return self

    def __exit__(self,*exc):
        drift=self.cfg.func is not self.wrapper
        self.cfg.func=self.original;self.active=False
        if drift:raise ValueError('reset callback changed during effective start context')
        return False

def verify_reset_geometry(env,pose):
    """Verify this frozen fixed-scene Z1 reset before the first action.

    Native site/reference positions already share the XML world frame. The scene
    terrain grid is layout metadata; this task's joint-only reset never translates
    fixed robot/nail roots by that grid. This helper is not for translated scenes.
    """
    import torch
    from mjlab.utils.lab_api.math import matrix_from_quat
    tracker=env._hammer_first_strike;data=tracker._robot.data;ref=env._strike_reference
    origins=env.scene.env_origins.double()
    head=data.site_pos_w[:,tracker._site_ids].squeeze(1).double()
    quat=data.site_quat_w[:,tracker._site_ids].squeeze(1).double()
    joint_ids=env.event_manager.get_term_cfg('reset_strike_route_joints').params['asset_cfg'].joint_ids
    q=data.joint_pos[:,joint_ids].double();v=data.joint_vel[:,joint_ids].double()
    def tensor(value):return torch.as_tensor(value,dtype=torch.float64,device=env.device)
    expected_head=tensor(pose['requested_head_m']);expected_q=tensor(pose['executed_joint_positions_rad'])
    expected_nail=tensor(pose['nail_m']);orientation=tensor(pose['requested_orientation_matrix'])
    rotation=matrix_from_quat(quat);relative=rotation@orientation.T
    skew=torch.stack((relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]),dim=-1)/2
    angle=torch.atan2(torch.linalg.vector_norm(skew,dim=-1),((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)).abs()
    residual=torch.linalg.vector_norm(head-expected_head,dim=-1)
    nail=ref._nail.double();target=ref._target.double();anchor=ref._head0.double()
    wanted_target=nail.clone();wanted_target[:,2]-=.02
    contact=(tracker._contact_sensor.data.found>0).any(dim=-1)
    finite=all(bool(torch.isfinite(x).all()) for x in (head,quat,q,v,nail,target,anchor,residual,angle))
    valid=(finite and head.shape==(env.num_envs,3) and q.shape==(env.num_envs,6)
        and bool((residual<=1e-6).all()) and bool((angle<=1e-5).all())
        and bool((q==expected_q).all()) and bool((v==0).all()) and not bool(contact.any())
        and bool((torch.linalg.vector_norm(nail-expected_nail,dim=-1)<=1e-6).all())
        and bool((torch.linalg.vector_norm(anchor-head,dim=-1)<=1e-6).all())
        and bool((torch.linalg.vector_norm(target-wanted_target,dim=-1)<=1e-6).all()))
    if not valid:raise ValueError('realized reset pose/reference geometry or velocity/contact differs')
    return [dict(world=i,head_world_m=head[i].tolist(),head_local_m=head[i].tolist(),
                 coordinate_frame='fixed_scene_native_world',scene_grid_origin_m=origins[i].tolist(),
                 scene_grid_origin_applied=False,
                 site_quaternion_wxyz=quat[i].tolist(),joint_positions_rad=q[i].tolist(),joint_velocities_rad_s=v[i].tolist(),
                 nail_world_m=nail[i].tolist(),reference_head0_w=anchor[i].tolist(),reference_target_w=target[i].tolist(),
                 position_residual_m=float(residual[i]),orientation_residual_rad=float(angle[i]),reset_contact=False)
            for i in range(env.num_envs)]
