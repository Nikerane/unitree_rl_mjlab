"""Prospective matched bank32 and event-only reset adapter; no physics calls.

Bank construction is stdlib pure. Runtime verification reads tensors; it never
forwards physics, computes observations, calls a policy or replaces env reset.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

TAG = 2026103001
ROUTES = (-1,)*11 + (0,)*10 + (1,)*11
CONTROLS = (0,11,21)
NAMES = tuple(f'joint{i}' for i in range(1,7))
REFERENCE = dict(path_shape='quintic',overshoot=.02,followthrough_mode='vertical',
                 horizontal_detour_m=0.,axis_tol=.05,descent_speed=.05)
LOAD_FIELDS = {'body_mass_kg':'body_mass','body_inertia_kg_m2':'body_inertia',
    'body_gravcomp':'body_gravcomp','dof_damping':'dof_damping',
    'dof_frictionloss_nm':'dof_frictionloss','jnt_stiffness':'jnt_stiffness'}
_ACTIVE = {}


def need(ok,message):
    if not ok: raise ValueError(message)


def encoded(value):
    try: return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    except (ValueError,TypeError): raise ValueError('nonfinite or non-JSON bank data') from None


def digest(value): return hashlib.sha256(encoded(value)).hexdigest()
def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _f32(value): return struct.unpack('<f',struct.pack('<f',value))[0]


def _vector(values,n):
    need(type(values) in (list,tuple) and len(values)==n,'vector shape differs')
    need(all(type(v) in (int,float) and math.isfinite(v) for v in values),'finite vector required')
    return [float(v) for v in values]


def build_bank(trained_table,joint_limits,reference,plant,assets_sha256,source_sha256):
    """Three native route controls plus29 hash-derived perturbations; no outcome input.

    Integer nanoradian deltas are uniformly mapped from SHA256(tag,row,joint)
    into[-499000,499000]. The1microradian margin keeps executed float32 deltas
    within0.0005rad after rounding. Nothing is selected by FK or policy outcome.
    """
    need(type(trained_table) in (list,tuple) and len(trained_table)==3,'trained table3x6 required')
    table=[_vector(row,6) for row in trained_table]
    need(type(joint_limits) in (list,tuple) and len(joint_limits)==6,'six joint limits required')
    limits=[_vector(pair,2) for pair in joint_limits]
    need(all(lo<hi for lo,hi in limits),'joint bounds inverted')
    need(encoded(reference)==encoded(REFERENCE),'unchanged reference recipe required')
    need(type(plant) is dict and set(plant)==set(LOAD_FIELDS),'complete physical load fields required')
    encoded(plant)
    for mapping in (assets_sha256,source_sha256):
        need(type(mapping) is dict and mapping and all(type(k) is str and type(h) is str
            and len(h)==64 and all(c in '0123456789abcdef'for c in h) for k,h in mapping.items()),'SHA identity map required')
    nail=dict(entity='nail_block',joint='nail_slide',qpos_m=0.,qvel_m_s=0.,top_world_m=[.5,0.,.102])
    rows=[]
    for i in range(32):
        nominal=[_f32(v)for v in table[ROUTES[i]+1]]
        delta=[0.]*6 if i in CONTROLS else [(int.from_bytes(hashlib.sha256(f'{TAG}:{i}:{j}'.encode()).digest()[:8],'big')%998001-499000)*1e-9 for j in range(6)]
        q=[_f32(v+d)for v,d in zip(nominal,delta)]
        need(all(lo<=v<=hi for v,(lo,hi)in zip(q,limits)),'bank joint limit violation')
        need(max(abs(v-n)for v,n in zip(q,nominal))<=.0005,'bank perturbation radius exceeded')
        row=dict(world=i,kind='reused_nominal_control'if i in CONTROLS else 'matched_small_perturbation',
            qpos_rad=q,qvel_rad_s=[0.]*6,route_sign=ROUTES[i],reference_sha256=digest(reference),nail_sha256=digest(nail))
        row['state_sha256']=digest(row);rows.append(row)
    need(len({tuple(r['qpos_rad'])for r in rows})==32,'float32 bank states not unique')
    return dict(schema='pilot_balanced_native_reset_bank32_v2',tag=TAG,worlds=32,
        interpretation='MATCHED_NOMINAL_PERTURBATION_BANK_NOT_32_IID_HELD_OUT',nominal_geometry_intentionally_reused=True,
        selection='SHA256_TAG_ROW_JOINT_NANORAD_NO_PERFORMANCE_SELECTION',max_linf_rad=.0005,
        route_counts={'-1':11,'0':10,'1':11},nominal_control_worlds=list(CONTROLS),
        reference_anchor_rule='NATIVE_REANCHOR_FROM_REALIZED_RESET_HEAD_NO_CARTESIAN_POSE_CLAIM',
        joint_names=list(NAMES),trained_route_table_rad=table,joint_limits_rad=limits,
        reference=dict(reference),nail=nail,plant=plant,assets_sha256=dict(assets_sha256),source_sha256=dict(source_sha256),rows=rows)


def validate_bank(value):
    need(type(value) is dict,'bank dictionary required')
    try:
        wanted=build_bank(value['trained_route_table_rad'],value['joint_limits_rad'],value['reference'],
                          value['plant'],value['assets_sha256'],value['source_sha256'])
        need(encoded(value)==encoded(wanted),'bank schema/hash/selection/state drift')
    except (KeyError,OverflowError): raise ValueError('malformed bank') from None
    return wanted


def load_bank(path,expected_sha256):
    body=Path(path).read_bytes()
    need(type(expected_sha256) is str and hashlib.sha256(body).hexdigest()==expected_sha256,'bank file SHA mismatch')
    def pairs(items):
        result={}
        for k,v in items:
            need(k not in result,'duplicate bank key');result[k]=v
        return result
    try: value=json.loads(body,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite bank'))
    except (json.JSONDecodeError,UnicodeDecodeError): raise ValueError('invalid bank JSON') from None
    return validate_bank(value)


def _ids(env,ids):
    import torch
    if ids is None: ids=torch.arange(env.num_envs,device=env.device)
    need(isinstance(ids,torch.Tensor) and ids.ndim==1 and ids.dtype in (torch.int32,torch.int64),'integer reset IDs required')
    values=ids.detach().cpu().tolist()
    need(len(values)>0 and len(values)==len(set(values)) and all(0<=i<32 for i in values),'duplicate/out-of-range reset IDs')
    return ids,values


class BankResetContext:
    def __init__(self,env,bank_path,expected_sha256):
        self.env,self.path,self.sha=env,Path(bank_path),expected_sha256
        self.bank=load_bank(self.path,self.sha)
        need(type(env.num_envs) is int and env.num_envs==32,'exact32-world bank required')
        self.active=False;self.sampled_worlds=set();self.attempted_worlds=set();self.applied_worlds=set()

    def _owner_ok(self): need(self.env._reset_idx==self.reset_owner,'controller reset owner changed')

    def __enter__(self):
        need(not self.active and id(self.env) not in _ACTIVE,'bank reset context already active')
        self.cfg=self.env.event_manager.get_term_cfg('reset_strike_route_joints')
        self.original=self.cfg.func;self.reset_owner=self.env._reset_idx
        self.route_cfg=self.env.event_manager.get_term_cfg('sample_strike_route_signs')
        self.original_route=self.route_cfg.func;self.route_params=dict(self.route_cfg.params)
        need(self.route_params.get('fixed_route_sign') is None,'native three-route sampler required')
        need(all(self.route_params.get(k)==REFERENCE[k] for k in ('path_shape','overshoot','followthrough_mode','horizontal_detour_m')),'native sampler reference recipe differs')
        table=self.bank['trained_route_table_rad']
        need(encoded(self.cfg.params['route_joint_positions'])==encoded(table),'trained route table differs')
        asset=self.cfg.params['asset_cfg']
        need(tuple(asset.joint_names)==NAMES and list(asset.joint_ids)==list(range(6)),'canonical ordered six arm joints required')
        def sample(env,env_ids,**params):
            import torch
            self._owner_ok();need(env is self.env and encoded(params)==encoded(self.route_params),'native sampler parameters drift')
            ids,values=_ids(env,env_ids)
            need(not self.sampled_worlds.intersection(values),'bank world sampler already attempted; no retry')
            self.sampled_worlds.update(values)
            result=self.original_route(env,env_ids,**params)
            ref=env._strike_reference;signs=ref.route_signs()
            signs[ids]=torch.tensor([self.bank['rows'][i]['route_sign']for i in values],dtype=signs.dtype,device=signs.device)
            ref.set_route_signs(signs);self._owner_ok()
            return result
        def reset(env,env_ids,*,route_joint_positions,asset_cfg):
            import torch
            self._owner_ok()
            need(env is self.env and asset_cfg is asset and encoded(route_joint_positions)==encoded(table),'native reset parameters drift')
            ids,values=_ids(env,env_ids)
            need(not self.attempted_worlds.intersection(values),'bank world reset already attempted; no retry')
            signs=env._strike_reference.route_signs()[ids]
            wanted=signs.new_tensor([self.bank['rows'][i]['route_sign']for i in values])
            need(set(values)<=self.sampled_worlds and torch.equal(signs,wanted),'native per-world route sampler must run before bank reset')
            self.attempted_worlds.update(values)
            result=self.original(env,env_ids,route_joint_positions=route_joint_positions,asset_cfg=asset_cfg)
            # Cached3-pose write is complete. Overlay each immutable world row
            # now, before native forward, reference anchoring and observations.
            q=torch.tensor([self.bank['rows'][i]['qpos_rad']for i in values],dtype=torch.float32,device=env.device).reshape(-1,6)
            v=torch.zeros_like(q)
            env.scene[asset_cfg.name].write_joint_state_to_sim(q,v,joint_ids=asset_cfg.joint_ids,env_ids=ids)
            self.applied_worlds.update(values);self._owner_ok()
            return result
        self.route_wrapper=sample;self.wrapper=reset
        self.route_cfg.func=sample;self.cfg.func=reset;self.active=True;_ACTIVE[id(self.env)]=self
        return self

    def __exit__(self,*exc):
        drift=self.cfg.func is not self.wrapper or self.route_cfg.func is not self.route_wrapper
        self.route_cfg.func=self.original_route;self.cfg.func=self.original;self.active=False;_ACTIVE.pop(id(self.env),None)
        self._owner_ok();need(not drift,'native event callback replaced during bank context')
        return False


def create_context(env,bank_path,expected_sha256):
    return BankResetContext(env,bank_path,expected_sha256)


def verify_reset(env,obs,bank_path,expected_sha256,*,env_ids=None):
    """Read actual post-native-reset tensors before first action; never recompute.

    Intended qpos/qvel are exactfloat32. Nail/head/reference geometry comparisons
    use1micrometre native floating tolerance, not claimed prequalified FK poses.
    """
    import torch
    bank=load_bank(bank_path,expected_sha256);ids,worlds=_ids(env,env_ids)
    ctx=_ACTIVE.get(id(env));need(ctx is not None and ctx.sha==expected_sha256,'matching active bank context required')
    ctx._owner_ok();need(set(worlds)<=ctx.applied_worlds,'world bank write not observed')
    def tensor(value,shape):
        need(isinstance(value,torch.Tensor) and tuple(value.shape)==shape and bool(torch.isfinite(value).all()),'finite realized tensor shape required')
        return value.detach()
    tr=env._hammer_first_strike;ref=env._strike_reference
    q=tensor(tr._robot.data.joint_pos[:,ctx.cfg.params['asset_cfg'].joint_ids],(32,6))[ids]
    v=tensor(tr._robot.data.joint_vel[:,ctx.cfg.params['asset_cfg'].joint_ids],(32,6))[ids]
    expected=torch.tensor([bank['rows'][i]['qpos_rad']for i in worlds],dtype=q.dtype,device=q.device).reshape(-1,6)
    need(torch.equal(q,expected) and bool((v==0).all()),'actual bank joint state differs')
    routes=[bank['rows'][i]['route_sign']for i in worlds]
    need(torch.equal(ref.route_signs()[ids],ref.route_signs().new_tensor(routes)),'actual route differs')
    for name,value in bank['reference'].items(): need(getattr(ref,name)==value,'actual reference recipe changed')
    head=tensor(tr._robot.data.site_pos_w[:,tr._site_ids].squeeze(1),(32,3))[ids]
    quat=tensor(tr._robot.data.site_quat_w[:,tr._site_ids].squeeze(1),(32,4))[ids]
    nail=tensor(tr._nail.data.site_pos_w[:,tr._nail_site_ids].squeeze(1),(32,3))[ids]
    nq=tensor(tr._nail.data.joint_pos[:,tr._joint_ids].reshape(32,1),(32,1))[ids]
    nv=tensor(tr._nail.data.joint_vel[:,tr._joint_ids].reshape(32,1),(32,1))[ids]
    need(bool((nq==0).all())and bool((nv==0).all()),'nail reset state changed')
    wanted_nail=nail.new_tensor(bank['nail']['top_world_m'])
    close=lambda x,y:bool((torch.abs(x-y)<=1e-6).all())
    need(close(nail,wanted_nail),'nail world geometry changed')
    anchor=tensor(ref._head0,(32,3))[ids];rnail=tensor(ref._nail,(32,3))[ids];target=tensor(ref._target,(32,3))[ids]
    wanted_target=nail.clone();wanted_target[:,2]-=bank['reference']['overshoot']
    need(bool(ref._anchored[ids].all()) and close(anchor,head) and close(rnail,nail) and close(target,wanted_target)
         and bool((head[:,2]>nail[:,2]).all()),'native reference not reanchored to realized bank geometry')
    contact=tensor(tr._contact_sensor.data.found,(32,tr._contact_sensor.data.found.shape[1]))[ids]
    need(not bool((contact>0).any()),'bank has reset contact; geometry not qualified')
    load={}
    for key,field in LOAD_FIELDS.items():
        value=getattr(env.sim.model,field).detach()
        need(isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[0] in (1,32)
             and bool(torch.isfinite(value).all()) and torch.equal(value,value[:1].expand_as(value)),'physical plant load heterogeneous/nonfinite')
        load[key]=value[0].detach().cpu().tolist()
    need(digest(load)==digest(bank['plant']),'physical plant load differs')
    cpu=lambda t:t.detach().cpu().tolist()
    observations={}
    for key,value in obs.items():
        need(type(key) is str and isinstance(value,torch.Tensor) and value.ndim>=1 and value.shape[0]==32
             and bool(torch.isfinite(value).all()),'actual observation tensor invalid')
        observations[key]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    realized=dict(qpos_rad=cpu(q),qvel_rad_s=cpu(v),route_signs=routes,nail_qpos_m=cpu(nq),
        nail_qvel_m_s=cpu(nv),nail_world_m=cpu(nail),head_world_m=cpu(head),head_quat_wxyz=cpu(quat),
        reference_head0_world_m=cpu(anchor),reference_target_world_m=cpu(target))
    return dict(schema='pilot_reset_bank32_realized_v1',bank_sha256=expected_sha256,world_ids=worlds,
        interpretation=bank['interpretation'],actual_qpos_sha256=digest(cpu(q)),actual_qvel_sha256=digest(cpu(v)),
        actual_state_sha256=digest(realized),actual_plant_sha256=digest(load),
        actual_route_sha256=digest(routes),actual_reference_recipe={k:getattr(ref,k)for k in bank['reference']},
        actual_reference_sha256=digest(dict(recipe={k:getattr(ref,k)for k in bank['reference']},head0_world_m=cpu(anchor),target_world_m=cpu(target))),
        actual_nail_sha256=digest(dict(qpos_m=cpu(nq),qvel_m_s=cpu(nv),world_m=cpu(nail))),
        observations=observations,realized=realized,
        verification='POST_NATIVE_RESET_PRE_ACTION_NO_FORWARD_OR_POLICY_CALL',structural_FK_contact_qualification='ROOT_FINITE_ENGINEERING_GATE_STILL_REQUIRED')
