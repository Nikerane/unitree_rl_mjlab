"""Pure, closed opt-in bindings for final100-update pilot evaluations.

This is not authority. Root's finite supervisor must bind a validated record to
its admitted policy map and actual training completion/source before issue.
No environment construction, model inference or checkpoint rewriting occurs.
"""
from __future__ import annotations
import hashlib
from collections import OrderedDict
import json
import math
from pathlib import Path

SETTINGS=tuple(f'credit{deadline}_{mode}'for deadline in (50,100,150,200)for mode in ('current','event'))
SEEDS=(2026102901,)
BASE='Unitree-Z1-Hammer-Quintic-ThreeStarts-SplineSpeedCurriculum-Observation-Age-LowNoise-DeferredCredit-Horizon-extended200'
SOURCE_SHA='eb504f5fb772932f11084e29fc88ef0943a2259756d096f1853ed9c1f48a6a06'
MANIFEST_SHA='7ca22bbd866fa069d8ffc520000b8e966bddf9693df70f03963aac7d6ce0fa79'
FIELDS={'schema','policy_id','setting','cell_index','credit_deadline_ms','critic_mode','training_seed','task','checkpoint_path','checkpoint_sha256','source_manifest_sha256','training_manifest_sha256','training_completion_sha256','final_iteration','completed_updates','common_step_counter','impact_speed_contract'}


def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(path):
    with Path(path).open('rb') as stream:
        d=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b''):d.update(chunk)
    return d.hexdigest()


def expected_marker(setting):
    need(type(setting)is str and setting in SETTINGS,'exact screen setting required')
    index=SETTINGS.index(setting);deadline=(50,100,150,200)[index//2];mode=('current','event')[index%2]
    return dict(schema_version='spline_credit_critic_v1',credit_deadline_ms=deadline,episode_contact_window_ms=200,critic_mode=mode,actor_width=43,critic_width=47 if mode=='event' else 43,critic_event_fields=['credit_contact_seen','credit_contact_age','credit_unpaid','credit_time_to_payout']if mode=='event'else[],actor_inputs_unchanged=True)

def validate_record(record):
    need(type(record)is dict and set(record)==FIELDS,'closed17-field screen final record required')
    need(record['schema']=='z1_credit_screen_final_policy_v1','explicit screen final schema required')
    marker=expected_marker(record['setting']);setting=record['setting'];index=SETTINGS.index(setting)
    need(type(record['cell_index'])is int and record['cell_index']==index and type(record['credit_deadline_ms'])is int and record['credit_deadline_ms']==marker['credit_deadline_ms']and record['critic_mode']==marker['critic_mode'],'screen recipe/cell differs')
    need(type(record['training_seed'])is int and record['training_seed']==2026102901,'exact screen training seed required')
    need(record['policy_id']==f'{setting}-seed2026102901'and record['task']==BASE+f"-Credit{marker['credit_deadline_ms']}-Critic{marker['critic_mode'].title()}",'screen policy/task differs')
    for key,value in (('final_iteration',99),('completed_updates',100),('common_step_counter',2400)):
        need(type(record[key])is int and record[key]==value,'exact final clock required: '+key)
    path=Path(record['checkpoint_path']);need(path.is_absolute()and path.name=='model_99.pt','absolute final model99 required')
    for key in ('checkpoint_sha256','source_manifest_sha256','training_manifest_sha256','training_completion_sha256'):
        value=record[key];need(type(value)is str and len(value)==64 and all(c in '0123456789abcdef'for c in value),'SHA256 required: '+key)
    need(record['source_manifest_sha256']==SOURCE_SHA and record['training_manifest_sha256']==MANIFEST_SHA,'screen source-map/manifest-file domains differ')
    contract=record['impact_speed_contract'];need(type(contract)is dict and contract.get('schema_version')=='impact_speed_v1','full saved impact contract required')
    need(canonical(contract.get('credit_critic_experiment'))==canonical(marker),'exact deadline/critic marker required')
    legacy=dict(schema_version='spline_learning_v1',variant='N',startup_age=dict(source='sampled_pd500_physics_substeps_since_own_reset',normalization_s=.1,clip=[0.,1.],observation_index=40),initial_exploration_std=[.3]*12+[1.],startup_target_ramp_s=0.)
    need(canonical(contract.get('spline_learning_variant'))==canonical(legacy),'exact unchanged legacyN required')
    need('spline_exploration_pilot'not in contract and 'pilot_physical_treatment'not in contract,'screen must not impersonate pilot metadata')
    need(contract.get('action_dim')==13 and contract.get('action_type')=='joint_position_variable_impedance_sampled_pd'and contract.get('controller_identity')=='sampled_vic500_v1','screen variable13 controller identity differs')
    need(contract.get('observation_widths')==dict(actor=43,critic=marker['critic_width']),'exact actor43/critic43or47 required')
    need(contract.get('physics_dt_s')==.00025 and contract.get('control_decimation')==80,'physical timing differs')
    need(contract.get('speed_bounds_mps')==[1.,1.5]and contract.get('speed_mode')=='policy','native learned selector calibration required')
    canonical(contract);return record


def finite_tree(value):
    if hasattr(value,'detach'):
        need(bool(value.detach().isfinite().all().item()),'nonfinite checkpoint tensor')
    elif isinstance(value,float):need(math.isfinite(value),'nonfinite checkpoint scalar')
    elif isinstance(value,dict):
        for item in value.values():finite_tree(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite_tree(item)


def expected_state_shapes(width,actor=False):
    shapes={f'obs_normalizer.physical.{key}':(1,40)for key in ('_mean','_var','_std')}
    shapes['obs_normalizer.physical.count']=()
    for layer,left,right in ((0,width,256),(2,256,128),(4,128,64),(6,64,13 if actor else 1)):
        shapes[f'mlp.{layer}.weight']=(right,left);shapes[f'mlp.{layer}.bias']=(right,)
    if actor:shapes['distribution.std_param']=(13,)
    return shapes

def validate_state_shapes(state,width,actor=False):
    shapes=expected_state_shapes(width,actor)
    need(type(state)in (dict,OrderedDict) and set(state)==set(shapes),'exact actor/critic saved tensor keys required')
    for key,shape in shapes.items():
        tensor=state[key]
        need(hasattr(tensor,'detach')and tuple(getattr(tensor,'shape',()))==shape,'actual saved tensor shape differs: '+key)


def validate_payload(payload,record):
    validate_record(record);need(type(payload)is dict,'checkpoint dictionary required')
    need(type(payload.get('iter'))is int and payload['iter']==99,'actual checkpoint final iteration differs')
    infos=payload.get('infos');need(type(infos)is dict,'actual checkpoint infos required')
    counter=(infos.get('env_state')or{}).get('common_step_counter')
    need(type(counter)is int and counter==2400,'actual checkpoint counter differs')
    need(infos.get('cat_pressure_treatment')is None,'unapproved pressure scheduler checkpoint')
    need(payload.get('actor_state_dict') and payload.get('critic_state_dict'),'finite actor/critic checkpoint states required')
    need(canonical(infos.get('impact_speed_contract'))==canonical(record['impact_speed_contract']),'full saved contract differs, including physical plant')
    validate_state_shapes(payload['actor_state_dict'],43,actor=True)
    validate_state_shapes(payload['critic_state_dict'],expected_marker(record['setting'])['critic_width'])
    finite_tree(payload)
    return dict(status='PASS_FINAL_SCREEN_PAYLOAD_ONLY_NOT_AUTHORITY',iteration=99,common_step_counter=2400,checkpoint_sha256=record['checkpoint_sha256'])


def gain_mode(record):validate_record(record);return 'variable'


def evaluator_arguments(record,*,output,seed):
    validate_record(record);need(type(seed)is int and 0<seed<2**31,'fresh explicit evaluation seed required')
    need(Path(output).is_absolute(),'absolute output required')
    return ['--task',record['task'],'--checkpoint',record['checkpoint_path'],'--speed-bounds','1.0','1.5','--speed-cost','0.8493676','--seed',str(seed),'--num-envs','32','--mode','policy','--output',str(output),'--device','cpu','--sampled-pd500','--sampled-pd500-gain-mode',gain_mode(record),'--measurement-protocol','native']
