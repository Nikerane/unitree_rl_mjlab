"""Truthful A/C/B metadata and 99/499 clocks, independent of legacy admission.

The complete contract comes from a verified frozen producer initial-state seal;
this module checks family/safety identity, then saved/live paths require exact
full contract equality. It does not synthesize scientific plant metadata.
"""
from collections.abc import Mapping
import math
from pathlib import Path
import re
from . import release as a
from .files import canonical, digest, need, read_json, sha_file

IDENTITIES_SHA = '90108d54324e318bf9527c6888e406d432d17414bb3a2fc9c7fa6d6534a29b0c'
FIELDS = {'schema','policy_id','case_index','training_cell_index','training_slot','training_family','training_cell',
          'checkpoint_label','completed_updates','common_step_counter','task','checkpoint_path','checkpoint_sha256',
          'source_manifest_sha256','training_manifest_sha256','training_completion_sha256','impact_speed_contract'}
BASE = 'Unitree-Z1-Hammer-Quintic-ThreeStarts-SplineSpeedCurriculum-Observation-Age-LowNoise-DeferredCredit-Horizon-extended200'


def expected_marker(case_index):
    descriptor = a.case_for_index(case_index)
    path = Path(__file__).with_name('identities.json')
    need(sha_file(path) == IDENTITIES_SHA, 'closed family identities differ')
    return read_json(path)[descriptor['training_cell_index']]


def expected_task(descriptor):
    cell = descriptor['training_cell']
    if descriptor['training_family'] == 'split_credit':
        return 'Unitree-Z1-Hammer-Spline-SplitCredit50-200-Current-'+cell['setting']
    return BASE+f"-Credit{cell['credit_deadline_ms']}-Critic{cell['critic_mode'].title()}-Pilot-"+cell['setting']


def record_for_binding(binding, source_manifest_sha256):
    d = binding['descriptor']; a.need(canonical(d) == canonical(a.case_for_index(d['index'])), 'bound descriptor differs')
    record = dict(schema='z1_thirty_evaluation_policy023',policy_id=f"cell-{d['training_cell_index']:02d}-model-{d['checkpoint_label']}",
        case_index=d['index'],training_cell_index=d['training_cell_index'],training_slot=d['training_slot'],
        training_family=d['training_family'],training_cell=d['training_cell'],checkpoint_label=d['checkpoint_label'],
        completed_updates=d['updates_completed'],common_step_counter=d['expected_common_step_counter'],task=expected_task(d),
        checkpoint_path=binding['checkpoint']['path'],checkpoint_sha256=binding['checkpoint']['sha256'],
        source_manifest_sha256=source_manifest_sha256,training_manifest_sha256=a.PRODUCER_SHA,
        training_completion_sha256=binding['completion']['sha256'],impact_speed_contract=binding['impact_speed_contract'])
    return validate_record(record)


def validate_record(record):
    need(type(record) is dict and set(record) == FIELDS, 'closed new policy fields required')
    need(record['schema'] == 'z1_thirty_evaluation_policy023', 'truthful training30 evaluation schema required')
    d = a.case_for_index(record['case_index']); cell = d['training_cell']; marker = expected_marker(d['index'])
    exact = dict(training_cell_index=d['training_cell_index'],training_slot=d['training_slot'],training_family=d['training_family'],
                 training_cell=cell,checkpoint_label=d['checkpoint_label'],completed_updates=d['updates_completed'],
                 common_step_counter=d['expected_common_step_counter'],checkpoint_path=d['checkpoint_path'],
                 policy_id=f"cell-{d['training_cell_index']:02d}-model-{d['checkpoint_label']}",task=expected_task(d),
                 training_manifest_sha256=a.PRODUCER_SHA)
    need(all(canonical(record[key]) == canonical(value) for key,value in exact.items()), 'policy descriptor/clock/path differs')
    for key in ('checkpoint_sha256','source_manifest_sha256','training_manifest_sha256','training_completion_sha256'):
        need(type(record[key]) is str and re.fullmatch('[0-9a-f]{64}',record[key]), 'actual SHA256 required: '+key)
    contract = record['impact_speed_contract']; need(type(contract) is dict, 'full saved contract required')
    split = d['training_family'] == 'split_credit'
    marker_key = 'spline_split_credit' if split else 'spline_exploration_pilot'
    physical_key = 'split_physical_treatment' if split else 'pilot_physical_treatment'
    need(canonical(contract.get(marker_key)) == canonical(marker), 'exact selected family marker differs')
    forbidden = ('spline_exploration_pilot','pilot_physical_treatment','credit_critic_experiment','spline_learning_variant') if split else ('spline_split_credit','split_physical_treatment','spline_learning_variant')
    need(not any(key in contract for key in forbidden), 'mixed or legacy family metadata forbidden')
    if not split:
        need(canonical(contract.get('credit_critic_experiment')) == canonical(marker['credit_critic_experiment']), 'pilot critic/deadline anchor differs')
    fields = dict(schema_version='impact_speed_v1',action_type='joint_position_variable_impedance_sampled_pd',
                  controller_identity='sampled_vic500_v1',action_dim=13,physics_dt_s=.00025,control_decimation=80,
                  speed_bounds_mps=[1.,1.5],speed_mode='policy',observation_widths={'actor':43,'critic':marker['critic_width']})
    need(all(canonical(contract.get(k)) == canonical(v) for k,v in fields.items()), 'physical timing/action/observation contract differs')
    controller = contract.get('sampled_pd'); need(type(controller) is dict, 'sampled controller required')
    expected = dict(version='variable_impedance_pd500_v1',gain_mode='variable',mapping_family='author_v1_exponential',C=1.25,
                    p_bounds=[-1.,1.],physics_dt_s=.00025,pd_dt_s=.002,policy_dt_s=.02,kp=marker['fixed_nominal_kp'],kd=marker['fixed_nominal_kd'])
    need(all(canonical(controller.get(k)) == canonical(v) for k,v in expected.items())
         and 'ignored_action_indices' not in controller and 'startup_target_ramp' not in controller, 'true VIC nominal gains/mapping/timing differ')
    physical = contract.get(physical_key)
    need(type(physical) is dict and canonical(physical) == canonical(contract.get('sampled_vic_treatment'))
         and all(type(physical.get(k)) is dict and physical[k] for k in ('reference','guidance','load')), 'full physical treatment required')
    safety = dict(use_vel=True,use_impulse=True,imp_limit=[.369,.246,.738,.369,.246,.0164],imp_max_p=.2,imp_seed=.001,
                  limit=3.1415,max_p=.5,min_p=0.,tau=.95,vel_detection='substep')
    need(canonical(contract.get('safety',{}).get('cat_soft')) == canonical(safety), 'unchanged joint velocity/impulse safety required')
    canonical(record)
    return record


def finite_tree(value):
    if hasattr(value,'detach'):
        need(bool(value.detach().isfinite().all().item()), 'nonfinite checkpoint tensor')
    elif isinstance(value,float):
        need(math.isfinite(value), 'nonfinite checkpoint scalar')
    elif isinstance(value,Mapping):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value,(tuple,list)):
        for item in value:
            finite_tree(item)


def expected_state_shapes(width, actor=False):
    need(type(width) is int and width in (43,47) and (not actor or width == 43), 'exact actor43/current43/event47 width required')
    shapes = {f'obs_normalizer.physical.{key}':(1,40) for key in ('_mean','_var','_std')}
    shapes['obs_normalizer.physical.count'] = ()
    for layer,left,right in ((0,width,256),(2,256,128),(4,128,64),(6,64,13 if actor else 1)):
        shapes[f'mlp.{layer}.weight'] = (right,left); shapes[f'mlp.{layer}.bias'] = (right,)
    if actor:
        shapes['distribution.std_param'] = (13,)
    return shapes


def validate_state_shapes(state, width, actor=False):
    shapes = expected_state_shapes(width,actor)
    need(isinstance(state,Mapping) and set(state) == set(shapes), 'exact actor/critic tensor keys required')
    for key,shape in shapes.items():
        tensor = state[key]
        need(hasattr(tensor,'detach') and tuple(getattr(tensor,'shape',())) == shape, 'saved tensor shape differs: '+key)
    finite_tree(state)


def validate_payload(payload, record):
    validate_record(record)
    need(isinstance(payload,Mapping), 'checkpoint Mapping required')
    need(type(payload.get('iter')) is int and payload['iter'] == record['checkpoint_label'], 'actual checkpoint iteration differs')
    infos = payload.get('infos'); need(isinstance(infos,Mapping), 'checkpoint infos Mapping required')
    state = infos.get('env_state'); need(isinstance(state,Mapping), 'checkpoint env state required')
    need(type(state.get('common_step_counter')) is int and state['common_step_counter'] == record['common_step_counter'], 'loaded checkpoint clock differs')
    need(infos.get('cat_pressure_treatment') is None, 'pressure scheduler forbidden')
    need(canonical(infos.get('impact_speed_contract')) == canonical(record['impact_speed_contract']), 'full saved contract differs from producer')
    validate_state_shapes(payload.get('actor_state_dict'),43,True)
    validate_state_shapes(payload.get('critic_state_dict'),record['impact_speed_contract']['observation_widths']['critic'])
    finite_tree(payload)
    return dict(status='PASS_PAYLOAD_ONLY_NOT_AUTHORITY',iteration=record['checkpoint_label'],
                common_step_counter=record['common_step_counter'],checkpoint_sha256=record['checkpoint_sha256'])


def evaluator_arguments(record, *, output, seed):
    validate_record(record)
    need(type(seed) is int and seed == a.EVALUATION_SEED, 'same prospective bank evaluation seed required')
    need(Path(output).is_absolute(), 'absolute output required')
    return ['--task',record['task'],'--checkpoint',record['checkpoint_path'],'--speed-bounds','1.0','1.5',
            '--speed-cost','0.8493676','--seed',str(seed),'--num-envs','32','--mode','policy','--output',str(output),
            '--device','cpu','--sampled-pd500','--sampled-pd500-gain-mode','variable','--measurement-protocol','native']
