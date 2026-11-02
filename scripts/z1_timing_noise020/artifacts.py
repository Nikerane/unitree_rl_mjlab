"""File-only completion checks. Never load models or import ML."""
import hashlib
import importlib
import math
from pathlib import Path
import re
from . import admission as a,compiler
from scripts.z1_exploration_pilot_campaign import (
    expected_pilot_identity, COMMON_INITIAL_HASHES, _EPISODE_KEYS,
    need, encoded, read_file, _pairs, validate_trace)
import json

def _number(value,low=None,high=None,integer=False):
    need(type(value) in ((int,) if integer else (int,float)) and math.isfinite(value),
         'finite population numeric domain required')
    need((low is None or value>=low) and (high is None or value<=high),'population numeric bound differs')
    return value

def _episode_counts(row,previous=None):
    need(type(row) is dict and set(row)==set(_EPISODE_KEYS),'complete population episode counters required')
    for key in _EPISODE_KEYS:
        _number(row[key],0,1024+1024*12000,integer=True)
        if previous is not None:need(row[key]>=previous[key],'population cumulative counter decreased')
    ncontact,nmiss,external=(row[k] for k in ('native_contact_episodes','native_no_contact_episodes','external_reset_episodes'))
    need(row['all_episode_starts']==1024+ncontact+nmiss+external,'population episode-start conservation differs')
    need(row['contact_resets_before_200ms']+row['contact_policy_envelopes_reached_200ms']==ncontact,
         'population contact-window conservation differs')
    need(row['native_completed_productive_episodes']<=ncontact and row['native_timeout_episodes']<=ncontact+nmiss
         and row['external_contact_censored_episodes']<=external
         and row['external_reset_observed_productive_episodes']<=external,'population episode subcounts differ')
    need(row['observed_contact_onsets']<=row['observed_episode_starts']<=row['all_episode_starts']
         and row['observed_productive_contact_episodes']<=row['observed_contact_onsets'],
         'population observed-episode denominator differs')

def validate_population(rows,*,threshold):
    _number(threshold,0);need(threshold>0,'positive live population threshold required')
    need(type(rows) is list and len(rows)==501,'500 population updates plus final row required')
    previous=None
    vector_keys=('peak_rad_s_max_per_joint','peak_rad_s_mean_per_joint','raw_margin_max_per_joint',
        'exceedance_policy_steps_per_joint','exceedance_policy_step_fraction_per_joint',
        'probability_mean_per_joint','probability_max_per_joint','ema_min_per_joint','ema_max_per_joint')
    for update,row in enumerate(rows[:-1]):
        need(type(row) is dict and row.get('schema')=='z1_velocity_population_v1' and row.get('kind')=='update'
             and type(row.get('update')) is int and row['update']==update,'population update order/schema differs')
        need(row.get('scope')=='ALL_WORLDS_POLICY_STEP_ENVELOPE'
             and type(row.get('worlds')) is int and row['worlds']==1024
             and type(row.get('world_transitions')) is int and row['world_transitions']==24576
             and type(row.get('common_step_counter')) is int and row['common_step_counter']==24*(update+1),
             'population world/transition clock differs')
        need('exact_200ms_contact_window_safety' in row and row['exact_200ms_contact_window_safety'] is None
             and row.get('envelope_caveat')=='ONSET_STEP_MAY_INCLUDE_PRECONTACT_FINAL_STEP_MAY_EXCEED_200MS',
             'population must preserve unknown exact-window safety')
        velocity=row.get('velocity')
        need(type(velocity) is dict and set(velocity)==set(vector_keys)|{'threshold_rad_s','joint_names'},
             'complete population velocity schema required')
        _number(velocity['threshold_rad_s'],0)
        need(velocity['threshold_rad_s']==threshold
             and velocity['joint_names']==[f'joint{i}' for i in range(1,7)],'population threshold/joint identity differs')
        for key in vector_keys:
            values=velocity[key];need(type(values) is list and len(values)==6,'population six-joint vector required')
            for value in values:
                low=None if key=='raw_margin_max_per_joint' else 0
                high=1 if key in ('probability_mean_per_joint','probability_max_per_joint','exceedance_policy_step_fraction_per_joint') else (24576 if key=='exceedance_policy_steps_per_joint' else None)
                _number(value,low,high,integer=key=='exceedance_policy_steps_per_joint')
        for j in range(6):
            at=lambda key:velocity[key][j]
            need(at('peak_rad_s_mean_per_joint')<=at('peak_rad_s_max_per_joint')+1e-5
                 and at('raw_margin_max_per_joint')+threshold<=at('peak_rad_s_max_per_joint')+1e-4,
                 'population raw peak/margin domains differ')
            need(math.isclose(at('exceedance_policy_step_fraction_per_joint'),at('exceedance_policy_steps_per_joint')/24576,
                              rel_tol=1e-5,abs_tol=1e-7),'population exceedance fraction denominator differs')
            need(at('probability_mean_per_joint')<=at('probability_max_per_joint')+1e-6
                 and at('ema_min_per_joint')<=at('ema_max_per_joint'),'population probability/EMA domains differ')
        winners=row.get('pressure_winner_policy_steps')
        need(type(winners) is dict and set(winners)=={'none','velocity','impulse','tie'},'population pressure winners required')
        for value in winners.values():_number(value,0,24576,integer=True)
        need(sum(winners.values())==24576,'population pressure winner conservation differs')
        counts=row.get('episodes_cumulative');_episode_counts(counts,previous);previous=counts
        encoded(row)  # Reject nonfinite extra fields as well.
    final=rows[-1]
    need(type(final) is dict and final.get('schema')=='z1_velocity_population_v1' and final.get('kind')=='final'
         and final.get('scope')=='ALL_WORLDS_POLICY_STEP_ENVELOPE'
         and final.get('status')=='COMPLETE_OBSERVATION_NOT_SAFETY_CERTIFICATION'
         and type(final.get('updates')) is int and final['updates']==500
         and type(final.get('partial_policy_steps')) is int and final['partial_policy_steps']==0,
         'complete population final envelope required')
    need('exact_200ms_contact_window_safety' in final and final['exact_200ms_contact_window_safety'] is None,
         'final exact-window safety must remain unknown')
    counts={k:final.get(k) for k in _EPISODE_KEYS};_episode_counts(counts,previous)
    need(counts==previous,'final population counts changed after final update')
    for key in ('active_at_stop_right_censored','active_contact_at_stop','unobserved_episode_starts_at_stop','active_observed_productive_at_stop'):
        _number(final.get(key),0,1024,integer=True)
    need(final['active_at_stop_right_censored']+final['unobserved_episode_starts_at_stop']==1024
         and final['active_contact_at_stop']<=final['active_at_stop_right_censored']
         and final['active_observed_productive_at_stop']<=final['active_contact_at_stop'],
         'final population censoring conservation differs')
    need(counts['observed_productive_contact_episodes']==counts['native_completed_productive_episodes']
         +counts['external_reset_observed_productive_episodes']+final['active_observed_productive_at_stop'],
         'final productive population conservation differs')
    encoded(final)

def _jsonl(path,cap,line_cap):
    body=read_file(path,cap);rows=[]
    for line in body.splitlines():
        need(len(line)<=line_cap,'trace line cap exceeded')
        rows.append(json.loads(line,object_pairs_hook=_pairs,parse_constant=lambda x:need(False,'nonfinite trace')))
    return body,rows

def file_ref(path):
    p=a.regular_path(path)
    return dict(path=str(p),size=p.stat().st_size,sha256=a.sha_file(p))

def validate_trace_rows(rows,*,split=False):
    a.need(len(rows)==500,'complete500 ordered trace rows required')
    if split:
        validate=importlib.import_module('scripts.z1_timing_noise020.split_contract').validate_trace_row
        previous=None
        for update,row in enumerate(rows):
            previous=validate(row,update,previous=previous)
            a.need(previous is not None,'split trace validator did not return continuity state')
    else:
        for update,row in enumerate(rows):validate_trace(row,update)


def seal_cell(release,global_index):
    a.need(type(global_index) is int and 0<=global_index<30,'invalid global cell')
    wrapper=release['cells'][global_index];cell=wrapper['training_cell'];output=Path(wrapper['output'])
    initial=a.read_json(output/'initial-state.json')
    a.need(initial.get('cell')==cell and type(initial.get('actual_counter')) is int and initial['actual_counter']==0
           and initial.get('phase')=='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT','initial identity/phase differs')
    contract=initial.get('contract');a.need(type(contract) is dict,'initial contract missing')
    if wrapper['family']=='split_credit':
        split=importlib.import_module('scripts.z1_timing_noise020.split_contract')
        marker=split.validate_initial(initial,cell)
    else:
        marker=expected_pilot_identity(cell['setting'],credit_deadline_ms=cell['credit_deadline_ms'],critic_mode=cell['critic_mode'])
        a.need(contract.get('spline_exploration_pilot')==marker and contract.get('spline_learning_variant') is None,'pilot marker differs')
    std=initial.get('initial_std');a.need(type(std) is list and len(std)==13,'initial std missing')
    expected=[.1]*6+[.3]*6+[1.] if cell['setting']=='vic_position_low' else [.3]*12+[1.]
    for value,target in zip(std,expected):
        _number(value,0);a.need(math.isclose(value,target,rel_tol=1e-6,abs_tol=1e-8),'initial std differs')
    hashes=initial.get('hashes',{});observations=initial.get('initial_observations',{})
    for key in (*COMMON_INITIAL_HASHES,'actor','critic'):
        a.need(type(hashes.get(key)) is str and re.fullmatch('[0-9a-f]{64}',hashes[key]),'initial hash missing')
    for key in ('actor_sha256','critic_shared43_sha256'):
        a.need(type(observations.get(key)) is str and re.fullmatch('[0-9a-f]{64}',observations[key]),'observation hash missing')
    a.need(observations.get('actor_shape')==[1024,43] and observations.get('critic_shape')==[1024,marker['critic_width']],'observation shape differs')
    labels=(0,24,49,99,249,499);train=output/'train'
    a.need({str(p.relative_to(train)) for p in train.rglob('*.pt')}=={f'model_{i}.pt' for i in labels},'exact six checkpoints required')
    refs={'initial':file_ref(output/'initial-state.json')};checkpoints={}
    for label in labels:
        path=train/f'model_{label}.pt';ref=file_ref(path);live=a.read_json(output/f'checkpoint-{label}-live.json')
        a.need(type(live.get('iteration')) is int and live['iteration']==label
               and type(live.get('common_step_counter')) is int and live['common_step_counter']==(label+1)*24
               and live.get('checkpoint_sha256')==ref['sha256'],'checkpoint clock/hash differs')
        a.need(all(type(live.get('hashes',{}).get(k)) is str and re.fullmatch('[0-9a-f]{64}',live['hashes'][k]) for k in ('actor','critic')),'checkpoint model hashes missing')
        checkpoints[str(label)]=dict(checkpoint=ref,live=file_ref(output/f'checkpoint-{label}-live.json'))
    body,rows=_jsonl(output/'training-trace.jsonl',release['limits']['trace_bytes'],1024**2)
    validate_trace_rows(rows,split=wrapper['family']=='split_credit')
    population,rows=_jsonl(output/'velocity-population.jsonl',release['limits']['population_bytes'],65536)
    validate_population(rows,threshold=contract.get('safety',{}).get('cat_soft',{}).get('limit'))
    compiler.verify_receipt(release,f"cell-{global_index:02d}",output,Path(wrapper['private']))
    refs.update(compiler=file_ref(output/'compiler-options.json'),trace=file_ref(output/'training-trace.jsonl'),population=file_ref(output/'velocity-population.jsonl'),gpu=file_ref(output/'gpu-runtime.json'))
    return dict(index=global_index,training_cell=cell,updates=500,common_step_counter=12000,trace_rows=500,population_rows=501,checkpoints=checkpoints,
                files=refs,initial_hashes=hashes,initial_observations=observations)

def compare_initial(result,reference,*,full=False):
    a.need(all(result['initial_hashes'][key]==reference['initial_hashes'][key] for key in (*COMMON_INITIAL_HASHES,*(('actor','critic') if full else ()))),'matched initial model/RNG mismatch')
    a.need(all(result['initial_observations'][key]==reference['initial_observations'][key] for key in ('actor_sha256','critic_shared43_sha256')),'matched initial observation mismatch')

def validate_split_reference(wrapper,reference):
    """B controls share the same seed AND exploration setting with A50."""
    cell=wrapper['training_cell'];control=reference.get('training_cell',{})
    expected=dict(index=cell['index']//2*4+cell['index']%2,setting=cell['setting'],seed=cell['seed'],
                  critic_mode='current',updates=500,worlds=1024,rollout_steps=24,credit_deadline_ms=50)
    a.need(wrapper['family']=='split_credit' and reference.get('index')==a.reference_index(wrapper)
           and a.canonical(control)==a.canonical(expected),'split A50 control seed/setting/cell differs')


def seal_slot(release,slot):
    wrappers=a.slot_cells(release,slot)
    results=[seal_cell(release,c['index']) for c in wrappers]
    for wrapper,result in zip(wrappers,results):
        if slot>=6:
            reference=seal_cell(release,a.reference_index(wrapper))
            validate_split_reference(wrapper,reference)
        else:reference=results[0]
        compare_initial(result,reference,full=slot>=6)
    return dict(slot=slot,cells=results,status='COMPLETE',manifest_sha256=a.digest(release))
