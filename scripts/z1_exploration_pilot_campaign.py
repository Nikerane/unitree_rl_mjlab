"""Finite independent four-setting seed block; stdlib-only before admission.

One allocated GPU runs four fresh child processes sequentially. Separate seed
blocks may run independently only after root admits aggregate active-job quota.
No job submission, scheduling, deletion, retry or longer-run admission is provided.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

OLD_GUARD_SHA256 = 'b462c8143a0eaf0d12e13dabbfe881c17c3d3eae1ea5218a1469e840ea7404c6'
OLD_GUARD = 'scripts/z1_credit_critic_campaign.py'
# Verify helper bytes before executing even its stdlib-only top level.
_helper_path=Path(__file__).with_name('z1_credit_critic_campaign.py')
_fd=os.open(_helper_path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
with os.fdopen(_fd,'rb') as _f:
    _info=os.fstat(_f.fileno())
    if not stat.S_ISREG(_info.st_mode) or _info.st_size>65536:
        raise ValueError('bounded regular pinned helper required')
    _body=_f.read(_info.st_size+1)
    if len(_body)!=_info.st_size or hashlib.sha256(_body).hexdigest()!=OLD_GUARD_SHA256:
        raise ValueError('stdlib guard helper source pin changed')
_old=importlib.util.module_from_spec(importlib.util.spec_from_file_location('_z1_pilot_stdlib_guard',_helper_path))
exec(compile(_body,str(_helper_path),'exec'),_old.__dict__)
QUOTA_ROOT='/cephhome/eunikhilr'
WORKER='scripts/z1_exploration_pilot_campaign.py'
LIMITS=dict(output_bytes=96*1024**2,cache_bytes=1024**3,quota_floor_bytes=512*1024**2,
            cell_wall_seconds=3600,total_wall_seconds=7200)
BLOCK_RESERVE_BYTES=5234491392
PILOT_SEEDS=(2026102902,2026102903,2026102904)
PILOT_SETTINGS=('vic_baseline','vic_position_low','vic_gain_mean','fixed_nominal')
COMMON_INITIAL_HASHES=('actor_shared_non_std','shared_critic43','actor_normalizer','critic_normalizer',
    'collection_torch_rng','collection_python_rng','collection_numpy_rng','collection_cuda_rng')
CRITICAL_SOURCE=_old.CRITICAL_SOURCE | {WORKER,'scripts/z1_exploration_pilot_training.py',
    'src/tasks/hammer/config/z1/spline_exploration_pilot_cfg.py',
    'src/tasks/hammer/mdp/spline_exploration_pilot.py',
    'src/tasks/hammer/rl/spline_exploration_pilot.py',
    'src/tasks/hammer/rl/spline_exploration_pilot_contract.py',
    'src/tasks/hammer/rl/velocity_population_trace.py'}
RUNTIME_SUFFIXES,RUNTIME_MODULES,CACHE_KEYS=_old.RUNTIME_SUFFIXES,_old.RUNTIME_MODULES,_old.CACHE_KEYS
for _name in ('need','encoded','mapping_digest','real','read_file','file_sha','_pairs','load_json',
              'write_once','_verify_map','allocation_identity','verify_import_origins','directory_bytes',
              'validate_trace','verify_cuda_device','_mkdir','_kill'):
    globals()[_name]=getattr(_old,_name)


def _study_anchor(value):
    need(type(value) is dict and set(value)=={'credit_deadline_ms','critic_mode'},'one explicit study anchor required')
    need(type(value['credit_deadline_ms']) is int and value['credit_deadline_ms'] in (50,100,150,200)
         and type(value['critic_mode']) is str and value['critic_mode'] in ('current','event'),
         'unknown study anchor')
    return dict(value)


def expected_cells(seed_slot,*,credit_deadline_ms=200,critic_mode='current'):
    need(type(seed_slot) is int and 0<=seed_slot<3,'seed slot must be integer0..2')
    anchor=_study_anchor(dict(credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode))
    return [dict(index=seed_slot*4+i,setting=setting,seed=PILOT_SEEDS[seed_slot],
        credit_deadline_ms=anchor['credit_deadline_ms'],critic_mode=anchor['critic_mode'],updates=100,worlds=1024,rollout_steps=24)
        for i,setting in enumerate(PILOT_SETTINGS)]


def admission_binding(manifest):
    return dict(schema=manifest['schema'],campaign_root=manifest['campaign_root'],seed_slot=manifest['seed_slot'],
        study_anchor=manifest['study_anchor'],cells=manifest['cells'],limits=manifest['limits'],worker_sha256=manifest['worker_sha256'],
        python_sha256=manifest['python_sha256'],source_files_sha256=mapping_digest(manifest['source_files']),
        runtime_files_sha256=mapping_digest(manifest['runtime_files']),asset_files_sha256=mapping_digest(manifest['asset_files']))

def verify_manifest(root, pin):
    root = real(root)
    need(root.parent == Path(QUOTA_ROOT)/'campaigns' and root.name.startswith('z1-exploration-pilot-'), 'fresh campaign namespace required')
    need(re.fullmatch('[0-9a-f]{64}', pin or ''), 'manifest SHA required')
    need(file_sha(root/'manifest.json') == pin, 'bound manifest changed')
    m = load_json(root/'manifest.json')
    need(m.get('schema') == 'z1_exploration_pilot_campaign_v1' and m.get('campaign_root') == str(root), 'manifest schema/root differs')
    need(encoded(m.get('cells')) == encoded(expected_cells(m.get('seed_slot'),**_study_anchor(m.get('study_anchor')))) and encoded(m.get('limits')) == encoded(LIMITS), 'finite cells or limits changed')
    source = real(root/'source'); declared = m.get('source_files', {})
    need(CRITICAL_SOURCE <= set(declared), 'critical source missing')
    files, directories = set(), set()
    for path in source.rglob('*'):
        need(not path.is_symlink() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'), 'source symlink/bytecode forbidden')
        mode = path.lstat().st_mode; name = str(path.relative_to(source))
        need(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'source special file forbidden')
        (directories if stat.S_ISDIR(mode) else files).add(name)
    allowed_dirs = {str(p) for name in declared for p in Path(name).parents if str(p) != '.'}
    need(files == set(declared) and directories == allowed_dirs, 'source member set differs')
    need(declared.get(OLD_GUARD)==OLD_GUARD_SHA256, 'stdlib helper source pin differs')
    _verify_map(declared, source)
    need(real(__file__) == source/WORKER and file_sha(__file__) == m.get('worker_sha256'), 'running worker differs')
    executable = Path(m.get('python_executable',''))
    need(executable.is_absolute() and str(executable) == sys.executable and executable.resolve().is_file(), 'invoked Python differs')
    need(file_sha(executable.resolve()) == m.get('python_sha256'), 'resolved Python bytes changed')
    for key in ('runtime_files','asset_files'): _verify_map(m.get(key))
    roles = m.get('runtime_roles', {})
    need(set(roles) == set(RUNTIME_SUFFIXES), 'complete runtime roles required')
    for role, suffix in RUNTIME_SUFFIXES.items():
        need(roles[role].endswith('/'+suffix) and roles[role] in m['runtime_files'], 'runtime role differs')
    assets = m.get('asset_roles', {})
    need(set(assets) == {'scene_xml','robot_xml'}, 'complete XML asset roles required')
    need(all(path.endswith('.xml') and path in m['asset_files'] for path in assets.values()), 'asset role differs')
    for key in ('root_approval','admission'):
        ref = m.get(key)
        need(isinstance(ref,dict) and set(ref) == {'path','sha256'}, 'bound approval/admission missing')
        path = real(ref['path']); need(root in path.parents and file_sha(path) == ref['sha256'], 'admission reference changed')
        record = load_json(path)
        need(record.get('status') == 'PASS' and encoded(record.get('binding')) == encoded(admission_binding(m)), 'approval/admission not PASS for this package')
    need(m['root_approval']['path'] != m['admission']['path'], 'approval and admission must be separate records')
    return m

def quota_free():
    return int(os.getxattr(QUOTA_ROOT,'ceph.quota.max_bytes')) - int(os.getxattr(QUOTA_ROOT,'ceph.dir.rbytes'))

def _cell(root, index):
    need(type(index) is int and 0 <= index < 12, 'cell index must be integer0..11')
    return root/'runs'/f'cell-{index:02d}', root/'private'/f'cell-{index:02d}'

def child_argv(root, pin, index):
    _cell(root,index)
    return [sys.executable,'-B','-I',str(root/'source'/WORKER),'--campaign',str(root),'--manifest-sha',pin,'--cell',str(index)]

def limit_reason(root, index, started, deadline, *, now=None):
    now = time.monotonic() if now is None else now
    if now >= deadline: return 'TOTAL_WALL_LIMIT'
    if now-started >= LIMITS['cell_wall_seconds']: return 'CELL_WALL_LIMIT'
    output, private = _cell(root,index)
    logs = root/'logs'; logbytes = sum((logs/f'cell-{index:02d}.{suffix}').stat().st_size for suffix in ('stdout','stderr') if (logs/f'cell-{index:02d}.{suffix}').exists())
    if directory_bytes(output)+logbytes > LIMITS['output_bytes']: return 'OUTPUT_LIMIT'
    if directory_bytes(private) > LIMITS['cache_bytes']: return 'CACHE_LIMIT'
    if quota_free() < LIMITS['quota_floor_bytes']: return 'QUOTA_FLOOR'
    return None

def expected_pilot_identity(setting,*,credit_deadline_ms=200,critic_mode='current'):
    need(type(setting) is str and setting in PILOT_SETTINGS,'approved pilot setting required')
    anchor=_study_anchor(dict(credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode))
    event=anchor['critic_mode']=='event'
    fixed=setting=='fixed_nominal';weight=1/6 if setting=='vic_gain_mean' else 1.
    experiment=dict(schema_version='spline_credit_critic_v1',credit_deadline_ms=anchor['credit_deadline_ms'],
        episode_contact_window_ms=200,critic_mode=anchor['critic_mode'],actor_width=43,critic_width=47 if event else 43,
        critic_event_fields=['credit_contact_seen','credit_contact_age','credit_unpaid','credit_time_to_payout'] if event else [],actor_inputs_unchanged=True)
    classes=('PilotBaselineActor','PilotPositionLowActor','PilotGainMeanActor','PilotFixedNominalActor')
    return dict(schema_version='spline_exploration_pilot_v1',setting=setting,
        gain_mode='fixed' if fixed else 'variable',actor_width=43,critic_width=47 if event else 43,
        actor_class='src.tasks.hammer.rl.spline_exploration_pilot:'+classes[PILOT_SETTINGS.index(setting)],
        critic_class='src.tasks.hammer.rl.spline_credit_critic:CreditEventCritic' if event else 'src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedAgeModel',
        distribution_class='src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian',
        initial_exploration_std=([.1]*6+[.3]*6+[1.] if setting=='vic_position_low' else [.3]*12+[1.]),
        gain_entropy_weight=weight,effective_gain_entropy_weight=0. if fixed else weight,
        gain_entropy_mode='masked_out' if fixed else ('mean' if setting=='vic_gain_mean' else 'sum'),
        likelihood_gain_mask=0 if fixed else 1,selector_mask='existing_raw_pending',action_draws=13,
        fixed_nominal_kp=[256.,384.,384.,256.,192.,128.],fixed_nominal_kd=[6.4]*6,
        ignored_action_indices=list(range(6,12)) if fixed else [],
        startup_age=dict(source='sampled_pd500_physics_substeps_since_own_reset',normalization_s=.1,
                         clip=[0.,1.],observation_index=40),credit_critic_experiment=experiment)


_EPISODE_KEYS=('all_episode_starts','observed_episode_starts','observed_contact_onsets',
    'native_contact_episodes','native_no_contact_episodes','native_completed_productive_episodes',
    'observed_productive_contact_episodes','external_reset_observed_productive_episodes',
    'native_timeout_episodes','contact_resets_before_200ms','contact_policy_envelopes_reached_200ms',
    'external_reset_episodes','external_contact_censored_episodes')


def _number(value,low=None,high=None,integer=False):
    need(type(value) in ((int,) if integer else (int,float)) and math.isfinite(value),
         'finite population numeric domain required')
    need((low is None or value>=low) and (high is None or value<=high),'population numeric bound differs')
    return value


def _episode_counts(row,previous=None):
    need(type(row) is dict and set(row)==set(_EPISODE_KEYS),'complete population episode counters required')
    for key in _EPISODE_KEYS:
        _number(row[key],0,1024+1024*2400,integer=True)
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
    need(type(rows) is list and len(rows)==101,'100 population updates plus final row required')
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
         and type(final.get('updates')) is int and final['updates']==100
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


def final_artifacts(root,index,manifest=None):
    manifest=load_json(root/'manifest.json') if manifest is None else manifest
    anchor=_study_anchor(manifest.get('study_anchor'));cells=expected_cells(manifest['seed_slot'],**anchor);need(index in [c['index'] for c in cells],'artifact index outside seed block')
    cell=next(c for c in cells if c['index']==index);output,_=_cell(root,index)
    initial=load_json(output/'initial-state.json')
    need(encoded(initial.get('cell'))==encoded(cell) and type(initial.get('actual_counter')) is int
         and initial['actual_counter']==0 and initial.get('phase')=='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT',
         'initial pilot gate missing or different')
    marker=expected_pilot_identity(cell['setting'],**anchor);contract=initial.get('contract')
    need(type(contract) is dict and encoded(contract.get('spline_exploration_pilot'))==encoded(marker)
         and contract.get('spline_learning_variant') is None,'initial full pilot identity differs')
    std=initial.get('initial_std');need(type(std) is list and len(std)==13,'actual initial pilot std required')
    for actual,expected in zip(std,marker['initial_exploration_std']):
        _number(actual,0);need(math.isclose(actual,expected,rel_tol=1e-6,abs_tol=1e-8),'initial treatment std differs')
    hashes=initial.get('hashes',{})
    need(type(hashes) is dict and all(type(hashes.get(k)) is str and re.fullmatch('[0-9a-f]{64}',hashes[k])
         for k in COMMON_INITIAL_HASHES+('actor','critic')),'initial fingerprints incomplete')
    observations=initial.get('initial_observations',{})
    need(all(type(observations.get(k)) is str and re.fullmatch('[0-9a-f]{64}',observations[k])
         for k in ('actor_sha256','critic_shared43_sha256')),'initial observation fingerprints incomplete')
    need(encoded(observations.get('actor_shape'))==b'[1024,43]'
         and encoded(observations.get('critic_shape'))==encoded([1024,marker['critic_width']]),'full initial observation shape differs')
    checkpoints={};train=real(output/'train')
    need({str(p.relative_to(train)) for p in train.rglob('*.pt')}=={f'model_{i}.pt' for i in (0,24,49,99)},
         'exact four retained checkpoints required')
    for label in (0,24,49,99):
        path=train/f'model_{label}.pt';digest=file_sha(path);record=load_json(output/f'checkpoint-{label}-live.json')
        need(type(record.get('iteration')) is int and record['iteration']==label
             and type(record.get('common_step_counter')) is int and record['common_step_counter']==(label+1)*24
             and record.get('checkpoint_sha256')==digest,'live pilot checkpoint counter/digest differs')
        need(type(record.get('hashes')) is dict and all(type(record['hashes'].get(k)) is str
             and re.fullmatch('[0-9a-f]{64}',record['hashes'][k]) for k in ('actor','critic')),'checkpoint model fingerprints incomplete')
        checkpoints[path.name]=digest
    credit_body,credit_rows=_jsonl(output/'training-trace.jsonl',32*1024**2,1024**2)
    need(len(credit_rows)==100,'complete100-update credit trace required')
    for update,row in enumerate(credit_rows):validate_trace(row,update)
    population_body,population_rows=_jsonl(output/'velocity-population.jsonl',16*1024**2,65536)
    safety=contract.get('safety',{});cat=safety.get('cat_soft',{}) if isinstance(safety,dict) else {}
    validate_population(population_rows,threshold=cat.get('limit'))
    return dict(updates=100,common_step_counter=2400,checkpoint_sha256=checkpoints,
        initial_state_sha256=file_sha(output/'initial-state.json'),initial_hashes=hashes,
        initial_observations=observations,trace_sha256=hashlib.sha256(credit_body).hexdigest(),trace_bytes=len(credit_body),
        population_sha256=hashlib.sha256(population_body).hexdigest(),population_bytes=len(population_body))


def supervise(root, pin):
    manifest=verify_manifest(root,pin); identity=allocation_identity(); root=Path(root)
    need(quota_free() >= BLOCK_RESERVE_BYTES, 'quota cannot reserve full seed-block allowance')
    for name in ('claims','runs','private','logs'): _mkdir(root/name)
    for index in [c['index'] for c in manifest['cells']]:
        output,private=_cell(root,index)
        need(not os.path.lexists(output) and not os.path.lexists(private) and not os.path.lexists(root/'claims'/f'cell-{index:02d}.json'), 'cell path/claim already consumed')
    context=dict(schema='z1_exploration_pilot_supervisor_v1',supervisor_pid=os.getpid(),seed_slot=manifest['seed_slot'],manifest_sha256=pin,allocation=identity)
    write_once(root/'claims/supervisor.json',context)
    completed=[]; global_deadline=time.monotonic()+LIMITS['total_wall_seconds']; reference=None
    for index in [c['index'] for c in manifest['cells']]:
        process=None; reason=None; started=time.monotonic(); code=None
        try:
            need(started < global_deadline, 'total wall allocation exhausted')
            need(quota_free() >= LIMITS['quota_floor_bytes']+LIMITS['output_bytes']+LIMITS['cache_bytes'], 'quota cannot reserve next cell')
            output,private=_cell(root,index); _mkdir(private)
            env=os.environ.copy()
            for key in CACHE_KEYS:
                directory=private/key.lower(); _mkdir(directory); env[key]=str(directory)
            env.update(LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1')
            with (root/'logs'/f'cell-{index:02d}.stdout').open('xb') as stdout, (root/'logs'/f'cell-{index:02d}.stderr').open('xb') as stderr:
                process=subprocess.Popen(child_argv(root,pin,index),stdout=stdout,stderr=stderr,env=env,start_new_session=True)
                while process.poll() is None:
                    reason=limit_reason(root,index,started,global_deadline)
                    if reason: _kill(process); break
                    time.sleep(1)
                code=process.returncode
            reason = reason or limit_reason(root,index,started,global_deadline)
            need(reason is None and code == 0, 'child failed or finite monitor stopped it')
            artifact=final_artifacts(root,index,manifest); terminal=load_json(output/'train-complete.json')
            need(terminal.get('status') == 'COMPLETE' and terminal.get('manifest_sha256') == pin and terminal.get('allocation') == identity and terminal.get('artifacts') == artifact, 'child completion not independently joined')
            common={k:artifact['initial_hashes'][k] for k in COMMON_INITIAL_HASHES}
            common.update({k:artifact['initial_observations'][k] for k in ('actor_sha256','critic_shared43_sha256')})
            if reference is None: reference=common
            need(common == reference, 'paired initialization differs across cells')
            completed.append(index)
        except BaseException as exc:
            _kill(process)
            reason=reason or f'{type(exc).__name__}: {str(exc)[:2048]}'
        result=dict(status='COMPLETE' if reason is None else 'FAILED_NO_RETRY',cell=index,
                    returncode=code,reason=reason,allocation=identity,automatic_retry=False,
                    started_monotonic=started,finished_epoch=time.time())
        write_once(root/'logs'/f'cell-{index:02d}-terminal.json',result)
        if reason is not None: break
    result=dict(status='COMPLETE' if len(completed)==4 else 'FAILED_NO_RETRY',completed_cells=completed,
                failed_cell=None if len(completed)==4 else index,
                remaining_cells_not_attempted=[] if len(completed)==4 else [c['index'] for c in manifest['cells'] if c['index']>index],
                allocation=identity,manifest_sha256=pin,automatic_retry=False,finished_epoch=time.time())
    write_once(root/'supervisor-result.json',result)
    return result

def run_cell(root, pin, index):
    manifest=verify_manifest(root,pin); identity=allocation_identity(); root=Path(root); output,private=_cell(root,index)
    need(index in [c['index'] for c in manifest['cells']], 'child index outside declared seed block')
    context=load_json(root/'claims/supervisor.json'); parent=os.getppid()
    need(context.get('schema') == 'z1_exploration_pilot_supervisor_v1' and context.get('supervisor_pid') == parent and context.get('seed_slot') == manifest['seed_slot'] and context.get('manifest_sha256') == pin and context.get('allocation') == identity, 'child requires same live parent/job/GPU claim')
    expected=child_argv(root,pin,index)[:-2]+['--supervise']
    need(sys.platform == 'linux' and Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')[:-1] == [x.encode() for x in expected], 'supervisor command identity differs')
    need(ctypes.CDLL(None).prctl(1,signal.SIGTERM)==0 and os.getppid()==parent, 'cannot bind supervisor death')
    need(not os.path.lexists(output) and private.is_dir() and directory_bytes(private)==0, 'fresh output and empty private cache required')
    need(quota_free() >= LIMITS['quota_floor_bytes']+LIMITS['output_bytes'], 'preservation quota unavailable')
    write_once(root/'claims'/f'cell-{index:02d}.json',dict(cell=manifest['cells'][index-manifest['seed_slot']*4],parent_pid=parent,allocation=identity,manifest_sha256=pin))
    _mkdir(output)
    try:
        for key in CACHE_KEYS:
            path=real(private/key.lower()); need(path.is_dir() and os.environ.get(key)==str(path), 'private cache binding differs')
        sys.dont_write_bytecode=True; os.environ['PYTHONDONTWRITEBYTECODE']='1'; os.chdir(private); sys.path.insert(0,str(root/'source'))
        import torch
        write_once(output/'gpu-runtime.json',verify_cuda_device(torch,identity))
        import src.tasks  # noqa: F401 — admitted package registration only.
        from scripts.train import run_train
        from scripts.z1_exploration_pilot_training import build_pilot_train_config, selected_pilot_checkpoint_saves, matched_pilot_initial_state_gate, pilot_cell_for_index
        anchor=_study_anchor(manifest['study_anchor'])
        need(encoded(pilot_cell_for_index(index,**anchor)) == encoded(manifest['cells'][index-manifest['seed_slot']*4]), 'adapter cell differs')
        for name in ('scripts.train','scripts.z1_exploration_pilot_training'):
            need(Path(sys.modules[name].__file__).resolve() == root/'source'/Path(*name.split('.')).with_suffix('.py'), 'training import escaped frozen source')
        verify_import_origins(manifest,root/'source')
        task,cfg=build_pilot_train_config(index,**anchor)
        first=manifest['seed_slot']*4
        reference = load_json(root/'runs'/f'cell-{first:02d}'/'initial-state.json') if index > first else None
        with selected_pilot_checkpoint_saves(index,output,**anchor), matched_pilot_initial_state_gate(index,output,enable_trace=True,reference_initial_state=reference,**anchor):
            run_train(task,cfg,output/'train')
        result=dict(status='COMPLETE',cell=index,manifest_sha256=pin,allocation=identity,artifacts=final_artifacts(root,index,manifest),automatic_retry=False,finished_epoch=time.time())
        write_once(output/'train-complete.json',result)
        return result
    except BaseException as exc:
        write_once(output/'failure.json',dict(status='FAILED_NO_RETRY',error_type=type(exc).__name__,error=str(exc)[:4096],manifest_sha256=pin,automatic_retry=False,finished_epoch=time.time()))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cell',type=int); mode.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode, 'run worker with exact Python -B -I flags')
    result=supervise(args.campaign,args.manifest_sha) if args.supervise else run_cell(args.campaign,args.manifest_sha,args.cell)
    print(encoded(result).decode())
    return 0 if result['status']=='COMPLETE' else 1

if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(encoded(dict(status='FAILED_NO_RETRY',error_type=type(exc).__name__,error=str(exc)[:4096])).decode(),file=sys.stderr)
        raise SystemExit(1)
