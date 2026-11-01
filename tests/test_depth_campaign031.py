"""Finite031 controls; no physical construction or cluster commands."""
import copy
import importlib
from pathlib import Path
import subprocess
import sys
import pytest


def module(name):
    return importlib.import_module('scripts.z1_depth_campaign031.'+name)


def test_matrix_is_closed_twelve_cells_six_slots_twenty_four_cases():
    m=module('manifest'); r=m.template()
    assert len(r['cells'])==12 and len(r['cases'])==24
    assert [x['cells'] for x in r['slots']]==[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
    assert m.validate(r) is r
    for mutate in (lambda x:x['cells'][1].update(seed=9),lambda x:x['cells'].append(x['cells'][0]),
                   lambda x:x['limits'].update(eval_concurrency=8),lambda x:x.update(resume=True)):
        bad=copy.deepcopy(r);mutate(bad)
        with pytest.raises(ValueError):m.validate(bad)


def test_quota_remaining_does_not_double_count_consumed():
    m=module('manifest'); life=module('lifecycle')
    assert m.CAP_BYTES==int(7.375*1024**3)
    assert life.required_free(0)==int(10.375*1024**3)
    assert life.required_free(256*1024**2)==int(10.125*1024**3)
    with pytest.raises(ValueError):life.admit_quota(100,0)
    for used in (-1,True,m.CAP_BYTES+1):
        with pytest.raises(ValueError):life.required_free(used)


@pytest.fixture
def original_clock(tmp_path,monkeypatch):
    from tests.test_depth_admission031 import origin, FIRST
    a,r,p=origin(monkeypatch,tmp_path)
    def create(manifest_sha='a'*64):
        return module('lifecycle').start_clock(tmp_path/'fixture-clock.json',manifest_sha,'123',FIRST+3000,host='gn52')
    return create


def test_global_clock_cannot_restart_or_use_local_clock(tmp_path,original_clock):
    l=module('lifecycle');first=1793500613
    clock=original_clock()
    assert clock['deadline_epoch']==1793529413
    with pytest.raises(FileExistsError):l.start_clock(tmp_path/'fixture-clock.json','a'*64,'124',first+3100,host='gn52')
    assert l.admit_time(clock,now=first+2001,allocation_end=first+19000,bound=16030)==first+19000
    for now,end,bound in ((first+14000,first+30000,16030),(first,first+16000,16030)):
        with pytest.raises(ValueError):l.admit_time(clock,now=now,allocation_end=end,bound=bound)
    with pytest.raises(ValueError):l.start_clock(tmp_path/'bad.json','a'*64,'125',first+2000,host='MacBook')


def test_durable_exclusive_json_rejects_duplicates_and_symlinks(tmp_path):
    f=module('files'); p=tmp_path/'x.json';f.write_once(p,{'a':1})
    assert f.read_json(p)=={'a':1}
    with pytest.raises(FileExistsError):f.write_once(p,{'a':2})
    d=tmp_path/'bad.json';d.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):f.read_json(d)
    q=tmp_path/'link';q.symlink_to(p)
    with pytest.raises(ValueError):f.reference(q)
    (tmp_path/'dir').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):f.write_once(tmp_path/'dir'/'new',{})


def test_source_snapshot_binds_workspace_not_git_head(tmp_path):
    m=module('manifest'); f=module('files')
    src=tmp_path/'src';src.mkdir();(src/'dirty.py').write_text('reviewed dirty bytes\n')
    dest=tmp_path/'frozen';pins=m.copy_sources(src,dest,['dirty.py'])
    assert pins=={'dirty.py':f.sha_file(src/'dirty.py')}
    (src/'dirty.py').write_text('later mutation')
    assert (dest/'dirty.py').read_text()=='reviewed dirty bytes\n'
    with pytest.raises(FileExistsError):m.copy_sources(src,dest,['dirty.py'])


def test_qualification_budget_includes_1024_smoke():
    m=module('manifest'); recipe=m.qualification_recipe()
    assert sum(x['seconds'] for x in recipe['gates'])+4*70+270==2470
    assert [x['constructor_worlds'] for x in recipe['gates'][:3]]==[2,4,16]
    assert [x['reset_worlds'] for x in recipe['gates'][:3]]==[10,4,48]
    assert [x['world_transitions'] for x in recipe['gates'][:3]]==[97,800,6480]
    assert recipe['gates'][-1]['production']['worlds']==1024
    assert recipe['gates'][-1]['production']['updates']==1
    assert recipe['gates'][-1]['production']['world_transitions']==24576


def test_seal_independent_model99_and_refuses_wrong_live(tmp_path):
    e=module('evaluation');f=module('files');m=module('manifest');r=m.template()
    out=tmp_path/'cell';out.mkdir();(out/'train').mkdir()
    ck=out/'train/model_99.pt';ck.write_bytes(b'checkpoint')
    marker=f.read_json(Path(e.__file__).with_name('identities031.json'))['A0']
    contract=dict(schema_version='impact_speed_v1',action_type='joint_position_variable_impedance_sampled_pd',
        controller_identity='sampled_vic500_v1',action_dim=13,physics_dt_s=.00025,control_decimation=80,
        speed_bounds_mps=[1.,1.5],speed_mode='policy',observation_widths={'actor':43,'critic':43},
        spline_depth_reward031=marker,spline_split_credit=marker['baseline_split'],
        safety={'cat_soft':dict(use_vel=True,use_impulse=True,imp_limit=[.369,.246,.738,.369,.246,.0164],imp_max_p=.2,imp_seed=.001,limit=3.1415,max_p=.5,min_p=0.,tau=.95,vel_detection='substep')})
    initial={'cell':m.training_cell(0),'actual_counter':0,'phase':'BEFORE_UPDATE_0_AND_FIRST_ROLLOUT',
        'contract':contract,'model_initialization_seed':m.SEEDS[0],'initial_std':[.3]*12+[1.],
        'hashes':dict.fromkeys(('actor','critic','actor_shared_non_std','shared_critic43','actor_normalizer','critic_normalizer','collection_torch_rng','collection_python_rng','collection_numpy_rng','collection_cuda_rng'),'a'*64),
        'initial_observations':dict(actor_shape=[1024,43],critic_shape=[1024,43],actor_sha256='b'*64,critic_shared43_sha256='c'*64)}
    f.write_once(out/'initial-state.json',initial)
    f.write_once(out/'checkpoint-99-live.json',{'iteration':99,'common_step_counter':2400,'checkpoint_sha256':f.sha_file(ck)})
    seal=e.checkpoint_seal(r,0,99,out)
    assert seal['checkpoint_label']==99 and not (out/'train/model_499.pt').exists()
    assert seal['checkpoint']['sha256']==f.sha_file(ck)
    with pytest.raises(ValueError):e.checkpoint_seal(r,0,499,out)


def test_owned_submission_readiness_does_not_await_a0_completion():
    l=module('lifecycle')
    assert l.ready_slots(set(),set())==[0,2,4]
    assert l.ready_slots({0,2,4},{0})==[1]
    assert l.ready_slots(set(range(6)),{0,4,8})==[]
    assert l.qualification_terminal({'JobState':'COMPLETED','ExitCode':'0:0'})
    assert not l.qualification_terminal({'JobState':'RUNNING','ExitCode':'0:0'})


def test_preservation_archive_checks_all_members_once(tmp_path):
    p=module('preservation');f=module('files');root=tmp_path/'raw';root.mkdir()
    (root/'zero').touch();(root/'sample').write_bytes(b'raw evidence')
    receipt=p.archive(root,tmp_path/'evidence.tar',identity='cell-00',cap=1024)
    assert receipt['members']=={'sample':{'sha256':f.sha_file(root/'sample'),'size':12},'zero':{'sha256':f.sha_file(root/'zero'),'size':0}}
    assert p.verify_archive(receipt)['status']=='VERIFIED_MEMBERS'
    with (tmp_path/'evidence.tar').open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError):p.verify_archive(receipt)


def test_probe_recipe_exact_real_and_synthetic_start_denominators():
    p=module('probe');r=p.recipe();c=r['counts']
    assert c['constructor_worlds']==12+1024
    assert c['explicit_reset_worlds']==12+1024
    assert c['world_transitions']==6*2*16+1024*24
    assert c['world_physics_substeps']==c['world_transitions']*80
    assert c['actor_forwards']==8+24+20 and c['critic_forwards']==8+24+1+20
    assert c['world_model_forwards']==32+2*1024*24+1024+2*20*(1024*24//4)
    assert c['optimizer_steps']==20 and c['algorithm_updates']==1
    assert c['synthetic_world_observations']==4*2*882 and r['synthetic_simulated_transitions']==0


def test_quota_monitor_checks_each_enforced_training_partition(tmp_path):
    w=module('worker');m=module('manifest')
    out=tmp_path/'out';out.mkdir();private=tmp_path/'private';private.mkdir()
    # Sparse files enforce actual logical Ceph byte reservations without RAM/disk growth.
    with (out/'training-trace.jsonl').open('wb') as stream:stream.truncate(m.LIMITS['trace']+1)
    with pytest.raises(ValueError,match='trace cap'):w._check_tree_caps(out,private,'training')


def test_foreign_job_cannot_be_cancelled_even_at_deadline():
    l=module('lifecycle');sent=[];sha='a'*64
    own=dict(manifest_sha256=sha,kind='training',job_id='123',name='depth031-slot-0')
    foreign=dict(JobId='123',JobName='foreign',UserId='eunikhilr(1)',Comment='depth031:'+sha,JobState='RUNNING')
    with pytest.raises(ValueError):l.cancel_expired_gpu_jobs([own],{'deadline_epoch':5,'manifest_sha256':sha},now=6,lookup=lambda _:foreign,run=lambda *a,**kw:sent.append(a))
    assert sent==[]
    foreign['JobName']=own['name']
    l.cancel_expired_gpu_jobs([own],{'deadline_epoch':5,'manifest_sha256':sha},now=6,lookup=lambda _:foreign,run=lambda *a,**kw:sent.append(a))
    assert sent==[(['scancel','123'],)]


def test_grant_rejects_old_authority_wrong_scope_and_expiry(tmp_path):
    l=module('lifecycle');f=module('files');m=module('manifest');r=m.template()
    for name in ('spec','resources'):f.write_once(tmp_path/name,{'finding':'reviewed'})
    g=dict(schema='z1_depth_root_grant031',manifest_sha256=f.digest(r),issued_epoch=1,expires_epoch=500,
        scope='campaign',issuer='root',nonce='a'*32,review_refs=[f.reference(tmp_path/'spec'),f.reference(tmp_path/'resources')])
    assert l.validate_grant(r,g,now=2,scope='campaign')==g
    for bad in (dict(g,schema='z1_recovery_grant029'),dict(g,scope='qualification'),dict(g,expires_epoch=2),dict(g,manifest_sha256='b'*64)):
        with pytest.raises(ValueError):l.validate_grant(r,bad,now=2,scope='campaign')


def test_controller_closed_allowlist_and_resource_bounds():
    w=module('worker');m=module('manifest');r=m.template();r['python']={'path':'/pinned/python'}
    plan=w.submission_plan(r)
    assert len(plan['jobs'])==30 and plan['controller_seconds']==18*3600
    assert plan['initial_slots']==[0,2,4] and plan['qualification_requires_terminal_completed']
    for row in plan['jobs']:
        cmd=row['command'];assert cmd[:4]==['sbatch','--parsable','--no-requeue','--export=NONE']
        if row['kind']=='training':assert '--gres=gpu:1' in cmd and '--time=04:30:00' in cmd
        else:assert '--cpus-per-task=2' in cmd and '--mem=8G' in cmd and '--time=02:15:00' in cmd


def test_ambiguous_submission_is_consumed_before_sbatch_no_retry(tmp_path,monkeypatch):
    w=module('worker');f=module('files');m=module('manifest');r=m.template();r['python']={'path':'/python'}
    monkeypatch.setattr(w,'ROOT',str(tmp_path));(tmp_path/'submissions').mkdir()
    calls=[]
    def fail(*args,**kwargs):calls.append(args);raise subprocess.TimeoutExpired('sbatch',30)
    monkeypatch.setattr(w.subprocess,'run',fail)
    with pytest.raises(subprocess.TimeoutExpired):w._submit_once(r,w.submission_plan(r),'training',0)
    assert (tmp_path/'submissions/training-00.consumed.json').is_file()
    with pytest.raises(FileExistsError):w._submit_once(r,w.submission_plan(r),'training',0)
    assert len(calls)==1


def test_source_pinned_bootstrap_real_isolated_process_rejects_mutation(tmp_path):
    m=module('manifest');f=module('files')
    source=tmp_path/'source';package=source/'scripts/z1_depth_campaign031';package.mkdir(parents=True)
    original=Path(module('__main__').__file__);entry=package/'__main__.py';entry.write_bytes(original.read_bytes())
    manifest=dict(source_root=str(source),source_files={'scripts/z1_depth_campaign031/__main__.py':f.sha_file(entry)})
    path=tmp_path/'manifest.json';f.write_once(path,manifest)
    entry.write_text(entry.read_text()+'\n# changed after freeze\n')
    proc=subprocess.run([sys.executable,'-B','-I','-S',str(entry),'--manifest',str(path),'--manifest-sha',f.digest(manifest),'verify'],capture_output=True,text=True)
    assert proc.returncode!=0 and 'bootstrap source bytes differ' in proc.stderr


def test_vendor_is_exact_25_file_413171_byte_closure():
    e=module('evaluation');pins=e.verify_copies()
    assert len(pins)==25 and pins['reset-bank001/BANK32-BALANCED-PROSPECTIVE002.json']==e.BANK_SHA


def test_archive_rejects_symlink_and_duplicate_identity(tmp_path):
    p=module('preservation');f=module('files');m=module('manifest')
    root=tmp_path/'raw';root.mkdir();(root/'file').write_bytes(b'one');(root/'link').symlink_to(root/'file')
    with pytest.raises(ValueError):p.archive(root,tmp_path/'bad.tar',identity='cell-00',cap=10)
    (root/'link').unlink();r=p.archive(root,tmp_path/'good.tar',identity='cell-00',cap=10)
    f.write_once(tmp_path/'receipt',r);ref=f.reference(tmp_path/'receipt')
    with pytest.raises(ValueError,match='duplicate'):p.final_join(m.template(),[None]*12,[None]*24,[ref,ref])


def test_scheduler_epochs_parse_explicit_utc(monkeypatch):
    l=module('lifecycle')
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setattr(l.socket,'gethostname',lambda:'gn52');monkeypatch.setattr(l,'gpu_allocation_identity',lambda: {'gpu_uuid':'fixture'})
    monkeypatch.setattr(l,'slurm_job',lambda _:dict(NodeList='gn52',BatchHost='gn52',JobName='depth031-slot-0',UserId='eunikhilr(1)',JobState='RUNNING',Requeue='0',Restarts='0',Account='d2026d06-166-users',QOS='normal',Partition='gpu',NumCPUs='4',NumNodes='1',AllocTRES='cpu=4,mem=32G,node=1,gres/gpu=1',StartTime='2026-11-01T01:00:00',EndTime='2026-11-01T05:30:00',TimeLimit='04:30:00',Deadline='2026-11-01T09:00:00'))
    a=l.allocation('123','depth031-slot-0')
    from datetime import datetime,timezone
    assert a['start_epoch']==datetime(2026,11,1,1,tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize('arm',['A0','A1','A2','A3'])
def test_probe_synthetic_real_manager_fixture_on_cpu_without_physics(arm):
    p=module('probe')
    from tests.test_depth_reward031 import scene_env
    env=scene_env(arm);counts={'synthetic_active':False,'synthetic_owner_reset_worlds':0}
    row=p.synthetic_fixture(env,arm,counts)
    assert row['label']=='SYNTHETIC_MANAGER_FIXTURE_NOT_A_PHYSICAL_STRIKE'
    assert row['simulated_transitions']==0 and counts['synthetic_owner_reset_worlds']==5
    assert row['transient']['first_attainment_tick']==[281,-1]
    assert counts['synthetic_active'] is False


def test_copied_owned_tree_tracks_detached_children_without_foreign_signals():
    l=module('lifecycle')
    from tests.test_startup_diagnostic_admission import Processes,process
    import signal
    api=Processes([process(200,100),process(201,200,pgid=201,sid=201),process(202,201,pgid=202,sid=202)])
    tree=l.OwnedTree(api,200);tree.signal_all(signal.SIGTERM)
    assert {identity[0] for identity,sig in api.sent}=={200,201,202}
    api.rows={};api.no_children=lambda:False
    assert tree.pending()
    api.no_children=lambda:True
    assert not tree.pending();tree.close()


def test_partial_initial_publication_is_not_ready_and_not_a_claim(tmp_path,monkeypatch):
    e=module('evaluation');m=module('manifest');r=m.template()
    r['cells'][0]['output']=str(tmp_path);(tmp_path/'initial-state.json').write_text('{"cell":')
    assert e.publish_seals(r,0)==[]
    assert not (tmp_path/'initial-seal.json').exists()


def test_readonly_package_import_isolated_has_no_scientific_imports():
    script="import sys; import scripts.z1_depth_campaign031.worker; import scripts.z1_depth_campaign031.probe; assert not any(n.split('.')[0] in ('torch','numpy','mjlab','mujoco','src') for n in sys.modules)"
    result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_copied_population_validator_keeps_censoring_and_unknown_semantics():
    w=module('worker')
    from tests.test_timing_noise_campaign020 import population_fixture
    rows=population_fixture();w.validate_population(rows,threshold=1.)
    rows[-1]['active_at_stop_right_censored']=1
    with pytest.raises(ValueError):w.validate_population(rows,threshold=1.)


def test_gpu_scheduler_deadline_is_absolute_even_without_live_controller(original_clock):
    m=module('manifest');w=module('worker');f=module('files');r=m.template();r['python']={'path':'/python'}
    clock=original_clock(f.digest(r))
    p=w.submission_plan(r,clock)
    assert p['slurm_gpu_deadline_utc']=='2026-11-01T10:33:53'
    for j in p['jobs']:
        if j['kind']=='training':
            assert '--deadline=2026-11-01T10:33:53' in j['command']
            assert '--time=04:30:00' in j['command'] and not any(x.startswith('--time-min') for x in j['command'])


def test_actual_resource_shape_rejects_extra_gpu_cpu_or_memory():
    l=module('lifecycle')
    row=dict(Account='d2026d06-166-users',QOS='normal',Partition='gpu',NumCPUs='4',NumNodes='1',TimeLimit='04:30:00',AllocTRES='cpu=4,mem=32G,node=1,gres/gpu=1')
    l.require_resource_shape(row,'depth031-slot-0')
    for extra in ({'NumCPUs':'8'},{'AllocTRES':'cpu=4,mem=64G,gres/gpu=1'},{'AllocTRES':'cpu=4,mem=32G,gres/gpu=2'},{'Partition':'cpu'}):
        with pytest.raises(ValueError):l.require_resource_shape(dict(row,**extra),'depth031-slot-0')


def test_compiler_pch_private_cache_is_configured_before_scientific_imports(tmp_path):
    c=module('compiler');f=module('files')
    from types import SimpleNamespace
    (tmp_path/'private').mkdir();out=tmp_path/'out';out.mkdir()
    config=SimpleNamespace(__file__=c._ROLES['warp.config'],use_precompiled_headers=True,version='1.14.0')
    modules={'warp':SimpleNamespace(config=config),
        'warp._src.context':SimpleNamespace(__file__=c._ROLES['warp._src.context'],runtime=None),
        'warp._src.build':SimpleNamespace(__file__=c._ROLES['warp._src.build'])}
    r={'compiler_policy':c.policy()}
    receipt=c.configure(r,tmp_path/'private',out,'fixture',modules=modules,importer=lambda name:modules[name])
    assert receipt['use_precompiled_headers'] is False and config.kernel_cache_dir==str(tmp_path/'private/cache/warp')
    assert c.verify_configured(r,tmp_path/'private',modules=modules)
    modules['src']=SimpleNamespace()
    with pytest.raises(ValueError,match='after scientific'):c.configure(r,tmp_path/'private',out,'fixture',modules=modules,importer=lambda name:modules[name])


def test_cpu_case_deadline_is_independent_of_controller_liveness(original_clock):
    m=module('manifest');w=module('worker');f=module('files');r=m.template();r['python']={'path':'/python'}
    clock=original_clock(f.digest(r))
    plan=w.submission_plan(r,clock)
    assert plan['slurm_cpu_case_deadline_utc']=='2026-11-01T20:33:53'
    for job in plan['jobs']:
        if job['kind']=='evaluation':assert '--deadline=2026-11-01T20:33:53' in job['command']


def _publication_fixture(tmp_path):
    # Reuse the complete regular-file031 checkpoint fixture; no native imports.
    test_seal_independent_model99_and_refuses_wrong_live(tmp_path)
    m=module('manifest');f=module('files');r=m.template()
    r['root']=str(tmp_path);r['cells'][0]['output']=str(tmp_path/'cell')
    (tmp_path/'claims').mkdir()
    f.write_once(tmp_path/'claims/cell-00.json',dict(schema='z1_depth_claim031',manifest_sha256=f.digest(r),kind='training',index=0))
    initial=dict(schema='z1_depth_initial_seal031',manifest_sha256=f.digest(r),cell=0,
        initial=f.reference(tmp_path/'cell/initial-state.json'),claim=f.reference(tmp_path/'claims/cell-00.json'))
    return r,tmp_path/'cell',initial


@pytest.mark.parametrize('target',['initial-seal.json','model-99-seal.json'])
def test_concurrent_identical_seal_publishers_both_succeed(tmp_path,monkeypatch,target):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    e=module('evaluation');f=module('files');r,out,initial=_publication_fixture(tmp_path)
    if target=='model-99-seal.json':f.write_once(out/'initial-seal.json',initial)
    barrier=threading.Barrier(2);original=e.write_once
    def simultaneous(path,value):
        if Path(path).name==target:barrier.wait(timeout=5)
        return original(path,value)
    monkeypatch.setattr(e,'write_once',simultaneous)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(e.publish_seals,r,0) for _ in range(2)]
        results=[future.result(timeout=10) for future in futures]
    assert results[0]==results[1]
    assert f.read_json(out/'initial-seal.json')==initial
    expected=e.checkpoint_seal(r,0,99);expected['claim']=initial['claim']
    assert f.read_json(out/'model-99-seal.json')==expected
    assert not list(out.glob('*.pending'))


@pytest.mark.parametrize('target',['initial-seal.json','model-99-seal.json'])
def test_conflicting_preexisting_seal_is_fatal_without_replacement(tmp_path,target):
    e=module('evaluation');f=module('files');r,out,initial=_publication_fixture(tmp_path)
    if target!='initial-seal.json':f.write_once(out/'initial-seal.json',initial)
    wrong=dict(initial,manifest_sha256='0'*64)
    f.write_once(out/target,wrong);before=f.reference(out/target)
    with pytest.raises(ValueError,match='seal'):e.publish_seals(r,0)
    assert f.reference(out/target)==before


def test_asset_copy_plan_matches_pinned_native_resolver_without_import(tmp_path):
    m=module('manifest');f=module('files');r=m.template()
    native=Path('src/tasks/hammer/nail_block.py');source=tmp_path/'source'
    target=source/'src/tasks/hammer/nail_block.py';target.parent.mkdir(parents=True);target.write_bytes(native.read_bytes())
    r['source_files']={'src/tasks/hammer/nail_block.py':f.sha_file(target)}
    rows=[dict(path='/old/immutable/assets/nail_block_scene.xml',sha256='a'*64)]
    plan=m.asset_copy_plan(rows,r['source_root'])
    destination=m.ROOT+'/safe_impact_manipulation/hammer_z1_env/assets/nail_block_scene.xml'
    assert plan['files'][0]['destination']==destination
    r['asset_files']={destination:'a'*64}
    assert m.verify_asset_layout(r,source_tree=source)==Path(destination).parent
    r['asset_files']={m.ROOT+'/source/safe_impact_manipulation/hammer_z1_env/assets/nail_block_scene.xml':'a'*64}
    with pytest.raises(ValueError,match='asset'):m.verify_asset_layout(r,source_tree=source)
    r['asset_files']={destination:'a'*64}
    target.write_text(target.read_text().replace('parents[4]','parents[3]'))
    r['source_files']['src/tasks/hammer/nail_block.py']=f.sha_file(target)
    with pytest.raises(ValueError,match='resolver'):m.verify_asset_layout(r,source_tree=source)


def _cpu_row(deadline='2026-11-01T20:33:53'):
    return dict(NodeList='cn0001',BatchHost='cn0001',JobName='depth031-eval-0',UserId='eunikhilr(1)',JobState='RUNNING',Requeue='0',Restarts='0',
        Account='d2026d06-166-users',QOS='normal',Partition='cpu',NumCPUs='2',NumNodes='1',
        AllocTRES='cpu=2,mem=8G,node=1',TimeLimit='02:15:00',StartTime='2026-11-01T01:00:00',
        EndTime='2026-11-01T03:15:00',Deadline=deadline)


@pytest.mark.parametrize('deadline',[None,'N/A','Unknown',''])
def test_cpu_allocation_requires_effective_scheduler_deadline(monkeypatch,deadline):
    l=module('lifecycle');row=_cpu_row(deadline)
    if deadline is None:row.pop('Deadline')
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setattr(l.socket,'gethostname',lambda:'cn0001')
    monkeypatch.setattr(l,'slurm_job',lambda _:row)
    with pytest.raises(ValueError,match='Deadline'):l.allocation('123','depth031-eval-0')


def test_cpu_allocation_reports_actual_deadline_and_time_limit(monkeypatch):
    from datetime import datetime,timezone
    l=module('lifecycle');monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setattr(l.socket,'gethostname',lambda:'cn0001');monkeypatch.setattr(l,'slurm_job',lambda _: _cpu_row())
    allocation=l.allocation('123','depth031-eval-0')
    assert allocation['deadline_epoch']==datetime(2026,11,1,20,33,53,tzinfo=timezone.utc).timestamp()
    assert allocation['time_limit_seconds']==8100


@pytest.mark.parametrize('fault',['missing','wrong','late','valid'])
def test_cpu_entry_clamps_to_immutable_clock_before_any_claim(tmp_path,monkeypatch,fault,original_clock):
    w=module('worker');l=module('lifecycle');m=module('manifest');f=module('files');r=m.template()
    clock=original_clock(f.digest(r))
    cutoff=1793500613+64800-180
    f.write_once(tmp_path/'clock.json',clock);monkeypatch.setattr(w,'ROOT',str(tmp_path))
    monkeypatch.setattr(w,'verify_package',lambda _:None);monkeypatch.setattr(w,'verify_qualification',lambda _:None)
    monkeypatch.setenv('SLURM_JOB_ID','123')
    alloc=dict(job_id='123',end_epoch=cutoff+100000,deadline_epoch=cutoff,time_limit_seconds=8100)
    if fault=='missing':alloc.pop('deadline_epoch')
    if fault=='wrong':alloc['deadline_epoch']+=1
    monkeypatch.setattr(l,'allocation',lambda *args:alloc)
    monkeypatch.setattr(w.time,'time',lambda:cutoff-7259 if fault=='late' else cutoff-8000)
    original=w.read_json
    monkeypatch.setattr(w,'read_json',lambda path:original(path) if Path(path).name=='clock.json' else {'fixture':'selected binding'})
    calls=[];monkeypatch.setattr(w,'run_one',lambda *args,**kwargs:calls.append((args,kwargs)))
    grant={'expires_epoch':cutoff+180}
    if fault=='valid':
        w.run_evaluation(r,grant,0)
        assert len(calls)==1 and calls[0][0][5]==cutoff
    else:
        with pytest.raises(ValueError):w.run_evaluation(r,grant,0)
        assert calls==[]


def test_controller_allocation_has_no_scheduler_deadline_requirement(monkeypatch):
    l=module('lifecycle');row=_cpu_row('N/A')
    row.update(JobName='depth031-controller',NumCPUs='1',AllocTRES='cpu=1,mem=2G,node=1',TimeLimit='18:00:00')
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setattr(l.socket,'gethostname',lambda:'cn0001')
    monkeypatch.setattr(l,'slurm_job',lambda _:row)
    allocation=l.allocation('123','depth031-controller')
    assert 'deadline_epoch' not in allocation
