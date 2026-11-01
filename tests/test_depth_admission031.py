"""Pure successor admission regressions; no scheduler or scientific execution."""
import copy
import hashlib
from pathlib import Path
import pytest
from scripts.z1_depth_campaign031 import lifecycle as life, manifest as m, files as f

EVIDENCE=Path(__file__).resolve().parents[1]/'.superpowers/sdd/2026-11-01-z1-depth-reward031'
FIRST=1793500613

def scheduler(monkeypatch,host='gn52',cpu=False):
    row=dict(JobId='123',JobName='depth031-eval-0' if cpu else 'depth031-qualification',
        UserId='eunikhilr(1)',JobState='RUNNING',Requeue='0',Restarts='0',Account=m.SLURM_ACCOUNT,
        QOS='normal',Partition='cpu' if cpu else 'gpu',NumCPUs='2' if cpu else '4',NumNodes='1',
        AllocTRES='cpu=2,mem=8G,node=1' if cpu else 'cpu=4,mem=32G,node=1,gres/gpu=1',
        TimeLimit='02:15:00' if cpu else '00:44:00',NodeList=host,BatchHost=host,
        StartTime='2026-11-01T03:00:00',EndTime='2026-11-01T03:44:00',Deadline='2026-11-01T20:33:53' if cpu else '2026-11-01T05:53:53')
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setattr(life.socket,'gethostname',lambda:host)
    monkeypatch.setattr(life,'slurm_job',lambda _:row)
    monkeypatch.setattr(life,'gpu_allocation_identity',lambda:dict(gpu_uuid='fixture'))
    return row

@pytest.mark.parametrize('host,cpu',[('gn01',False),('gn52.vega.izum.si',False),('gn60',False),('cn0001',True),('cn0959.vega.izum.si',True)])
def test_actual_compute_membership(monkeypatch,host,cpu):
    row=scheduler(monkeypatch,host,cpu);row['NodeList']=host.split('.')[0]
    assert life.allocation('123',row['JobName'])['job_id']=='123'

@pytest.mark.parametrize('host',['gn00','gn61','gn52.foreign.example','cn001','vglogin0007','vgpu0001','vcpu0001','localhost','MacBook'])
def test_foreign_hosts_rejected(host):
    with pytest.raises(ValueError):life.remote_host(host)

@pytest.mark.parametrize('change',[{'NodeList':'gn51'},{'BatchHost':'gn51'},{'NodeList':'gn[51-52]'},
    {'NodeList':None},{'BatchHost':None},{'NodeList':'cn0001','BatchHost':'cn0001'},
    {'NumNodes':'2'},{'NumCPUs':'8'},{'AllocTRES':'cpu=4,mem=64G,node=1,gres/gpu=1'},
    {'AllocTRES':'cpu=4,mem=32G,node=1,gres/gpu=2'},{'UserId':'foreign(1)'},{'QOS':'other'},
    {'Requeue':'1'},{'Restarts':'1'},{'TimeLimit':'00:45:00'}])
def test_allocation_rejects_membership_and_shape(monkeypatch,change):
    row=scheduler(monkeypatch);row.update(change)
    with pytest.raises(ValueError):life.allocation('123',row['JobName'])

def test_successor_namespace_and_aggregate_qualification():
    r=m.template();recipe=m.qualification_recipe()
    assert r['root'].endswith('20261101-v3') and r['evaluation_root'].endswith('20261101-v3')
    assert recipe['allocation_seconds']==2640
    assert recipe['total_bound_seconds']==2470 and m.LIMITS['qualification_seconds']==2880
    assert 190+recipe['allocation_seconds']==2830<=2880

def origin(monkeypatch,tmp_path):
    from scripts.z1_depth_campaign031 import admission as a
    wrapper=f.read_json(EVIDENCE/'first-gpu-allocation-accounting031.json')
    p=tmp_path/'origin.json';f.write_once(p,wrapper['record'])
    assert f.sha_file(p)==wrapper['reference']['sha256']
    monkeypatch.setattr(a,'ORIGIN_REF',dict(wrapper['reference'],path=str(p)))
    second=f.read_json(EVIDENCE/'second-technical-allocation-accounting031.json')
    q=tmp_path/'second.json';f.write_once(q,second['record'])
    assert q.read_bytes()==(EVIDENCE/'namespace-diagnosis031/second-technical-allocation-accounting-body031.json').read_bytes()
    assert q.stat().st_size==5013 and f.sha_file(q)==second['reference']['sha256']
    monkeypatch.setattr(a,'SECOND_REF',dict(second['reference'],path=str(q)),raising=False)
    r=m.template();return a,r,p

def test_origin_clock_carry_forward_and_plan(tmp_path,monkeypatch):
    from scripts.z1_depth_campaign031 import worker
    a,r,p=origin(monkeypatch,tmp_path);r['python']={'path':'/python'}
    clock=life.start_clock(tmp_path/'clock.json',f.digest(r),'123',FIRST+3000,host='gn52')
    assert clock['first_allocation_epoch']==FIRST and clock['qualification_job_id']=='49072485'
    assert clock['technical_allocation']==dict(job_id='123',start_epoch=FIRST+3000)
    assert clock['prior_technical_allocation']==dict(job_id='49072528',start_epoch=1793503232)
    assert clock['second_origin']==a.SECOND_REF
    assert clock['deadline_epoch']==1793529413 and clock['origin']==f.reference(p)
    plan=worker.submission_plan(r,clock)
    assert plan['slurm_gpu_deadline_utc']=='2026-11-01T10:33:53'
    assert plan['slurm_cpu_case_deadline_utc']=='2026-11-01T20:33:53'
    with pytest.raises(FileExistsError):life.start_clock(tmp_path/'clock.json',f.digest(r),'124',FIRST+3100,host='gn52')
    for key,value in [('first_allocation_epoch',FIRST+1),('qualification_job_id','123'),('origin',{})]:
        bad=copy.deepcopy(clock);bad[key]=value
        with pytest.raises(ValueError):worker.submission_plan(r,bad)
    p.write_text('{}')
    with pytest.raises(ValueError):life.start_clock(tmp_path/'bad.json',f.digest(r),'125',FIRST+2200,host='gn52')

def test_origin_missing_and_new_start_substitution(tmp_path,monkeypatch):
    a,r,p=origin(monkeypatch,tmp_path)
    for start in [FIRST-1,FIRST+11821,float('nan')]:
        with pytest.raises(ValueError):life.start_clock(tmp_path/'bad.json',f.digest(r),'123',start,host='gn52')
    p.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):life.start_clock(tmp_path/'bad.json',f.digest(r),'123',FIRST+3000,host='gn52')

def test_replacement_admission_budget_and_remaining_slot_margin(tmp_path,monkeypatch):
    a,r,p=origin(monkeypatch,tmp_path)
    alloc=dict(job_id='123',start_epoch=FIRST+3000,end_epoch=FIRST+5640)
    assert a.admit_replacement(alloc,now=FIRST+3010)==FIRST+5640
    for changes,now in [({'end_epoch':FIRST+5700},FIRST+3010),({},FIRST+3200),
                         ({'start_epoch':FIRST+10000,'end_epoch':FIRST+12640},FIRST+10000),
                         ({'job_id':'49072485'},FIRST+3010),({},FIRST-1)]:
        with pytest.raises(ValueError):a.admit_replacement(dict(alloc,**changes),now=now)

def test_preserved_namespaces_count_once_and_source_cap(tmp_path,monkeypatch):
    from scripts.z1_depth_campaign031 import admission as a
    roots=[tmp_path/str(i) for i in range(4)]
    for root in roots:
        root.mkdir();(root/'source').mkdir();(root/'source'/'x').write_bytes(b'12345')
    monkeypatch.setattr(a,'SCOPE_ROOTS',tuple(map(str,roots)))
    assert a.scope_usage()==dict(total_bytes=20,source_assets_controls_bytes=20,failed_qualification_bytes=0,contingency_liabilities_bytes=0,remaining_contingency_bytes=256*m.MIB)
    monkeypatch.setattr(a,'SOURCE_CAP',20)
    with pytest.raises(ValueError,match='source'):a.scope_usage(additional_inputs=1)
    (roots[0]/'source'/'x').unlink();(roots[0]/'source').rmdir();roots[0].rmdir()
    with pytest.raises(ValueError,match='preserved'):a.scope_usage()

@pytest.mark.parametrize('deadline',[None,'N/A','2026-11-01T05:53:54'])
def test_qualifier_requires_fixed_latest_end_scheduler_deadline(monkeypatch,deadline):
    row=scheduler(monkeypatch);row['Deadline']=deadline
    with pytest.raises(ValueError,match='Deadline'):life.allocation('123',row['JobName'])


def test_quota_uses_retained_and_successor_consumption_once(tmp_path,monkeypatch):
    a,r,p=origin(monkeypatch,tmp_path)
    roots=[tmp_path/str(i) for i in range(4)]
    for root in roots:root.mkdir();(root/'input').write_bytes(b'12345')
    monkeypatch.setattr(a,'SCOPE_ROOTS',tuple(map(str,roots)))
    monkeypatch.setattr(life.os,'getxattr',lambda _,key:str(20 if key=='ceph.dir.rbytes' else m.CAP_BYTES+3*m.GIB).encode(),raising=False)
    result=life.current_quota()
    assert result['used031_bytes']==20 and result['required_free_bytes']==m.CAP_BYTES+3*m.GIB-20


def test_qualify_rejects_late_origin_before_clock_claim_or_science(tmp_path,monkeypatch):
    from scripts.z1_depth_campaign031 import worker as w
    a,r,p=origin(monkeypatch,tmp_path);calls=[]
    monkeypatch.setattr(w,'verify_package',lambda _:None);monkeypatch.setattr(life,'validate_grant',lambda *x,**k:None)
    monkeypatch.setattr(life,'current_quota',lambda:None);monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setattr(life,'allocation',lambda *x:dict(job_id='123',start_epoch=FIRST+10000,end_epoch=FIRST+12640))
    monkeypatch.setattr(w.time,'time',lambda:FIRST+10001)
    monkeypatch.setattr(life,'start_clock',lambda *x:calls.append('clock'))
    monkeypatch.setattr(w,'run_one',lambda *x:calls.append('science'))
    with pytest.raises(ValueError):w.qualify(r,{})
    assert calls==[]


def failed_lineage(tmp_path,monkeypatch):
    import shutil
    a,r,p=origin(monkeypatch,tmp_path)
    roots=[tmp_path/('root-'+str(i)) for i in range(6)]
    for root in roots:root.mkdir()
    saved=EVIDENCE/'namespace-diagnosis031/saved-failure49072528'
    for name in ('claims','intents','qualification'):
        shutil.copytree(saved/name,roots[2]/name)
    for name in ('clock.json','manifest.json'):
        shutil.copyfile(saved/name,roots[2]/name)
    monkeypatch.setattr(a,'PRESERVED_ROOTS',tuple(map(str,roots[:4])))
    monkeypatch.setattr(a,'SCOPE_ROOTS',tuple(map(str,roots)))
    return a,roots


def test_actual_consumed_technical_claims_are_retained_and_truthful(tmp_path,monkeypatch):
    a,roots=failed_lineage(tmp_path,monkeypatch)
    proof=a.prove_unused()
    assert proof['training_evaluation_claims_checkpoints']==0
    assert proof['gate_claims']==1 and proof['child_claims']==1
    assert proof['failed_technical_job_ids']==['49072485','49072528']
    assert len(a.authenticate_second()['failed_gate_files'])==14
    (roots[2]/'private').mkdir();(roots[2]/'private/empty').mkdir()
    assert a.prove_unused()==proof  # empty private directories are not starts


@pytest.mark.parametrize('which',['first','second'])
@pytest.mark.parametrize('fault',['missing','bytes','job','counts','clock'])
def test_both_records_authenticate_before_clock_or_admission(tmp_path,monkeypatch,which,fault):
    a,r,p=origin(monkeypatch,tmp_path)
    q=Path(a.ORIGIN_REF['path'] if which=='first' else a.SECOND_REF['path'])
    if fault=='missing':q.unlink()
    elif fault=='bytes':q.write_bytes(q.read_bytes()+b' ')
    else:
        bad=f.read_json(q)
        if fault=='job':bad['accounting']['job_id']='123'
        elif fault=='counts':bad['remote_model_or_physics_starts']=1
        else:bad['first_allocation_epoch']=FIRST+1
        q.write_bytes(f.canonical(bad)+b'\n')
    with pytest.raises((ValueError,FileNotFoundError)):
        life.start_clock(tmp_path/'clock.json',f.digest(r),'123',FIRST+3000,host='gn52')
    assert not (tmp_path/'clock.json').exists()
    with pytest.raises((ValueError,FileNotFoundError)):
        a.admit_replacement(dict(job_id='123',start_epoch=FIRST+3000,end_epoch=FIRST+5640),now=FIRST+3001)


@pytest.mark.parametrize('fault',['extra_gate','empty_extra_gate','changed_claim','nonzero_counts',
    'checkpoint','evalclaim','v1claim','clock','oldmanifest','receipt','symlink','fifo'])
def test_retained_failed_inventory_rejects_added_changed_or_scientific_members(tmp_path,monkeypatch,fault):
    import os
    a,roots=failed_lineage(tmp_path,monkeypatch);old=roots[2]
    if fault=='extra_gate':(old/'qualification/gate-1.json').write_text('{}')
    elif fault=='empty_extra_gate':(old/'qualification/gate-1').mkdir()
    elif fault=='changed_claim':(old/'claims/gate-0.json').write_text('{}')
    elif fault=='nonzero_counts':(old/'qualification/gate-0/actual-start-counts.json').write_text('{"updates":1}')
    elif fault=='checkpoint':(old/'outputs').mkdir();(old/'outputs/model_99.pt').write_bytes(b'x')
    elif fault=='evalclaim':(roots[3]/'claims').mkdir();(roots[3]/'claims/case-0.json').write_text('{}')
    elif fault=='v1claim':(roots[0]/'claims').mkdir();(roots[0]/'claims/gate-0.json').write_text('{}')
    elif fault=='clock':(old/'clock.json').write_text('{}')
    elif fault=='oldmanifest':(old/'manifest.json').write_text('{}')
    elif fault=='receipt':(old/'qualification-receipt.json').write_text('{}')
    elif fault=='symlink':(old/'qualification/link').symlink_to(old/'clock.json')
    else:os.mkfifo(old/'qualification/pipe')
    with pytest.raises(ValueError):a.prove_unused()


@pytest.mark.parametrize('job',['49072485','49072528'])
def test_third_allocation_cannot_reuse_either_prior_job(tmp_path,monkeypatch,job):
    a,r,p=origin(monkeypatch,tmp_path)
    with pytest.raises(ValueError):life.start_clock(tmp_path/'clock.json',f.digest(r),job,FIRST+3000,host='gn52')


def test_third_clock_binds_both_failures_and_cannot_reset(tmp_path,monkeypatch):
    a,r,p=origin(monkeypatch,tmp_path)
    clock=life.start_clock(tmp_path/'clock.json',f.digest(r),'123',FIRST+3000,host='gn52')
    assert a.validate_clock(clock,f.digest(r))==clock
    for key,value in [('second_origin',{}),('prior_technical_allocation',{}),
                      ('technical_allocation',dict(job_id='49072528',start_epoch=FIRST+3000)),
                      ('first_allocation_epoch',FIRST+3000)]:
        with pytest.raises(ValueError):a.validate_clock(dict(clock,**{key:value}))


def test_fixed_six_root_components_and_failed_evidence_contingency(tmp_path,monkeypatch):
    a,roots=failed_lineage(tmp_path,monkeypatch)
    assert len(a.origin_spec()['preserved_roots'])==4 and len(a.SCOPE_ROOTS)==6
    assert m.LIMITS['source_assets_controls']==a.SOURCE_CAP==768*m.MIB
    assert m.LIMITS['contingency']==256*m.MIB and m.CAP_BYTES==7918845952
    for root in roots:(root/'source').mkdir();(root/'source/a').write_bytes(b'12345')
    used=a.scope_usage()
    failed=life.tree_bytes(roots[2]/'qualification')
    assert used['total_bytes']==sum(life.tree_bytes(root) for root in roots)
    assert used['failed_qualification_bytes']==failed
    assert used['remaining_contingency_bytes']==256*m.MIB-failed
    assert a.scope_usage(contingency_liabilities=256*m.MIB-failed)['remaining_contingency_bytes']==0
    with pytest.raises(ValueError,match='contingency'):a.scope_usage(contingency_liabilities=256*m.MIB-failed+1)
    with pytest.raises(ValueError,match='source'):a.scope_usage(additional_inputs=a.SOURCE_CAP-used['source_assets_controls_bytes']+1)
    roots[2].rename(tmp_path/'missing-retained')
    with pytest.raises(ValueError,match='preserved'):a.scope_usage()


def test_dangling_receipt_and_scope_alias_are_not_absence(tmp_path,monkeypatch):
    a,roots=failed_lineage(tmp_path,monkeypatch)
    (roots[2]/'qualification-receipt.json').symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError):a.prove_unused()
    roots[5].rmdir();roots[5].symlink_to(tmp_path/'absent-root')
    with pytest.raises(ValueError):a.scope_usage()


def test_first_and_second_eval_cannot_contain_receipts(tmp_path,monkeypatch):
    a,roots=failed_lineage(tmp_path,monkeypatch)
    (roots[1]/'qualification-receipt.json').write_text('{}')
    with pytest.raises(ValueError):a.prove_unused()


def test_live_quota_monitor_authenticates_both_records_before_any_claim(tmp_path,monkeypatch):
    a,r,p=origin(monkeypatch,tmp_path)
    roots=[tmp_path/('quota-'+str(i)) for i in range(6)]
    for root in roots:root.mkdir()
    monkeypatch.setattr(a,'SCOPE_ROOTS',tuple(map(str,roots)))
    monkeypatch.setattr(life.os,'getxattr',lambda _,key:str(0 if key=='ceph.dir.rbytes' else m.CAP_BYTES+3*m.GIB).encode(),raising=False)
    Path(a.SECOND_REF['path']).unlink()
    with pytest.raises(FileNotFoundError):life.current_quota()
