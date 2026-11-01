"""Synthetic file-only releases/Q receipts; no real authority/claims/physics."""
import copy,json,runpy,sys
from pathlib import Path
import pytest
from test_screen_metadata import m,D,N

VIRTUAL={}

@pytest.fixture(autouse=True)
def isolated_file_reads(monkeypatch):
 VIRTUAL.clear();original_bound=m.bound;original_read=m.read
 def mapped_bound(ref,root=None):
  physical=VIRTUAL.get(ref['path'])
  return original_bound(dict(ref,path=str(physical))if physical else ref,root)
 monkeypatch.setattr(m,'bound',mapped_bound)
 monkeypatch.setattr(m,'read',lambda p:original_read(VIRTUAL.get(str(p),p)))

def put(p,v):
 logical=str(p);physical=VIRTUAL.get(logical,p)
 physical.write_text(json.dumps(v,sort_keys=True,allow_nan=False));return dict(path=logical,sha256=m.sha(physical))

def fixture(tmp,index=0,main=False):
 pcpath=N/'pilot-compat001/pilot_contract.py';fpath=N/'finite-worker001/finite_contract.py'
 sys.modules['pilot_contract']=m.source(pcpath,m.sha(pcpath),'pilot_contract')
 sys.path.insert(0,str(fpath.parent));sys.path.insert(0,str(pcpath.parent))
 finite=m.source(fpath,m.sha(fpath),'screen_fixture_finite');sys.modules['finite_contract']=finite
 r=runpy.run_path(str(fpath.with_name('test_finite_contract.py')))['release']()
 root=Path(r['campaign_root']);physical=tmp/'SYNTHETIC-files';physical.mkdir(parents=True)
 for relative in ('authority/release-canary001.json','authority/release-main001.json','evidence/software-qualification001/receipt.json'):
  dest=physical/relative;dest.parent.mkdir(parents=True,exist_ok=True);VIRTUAL[str(root/relative)]=dest
 script=tmp/'software-qualification001';script.mkdir();sref=put(script/'screen_software_qualification.py',{'SYNTHETIC':'NOT_EXECUTABLE'});cref=put(script/'qualification-config001.json',{'SYNTHETIC':'NOT_QUALIFIED'})
 r.update(package_files=[sref,cref,dict(path=str(pcpath),sha256=m.sha(pcpath)),dict(path=str(fpath),sha256=m.sha(fpath))])
 original_ref=put(root/'authority/release-canary001.json',r)
 q=dict(schema='z1_screen_software_qualification_v1',status='PASS_FINAL99_CURRENT_EVENT_CPU32_RESET_ONLY',script_sha256=sref['sha256'],config_sha256=cref['sha256'],source_map_sha256=r['slots'][0]['policy']['source_manifest_sha256'],reset_bank_sha256=r['reset_bank']['sha256'],policy_calls=0,physics_integration_steps=0,scientific_attempts=0,reset_contexts=64,guard_hits=[],release=original_ref,allocation_sha256=finite.allocation_digest(r),allocation=dict(job_id='123',hostname='cn0724',cpus=2,mem_mib=8192,affinity=[1,2],torch_intraop=2,torch_interop=2,thread_environment=dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')),cases=[dict(policy_id=r['slots'][i]['policy']['policy_id'],critic_width=43 if i==0 else 47,loaded_counter=2400,actual_saved_iteration=99,actor_only_load=True,actor_unchanged=True,normalizer_unchanged=True)for i in(0,1)])
 qref=put(root/'evidence/software-qualification001/receipt.json',q)
 if main:
  r=copy.deepcopy(r);r['authority']=dict(path='/SYNTHETIC-main-approval.json',sha256='d'*64);r['canary_receipt']=dict(path='/SYNTHETIC-canary-audit.json',sha256='e'*64)
 release_ref=put(root/'authority/release-main001.json',r)if main else original_ref
 ctx=dict(slot=index,parent=90001,release_path=release_ref['path'],release_sha256=release_ref['sha256'],software_qualification=qref)
 intent=finite.build_intent(r,index,deadline=12345.,context=ctx);raw=Path(intent['output_root']);dest=physical/f'intent-{index}.json';VIRTUAL[str(raw/'intent.json')]=dest;iref=put(raw/'intent.json',intent)
 hardware={'allocation':{'SLURM_JOB_ID':'123'if index==0 else'456'}}
 return r,intent,hardware,q,iref,finite

def save_q(intent,q):
 ref=put(Path(intent['supervisor_context']['software_qualification']['path']),q);intent['supervisor_context']['software_qualification']=ref

def test_pinned_real_software_gate_accepts_original_and_main_phase_file_bindings(tmp_path):
 r,intent,hardware,q,iref,finite=fixture(tmp_path)
 assert m.software_qualification(intent,hardware)['receipt']==intent['supervisor_context']['software_qualification']
 r,intent,hardware,q,iref,finite=fixture(tmp_path/'later',index=1,main=True)
 assert m.software_qualification(intent,hardware)['original_qualification_release']==q['release']

@pytest.mark.parametrize('bad',['job','width','counter','calls','guard','release_bytes','intent_extra','original_allocation'])
def test_software_qualification_context_and_actual_original_release_fail_closed(tmp_path,bad):
 r,intent,hardware,q,iref,finite=fixture(tmp_path,index=1 if bad=='original_allocation'else 0,main=bad=='original_allocation')
 if bad=='job':hardware['allocation']['SLURM_JOB_ID']='999'
 elif bad=='width':q['cases'][1]['critic_width']=43;save_q(intent,q)
 elif bad=='counter':q['cases'][0]['loaded_counter']=48;save_q(intent,q)
 elif bad=='calls':q['physics_integration_steps']=1;save_q(intent,q)
 elif bad=='guard':q['guard_hits']=['FORBIDDEN'];save_q(intent,q)
 elif bad=='release_bytes':put(Path(q['release']['path']),{})
 elif bad=='intent_extra':intent['unexpected']=True
 else:
  original=m.read(q['release']['path']);original['slots'][2]['runtime_identity']={'changed':True};q['release']=put(Path(q['release']['path']),original);save_q(intent,q)
 with pytest.raises(ValueError):m.software_qualification(intent,hardware)

class AuditRequired(RuntimeError):pass

def test_canary_phase_refs_only_reach_mandatory_audit_not_status_only(tmp_path,monkeypatch):
 r,intent,hardware,q,iref,finite=fixture(tmp_path);main=copy.deepcopy(r);main['authority']=dict(path='/SYNTHETIC-new-phase.json',sha256='f'*64);main['canary_receipt']=dict(path='/SYNTHETIC-stored-canary.json',sha256='e'*64)
 receipt={'status':'PASS','request':dict(expected_policy=r['slots'][0]['policy'],raw_root=intent['output_root'],intent=iref)}
 def actual_audit_required(request):raise AuditRequired('real audit_case must execute next')
 monkeypatch.setattr(m,'audit_case',actual_audit_required)
 with pytest.raises(AuditRequired):m.verify_canary(receipt,main)

@pytest.mark.parametrize('bad',['cohort','resources','package','intent'])
def test_canary_same_baseline_and_bank_do_not_qualify_changed_allocation(tmp_path,bad):
 r,intent,hardware,q,iref,finite=fixture(tmp_path);main=copy.deepcopy(r)
 if bad=='cohort':main['slots'][2]['policy']['checkpoint_sha256']='a'*64
 elif bad=='resources':main['max_concurrent']=6
 elif bad=='package':main['package_files'][0]['sha256']='b'*64
 else:intent['unexpected']=True;iref=put(Path(iref['path']),intent)
 receipt={'status':'PASS','request':dict(expected_policy=r['slots'][0]['policy'],raw_root=intent['output_root'],intent=iref)}
 with pytest.raises(ValueError):m.verify_canary(receipt,main)
