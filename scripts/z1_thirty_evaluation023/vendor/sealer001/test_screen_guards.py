"""Synthetic file-only guard checks; no real claims or qualification generated."""
import ast,copy,json
from pathlib import Path
import pytest
from test_screen_metadata import m,D,N,data
A=N.parent.parent
E=N  # Fork's own hardware guard; original test used byte-identical old12 guard.

def put(p,v):p.write_text(json.dumps(v,sort_keys=True));return {'path':str(p),'sha256':m.sha(p)}

def claim_fixture(tmp):
 p=data(0)[0];root=tmp/'z1-screen-evaluation-SYNTHETIC'/'runs'/p['policy_id'];root.mkdir(parents=True);claims=root.parent.parent/'claims';claims.mkdir()
 runtime=tmp/'SYNTHETIC-runtime';runtime.write_text('SYNTHETIC');runtime_ref={'path':str(runtime),'sha256':m.sha(runtime)}
 guard=E/'pilot-compat001/qualified007/supervisor/cpu_supervisor.py';gref={'path':str(guard),'sha256':'b6a2d726bb58ed63451567089ec5e9b295bed3b979f427091d42becb5c03c757'}
 bank={'path':'/SYNTHETIC-bank.json','sha256':m.BANK_SHA};key=p['policy_id']+'-bank32';path=claims/(key+'.json');ctx={'parent':90001,'slot':0}
 intent=dict(policy=p,supervisor_context=ctx,package_files=[gref],claim_path=str(path),claim_key=key,reset_bank=bank,runtime_files=[runtime_ref],runtime_roles={'synthetic':str(runtime)})
 allocation=dict(hostname='cn0724',SLURM_JOB_ID='SYNTHETIC_ONLY',SLURM_CPUS_PER_TASK='2',SLURM_JOB_NODELIST='cn0724',SLURM_MEM_PER_NODE='8192')
 sup=put(path.with_suffix('.supervisor.json'),dict(schema='z1_screen_eval_supervisor_claim_v1',parent=90001,allocation=allocation,intent_binding_sha256=m.digest(intent),slot=0,retry_allowed=False));ctx['supervisor_claim_sha256']=sup['sha256'];intentref=put(root/'intent.json',intent)
 put(path,dict(schema='z1_screen_eval_physical_claim_v1',attempts=32,intent_sha256=intentref['sha256'],policy_id=p['policy_id'],bank=bank,retry_allowed=False))
 hardware=dict(status='PASS',before_environment_construction=True,allocation=allocation,affinity=[0,1],torch_intraop=2,torch_interop=2,thread_environment={k:'2'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},runtime_origins={'synthetic':dict(path=str(runtime.resolve()),sha256=m.sha(runtime))})
 put(root/'hardware-prolog.json',hardware)
 return intent,root,intentref['sha256'],hardware,path

def test_screen_claim_and_hardware_join_valid_only_synthetic(tmp_path):
 args=claim_fixture(tmp_path);r=m.hardware_and_claims(*args[:4]);assert set(r)=={'physical','supervisor','hardware_guard'}

@pytest.mark.parametrize('bad',['pilot_schema','wrong_attempts','wrong_bank','wrong_intent','wrong_parent','wrong_hardware','wrong_origin'])
def test_screen_claim_hardware_mutations_refused(tmp_path,bad):
 intent,root,pin,hardware,path=claim_fixture(tmp_path)
 if bad in('pilot_schema','wrong_attempts','wrong_bank','wrong_intent'):
  c=m.read(path)
  if bad=='pilot_schema':c['schema']='z1_pilot_eval_physical_claim_v1'
  if bad=='wrong_attempts':c['attempts']=31
  if bad=='wrong_bank':c['bank']['sha256']='0'*64
  if bad=='wrong_intent':c['intent_sha256']='0'*64
  put(path,c)
 if bad=='wrong_parent':intent['supervisor_context']['parent']=90002
 if bad=='wrong_hardware':hardware['torch_intraop']=4;put(root/'hardware-prolog.json',hardware)
 if bad=='wrong_origin':hardware['runtime_origins']['synthetic']['sha256']='0'*64;put(root/'hardware-prolog.json',hardware)
 with pytest.raises(ValueError):m.hardware_and_claims(intent,root,pin,hardware)

def test_qualified_numeric_functions_are_exact_c791_ast_and_files():
 old=ast.parse((D/'PRESERVED-PILOT-READER-c791.py.txt').read_text());new=ast.parse((D/'measurement_reader.py').read_text());names=('dependencies','realized_bank','audit_measurements')
 nodes=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for name in names:assert ast.dump(nodes(old)[name],include_attributes=False)==ast.dump(nodes(new)[name],include_attributes=False)
 for name,pin in m.PINS.items():assert m.sha(D/'qualified-origin'/name)==pin

def test_old_pilot_request_schema_refused_before_any_producer_read():
 req=dict(schema='z1_pilot_case32_audit_request_v1',raw_root='/SYNTHETIC',intent={},expected_policy={},independent_backup={},pilot_contract={})
 with pytest.raises(ValueError,match='closed'):m.audit_case(req)
