"""Actual archived8 metadata plus synthetic mutations; never tensors or claims."""
import copy,importlib.util,json
from pathlib import Path
import pytest
D=Path(__file__).resolve().parent;N=D.parent
sp=importlib.util.spec_from_file_location('screen_reader',D/'measurement_reader.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def data(i):
 files=json.loads((N/'ACTUAL-EIGHT-METADATA001.json').read_text())['files'];payload=json.loads(json.loads((N/'ACTUAL-EIGHT-PAYLOAD-METADATA002.json').read_text())['stdout'])['cells'][i]
 manifest=copy.deepcopy(files['manifest.json']['value']);complete=copy.deepcopy(files[f'runs/cell-{i:02d}/train-complete.json']['value']);initial=files[f'runs/cell-{i:02d}/initial-state.json'];experiment=initial['value']['contract'];setting=f"credit{experiment['credit_deadline_ms']}_{experiment['critic_mode']}"
 p=dict(policy_id=f'{setting}-seed2026102901',setting=setting,cell_index=i,training_seed=2026102901,credit_deadline_ms=experiment['credit_deadline_ms'],critic_mode=experiment['critic_mode'],source_manifest_sha256='eb504f5fb772932f11084e29fc88ef0943a2259756d096f1853ed9c1f48a6a06',training_manifest_sha256=files['manifest.json']['sha256'],checkpoint_sha256=payload['checkpoint_sha256'],impact_speed_contract=payload['infos']['impact_speed_contract'])
 # Metadata-only terminal shape derived from actual completion. No physical
 # fixture/claim is produced or represented as a real terminal file.
 terminal=dict(status='COMPLETE',returncode=0,cell=i,allocation=complete['allocation'])
 audit=dict(status='PASS_FINAL_SCREEN_ARTIFACTS',policy_id=p['policy_id'],setting=setting,training_seed=2026102901,iteration=99,common_step_counter=2400,checkpoint_sha256=p['checkpoint_sha256'],training_manifest_sha256=p['training_manifest_sha256'],canonical_source_map_sha256=p['source_manifest_sha256'],artifacts=complete['artifacts'])
 return p,manifest,complete,terminal,audit,initial

@pytest.mark.parametrize('i',range(8))
def test_actual_eight_manifest_payload_and_artifacts_domains(i):
 p,manifest,c,t,a,initial=data(i)
 assert p['training_manifest_sha256']!=p['source_manifest_sha256']
 assert m.validate_training_metadata(p,manifest,p['training_manifest_sha256'],c,t,a)==c['artifacts']
 assert m.validate_initial_marker(p,initial['value'],initial['sha256'],c['artifacts'])==p['impact_speed_contract']['credit_critic_experiment']
 assert p['impact_speed_contract']['observation_widths']['critic']==(43 if i%2==0 else 47)

@pytest.mark.parametrize('bad',['worker','source_map','manifest_file','cell','terminal','artifact','audit_status','audit_domain'])
def test_training_provenance_mutations_refused(bad):
 p,manifest,c,t,a,initial=data(1)
 if bad=='worker':manifest['worker_sha256']='60507ea8023fdb713cacc16c5ccc57a35ebdaad2f05dd2e679b65a071962f05a'
 if bad=='source_map':p['source_manifest_sha256']=p['training_manifest_sha256']
 if bad=='manifest_file':p['training_manifest_sha256']=p['source_manifest_sha256']
 if bad=='cell':c['cell']=0
 if bad=='terminal':t['returncode']=1
 if bad=='artifact':c['artifacts']['common_step_counter']=48
 if bad=='audit_status':a['status']='PASS_FINAL_PILOT_ARTIFACTS'
 if bad=='audit_domain':a['training_manifest_sha256']=p['source_manifest_sha256']
 with pytest.raises(ValueError):m.validate_training_metadata(p,manifest,p['training_manifest_sha256'],c,t,a)

@pytest.mark.parametrize('bad',['full_instead_short','deadline','width','initial_sha'])
def test_initial_short_marker_must_match_actual_experiment(bad):
 p,manifest,c,t,a,initial=data(1);row=copy.deepcopy(initial['value']);pin=initial['sha256']
 if bad=='full_instead_short':row['contract']=p['impact_speed_contract']
 if bad=='deadline':row['contract']['credit_deadline_ms']=200
 if bad=='width':row['contract']['critic_width']=43
 if bad=='initial_sha':pin='0'*64
 with pytest.raises(ValueError):m.validate_initial_marker(p,row,pin,c['artifacts'])
