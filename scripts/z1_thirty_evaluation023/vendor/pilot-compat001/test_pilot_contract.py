"""Actual CPU-inspected eight-screen metadata fixtures; no tensors imported."""
import copy,json,unittest
from pathlib import Path
from pilot_contract import validate_record,validate_payload,expected_marker,SETTINGS
N=Path(__file__).resolve().parent.parent
ACTUAL=json.loads(json.loads((N/'ACTUAL-EIGHT-PAYLOAD-METADATA002.json').read_text())['stdout'])
META=json.loads((N/'ACTUAL-EIGHT-METADATA001.json').read_text())
def record(setting=SETTINGS[0]):
 i=SETTINGS.index(setting);cell=ACTUAL['cells'][i];c=cell['infos']['impact_speed_contract'];audit=json.loads((N/'CONTRACT-AUDIT001.json').read_text())['cells'][i]
 return dict(schema='z1_credit_screen_final_policy_v1',policy_id=setting+'-seed2026102901',setting=setting,cell_index=i,credit_deadline_ms=c['credit_critic_experiment']['credit_deadline_ms'],critic_mode=c['credit_critic_experiment']['critic_mode'],training_seed=2026102901,task='Unitree-Z1-Hammer-Quintic-ThreeStarts-SplineSpeedCurriculum-Observation-Age-LowNoise-DeferredCredit-Horizon-extended200-Credit'+str(c['credit_critic_experiment']['credit_deadline_ms'])+'-Critic'+c['credit_critic_experiment']['critic_mode'].title(),checkpoint_path=cell['checkpoint_path'],checkpoint_sha256=cell['checkpoint_sha256'],source_manifest_sha256='eb504f5fb772932f11084e29fc88ef0943a2259756d096f1853ed9c1f48a6a06',training_manifest_sha256='7ca22bbd866fa069d8ffc520000b8e966bddf9693df70f03963aac7d6ce0fa79',training_completion_sha256=audit['completion_sha256'],final_iteration=99,completed_updates=100,common_step_counter=2400,impact_speed_contract=copy.deepcopy(c))
class FakeTensor:
 def __init__(self,shape,finite=True):self.shape=tuple(shape);self.finite=finite
 def detach(self):return self
 def isfinite(self):return self
 def all(self):return self
 def item(self):return self.finite

def payload(r):
 cell=ACTUAL['cells'][r['cell_index']]
 return dict(iter=99,infos=dict(env_state={'common_step_counter':2400},impact_speed_contract=r['impact_speed_contract']),actor_state_dict={k:FakeTensor(v)for k,v in cell['actor_shapes'].items()},critic_state_dict={k:FakeTensor(v)for k,v in cell['critic_shapes'].items()})
class Tests(unittest.TestCase):
 def test_actual_all8_exactmetadata(self):
  for setting in SETTINGS:
   r=record(setting);validate_record(r);validate_payload(payload(r),r)
 def test_identity_and_marker_domains(self):
  for key,value in [('setting','vic_baseline'),('training_seed',2026102902),('cell_index',1),('credit_deadline_ms',100),('critic_mode','event'),('final_iteration',499),('common_step_counter',48),('training_manifest_sha256','eb504f5fb772932f11084e29fc88ef0943a2259756d096f1853ed9c1f48a6a06'),('source_manifest_sha256','7ca22bbd866fa069d8ffc520000b8e966bddf9693df70f03963aac7d6ce0fa79')]:
   r=record();r[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):validate_record(r)
  r=record();r['unexpected']=True
  with self.assertRaises(ValueError):validate_record(r)
 def test_critic_and_legacyN_not_pilot(self):
  for mutation in ('width','marker','N','pilot'):
   r=record(SETTINGS[1]);c=r['impact_speed_contract']
   if mutation=='width':c['observation_widths']['critic']=43
   elif mutation=='marker':c['credit_critic_experiment']['critic_width']=43
   elif mutation=='N':c['spline_learning_variant']['variant']='A'
   else:c['spline_exploration_pilot']={}
   with self.assertRaises(ValueError):validate_record(r)
 def test_payload_counter_fullcontract_and_nonfinite(self):
  r=record()
  for mutation in ('counter','contract','finite'):
   p=payload(r)
   if mutation=='counter':p['infos']['env_state']['common_step_counter']=48
   elif mutation=='contract':p['infos']['impact_speed_contract']={}
   else:p['actor_state_dict']['distribution.std_param'].finite=False
   with self.assertRaises(ValueError):validate_payload(p,r)
 def test_saved_state_shapes_are_exact_actual_current_and_event(self):
  for setting in (SETTINGS[0],SETTINGS[1]):
   r=record(setting)
   for mutation in ('actor_input','actor_output','critic_input','std','normalizer','missing','extra','nontensor'):
    p=payload(r)
    if mutation=='actor_input':p['actor_state_dict']['mlp.0.weight'].shape=(256,47)
    elif mutation=='actor_output':p['actor_state_dict']['mlp.6.weight'].shape=(12,64)
    elif mutation=='critic_input':p['critic_state_dict']['mlp.0.weight'].shape=(256,47 if setting==SETTINGS[0]else 43)
    elif mutation=='std':p['actor_state_dict']['distribution.std_param'].shape=(12,)
    elif mutation=='normalizer':p['critic_state_dict']['obs_normalizer.physical._mean'].shape=(1,43)
    elif mutation=='missing':p['actor_state_dict'].pop('mlp.0.weight')
    elif mutation=='extra':p['critic_state_dict']['extra']=FakeTensor([1])
    else:p['actor_state_dict']['distribution.std_param']=[.3]*13
    with self.subTest(setting=setting,mutation=mutation),self.assertRaises(ValueError):validate_payload(p,r)
if __name__=='__main__':unittest.main()
