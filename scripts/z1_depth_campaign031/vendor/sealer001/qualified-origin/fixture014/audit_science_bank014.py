"""Read-only terminal science160 file audit. Never runs a policy or creates seals.

The e1 reader, literal endpoint arithmetic and classification stay unchanged.
Only the closed case/report arguments of e1/15e are adapted in memory by AST.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,json,math,re,sys,types

CAMPAIGN='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
ADMISSION=CAMPAIGN+'/admission-staged-integration002/runtime-budget007'
RELEASE=ADMISSION+'/science/release.json'
RELEASE_SHA='ab4970f492f43056df91d957541e9d8bb4e4d6edde7b1bb2394cd31e8938165c'
MANIFEST_SHA='251f1f2c4b8465733dcd6e22ac3acf302fd89bb2f1eca963c458a66dd806b5e6'
SUPERVISOR_PATH=CAMPAIGN+'/package-runtime-budget007/supervisor/cpu_supervisor.py'
SUPERVISOR_SHA='f4b40175786581063848c8b8d9b40651f203f9670418fcbe6d3da7274f3af449'
SEALER_SHA='e1bc82fe9feb7a57ce6ea220fd502d07474eeb1397bd46da80af530e1ee52641'
OWN_SHA='15e220f75ae3bdf1d16454dfb68a5b98b47d27192e035f44c8afdaabcfed6e45'
TRACE_PRODUCER_SHA='16137ae895ec33f9898b4621b067eb6b10f665d997d14fabf062fc42488b147e'
TRACE_SEMANTICS='onset sample included; literal200ms at slot800; native boundary marked; no precontact dense trace; padding invalid beyond valid_counts'
SCIENCE=(1,2,4,5,7,8,10,11,13,14,16,17)
CASES=['original-route-1','start-minus10mm','original-route+0','start-plus10mm','original-route+1']
ROUTES=(-1,0,0,0,1)
POLICIES=('native50-seed2026102301','native50-seed2026102301','extended200-seed2026102301','extended200-seed2026102301','extended200-seed2026102302','extended200-seed2026102302','native50-seed2026102302','native50-seed2026102302','native50-seed2026102303','native50-seed2026102303','extended200-seed2026102303','extended200-seed2026102303')
FORBIDDEN={'torch','mujoco','mujoco_warp','mjlab','rsl_rl'}

def need(ok,message):
 if not ok:raise ValueError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def declared_equal(actual,expected):return canonical({k:actual.get(k)for k in expected})==canonical(expected)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def source(path,pin,name):
 path=Path(path)
 need(path.is_absolute()and not any(p.is_symlink()for p in(path,*path.parents)),'real absolute pinned source required')
 body=path.read_bytes();need(hashlib.sha256(body).hexdigest()==pin,'audited source SHA differs')
 module=types.ModuleType(name);module.__file__=str(path);exec(compile(body,str(path),'exec'),module.__dict__)
 return module,body

class Adapt(ast.NodeTransformer):
 def __init__(self,kind):self.kind=kind;self.counts={}
 def hit(self,key):self.counts[key]=self.counts.get(key,0)+1
 def visit_Constant(self,node):
  if node.value=='original-route+1':
   self.hit(self.kind+'_case_literal');return ast.copy_location(ast.Name(id='case_id',ctx=ast.Load()),node)
  return node
 def visit_Assign(self,node):
  node=self.generic_visit(node)
  if self.kind=='e1'and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='reports':
   self.hit('e1_reports_assignment');node.value=ast.parse("[row for row in result.get('reports',[]) if row.get('case_id')==case_id]",mode='eval').body
  return node
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if self.kind=='own'and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=='w'and node.func.attr=='audit_raw':
   self.hit('own_audit_call');node.args.append(ast.Name(id='case_id',ctx=ast.Load()))
  return node
 def visit_Compare(self,node):
  original=ast.dump(node,include_attributes=False);node=self.generic_visit(node)
  if self.kind=='own'and original==ast.dump(ast.parse("ep.get('route_sign')==1",mode='eval').body,include_attributes=False):
   self.hit('own_route_compare');node.comparators=[ast.Name(id='route_sign',ctx=ast.Load())]
  return node

def derive(body,name,kind,namespace):
 nodes=[n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef)and n.name==name];need(len(nodes)==1,'one audited function required')
 node=copy.deepcopy(nodes[0]);transform=Adapt(kind);node=transform.visit(node)
 for arg in(['case_id']if kind=='e1'else['case_id','route_sign']):node.args.args.append(ast.arg(arg=arg))
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),kind+'_closed_case_derivative','exec'),namespace)
 return namespace[name],transform.counts

def dependencies(sealer_path,own_path):
 w,ebody=source(sealer_path,SEALER_SHA,'science014_e1');o,obody=source(own_path,OWN_SHA,'science014_15e')
 w.audit_raw,ecounts=derive(ebody,'audit_raw','e1',w.__dict__)
 w.audit_case_measurements,ocounts=derive(obody,'audit_measurements','own',o.__dict__)
 expected=dict(e1_case_literal=2,e1_reports_assignment=1,own_case_literal=1,own_audit_call=1,own_route_compare=1)
 need(ecounts|ocounts==expected,'closed derivative structure changed')
 w.adaptation_counts=ecounts|ocounts;w.metadata=o.metadata;w.validate_terminal=o.validate_terminal
 return w

def expected_target(index):
 need(type(index)is int and index in SCIENCE,'original science slot required')
 pos=SCIENCE.index(index);pid=POLICIES[pos];arm=pid.split('-seed')[0];bank=(20261041,2026101601)[pos%2]
 return dict(schema='z1_horizon_cpu_worker_v1',status='ADMITTED_ONCE',scope='science160',attempts=160,cases=list(CASES),evaluation_seed=bank,retry_allowed=False,resume_allowed=False,policy=dict(policy_id=pid,arm=arm,training_seed=int(pid.split('seed')[1]),native_window_s={'native50':.05,'extended200':.2}[arm]),output_root=CAMPAIGN+'/runs/'+pid+'/science160-bank'+str(bank))

def validate_target(intent,index):
 expected=expected_target(index)
 need(declared_equal(intent,{k:v for k,v in expected.items()if k!='policy'}),'closed original science160 target differs')
 need(declared_equal(intent.get('policy',{}),expected['policy']),'closed training policy identity differs')

def validate_context(context,index):
 need(type(context.get('slot'))is int and declared_equal(context,dict(slot=index,supervisor_path=SUPERVISOR_PATH,supervisor_sha256=SUPERVISOR_SHA,release_path=RELEASE,release_sha256=RELEASE_SHA)),'actual current science supervisor/release context differs')

def distance(a,b):
 need(isinstance(a,list)and len(a)==len(b)and all(type(x)in(int,float)and math.isfinite(x)for x in a),'finite geometry vector required')
 return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def geometry(reset,case_id,pose):
 rows=reset.get('geometry')
 if not case_id.startswith('start-'):
  need(rows is None,'original route cannot substitute an intermediate pose');return
 need(isinstance(rows,list)and len(rows)==32,'all32 realized intermediate poses required')
 need([r.get('world')for r in rows]==list(range(32)),'explicit ordered geometry worlds required')
 for row in rows:
  need(row.get('joint_positions_rad')==pose['executed_joint_positions_rad']and row.get('joint_velocities_rad_s')==[0]*6 and row.get('reset_contact')is False,'realized reset joints/velocity/contact differs')
  need(row.get('coordinate_frame')=='fixed_scene_native_world'and row.get('scene_grid_origin_applied')is False,'fixed scene coordinate frame required')
  for field,target in [('head_world_m',pose['requested_head_m']),('head_local_m',pose['requested_head_m']),('nail_world_m',pose['nail_m']),('reference_head0_w',row['head_world_m']),('reference_target_w',[pose['nail_m'][0],pose['nail_m'][1],pose['nail_m'][2]-.02])]:
   need(distance(row.get(field),target)<=1e-6,'realized reset/reference geometry differs')
  q=row.get('site_quaternion_wxyz');distance(q,[0]*4);need(abs(sum(x*x for x in q)-1)<=1e-6,'unit orientation required')
  a,b,c,d=q;r=[[1-2*(c*c+d*d),2*(b*c-a*d),2*(b*d+a*c)],[2*(b*c+a*d),1-2*(b*b+d*d),2*(c*d-a*b)],[2*(b*d-a*c),2*(c*d+a*b),1-2*(b*b+c*c)]]
  want=pose['requested_orientation_matrix'];relative=[[sum(r[i][k]*want[j][k]for k in range(3))for j in range(3)]for i in range(3)]
  skew=[(relative[2][1]-relative[1][2])/2,(relative[0][2]-relative[2][0])/2,(relative[1][0]-relative[0][1])/2]
  angle=abs(math.atan2(math.sqrt(sum(x*x for x in skew)),max(-1,min(1,(sum(relative[i][i]for i in range(3))-1)/2))))
  need(angle<=1e-5,'realized reset orientation differs')
  for key,limit in [('position_residual_m',1e-6),('orientation_residual_rad',1e-5)]:need(type(row.get(key))in(int,float)and math.isfinite(row[key])and 0<=row[key]<=limit,'saved reset residual invalid')

def producer_reference(w,case,field,filename):
 """Compare the complete native descriptor to its original producer data.

 Mapping and endpoint references remain their original two-field objects.
 Native trace metadata comes from the pinned e1 reader, without NumPy imports.
 All-miss traces may legitimately contain only the four structural arrays.
 """
 path=case/filename;reference=dict(path=str(path),sha256=sha(path))
 if field=='native_onset_trace':
  trace=w.read_trace(path)
  dtype_names={'<f4':'float32','<f8':'float64','<i8':'int64','|b1':'bool'}
  reference.update(arrays={name:dict(shape=list(row['shape']),dtype=dtype_names[row['dtype']])for name,row in trace.items()},
   valid_counts=list(trace['valid_counts']['values']),capacity_per_world=w.CAPACITY,semantics=TRACE_SEMANTICS)
 return reference

def audit_measurements(w,intent,root):
 root=Path(root)
 need(intent.get('cases')==CASES and intent.get('attempts')==160 and intent.get('scope')=='science160'and intent.get('evaluation_seed')in(20261041,2026101601),'complete ordered original science bank required')
 need({p.name for p in root.iterdir()if p.is_dir()}==set(CASES),'missing/extra scientific case directory')
 worker=read(root/'worker-result.json');reports=worker.get('reports',[])
 need(worker.get('attempts')==160 and [r.get('case_id')for r in reports]==CASES,'all five ordered worker case reports required')
 for key,field in [('observer_mappings','observer_mapping'),('native_onset_traces','native_onset_trace'),('safety_endpoint_summaries','safety_endpoint_summary')]:need(canonical(worker.get(key))==canonical([r.get(field)for r in reports]),'worker aggregate producer references differ')
 cases=[];loaded=[]
 for case_id,sign,report_ref in zip(CASES,ROUTES,reports):
  case=root/case_id;report=read(case/'report.json');reset=read(case/'reset-observed.json')
  need(report_ref.get('report_sha256')==sha(case/'report.json'),'case report hash differs')
  references={}
  for field,filename in [('observer_mapping','observer-mapping.json'),('native_onset_trace','onset-native-producers.npz'),('safety_endpoint_summary','inclusive-safety-endpoints.json')]:
   reference=producer_reference(w,case,field,filename);references[field]=reference
   need(canonical(report_ref.get(field))==canonical(report.get(field))==canonical(reference),'case producer reference escaped or swapped')
   for name in ['episodes.json','observer-records.json','native-boundary-records.json']:
    need(all(canonical(row.get(field))==canonical(reference) for row in read(case/name)),'per-world case producer reference differs')
  endpoints=read(case/'inclusive-safety-endpoints.json')
  need(all(canonical(endpoints.get(field))==canonical(references[field])for field in('native_onset_trace','observer_mapping')),'endpoint summary producer reference differs')
  need(report.get('first_episode_route_signs')==[sign]*32 and report.get('measurement',{}).get('protocol')=='native'and report['measurement'].get('physics_dt_s')==.00025,'case native route/physics report differs')
  need(reset.get('case_id')==case_id and reset.get('worlds')==list(range(32)),'reset case/world identity differs')
  hashes=tuple(reset.get(k)for k in('actor_sha256','normalizer_sha256'));need(all(isinstance(h,str)and re.fullmatch('[a-f0-9]{64}',h)for h in hashes),'saved loaded policy hashes required');loaded.append(hashes)
  geometry(reset,case_id,intent.get('pose_catalog',{}).get(case_id))
  result=w.audit_case_measurements(w,intent,root,case_id,sign)
  result.pop('unsafe_allowed_for_technical_qualification',None)
  result.update(case_id=case_id,route_sign=sign,report_sha256=report_ref['report_sha256'])
  endpoints=w.keyed(read(case/'inclusive-safety-endpoints.json')['worlds'],'world_id')
  native_records=w.keyed(read(case/'native-boundary-records.json'),'world_id')
  for row in result['worlds']:
   i=row['world_id'];native=native_records[i]['native']
   row.update(case_id=case_id,evaluation_seed=intent['evaluation_seed'],safety_windows=endpoints[i]['windows'],native_boundary_measurement=native['boundary_measurement'],native_snapshot=native['snapshot'])
   for key in('measured_mps','depth_50ms_m','depth_200ms_m'):
    need(row[key]is None or type(row[key])in(int,float)and math.isfinite(row[key]),'nonfinite recorded utility')
  cases.append(result)
 need(len(set(loaded))==1,'actor/normalizer changed between scientific cases')
 totals={key:sum(c['denominators'][key]for c in cases)for key in('all_first_episodes','contact_801','genuine_no_contact')}
 need(totals['all_first_episodes']==160 and totals['contact_801']+totals['genuine_no_contact']==160,'full160 denominator differs')
 return dict(schema='z1_science_bank_existing_data_audit_v1',cases=cases,denominators=totals,scientific_safety_claim=False,policy_qualification=False,source_pins=dict(sealer=SEALER_SHA,own_auditor=OWN_SHA,supervisor=SUPERVISOR_SHA),adaptation_counts=w.adaptation_counts)

def validate_mapping(mapping,w,source_hashes):
 need(mapping.get('joint_names')==['joint'+str(j)for j in range(1,7)]and all(mapping.get(key)==list(range(6))for key in('entity_ids','native_joint_ids','qpos_ids','dof_ids'))and mapping.get('actuator_ids')==[0,5,1,2,3,4],'actual canonical joint/actuator mapping differs')
 imp=mapping.get('impulse_provenance',{})
 need(imp.get('window_substeps')==200 and imp.get('baseline_subtracted')is True and imp.get('thresholds_nms')==list(w.LOAD_CAPS)and imp.get('enforced_relation')=='maximum(raw_load,held_load)'and imp.get('source_sha256')in source_hashes,'impulse producer semantics differ')

def audit(sealer_path,own_path,slot_index,supervisor_path=SUPERVISOR_PATH):
 need(str(supervisor_path)==SUPERVISOR_PATH,'exact actual f4 supervisor required')
 target=expected_target(slot_index);root=Path(target['output_root'])
 need(root.is_dir()and not any(p.is_symlink()for p in(root,*root.parents)),'actual absolute science raw tree required')
 w=dependencies(sealer_path,own_path);c,_=source(supervisor_path,SUPERVISOR_SHA,'science014_f4')
 hardware=read(root/'hardware-prolog.json');terminal=read(root/'supervisor-terminal.json');w.validate_terminal(terminal,hardware)
 before_meta=w.metadata(root);before=w.inventory(root);intent=read(root/'intent.json');validate_target(intent,slot_index)
 context=intent['supervisor_context'];validate_context(context,slot_index)
 need(sha(RELEASE)==RELEASE_SHA and read(RELEASE).get('manifest',{}).get('sha256')==MANIFEST_SHA,'actual final science release/manifest changed')
 manifest=c.load_release(RELEASE);c.require_selected(manifest,slot_index);slot=manifest['slots'][slot_index]
 need(declared_equal(intent,slot['intent']),'actual typed intent differs from admitted science slot')
 need(intent.get('own_cpu32_seal_path')==slot['qualification_seal']['path']and intent.get('own_cpu32_seal_sha256')==slot['qualification_seal']['sha256'],'admitted prerequisite own32 reference differs')
 c.verify_intent_inventory(intent);c.validate_hardware_prolog(root/'hardware-prolog.json',intent=intent)
 need(read(root/'actual-runtime.json')==intent['runtime_identity'],'actual runtime differs')
 training=c.audit_training(slot['training_bindings'],tensor_audit=False);saved=read(root/'training-audit.json');preflight=read(root/'training-metadata-preflight.json')
 need(saved.get('status')=='PASS'and saved.get('finite_checkpoint')is True and saved.get('policy')==intent['policy']and saved.get('checkpoint_sha256')==intent['policy']['checkpoint_sha256']and saved.get('iteration')==499 and saved.get('common_step_counter')==12000 and saved.get('last_transition_index')==11999 and saved.get('tree_sha256')==training['tree_sha256']==preflight.get('tree_sha256'),'supervised finite tensor-audit provenance differs')
 train_manifest=c.bound(slot['training_bindings']['manifest']);source_map={str(Path(r['path']).resolve().relative_to(Path(intent['source_root']).resolve())):r['sha256']for r in intent['source_files']}
 need(source_map==train_manifest['source_files']and all({r['path']:r['sha256']for r in intent[group]}==train_manifest[group]for group in('runtime_files','asset_files')),'source/runtime/assets differ from own training')
 need(intent['source_identity']==dict(training_manifest_sha256=slot['training_bindings']['manifest']['sha256'],source_tree_sha256=w.tree_digest(source_map)),'source identity differs')
 packages={Path(r['path']).name:r for r in intent['package_files']}
 need(packages.get('onset_trace.py',{}).get('sha256')==TRACE_PRODUCER_SHA,'native trace descriptor producer source differs')
 need(len(packages)==10 and intent['observer_identity']=={name:r['sha256']for name,r in packages.items()}and context['worker_sha256']==packages['horizon_worker.py']['sha256']and intent['reviewed_hook_path']==packages['horizon_integration.py']['path']and intent['reviewed_hook_sha256']==packages['horizon_integration.py']['sha256'],'observer/hook/worker identities differ')
 worker=read(root/'worker-result.json');need(worker.get('status')=='RAW_REVIEW_REQUIRED'and worker.get('attempts')==160 and worker.get('checkpoint_contract_validated_before_overlay')is True and worker.get('final_contract_restored')is True and worker.get('reset_frames_sampled')==0 and worker.get('hardware_prolog_sha256')==sha(root/'hardware-prolog.json'),'actual science worker contract incomplete')
 ownership=read(root/'process-group-ownership.json');need(ownership.get('intent_sha256')==sha(root/'intent.json')and ownership.get('parent')==context['parent'],'actual process-group/intent provenance differs')
 for case_id in CASES:validate_mapping(read(root/case_id/'observer-mapping.json'),w,set(source_map.values()))
 result=audit_measurements(w,intent,root);after=w.inventory(root);after_meta=w.metadata(root)
 need(before==after,'raw content changed during audit')
 need(before_meta.keys()==after_meta.keys()and all(all(before_meta[name][k]==after_meta[name][k]for k in before_meta[name]if k!='st_atime_ns')for name in before_meta),'raw metadata changed during audit')
 need(not any(n.split('.')[0]in FORBIDDEN for n in sys.modules),'target runtime imported')
 result.update(status='PASS_EXISTING_SCIENCE_DATA_AUDIT_NOT_A_SEAL',raw_root=str(root),slot_index=slot_index,job_id=hardware['allocation']['SLURM_JOB_ID'],intent_sha256=sha(root/'intent.json'),raw_tree_sha256=w.tree_digest(before),raw_members=before,identity={k:intent[k]for k in w.IDENTITY_KEYS},hardware_prolog_sha256=sha(root/'hardware-prolog.json'),raw_content_drift=False,raw_metadata_drift=False,atime_changes={n:dict(before=before_meta[n]['st_atime_ns'],after=after_meta[n]['st_atime_ns'])for n in before_meta if before_meta[n]['st_atime_ns']!=after_meta[n]['st_atime_ns']},new_tensor_deserialization=False,physical_attempts_created=0,qualification_seal_created=False)
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sealer',type=Path,required=True);parser.add_argument('--own-auditor',type=Path,required=True);parser.add_argument('--slot',type=int,choices=SCIENCE,required=True)
 args=parser.parse_args();print(json.dumps(audit(args.sealer,args.own_auditor,args.slot),sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
