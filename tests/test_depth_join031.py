"""File-only final joins: real immutable receipts/download mappings/tar members."""
import copy
from pathlib import Path
import pytest
from scripts.z1_depth_campaign031 import files as f, manifest as m, preservation as p, evaluation as e


def contract():
    marker=f.read_json(e.HERE/'identities031.json')['A0']
    return dict(schema_version='impact_speed_v1',action_type='joint_position_variable_impedance_sampled_pd',
        controller_identity='sampled_vic500_v1',action_dim=13,physics_dt_s=.00025,control_decimation=80,
        speed_bounds_mps=[1.,1.5],speed_mode='policy',observation_widths={'actor':43,'critic':43},
        spline_depth_reward031=marker,spline_split_credit=marker['baseline_split'],
        safety={'cat_soft':dict(use_vel=True,use_impulse=True,imp_limit=[.369,.246,.738,.369,.246,.0164],imp_max_p=.2,imp_seed=.001,limit=3.1415,max_p=.5,min_p=0.,tau=.95,vel_detection='substep')})


CASE_RAW=['worker-result.json','checkpoint-audit.json']+['pilot-bank32/'+name for name in (
    'report.json','episodes.json','observer-mapping.json','onset-native-producers.npz','inclusive-safety-endpoints.json',
    'reset-observed.json','pilot-reset-bank-evidence.json','observer-records.json','native-boundary-records.json')]


def fixture(tmp_path,kind='training',fault=None):
    manifest=m.template();sha=f.digest(manifest);index=0;identity='cell-00' if kind=='training' else 'case-00'
    remote=Path(manifest['cells' if kind=='training' else 'cases'][0]['output'])
    local=tmp_path/'download';local.mkdir();mapping={};refs={}
    def put(name,value,remote_path=None):
        path=local/name;path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,bytes):path.write_bytes(value)
        else:f.write_once(path,value)
        ref=f.reference(path);ref['path']=str(remote_path or remote/name);mapping[ref['path']]=str(path);refs[name]=ref
        return ref
    claim_path=Path(m.ROOT if kind=='training' else m.EVAL_ROOT)/'claims'/('cell-00.json' if kind=='training' else 'case-00.json')
    intent_path=Path(m.ROOT if kind=='training' else m.EVAL_ROOT)/'intents'/('cell-00.json' if kind=='training' else 'case-00.json')
    claim=dict(schema='z1_depth_claim031',manifest_sha256=sha,kind=kind,index=0,nonce='fresh',parent={'pid':10,'startticks':'20'})
    if fault=='wrong_claim_kind':claim['kind']='qualification'
    cr=put('producer-claim.json',claim,claim_path)
    intent=dict(schema='z1_depth_intent031',manifest=manifest,kind=kind,index=0,output=str(remote),claim=cr,nonce='fresh',parent=claim['parent'])
    cell=m.training_cell(0)
    base=dict(manifest_sha256=sha,kind=kind,index=0,cell=cell)
    if kind=='training':
        for name in ('training-trace.jsonl','velocity-population.jsonl','initial-state.json','checkpoint-499-final-save-deduplicated.json'):put(name,b'fixture raw\n')
        seals=[]
        for label in (0,24,49,99,249,499):
            ck=put(f'train/model_{label}.pt',b'fixture checkpoint '+str(label).encode())
            live=put(f'checkpoint-{label}-live.json',{'iteration':label,'common_step_counter':(label+1)*24})
            seals.append(dict(schema='z1_depth_checkpoint_seal031',manifest_sha256=sha,cell=0,checkpoint_label=label,
                counter=(label+1)*24,checkpoint=ck,initial=refs['initial-state.json'],live=live,impact_speed_contract=contract()))
        audit=dict(base,schema='z1_depth_training_audit031',status='COMPLETE',trace_rows=500,population_rows=501,checkpoints=seals)
    else:
        ck=dict(path=manifest['cells'][0]['output']+'/train/model_99.pt',sha256='a'*64,size=24)
        record=dict(policy_id='cell-00-model-99',case_index=0,training_cell=cell,checkpoint_label=99,completed_updates=100,
            common_step_counter=2400,task='Unitree-Z1-Hammer-Spline-DepthReward031-A0',checkpoint_path=ck['path'],checkpoint_sha256=ck['sha256'],impact_speed_contract=contract())
        binding=dict(schema='z1_depth_checkpoint_seal031',manifest_sha256=sha,cell=0,checkpoint_label=99,counter=2400,
            checkpoint=ck,impact_speed_contract=contract())
        intent.update(record=record,binding=binding)
        for name in CASE_RAW:put(name,b'fixture raw\n')
        audit=dict(base,schema='z1_depth_measurement_audit031',status='PASS_FILE_AUDIT_ONLY',record=record,binding=binding,
            reset_bank=dict(path=manifest['source_root']+'/scripts/z1_depth_campaign031/vendor/reset-bank001/BANK32-BALANCED-PROSPECTIVE002.json',sha256=e.BANK_SHA),
            evaluation_seed=e.EVALUATION_SEED,checkpoint_sha256=ck['sha256'],claim_sha256=cr['sha256'],
            measurement=dict(denominators={'all_first_episodes':32},reset_bank_sha256=e.BANK_SHA,worlds=[{'world_id':i} for i in range(32)],world_classifications=[{'world_id':i} for i in range(32)],
                window_metrics={'saved_window':{'all_first_episodes':32}},scientific_safety_claim=False))
    if fault=='wrong_intent_nonce':intent['nonce']='another'
    ir=put('producer-intent.json',intent,intent_path)
    consumed=dict(schema='z1_depth_child_consumed031',manifest_sha256=sha,intent_sha256=ir['sha256'],nonce='fresh',parent=claim['parent'],identity={'pid':11,'startticks':'21'})
    if fault=='wrong_consumed_intent':consumed['intent_sha256']='0'*64
    co=put('producer-child-consumed.json',consumed,claim_path.with_suffix('.child.json'))
    audit['raw_files']=f.inventory(local,1024**2)
    if fault=='wrong_kind':audit['kind']='evaluation' if kind=='training' else 'training'
    if fault=='old_schema':audit['schema']='z1_depth_training_audit029'
    if fault=='wrong_arm':audit['cell']=dict(cell,arm='A3')
    if fault=='wrong_seed':audit['cell']=dict(cell,seed=1)
    if fault=='counter_float':audit['record']['common_step_counter']=2400.0
    if fault=='wrong_counter':audit['record']['common_step_counter']=2401
    if fault=='wrong_bank':audit['reset_bank']['sha256']='0'*64
    if fault=='wrong_checkpoint':audit['checkpoint_sha256']='0'*64
    if fault=='short_training':audit['trace_rows']=499
    if fault=='missing_final_population':audit['population_rows']=500
    if fault=='short_evaluation':audit['measurement']['denominators']['all_first_episodes']=31
    if fault=='missing_numerical':audit['measurement'].pop('window_metrics')
    if fault=='missing_dedup':audit['raw_files_missing_dedup']=True
    if fault=='missing_raw':audit['raw_files'].pop('training-trace.jsonl' if kind=='training' else 'pilot-bank32/report.json')
    if audit.pop('raw_files_missing_dedup',False):audit['raw_files'].pop('checkpoint-499-final-save-deduplicated.json')
    ar=put('audit.json',audit)
    result=dict(audit,intent_sha256=ir['sha256'],audit=ar,started_epoch=100,finished_epoch=200)
    if fault=='unlinked_result':result['audit']=dict(ar,sha256='0'*64)
    rr=put('child-result.json',result)
    terminal=dict(schema='z1_depth_terminal031',manifest_sha256=sha,kind=kind,index=0,status='COMPLETE',claim=cr,intent=ir,child_consumed=co,result=rr)
    if fault=='failed_after_audit':terminal.update(status='FAILED_NO_RETRY',error='post-audit finalization failed',scientific_outcome='UNKNOWN');terminal.pop('result')
    if fault=='wrong_terminal_kind':terminal['kind']='qualification'
    if fault=='wrong_terminal_schema':terminal['schema']='z1_depth_terminal029'
    if fault=='wrong_terminal_manifest':terminal['manifest_sha256']='0'*64
    if fault=='wrong_terminal_index':terminal['index']=1
    if fault=='duplicate_proof':terminal['child_consumed']=cr
    if fault=='missing_result':terminal.pop('result')
    tr=put('terminal.json',terminal)
    if fault=='archive_proof_mismatch':(local/'child-result.json').write_bytes(b'other bytes')
    receipt=p.archive(local,tmp_path/'saved.tar',identity=identity,cap=2*1024**2)
    f.write_once(tmp_path/'backup.json',receipt)
    if fault=='archive_proof_mismatch':(local/'child-result.json').write_bytes(f.canonical(result)+b'\n')
    outcome=dict(schema='z1_depth_outcome031',kind=kind,index=0,terminal=tr,audit=ar)
    if fault=='missing_terminal':outcome['terminal']=None
    if fault=='missing_audit':outcome['audit']=None
    rows=[None]*(12 if kind=='training' else 24);rows[0]=outcome
    return manifest,rows if kind=='training' else [None]*12,rows if kind=='evaluation' else [None]*24,[f.reference(tmp_path/'backup.json')],mapping


def join(args):return p.final_join(*args[:4],downloads=args[4])


@pytest.mark.parametrize('kind',['training','evaluation'])
def test_valid_downloaded_producer_chain_and_archive_join(tmp_path,kind):
    result=join(fixture(tmp_path,kind));row=result['outcomes'][0 if kind=='training' else 12]
    assert row['status']=='COMPLETE' and row['preserved'] is True and row['scientific_outcome']=='SEE_SAVED_AUDIT'


@pytest.mark.parametrize('fault',['failed_after_audit','missing_terminal','missing_audit','missing_raw','missing_result'])
def test_incomplete_or_failed_producer_never_promotes_saved_audit(tmp_path,fault):
    result=join(fixture(tmp_path,fault=fault));row=result['outcomes'][0]
    assert row['scientific_outcome']=='UNKNOWN' and row['status']!='COMPLETE'
    if fault=='failed_after_audit':assert row['status']=='FAILED_NO_RETRY' and row['preserved'] is True


@pytest.mark.parametrize('kind,fault',[(k,x) for k in ('training','evaluation') for x in ('wrong_kind','old_schema','wrong_arm','wrong_seed','unlinked_result','archive_proof_mismatch')]+[('training','short_training'),('training','missing_final_population'),('evaluation','wrong_bank'),('evaluation','wrong_checkpoint'),('evaluation','short_evaluation')])
def test_wrong_identity_or_proof_bytes_rejected(tmp_path,kind,fault):
    with pytest.raises(ValueError):join(fixture(tmp_path,kind,fault))


def test_duplicate_outcome_assignment_rejected(tmp_path):
    args=fixture(tmp_path);args[1][1]=copy.deepcopy(args[1][0])
    with pytest.raises(ValueError):join(args)


def test_missing_download_does_not_reopen_vega_path(tmp_path):
    args=fixture(tmp_path);args[4].pop(args[1][0]['terminal']['path'])
    row=join(args)['outcomes'][0]
    assert row['status']=='INCOMPLETE' and row['scientific_outcome']=='UNKNOWN'


def test_review_wrong_kind_reference_is_not_an_outcome(tmp_path):
    r=m.template();path=tmp_path/'terminal.json'
    f.write_once(path,dict(schema='z1_depth_terminal031',kind='training',status='COMPLETE',manifest_sha256=f.digest(r),index=0))
    with pytest.raises(ValueError):p.final_join(r,[None]*12,[f.reference(path)]+[None]*23,[])


def test_review_saved_audit_alone_cannot_override_failed_terminal(tmp_path):
    r=m.template();raw=tmp_path/'raw';raw.mkdir()
    f.write_once(raw/'audit.json',dict(schema='z1_depth_training_audit031',status='COMPLETE',manifest_sha256=f.digest(r),index=0,raw_files={}))
    f.write_once(raw/'terminal.json',dict(schema='z1_depth_terminal031',kind='training',status='FAILED_NO_RETRY',manifest_sha256=f.digest(r),index=0))
    backup=p.archive(raw,tmp_path/'proof.tar',identity='cell-00',cap=4096);f.write_once(tmp_path/'backup.json',backup)
    with pytest.raises(ValueError):p.final_join(r,[f.reference(raw/'audit.json')]+[None]*11,[None]*24,[f.reference(tmp_path/'backup.json')])


def test_producer_proof_copies_preserve_original_refs_and_bytes(tmp_path):
    from scripts.z1_depth_campaign031 import worker
    controls=tmp_path/'controls';controls.mkdir();out=tmp_path/'output';out.mkdir()
    claim=controls/'claim.json';intent=controls/'intent.json';consumed=claim.with_suffix('.child.json')
    for path in (claim,intent,consumed):f.write_once(path,{'source':path.name})
    refs=worker.producer_refs(out,claim,intent)
    assert refs==dict(claim=f.reference(claim),intent=f.reference(intent),child_consumed=f.reference(consumed))
    for key,path in (('claim',claim),('intent',intent),('child-consumed',consumed)):
        assert (out/('producer-'+key+'.json')).read_bytes()==path.read_bytes()
    assert worker.producer_refs(out,claim,intent)==refs
    (out/'producer-intent.json').write_bytes(b'conflict')
    with pytest.raises(ValueError):worker.producer_refs(out,claim,intent)


def test_complete_producer_without_independent_archive_stays_incomplete(tmp_path):
    args=fixture(tmp_path);args[3].clear()
    row=join(args)['outcomes'][0]
    assert row['status']=='INCOMPLETE' and row['scientific_outcome']=='UNKNOWN' and row['preserved'] is False


def test_each_downloaded_proof_is_read_once_per_join(tmp_path,monkeypatch):
    from scripts.z1_depth_campaign031 import join as implementation
    args=fixture(tmp_path);reads={};original=implementation.read_bytes
    def counted(path,*a,**kw):
        reads[str(path)]=reads.get(str(path),0)+1
        return original(path,*a,**kw)
    monkeypatch.setattr(implementation,'read_bytes',counted)
    assert join(args)['outcomes'][0]['status']=='COMPLETE'
    assert reads and all(count==1 for count in reads.values())


def test_join_cli_requires_new_input_schema_and_preserves_missing_outcomes(tmp_path):
    import subprocess,sys
    manifest=m.template();f.write_once(tmp_path/'manifest.json',manifest)
    inputs=dict(schema='z1_depth_join_inputs031',training=[None]*12,evaluations=[None]*24,backups=[],downloads={})
    f.write_once(tmp_path/'inputs.json',inputs)
    command=[sys.executable,'-B','-m','scripts.z1_depth_campaign031','--manifest',str(tmp_path/'manifest.json'),
        '--manifest-sha',f.digest(manifest),'join','--inputs',str(tmp_path/'inputs.json'),'--output',str(tmp_path/'joined.json')]
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    result=f.read_json(tmp_path/'joined.json')
    assert len(result['outcomes'])==36 and all(row['scientific_outcome']=='UNKNOWN' for row in result['outcomes'])
    del inputs['schema'];f.write_once(tmp_path/'old-inputs.json',inputs)
    command[command.index('--inputs')+1]=str(tmp_path/'old-inputs.json')
    command[command.index('--output')+1]=str(tmp_path/'bad-joined.json')
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode!=0 and 'discriminated031 join inputs required' in run.stderr
    assert not (tmp_path/'bad-joined.json').exists()


@pytest.mark.parametrize('kind,fault',[('training','missing_dedup'),('evaluation','missing_numerical')])
def test_required_saved_science_payload_missing_stays_unknown(tmp_path,kind,fault):
    row=join(fixture(tmp_path,kind,fault))['outcomes'][0 if kind=='training' else 12]
    assert row['status']=='INCOMPLETE' and row['scientific_outcome']=='UNKNOWN'


@pytest.mark.parametrize('fault',['wrong_claim_kind','wrong_intent_nonce','wrong_consumed_intent','wrong_terminal_kind',
    'wrong_terminal_schema','wrong_terminal_manifest','wrong_terminal_index','duplicate_proof'])
def test_terminal_and_consumption_chain_identity_rejected(tmp_path,fault):
    with pytest.raises(ValueError):join(fixture(tmp_path,fault=fault))


def test_missing_archive_file_cannot_promote_complete_producer(tmp_path):
    args=fixture(tmp_path);(tmp_path/'saved.tar').rename(tmp_path/'held.tar')
    row=join(args)['outcomes'][0]
    assert row['status']=='INCOMPLETE' and row['scientific_outcome']=='UNKNOWN'


@pytest.mark.parametrize('fault',['counter_float','wrong_counter'])
def test_selected_case_counters_require_exact_planned_integer(tmp_path,fault):
    with pytest.raises(ValueError):join(fixture(tmp_path,'evaluation',fault))
