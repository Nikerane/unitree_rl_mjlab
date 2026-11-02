"""Read-only existing-data audit for replacement007. Emits JSON; never seals/jobs.

Run only after actual terminal. Exact unchanged e1 sealer and f4 consumer source
are compiled. Partial/running/wall-failed cases cannot return a PASS audit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import types

CAMPAIGN='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
RAW_ROOT=CAMPAIGN+'/runs/extended200-seed2026102301/own_cpu32-runtime-budget007'
SUPERVISOR_PATH=CAMPAIGN+'/package-runtime-budget007/supervisor/cpu_supervisor.py'
SUPERVISOR_SHA256='f4b40175786581063848c8b8d9b40651f203f9670418fcbe6d3da7274f3af449'
SEALER_SHA256='e1bc82fe9feb7a57ce6ea220fd502d07474eeb1397bd46da80af530e1ee52641'
JOB_ID='48760576'
FORBIDDEN={'torch','mujoco','mujoco_warp','mjlab','rsl_rl'}


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_module(path,pin,name):
    path=Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path,*path.parents)),'real absolute audited source required')
    source=path.read_bytes();require(hashlib.sha256(source).hexdigest()==pin,'audited source SHA differs')
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def validate_target(intent,hardware):
    policy=intent['policy'];context=intent['supervisor_context']
    require(intent.get('schema')=='z1_horizon_cpu_worker_v1' and intent.get('status')=='ADMITTED_ONCE'
            and intent.get('scope')=='own_cpu32' and intent.get('attempts')==32
            and intent.get('cases')==['original-route+1'] and intent.get('evaluation_seed')==2026101602
            and intent.get('retry_allowed') is False and intent.get('resume_allowed') is False
            and intent.get('output_root')==RAW_ROOT,'exact replacement007 own32 population required')
    require((policy.get('policy_id'),policy.get('arm'),policy.get('training_seed'),policy.get('native_window_s'))
            ==('extended200-seed2026102301','extended200',2026102301,.2),'exact extended200 replacement policy required')
    require(type(context.get('slot')) is int and context['slot']==3
            and context.get('supervisor_path')==SUPERVISOR_PATH
            and context.get('supervisor_sha256')==SUPERVISOR_SHA256,'exact slot3/f4 supervisor required')
    allocation=hardware.get('allocation',{})
    require(allocation.get('SLURM_JOB_ID')==JOB_ID and allocation.get('hostname','').split('.')[0]=='cn0290','exact replacement job/cn0290 required')


def validate_terminal(terminal,hardware):
    require(terminal.get('status')=='RAW_REVIEW_REQUIRED' and terminal.get('returncode')==0
            and terminal.get('reason') is None and terminal.get('cleanup_error') is None
            and terminal.get('measurement_errors')=={} and terminal.get('retry_allowed') is False
            and terminal.get('allocation')==hardware.get('allocation'),'complete clean terminal required; partial/wall failure is not qualifying')


def near(actual,expected):
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(near(a,b) for a,b in zip(actual,expected))
    return isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12)


def dense_metrics(w,trace,world,start,stop):
    result={}
    for name in ('qvel','rolling','pulse','enforced'):
        peaks=[];exceeds=[]
        for joint in range(6):
            def channel(key):return [trace[key]['values'][(world*881+k)*6+joint] for k in range(start,stop)]
            values=[max(x,y) for x,y in zip(channel('rolling'),channel('pulse'))] if name=='enforced' else channel(name)
            if name=='qvel':values=list(map(abs,values))
            limit=w.VELOCITY_LIMIT if name=='qvel' else w.LOAD_CAPS[joint]
            peaks.append(max(values));exceeds.append(sum(value>limit for value in values))
        result[name]=dict(peak=peaks,exceedance_samples=exceeds)
    return result


def audit_measurements(w,intent,root):
    """Reuse actual e1 audit; add native/reset parity and class-aware summary."""
    root=Path(root);case=root/'original-route+1'
    classifications=w.audit_raw(intent,root)  # Full32 and801 failure semantics remain unchanged.
    episodes=w.keyed(read(case/'episodes.json'),'world');observers=w.keyed(read(case/'observer-records.json'),'world_id')
    boundaries=w.keyed(read(case/'native-boundary-records.json'),'world_id')
    trace=w.read_trace(case/'onset-native-producers.npz');endpoints=w.keyed(read(case/'inclusive-safety-endpoints.json')['worlds'],'world_id')
    resets=sorted(case.glob('reset-event-*.json'))
    require(len(resets)==1 and read(resets[0]).get('worlds')==list(range(32)),'exact one reset event required')
    require(read(case/'first-step-observed.json').get('observed_substeps')==[80]*32,'first step must observe80 substeps for every world')
    worlds=[];contact=[];window_rows={name:[] for name in w.WINDOWS}
    for classification in classifications:
        i=classification['world_id'];ep=episodes[i];obs=observers[i];native=boundaries[i]['native'];snapshot=native.get('snapshot',{})
        require(ep.get('episode_id')==f'{i}:0' and type(ep.get('finalized')) is bool and type(ep.get('incomplete')) is bool and ep.get('route_sign')==1,'actual first route+1 episode and recorded utility labels required')
        require(native.get('boundary_measurement',{}).get('samples')==native.get('observed_samples')==snapshot.get('observed_samples'),'native snapshot timestamp differs')
        expected_reasons={name for name,value in native['termination_terms'].items() if value}
        require(isinstance(snapshot.get('termination_reasons'),str) and set(snapshot['termination_reasons'].split('+'))==expected_reasons,'native snapshot termination reasons differ')
        for key in ('peak_nms','exceedance_samples'):
            field='joint_load_'+key
            actual=native['boundary_measurement'].get(field)
            require(near(actual,snapshot.get('loads',{}).get('episode',{}).get(key)) if isinstance(snapshot.get('loads',{}).get('episode',{}).get(key),list) else False,'native whole-episode load parity differs')
        row=dict(world_id=i,outcome=classification['classification'],accepted_contact_sample=obs['accepted_contact_sample'],first_raw_contact_sample=obs['first_raw_contact_sample'],native_contact_samples=native['contact_samples'],native_termination_terms=native['termination_terms'],native_snapshot_termination_reasons=snapshot['termination_reasons'],observed_samples=obs['observed_samples'],policy_steps=ep['steps'],physical_end_reason=obs.get('end_reason'),timeout=ep.get('timeout'),utility_incomplete=ep.get('incomplete'),speed_finalized=ep['finalized'],measured_mps=ep.get('measured_mps'),productive=ep.get('productive'),depth_50ms_m=None,depth_200ms_m=None,safety_0_200ms='UNKNOWN_NO_CONTACT')
        if classification['classification']=='CONTACT_COMPLETE':
            contact.append(i)
            if native['termination_terms'].get('nail_driven') is True:
                require(snapshot.get('native_success_sample')==native['observed_samples'],'native success timestamp differs')
            else:
                require(snapshot.get('native_success_sample') is None,'native success invented for nonsuccess termination')
            common=dense_metrics(w,trace,i,0,200)
            require(snapshot.get('common_window',{}).get('window_complete') is True and snapshot['common_window']['window_samples']==200 and snapshot['loads']['common_window']['observed_samples']==200,'native common50ms endpoint missing')
            for key in ('peak_nms','exceedance_samples'):
                expected=common['enforced']['peak' if key=='peak_nms' else key]
                require(near(snapshot['loads']['common_window'].get(key),expected),'native common-window dense parity differs')
            for name,a,b in [('contact_0_50ms',0,200),('contact_0_200ms',0,800),('contact_50_200ms',200,800)]:
                metric=dense_metrics(w,trace,i,a,b);record=obs[name]
                require(record.get('complete') is True and record.get('samples')==b-a,'legacy observer window incomplete')
                for field,expected in [('joint_velocity_peak_rad_s',metric['qvel']['peak']),('joint_velocity_exceedance_samples',metric['qvel']['exceedance_samples']),('joint_load_peak_nms',metric['enforced']['peak']),('joint_load_exceedance_samples',metric['enforced']['exceedance_samples'])]:require(near(record.get(field),expected),'legacy observer dense parity differs')
            require(obs.get('authoritative_contact_0_200ms_inclusive')==endpoints[i]['windows']['authoritative_0_200ms_inclusive'],'inclusive observer endpoint differs')
            for label,k in [('50ms',201),('200ms',801)]:
                d=obs['literal_depth'][label];sample=obs['accepted_contact_sample']+k-1
                require(d.get('sample')==d.get('lower_sample')==d.get('upper_sample')==sample and near(d.get('depth_m'),d.get('lower_depth_m')) and near(d.get('depth_m'),d.get('upper_depth_m')) and near(d.get('target_time_s'),sample*w.DT),'literal depth timestamp/value differs')
                row['depth_'+label+'_m']=d['depth_m']
            for name,(a,b) in w.WINDOWS.items():window_rows[name].append(dense_metrics(w,trace,i,a,b))
            row['safety_0_200ms']='OBSERVED_COMPLETE_801'
        else:
            require(snapshot.get('common_window',{}).get('window_complete') is False and snapshot.get('loads',{}).get('common_window',{}).get('observed_samples')==0,'no-contact common window must remain absent')
            if classification['classification']=='NO_CONTACT_TIMEOUT':
                require(ep['steps']==1000 and obs['observed_samples']==80000 and native['observed_samples']==80000 and native['termination_terms'].get('time_out') is True,'no-contact timeout must retain full20s/80000 substeps')
        worlds.append(row)
    metrics={}
    for name,rows in window_rows.items():
        metrics[name]=dict(contact_worlds_observed=len(rows),all_first_episodes=32,unknown_no_contact_worlds=32-len(rows),maximum_velocity_per_joint=None if not rows else [max(row['qvel']['peak'][j] for row in rows) for j in range(6)],maximum_enforced_impulse_per_joint=None if not rows else [max(row['enforced']['peak'][j] for row in rows) for j in range(6)],worlds_with_any_exceedance=None)
        if rows:metrics[name]['worlds_with_any_exceedance']=sum(any(row['qvel']['exceedance_samples']) or any(row['enforced']['exceedance_samples']) for row in rows)
    return dict(world_classifications=classifications,denominators=dict(all_first_episodes=32,contact_801=len(contact),genuine_no_contact=32-len(contact)),worlds=worlds,window_metrics=metrics,scientific_safety_claim=False,unsafe_allowed_for_technical_qualification=True)


def metadata(root):
    return {str(path.relative_to(root)):{key:getattr(path.lstat(),key) for key in ('st_mode','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns','st_atime_ns','st_ino','st_dev','st_nlink')} for path in (root,*sorted(root.rglob('*')))}


def audit(sealer_path,supervisor_path=SUPERVISOR_PATH):
    """Existing terminal bytes only. No tensor load or seal/review publication."""
    require(str(supervisor_path)==SUPERVISOR_PATH,'exact actual primary supervisor path required')
    root=Path(RAW_ROOT)
    require(root.is_dir() and not any(path.is_symlink() for path in (root,*root.parents)),'actual replacement raw root required')
    hardware=read(root/'hardware-prolog.json');terminal=read(root/'supervisor-terminal.json')
    validate_terminal(terminal,hardware)
    w=source_module(sealer_path,SEALER_SHA256,'post_terminal_e1_sealer')
    c=source_module(supervisor_path,SUPERVISOR_SHA256,'post_terminal_f4_consumer')
    before_meta=metadata(root);before=w.inventory(root);intent=read(root/'intent.json')
    validate_target(intent,hardware);c.verify_intent_inventory(intent)
    c.validate_hardware_prolog(root/'hardware-prolog.json',intent=intent)
    require(read(root/'actual-runtime.json')==intent['runtime_identity'],'actual runtime differs')
    context=intent['supervisor_context'];require(sha(context['release_path'])==context['release_sha256'],'release hash differs')
    manifest=c.load_release(context['release_path']);c.require_selected(manifest,3);slot=manifest['slots'][3]
    require(all(intent.get(key)==value for key,value in slot['intent'].items()),'actual intent differs from admitted slot3')
    training=c.audit_training(slot['training_bindings'],tensor_audit=False)
    saved=read(root/'training-audit.json');preflight=read(root/'training-metadata-preflight.json')
    require(saved.get('status')=='PASS' and saved.get('finite_checkpoint') is True and saved.get('policy')==intent['policy'] and saved.get('checkpoint_sha256')==intent['policy']['checkpoint_sha256'] and saved.get('iteration')==499 and saved.get('common_step_counter')==12000 and saved.get('last_transition_index')==11999 and saved.get('tree_sha256')==training['tree_sha256']==preflight.get('tree_sha256'),'actual saved supervised finite audit provenance differs')
    train_manifest=c.bound(slot['training_bindings']['manifest'])
    source={str(Path(row['path']).resolve().relative_to(Path(intent['source_root']).resolve())):row['sha256'] for row in intent['source_files']}
    require(source==train_manifest['source_files'] and all({row['path']:row['sha256'] for row in intent[group]}==train_manifest[group] for group in ('runtime_files','asset_files')),'source/runtime/assets differ from own training')
    require(intent['source_identity']==dict(training_manifest_sha256=slot['training_bindings']['manifest']['sha256'],source_tree_sha256=w.tree_digest(source)),'source identity differs')
    packages={Path(row['path']).name:row for row in intent['package_files']}
    require(len(packages)==10 and intent['observer_identity']=={name:row['sha256'] for name,row in packages.items()} and context['worker_sha256']==packages['horizon_worker.py']['sha256'] and intent['reviewed_hook_path']==packages['horizon_integration.py']['path'] and intent['reviewed_hook_sha256']==packages['horizon_integration.py']['sha256'],'observer/hook/worker identities differ')
    worker=read(root/'worker-result.json')
    require(worker.get('attempts')==32 and worker.get('checkpoint_contract_validated_before_overlay') is True and worker.get('final_contract_restored') is True and worker.get('reset_frames_sampled')==0 and worker.get('hardware_prolog_sha256')==sha(root/'hardware-prolog.json'),'actual worker contract incomplete')
    ownership=read(root/'process-group-ownership.json')
    require(ownership.get('intent_sha256')==sha(root/'intent.json') and ownership.get('parent')==context['parent'],'actual process-group/intent provenance differs')
    result=audit_measurements(w,intent,root)
    mapping=read(root/'original-route+1/observer-mapping.json');imp=mapping['impulse_provenance']
    require(mapping.get('joint_names')==['joint'+str(j) for j in range(1,7)] and all(mapping.get(key)==list(range(6)) for key in ('entity_ids','native_joint_ids','qpos_ids','dof_ids')) and mapping.get('actuator_ids')==[0,5,1,2,3,4],'actual canonical joint/actuator mapping differs')
    require(imp.get('window_substeps')==200 and imp.get('baseline_subtracted') is True and imp.get('thresholds_nms')==list(w.LOAD_CAPS) and imp.get('enforced_relation')=='maximum(raw_load,held_load)' and imp.get('source_sha256') in source.values(),'impulse producer semantics differ')
    after=w.inventory(root);after_meta=metadata(root)
    require(before==after,'raw content changed during audit')
    require(before_meta.keys()==after_meta.keys() and all(all(before_meta[name][key]==after_meta[name][key] for key in before_meta[name] if key!='st_atime_ns') for name in before_meta),'raw metadata changed during audit')
    require(not any(name.split('.')[0] in FORBIDDEN for name in sys.modules),'target runtime imported')
    result.update(status='PASS_EXISTING_DATA_AUDIT_NOT_A_SEAL',raw_root=RAW_ROOT,job_id=JOB_ID,slot_index=3,intent_sha256=sha(root/'intent.json'),raw_tree_sha256=w.tree_digest(before),raw_members=before,identity={key:intent[key] for key in w.IDENTITY_KEYS},hardware_prolog_sha256=sha(root/'hardware-prolog.json'),source_pins=dict(sealer=SEALER_SHA256,supervisor=SUPERVISOR_SHA256),raw_content_drift=False,raw_metadata_drift=False,atime_changes={name:dict(before=before_meta[name]['st_atime_ns'],after=after_meta[name]['st_atime_ns']) for name in before_meta if before_meta[name]['st_atime_ns']!=after_meta[name]['st_atime_ns']},new_tensor_deserialization=False,review_receipts_created=False,qualification_seal_created=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sealer',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.sealer),sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
