"""Bounded file-only031 producer chains; never imports or reruns science."""
import hashlib
import json
from pathlib import Path
from .files import canonical,digest,need,read_bytes,relative_name
from .manifest import training_cell,validate
from .evaluation import BANK_SHA,EVALUATION_SEED,validate_contract031

LABELS=(0,24,49,99,249,499)
CASE_RAW={'worker-result.json','checkpoint-audit.json'}|{'pilot-bank32/'+name for name in (
    'report.json','episodes.json','observer-mapping.json','onset-native-producers.npz','inclusive-safety-endpoints.json',
    'reset-observed.json','pilot-reset-bank-evidence.json','observer-records.json','native-boundary-records.json')}


class MissingProof(Exception):
    """Absent evidence is an incomplete outcome, never successful science."""


class Proofs:
    def __init__(self,manifest,downloads):
        need(type(downloads) is dict and len(downloads)<=512,'bounded explicit download mapping required')
        need(all(type(k) is str and type(v) is str and Path(k).is_absolute() and Path(v).is_absolute()
                 for k,v in downloads.items()),'absolute download paths required')
        self.downloads=downloads;self.roots=(Path(manifest['root']),Path(manifest['evaluation_root']))
        self.cache={};self.assigned={}

    def local(self,path):
        if path in self.downloads:return self.downloads[path]
        if any(Path(path).is_relative_to(root) for root in self.roots):raise MissingProof('missing explicit download: '+path)
        return path

    def read(self,ref,*,expected=None,owner=None):
        need(type(ref) is dict and set(ref)=={'path','sha256','size'} and type(ref['path']) is str
             and Path(ref['path']).is_absolute(),'closed031 proof reference required')
        if expected is not None:need(ref['path']==str(expected),'proof path identity differs')
        key=(ref['path'],ref['sha256'],ref['size'])
        if owner is not None:
            fingerprint=(ref['sha256'],ref['size'])
            need(fingerprint not in self.assigned or self.assigned[fingerprint]==owner,'duplicate producer proof assignment')
            self.assigned[fingerprint]=owner
        if key not in self.cache:
            try:body=read_bytes(self.local(ref['path']),16*1024**2)
            except FileNotFoundError as exc:raise MissingProof(str(exc)) from exc
            need(len(body)==ref['size'] and hashlib.sha256(body).hexdigest()==ref['sha256'],'proof bytes differ')
            def pairs(items):
                result={}
                for k,v in items:need(k not in result,'duplicate proof JSON key');result[k]=v
                return result
            def invalid(value):raise ValueError('nonfinite proof JSON: '+value)
            self.cache[key]=json.loads(body,object_pairs_hook=pairs,parse_constant=invalid)
        return self.cache[key]


def exact_identity(row,schema,sha,kind,index):
    need(type(row) is dict and row.get('schema')==schema and row.get('manifest_sha256')==sha
         and row.get('kind')==kind and type(row.get('index')) is int and row['index']==index,'031 proof schema/kind/identity differs')


def member(ref,name,members):
    relative_name(name)
    need(members.get(name)=={k:ref[k] for k in ('sha256','size')},'archive proof/raw member differs: '+name)


def seal_identity(seal,manifest,index,label):
    need(seal.get('schema')=='z1_depth_checkpoint_seal031' and seal.get('manifest_sha256')==digest(manifest)
         and type(seal.get('cell')) is int and seal['cell']==index and seal.get('checkpoint_label')==label
         and seal.get('counter')==(label+1)*24,'031 checkpoint seal identity/counter differs')
    need(seal['checkpoint']['path']==manifest['cells'][index]['output']+f'/train/model_{label}.pt','selected checkpoint path differs')
    validate_contract031(seal['impact_speed_contract'],index)


def audit_identity(audit,manifest,kind,index):
    sha=digest(manifest);cell_index=index if kind=='training' else manifest['cases'][index]['cell']
    schema='z1_depth_training_audit031' if kind=='training' else 'z1_depth_measurement_audit031'
    exact_identity(audit,schema,sha,kind,index)
    need(canonical(audit.get('cell'))==canonical(training_cell(cell_index)),'audit cell/arm/seed differs')
    raw=audit.get('raw_files')
    if not isinstance(raw,dict) or not raw:raise MissingProof('required raw science inventory missing')
    need(len(raw)<=256,'bounded raw inventory required')
    for name,row in raw.items():
        relative_name(name)
        need(type(row) is dict and set(row)=={'sha256','size'} and type(row['size']) is int and row['size']>=0,'raw reference differs')
    if kind=='training':
        need(audit.get('status')=='COMPLETE' and audit.get('trace_rows')==500 and audit.get('population_rows')==501,'complete500 training audit required')
        required={'training-trace.jsonl','velocity-population.jsonl','initial-state.json','checkpoint-499-final-save-deduplicated.json'}
        seals=audit.get('checkpoints',[])
        need(len(seals)==6,'all six training checkpoint audits required')
        for label,seal in zip(LABELS,seals):
            seal_identity(seal,manifest,index,label)
            for key,name in (('checkpoint',f'train/model_{label}.pt'),('live',f'checkpoint-{label}-live.json'),('initial','initial-state.json')):
                required.add(name)
                need(seal[key]['path']==manifest['cells'][index]['output']+'/'+name,'training checkpoint proof path differs')
                if name in raw:member(seal[key],name,raw)
        if not required<=set(raw):raise MissingProof('required500 training raw science missing')
    else:
        case=manifest['cases'][index];record=audit.get('record',{});binding=audit.get('binding',{})
        seal_identity(binding,manifest,cell_index,case['label'])
        expected=dict(policy_id=f"cell-{cell_index:02d}-model-{case['label']}",case_index=index,training_cell=training_cell(cell_index),
            checkpoint_label=case['label'],completed_updates=case['label']+1,common_step_counter=case['counter'],
            task='Unitree-Z1-Hammer-Spline-DepthReward031-'+training_cell(cell_index)['arm'],checkpoint_path=binding['checkpoint']['path'],
            checkpoint_sha256=binding['checkpoint']['sha256'],impact_speed_contract=binding['impact_speed_contract'])
        need(canonical(record)==canonical(expected) and audit.get('checkpoint_sha256')==record['checkpoint_sha256'],'selected checkpoint record differs')
        bank=dict(path=manifest['source_root']+'/scripts/z1_depth_campaign031/vendor/reset-bank001/BANK32-BALANCED-PROSPECTIVE002.json',sha256=BANK_SHA)
        measured=audit.get('measurement',{})
        need(audit.get('status')=='PASS_FILE_AUDIT_ONLY' and audit.get('reset_bank')==bank
             and audit.get('evaluation_seed')==EVALUATION_SEED and measured.get('reset_bank_sha256')==BANK_SHA,'prospective bank/evaluation identity differs')
        need(measured.get('denominators',{}).get('all_first_episodes')==32
             and [r.get('world_id') for r in measured.get('worlds',[])]==list(range(32)),'saved32-context numerical audit required')
        if not (isinstance(measured.get('window_metrics'),dict) and measured['window_metrics']
                and isinstance(measured.get('world_classifications'),list)):
            raise MissingProof('saved numerical audit payload missing')
        need([r.get('world_id') for r in measured['world_classifications']]==list(range(32))
             and measured.get('scientific_safety_claim') is False,'saved numerical audit scope differs')
        if not CASE_RAW<=set(raw):raise MissingProof('required32-context raw science missing')
    return raw


def outcome(manifest,kind,index,entry,proofs,archive):
    identity=('cell-' if kind=='training' else 'case-')+f'{index:02d}'
    result=dict(identity=identity,status='MISSING' if entry is None else 'INCOMPLETE',scientific_outcome='UNKNOWN',preserved=False)
    if entry is None:return result
    need(type(entry) is dict and set(entry)=={'schema','kind','index','terminal','audit'}
         and entry['schema']=='z1_depth_outcome031' and entry['kind']==kind and type(entry['index']) is int and entry['index']==index,
         'discriminated031 outcome kind/index required')
    root=Path(manifest['cells' if kind=='training' else 'cases'][index]['output']);sha=digest(manifest)
    members=None if archive is None else archive['members'];audit=None
    def read(ref,name,expected=None):
        if ref is None:raise MissingProof('missing '+name)
        row=proofs.read(ref,expected=root/name if expected is None else expected,owner=identity+':'+name)
        if members is not None:member(ref,name,members)
        return row
    try:
        if entry['audit'] is not None:
            audit=read(entry['audit'],'audit.json')
            raw=audit_identity(audit,manifest,kind,index)
            if members is not None:
                for name,pin in raw.items():member(pin,name,members)
        terminal=read(entry['terminal'],'terminal.json')
        exact_identity(terminal,'z1_depth_terminal031',sha,kind,index)
        need(terminal.get('status') in ('COMPLETE','FAILED_NO_RETRY'),'unknown031 producer terminal state')
        control=Path(manifest['root'] if kind=='training' else manifest['evaluation_root'])
        basename=('cell-' if kind=='training' else 'case-')+f'{index:02d}.json'
        claim=read(terminal.get('claim'),'producer-claim.json',control/'claims'/basename)
        exact_identity(claim,'z1_depth_claim031',sha,kind,index)
        intent=read(terminal.get('intent'),'producer-intent.json',control/'intents'/basename)
        need(intent.get('schema')=='z1_depth_intent031' and digest(intent.get('manifest'))==sha and intent.get('kind')==kind
             and type(intent.get('index')) is int and intent['index']==index and intent.get('output')==str(root)
             and intent.get('claim')==terminal['claim'] and intent.get('nonce')==claim.get('nonce')
             and intent.get('parent')==claim.get('parent'),'consumed intent/claim linkage differs')
        if terminal['status']=='FAILED_NO_RETRY':
            result['status']='FAILED_NO_RETRY'
            # A failed producer may never have spawned/published a child result.
            if terminal.get('child_consumed') is not None:
                consumed=read(terminal['child_consumed'],'producer-child-consumed.json',(control/'claims'/basename).with_suffix('.child.json'))
                consumed_identity(consumed,sha,terminal['intent'],claim)
        else:
            consumed=read(terminal.get('child_consumed'),'producer-child-consumed.json',(control/'claims'/basename).with_suffix('.child.json'))
            consumed_identity(consumed,sha,terminal['intent'],claim)
            if audit is None:raise MissingProof('required saved audit missing')
            child=read(terminal.get('result'),'child-result.json')
            need(child.get('intent_sha256')==terminal['intent']['sha256'] and child.get('audit')==entry['audit'],'child result/audit linkage differs')
            expected=dict(audit,intent_sha256=terminal['intent']['sha256'],audit=entry['audit'],
                started_epoch=child.get('started_epoch'),finished_epoch=child.get('finished_epoch'))
            need(child==expected,'child result differs from saved audit')
            if kind=='evaluation':
                need(audit['record']==intent.get('record') and audit['binding']==intent.get('binding')
                     and audit.get('claim_sha256')==terminal['claim']['sha256'],'evaluation intent/checkpoint/claim linkage differs')
            if archive is None:raise MissingProof('independently verified archive missing')
            result.update(status='COMPLETE',scientific_outcome='SEE_SAVED_AUDIT',audit=entry['audit'])
        result.update(terminal=entry['terminal'],preserved=archive is not None,backup=None if archive is None else archive['verified'])
    except MissingProof as exc:result['reason']=str(exc)
    return result


def consumed_identity(row,sha,intent,claim):
    need(row.get('schema')=='z1_depth_child_consumed031' and row.get('manifest_sha256')==sha
         and row.get('intent_sha256')==intent['sha256'] and row.get('nonce')==claim.get('nonce')
         and row.get('parent')==claim.get('parent') and type(row.get('identity')) is dict,'consumed child/intent linkage differs')


def final_join(manifest,training,evaluations,backups,*,downloads=None):
    from .preservation import verify_archive
    validate(manifest)
    need(type(training) is list and type(evaluations) is list and len(training)==12 and len(evaluations)==24,'all planned031 outcomes required')
    need(type(backups) is list and len(backups)<=36,'bounded031 preservation list required')
    proofs=Proofs(manifest,{} if downloads is None else downloads);archives={}
    identities={f'cell-{i:02d}' for i in range(12)}|{f'case-{i:02d}' for i in range(24)}
    declared=set();backup_refs=set()
    for ref in backups:
        key=canonical(ref)
        need(key not in backup_refs,'duplicate preservation reference');backup_refs.add(key)
        try:receipt=proofs.read(ref)
        except MissingProof:continue
        identity=receipt.get('identity')
        need(identity in identities and identity not in declared,'unknown or duplicate preservation identity');declared.add(identity)
        need(receipt.get('schema')=='z1_depth_archive031','new031 archive receipt required')
        # Only archive location changes on a download; identity/hash/size stay fixed.
        try:
            local=dict(receipt,archive=dict(receipt['archive'],path=proofs.local(receipt['archive']['path'])))
            verified=verify_archive(local)
        except (MissingProof,FileNotFoundError):continue
        archives[identity]=dict(members=receipt['members'],verified=verified)
    joined=[outcome(manifest,kind,index,row,proofs,archives.get(('cell-' if kind=='training' else 'case-')+f'{index:02d}'))
        for kind,rows in (('training',training),('evaluation',evaluations)) for index,row in enumerate(rows)]
    return dict(schema='z1_depth_final_join031',manifest_sha256=digest(manifest),outcomes=joined,
        all_evidence_preserved=all(row.get('preserved',False) for row in joined),scientific_qualification=False,
        old029_evidence_unchanged=True,immutable_audits_checked_once=True)
