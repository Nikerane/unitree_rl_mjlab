"""All-eighteen capacity ledger and one in-flight transaction; no scheduler IO.

Nonzero credit requires the full archive proof authenticator, never a boolean.
Raw/private usage consumes the frozen transaction and never changes ledger R.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

CPU_ROOT='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
BACKUP_ROOT='/cephhome/eunikhilr/campaign-backups/z1-horizon-evaluation-20261027-autonomous005'
CONTROL_ROOT=CPU_ROOT+'/admission-staged-integration002/autonomous005'
EVIDENCE_ROOT=CPU_ROOT+'/evidence-autonomous005'
CAPS=dict(raw=167772160,archive=201326592,provenance=92274688)
SHARED=dict(derived=671088640,controls=67108864)
IMMUTABLE=11844714496
G=461373440

def require(value,message):
    if not value:raise ValueError(message)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def layout_rows():
    rows=[]
    for arm in ('native50','extended200'):
        for seed in (2026102301,2026102302,2026102303):
            policy='%s-seed%d'%(arm,seed)
            for scope,bank in [('own_cpu32',None),('science160',20261041),('science160',2026101601)]:
                leaf='own_cpu32-staged-integration002' if bank is None else 'science160-bank%d'%bank
                rows.append(dict(policy_id=policy,scope=scope,bank=bank,run_root=CPU_ROOT+'/runs/'+policy+'/'+leaf,private_root=CPU_ROOT+'/private/'+policy+'/'+leaf))
    return rows

def indexed_rows():
    rows=layout_rows();order=['native50-seed2026102301','extended200-seed2026102301','extended200-seed2026102302','native50-seed2026102302','native50-seed2026102303','extended200-seed2026102303']
    return [row for policy in order for row in rows if row['policy_id']==policy]

def preissue_descriptor(*,index,release,nonce):
    require(type(index)is int and index in range(1,18),'closed non-first CPU index required')
    row=indexed_rows()[index]
    require(isinstance(release,dict) and set(release)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',release['sha256']),'full release reference required')
    phase='remaining' if index%3==0 else 'science'
    require(release['path']==CPU_ROOT+'/admission-staged-integration002/'+phase+'/release.json','closed actual phase release path required')
    require(isinstance(nonce,str) and re.fullmatch('[0-9a-f]{32,64}',nonce),'approved nonce required')
    return dict(schema='z1_vega_preissue_descriptor_v1',kind='SLOT',index=index,slot=row,release=release,nonce=nonce,caps=dict(CAPS),shared_caps=dict(SHARED))


def zero_ledger(*,preissue,layout_sha256,staging_bytes=0):
    """Create conservative all18 ledger. Add only concrete verified objects later."""
    require(isinstance(layout_sha256,str) and re.fullmatch('[0-9a-f]{64}',layout_sha256),'layout SHA required')
    return dict(schema='z1_vega_autonomous_capacity_ledger_v1',rule_version='capacity-ledger-ruling002',preissue_descriptor=preissue,preissue_intent_sha256=digest(preissue),release_sha256=preissue['release']['sha256'] if preissue.get('kind')=='SLOT' else None,layout_sha256=layout_sha256,slots=[dict(row,caps=dict(CAPS),retained={key:[] for key in CAPS}) for row in layout_rows()],shared={key:dict(cap=value,retained=[]) for key,value in SHARED.items()},staging_bytes=staging_bytes)


def _ledger_budget(ledger):
    # Standalone arithmetic uses the same exact rows and proof grouping rules.
    expected=layout_rows()
    require(ledger.get('schema')=='z1_vega_autonomous_capacity_ledger_v1' and ledger.get('rule_version')=='capacity-ledger-ruling002','fixed ledger schema required')
    require(len(ledger.get('slots',[]))==18,'all eighteen slots required')
    descriptor=ledger.get('preissue_descriptor');require(isinstance(descriptor,dict) and ledger.get('preissue_intent_sha256')==digest(descriptor),'explicit preissue descriptor SHA differs')
    if descriptor.get('kind')=='SLOT':
        require(descriptor==preissue_descriptor(index=descriptor['index'],release=descriptor['release'],nonce=descriptor['nonce']),'exact preissue descriptor differs')
        require(ledger.get('release_sha256')==descriptor['release']['sha256'],'preissue release SHA differs')
    else:require(descriptor==dict(schema='z1_vega_preissue_descriptor_v1',kind='CONTROL_ONLY') and ledger.get('release_sha256') is None,'explicit generic capacity descriptor required')
    for key in ('preissue_intent_sha256','layout_sha256'):
        require(isinstance(ledger.get(key),str) and re.fullmatch('[0-9a-f]{64}',ledger[key]),'ledger binding required')
    seen=set();inodes={};credit=0
    for index,slot in enumerate(ledger['slots']):
        row={k:slot.get(k) for k in expected[index]}
        require(row==expected[index],'fixed ordered slot roots required')
        require(slot.get('caps')==CAPS and all(type(v)is int for v in slot['caps'].values()),'fixed slot caps required')
        require(set(slot.get('retained',{}))==set(CAPS),'complete retained roles required')
        for role,cap in CAPS.items():
            total=0
            for item in slot['retained'][role]:
                require(isinstance(item,dict) and set(item)=={'path','bytes','sha256','device','inode','link_count','proof'},'concrete credit object required')
                path=Path(item['path']);require(path.is_absolute() and '..' not in path.parts and str(path)==item['path'],'canonical credit path required')
                require(type(item['bytes'])is int and item['bytes']>=0,'typed credit size required')
                require(all(type(item[k])is int and item[k]>=0 for k in ('device','inode','link_count')) and item['link_count']>0,'credit identity required')
                require(isinstance(item['sha256'],str) and re.fullmatch('[0-9a-f]{64}',item['sha256']),'credit SHA required')
                proof=item['proof'];require(isinstance(proof,dict) and set(proof)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',proof['sha256']) and Path(proof['path']).is_absolute(),'proof reference required')
                raw=any(Path(slot[k]) in path.parents for k in ('run_root','private_root'))
                base=Path(BACKUP_ROOT)/slot['policy_id']/Path(slot['run_root']).name
                evidence=Path(EVIDENCE_ROOT)/('slot%02d'%index)
                require(raw if role=='raw' else base in path.parents or (role=='provenance' and evidence in path.parents),'credit outside exact slot role roots')
                require(item['path'] not in seen,'duplicate credit path');seen.add(item['path'])
                inode=(item['device'],item['inode']);signature=(index,role,item['bytes'],item['sha256'])
                require(inode not in inodes or (role=='raw' and inodes[inode]==signature),'cross-slot/cross-role inode credit')
                if inode not in inodes:credit+=item['bytes'];inodes[inode]=signature
                total+=item['bytes'];require(total<=cap,'credit cap exceeded')
    require(set(ledger.get('shared',{}))==set(SHARED),'shared allocations required')
    for role,cap in SHARED.items():
        obj=ledger['shared'][role];require(obj==dict(cap=cap,retained=[]),'shared credit disabled until separate verified shared proof profile exists')
    staging=ledger.get('staging_bytes');require(type(staging)is int and staging>=0,'explicit staging bytes required')
    return IMMUTABLE+18*G+sum(SHARED.values())+staging-credit

def _authenticate(ledger,request,deadline):
    items=[(slot,role,item) for slot in ledger['slots'] for role in CAPS for item in slot['retained'][role]]
    if not items:return
    require(request is not None and deadline is not None,'nonzero credit requires actual full proof authentication')
    from vega_policy_archive import resident_archive as archive
    groups={}
    for slot,role,item in items:groups.setdefault(item['proof']['sha256'],[]).append((slot,role,item))
    for values in groups.values():
        ref=values[0][2]['proof'];raw=archive.bound_bytes(ref,deadline,1024**2)
        proof=archive._credit_closure(request,archive.parse_json(raw),deadline)
        for slot,role,item in values:archive._authenticate_credit_item(slot,role,item,proof,deadline)

def make_reservation(ledger,*,index,release,nonce,ledger_sha256,active,credit_request=None,deadline=None):
    require(not active,'overlapping reservation forbidden')
    R=_ledger_budget(ledger)
    require(type(index)is int and index in range(1,18),'first qualified slot cannot be reserved')
    require(isinstance(release,dict) and set(release)=={'path','sha256'},'release reference required')
    phase='remaining' if index%3==0 else 'science'
    require(release['path']==CPU_ROOT+'/admission-staged-integration002/'+phase+'/release.json' and release['sha256']==ledger['release_sha256'],'phase/release differs')
    require(isinstance(nonce,str) and re.fullmatch('[0-9a-f]{32,64}',nonce),'root-approved nonce required')
    require(ledger_sha256==digest(ledger),'ledger state SHA differs')
    require(ledger['preissue_descriptor']==preissue_descriptor(index=index,release=release,nonce=nonce),'preissue descriptor differs from current reservation')
    _authenticate(ledger,credit_request,deadline)
    row=indexed_rows()[index];base=BACKUP_ROOT+'/'+row['policy_id']+'/'+Path(row['run_root']).name
    return dict(schema='z1_vega_slot_reservation_v1',index=index,slot=row,release=release,ledger_sha256=ledger_sha256,ledger=ledger,R=R,G=G,caps=dict(CAPS),shared_caps=dict(SHARED),roots=dict(raw=[row['run_root'],row['private_root']],archive=base,provenance=EVIDENCE_ROOT+'/slot%02d'%index,controls=CONTROL_ROOT),nonce=nonce)

def validate_reservation(reservation,ledger,*,credit_request=None,deadline=None):
    require(reservation.get('ledger')==ledger,'frozen ledger differs')
    expected=make_reservation(ledger,index=reservation['index'],release=reservation['release'],nonce=reservation['nonce'],ledger_sha256=reservation['ledger_sha256'],active=[],credit_request=credit_request,deadline=deadline)
    require(reservation==expected,'reservation exact binding differs')
    return expected

def check_reservation(reservation,usage,*,free_bytes,pending=None):
    # R and caps cannot be changed independently of the frozen complete ledger.
    require(reservation.get('schema')=='z1_vega_slot_reservation_v1' and reservation['ledger_sha256']==digest(reservation['ledger']),'reservation ledger state differs')
    require(reservation['R']==_ledger_budget(reservation['ledger']) and reservation['G']==G and reservation['caps']==CAPS,'frozen budget differs')
    row=indexed_rows()[reservation['index']]
    require(reservation['slot']==row and reservation['roots']==dict(raw=[row['run_root'],row['private_root']],archive=BACKUP_ROOT+'/'+row['policy_id']+'/'+Path(row['run_root']).name,provenance=EVIDENCE_ROOT+'/slot%02d'%reservation['index'],controls=CONTROL_ROOT),'frozen slot roots differ')
    require(isinstance(usage,dict) and set(usage)==set(CAPS),'exact usage roles required')
    pending=pending or {key:0 for key in CAPS}
    require(set(pending)==set(CAPS),'exact pending roles required')
    for role,cap in CAPS.items():
        require(type(usage[role])is int and type(pending[role])is int and min(usage[role],pending[role])>=0,'nonnegative typed usage required')
        require(usage[role]+pending[role]<=cap,'individual role cap exceeded')
    used=sum(usage.values());write=sum(pending.values());require(used+write<=G,'transaction G exceeded')
    protected=max(IMMUTABLE,reservation['R']-G)
    required=max(protected,reservation['R']-used)
    require(type(free_bytes)is int and free_bytes-write>=max(protected,reservation['R']-used-write),'current quota below frozen reservation')
    return dict(consumed_bytes=used,protected_bytes=protected,required_free_bytes=required,remaining_growth_bytes=G-used)

def current_usage(reservation,*,deadline,max_nodes=8192):
    """Bounded no-follow lstat walk; count only the four exact frozen roots.

    Archive files count to archive; all other backup files plus slot logs/reviews
    count to provenance. Hardlink aliases are rejected across roots/roles.
    """
    check_reservation(reservation,{key:0 for key in CAPS},free_bytes=reservation['R'])
    require(type(max_nodes)is int and 0<max_nodes<=8192,'finite node bound required')
    totals={key:0 for key in CAPS};seen=set();count=0
    def walk(path,role,backup=False):
        nonlocal count
        require(time.monotonic()<deadline,'usage deadline expired')
        for parent in reversed(path.parents):
            if parent.exists():require(stat.S_ISDIR(parent.lstat().st_mode),'usage parent symlink forbidden')
        try:info=path.lstat()
        except FileNotFoundError:return
        count+=1;require(count<=max_nodes,'usage node cap exceeded')
        if stat.S_ISDIR(info.st_mode):
            fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                with os.scandir(fd) as entries:
                    for entry in entries:walk(path/entry.name,role,backup)
            finally:os.close(fd)
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'usage symlink/special/hardlink forbidden')
            identity=(info.st_dev,info.st_ino);require(identity not in seen,'usage inode overlap');seen.add(identity)
            charged='archive' if backup and path.name=='terminal-policy-full.tar.gz' else ('provenance' if backup else role)
            totals[charged]+=info.st_size;require(totals[charged]<=CAPS[charged],'observed role cap exceeded')
    for value in reservation['roots']['raw']:walk(Path(value),'raw')
    walk(Path(reservation['roots']['archive']),'archive',True)
    walk(Path(reservation['roots']['provenance']),'provenance')
    return totals
