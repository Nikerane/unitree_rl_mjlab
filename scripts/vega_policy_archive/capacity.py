"""Fixed eighteen-slot byte ledger and exclusive bounded Mac controls.

Arithmetic is pure. Callers MUST authenticate every nonzero retained object
against its proof/current remote descriptor before using remaining_budget.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

MIB=1024**2
FUTURE=11274289152
FLOOR=570425344
IMMUTABLE=FUTURE+FLOOR
CAPS=dict(raw=160*MIB,archive=192*MIB,provenance=88*MIB)
SHARED=dict(derived=640*MIB,controls=64*MIB)
MAC_ENVELOPE=1400897536
MAC_RETAINED=64*MIB
MAC_WORKING=88*MIB
CPU_ROOT='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
RULE='capacity-ledger-ruling002'


def require(value,message):
    if not value: raise ValueError(message)


def deadline_check(deadline):
    if time.monotonic()>=deadline: raise TimeoutError('capacity/control deadline expired')


def plain(value):
    path=Path(value)
    require(path.is_absolute() and str(path)==str(value) and '..' not in path.parts,'canonical absolute path required')
    return path


def parents(path):
    for item in reversed((path,*path.parents)):
        require(stat.S_ISDIR(item.lstat().st_mode),'nonsymlink control parents required')


def reference(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'},'exact proof reference required')
    plain(value['path'])
    require(isinstance(value['sha256'],str) and re.fullmatch('[0-9a-f]{64}',value['sha256']),'proof SHA required')


def layout_rows():
    rows=[]
    for arm in ('native50','extended200'):
        for seed in (2026102301,2026102302,2026102303):
            policy='%s-seed%d'%(arm,seed)
            for scope,bank in [('own_cpu32',None),('science160',20261041),('science160',2026101601)]:
                leaf='own_cpu32-staged-integration002' if bank is None else 'science160-bank%d'%bank
                rows.append(dict(policy_id=policy,scope=scope,bank=bank,run_root=CPU_ROOT+'/runs/'+policy+'/'+leaf,
                                 private_root=CPU_ROOT+'/private/'+policy+'/'+leaf))
    return rows


def retained_objects(ledger):
    """Yield typed materialization records after the complete ledger is checked."""
    seen_paths=set(); seen_inodes={}
    for slot in ledger['slots']:
        for role,cap in CAPS.items():
            items=slot['retained'][role]
            require(isinstance(items,list),'retained file set required')
            total=0
            for item in items:
                require(isinstance(item,dict) and set(item)=={'path','bytes','sha256','device','inode','link_count','proof'},'supported retained object required')
                path=plain(item['path']); reference(item['proof'])
                require(type(item['bytes']) is int and item['bytes']>=0,'nonnegative retained bytes required')
                require(isinstance(item['sha256'],str) and re.fullmatch('[0-9a-f]{64}',item['sha256']),'retained SHA required')
                require(all(type(item[k]) is int and item[k]>=0 for k in ('device','inode','link_count')) and item['link_count']>0,'retained identity required')
                require(str(path) not in seen_paths,'duplicate retained path across roles')
                inode=(item['device'],item['inode'])
                signature=(role,item['bytes'],item['sha256'])
                require(inode not in seen_inodes or (role=='raw' and seen_inodes[inode]==signature),'duplicate retained inode across roles/inconsistent raw identity')
                seen_paths.add(str(path));seen_inodes[inode]=signature
                if role=='raw':
                    require(any(plain(slot[key]) in path.parents for key in ('run_root','private_root')),'raw credit outside exact source roots')
                else:
                    require(CPU_ROOT not in str(path),'archive/provenance credit cannot alias raw/runtime')
                    if role=='archive': require(item['link_count']==1,'new separate archive link_count must be one')
                total+=item['bytes'];require(total<=cap,'retained credit exceeds cap')
                yield slot,role,item
    for role,cap in SHARED.items():
        obj=ledger['shared'][role];require(set(obj)=={'cap','retained'} and obj['cap']==cap and type(obj['cap']) is int,'fixed shared cap required')
        # Shared retained files use the identical concrete file proof contract.
        total=0
        require(isinstance(obj['retained'],list),'shared retained file set required')
        for item in obj['retained']:
            require(set(item)=={'path','bytes','sha256','device','inode','link_count','proof'},'supported shared object required')
            reference(item['proof']); path=plain(item['path'])
            require(type(item['bytes']) is int and item['bytes']>=0,'shared retained bytes required')
            require(isinstance(item['sha256'],str) and re.fullmatch('[0-9a-f]{64}',item['sha256']),'shared retained SHA required')
            require(all(type(item[k]) is int and item[k]>=0 for k in ('device','inode','link_count')) and item['link_count']>0,'shared identity required')
            inode=(item['device'],item['inode'])
            require(str(path) not in seen_paths and inode not in seen_inodes,'duplicate retained path/inode across roles')
            seen_paths.add(str(path));seen_inodes[inode]=(role,item['bytes'],item['sha256']);total+=item['bytes']
            require(total<=cap,'shared credit exceeds cap')
            yield None,role,item


def remaining_budget(ledger):
    require(isinstance(ledger,dict) and ledger.get('schema')=='z1_vega_resident_capacity_ledger_v1' and ledger.get('rule_version')==RULE,'capacity ledger schema/rule required')
    for key in ('request_identity_sha256','release_sha256','layout_sha256'):
        require(isinstance(ledger.get(key),str) and re.fullmatch('[0-9a-f]{64}',ledger[key]),'ledger binding required: '+key)
    expected={json.dumps(x,sort_keys=True) for x in layout_rows()};actual=set()
    require(isinstance(ledger.get('slots'),list) and len(ledger['slots'])==18,'all eighteen slots required')
    for slot in ledger['slots']:
        require(all(key in slot for key in ('policy_id','scope','bank','run_root','private_root','caps','retained')),'complete slot required')
        row={key:slot[key] for key in ('policy_id','scope','bank','run_root','private_root')}
        encoded=json.dumps(row,sort_keys=True)
        require(encoded in expected and encoded not in actual,'foreign/duplicate slot')
        actual.add(encoded)
        require(slot['caps']==CAPS and all(type(v) is int for v in slot['caps'].values()),'fixed typed slot caps required')
        require(isinstance(slot['retained'],dict) and set(slot['retained'])==set(CAPS),'all residual roles required')
    require(isinstance(ledger.get('shared'),dict) and set(ledger['shared'])==set(SHARED),'all shared allocations required')
    staging=ledger.get('staging_bytes');require(type(staging) is int and staging>=0,'explicit nonnegative staging bytes required')
    objects=list(retained_objects(ledger))
    unique={(item['device'],item['inode']):item['bytes'] for _,_,item in objects}
    credit=sum(unique.values())
    return IMMUTABLE+18*sum(CAPS.values())+sum(SHARED.values())+staging-credit


def operation_budget(ledger,growth_bytes):
    remaining=remaining_budget(ledger)
    require(type(growth_bytes) is int and 0<growth_bytes<=sum(CAPS.values())+sum(SHARED.values()),'bounded positive operation growth required')
    protected=max(IMMUTABLE,remaining-growth_bytes)
    return dict(remaining_bytes=remaining,growth_bytes=growth_bytes,protected_bytes=protected,
                future_reserve_bytes=max(FUTURE,protected-FLOOR))


class ControlRoot:
    """A single explicit retained root, guarded by an advisory directory lock.

    Every file (including failed/partial writes) counts. No symlink is followed;
    write never overwrites or removes evidence. Working data is memory-only.
    """
    def __init__(self,root,*,free_reader=None,deadline):
        self.root=plain(root);parents(self.root);self.deadline=deadline;self._lockfd=None
        self.free_reader=free_reader or (lambda:shutil.disk_usage(self.root).free)

    def _total(self):
        total=0;count=0
        def walk(fd):
            nonlocal total,count
            deadline_check(self.deadline)
            with os.scandir(fd) as entries:
                for entry in entries:
                    count+=1;require(count<=8192,'control node cap exceeded')
                    info=entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        child=os.open(entry.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                        try:walk(child)
                        finally:os.close(child)
                    else:
                        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'control root symlink/special/hardlink forbidden')
                        total+=info.st_size;require(total<=MAC_RETAINED,'retained aggregate exceeds64MiB')
        fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:walk(fd)
        finally:os.close(fd)
        return total

    def check(self,pending=0):
        deadline_check(self.deadline);parents(self.root)
        total=self._total()
        require(total+pending<=MAC_RETAINED,'retained aggregate exceeds64MiB')
        # Keep the entire admitted working/media/helper/floor and future retained
        # envelope, less retained material already charged to observed free.
        free=self.free_reader();deadline_check(self.deadline)
        require(type(free) is int and free-pending>=MAC_ENVELOPE-total,'Mac envelope/free reservation exhausted')
        return total

    @contextmanager
    def lease(self):
        """Serialize the full allocation, including all remote RPCs/publication."""
        if self._lockfd is not None:
            yield self._lockfd
            return
        deadline_check(self.deadline);parents(self.root)
        lock=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            while True:
                deadline_check(self.deadline)
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(.01)
            self._lockfd=lock
            yield lock
        finally:
            self._lockfd=None
            fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)

    def write(self,name,data):
        require(isinstance(name,str) and Path(name).name==name and name not in ('','.','..'),'single control filename required')
        require(isinstance(data,bytes) and len(data)<=MIB,'compact control exceeds1MiB')
        with self.lease() as lock:
            self.check(len(data))
            info=os.fstat(lock);current=self.root.lstat()
            require((info.st_dev,info.st_ino)==(current.st_dev,current.st_ino),'control root identity changed')
            fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=lock)
            try:
                view=memoryview(data)
                while view:
                    deadline_check(self.deadline)
                    self.check(len(view))
                    size=os.write(fd,view[:MIB]);require(size>0,'zero control write')
                    view=view[size:]
                os.fsync(fd);deadline_check(self.deadline)
            finally:os.close(fd)
            os.fsync(lock);deadline_check(self.deadline)
            return dict(path=str(self.root/name),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
