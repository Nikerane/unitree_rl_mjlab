"""Permanent exclusive ownership and bounded, fsynced hash-chain decisions.

Only the owning controller thread writes. Existing leases are never reclaimed,
including after a clean completion; read_chain/reconcile never repair evidence.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import threading

MAX_EVENT = 1048576
MAX_JOURNAL = 8388608
REGULAR_JOURNAL = 6291456  # 2MiB reserved for bounded drain/HOLD decisions
MAX_CONTROLS = 67108864
EMERGENCY_CONTROLS = 2097152

class Hold(RuntimeError):
    pass

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()

def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise Hold('duplicate JSON key')
        result[key] = value
    return result

def decode(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(Hold('nonfinite JSON')))
    except (ValueError, UnicodeError) as exc:
        raise Hold('invalid JSON') from exc

def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)

def read_file(path, cap=MAX_EVENT):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > cap:
            raise Hold('not a bounded private regular file')
        result = bytearray()
        while True:
            block = os.read(fd, min(65536, cap+1-len(result)))
            if not block: break
            result.extend(block)
            if len(result)>cap: raise Hold('file limit')
        after = os.fstat(fd)
        if (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
            raise Hold('file changed during read')
        return bytes(result)
    finally: os.close(fd)

def control_budget(root, pending, *, emergency=False):
    total = pending
    count = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs+files:
            count += 1
            st = os.lstat(Path(base)/name)
            if stat.S_ISLNK(st.st_mode) or not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
                raise Hold('control special node')
            if stat.S_ISREG(st.st_mode):
                if st.st_nlink != 1: raise Hold('control hard link')
                total += st.st_size
            if total>(MAX_CONTROLS if emergency else MAX_CONTROLS-EMERGENCY_CONTROLS) or count>8192: raise Hold('shared controller cap')

def exclusive_json(path, value, *, cap=MAX_EVENT):
    path = Path(path)
    data = encoded(value)+b'\n'
    if len(data)>cap: raise Hold('control size cap')
    control_budget(path.parent, len(data))
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view[:65536])
            if n<=0: raise Hold('short control write')
            view = view[n:]
        os.fsync(fd)
    finally: os.close(fd)
    sync_directory(path.parent)
    return {'path':str(path), 'sha256':hashlib.sha256(data).hexdigest()}

class Lease:
    def __init__(self, fd, identity): self.fd, self.identity = fd, identity
    def close(self):
        if self.fd is not None:
            os.close(self.fd); self.fd = None

def acquire_new_lease(root, config):
    root = Path(root)
    if not root.is_dir() or root.is_symlink(): raise Hold('lease root must already exist')
    try: fd = os.open(root/'owner.lease', os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    except OSError as exc: raise Hold('existing/unknown owner; reconcile only') from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity = dict(config, hostname=socket.gethostname(), pid=os.getpid())
        data = encoded(identity)+b'\n'
        if len(data)>16384: raise Hold('lease cap')
        if os.write(fd,data)!=len(data): raise Hold('short lease write')
        os.fsync(fd); sync_directory(root)
        return Lease(fd,identity)
    except BaseException:
        os.close(fd)
        raise  # The permanent failed lease deliberately remains.

def read_chain(path):
    path = Path(path)
    if not path.exists(): return []
    raw = read_file(path,MAX_JOURNAL)
    if raw and not raw.endswith(b'\n'): raise Hold('torn journal')
    result, previous, intents = [], '0'*64, set()
    for line in raw.splitlines():
        if len(line)>MAX_EVENT: raise Hold('event cap')
        value = decode(line)
        if not isinstance(value,dict) or set(value)!={'sequence','previous','event','sha256'}:
            raise Hold('journal schema')
        body={k:value[k] for k in ('sequence','previous','event')}
        if type(value['sequence']) is not int or value['sequence']!=len(result)+1 or value['previous']!=previous or digest(body)!=value['sha256']:
            raise Hold('broken journal chain')
        event=value['event']
        if not isinstance(event,dict) or not isinstance(event.get('kind'),str): raise Hold('journal event schema')
        if event['kind']=='SUBMIT_INTENT':
            if type(event.get('index')) is not int or not 1<=event['index']<=17 or event['index'] in intents: raise Hold('duplicate/invalid submit intent')
            intents.add(event['index'])
        previous=value['sha256']; result.append(value)
    return result

class Journal:
    def __init__(self,path):
        self.path=Path(path)
        if self.path.exists(): raise Hold('journal already exists; reconcile only')
        fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        os.fsync(fd); os.close(fd); sync_directory(self.path.parent)
        self.sequence=0; self.previous='0'*64; self.bytes=0
        self.thread=threading.get_ident(); self.intents=set()
    def append(self,event,*,emergency=False):
        if threading.get_ident()!=self.thread: raise Hold('only owner writes journal')
        if not isinstance(event,dict) or not isinstance(event.get('kind'),str): raise Hold('event schema')
        if event['kind']=='SUBMIT_INTENT':
            if type(event.get('index')) is not int or not 1<=event['index']<=17 or event['index'] in self.intents: raise Hold('duplicate/invalid submit intent')
            self.intents.add(event['index'])
        body={'sequence':self.sequence+1,'previous':self.previous,'event':event}
        row=dict(body,sha256=digest(body)); data=encoded(row)+b'\n'
        if len(data)>MAX_EVENT or self.bytes+len(data)>(MAX_JOURNAL if emergency else REGULAR_JOURNAL): raise Hold('journal cap')
        control_budget(self.path.parent,len(data),emergency=emergency)
        fd=os.open(self.path,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        try:
            st=os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size!=self.bytes: raise Hold('journal changed')
            view=memoryview(data)
            while view:
                n=os.write(fd,view[:65536])
                if n<=0: raise Hold('short journal write')
                view=view[n:]
            os.fsync(fd)
        finally: os.close(fd)
        self.sequence=row['sequence']; self.previous=row['sha256']; self.bytes+=len(data)
        return {'sequence':self.sequence,'sha256':self.previous}
