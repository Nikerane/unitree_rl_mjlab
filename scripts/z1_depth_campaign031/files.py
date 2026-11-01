"""Strict bounded regular-file hashes and exclusive durable publication."""
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_path(path):
    path = Path(path).absolute()
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink path refused: '+str(path))
    return path


def regular_path(path):
    path = safe_path(path)
    need(stat.S_ISREG(path.stat().st_mode), 'regular file required: '+str(path))
    return path


def read_bytes(path, cap=256*1024**2):
    path = regular_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(before.st_size <= cap, 'file cap exceeded')
        body = stream.read(cap+1)
        after = os.fstat(stream.fileno())
    need(len(body) <= cap and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'file changed during read')
    need(path.stat().st_ino == after.st_ino, 'file replaced during read')
    return body


def sha_file(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def read_json(path, cap=16*1024**2):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(read_bytes(path, cap), object_pairs_hook=pairs, parse_constant=invalid)


def sync_dir(path):
    fd = os.open(safe_path(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, value):
    """Hardlink promotion publishes only complete fsynced bytes; never replaces."""
    path = safe_path(path)
    need(path.parent.is_dir(), 'existing parent required')
    tmp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.pending')
    body = canonical(value)+b'\n'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path, follow_symlinks=False)
        sync_dir(path.parent)
    finally:
        tmp.unlink()
        sync_dir(path.parent)
    return reference(path)


claim_once = write_once


def reference(path):
    body = read_bytes(path)
    return dict(path=str(Path(path).absolute()), sha256=hashlib.sha256(body).hexdigest(), size=len(body))


def verify_ref(ref, expected_path=None):
    need(type(ref) is dict and set(ref)=={'path','sha256','size'}, 'closed file reference required')
    if expected_path is not None:
        need(ref['path']==str(Path(expected_path).absolute()), 'reference path differs')
    need(reference(ref['path'])==ref, 'reference bytes differ: '+ref['path'])
    return Path(ref['path'])


def relative_name(name):
    p = Path(name)
    need(type(name) is str and name and not p.is_absolute() and '..' not in p.parts and str(p)==name,
         'unsafe relative member')
    return p


def inventory(root, cap):
    root = safe_path(root)
    need(root.is_dir(), 'inventory root missing')
    result = {}; total = 0
    for path in sorted(root.rglob('*')):
        safe_path(path)
        if path.is_dir():
            continue
        ref = reference(path)
        result[str(path.relative_to(root))] = dict(sha256=ref['sha256'], size=ref['size'])
        total += ref['size']
        need(total<=cap and len(result)<=100000, 'inventory cap exceeded')
    return result


def durable_reference(path):
    """Make inherited completed publications durable before issuing a new seal."""
    path=regular_path(path);before=reference(path)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    sync_dir(path.parent)
    need(reference(path)==before,'publication changed during durability fence')
    return before
