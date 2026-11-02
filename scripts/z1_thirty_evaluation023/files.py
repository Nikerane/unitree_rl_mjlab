"""Bounded file-only primitives. No model loading, jobs, claims or authority minting."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with regular_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def regular_path(path):
    path = Path(path)
    need(path.is_absolute() and str(path) == str(path.absolute()) and '..' not in path.parts, 'absolute canonical path required')
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink forbidden')
    need(stat.S_ISREG(path.stat().st_mode), 'regular file required')
    return path


def reference(path):
    path = regular_path(path)
    return dict(path=str(path), sha256=sha_file(path), size=path.stat().st_size)


def verify_ref(ref, *, expected_path=None, max_bytes=128*1024**2):
    need(type(ref) is dict and set(ref) == {'path', 'sha256', 'size'}, 'exact file reference required')
    need(type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'SHA256 required')
    need(type(ref['size']) is int and 0 < ref['size'] <= max_bytes, 'reference byte limit')
    if expected_path is not None:
        need(ref['path'] == str(expected_path), 'reference path differs')
    path = regular_path(ref['path'])
    need(path.stat().st_size == ref['size'] and sha_file(path) == ref['sha256'], 'reference bytes differ')
    return path


def read_json(path, *, cap=4*1024**2):
    path = regular_path(path)
    need(path.stat().st_size <= cap, 'JSON byte cap exceeded')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, 'nonfinite JSON'))


def write_once(path, value, *, cap=4*1024**2):
    """Exclusive bounded durable file; caller owns the directory and admission."""
    path = Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink output required')
    body = canonical(value) + b'\n'
    need(len(body) <= cap, 'JSON byte cap exceeded before write')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
