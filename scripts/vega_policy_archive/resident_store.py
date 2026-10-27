"""Exclusive Vega-resident archives, with local primary verification.

All deadlines are absolute monotonic timestamps, bounded to 600 seconds. Callback
inventory readers take the request; quota readers return current integer Ceph
headroom, never filesystem-wide free space. No networking or target imports.
"""
from __future__ import annotations

import base64
import ctypes
from decimal import Decimal
import errno
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import time

from .compat import load_modules
from .stream_reader import verify_stream

BACKUP_ROOT = Path('/cephhome/eunikhilr/campaign-backups/z1-horizon-evaluation-20261027-resident004')
FUTURE_MIN = 11274289152
FLOOR_CONTROLS = 570425344
SOURCE_CAP = 167772160
ARCHIVE_CAP = 201326592
METADATA_CAP = 4194304
EVIDENCE_CAP = 8388608
CHUNK = 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _deadline(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'finite monotonic deadline required')
    if time.monotonic() >= value:
        raise TimeoutError('resident store deadline expired')


def _limit_deadline(value):
    _deadline(value)
    return min(value, time.monotonic() + 600)


def _json(value, cap=METADATA_CAP):
    result = bytearray()
    for piece in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value):
        result.extend(piece.encode('utf-8'))
        _require(len(result) <= cap, 'metadata JSON byte cap exceeded')
    return bytes(result)


def _plain(path):
    value = str(path)
    path = Path(value)
    _require(path.is_absolute() and str(path) == value and '..' not in path.parts,
             'canonical absolute path required')
    return path


def _parents(path, *, include=True):
    """Reject symlinks at every existing directory, including ancestors."""
    path = _plain(path)
    for parent in reversed((path, *path.parents) if include else tuple(path.parents)):
        info = parent.lstat()
        _require(stat.S_ISDIR(info.st_mode), 'all parents must be nonsymlink directories')
    return path


def _separate(a, b):
    return a != b and a not in b.parents and b not in a.parents


def validate_destination(request, destination=None):
    """Pure closed CPU mapping; this exposes no alternate root argument."""
    modules = load_modules()
    roots = modules['contract'].validate_paths(request)
    _require(request['scope'] in ('own_cpu32', 'science160'), 'CPU archive scope required')
    _require(re.fullmatch(r'[0-9]+', request['job_id']) is not None, 'concrete numeric job ID required')
    raw, private = (_plain(roots[key]) for key in ('policy-tree', 'private-tree'))
    _require(_separate(raw, private), 'raw/private roots must be distinct and separate')
    backup = _plain(BACKUP_ROOT)
    expected = backup / request['policy_id'] / raw.name / ('job-' + request['job_id'])
    if destination is not None:
        _require(_plain(destination) == expected, 'destination differs from closed backup mapping')
    for source in (raw, private, _plain(request['cpu_layout']['root']) if 'cpu_layout' in request else raw):
        _require(_separate(source, backup), 'source/runtime and backup roots must be separate')
    for key in ('runtime_root', 'physical_runtime_root'):
        if key in request:
            _require(_separate(_plain(request[key]), backup), 'runtime and backup roots must be separate')
    return expected


def _bounds(request):
    for key, hard in [('max_source_bytes', SOURCE_CAP), ('max_archive_bytes', ARCHIVE_CAP),
                      ('max_metadata_bytes', METADATA_CAP), ('max_evidence_bytes', EVIDENCE_CAP),
                      ('max_members', 8192)]:
        _require(type(request.get(key)) is int and 0 < request[key] <= hard, 'invalid finite cap: ' + key)
    for key in ('future_reserve_bytes', 'remote_future_reserve_bytes'):
        _require(type(request.get(key, 0)) is int and request.get(key, 0) >= 0, 'invalid future reserve')
    return FLOOR_CONTROLS + max(FUTURE_MIN, request.get('future_reserve_bytes', 0),
                                request.get('remote_future_reserve_bytes', 0))


def _quota(free_reader, size, reserve, deadline):
    _deadline(deadline)
    free = free_reader()
    _deadline(deadline)
    _require(type(free) is int and free - size >= reserve, 'Ceph quota headroom below required future reserve')


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(fd)


class _Output:
    def __init__(self, fd, cap, free_reader, reserve, deadline):
        self.fd, self.cap, self.free_reader = fd, cap, free_reader
        self.reserve, self.deadline = reserve, deadline
        self.count, self.digest = 0, hashlib.sha256()

    def write(self, data):
        view = memoryview(data)
        original = len(view)
        _require(self.count + original <= self.cap, 'compressed/control byte cap exceeded')
        while view:
            piece = view[:CHUNK]
            _quota(self.free_reader, len(piece), self.reserve, self.deadline)
            count = os.write(self.fd, piece)
            _require(type(count) is int and 0 < count <= len(piece), 'zero/invalid short write')
            self.count += count
            self.digest.update(piece[:count])
            view = view[count:]
            _deadline(self.deadline)
        return original

    def flush(self):
        _deadline(self.deadline)


def _write_file(path, data, free_reader, reserve, deadline, cap):
    _require(isinstance(data, bytes) and len(data) <= cap, 'bounded control bytes required')
    _parents(path.parent)
    _quota(free_reader, len(data), reserve, deadline)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        output = _Output(fd, cap, free_reader, reserve, deadline)
        output.write(data)
        _deadline(deadline)
        os.fsync(fd)
        _deadline(deadline)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)
    _deadline(deadline)
    return dict(sha256=output.digest.hexdigest(), bytes=output.count)


def publish_control(destination, data, expected_sha256, *, free_reader, deadline, future_reserve_bytes=FUTURE_MIN):
    """Publish a NEW control file exclusively under the fixed backup namespace.

    The hard future reserve always applies; callers can declare a larger reserve.
    """
    deadline = _limit_deadline(deadline)
    destination = _plain(destination)
    _require(type(future_reserve_bytes) is int and future_reserve_bytes >= FUTURE_MIN, 'invalid control future reserve')
    _require(_plain(BACKUP_ROOT) in destination.parents, 'control outside owned backup namespace')
    _require(isinstance(data, bytes) and len(data) <= METADATA_CAP, 'bounded control bytes required')
    _require(hashlib.sha256(data).hexdigest() == expected_sha256,
             'control SHA-256 differs')
    return _write_file(destination, data, free_reader, FLOOR_CONTROLS + future_reserve_bytes,
                       deadline, METADATA_CAP)


def _xattrs(path):
    if hasattr(os, 'listxattr'):
        return {name: os.getxattr(path, name, follow_symlinks=False)
                for name in os.listxattr(path, follow_symlinks=False)}
    _require(sys.platform == 'darwin', 'native nonsymlink xattr reader unavailable')
    # Darwin Python builds may omit the os xattr APIs. XATTR_NOFOLLOW is 0x1.
    libc = ctypes.CDLL(None, use_errno=True)
    encoded = os.fsencode(path)
    listxattr = libc.listxattr
    listxattr.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    listxattr.restype = ctypes.c_ssize_t
    getxattr = libc.getxattr
    getxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                        ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
    getxattr.restype = ctypes.c_ssize_t
    def checked(result):
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(path))
        _require(result <= METADATA_CAP, 'xattr byte cap exceeded')
        return result
    size = checked(listxattr(encoded, None, 0, 1))
    buffer = ctypes.create_string_buffer(size)
    count = checked(listxattr(encoded, buffer, size, 1))
    _require(count == size, 'xattr list changed during read')
    values = {}
    for name in buffer.raw[:size].split(b'\0'):
        if not name:
            continue
        size = checked(getxattr(encoded, name, None, 0, 0, 1))
        value = ctypes.create_string_buffer(size)
        count = checked(getxattr(encoded, name, value, size, 0, 1))
        _require(count == size, 'xattr changed during read')
        values[os.fsdecode(name)] = value.raw[:size]
    return values


def _metadata(path):
    info = path.lstat()
    restore = dict(atime_ns=info.st_atime_ns, xattrs={})
    for attr in ('st_birthtime_ns', 'st_flags'):
        if hasattr(info, attr):
            restore[{'st_birthtime_ns':'birthtime_ns', 'st_flags':'flags'}[attr]] = getattr(info, attr)
    try:
        attrs = _xattrs(path)
    except OSError as exc:
        if exc.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP):
            raise
        attrs = {}
    for name, value in sorted(attrs.items()):
        _require(len(value) <= METADATA_CAP, 'xattr byte cap exceeded')
        restore['xattrs'][name] = dict(value_b64=base64.b64encode(value).decode('ascii'),
                                     sha256=hashlib.sha256(value).hexdigest(), bytes=len(value))
        _json(restore)
    identity = dict(inode=info.st_ino, device=info.st_dev, ctime_ns=info.st_ctime_ns,
                    link_count=info.st_nlink)
    row = dict(source_path=str(path), mode=stat.S_IMODE(info.st_mode), uid=info.st_uid,
               gid=info.st_gid, mtime_ns=info.st_mtime_ns, restore_metadata=restore,
               source_identity=identity)
    if stat.S_ISREG(info.st_mode): row.update(type='file', bytes=info.st_size)
    elif stat.S_ISDIR(info.st_mode): row['type'] = 'directory'
    elif stat.S_ISLNK(info.st_mode): row.update(type='symlink', target=os.readlink(path))
    else: raise ValueError('special source node forbidden')
    return row


def _walk(request, deadline):
    nodes = {}
    def visit(path, name):
        _deadline(deadline)
        row = _metadata(path)
        nodes[name] = row
        _require(len(nodes) <= request['max_members'], 'source member cap exceeded')
        if row['type'] == 'directory':
            _parents(path.parent)
            directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.fstat(directory_fd)
                _require(info.st_ino == row['source_identity']['inode'] and
                         info.st_dev == row['source_identity']['device'], 'directory identity changed before listing')
                with os.scandir(directory_fd) as entries:
                    children = []
                    for entry in entries:
                        _deadline(deadline)
                        children.append(entry.name)
                        _require(len(nodes) + len(children) <= request['max_members'], 'source member cap exceeded')
                    children.sort()
            finally:
                os.close(directory_fd)
            for child in children:
                visit(path / child, name + '/' + child)
    for prefix, key in [('policy-tree', 'source_root'), ('private-tree', 'private_root')]:
        visit(Path(request[key]), prefix)
    _json(nodes, request['max_metadata_bytes'])
    return nodes


def _stable(row):
    row = json.loads(_json(row))
    row.get('restore_metadata', {}).pop('atime_ns', None)
    return row


def _snapshot(request, inventory_reader, deadline):
    original = _walk(request, deadline)
    inventory = inventory_reader(request)
    _deadline(deadline)
    current = _walk(request, deadline)
    _require({k:_stable(v) for k,v in original.items()} == {k:_stable(v) for k,v in current.items()},
             'source metadata/identity drift during inventory')
    _require(isinstance(inventory, dict) and set(inventory.get('nodes', {})) == set(original),
             'full inventory member set differs')
    for name, row in original.items():
        item = inventory['nodes'][name]
        for key, value in row.items():
            if key not in ('restore_metadata', 'source_identity'):
                _require(item.get(key) == value, 'inventory source metadata differs: ' + name)
        inventory['nodes'][name] = dict(item, restore_metadata=row['restore_metadata'], source_identity=row['source_identity'])
    _json(inventory, request['max_metadata_bytes'])
    job = inventory.get('job', {})
    _require(job.get('job_id') == request['job_id'] and job.get('terminal') is True,
             'actual terminal job identity required')
    _require(job.get('state') in load_modules()['remote'].TERMINAL, 'job must be terminal')
    modules = load_modules()
    modules['remote'].validate_guard(request, job)
    capture = request.get('scheduler_identity_capture', {})
    _require(capture.get('job_id') == request['job_id'], 'saved scheduler identity differs')
    identity = modules['contract'].accounting_identity(request['job_id'], capture.get('accounting_job_id'),
                                                      request['job_id'], capture.get('capture'))
    _require(job.get('identity') == identity, 'inventory scheduler identity differs from saved capture')
    for key in ('policy_id', 'scope', 'source_root', 'private_root'):
        _require(inventory.get(key) == request[key], 'inventory request identity differs: ' + key)
    _require(sum(row.get('bytes', 0) for row in original.values()) <= request['max_source_bytes'], 'source byte cap exceeded')
    return inventory


def inspect_source(request, *, inventory_reader, deadline):
    """Return a complete bounded source inventory with original restore metadata."""
    deadline = _limit_deadline(deadline)
    _bounds(request)
    validate_destination(request)
    for key in ('source_root', 'private_root'):
        _parents(Path(request[key]))
    return _snapshot(request, inventory_reader, deadline)


def compare_source_inventories(before, after):
    """Reject all drift except read-induced atime; scheduler capture time is volatile."""
    _require(isinstance(before, dict) and isinstance(after, dict), 'source inventories required')
    _require({k:_stable(v) for k,v in before['nodes'].items()} ==
             {k:_stable(v) for k,v in after['nodes'].items()}, 'source content/metadata/identity drift')
    # job_proof samples time on every call; every other scheduler field is bound.
    _require({k:v for k,v in before['job'].items() if k != 'captured_epoch'} ==
             {k:v for k,v in after['job'].items() if k != 'captured_epoch'}, 'terminal scheduler identity drift')
    for key in ('scope', 'policy_id', 'source_root', 'private_root'):
        _require(before.get(key) == after.get(key), 'source inventory request identity drift')
    return dict(status='STABLE', read_induced_atime_changes={
        name: dict(before=row['restore_metadata']['atime_ns'], after=after['nodes'][name]['restore_metadata']['atime_ns'])
        for name,row in before['nodes'].items()
        if row['restore_metadata']['atime_ns'] != after['nodes'][name]['restore_metadata']['atime_ns']})


def _same_fd(info, row):
    return (stat.S_ISREG(info.st_mode) and info.st_size == row['bytes'] and
            stat.S_IMODE(info.st_mode) == row['mode'] and info.st_uid == row['uid'] and
            info.st_gid == row['gid'] and info.st_mtime_ns == row['mtime_ns'] and
            dict(inode=info.st_ino, device=info.st_dev, ctime_ns=info.st_ctime_ns,
                 link_count=info.st_nlink) == row['source_identity'])


class _Source:
    def __init__(self, stream, deadline): self.stream, self.deadline = stream, deadline
    def read(self, size):
        _deadline(self.deadline)
        data = self.stream.read(min(size, CHUNK))
        _deadline(self.deadline)
        return data


def _application_names(request, before):
    records = request.get('_bound_observation', {}).get('binding', {}).get('records')
    names = set(records or ('supervisor-terminal.json', 'intent.json'))
    if request['guard_status'] == 'RAW_REVIEW_REQUIRED': names.add('worker-result.json')
    return ['policy-tree/' + name for name in sorted(names)]


def _create_namespace(parent, created, free_reader, reserve, deadline):
    backup = _plain(BACKUP_ROOT)
    _parents(backup.parent)
    _require(parent == backup or backup in parent.parents, 'namespace parent outside backup root')
    for path in [backup] + [backup.joinpath(*parent.relative_to(backup).parts[:i])
                             for i in range(1, len(parent.relative_to(backup).parts)+1)]:
        _deadline(deadline)
        try:
            info = path.lstat()
        except FileNotFoundError:
            _parents(path.parent)
            _quota(free_reader, 0, reserve, deadline)
            os.mkdir(path, 0o700)
            created.append(str(path))
            _fsync_dir(path)
            _fsync_dir(path.parent)
        else:
            _require(stat.S_ISDIR(info.st_mode), 'all parents must be nonsymlink directories')


def create_archive(request, destination, *, inventory_reader, free_reader, deadline):
    """Create a new private destination and retain all partial/failure evidence."""
    deadline = _limit_deadline(deadline)
    reserve = _bounds(request)
    destination = validate_destination(request, destination)
    for key in ('source_root', 'private_root'): _parents(Path(request[key]))
    # Validate actual terminal identity before creating any namespace parents.
    before = _snapshot(request, inventory_reader, deadline)
    created = []
    try:
        _create_namespace(destination.parent, created, free_reader, reserve, deadline)
        _parents(destination.parent)
        _quota(free_reader, 0, reserve, deadline)
        os.mkdir(destination, 0o700)
    except BaseException as exc:
        exc.created_parents = tuple(created)
        raise
    try:
        _fsync_dir(destination.parent)
        controls = {}
        def save(name, value, key=None):
            binding = _write_file(destination/name, _json(value, request['max_metadata_bytes']),
                                  free_reader, reserve, deadline, request['max_metadata_bytes'])
            controls[key or name.removesuffix('.json')] = dict(binding, path=str(destination/name))
            return controls[key or name.removesuffix('.json')]
        save('attempt.json', dict(status='INCOMPLETE', policy_id=request['policy_id'], job_id=request['job_id'], created_parents=created))
        save('before.json', before)
        archive_path = destination/'terminal-policy-full.tar.gz'
        _quota(free_reader, 0, reserve, deadline)
        fd = os.open(archive_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            output = _Output(fd, request['max_archive_bytes'], free_reader, reserve, deadline)
            with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0, compresslevel=1) as zipped:
                with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                    for name, row in sorted(before['nodes'].items()):
                        _deadline(deadline)
                        path = Path(row['source_path'])
                        _parents(path.parent)
                        _require(_stable(_metadata(path)) == _stable({k:v for k,v in row.items() if k != 'sha256'}),
                                 'source node drift before archive body')
                        member = tarfile.TarInfo(name)
                        member.mode, member.uid, member.gid = row['mode'], row['uid'], row['gid']
                        member.mtime = row['mtime_ns'] // 1000000000
                        member.pax_headers = {'z1.mtime_ns': str(row['mtime_ns']),
                            'mtime': str(Decimal(row['mtime_ns']) / Decimal(1000000000)),
                            'z1.restore_metadata': _json(row['restore_metadata']).decode(),
                            'z1.source_identity': _json(row['source_identity']).decode()}
                        if row['type'] == 'file':
                            member.size = row['bytes']
                            source_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                            with os.fdopen(source_fd, 'rb') as source:
                                _require(_same_fd(os.fstat(source.fileno()), row), 'source descriptor identity differs')
                                archive.addfile(member, _Source(source, deadline))
                                _require(_same_fd(os.fstat(source.fileno()), row), 'source descriptor drift after body')
                        elif row['type'] == 'directory':
                            member.type = tarfile.DIRTYPE
                            archive.addfile(member)
                        else:
                            member.type, member.linkname = tarfile.SYMTYPE, row['target']
                            archive.addfile(member)
            _deadline(deadline)
            os.fsync(fd)
            _deadline(deadline)
        finally:
            os.close(fd)
        _fsync_dir(destination)
        after = _snapshot(request, inventory_reader, deadline)
        save('after.json', after)
        stability = compare_source_inventories(before, after)
        with os.fdopen(os.open(archive_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
            result = verify_stream(stream, before, max_archive_bytes=request['max_archive_bytes'],
                max_source_bytes=request['max_source_bytes'], max_metadata_bytes=request['max_metadata_bytes'],
                application_names=_application_names(request, before), max_evidence_bytes=request['max_evidence_bytes'], deadline=deadline)
        _require(result['archive_bytes'] == output.count and result['archive_sha256'] == output.digest.hexdigest(),
                 'stored archive differs from written byte binding')
        binding = request.get('_bound_observation', {}).get('binding') or {'records': {
            name.removeprefix('policy-tree/'): dict(value=value, sha256=result['origin_map'][name]['sha256'])
            for name,value in result['application_records'].items()}}
        load_modules()['policy'].validate_archived_application_records(request, binding, result['origin_map'],
            archive_path, deadline, record_reader=lambda _path,name,_deadline,_maximum: result['application_records'][name])
        save('verification.json', result, 'primary')
        save('origin.json', result['origin_map'])
        index = dict(status='ARCHIVE_BOUND', archive_name=archive_path.name, archive_sha256=result['archive_sha256'],
                     archive_bytes=result['archive_bytes'], controls=dict(controls),
                     read_induced_atime_changes=stability['read_induced_atime_changes'],
                     policy_id=request['policy_id'], job_id=request['job_id'], created_parents=created)
        save('index.json', index)
        _deadline(deadline)
        return dict(status='COMPLETE', destination=str(destination), archive_path=str(archive_path),
                    archive_sha256=result['archive_sha256'], archive_bytes=result['archive_bytes'],
                    node_count=result['node_count'], file_count=result['file_count'], regular_bytes=result['regular_bytes'],
                    controls=controls, created_parents=created)
    except BaseException as exc:
        try:
            _write_file(destination/'failure.json', _json(dict(status='FAILED_NO_RETRY', error_type=type(exc).__name__, error=str(exc)[:4096], created_parents=created)),
                        free_reader, reserve, deadline, request['max_metadata_bytes'])
        except BaseException:
            pass  # attempt.json and all partial bytes remain; failure is also raised.
        raise


def _descriptor_identity(info):
    """All stable stat fields, preserving ns precision and exempting only atime."""
    fields = ('st_mode', 'st_uid', 'st_gid', 'st_size', 'st_ino', 'st_dev',
              'st_nlink', 'st_mtime_ns', 'st_ctime_ns', 'st_flags', 'st_birthtime_ns')
    return tuple(getattr(info, key, None) for key in fields)


def _read_control(path, maximum, deadline, expected_sha256):
    _parents(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_size <= maximum, 'bounded regular control required')
        chunks, count = [], 0
        while True:
            _deadline(deadline)
            block = stream.read(min(CHUNK, maximum - count + 1))
            count += len(block)
            _require(count <= maximum, 'control metadata byte cap exceeded')
            if not block:
                break
            chunks.append(block)
        body = b''.join(chunks)
        _require(hashlib.sha256(body).hexdigest() == expected_sha256, 'control/index SHA-256 differs')
        _require(_descriptor_identity(os.fstat(stream.fileno())) == _descriptor_identity(info), 'control identity drift during read')
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'duplicate control JSON key')
            value[key] = item
        return value
    def constant(value):
        raise ValueError('nonfinite control JSON')
    value = json.loads(body, object_pairs_hook=pairs, parse_constant=constant)
    _json(value, maximum)
    return value


def stream_back(request, destination, sink, *, expected_index_sha256,
                expected_archive_sha256, expected_archive_bytes, deadline):
    """Stream only a caller-bound completed create response, never discover a PASS.

    Task3 must first require the successful create response and process returncode
    zero. An ARCHIVE_BOUND index alone is recovery evidence, never admission.
    Local data is prehashed on one nonsymlink descriptor before any sink write;
    a second digest/fstat checks concurrent drift during emission.
    """
    deadline = _limit_deadline(deadline)
    _bounds(request)
    destination = validate_destination(request, destination)
    _parents(destination)
    _require(not os.path.lexists(destination/'failure.json'), 'archive has retained failure evidence')
    for value in (expected_index_sha256, expected_archive_sha256):
        _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'exact SHA binding required')
    _require(type(expected_archive_bytes) is int and 0 < expected_archive_bytes <= request['max_archive_bytes'],
             'bounded archive size binding required')
    index = _read_control(destination/'index.json', request['max_metadata_bytes'], deadline, expected_index_sha256)
    _require(index.get('status') == 'ARCHIVE_BOUND' and index.get('policy_id') == request['policy_id'] and
             index.get('job_id') == request['job_id'] and index.get('archive_name') == 'terminal-policy-full.tar.gz' and
             index.get('archive_sha256') == expected_archive_sha256 and index.get('archive_bytes') == expected_archive_bytes,
             'index destination/policy/job/archive binding differs')
    for key, name in [('before','before.json'),('after','after.json'),('origin','origin.json'),('primary','verification.json')]:
        ref = index.get('controls', {}).get(key, {})
        _require(ref.get('path') == str(destination/name) and type(ref.get('bytes')) is int and
                 0 < ref['bytes'] <= request['max_metadata_bytes'], 'index control path/size differs')
        value = _read_control(destination/name, request['max_metadata_bytes'], deadline, ref.get('sha256'))
        _require((destination/name).stat().st_size == ref['bytes'], 'index control size binding differs')
        if key in ('before', 'after'):
            for field in ('policy_id', 'scope', 'source_root', 'private_root'):
                _require(value.get(field) == request[field], 'index source identity differs')
            _require(value.get('job', {}).get('job_id') == request['job_id'], 'index source job identity differs')
        if key == 'primary':
            _require(value.get('archive_sha256') == expected_archive_sha256 and value.get('archive_bytes') == expected_archive_bytes,
                     'primary verification archive binding differs')
    path = destination/'terminal-policy-full.tar.gz'
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_size == expected_archive_bytes, 'stored archive size/type differs')
        digest = hashlib.sha256()
        while True:
            _deadline(deadline)
            block = stream.read(CHUNK)
            if not block:
                break
            digest.update(block)
        _require(digest.hexdigest() == expected_archive_sha256, 'stored archive SHA differs')
        _require(_descriptor_identity(os.fstat(stream.fileno())) == _descriptor_identity(info), 'archive identity changed before stream-back')
        stream.seek(0)
        digest, count = hashlib.sha256(), 0
        while True:
            _deadline(deadline)
            block = stream.read(CHUNK)
            if not block:
                break
            digest.update(block)
            count += len(block)
            view = memoryview(block)
            while view:
                _deadline(deadline)
                written = sink.write(view)
                _require(type(written) is int and 0 < written <= len(view), 'stream-back short write failed')
                view = view[written:]
        _deadline(deadline)
        _require(count == expected_archive_bytes and digest.hexdigest() == expected_archive_sha256 and
                 _descriptor_identity(os.fstat(stream.fileno())) == _descriptor_identity(info), 'archive drift during stream-back')
    return dict(archive_sha256=expected_archive_sha256, archive_bytes=count, index_sha256=expected_index_sha256)
