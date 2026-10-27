"""Verify a single gzip/PAX tar from a bounded, nonseekable byte stream.

No archive body is written, extracted, imported, or retained. Only requested JSON
records and validated inventory metadata are returned. ``deadline`` is an
absolute monotonic timestamp; callers must separately bound blocking transport
reads, since Python cannot interrupt an arbitrary supplied ``read`` method.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
import tarfile
import time
import zlib

_CHUNK = 1024 * 1024
_MAX_NODES = 8192
_ROOTS = {'policy-tree', 'private-tree'}
_OPTIONAL = ('restore_metadata', 'source_identity')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('archive verification deadline expired')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _safe_name(name):
    _require(isinstance(name, str) and name and '\\' not in name and
             all(ord(char) >= 32 and ord(char) != 127 for char in name),
             'unsafe archive member name')
    parts = name.split('/')
    _require(parts[0] in _ROOTS and all(part not in ('', '.', '..') for part in parts),
             'unsafe archive member path')
    return name


def _json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            _require(key not in result, 'duplicate JSON object key')
            result[key] = value
        return result

    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)

    def bounded_int(value):
        _require(len(value.lstrip('-')) <= 4096, 'JSON integer token digit cap exceeded')
        return int(value)

    def finite_float(value):
        result = float(value)
        _require(math.isfinite(result), 'nonfinite JSON number')
        return result

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant,
                          parse_float=finite_float, parse_int=bounded_int)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('invalid bounded JSON') from exc


class _GzipReader:
    """Incremental single-member gzip decoder, including CRC and transport EOF."""
    def __init__(self, stream, compressed_cap, expanded_cap, deadline):
        self.stream = stream
        self.compressed_cap = compressed_cap
        self.expanded_cap = expanded_cap
        self.deadline = deadline
        self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.pending = b''
        self.count = 0
        self.expanded = 0
        self.digest = hashlib.sha256()
        self.finished = False

    def _compressed(self):
        _deadline(self.deadline)
        # One extra byte establishes exhaustion without an unbounded read.
        size = min(_CHUNK, max(1, self.compressed_cap - self.count + 1))
        data = self.stream.read(size)
        _deadline(self.deadline)
        _require(isinstance(data, bytes) and len(data) <= size,
                 'transport must return bounded bytes')
        self.count += len(data)
        _require(self.count <= self.compressed_cap, 'compressed archive byte cap exceeded')
        self.digest.update(data)
        return data

    def read(self, size):
        _require(type(size) is int and 0 < size <= _CHUNK, 'bounded read size required')
        while not self.finished:
            _deadline(self.deadline)
            if self.decoder.eof:
                _require(not self.decoder.unused_data, 'extra compressed payload after gzip member')
                _require(not self._compressed(), 'extra compressed payload after gzip member')
                self.finished = True
                return b''
            if not self.pending:
                self.pending = self._compressed()
                _require(self.pending, 'truncated gzip stream')
            try:
                block = self.decoder.decompress(self.pending, size)
            except zlib.error as exc:
                raise ValueError('invalid gzip stream or CRC') from exc
            self.pending = self.decoder.unconsumed_tail
            self.expanded += len(block)
            _require(self.expanded <= self.expanded_cap, 'decompressed archive byte cap exceeded')
            _require(not self.decoder.unused_data, 'extra compressed payload after gzip member')
            if block:
                return block
        return b''

    def exact(self, size):
        _require(0 <= size <= _CHUNK, 'bounded exact read required')
        chunks = []
        remaining = size
        while remaining:
            block = self.read(remaining)
            _require(block, 'truncated tar record')
            chunks.append(block)
            remaining -= len(block)
        return b''.join(chunks)

    def padding(self, size):
        if size:
            _require(not any(self.exact(size)), 'nonzero tar member padding')


def _pax(data):
    """Parse bounded POSIX PAX length records without tarfile's buffering."""
    values, position = {}, 0
    while position < len(data):
        space = data.find(b' ', position)
        _require(space > position and space - position <= 12, 'malformed PAX record length')
        length_bytes = data[position:space]
        _require(length_bytes.isdigit() and not length_bytes.startswith(b'0'),
                 'malformed PAX record length')
        length = int(length_bytes)
        end = position + length
        _require(space + 2 < end <= len(data) and data[end - 1:end] == b'\n',
                 'truncated PAX record')
        key, separator, value = data[space + 1:end - 1].partition(b'=')
        _require(key and separator, 'malformed PAX key/value')
        try:
            key, value = key.decode('utf-8'), value.decode('utf-8')
        except UnicodeError as exc:
            raise ValueError('invalid PAX UTF-8') from exc
        _require(key not in values and '\x00' not in key and '\x00' not in value,
                 'duplicate or invalid PAX key')
        values[key] = value
        position = end
    return values


def _integral(value):
    digits = value[1:] if value[:1] in ('+', '-') else value
    _require(0 < len(digits) <= 20 and all('0' <= char <= '9' for char in digits),
             'PAX integral field syntax/digit cap exceeded')
    return int(value)


def _timestamp(value):
    _require(len(value) <= 128 and re.fullmatch(
        r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', value),
        'PAX timestamp syntax/length cap exceeded')
    try:
        stamp = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('invalid PAX timestamp') from exc
    _require(stamp.is_finite() and (stamp.is_zero() or stamp.adjusted() <= 20),
             'PAX timestamp range exceeded')


def _inventory(before, source_cap, metadata_cap, deadline):
    _require(isinstance(before, dict) and isinstance(before.get('nodes'), dict),
             'two-root inventory nodes required')
    nodes = before['nodes']
    _require(2 <= len(nodes) <= _MAX_NODES and _ROOTS.issubset(nodes),
             'inventory root/member count differs')
    regular_bytes = 0
    for name, item in nodes.items():
        _deadline(deadline)
        _safe_name(name)
        _require(isinstance(item, dict), 'inventory row must be an object')
        _require(isinstance(item.get('source_path'), str) and item['source_path'].startswith('/'),
                 'absolute inventory source path required')
        for key in ('mode', 'uid', 'gid', 'mtime_ns'):
            _require(type(item.get(key)) is int, 'integer inventory metadata required: ' + key)
        _require(0 <= item['mode'] <= 0o7777 and item['uid'] >= 0 and item['gid'] >= 0,
                 'invalid inventory mode/ownership')
        kind = item.get('type')
        _require(kind in ('file', 'directory', 'symlink'), 'unsupported inventory node type')
        _require(name not in _ROOTS or kind == 'directory', 'inventory root must be directory')
        if '/' in name:
            parent = nodes.get(name.rsplit('/', 1)[0], {})
            _require(parent.get('type') == 'directory', 'missing or nondirectory inventory parent')
        if kind == 'file':
            _require(type(item.get('bytes')) is int and item['bytes'] >= 0,
                     'nonnegative inventory file bytes required')
            sha = item.get('sha256')
            _require(isinstance(sha, str) and len(sha) == 64 and
                     all(char in '0123456789abcdef' for char in sha), 'invalid inventory SHA-256')
            regular_bytes += item['bytes']
            _require(regular_bytes <= source_cap, 'source byte cap exceeded')
        elif kind == 'symlink':
            _require(isinstance(item.get('target'), str) and '\x00' not in item['target'],
                     'invalid symlink target')
        for key in _OPTIONAL:
            if key in item:
                _require(isinstance(item[key], dict), 'optional restore/identity metadata must be object')
    # Bound the caller's metadata before retaining a second representation.
    count = 0
    try:
        for piece in json.JSONEncoder(allow_nan=False, separators=(',', ':')).iterencode(nodes):
            _deadline(deadline)
            count += len(piece.encode('utf-8'))
            _require(count <= metadata_cap, 'inventory metadata byte cap exceeded')
    except (TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError('invalid inventory JSON metadata') from exc
    return nodes, regular_bytes


def verify_stream(stream, before, *, max_archive_bytes, max_source_bytes,
                  max_metadata_bytes, application_names=(),
                  max_evidence_bytes=8388608, deadline=None):
    """Return exact validated maps/counts only after gzip CRC and transport EOF.

    Accepts the existing ``policy-tree``/``private-tree`` inventory schema.
    Optional ``restore_metadata`` and ``source_identity`` dictionaries must have
    exact canonical JSON counterparts in ``z1.<field>`` PAX records. The metadata
    cap independently bounds the inventory JSON and aggregate PAX payload bytes.
    Tar node count is limited to 8192. A single gzip member is required; all bytes
    after its trailer are rejected, including a second member or zero padding.
    """
    for name, value, hard in (
            ('max_archive_bytes', max_archive_bytes, 201326592),
            ('max_source_bytes', max_source_bytes, 167772160),
            ('max_metadata_bytes', max_metadata_bytes, 4194304),
            ('max_evidence_bytes', max_evidence_bytes, 8388608)):
        _require(type(value) is int and 0 < value <= hard, 'invalid finite bound: ' + name)
    if deadline is None:
        deadline = time.monotonic() + 600
    _require(type(deadline) in (int, float) and math.isfinite(deadline),
             'finite monotonic deadline required')
    deadline = min(deadline, time.monotonic() + 600)
    _deadline(deadline)
    expected, expected_bytes = _inventory(before, max_source_bytes, max_metadata_bytes, deadline)
    _require(isinstance(application_names, (tuple, list, set, frozenset)) and
             len(application_names) <= _MAX_NODES, 'bounded application names required')
    applications = set()
    for name in application_names:
        _safe_name(name)
        _require(name not in applications, 'duplicate application record name')
        applications.add(name)
        _require(expected.get(name, {}).get('type') == 'file', 'application record must be inventory file')
        _require(expected[name]['bytes'] <= max_evidence_bytes, 'application record exceeds evidence cap')
    _require(sum(expected[name]['bytes'] for name in applications) <= 2 * max_evidence_bytes,
             'aggregate application records exceed evidence cap')
    # Per node: normal header, extension header, and both padding allowances.
    reader = _GzipReader(stream, max_archive_bytes,
                         expected_bytes + len(expected) * 2048 + max_metadata_bytes + 10240,
                         deadline)
    origin_map, application_records = {}, {}
    pax, metadata_bytes, header_count, file_count, regular_bytes = {}, 0, 0, 0, 0
    while True:
        _deadline(deadline)
        header = reader.exact(512)
        if not any(header):
            _require(not pax, 'orphan PAX extension')
            _require(not any(reader.exact(512)), 'tar requires two zero terminator blocks')
            while True:
                block = reader.read(_CHUNK)
                if not block:
                    break
                _require(not any(block), 'nonzero data after tar terminator')
            break
        header_count += 1
        _require(header_count <= 2 * len(expected), 'tar member/header count cap exceeded')
        try:
            member = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
        except (tarfile.HeaderError, UnicodeError, ValueError) as exc:
            raise ValueError('invalid tar header') from exc
        _require(member.size >= 0, 'negative tar member size')
        if member.type == tarfile.XHDTYPE:
            _require(not pax, 'stacked PAX extension')
            metadata_bytes += member.size
            _require(metadata_bytes <= max_metadata_bytes, 'PAX metadata byte cap exceeded')
            chunks, remaining = [], member.size
            while remaining:
                block = reader.exact(min(remaining, _CHUNK))
                chunks.append(block)
                remaining -= len(block)
            pax = _pax(b''.join(chunks))
            reader.padding((-member.size) % 512)
            continue
        _require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE),
                 'unsupported tar member type (hardlinks/sparse/special nodes forbidden)')
        _require(not any(key.startswith('GNU.sparse') for key in pax), 'sparse archive member forbidden')
        name = pax.get('path', member.name)
        if member.isdir() and name.endswith('/'):
            name = name[:-1]
        _safe_name(name)
        _require(name in expected and name not in origin_map, 'unexpected or duplicate archive member')
        item = expected[name]
        size = member.size
        if 'size' in pax:
            size = _integral(pax['size'])
            _require(size >= 0, 'negative PAX member size')
        for key in ('uid', 'gid'):
            value = pax.get(key, str(getattr(member, key)))
            _require(_integral(value) == item[key], 'archive ownership differs')
        for key in ('mtime', 'atime', 'ctime'):
            if key in pax:
                _timestamp(pax[key])
        _require(member.mode == item['mode'] and pax.get('z1.mtime_ns') == str(item['mtime_ns']),
                 'archive node metadata differs from inventory')
        allowed_z1 = {'z1.mtime_ns'}
        record = {key: item[key] for key in ('source_path', 'type', 'mode', 'uid', 'gid', 'mtime_ns')}
        for key in _OPTIONAL:
            if key in item:
                allowed_z1.add('z1.' + key)
                _require(pax.get('z1.' + key) == _canonical(item[key]),
                         'archive optional metadata differs: ' + key)
                # JSON round-trip avoids aliasing mutable caller-owned metadata.
                record[key] = _json(_canonical(item[key]))
        _require(all(not key.startswith('z1.') or key in allowed_z1 for key in pax),
                 'unbound z1 archive metadata')
        kind = item['type']
        if kind == 'directory':
            _require(member.isdir() and size == 0, 'directory archive type/size differs')
        elif kind == 'symlink':
            _require(member.issym() and size == 0 and
                     pax.get('linkpath', member.linkname) == item['target'], 'symlink archive target differs')
            record['target'] = item['target']
        else:
            _require(member.isfile() and size == item['bytes'], 'regular archive size/type differs')
            regular_bytes += size
            _require(regular_bytes <= max_source_bytes, 'source byte cap exceeded')
            digest, chunks, remaining = hashlib.sha256(), [], size
            while remaining:
                block = reader.exact(min(remaining, _CHUNK))
                digest.update(block)
                if name in applications:
                    chunks.append(block)
                remaining -= len(block)
            _require(digest.hexdigest() == item['sha256'], 'regular archive SHA differs from inventory')
            record.update(bytes=size, sha256=digest.hexdigest())
            file_count += 1
            if name in applications:
                try:
                    application_records[name] = _json(b''.join(chunks))
                except (ValueError, OverflowError) as exc:
                    raise ValueError('invalid application JSON: ' + name) from exc
        reader.padding((-size) % 512)
        origin_map[name] = record
        pax = {}
    _require(set(origin_map) == set(expected), 'archive member set incomplete')
    _require(regular_bytes == expected_bytes, 'archive regular byte count differs')
    _deadline(deadline)
    return dict(origin_map=origin_map, application_records=application_records,
                archive_sha256=reader.digest.hexdigest(), archive_bytes=reader.count,
                node_count=len(origin_map), file_count=file_count, regular_bytes=regular_bytes)
