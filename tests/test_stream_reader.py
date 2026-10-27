"""Pure, synthetic, nonseekable archive contract tests (no repository imports)."""
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from vega_policy_archive.stream_reader import verify_stream
except ModuleNotFoundError:
    verify_stream = None

MIB = 1024 * 1024


class Nonseekable:
    def __init__(self, data, fragment=97):
        self.data = data
        self.position = 0
        self.fragment = fragment
        self.eof_seen = False
        self.read_sizes = []

    def read(self, size):
        if not 0 < size <= MIB:
            raise AssertionError('unbounded read')
        self.read_sizes.append(size)
        end = min(len(self.data), self.position + min(size, self.fragment))
        result = self.data[self.position:end]
        self.position = end
        if not result:
            self.eof_seen = True
        return result

    def seek(self, *args):
        raise AssertionError('seek forbidden')


def fixture():
    bodies = {'policy-tree/terminal.json': b'{"status":"FAILED_NO_RETRY"}',
              'private-tree/partial.bin': b'partial\x00data'}
    nodes = {}
    for name in ('policy-tree', 'private-tree', 'policy-tree/empty',
                 'policy-tree/terminal.json', 'private-tree/partial.bin',
                 'private-tree/link'):
        row = dict(source_path='/synthetic/' + name, mode=0o750, uid=123,
                   gid=456, mtime_ns=1700000000123456789)
        if name in bodies:
            row.update(type='file', bytes=len(bodies[name]),
                       sha256=hashlib.sha256(bodies[name]).hexdigest())
        elif name.endswith('/link'):
            row.update(type='symlink', target='partial.bin')
        else:
            row['type'] = 'directory'
        nodes[name] = row
    return {'nodes': nodes}, bodies


def make_tar(before, bodies, mutate=None):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w', format=tarfile.PAX_FORMAT) as archive:
        entries = []
        for name, row in before['nodes'].items():
            item = tarfile.TarInfo(name)
            item.mode, item.uid, item.gid = row['mode'], row['uid'], row['gid']
            item.mtime = row['mtime_ns'] / 1e9
            item.pax_headers = {'z1.mtime_ns': str(row['mtime_ns'])}
            for key in ('restore_metadata', 'source_identity'):
                if key in row:
                    item.pax_headers['z1.' + key] = json.dumps(row[key], sort_keys=True, separators=(',', ':'), allow_nan=False)
            body = bodies.get(name, b'')
            if row['type'] == 'file':
                item.size = len(body)
            elif row['type'] == 'directory':
                item.type = tarfile.DIRTYPE
            else:
                item.type, item.linkname = tarfile.SYMTYPE, row['target']
            entries.append((item, body))
        if mutate:
            entries = mutate(entries)
        for item, body in entries:
            archive.addfile(item, io.BytesIO(body))
    return output.getvalue()


class StreamReaderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(verify_stream, 'bounded stream verifier is not implemented')
        self.before, self.bodies = fixture()

    def verify(self, payload=None, before=None, **kwargs):
        if payload is None:
            payload = gzip.compress(make_tar(self.before, self.bodies), mtime=0)
        stream = Nonseekable(payload)
        options = dict(max_archive_bytes=MIB, max_source_bytes=MIB,
                       max_metadata_bytes=65536,
                       application_names=('policy-tree/terminal.json',))
        options.update(kwargs)
        result = verify_stream(stream, self.before if before is None else before, **options)
        return result, stream

    def rejects(self, payload=None, before=None, **kwargs):
        with self.assertRaises((ValueError, TimeoutError)):
            self.verify(payload, before, **kwargs)

    def changed(self, mutate):
        return gzip.compress(make_tar(self.before, self.bodies, mutate), mtime=0)

    def test_nonseekable_exact_two_roots_and_full_compressed_digest(self):
        payload = gzip.compress(make_tar(self.before, self.bodies), mtime=0)
        result, stream = self.verify(payload)
        self.assertEqual(result['origin_map'], self.before['nodes'])
        self.assertEqual(result['application_records'], {
            'policy-tree/terminal.json': {'status': 'FAILED_NO_RETRY'}})
        self.assertEqual(result['archive_sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result['archive_bytes'], len(payload))
        self.assertEqual(result['node_count'], 6)
        self.assertEqual(result['file_count'], 2)
        self.assertEqual(result['regular_bytes'], sum(map(len, self.bodies.values())))
        self.assertTrue(stream.eof_seen)
        self.assertLessEqual(max(stream.read_sizes), MIB)

    def test_optional_restore_and_source_identity_preserved(self):
        row = self.before['nodes']['policy-tree']
        row['restore_metadata'] = {'atime_ns': 123, 'xattrs': {
            'user.note': {'value_b64': 'eA==', 'sha256': hashlib.sha256(b'x').hexdigest(), 'bytes': 1}}}
        row['source_identity'] = {'inode': 1, 'device': 2, 'ctime_ns': 3, 'link_count': 1}
        result, _ = self.verify()
        self.assertEqual(result['origin_map'], self.before['nodes'])

    def test_optional_restore_metadata_missing_or_different(self):
        self.before['nodes']['policy-tree']['restore_metadata'] = {'atime_ns': 123}
        for value in (None, '{"atime_ns":124}', '{ "atime_ns": 123 }'):
            with self.subTest(value=value):
                def change(items):
                    if value is None:
                        del items[0][0].pax_headers['z1.restore_metadata']
                    else:
                        items[0][0].pax_headers['z1.restore_metadata'] = value
                    return items
                self.rejects(self.changed(change))

    def test_unbound_extra_z1_metadata(self):
        for key in ('z1.restore_metadata', 'z1.source_identity', 'z1.unknown'):
            with self.subTest(key=key):
                def change(items):
                    items[0][0].pax_headers[key] = '{}'
                    return items
                self.rejects(self.changed(change))

    def test_file_spanning_multiple_bounded_hash_chunks(self):
        body = b'0123456789abcdef' * (MIB // 8) + b'end'
        self.bodies['private-tree/partial.bin'] = body
        row = self.before['nodes']['private-tree/partial.bin']
        row.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        result, _ = self.verify(max_source_bytes=3 * MIB)
        self.assertEqual(result['origin_map']['private-tree/partial.bin'], row)

    def test_exact_compressed_and_source_caps_succeed(self):
        payload = gzip.compress(make_tar(self.before, self.bodies), mtime=0)
        result, _ = self.verify(payload, max_archive_bytes=len(payload),
                                max_source_bytes=sum(map(len, self.bodies.values())),
                                max_evidence_bytes=len(self.bodies['policy-tree/terminal.json']))
        self.assertEqual(result['archive_bytes'], len(payload))

    def test_one_byte_transport_fragments(self):
        payload = gzip.compress(make_tar(self.before, self.bodies), mtime=0)
        stream = Nonseekable(payload, fragment=1)
        result = verify_stream(stream, self.before, max_archive_bytes=MIB,
                               max_source_bytes=MIB, max_metadata_bytes=65536)
        self.assertEqual(result['archive_bytes'], len(payload))
        self.assertTrue(stream.eof_seen)

    def test_missing_member(self):
        self.rejects(self.changed(lambda items: items[:-1]))

    def test_extra_member(self):
        def change(items):
            item = tarfile.TarInfo('policy-tree/unexpected')
            return items + [(item, b'')]
        self.rejects(self.changed(change))

    def test_duplicate_member(self):
        self.rejects(self.changed(lambda items: items + [items[0]]))

    def test_unsafe_names(self):
        for name in ('/absolute', 'policy-tree/../escape', 'policy-tree//alias',
                     './policy-tree', 'policy-tree/./alias', 'foreign-tree/x',
                     'policy-tree\\escape'):
            with self.subTest(name=name):
                def change(items):
                    items[0][0].name = name
                    return items
                self.rejects(self.changed(change))

    def test_hardlink_rejected_even_with_expected_file_name(self):
        def change(items):
            item = items[3][0]
            item.type, item.linkname, item.size = tarfile.LNKTYPE, 'private-tree/partial.bin', 0
            return items
        self.rejects(self.changed(change))

    def test_special_node(self):
        def change(items):
            items[0][0].type = tarfile.FIFOTYPE
            return items
        self.rejects(self.changed(change))

    def test_file_body_drift(self):
        self.bodies['private-tree/partial.bin'] = b'X' * len(self.bodies['private-tree/partial.bin'])
        self.rejects()

    def test_file_size_drift(self):
        self.bodies['private-tree/partial.bin'] += b'X'
        self.rejects()

    def test_metadata_drift(self):
        for key in ('mode', 'uid', 'gid'):
            with self.subTest(key=key):
                def change(items):
                    setattr(items[0][0], key, getattr(items[0][0], key) + 1)
                    return items
                self.rejects(self.changed(change))

    def test_mtime_ns_drift_or_missing(self):
        for value in (None, '1700000000123456790'):
            with self.subTest(value=value):
                def change(items):
                    items[0][0].pax_headers = {} if value is None else {'z1.mtime_ns': value}
                    return items
                self.rejects(self.changed(change))

    def test_symlink_target_drift(self):
        def change(items):
            items[-1][0].linkname = '../../outside'
            return items
        self.rejects(self.changed(change))

    def test_truncated_gzip_header_body_and_trailer(self):
        payload = gzip.compress(make_tar(self.before, self.bodies), mtime=0)
        for cut in (1, 10, len(payload) // 2, len(payload) - 1, len(payload) - 8):
            with self.subTest(cut=cut):
                self.rejects(payload[:cut])

    def test_bad_gzip_crc(self):
        payload = bytearray(gzip.compress(make_tar(self.before, self.bodies), mtime=0))
        payload[-8] ^= 1
        self.rejects(bytes(payload))

    def test_gzip_trailing_payload_and_concatenated_member(self):
        payload = gzip.compress(make_tar(self.before, self.bodies), mtime=0)
        for tail in (b'garbage', b'\x00', gzip.compress(b'\x00' * 1024)):
            with self.subTest(tail=tail[:10]):
                self.rejects(payload + tail)

    def test_nonzero_tar_tail(self):
        raw = make_tar(self.before, self.bodies)
        self.rejects(gzip.compress(raw + b'X'))

    def test_hidden_tar_member_after_end(self):
        raw = make_tar(self.before, self.bodies)
        self.rejects(gzip.compress(raw + make_tar(self.before, self.bodies)))

    def test_missing_tar_terminator(self):
        raw = make_tar(self.before, self.bodies)
        last = raw.rfind(b'partial\x00data')
        self.rejects(gzip.compress(raw[:((last + len(b'partial\x00data') + 511) // 512) * 512]))

    def test_bad_tar_checksum(self):
        raw = bytearray(make_tar(self.before, self.bodies))
        raw[0] ^= 1
        self.rejects(gzip.compress(bytes(raw)))

    def test_pax_metadata_bound(self):
        def change(items):
            items[0][0].pax_headers['comment'] = 'X' * 70000
            return items
        self.rejects(self.changed(change))

    def test_decompression_bomb_tail(self):
        raw = make_tar(self.before, self.bodies)
        self.rejects(gzip.compress(raw + b'\x00' * (15 * MIB)))

    def test_compressed_byte_cap(self):
        payload = gzip.compress(make_tar(self.before, self.bodies), mtime=0)
        self.rejects(payload, max_archive_bytes=len(payload) - 1)

    def test_source_byte_cap(self):
        self.rejects(max_source_bytes=sum(map(len, self.bodies.values())) - 1)

    def test_inventory_node_cap(self):
        before = copy.deepcopy(self.before)
        before['nodes'].update({f'policy-tree/{i}': before['nodes']['policy-tree'] for i in range(8193)})
        self.rejects(before=before)

    def test_inventory_root_contract(self):
        before = copy.deepcopy(self.before)
        del before['nodes']['private-tree']
        self.rejects(before=before)

    def test_inventory_symlink_parent(self):
        before = copy.deepcopy(self.before)
        before['nodes']['private-tree/link/child'] = before['nodes']['policy-tree']
        self.rejects(before=before)

    def test_invalid_json(self):
        for body in (b'not json', b'{"x":NaN}', b'{"x":1,"x":2}', b'\xff'):
            with self.subTest(body=body):
                self.bodies['policy-tree/terminal.json'] = body
                row = self.before['nodes']['policy-tree/terminal.json']
                row.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
                self.rejects()

    def test_nonfinite_numeric_json(self):
        self.bodies['policy-tree/terminal.json'] = b'{"x":1e9999}'
        row = self.before['nodes']['policy-tree/terminal.json']
        row.update(bytes=len(self.bodies['policy-tree/terminal.json']),
                   sha256=hashlib.sha256(self.bodies['policy-tree/terminal.json']).hexdigest())
        self.rejects()

    def test_integer_json_digit_bound(self):
        body = b'{"x":' + b'1' * 4097 + b'}'
        self.bodies['policy-tree/terminal.json'] = body
        row = self.before['nodes']['policy-tree/terminal.json']
        row.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        self.rejects()

    def test_pax_integral_digit_bound(self):
        for key in ('uid', 'gid', 'size'):
            with self.subTest(key=key):
                def change(items):
                    item = items[3 if key == 'size' else 0][0]
                    item.pax_headers[key] = '0' * 21 + str(getattr(item, key))
                    return items
                self.rejects(self.changed(change))

    def test_pax_timestamp_syntax_bound(self):
        for value in ('Infinity', 'NaN', '1e9999', 'x', '1' * 129):
            with self.subTest(value=value):
                def change(items):
                    items[0][0].pax_headers['mtime'] = value
                    return items
                self.rejects(self.changed(change))

    def test_negative_ns_timestamp_and_decimal_pax_timestamp(self):
        self.before['nodes']['policy-tree']['mtime_ns'] = -123456789
        result, _ = self.verify()
        self.assertEqual(result['origin_map']['policy-tree']['mtime_ns'], -123456789)

    def test_declared_member_size_overrides_are_bounded(self):
        def change(items):
            items[3][0].pax_headers['size'] = '99999999999999999999999999999'
            return items
        self.rejects(self.changed(change))

    def test_pax_duplicate_keys_and_malformed_length(self):
        def pax_line(key, value):
            tail = (' ' + key + '=' + value + '\n').encode()
            length = len(tail) + 1
            while len(str(length)) + len(tail) != length:
                length = len(str(length)) + len(tail)
            return str(length).encode() + tail
        raw = make_tar(self.before, self.bodies)
        first = tarfile.TarInfo.frombuf(raw[:512], 'utf-8', 'strict')
        remainder = raw[512 + ((first.size + 511) // 512) * 512:]
        line = pax_line('z1.mtime_ns', str(self.before['nodes']['policy-tree']['mtime_ns']))
        for body in (line + line, b'99999 key=value\n', b'0 x=y\n', b'8 x=\xff\n'):
            with self.subTest(body=body):
                first.size = len(body)
                changed = first.tobuf(format=tarfile.USTAR_FORMAT) + body
                changed += b'\x00' * ((-len(body)) % 512) + remainder
                self.rejects(gzip.compress(changed))

    def test_pax_unsafe_path_override(self):
        def change(items):
            items[0][0].pax_headers['path'] = 'policy-tree/../private-tree'
            return items
        self.rejects(self.changed(change))

    def test_optional_source_identity_drift(self):
        self.before['nodes']['policy-tree']['source_identity'] = {'inode': 1}
        def change(items):
            items[0][0].pax_headers['z1.source_identity'] = '{"inode":2}'
            return items
        self.rejects(self.changed(change))

    def test_transport_contract_invalid_returns(self):
        for value in (None, bytearray(b'x'), b'x' * (MIB + 1)):
            with self.subTest(kind=type(value)):
                class Invalid:
                    def read(self, size):
                        return value
                with self.assertRaises(ValueError):
                    verify_stream(Invalid(), self.before, max_archive_bytes=MIB,
                                  max_source_bytes=MIB, max_metadata_bytes=65536)

    def test_inventory_metadata_bound(self):
        self.rejects(max_metadata_bytes=20)

    def test_duplicate_application_names(self):
        self.rejects(application_names=('policy-tree/terminal.json',) * 2)

    def test_individual_application_bound(self):
        self.rejects(max_evidence_bytes=4)

    def test_aggregate_application_bound(self):
        before = copy.deepcopy(self.before)
        for i in range(3):
            name = f'policy-tree/{i}.json'
            self.bodies[name] = b'{}' + b' ' * 8
            before['nodes'][name] = dict(before['nodes']['policy-tree/terminal.json'],
                                        bytes=10, sha256=hashlib.sha256(self.bodies[name]).hexdigest())
        self.rejects(gzip.compress(make_tar(before, self.bodies)), before=before,
                     max_evidence_bytes=10,
                     application_names=tuple(f'policy-tree/{i}.json' for i in range(3)))

    def test_application_must_exist_and_be_file(self):
        for name in ('policy-tree/missing.json', 'private-tree/link', 'policy-tree'):
            with self.subTest(name=name):
                self.rejects(application_names=(name,))

    def test_expired_deadline(self):
        self.rejects(deadline=time.monotonic() - 1)

    def test_deadline_checked_after_read(self):
        class Expiring(Nonseekable):
            def read(self, size):
                time.sleep(0.002)
                return super().read(size)
        payload = gzip.compress(make_tar(self.before, self.bodies))
        with self.assertRaises(TimeoutError):
            verify_stream(Expiring(payload), self.before, max_archive_bytes=MIB,
                          max_source_bytes=MIB, max_metadata_bytes=65536,
                          deadline=time.monotonic() + 0.001)

    def test_invalid_bounds(self):
        for key, value in (('max_archive_bytes', 0), ('max_source_bytes', -1),
                           ('max_metadata_bytes', True), ('max_evidence_bytes', 0)):
            with self.subTest(key=key):
                self.rejects(**{key: value})


if __name__ == '__main__':
    unittest.main()
