"""File-only deferred checkpoint binding. No authority creation or model imports."""
import importlib
from pathlib import Path
import re
import sys
from . import release as a
from .files import canonical, digest, need, read_json, reference, regular_path, sha_file, verify_ref


def validate_producer(producer):
    need(type(producer) is dict and digest(producer) == a.PRODUCER_SHA, 'frozen producer manifest differs')
    need(producer['root'] == a.PRODUCER_ROOT and producer['source_root'] == a.PRODUCER_ROOT+'/source', 'producer source/root differs')
    need(len(producer['source_files']) == 688 and len(producer['runtime_files']) == 375
         and len(producer['asset_files']) == 32 and len(producer['compiler_policy']['runtime_files']) == 3,
         'complete frozen producer inventories required')
    return producer


def join_completed_producer(producer, slot_seals, completions, slot_finals):
    """Pure join. Inputs are recomputed seals, never trusted as self-attestation."""
    validate_producer(producer)
    need(type(slot_seals) is list and len(slot_seals) == 9 and type(slot_finals) is list and len(slot_finals) == 9,
         'all nine producer slot seals/finals required')
    need(type(completions) is list and len(completions) == 30, 'all thirty train completions required')
    results = {}
    for slot, (seal, final) in enumerate(zip(slot_seals, slot_finals)):
        indices = [c['index'] for c in producer['cells'] if c['slot'] == slot]
        need(type(seal) is dict and seal.get('slot') == slot and seal.get('status') == 'COMPLETE'
             and seal.get('manifest_sha256') == a.PRODUCER_SHA
             and [c['index'] for c in seal.get('cells',[])] == indices, 'recomputed slot seal differs')
        need(type(final) is dict and final.get('schema') == 'z1_thirty_slot020'
             and final.get('status') == 'COMPLETE' and final.get('error') is None
             and final.get('slot') == slot and final.get('manifest_sha256') == a.PRODUCER_SHA
             and final.get('job_id') == a.PRODUCER_JOBS[slot]
             and final.get('completed') == indices and canonical(final.get('artifacts')) == canonical(seal),
             'producer slot final/seal/job join differs')
        for result in seal['cells']:
            index = result['index']; wrapper = producer['cells'][index]; completion = completions[index]
            need(type(index) is int and index not in results and result.get('training_cell') == wrapper['training_cell']
                 and result.get('updates') == 500 and result.get('common_step_counter') == 12000,
                 'producer cell identity/clock differs')
            expected = dict(status='COMPLETE',index=index,manifest_sha256=a.PRODUCER_SHA,artifacts=result)
            need(canonical(completion) == canonical(expected), 'train-complete differs from recomputed cell seal')
            results[index] = result
    need(set(results) == set(range(30)), 'thirty unique completed producer cells required')
    return results


def resolve_case_references(by_cell):
    """Deterministic mapping only, with no claimed/authorized/success status."""
    need(type(by_cell) is dict and set(by_cell) == set(range(30)), 'thirty completed cells required')
    result = []
    for descriptor in a.expected_cases():
        label = descriptor['checkpoint_label']
        refs = by_cell[descriptor['training_cell_index']]['checkpoints'][str(label)]
        for key, expected_path in (('checkpoint',descriptor['checkpoint_path']),('live',descriptor['checkpoint_live_receipt'])):
            ref = refs[key]
            need(type(ref) is dict and set(ref) == {'path','sha256','size'}
                 and ref['path'] == expected_path and type(ref['size']) is int and 0 < ref['size'] <= 128*a.MIB
                 and type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']),
                 'resolved checkpoint/live reference differs')
        result.append(dict(descriptor=descriptor,checkpoint=refs['checkpoint'],live=refs['live']))
    return result


def verify_package(release):
    a.validate_release(release)
    root = Path(release['package_root'])
    need(root.is_dir() and not any(p.is_symlink() for p in (root,*root.parents)), 'regular package root required')
    actual = set()
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'package symlink forbidden')
        if path.is_dir():
            continue
        regular_path(path)
        actual.add(str(path.relative_to(root)))
    need(actual == set(release['package_files']), 'package file closure differs')
    for path, pin in release['package_files'].items():
        need(sha_file(root/path) == pin, 'package SHA differs: '+path)
    need(Path(__file__).resolve() == root/'producer.py', 'producer adapter import origin differs')


def verify_frozen_files(producer):
    """Read-only full inventory check reusable by already isolated parent/child."""
    validate_producer(producer)
    source = Path(producer['source_root'])
    actual = set()
    need(source.is_dir() and not any(p.is_symlink() for p in (source,*source.parents)), 'frozen source directory required')
    for path in source.rglob('*'):
        need(not path.is_symlink(), 'source symlink forbidden')
        if path.is_dir():
            continue
        regular_path(path); actual.add(str(path.relative_to(source)))
    need(actual == set(producer['source_files']), 'frozen source closure differs')
    for kind in ('source_files','runtime_files','asset_files'):
        for path, pin in producer[kind].items():
            target = source/path if kind == 'source_files' else Path(path)
            need(sha_file(target) == pin, 'frozen inventory SHA differs: '+str(target))
    for path,pin in producer['compiler_policy']['runtime_files'].items():
        need(sha_file(path) == pin, 'compiler runtime SHA differs')


def frozen_producer_modules(producer):
    """Verify every frozen source/runtime/asset before importing stdlib sealers.

    Must run in the isolated new-package process. It refuses an ambient scripts
    namespace, and never adds the editable workspace to sys.path.
    """
    verify_frozen_files(producer)
    source = Path(producer['source_root'])
    need(not any(n == 'src' or n.startswith('src.') or n == 'scripts' or n.startswith('scripts.') for n in sys.modules),
         'ambient scientific/producer modules refused before frozen import')
    sys.path.insert(0,str(source))
    admission = importlib.import_module('scripts.z1_timing_noise020.admission')
    admission.verify_inventory(admission.inventory_for(producer))
    artifacts = importlib.import_module('scripts.z1_timing_noise020.artifacts')
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path = Path(module.__file__)
            need(path.is_relative_to(source) and sha_file(path) == producer['source_files'].get(str(path.relative_to(source))),
                 'producer import escaped frozen source')
    return admission, artifacts


def collect_completed_producer(release, grant, *, now):
    """Read-only live binder; authority, quota admission and durable write stay separate.

    Returns verified deferred bindings and all references. Caller must refresh
    actual quota and authority clock before exclusively publishing them.
    """
    a.validate_grant(release,grant,now=now)
    verify_package(release)
    producer_path = Path(release['producer_manifest']['path'])
    need(sha_file(producer_path) == a.PRODUCER_SHA, 'producer manifest file changed')
    producer = validate_producer(read_json(producer_path))
    _, artifacts = frozen_producer_modules(producer)
    seals = [artifacts.seal_slot(producer,slot) for slot in range(9)]
    completions = [read_json(Path(c['output'])/'train-complete.json') for c in producer['cells']]
    finals = [read_json(Path(a.PRODUCER_ROOT)/'slots'/f'slot-{slot}'/'final.json') for slot in range(9)]
    by_cell = join_completed_producer(producer,seals,completions,finals)
    bindings = resolve_case_references(by_cell)
    for bound in bindings:
        descriptor = bound['descriptor']
        for key in ('checkpoint','live'):
            verify_ref(bound[key])
        live = read_json(bound['live']['path'])
        need(type(live.get('iteration')) is int and live['iteration'] == descriptor['checkpoint_label']
             and type(live.get('common_step_counter')) is int and live['common_step_counter'] == descriptor['expected_common_step_counter']
             and live.get('checkpoint_sha256') == bound['checkpoint']['sha256'], 'selected checkpoint live receipt differs')
        initial_path = Path(descriptor['producer_case_output'])/'initial-state.json'
        initial = read_json(initial_path)
        bound.update(initial=reference(initial_path),impact_speed_contract=initial['contract'],
                     completion=reference(descriptor['producer_cell_final']),slot_final=reference(descriptor['producer_slot_final']))
    return dict(schema='z1_thirty_evaluation_resolved023',release_sha256=digest(release),grant_sha256=digest(grant),
                producer_manifest_sha256=a.PRODUCER_SHA,source_manifest_sha256=digest(producer['source_files']),
                status='PRODUCER_VERIFIED_QUOTA_NOT_ADMITTED',cases=bindings,
                cell_seals=by_cell,slot_seals=seals)
