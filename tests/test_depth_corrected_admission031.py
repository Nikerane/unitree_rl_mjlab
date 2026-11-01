"""Fixed three-failure corrected admission, using actual retained native evidence."""
import copy
from pathlib import Path
import pytest
from scripts.z1_depth_campaign031 import admission as a, files as f, lifecycle as life, manifest as m
from tests.test_depth_admission031 import EVIDENCE, FIRST, origin, failed_lineage


def test_exact_third_record_and_native_failure_semantics(tmp_path,monkeypatch):
    origin(monkeypatch,tmp_path)
    r=a.authenticate_third()
    assert len(r['failed_files'])==118 and a.THIRD_REF['size']==31088
    assert r['actual_start_counts']==dict(constructor_worlds=2,reset_worlds=11,
        native_autoreset_worlds=3,world_transitions=38,learned_forwards=0,updates=0)
    assert r['canonical_assertion_marker_present'] is True
    assert r['qualification_authority'] is False and r['child_result_present'] is False
    saved=EVIDENCE/'qualification-failure49072580'
    for rel,ref in r['failed_files'].items():
        actual=f.reference(saved/rel)
        assert {k:actual[k] for k in ('sha256','size')}==ref


def test_three_failure_clock_and_exact_reservation(tmp_path,monkeypatch):
    origin(monkeypatch,tmp_path)
    clock=life.start_clock(tmp_path/'clock.json','release','123',1793506000,host='gn52')
    assert clock['third_origin']==a.THIRD_REF
    assert clock['prior_native_qualification']==dict(job_id='49072580',start_epoch=1793505650)
    for key in ('third_origin','prior_native_qualification'):
        with pytest.raises(ValueError):a.validate_clock(dict(clock,**{key:{}}))
    spec=a.origin_spec()
    assert (spec['consumed_qualification_seconds'],spec['remaining_qualification_seconds'],spec['aggregate_qualification_seconds'])==(466,2774,3240)
    assert spec['consumed_qualification_seconds']+spec['replacement_allocation_seconds']==3106
    assert a.valid_replacement_start('123',1793505927)
    assert not a.valid_replacement_start('123',1793505926)
    assert not a.valid_replacement_start('123',1793505925)
    assert a.valid_replacement_start('123',1793509793)
    assert not a.valid_replacement_start('123',1793509794)


@pytest.mark.parametrize('fault',['gate','emptygate','cell','eval','receipt','clock','manifest','counts','nativefile','symlink','fifo'])
def test_actual_native_failed_tree_is_fixed_not_qualified(tmp_path,monkeypatch,fault):
    import os
    _,roots=failed_lineage(tmp_path,monkeypatch)
    proof=a.prove_unused()
    assert proof['prior_native_work']==dict(constructor_worlds=2,reset_worlds=11,native_autoreset_worlds=3,world_transitions=38)
    old=roots[4]
    if fault=='gate':(old/'qualification/gate-1.json').write_text('{}')
    elif fault=='emptygate':(old/'qualification/gate-1').mkdir()
    elif fault=='cell':(old/'claims/cell-0.json').write_text('{}')
    elif fault=='eval':(roots[5]/'claims').mkdir();(roots[5]/'claims/case-0.json').write_text('{}')
    elif fault=='receipt':(old/'qualification-receipt.json').write_text('{}')
    elif fault=='clock':(old/'clock.json').write_text('{}')
    elif fault=='manifest':(old/'manifest.json').write_text('{}')
    elif fault=='counts':(old/'qualification/gate-0/actual-start-counts.json').write_text('{}')
    elif fault=='nativefile':next((old/'qualification/gate-0-private').rglob('*.cpp')).write_text('changed')
    elif fault=='symlink':(old/'qualification/link').symlink_to(old/'clock.json')
    else:os.mkfifo(old/'qualification/pipe')
    with pytest.raises(ValueError):a.prove_unused()


def test_quota_reauthenticates_third_and_eight_root_sum(tmp_path,monkeypatch):
    _,roots=failed_lineage(tmp_path,monkeypatch)
    assert len(roots)==8
    assert a.scope_usage()['total_bytes']==sum(life.tree_bytes(root) for root in roots)
    monkeypatch.setattr(life.os,'getxattr',lambda _,key:str(0 if key=='ceph.dir.rbytes' else m.CAP_BYTES+3*m.GIB).encode(),raising=False)
    Path(a.THIRD_REF['path']).unlink()
    with pytest.raises(FileNotFoundError):life.current_quota()


@pytest.mark.parametrize('index',range(6))
@pytest.mark.parametrize('artifact',['model.pt','checkpoint.pt','gate-1.json','evaluation-receipt.json'])
def test_every_preserved_root_rejects_unexplained_regular_artifact(tmp_path,monkeypatch,index,artifact):
    _,roots=failed_lineage(tmp_path,monkeypatch)
    assert a.prove_unused()['training_evaluation_claims_checkpoints']==0
    (roots[index]/artifact).write_bytes(b'unexplained prior evidence')
    with pytest.raises(ValueError,match='preserved root entry'):a.prove_unused()


@pytest.mark.parametrize('index',range(6))
@pytest.mark.parametrize('fault',['unknown_directory','directory_as_file','symlink'])
def test_closed_preserved_root_entry_names_and_types(tmp_path,monkeypatch,index,fault):
    _,roots=failed_lineage(tmp_path,monkeypatch);root=roots[index]
    if fault=='unknown_directory':(root/'unexplained').mkdir()
    else:
        name='source' if index%2==0 else 'private'
        if fault=='directory_as_file':(root/name).write_bytes(b'wrong type')
        else:(root/name).symlink_to(tmp_path/'dangling')
    with pytest.raises(ValueError):a.prove_unused()


def test_exact_observed_root_classification_accepts_known_subsets(tmp_path,monkeypatch):
    _,roots=failed_lineage(tmp_path,monkeypatch)
    layout=f.read_json(EVIDENCE/'retained-root-layout031.json')['roots']
    for index,remote in enumerate(a.FIRST_ROOTS+a.SECOND_ROOTS+a.THIRD_ROOTS):
        for name,kind in layout[remote].items():
            path=roots[index]/name
            if path.exists():continue
            if kind=='directory':path.mkdir()
            else:path.write_bytes(b'known root metadata; completeness checked separately')
    assert a.prove_unused()['training_evaluation_claims_checkpoints']==0


def test_known_regular_metadata_name_requires_file(tmp_path,monkeypatch):
    _,roots=failed_lineage(tmp_path,monkeypatch)
    (roots[0]/'manifest.json').mkdir()
    with pytest.raises(ValueError,match='preserved root entry'):a.prove_unused()
