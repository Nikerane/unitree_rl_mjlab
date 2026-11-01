"""Synthetic file/tensor-free tests; never evaluate, import policies or create seals."""
import array
import importlib.util
from pathlib import Path
import pytest
D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('measurement_reader',D/'measurement_reader.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)


def native_miss(raw=None):
    ep=dict(complete=True,steps=1000,hit=False,timeout=True)
    obs=dict(active=False,whole_attempt={'complete':True},accepted_contact_sample=None,first_raw_contact_sample=raw)
    boundary={'native':dict(latched=True,observed_samples=80000,termination_terms={'time_out':True,'nail_driven':False})}
    return ep,obs,boundary


def test_pinned_019_numeric_derivation_has_only_declared_seams():
    w=m.dependencies();assert w.CAPACITY==881 and w.WORLDS==32 and w.WINDOWS['authoritative_0_200ms_inclusive']==(0,801)
    assert w.pilot_adaptation_counts=={'per_world_route_compare':1,'raw_unaccepted_classification':1}
    assert w.native_adaptation_counts==dict(native_coverage_alternative=1,native_common_prefix_assignment=1,native_common_check_moved_to_regular_branch=1)


def test_raw_unaccepted_preserves_unknown_accepted_window_instead_of_miss():
    w=m.dependencies();args=native_miss(100)
    assert w.classify_world(*args,0)=='RAW_CONTACT_UNACCEPTED'
    assert w.classify_world(*native_miss(),0)=='NO_CONTACT_TIMEOUT'

@pytest.mark.parametrize('bad',['late_raw','bool_raw','hit_without_accepted','positive_dense','short_timeout','incomplete'])
def test_raw_unaccepted_invalid_lifecycle_not_admitted(bad):
    w=m.dependencies();ep,obs,bound=native_miss(100);count=0
    if bad=='late_raw':obs['first_raw_contact_sample']=80001
    if bad=='bool_raw':obs['first_raw_contact_sample']=True
    if bad=='hit_without_accepted':ep['hit']=True
    if bad=='positive_dense':count=801
    if bad=='short_timeout':ep['steps']=999;bound['native']['observed_samples']=79920
    if bad=='incomplete':ep['complete']=False
    with pytest.raises(ValueError):w.classify_world(ep,obs,bound,count)


def test_literal801_last_point_risk_and_held_max_are_unchanged():
    w=m.dependencies();n=32*881*6
    t={k:{'values':array.array('d',[0.])*n}for k in ('qvel','rolling','pulse')}
    t['qvel']['values'][800*6]=4.;t['rolling']['values'][800*6]=.1;t['pulse']['values'][800*6]=.5
    prefix=w.dense_metrics(w,t,0,0,800);full=w.dense_metrics(w,t,0,0,801)
    assert prefix['qvel']['exceedance_samples']==[0]*6 and full['qvel']['exceedance_samples']==[1,0,0,0,0,0]
    assert full['rolling']['peak'][0]==.1 and full['pulse']['peak'][0]==full['enforced']['peak'][0]==.5
    assert full['enforced']['exceedance_samples'][0]==1


def test_real_native_timeout_019_prefix_keeps_full_shadow_separate():
    w=m.dependencies();r=100;onset=79901
    native=dict(contact_samples=r,observed_samples=80000,termination_terms={'time_out':True,'nail_driven':False},snapshot=dict(termination_reasons='time_out',native_success_sample=None,native_success_time_s=None,miss=False,observed_samples=80000,observed_duration_s=20.,time_to_contact_s=onset*.00025))
    assert w.native_censor_coverage(native,onset,800)
    native['observed_samples']-=1
    with pytest.raises(ValueError):w.native_censor_coverage(native,onset,800)


