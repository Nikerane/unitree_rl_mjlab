"""Pure tensor ledger equivalence; no simulator construction or policy rollout."""
import importlib
from copy import deepcopy
import pytest
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from tests.test_spline_credit_deadline import setup as deadline_setup
from tests.test_spline_boundary_credit import accepted_onset


def setup(alpha=1., dt=.00025, batch=80):
    name='src.tasks.hammer.mdp.spline_split_credit'
    assert importlib.util.find_spec(name), 'split credit implementation missing'
    env,_,t,p,old,_=deadline_setup(50,batch=batch,dt=dt)
    del env._spline_deadline_credit
    params={k:v for k,v in p.items() if k!='credit_deadline_ms'}
    params.update(early_fraction=alpha,early_deadline_ms=50,reserve_deadline_ms=200)
    cls=importlib.import_module(name).SplitSplineStrikeReward
    split=cls(RewardTermCfg(func=cls,weight=1.,params=params),env)
    return env,t,p,old,params,split


@pytest.mark.parametrize('dt',[.00025,.002])
@pytest.mark.parametrize('kind',['accepted','upward','off_endpoint','off_path','missing'])
@pytest.mark.parametrize('cause',['horizon','early','timeout','simultaneous','manual','subset'])
@pytest.mark.parametrize('late',[False,True])
def test_alpha1_matches_deadline50_all_onset_phases_and_reset_causes(dt,kind,cause,late):
    period=round(.02/dt);env,t,p,old,params,split=setup(dt=dt,batch=period)
    accepted_onset(t)
    if kind=='upward':t.v_precontact[:]=-1
    if kind=='off_endpoint':t.onset_head_w[:,2]=.15
    if kind=='off_path':t.onset_head_w[:,0]=.02
    if kind=='missing':t.started[:]=False
    offset=round(19.96/dt) if late else 0
    onset=torch.arange(1,period+1)+offset;t.first_contact_time_s=onset.float()*dt
    for step in range(1,14):
        t._substep_count[:]=offset+step*period
        native=(step==2 if cause in ('early','simultaneous') else step==3 if cause=='timeout' else step==11)
        env.reset_buf[:]=native
        env.reset_time_outs[:]=native and cause in ('timeout','simultaneous')
        env.reset_terminated[:]=native and cause!='timeout'
        torch.testing.assert_close(split(env,**params),old(env,**p),rtol=0,atol=0)
        torch.testing.assert_close(split.early_credit+split.reserve_credit,old.unpaid_credit,rtol=0,atol=0)
        assert torch.equal(split._consumed,old._consumed)
        if step==1 and cause in ('manual','subset'):
            ids=None if cause=='manual' else torch.arange(0,period,2)
            split.reset(ids);old.reset(ids)
            t.started[slice(None) if ids is None else ids]=False
    assert split.reserve_credit.eq(0).all()


@pytest.mark.parametrize('dt',[.00025,.002])
def test_fractional_quantized_tranches_conserve_one_unit_all_phases(dt):
    n=round(.02/dt);env,t,_,_,p,r=setup(.5,dt,n);accepted_onset(t)
    onset=torch.arange(1,n+1);t.first_contact_time_s=onset.float()*dt
    early=((onset+round(.05/dt)+n-1)//n)*n
    reserve=((onset+round(.2/dt)+n-1)//n)*n
    total=torch.zeros(n)
    for step in range(1,14):
        t._substep_count[:]=step*n
        paid=r(env,**p)*env.step_dt;total+=paid
        expected=.5*((early==step*n).float()+(reserve==step*n).float())
        torch.testing.assert_close(paid,expected,rtol=0,atol=0)
        snap=r.last_snapshot
        for bucket,fraction in [('early',.5),('reserve',.5)]:
            torch.testing.assert_close(snap[bucket+'_before']+snap['new_accepted']*fraction,
                snap[bucket+'_paid']+snap[bucket+'_after'],rtol=0,atol=0)
    assert total.eq(1).all()


@pytest.mark.parametrize('fault',['alpha','early','reserve','dt','reset_shape','reset_dtype','step_dt','width'])
def test_invalid_input_is_rejected_before_consuming_event(fault):
    env,t,_,_,p,r=setup(.5,batch=3);accepted_onset(t)
    if fault=='alpha':p['early_fraction']=.2
    elif fault=='early':p['early_deadline_ms']=50.
    elif fault=='reserve':p['reserve_deadline_ms']=100
    elif fault=='dt':env.physics_dt=.001
    elif fault=='reset_shape':env.reset_buf=torch.zeros(2,dtype=torch.bool)
    elif fault=='reset_dtype':env.reset_buf=torch.zeros(3)
    elif fault=='step_dt':env.step_dt=float('nan')
    elif fault=='width':p['width_m']=.02
    with pytest.raises(ValueError):r(env,**p)
    assert not r._consumed.any() and r.early_credit.eq(0).all() and r.reserve_credit.eq(0).all()


def test_external_subset_reset_provenance_and_native_snapshot_survive_reset():
    env,t,_,_,p,r=setup(.5,batch=3);accepted_onset(t);r(env,**p)
    before=deepcopy(r.last_snapshot);r.reset(torch.tensor([1]));t.started[1]=False
    assert all(torch.equal(before[k],r.last_snapshot[k]) for k in before)
    env.reset_buf[0]=True;r(env,**p)
    assert r.last_snapshot['early_paid'].tolist()==[.5,0.,0.]
    assert r.last_snapshot['reserve_paid'].tolist()==[.5,0.,0.]
    assert r.last_snapshot['externally_discarded_early'].tolist()==[0.,.5,0.]
    snapshot=deepcopy(r.last_snapshot);r.reset(torch.tensor([0]))
    assert all(torch.equal(snapshot[k],r.last_snapshot[k]) for k in snapshot)
    assert r.generation.tolist()==[1,1,0]


@pytest.mark.parametrize('alpha',[.5,1.])
@pytest.mark.parametrize('kind',['upward','off_path','off_endpoint'])
def test_rejected_first_contact_cannot_become_eligible_on_recontact(alpha,kind):
    env,t,p,old,params,r=setup(alpha,batch=3);accepted_onset(t)
    if kind=='upward':t.v_precontact[:]=-1
    elif kind=='off_path':t.onset_head_w[:,0]+=.1
    else:t.onset_head_w[:,2]+=.1
    assert r(env,**params).eq(0).all()
    accepted_onset(t);t.v_precontact[:]=1.;t._substep_count[:]=1000;env.reset_buf[:]=True
    assert r(env,**params).eq(0).all()
    assert r.last_snapshot['new_accepted'].eq(0).all()


@pytest.mark.parametrize('dt',[.00025,.002])
def test_alpha1_all_late_onset_phases_flushed_at_exact20s_timeout(dt):
    n=round(.02/dt);env,t,p,old,params,r=setup(1.,dt,n);accepted_onset(t)
    onset=round(19.98/dt)+torch.arange(1,n+1);t.first_contact_time_s=onset.float()*dt
    t._substep_count[:]=round(20./dt);env.reset_buf[:]=True;env.reset_time_outs[:]=True
    assert torch.equal(r(env,**params),old(env,**p))
    assert r.last_snapshot['early_paid'].eq(1).all() and r.last_snapshot['reserve_paid'].eq(0).all()


@pytest.fixture(autouse=True)
def forbid_environment_or_checkpoint(monkeypatch):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    def forbidden(*a,**k):raise AssertionError("physical environment/checkpoint loading forbidden in pure task3 tests")
    monkeypatch.setattr(torch,"load",forbidden)
    monkeypatch.setattr(ManagerBasedRlEnv,"__init__",forbidden)
