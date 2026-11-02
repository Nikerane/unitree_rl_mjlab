"""Pure event/reward tests: no simulator, checkpoints, or policy rollout."""
import importlib
from types import SimpleNamespace
import pytest
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from tests.test_spline_boundary_credit import setup_credit, accepted_onset


def module():
    name = 'src.tasks.hammer.mdp.spline_credit_deadline'
    assert importlib.util.find_spec(name) is not None, 'deadline treatment is missing'
    return importlib.import_module(name)


def setup(ms=50, batch=3, dt=.00025):
    m = module()
    env, head, tracker, params, old = setup_credit(0., batch=batch)
    env.physics_dt = dt
    env.cfg = SimpleNamespace(episode_length_s=20.)
    env.observation_manager = object()
    tracker.first_contact_time_s = torch.zeros(batch)
    tracker._substep_count = torch.zeros(batch, dtype=torch.int64)
    params.pop('onset_fraction')
    params['credit_deadline_ms'] = ms
    cfg = RewardTermCfg(func=m.DeadlineSplineStrikeReward, weight=1., params=params)
    reward = cfg.func(cfg, env)
    return env, head, tracker, params, reward, old


@pytest.mark.parametrize('ms', [50,100,150,200])
@pytest.mark.parametrize('dt', [.00025,.002])
def test_deadline_all_control_phases_never_pays_early_or_twice(ms, dt):
    decimation = round(.02/dt)
    env, _, t, p, r, _ = setup(ms, batch=decimation, dt=dt)
    accepted_onset(t)
    onset = torch.arange(1, decimation+1)
    t.first_contact_time_s = onset.float()*dt
    due = ((onset+round(ms/1000/dt)+decimation-1)//decimation)*decimation
    t._substep_count = due-decimation
    assert r(env, **p).eq(0).all()
    t._substep_count = due
    torch.testing.assert_close(r(env, **p)*env.step_dt, torch.ones(decimation))
    assert r(env, **p).eq(0).all()
    t._substep_count += 10*decimation
    env.reset_buf[:] = True
    assert r(env, **p).eq(0).all()


@pytest.mark.parametrize('ms', [50,100,150,200])
def test_same_step_onset_early_native_reset_and_external_async_clear(ms):
    env, _, t, p, r, _ = setup(ms)
    accepted_onset(t)
    env.reset_buf[0] = True
    torch.testing.assert_close(r(env, **p)*env.step_dt, torch.tensor([1.,0.,0.]))
    assert r(env, **p).eq(0).all()
    r.reset(torch.tensor([1]))
    t.started[1] = False
    t._substep_count[:] = round(.2/env.physics_dt)
    env.reset_buf[:] = True
    torch.testing.assert_close(r(env, **p)*env.step_dt, torch.tensor([0.,0.,1.]))
    assert r.unpaid_credit.eq(0).all()
    r.reset(None)
    t.started.zero_()
    assert r(env, **p).eq(0).all()


@pytest.mark.parametrize('kind', ['no_contact','upward','off_endpoint'])
def test_rejected_or_absent_first_event_and_later_recontact_never_pay(kind):
    env, _, t, p, r, _ = setup()
    accepted_onset(t)
    if kind=='no_contact':t.started.zero_()
    elif kind=='upward':t.v_precontact[:] = -1
    else:t.onset_head_w[:,2] = .15
    assert r(env, **p).eq(0).all()
    if kind!='no_contact':
        accepted_onset(t);t.v_precontact[:] = 1
    t._substep_count[:] = 800
    env.reset_buf[:] = True
    assert r(env, **p).eq(0).all()


def test_200_control_matches_old_d_on_admissible_native_trace():
    env, _, t, p, r, old = setup(200)
    oldp={k:v for k,v in p.items()if k!='credit_deadline_ms'};oldp['onset_fraction']=0.
    accepted_onset(t)
    for count in [80,160,400,720,800]:
        t._substep_count[:] = count
        env.reset_buf[:] = count>=800
        torch.testing.assert_close(r(env, **p),old(env, **oldp))


def test_critic_features_use_continuing_clock_pending_balance_and_do_not_mutate():
    m=module();env, _, t, p, r, _ = setup(150)
    accepted_onset(t);t._substep_count[:] = torch.tensor([80,400,560])
    assert r(env, **p).eq(0).all()
    # A shorter event-finalization clock must not freeze these observations.
    t._event_age=torch.full((3,),25)
    before=(r.unpaid_credit.clone(),r._consumed.clone(),t._substep_count.clone())
    out=torch.cat([m.credit_event_observation(env,field=f,credit_deadline_ms=150)for f in m.CREDIT_EVENT_FIELDS],-1)
    torch.testing.assert_close(out,torch.tensor([[1.,.1,1.,.7],[1.,.5,1.,.3],[1.,.7,1.,.1]]))
    for a,b in zip(before,(r.unpaid_credit,r._consumed,t._substep_count)):torch.testing.assert_close(a,b)
    out[:,2]=0
    assert r.unpaid_credit.eq(1).all()
    t._substep_count[:]=640;r(env,**p)
    assert m.credit_event_observation(env,field='credit_unpaid',credit_deadline_ms=150).eq(0).all()
    assert m.credit_event_observation(env,field='credit_time_to_payout',credit_deadline_ms=150).eq(0).all()


def test_construction_probe_only_and_live_missing_state_rejected():
    m=module();env=SimpleNamespace(num_envs=2,device='cpu')
    assert m.credit_event_observation(env,field='credit_contact_age',credit_deadline_ms=50).eq(0).all()
    env.observation_manager=object()
    with pytest.raises(RuntimeError,match='credit'):
        m.credit_event_observation(env,field='credit_contact_age',credit_deadline_ms=50)


@pytest.mark.parametrize('bad',[True,50.,0,75,None,float('nan')])
def test_rejects_unapproved_or_mistyped_deadline(bad):
    with pytest.raises(ValueError,match='deadline'):setup(bad)


def test_live_observation_and_reward_deadline_mismatch_is_rejected():
    m=module();env,_,t,p,r,_=setup(50)
    with pytest.raises(ValueError,match='deadline'):r(env,**dict(p,credit_deadline_ms=100))
    with pytest.raises(ValueError,match='deadline'):
        m.credit_event_observation(env,field='credit_contact_seen',credit_deadline_ms=100)


@pytest.mark.parametrize('kind',['accepted','upward','no_contact'])
@pytest.mark.parametrize('cause',['completion','failure','timeout'])
def test_200_control_matches_d_with_async_native_causes_and_rejection(kind,cause):
    env,_,t,p,r,old=setup(200)
    oldp={k:v for k,v in p.items()if k!='credit_deadline_ms'};oldp['onset_fraction']=0.
    accepted_onset(t)
    if kind=='upward':t.v_precontact[1]=-1
    elif kind=='no_contact':t.started[1]=False
    t._substep_count[:]=80
    torch.testing.assert_close(r(env,**p),old(env,**oldp))
    if cause=='completion':t._substep_count[0]=800
    elif cause=='timeout':env.reset_time_outs[0]=True
    else:env.reset_terminated[0]=True
    env.reset_buf[0]=True
    torch.testing.assert_close(r(env,**p),old(env,**oldp))
    r.reset(torch.tensor([0]));old.reset(torch.tensor([0]));t.started[0]=False
    t._substep_count[0]=0;env.reset_buf[0]=False
    t._substep_count[1:]=800;env.reset_buf[1:]=True
    torch.testing.assert_close(r(env,**p),old(env,**oldp))


@pytest.mark.parametrize('ms',[50,100,150,200])
def test_late_onset_float_recovery_and_observation_reset_are_exact(ms):
    env,_,t,p,r,_=setup(ms,batch=4)
    accepted_onset(t)
    onset=torch.tensor([1,40001,78001,79199])
    t.first_contact_time_s=onset.float()*env.physics_dt
    due=((onset+round(ms/1000/env.physics_dt)+79)//80)*80
    t._substep_count=due-80
    assert r(env,**p).eq(0).all()
    t._substep_count=due
    torch.testing.assert_close(r(env,**p)*env.step_dt,torch.ones(4))
    r.reset(None);t.started.zero_();t.first_contact_time_s.zero_();t._substep_count.zero_()
    for field in module().CREDIT_EVENT_FIELDS:
        assert module().credit_event_observation(env,field=field,credit_deadline_ms=ms).eq(0).all()


@pytest.mark.parametrize('ms',[50,100,150,200])
@pytest.mark.parametrize('dt',[.00025,.002])
def test_time_to_payout_includes_next_policy_boundary_for_all_onset_phases(ms,dt):
    period=round(.02/dt)
    env,_,t,p,r,_=setup(ms,batch=period,dt=dt)
    accepted_onset(t)
    onset=torch.arange(1,period+1)
    t.first_contact_time_s=onset.float()*dt
    t._substep_count[:]=period
    assert r(env,**p).eq(0).all()
    due=((onset+round(ms/1000/dt)+period-1)//period)*period
    want=(due-period).float()/round(.2/dt)
    got=module().credit_event_observation(env,field='credit_time_to_payout',credit_deadline_ms=ms).squeeze(1)
    torch.testing.assert_close(got,want)
    t._substep_count=due;r(env,**p)
    assert module().credit_event_observation(env,field='credit_time_to_payout',credit_deadline_ms=ms).eq(0).all()


def test_actual_tracker_finalization_does_not_clear_contact_clock_or_pending_credit():
    from tests.test_first_strike_event import _tracker
    env,_,_,p,r,_=setup(200,batch=2)
    tracker,step=_tracker(batch=2,window=1,quality_instrumentation=False,physics_dt=env.physics_dt)
    env._hammer_first_strike=tracker
    # Actual tracker starts with finite zero sentinels and integer vector clock.
    for field in module().CREDIT_EVENT_FIELDS:
        assert module().credit_event_observation(env,field=field,credit_deadline_ms=200).eq(0).all()
    step(head_z=.104,depth=0.,contact_on=False)
    step(head_z=.102,depth=0.,contact_on=False)
    step(head_z=.100,depth=0.,contact_on=[True,False])
    assert tracker.finalized.tolist()==[True,False]
    assert tracker.started.tolist()==[True,False]
    assert r(env,**p).eq(0).all()
    frozen_age=tracker._event_age.clone()
    for _ in range(400):step(head_z=.1,depth=0.,contact_on=False)
    assert torch.equal(tracker._event_age,frozen_age)
    age=module().credit_event_observation(env,field='credit_contact_age',credit_deadline_ms=200)
    torch.testing.assert_close(age,torch.tensor([[.5],[0.]]))
    assert module().credit_event_observation(env,field='credit_unpaid',credit_deadline_ms=200).tolist()==[[1.],[0.]]
    tracker.reset(None);r.reset(None)
    for field in module().CREDIT_EVENT_FIELDS:
        assert module().credit_event_observation(env,field=field,credit_deadline_ms=200).eq(0).all()


@pytest.mark.parametrize('dt',[.00025,.002])
@pytest.mark.parametrize('cause',['completion','failure','timeout'])
def test_200_equivalence_uses_real_horizon_predicate_all_phases(dt,cause):
    from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
    period=round(.02/dt)
    env,_,t,p,r,old=setup(200,batch=period,dt=dt)
    oldp={k:v for k,v in p.items()if k!='credit_deadline_ms'};oldp['onset_fraction']=0.
    accepted_onset(t);t.v_precontact[0]=-1
    onset=torch.arange(1,period+1)+round(10./dt)
    t.first_contact_time_s=onset.float()*dt
    due=((onset+round(.2/dt)+period-1)//period)*period
    for offset in [-period,0,period]:
        t._substep_count=due+offset
        env.reset_buf=horizon_window_complete(env,.2)
        if offset<0 and cause!='completion':
            env.reset_buf[1::2]=True
            (env.reset_time_outs if cause=='timeout'else env.reset_terminated)[1::2]=True
        torch.testing.assert_close(r(env,**p),old(env,**oldp))


@pytest.mark.parametrize('mutation',['dt','onset_negative','onset_nan','onset_future','counter_dtype','counter_shape','reset_dtype','reset_shape'])
def test_malformed_clock_and_reset_inputs_fail_before_consuming_event(mutation):
    env,_,t,p,r,_=setup(50)
    accepted_onset(t)
    if mutation=='dt':env.physics_dt=.001
    elif mutation=='onset_negative':t.first_contact_time_s[0]=-1
    elif mutation=='onset_nan':t.first_contact_time_s[0]=float('nan')
    elif mutation=='onset_future':t.first_contact_time_s[0]=1
    elif mutation=='counter_dtype':t._substep_count=t._substep_count.float()
    elif mutation=='counter_shape':t._substep_count=torch.zeros(1,dtype=torch.int64)
    elif mutation=='reset_dtype':env.reset_buf=env.reset_buf.float()
    else:env.reset_buf=torch.zeros(1,dtype=torch.bool)
    with pytest.raises(ValueError):r(env,**p)
    assert not r._consumed.any() and r.unpaid_credit.eq(0).all()


def test_rejected_onset_critic_context_has_no_pending_credit_or_payout():
    env,_,t,p,r,_=setup(50)
    accepted_onset(t);t.v_precontact[:]=-1;t._substep_count[:]=80
    assert r(env,**p).eq(0).all()
    out=torch.cat([module().credit_event_observation(env,field=f,credit_deadline_ms=50)for f in module().CREDIT_EVENT_FIELDS],-1)
    torch.testing.assert_close(out,torch.tensor([[1.,.1,0.,0.]]).expand(3,-1))
