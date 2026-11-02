"""Pure tensor reward/trace/storage joins. No simulator, policy or checkpoint IO."""
import importlib
import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from tests.test_credit_training_trace import setup as old_setup,Algorithm,restored
from tests.test_spline_split_credit021 import setup as ledger_setup
from tests.test_spline_boundary_credit import accepted_onset


def api():
    name='src.tasks.hammer.rl.split_credit_training_trace'
    assert importlib.util.find_spec(name),'split trace missing'
    return importlib.import_module(name)


def setup():
    r,raw,pos=old_setup();e=r.env.unwrapped
    le,t,_,_,params,owner=ledger_setup(.5,batch=3)
    e.device='cpu';e.cfg=le.cfg;e.physics_dt=le.physics_dt
    e._hammer_first_strike=t;e._strike_reference=le._strike_reference;e._spline_split_credit=owner
    del e._spline_deadline_credit
    e.reset_terminated=torch.zeros(3,dtype=torch.bool)
    e.delta=torch.zeros(3);e.r_pos=torch.zeros(3)
    e.termination_manager=SimpleNamespace(active_terms=['time_out','nail_driven'],
        get_term=lambda name:e.reset_time_outs if name=='time_out' else e.reset_terminated,
        get_term_cfg=lambda name:SimpleNamespace(time_out=name=='time_out'))
    old_read=e.hook.constraint_telemetry
    def read():
        v=old_read();d=e.delta
        v.update(delta=d,delta_velocity_per_joint=d[:,None].expand(-1,6),delta_impulse_per_joint=torch.zeros(3,6),
            winner_constraint=torch.where(d>0,1,0),winner_joint=torch.where(d>0,0,-1),
            cap_utilization_per_joint=torch.ones(3,6)*.5,raw_margin_per_joint=torch.zeros(3,6),positive_margin_per_joint=torch.zeros(3,6))
        return v
    e.hook.constraint_telemetry=read
    return r,params


def collect(r,p,count=24,*,onset_step=0,timeout_step=1,terminated_step=9,simultaneous=False):
    e,a=r.env.unwrapped,r.alg;st=a.storage;t=e._hammer_first_strike;owner=e._spline_split_credit
    for step in range(count):
        st.step=step
        if step==onset_step:
            accepted_onset(t)
            t.first_contact_time_s=t._substep_count.float()*e.physics_dt
        t._substep_count+=80;t.elapsed_since_contact_s=t._substep_count.float()*e.physics_dt
        e.reset_time_outs=torch.tensor([False,step==timeout_step,False]);e.reset_terminated=torch.tensor([step==terminated_step,False,step==terminated_step])
        if simultaneous:e.reset_terminated|=e.reset_time_outs
        e.reset_buf=e.reset_time_outs|e.reset_terminated
        paid=owner(e,**p)*e.step_dt
        e.delta=torch.tensor([.2,.1,0.]);e.r_pos=paid.clone();e.reward_buf=paid-.1
        e.reward_manager._step_reward=torch.stack((torch.full((3,),-.1),paid),-1)/e.step_dt
        st.soft_dones[step,:,0]=e.delta;st.dones[step,:,0]=e.reset_buf
        st.rewards[step,:,0]=e.reward_buf-e.delta*e.r_pos+(1-e.delta)*.99*st.values[step,:,0]*e.reset_time_outs
        a.transition.values=st.values[step];a.transition.actions=st.actions[step]
        a.transition.distribution_params=(st.actions[step],torch.ones(3,13),torch.ones(3,13))
        e.action_manager.action=st.actions[step].clamp(-1,1)
        e.metrics_manager.compute()
        if e.reset_buf.any():
            ids=e.reset_buf.nonzero().flatten();owner.reset(ids);t.started[ids]=False;t._substep_count[ids]=0
    st.step=count


def make_row(tmp_path):
    r,p=setup();path=tmp_path/'trace.jsonl'
    with api().attach_credit_training_trace(r,path):collect(r,p);r.alg.compute_returns(None)
    restored(r)
    return json.loads(path.read_text()),r


def test_fractional_collector_and_stdlib_validator_join_actual_cat_and_restore(tmp_path):
    rng=torch.random.get_rng_state().clone();row,r=make_row(tmp_path)
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    validate_trace_row(row,0)
    assert row['schema']=='z1_split_credit_training_trace_v1' and row['max_updates']==500
    assert r.alg.critic.calls==1 and r.env.unwrapped.metrics_manager.calls==24
    assert torch.equal(rng,torch.random.get_rng_state())
    first=row['steps'][0]['pre_reset']['split_credit'];assert first['early_after']==[.5,.5]
    flush=row['steps'][1]['pre_reset'];assert flush['split_credit']['early_paid']==[0.,.5]
    assert flush['split_credit']['reserve_paid']==[0.,.5]
    assert flush['accepted_credit_paid']==[0.,1.]
    assert row['steps'][2]['pre_reset']['split_credit']['early_paid']==[.5,0.]
    assert row['steps'][9]['pre_reset']['split_credit']['reserve_paid']==[.5,0.]


@pytest.mark.parametrize('fault',['old_schema','extra','updates','shape','nan','payment','component','balance','consumption',
    'native_union','timeout','discount','td','advantage','cat_reward','next_value','generation','unpaid','cause','late_payment'])
def test_reader_rejects_fractional_semantic_or_storage_mutations(tmp_path,fault):
    row,_=make_row(tmp_path)
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    pre=row['steps'][2]['pre_reset'];ledger=pre['split_credit'];stored=row['steps'][2]['stored']
    if fault=='old_schema':row['schema']='z1_credit_training_trace_v1'
    elif fault=='extra':ledger['extra']=0
    elif fault=='updates':row['max_updates']=100
    elif fault=='shape':ledger['early_paid']=[.5]
    elif fault=='nan':ledger['reserve_after'][0]=float('nan')
    elif fault=='payment':ledger['early_paid'][0]=1.
    elif fault=='component':pre['accepted_credit_paid'][0]=.7
    elif fault=='balance':ledger['early_before'][0]=0.
    elif fault=='consumption':ledger['new_accepted'][0]=1.
    elif fault=='native_union':pre['native_reset'][0]=True
    elif fault=='timeout':pre['timeout'][0]=True
    elif fault=='discount':row['gamma']=.5
    elif fault=='td':stored['td_residual'][0]+=.1
    elif fault=='advantage':stored['raw_advantage'][0]+=.1
    elif fault=='cat_reward':stored['rewards_after_cat_and_bootstrap'][0]+=.1
    elif fault=='next_value':stored['next_values'][0]+=.1
    elif fault=='generation':ledger['generation'][0]+=1
    elif fault=='unpaid':pre['unpaid_credit'][0]=1.
    elif fault=='cause':pre['native_causes']['nail_driven']['flags'][0]=True
    elif fault=='late_payment':ledger['elapsed_ticks'][0]=1
    with pytest.raises(ValueError):validate_trace_row(row,0)


@pytest.mark.parametrize('fault',['cap','partial','metrics','returns','limits','missing_snapshot'])
def test_collector_failures_restore_and_preserve_bounded_partial_evidence(tmp_path,fault):
    r,p=setup();path=tmp_path/'trace';kwargs={'max_bytes':64} if fault=='cap' else {}
    if fault=='limits':kwargs['max_updates']=100
    with pytest.raises((ValueError,RuntimeError)):
        with api().attach_credit_training_trace(r,path,**kwargs):
            if fault=='metrics':r.env.unwrapped.metrics_manager.fail=True
            if fault=='missing_snapshot':r.env.unwrapped.metrics_manager.compute()
            collect(r,p,count=1 if fault=='partial' else 24)
            if fault=='returns':r.alg.fail=True
            r.alg.compute_returns(None)
    restored(r)
    assert not path.exists() or path.read_bytes()==b''


def test_stale_snapshot_is_rejected_without_second_reward_call(tmp_path):
    r,p=setup();path=tmp_path/'stale'
    with pytest.raises(ValueError,match='reward call'):
        with api().attach_credit_training_trace(r,path):
            collect(r,p,count=1)
            r.env.unwrapped.metrics_manager.compute()
    restored(r)


def test_exact_500_rows_cap_and_final_hook_forward_count(tmp_path):
    r,p=setup();path=tmp_path/'full'
    with api().attach_credit_training_trace(r,path) as trace:
        for update in range(500):
            collect(r,p);r.alg.compute_returns(None)
        assert trace.updates==500 and trace.written<=96*1024**2
        with pytest.raises(ValueError,match='cap'):collect(r,p,count=1)
    assert r.alg.critic.calls==500
    rows=path.read_text().splitlines();assert len(rows)==500
    assert json.loads(rows[-1])['update']==499
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    previous=None
    for update,line in enumerate(rows):
        previous=validate_trace_row(json.loads(line),update,previous=previous)
    assert previous['split_credit']['call_index']==[12000,12000]
    restored(r)


@pytest.fixture(autouse=True)
def forbid_environment_or_checkpoint(monkeypatch):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    def forbidden(*a,**k):raise AssertionError("physical environment/checkpoint loading forbidden in pure task3 tests")
    monkeypatch.setattr(torch,"load",forbidden)
    monkeypatch.setattr(ManagerBasedRlEnv,"__init__",forbidden)


def test_reader_rejects_missing_native_reset_generation(tmp_path):
    row,_=make_row(tmp_path)
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    # Timeout at step1 must reset the environment before step2 collection.
    for record in row['steps'][2:]:
        ledger=record['pre_reset']['split_credit'];ledger['generation'][1]-=1
        ledger['consumed_before'][1]=True;ledger['consumed_after'][1]=True
    ledger=row['steps'][2]['pre_reset']['split_credit'];ledger['resets_since_call'][1]=0
    ledger['consumed_before'][1]=True
    ledger['consumed_after'][1]=True
    with pytest.raises(ValueError):validate_trace_row(row,0)


def test_fixture_timeout_storage_matches_actual_cat_process_env_step(monkeypatch,tmp_path):
    from src.tasks.hammer.rl.cat_ppo import CatPPO
    from rsl_rl.algorithms.ppo import PPO
    row,_=make_row(tmp_path);record=row['steps'][1];pre=record['pre_reset'];stored=record['stored']
    alg=object.__new__(CatPPO);alg.gamma=.99;alg._cat_checked=True
    alg.transition=SimpleNamespace(values=torch.tensor(stored['values']).reshape(2,1));captured=[]
    monkeypatch.setattr(PPO,'process_env_step',lambda self,obs,rewards,dones,extras:captured.append((rewards,extras)))
    alg.process_env_step(None,torch.tensor(pre['raw_reward']),torch.tensor(pre['native_reset']),
        {'cat_delta':torch.tensor(pre['delta']),'cat_r_pos':torch.tensor(pre['r_pos']),'time_outs':torch.tensor(pre['timeout'])})
    torch.testing.assert_close(captured[0][0],torch.tensor(stored['rewards_after_cat_and_bootstrap']),rtol=0,atol=0)
    assert 'time_outs' not in captured[0][1]


def make_chain(tmp_path,*,external=False,native_boundary=False):
    r,p=setup();path=tmp_path/'chain';e=r.env.unwrapped;t=e._hammer_first_strike
    with api().attach_credit_training_trace(r,path):
        collect(r,p,onset_step=21,timeout_step=23 if native_boundary else None,
                terminated_step=23 if native_boundary else None,simultaneous=native_boundary)
        r.alg.compute_returns(None)
        if external:
            e._spline_split_credit.reset(None);t.started[:]=False;t._substep_count[:]=0
        collect(r,p,onset_step=None,timeout_step=None,terminated_step=None)
        r.alg.compute_returns(None)
    restored(r)
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_reader_joins_pending_reserve_across_update_boundary_and_returns_detached_copy(tmp_path):
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    first,second=make_chain(tmp_path)
    assert first['steps'][-1]['pre_reset']['split_credit']['reserve_after']==[.5,.5]
    assert second['steps'][0]['pre_reset']['split_credit']['reserve_before']==[.5,.5]
    previous=validate_trace_row(first,0)
    assert previous==first['steps'][-1]['pre_reset']
    saved=deepcopy(previous);first['steps'][-1]['pre_reset']['split_credit']['reserve_after'][0]=0.
    assert previous==saved
    final=validate_trace_row(second,1,previous=previous)
    assert final==second['steps'][-1]['pre_reset'] and previous==saved


@pytest.mark.parametrize('fault',['erased','invented','consumed','generation','unaccounted_discard','unreset_discard','missing_previous','wrong_call_index'])
def test_boundary_carry_corruption_rejected(tmp_path,fault):
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    first,second=make_chain(tmp_path)
    previous=validate_trace_row(first,0)
    assert previous is not None,'validator must return terminal state'
    # Mutate only the supplied validated terminal-state copy; the next envelope
    # remains internally consistent, isolating the cross-envelope comparison.
    if fault=='erased':previous['split_credit']['reserve_after'][0]=0.
    elif fault=='invented':previous['split_credit']['early_after'][0]=.5
    elif fault=='consumed':previous['split_credit']['consumed_after'][0]=False
    elif fault=='generation':previous['split_credit']['generation'][0]+=1
    elif fault=='unaccounted_discard':
        second['steps'][0]['pre_reset']['split_credit']['externally_discarded_reserve'][0]=.5
    elif fault=='unreset_discard':
        previous['split_credit']['early_after'][0]=.5
        second['steps'][0]['pre_reset']['split_credit']['externally_discarded_early'][0]=.5
    elif fault=='missing_previous':previous=None
    elif fault=='wrong_call_index':previous['split_credit']['call_index'][0]-=1
    with pytest.raises(ValueError):validate_trace_row(second,1,previous=previous)


@pytest.mark.parametrize('native_boundary',[False,True])
def test_boundary_external_clear_and_native_reset_overlap_are_valid(tmp_path,native_boundary):
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    first,second=make_chain(tmp_path,external=True,native_boundary=native_boundary)
    previous=validate_trace_row(first,0)
    ledger=second['steps'][0]['pre_reset']['split_credit']
    assert ledger['externally_discarded_reserve']==([0.,0.] if native_boundary else [.5,.5])
    assert ledger['resets_since_call']==([2,2] if native_boundary else [1,1])
    validate_trace_row(second,1,previous=previous)


def test_boundary_native_reset_without_generation_advance_is_rejected(tmp_path):
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    first,second=make_chain(tmp_path,native_boundary=True)
    previous=validate_trace_row(first,0)
    # Coherently erase reset state throughout the second envelope. Only its link
    # to the validated preceding native terminal can reveal this corruption.
    for record in second['steps']:
        ledger=record['pre_reset']['split_credit']
        ledger['generation']=[0,0];ledger['resets_since_call']=[0,0]
        ledger['consumed_before']=[True,True];ledger['consumed_after']=[True,True]
    with pytest.raises(ValueError):validate_trace_row(second,1,previous=previous)


@pytest.mark.parametrize('fault',['carry','consumed','discard','previous','setup_generation'])
def test_fresh_first_record_rejects_invented_provenance(tmp_path,fault):
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    row,_=make_row(tmp_path);ledger=row['steps'][0]['pre_reset']['split_credit'];kwargs={}
    if fault=='carry':
        # Replace the genuine first acceptance by invented .5+.5 carry-in.
        ledger['new_accepted']=[0.,0.];ledger['early_before']=[.5,.5];ledger['reserve_before']=[.5,.5]
    elif fault=='consumed':
        ledger['new_accepted']=[0.,0.];ledger['consumed_before']=[True,True]
        ledger['early_before']=[.5,.5];ledger['reserve_before']=[.5,.5]
    elif fault=='discard':ledger['externally_discarded_reserve']=[.5,0.]
    elif fault=='previous':kwargs['previous']=row['steps'][-1]['pre_reset']
    elif fault=='setup_generation':ledger['generation'][0]+=1
    with pytest.raises(ValueError):validate_trace_row(row,0,**kwargs)


def test_fresh_setup_resets_and_first_policy_native_flush_are_allowed(tmp_path):
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    r,p=setup();owner=r.env.unwrapped._spline_split_credit;owner.reset(None);owner.reset(None)
    path=tmp_path/'fresh'
    with api().attach_credit_training_trace(r,path):
        collect(r,p,timeout_step=0,terminated_step=0);r.alg.compute_returns(None)
    row=json.loads(path.read_text());first=row['steps'][0]['pre_reset']['split_credit']
    assert first['generation']==[2,2] and first['resets_since_call']==[2,2]
    assert first['new_accepted']==[1.,1.] and first['early_paid']==[.5,.5] and first['reserve_paid']==[.5,.5]
    validate_trace_row(row,0)
