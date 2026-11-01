"""031 integration contracts on real managers and clocked tensor scenes; no physics."""
from copy import deepcopy
from dataclasses import asdict
import importlib
from types import SimpleNamespace
import pytest
import torch
from mjlab.managers.metrics_manager import MetricsManager
from mjlab.managers.reward_manager import RewardManager


def api(name='mdp.depth_reward031'):
    module='src.tasks.hammer.'+name
    assert importlib.util.find_spec(module), '031 integration implementation missing'
    return importlib.import_module(module)


def scene_env(arm='A3'):
    m=api(); cfg=api('config.z1.spline_depth_reward031_cfg').depth_env_cfg(play=False,arm=arm)
    # Real SceneEntityCfg resolution needs the named nail joint interface only.
    nail=SimpleNamespace(joint_names=['nail_slide'],num_joints=1,data=SimpleNamespace(joint_pos=torch.zeros(2,1)),
        find_joints=lambda names,preserve_order=False:([0],['nail_slide']))
    env=SimpleNamespace(num_envs=2,device='cpu',scene={'nail_block':nail},cfg=cfg,
        max_episode_length_s=2.,common_step_counter=0,physics_dt=.00025,step_dt=.02,reset_buf=torch.zeros(2,dtype=torch.bool))
    env._hammer_first_strike=SimpleNamespace(started=torch.ones(2,dtype=torch.bool),
        first_contact_time_s=torch.full((2,),.00025),_substep_count=torch.zeros(2,dtype=torch.int64))
    env.metrics_manager=MetricsManager({k:v for k,v in cfg.metrics.items() if k.startswith('native_depth_')},env)
    env.reward_manager=RewardManager({k:v for k,v in cfg.rewards.items() if k.startswith('native_depth_')},env)
    return env


def sample(env,depth):
    env.scene['nail_block'].data.joint_pos[:,0]=torch.tensor(depth)
    env._hammer_first_strike._substep_count+=1
    env.metrics_manager.compute_substep()


def boundary(env,*,terminal=False):
    env.common_step_counter+=1
    env.reset_buf[:]=terminal
    reward=env.reward_manager.compute(dt=env.step_dt)
    env.metrics_manager.compute()
    return reward


@pytest.mark.parametrize('arm,progress,completion',[('A0',False,False),('A1',True,False),('A2',False,True),('A3',True,True)])
def test_actual_factories_keep_baseline_and_install_passive_owner_in_all_arms(arm,progress,completion):
    m=api('config.z1.spline_depth_reward031_cfg')
    from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_env_cfg
    from tests.test_spline_speed_variants import config_value
    cfg=m.depth_env_cfg(play=False,arm=arm)
    assert m.depth_task_id(arm)==f'Unitree-Z1-Hammer-Spline-DepthReward031-{arm}'
    assert cfg.rewards['native_depth_progress031'].weight==float(progress)
    assert cfg.rewards['native_depth_completion031'].weight==float(completion)
    assert cfg.metrics['native_depth_owner031'].per_substep is True
    assert cfg.metrics['native_depth_collect031'].per_substep is False
    a=asdict(cfg); b=asdict(split_env_cfg(play=False,setting='vic_baseline'))
    for group in ('rewards','metrics'):
        a[group]={k:v for k,v in a[group].items() if not k.startswith('native_depth_')}
    assert config_value(a)==config_value(b)
    assert not any(k.startswith('native_depth_') for k in b['metrics'])


@pytest.mark.parametrize('arm,expected',[('A0',[0.,0.]),('A1',[1.,.5]),('A2',[1.,0.]),('A3',[2.,.5])])
@pytest.mark.parametrize('crossing_tick',[281,879,880])
def test_real_managers_capture_transient_crossing_after_legacy_freeze_and_pay_once(arm,expected,crossing_tick):
    env=scene_env(arm); rng=torch.random.get_rng_state().clone()
    # Crossing at 70.25ms, gone before next 20ms boundary; horizon overshoots
    # nominal800 because contact onset may be off-policy-grid.
    total=torch.zeros(2)
    for tick in range(1,881):
        sample(env,[.030 if tick==crossing_tick else .001,.017])
        if tick%80==0:total+=boundary(env,terminal=tick==880)
    torch.testing.assert_close(total,torch.tensor(expected),rtol=0,atol=1e-6)
    owner=env._z1_depth_credit031; snap=owner.last_snapshot
    assert snap['physics_sample_count'].tolist()==[880,880]
    assert snap['first_attainment_tick'].tolist()==[crossing_tick,-1]
    assert snap['peak_depth_m'].tolist()==pytest.approx([.030,.017])
    for read in (api().depth_completion_reward031,api().depth_progress_reward031):
        read(env);read(env)
    assert owner.last_snapshot['completion_paid'].tolist()==snap['completion_paid'].tolist()
    assert torch.equal(rng,torch.random.get_rng_state())


def test_reset_reads_new_scene_once_and_preserves_pre_reset_receipt():
    env=scene_env();sample(env,[.030,.030]);env._hammer_first_strike._substep_count[:]=801
    boundary(env,terminal=True);owner=env._z1_depth_credit031;prior=owner.last_snapshot
    env.scene['nail_block'].data.joint_pos[0,0]=.017
    env.reward_manager.reset(torch.tensor([0]));env.metrics_manager.reset(torch.tensor([0]))
    assert owner.last_snapshot['completion_paid'].tolist()==[1.,1.]
    assert owner.last_snapshot['generation'].tolist()==prior['generation'].tolist()
    # Duplicate reward reader after automatic reset must not collect new scene.
    api().depth_progress_reward031(env)
    sample(env,[.030,.030]);env._hammer_first_strike._substep_count[:]=80
    total=boundary(env)
    assert total.tolist()==pytest.approx([.5,0.])
    assert (owner.last_snapshot['generation']-prior['generation']).tolist()==[1,0]


def test_timeout_forfeits_completion_horizon_wins_simultaneous_timeout():
    env=scene_env();sample(env,[.030,.030]);env._hammer_first_strike._substep_count[:]=torch.tensor([721,801])
    assert boundary(env,terminal=True).tolist()==[1.,2.]
    snap=env._z1_depth_credit031.last_snapshot
    assert snap['completion_forfeited'].tolist()==[1.,0.]
    assert snap['completion_paid'].tolist()==[0.,1.]


def live_env(arm='A3', env=None):
    from tests.test_split_credit_contract021 import convert_env
    from tests.test_spline_exploration_pilot_contract import pilot_env
    env=convert_env(pilot_env() if env is None else env);env.device='cpu';env.common_step_counter=0
    cfg=api('config.z1.spline_depth_reward031_cfg').depth_env_cfg(play=False,arm=arm)
    nail=SimpleNamespace(joint_names=['nail_slide'],num_joints=1,
        data=SimpleNamespace(joint_pos=torch.zeros(env.num_envs,1)),
        find_joints=lambda names,preserve_order=False:([0],['nail_slide']))
    env.scene={'nail_block':nail}
    env.cfg.metrics.update({k:v for k,v in cfg.metrics.items() if k.startswith('native_depth_')})
    env.cfg.rewards.update({k:v for k,v in cfg.rewards.items() if k.startswith('native_depth_')})
    env.metrics_manager=MetricsManager({k:v for k,v in cfg.metrics.items() if k.startswith('native_depth_')},env)
    old=env.reward_manager.get_term_cfg
    env.reward_manager.get_term_cfg=lambda k:env.cfg.rewards[k] if k.startswith('native_depth_') else old(k)
    env.reward_manager.active_terms.extend(k for k,v in cfg.rewards.items() if k.startswith('native_depth_') and v.weight)
    return env


@pytest.mark.parametrize('arm',['A0','A1','A2','A3'])
def test_truthful_live_metadata_roundtrip_models_and_exact_arm_reload(arm):
    m=api('rl.depth_reward031_contract');env=live_env(arm)
    identity=m.depth_metadata031(env)
    assert identity['arm']==arm and identity['critic_width']==43
    assert identity['progress']['episode_budget_units']==1.
    assert identity['completion']['episode_budget_units']==1.
    from tests.test_spline_exploration_pilot_contract import models
    a,c=models(); metadata={'spline_depth_reward031':identity,'spline_split_credit':m.split_identity('vic_baseline')}
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models,validate_impact_speed_metadata
    validate_impact_speed_models(a,c,metadata)
    import json
    saved=json.loads(json.dumps(dict(schema_version='impact_speed_v1',**metadata)))
    validate_impact_speed_metadata(saved,saved)
    wrong=deepcopy(saved);wrong['spline_depth_reward031']['arm']='A0' if arm!='A0' else 'A3'
    with pytest.raises(ValueError):validate_impact_speed_metadata(saved,wrong)


@pytest.mark.parametrize('fault',['arm','weight','owner','collector','substep','nail','ledger','semantics','mixed'])
def test_identity_lies_rejected(fault):
    m=api('rl.depth_reward031_contract');env=live_env();identity=m.depth_metadata031(env)
    if fault=='semantics':
        identity['completion']['boundary']='timeout_flush'
        with pytest.raises(ValueError):m.validate_depth_identity031(identity)
        return
    if fault=='mixed':
        from tests.test_spline_exploration_pilot_contract import models
        a,c=models()
        with pytest.raises(ValueError):m.validate_depth_models031(a,c,{'spline_depth_reward031':identity,'spline_split_credit':m.split_identity('vic_baseline'),'spline_exploration_pilot':{}})
        return
    if fault=='arm':env.metrics_manager.cfg['native_depth_owner031'].params['arm']='A0'
    elif fault=='weight':env.cfg.rewards['native_depth_progress031'].weight=2.
    elif fault=='owner':env._z1_depth_credit031=object()
    elif fault=='collector':env.metrics_manager.cfg['native_depth_collect031'].func=lambda e:torch.zeros(3)
    elif fault=='substep':env.metrics_manager.cfg['native_depth_owner031'].per_substep=False
    elif fault=='nail':env.metrics_manager.cfg['native_depth_owner031'].params['nail_cfg'].name='robot'
    elif fault=='ledger':env._z1_depth_credit031.ledger.progress_enabled=False
    with pytest.raises(ValueError):m.depth_metadata031(env)


def test_fresh_training_cells_and_factories_do_not_reuse_old_seed_or_setting():
    m=importlib.import_module('scripts.z1_depth_reward_training031')
    for index in range(12):
        cell=m.cell_for_index(index)
        assert cell['seed']==(2026110101,2026110102,2026110103)[index//4]
        assert cell['arm']==('A0','A1','A2','A3')[index%4]
        task,cfg=m.build_train_config(index)
        assert task.endswith(cell['arm']) and 'DepthReward031' in task
        assert cfg.env.seed==cfg.agent.seed==cell['seed']
        assert cfg.agent.max_iterations==500 and cfg.agent.num_steps_per_env==24 and cfg.env.scene.num_envs==1024
        assert cfg.agent.resume is False and cfg.sampled_pd500_gain_mode=='variable'
        assert cfg.env.terminations['nail_driven'].params=={'window_s':.2}
    for invalid in (True,-1,12,0.,'0'):
        with pytest.raises(ValueError):m.cell_for_index(invalid)


def trace_setup(arm='A3'):
    from tests.test_split_credit_trace021 import setup
    r,p=setup();env=r.env.unwrapped;env.common_step_counter=0
    cfg=api('config.z1.spline_depth_reward031_cfg').depth_env_cfg(play=False,arm=arm)
    nail=SimpleNamespace(joint_names=['nail_slide'],num_joints=1,data=SimpleNamespace(joint_pos=torch.zeros(3,1)),
        find_joints=lambda names,preserve_order=False:([0],['nail_slide']))
    env.scene={'nail_block':nail};env.max_episode_length_s=2.
    depth=MetricsManager({k:v for k,v in cfg.metrics.items() if k.startswith('native_depth_')},env)
    original=env.metrics_manager.compute
    env.metrics_manager.compute=lambda:(original(),depth.compute())
    env.metrics_manager.cfg.update(depth.cfg)
    original_reset=env._spline_split_credit.reset
    def reset(ids):
        original_reset(ids);env.scene['nail_block'].data.joint_pos[ids]=0;depth.reset(ids)
    env._spline_split_credit.reset=reset
    return r,p,depth


def collect_depth_trace(r,p,depth):
    from tests.test_spline_boundary_credit import accepted_onset
    env,a=r.env.unwrapped,r.alg;st=a.storage;t=env._hammer_first_strike
    for step in range(24):
        st.step=step;env.common_step_counter+=1
        if step==0:accepted_onset(t);t.first_contact_time_s=t._substep_count.float()*env.physics_dt
        for tick in range(80):
            t._substep_count+=1
            env.scene['nail_block'].data.joint_pos[:,0]=.030 if tick==40 else .001
            depth.compute_substep()
        t.elapsed_since_contact_s=t._substep_count.float()*env.physics_dt
        env.reset_time_outs=torch.tensor([False,step==1,False])
        env.reset_terminated=torch.tensor([step==9,False,step==9]);env.reset_buf=env.reset_time_outs|env.reset_terminated
        paid=env._spline_split_credit(env,**p)*env.step_dt
        progress=api().depth_progress_reward031(env)*env.step_dt
        completion=api().depth_completion_reward031(env)*env.step_dt
        env.delta=torch.tensor([.2,.1,0.]);env.r_pos=paid+progress+completion;env.reward_buf=env.r_pos-.1
        env.reward_manager.active_terms=['negative','accepted_spline_strike','native_depth_progress031','native_depth_completion031']
        env.reward_manager._step_reward=torch.stack((torch.full((3,),-.1),paid,progress,completion),-1)/env.step_dt
        st.soft_dones[step,:,0]=env.delta;st.dones[step,:,0]=env.reset_buf
        st.rewards[step,:,0]=env.reward_buf-env.delta*env.r_pos+(1-env.delta)*.99*st.values[step,:,0]*env.reset_time_outs
        a.transition.values=st.values[step];a.transition.actions=st.actions[step]
        a.transition.distribution_params=(st.actions[step],torch.ones(3,13),torch.ones(3,13))
        env.action_manager.action=st.actions[step].clamp(-1,1)
        env.metrics_manager.compute()
        if env.reset_buf.any():
            ids=env.reset_buf.nonzero().flatten();env._spline_split_credit.reset(ids);t.started[ids]=False;t._substep_count[ids]=0
    st.step=24


def test_bounded_trace_joins_real_depth_payments_split_and_population_without_extra_forwards(tmp_path):
    m=api('rl.depth_reward031_trace');r,p,depth=trace_setup();path=tmp_path/'trace.jsonl'
    before=torch.random.get_rng_state().clone()
    with m.attach_credit_training_trace(r,path) as trace:
        collect_depth_trace(r,p,depth);r.alg.compute_returns(None)
        assert trace.updates==1
    import json
    row=json.loads(path.read_text())
    assert row['schema']=='z1_depth_reward031_training_trace_v1'
    assert row['spline_depth_reward031']['arm']=='A3'
    assert row['steps'][0]['pre_reset']['depth_credit031']['progress_paid']==[1.,1.]
    assert row['steps'][9]['pre_reset']['depth_credit031']['completion_paid']==[1.,0.]
    assert row['steps'][9]['pre_reset']['split_credit']['reserve_paid']==[.5,0.]
    assert row['depth_population031']['native_episodes']==3
    assert row['depth_population031']['attained_native_episodes']==3
    assert row['depth_population031']['completion_paid_units']==2.
    assert r.alg.critic.calls==1 and torch.equal(before,torch.random.get_rng_state())
    m.validate_depth_trace_row031(row,0)
    bad=deepcopy(row);bad['steps'][0]['pre_reset']['reward_components']['native_depth_progress031'][0]=.5
    with pytest.raises(ValueError):m.validate_depth_trace_row031(bad,0)
    bad=deepcopy(row);bad['steps'][9]['pre_reset']['depth_credit031']['completion_paid'][0]=0.
    with pytest.raises(ValueError):m.validate_depth_trace_row031(bad,0)


def training_runner(index):
    from tests.test_z1_exploration_pilot_training import runner
    from tests.test_cat_soft_hook import _ihook
    r=runner(0);m=importlib.import_module('scripts.z1_depth_reward_training031');cell=m.cell_for_index(index)
    env=live_env(cell['arm'],r.env.unwrapped)
    env.metrics_manager.cfg['cat_soft']=SimpleNamespace(func=_ihook(imp_max_p=.2,use_vel=True))
    env.cfg.seed=cell['seed'];r.cfg.update(seed=cell['seed'],resume=False,max_iterations=500)
    return r


def test_all_arms_actually_match_actor_critic_std_rng_and_saved_observations(monkeypatch):
    import random
    import numpy as np
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from src.tasks.hammer.rl import impact_speed_contract
    from src.tasks.hammer.rl.depth_reward031_contract import depth_metadata031,split_metadata
    m=importlib.import_module('scripts.z1_depth_reward_training031');rows=[];draws=[]
    # Physical metadata extraction needs a live plant. Keep the real recipe,
    # live owner/model validators and actual deterministic initialization.
    monkeypatch.setattr(impact_speed_contract,'build_impact_speed_metadata',lambda e,clip:{
        'spline_depth_reward031':depth_metadata031(e),'spline_split_credit':split_metadata(e)})
    def learn(r,n,init_at_random_ep_len=False):
        draws.append((torch.rand(3).tolist(),np.random.random(3).tolist(),random.random()))
        return 'intercepted_without_training'
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',learn)
    for index in range(12):
        r=training_runner(index)
        def forbidden(*a,**k):raise AssertionError('extra forward/observations forbidden')
        monkeypatch.setattr(r.alg.actor,'forward',forbidden);monkeypatch.setattr(r.alg.critic,'forward',forbidden)
        r.env.get_observations=r.env.unwrapped.get_observations=forbidden
        baseline=rows[index//4*4] if index%4 else None
        with m.matched_initial_state_gate(index,None,enable_trace=False,reference_initial_state=baseline) as evidence:
            assert HammerOnPolicyRunner.learn(r,500)=='intercepted_without_training'
        row=evidence[0];rows.append(row)
        assert row['initial_std']==pytest.approx([.3]*12+[1.]) and row['critic_width']==43
        if baseline:
            assert row['hashes']==baseline['hashes']
            assert row['initial_observations']==baseline['initial_observations']
            assert draws[-1]==draws[index//4*4]
        elif index:assert row['hashes']['actor']!=rows[0]['hashes']['actor']


def test_six_checkpoint_labels_use031_contract_and_exact_after_update_counter(monkeypatch,tmp_path):
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from src.tasks.hammer.rl import impact_speed_contract
    from src.tasks.hammer.rl.depth_reward031_contract import depth_metadata031,split_metadata
    from pathlib import Path
    m=importlib.import_module('scripts.z1_depth_reward_training031');r=training_runner(3);calls=[]
    monkeypatch.setattr(impact_speed_contract,'build_impact_speed_metadata',lambda e,clip:{
        'spline_depth_reward031':depth_metadata031(e),'spline_split_credit':split_metadata(e)})
    def save(r,path,infos=None):
        assert infos['impact_speed_contract']['spline_depth_reward031']['arm']=='A3'
        calls.append(Path(path).name);Path(path).write_bytes(b'small-pure-fixture')
    monkeypatch.setattr(HammerOnPolicyRunner,'save',save);(tmp_path/'train').mkdir()
    with m.selected_checkpoint_saves(3,tmp_path) as rows:
        for label in (0,1,24,49,99,100,249,498,499,499):
            r.current_learning_iteration=label;r.env.unwrapped.common_step_counter=(label+1)*24
            HammerOnPolicyRunner.save(r,str(tmp_path/'train'/f'model_{label}.pt'))
    assert calls==['model_0.pt','model_24.pt','model_49.pt','model_99.pt','model_249.pt','model_499.pt']
    assert [x['common_step_counter'] for x in rows]==[24,600,1200,2400,6000,12000,12000]
    assert rows[-1]['deduplicated_final_save'] is True


def test_saved_metadata_cannot_validate_its_own_semantic_lie():
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_metadata
    m=api('rl.depth_reward031_contract');bad=dict(schema_version='impact_speed_v1',
        spline_depth_reward031=m.depth_identity031('A3'),spline_split_credit=m.split_identity('vic_baseline'))
    bad['spline_depth_reward031']['measurement']['cadence']='policy_only'
    with pytest.raises(ValueError):validate_impact_speed_metadata(bad,bad)


def test_actual_observer_substep_has_no_host_scalar_sync_or_rng():
    from torch.utils._python_dispatch import TorchDispatchMode
    class NoScalar(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:raise AssertionError('host scalar sync')
            return func(*args,**(kwargs or {}))
    env=scene_env('A0');before=torch.random.get_rng_state().clone()
    with NoScalar():
        for _ in range(80):sample(env,[.030,.001])
    assert torch.equal(before,torch.random.get_rng_state())


def test_trace_cap_and_partial_rollout_restore_hooks_and_preserve_evidence(tmp_path):
    m=api('rl.depth_reward031_trace')
    for reason in ('cap','partial'):
        r,p,depth=trace_setup();env=r.env.unwrapped
        metric=env.metrics_manager.compute;returns=r.alg.compute_returns
        path=tmp_path/reason
        with pytest.raises(ValueError):
            with m.attach_credit_training_trace(r,path,max_bytes=64 if reason=='cap' else 96*1024**2):
                collect_depth_trace(r,p,depth)
                if reason=='cap':r.alg.compute_returns(None)
        assert env.metrics_manager.compute==metric and r.alg.compute_returns==returns
        assert path.read_bytes()==b''


def test_pre_reset_receipt_records_previous_potential_and_completion_state():
    env=scene_env();sample(env,[.017,.001]);boundary(env)
    sample(env,[.030,.001]);env._hammer_first_strike._substep_count[:]=801;boundary(env,terminal=True)
    snap=env._z1_depth_credit031.last_snapshot
    assert snap['previous_progress_potential'].tolist()==pytest.approx([.5,0.])
    assert snap['completion_paid_before'].tolist()==[False,False]
    assert snap['completion_paid_after'].tolist()==[True,False]


def test_real_small_model_checkpoint_roundtrip_preserves031_identity(tmp_path):
    from tests.test_spline_exploration_pilot_contract import models
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models,validate_impact_speed_metadata
    m=api('rl.depth_reward031_contract');a,c=models();marker=m.depth_metadata031(live_env())
    meta=dict(schema_version='impact_speed_v1',spline_depth_reward031=marker,spline_split_credit=m.split_identity('vic_baseline'))
    path=tmp_path/'small-model-only.pt';torch.save({'actor':a.state_dict(),'critic':c.state_dict(),'metadata':meta},path)
    saved=torch.load(path,weights_only=True);a2,c2=models();a2.load_state_dict(saved['actor']);c2.load_state_dict(saved['critic'])
    validate_impact_speed_models(a2,c2,saved['metadata']);validate_impact_speed_metadata(saved['metadata'],meta)
    for key,value in a.state_dict().items():assert torch.equal(value,a2.state_dict()[key])


@pytest.fixture(autouse=True)
def prohibit_physical_environment(monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    def forbidden(*a,**k):raise AssertionError('physical environment construction prohibited in031 pure tests')
    monkeypatch.setattr(ManagerBasedRlEnv,'__init__',forbidden)


def test_full_builder_adds031_only_when_present_and_preserves_frozen_baseline_bytes(monkeypatch):
    import json
    from pathlib import Path
    from tests.test_spline_exploration_pilot_contract import pilot_env
    from tests.test_split_credit_contract021 import convert_env,initial_record
    from src.tasks.hammer.rl import impact_speed_contract as contract,runner
    from src.tasks.hammer.mdp.impact_speed import ImpactSpeedAction,ImpactSpeedCost,ImpactSpeedState
    from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker
    path=Path('.superpowers/sdd/2026-11-01-z1-depth-reward031/baseline-source/impact_speed_contract.py')
    frozen={'__name__':'frozen_before031'};exec(compile(path.read_bytes(),str(path),'exec'),frozen)
    physical=initial_record()['contract']['split_physical_treatment']
    monkeypatch.setattr(contract,'_sampled_vic_treatment_metadata',lambda e:deepcopy(physical))
    frozen['_sampled_vic_treatment_metadata']=lambda e:deepcopy(physical)
    monkeypatch.setattr(runner,'_joint_position_metadata',lambda env,path,**kw:dict(run_path=path,**kw['observation_metadata']))
    monkeypatch.setattr(runner,'_vic_controller_metadata',lambda *a,**kw:{})
    monkeypatch.setattr(runner,'_with_vic_metadata',lambda env,meta,c:dict(meta,variable_impedance={}))
    def prepare(env):
        env.scene['robot']=object()
        get=env.reward_manager.get_term_cfg
        progress=deepcopy(env.cfg.rewards['spline_progress']);progress.func=progress.func(progress,env)
        cost=object.__new__(ImpactSpeedCost)
        env.reward_manager.get_term_cfg=lambda name:SimpleNamespace(func=cost,params={},weight=-.8493676) if name=='speed_goal' else progress if name=='spline_progress' else get(name)
        env.reward_manager.active_terms.append('spline_path_cost');prior=env.reward_manager.get_term_cfg
        env.reward_manager.get_term_cfg=lambda name:env.cfg.rewards[name] if name=='spline_path_cost' else prior(name)
        tracker=object.__new__(FirstStrikeEventTracker);tracker._axis=torch.tensor([0.,0.,-1.]);env._hammer_first_strike=tracker
        speed=object.__new__(ImpactSpeedAction);speed.cfg=SimpleNamespace(bounds=(1.,1.5),mode='policy');speed.state=ImpactSpeedState(3,'cpu',(1.,1.5),'policy')
        env.action_manager=SimpleNamespace(active_terms=('joint_position','joint_stiffness','strike_speed'),action_term_dim=[6,6,1],total_action_dim=13,get_term=lambda name:speed if name=='strike_speed' else object())
        return env
    before=torch.random.get_rng_state().clone()
    for env in (prepare(pilot_env()),prepare(convert_env(pilot_env()))):
        assert json.dumps(contract.build_impact_speed_metadata(env,1.),sort_keys=True)==json.dumps(frozen['build_impact_speed_metadata'](env,1.),sort_keys=True)
        assert 'spline_depth_reward031' not in contract.build_impact_speed_metadata(env,1.)
    for arm in ('A0','A1','A2','A3'):
        env=prepare(live_env(arm));actual=contract.build_impact_speed_metadata(env,1.)
        assert actual['spline_depth_reward031']['arm']==arm
        assert actual['spline_split_credit']==api('rl.depth_reward031_contract').split_identity('vic_baseline')
        assert not {'spline_exploration_pilot','credit_critic_experiment','spline_learning_variant'} & actual.keys()
    assert torch.equal(before,torch.random.get_rng_state())


def test_depth_reward_without_observer_cannot_fall_through_as_legacy_metadata():
    env=live_env();del env._z1_depth_credit031;env.metrics_manager.cfg={}
    with pytest.raises(ValueError):api('rl.depth_reward031_contract').depth_metadata031(env)


@pytest.mark.parametrize('fault',['previous_potential','paid_before','due'])
def test_trace_reader_rejects_depth_state_lies_even_with_unchanged_reward_ledger(tmp_path,fault):
    import json
    m=api('rl.depth_reward031_trace');r,p,depth=trace_setup();path=tmp_path/'state-lie'
    with m.attach_credit_training_trace(r,path):collect_depth_trace(r,p,depth);r.alg.compute_returns(None)
    row=json.loads(path.read_text())
    if fault=='previous_potential':row['steps'][0]['pre_reset']['depth_credit031']['previous_progress_potential'][0]=.5
    elif fault=='paid_before':row['steps'][9]['pre_reset']['depth_credit031']['completion_paid_before'][0]=True
    else:row['steps'][9]['pre_reset']['depth_credit031']['completion_due'][0]=0.
    with pytest.raises(ValueError):m.validate_depth_trace_row031(row,0)


def test_a0_preserves_baseline_split_rewards_actions_cat_returns_and_rng_on_clocked_fixture():
    from tests.test_split_credit_trace021 import setup,collect
    baseline,p=setup();treatment,q,depth=trace_setup('A0')
    before=torch.random.get_rng_state().clone()
    collect(baseline,p);baseline.alg.compute_returns(None)
    after_baseline=torch.random.get_rng_state().clone()
    collect_depth_trace(treatment,q,depth);treatment.alg.compute_returns(None)
    for field in ('actions','values','rewards','soft_dones','dones','returns'):
        assert torch.equal(getattr(baseline.alg.storage,field),getattr(treatment.alg.storage,field))
    assert torch.equal(before,after_baseline) and torch.equal(before,torch.random.get_rng_state())
    assert treatment.env.unwrapped._z1_depth_credit031.last_snapshot['progress_paid'].tolist()==[0.,0.,0.]


def test_checkpoint_validation_rejects_split_schema_type_lie_with031_marker():
    m=api('rl.depth_reward031_contract')
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_metadata
    bad=dict(schema_version='impact_speed_v1',spline_depth_reward031=m.depth_identity031('A0'),
        spline_split_credit=m.split_identity('vic_baseline'))
    bad['spline_split_credit']['critic_width']=43.
    with pytest.raises(ValueError):validate_impact_speed_metadata(bad,bad)


@pytest.mark.parametrize('index',[1,2,3,5,6,7,9,10,11])
def test_treatment_gate_requires_durable_a0_reference_before_entering_generic_gate(index,tmp_path):
    m=importlib.import_module('scripts.z1_depth_reward_training031')
    with pytest.raises(ValueError):m.matched_initial_state_gate(index,tmp_path,enable_trace=False)
    assert not list(tmp_path.iterdir())


def test_production_gate_constructs_real500_update_collectors_with_explicit160mib_budget(monkeypatch,tmp_path):
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from src.tasks.hammer.rl import impact_speed_contract
    from src.tasks.hammer.rl.depth_reward031_contract import depth_metadata031,split_metadata
    from src.tasks.hammer.rl.depth_reward031_trace import _DepthTrace031
    from src.tasks.hammer.rl.velocity_population_trace import _Population
    m=importlib.import_module('scripts.z1_depth_reward_training031');r=training_runner(0)
    r.alg.storage=SimpleNamespace(num_transitions_per_env=24,step=0)
    r.alg.compute_returns=lambda *a:None
    hook=r.env.unwrapped.metrics_manager.cfg['cat_soft'].func
    hook._vel_detection='substep';hook._robot_cfg.joint_names=tuple(f'joint{i}' for i in range(1,7))
    monkeypatch.setattr(impact_speed_contract,'build_impact_speed_metadata',lambda e,clip:{
        'spline_depth_reward031':depth_metadata031(e),'spline_split_credit':split_metadata(e)})
    seen=[]
    def learn(r,n,init_at_random_ep_len=False):
        population=r.env.unwrapped.metrics_manager.compute.__self__
        depth=population.saved[0][2].__self__
        assert type(population) is _Population and population.max_updates==500 and population.cap==16*1024**2
        assert type(depth) is _DepthTrace031 and depth.max_updates==500 and depth.cap==160*1024**2
        assert (tmp_path/'initial-state.json').is_file() and n==500
        seen.append(True)
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',learn)
    with m.matched_initial_state_gate(0,tmp_path,enable_trace=True):HammerOnPolicyRunner.learn(r,500)
    assert seen==[True]
    assert (tmp_path/'training-trace.jsonl').stat().st_size==0
    assert (tmp_path/'velocity-population.jsonl').is_file()


def test_160mib_covers_pessimistic500_update_serialization_with_all_production_fields(tmp_path):
    import json
    m=api('rl.depth_reward031_trace');r,p,depth=trace_setup();path=tmp_path/'bound'
    with m.attach_credit_training_trace(r,path,max_bytes=160*1024**2):
        collect_depth_trace(r,p,depth);r.alg.compute_returns(None)
    row=json.loads(path.read_text())
    cfg=api('config.z1.spline_depth_reward031_cfg').depth_env_cfg(play=False,arm='A3')
    for record in row['steps']:
        pre=record['pre_reset']
        pre['reward_components']={name:[0.,0.] for name in cfg.rewards}
        pre['native_causes']={name:{'time_out':term.time_out,'flags':[False,False]} for name,term in cfg.terminations.items()}
    # Finite Python binary64 numeric repr needs at most24 bytes. All actual
    # float32/64 leaves and bounded500-update integers fit this deliberately
    # excessive token. Preserve fixed strings, configured names and all shapes.
    def pessimistic(value):
        if type(value) in (int,float):return -1.7976931348623157e+308
        if type(value) is bool:return False
        if type(value) is list:return [pessimistic(v) for v in value]
        if type(value) is dict:return {k:pessimistic(v) for k,v in value.items()}
        return value
    upper_row_bytes=len((json.dumps(pessimistic(row),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    assert upper_row_bytes*500 < 160*1024**2
    assert upper_row_bytes*500 > 96*1024**2
    (tmp_path/'bound-proof.json').write_text(json.dumps({'upper_row_bytes':upper_row_bytes,'updates':500,'upper_total_bytes':upper_row_bytes*500}))
