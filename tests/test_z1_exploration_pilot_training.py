"""Pure finite pilot admission tests; no simulated steps or checkpoint loads."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
from tests.test_cat_soft_hook import _ihook
from tests.test_spline_exploration_pilot_contract import pilot_env,models
from src.tasks.hammer.rl.spline_exploration_pilot import PILOT_SETTINGS


def api():
    name='scripts.z1_exploration_pilot_training'
    assert importlib.util.find_spec(name), 'separate pilot training adapter missing'
    return importlib.import_module(name)


def runner(index=0,mode='current'):
    setting=PILOT_SETTINGS[index%4];seed=(2026102902,2026102903,2026102904)[index//4]
    env=pilot_env(setting,mode);env.num_envs=1024;env.common_step_counter=0
    env.cfg.scene.num_envs=1024;env.cfg.seed=seed
    c=env._sampled_pd500;c.startup_age=torch.zeros(1024)
    for name in ('_kp','_kd'):setattr(c.gains,name,getattr(c.gains,name)[:1].expand(1024,-1).clone())
    initial=torch.zeros(1024,43)
    env.obs_buf={'actor':initial,'critic':torch.cat((initial,torch.zeros(1024,4)),-1) if mode=='event' else initial.clone()}
    env.metrics_manager.cfg={'cat_soft':SimpleNamespace(func=_ihook(imp_max_p=.2,use_vel=True))}
    a,c=models(setting,mode)
    cfg={'actor':{'hidden_dims':[16,8],'activation':'elu','obs_normalization':True,'distribution_cfg':{'class_name':'src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian','init_std':1.,'std_type':'scalar'}},'critic':{'hidden_dims':[16,8],'activation':'elu','obs_normalization':True},'num_steps_per_env':24,'seed':seed}
    return SimpleNamespace(env=SimpleNamespace(unwrapped=env,clip_actions=1.),alg=SimpleNamespace(actor=a,critic=c,optimizer=SimpleNamespace(state={})),cfg=cfg,device='cpu',current_learning_iteration=0)


@pytest.fixture
def live_metadata(monkeypatch):
    import src.tasks.hammer.rl.impact_speed_contract as contract
    from src.tasks.hammer.rl.spline_exploration_pilot_contract import pilot_metadata
    def full(env,clip):
        assert clip==1.
        marker=pilot_metadata(env)
        return {'spline_exploration_pilot':marker,'credit_critic_experiment':marker['credit_critic_experiment']}
    monkeypatch.setattr(contract,'build_impact_speed_metadata',full)


def test_exact_twelve_seed_major_cells_independent_configs_and_anchor_registration():
    m=api();seen=[]
    for i in range(12):
        cell=m.pilot_cell_for_index(i);task,cfg=m.build_pilot_train_config(i)
        assert cell==dict(index=i,setting=PILOT_SETTINGS[i%4],seed=m.PILOT_SEEDS[i//4],credit_deadline_ms=200,critic_mode='current',updates=100,worlds=1024,rollout_steps=24)
        assert task.endswith('-Pilot-'+cell['setting'])
        assert cfg.sampled_pd500 and cfg.sampled_pd500_gain_mode==('fixed' if i%4==3 else 'variable')
        assert cfg.speed_bounds==(1.,1.5) and cfg.speed_cost==.8493676 and cfg.speed_mode=='policy'
        assert cfg.qualification_checkpoint is None and cfg.enable_nan_guard
        assert cfg.env.seed==cfg.agent.seed==cell['seed'] and cfg.env.scene.num_envs==1024
        assert cfg.agent.max_iterations==100 and not cfg.agent.resume and cfg.agent.save_interval==1
        assert cfg.agent.num_steps_per_env==24 and cfg.agent.algorithm.entropy_coef==.02
        assert cfg.env.terminations['nail_driven'].params=={'window_s':.2}
        seen.append(cfg)
    assert len({id(x) for x in seen})==12
    # A second request for an already registered anchor must not duplicate registration.
    m.build_pilot_train_config(0)
    task,cfg=m.build_pilot_train_config(1,credit_deadline_ms=100,critic_mode='event')
    assert 'Credit100-CriticEvent' in task and cfg.env.rewards['accepted_spline_strike'].params['credit_deadline_ms']==100


@pytest.mark.parametrize('bad',[True,-1,12,'0',0.,None])
def test_bad_indices_refused(bad):
    with pytest.raises(ValueError):api().pilot_cell_for_index(bad)


@pytest.mark.parametrize('anchor',[{'credit_deadline_ms':True},{'credit_deadline_ms':75},{'critic_mode':'bad'}])
def test_bad_anchor_refused(anchor):
    with pytest.raises(ValueError):api().pilot_cell_for_index(0,**anchor)


@pytest.mark.parametrize('mode',['current','event'])
def test_all_twelve_match_non_treatment_state_std_rng_and_cached_observations(monkeypatch,live_metadata,mode):
    m=api();rows=[];draws=[]
    def learn(r,n,init_at_random_ep_len=False):
        draws.append((torch.rand(3).tolist(),np.random.random(3).tolist(),random.random()))
        return 'trained'
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',learn)
    for i in range(12):
        r=runner(i,mode);before=deepcopy(r.cfg)
        r.cfg['actor']['distribution_cfg'].pop('class_name')
        baseline=rows[i//4*4] if i%4 else None
        with m.matched_pilot_initial_state_gate(i,critic_mode=mode,enable_trace=False,reference_initial_state=baseline) as evidence:
            assert HammerOnPolicyRunner.learn(r,100)=='trained'
        row=evidence[0];rows.append(row)
        assert row['hashes']['actor_shared_non_std']
        assert row['initial_std']==pytest.approx([.1]*6+[.3]*6+[1.] if i%4==1 else [.3]*12+[1.])
        assert row['contract']['spline_exploration_pilot']['setting']==PILOT_SETTINGS[i%4]
        before['actor']['distribution_cfg'].pop('class_name');assert r.cfg==before
        if i%4:
            for key in m.COMMON_INITIAL_HASHES:assert row['hashes'][key]==baseline['hashes'][key]
            assert draws[-1]==draws[i//4*4]
        if i>=4 and i%4==0:assert row['hashes']['actor_shared_non_std']!=rows[0]['hashes']['actor_shared_non_std']


@pytest.mark.parametrize('fault',['critic_shape','critic_alias','actor_alias','critic_keys'])
def test_atomic_reference_validation_before_either_model_mutation(monkeypatch,live_metadata,fault):
    m=api();r=runner();calls=[]
    if fault=='critic_shape':r.alg.critic.mlp[0].weight=torch.nn.Parameter(torch.zeros(15,43))
    if fault=='critic_alias':r.alg.critic.mlp[0].weight=torch.nn.Parameter(r.alg.actor.mlp[0].weight)
    if fault=='actor_alias':r.alg.actor.mlp[0].bias=torch.nn.Parameter(r.alg.actor.mlp[2].weight.flatten()[:16])
    if fault=='critic_keys':r.alg.critic.register_buffer('unknown',torch.ones(1))
    before=[deepcopy(x.state_dict()) for x in (r.alg.actor,r.alg.critic)]
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',lambda *a,**k:calls.append(True))
    with pytest.raises(ValueError):
        with m.matched_pilot_initial_state_gate(0,enable_trace=False):HammerOnPolicyRunner.learn(r,100)
    assert not calls
    for model,state in zip((r.alg.actor,r.alg.critic),before):
        for key,value in state.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)


@pytest.mark.parametrize('fault',['updates','counter','optimizer','seed','worlds','steps','velocity','impulse','disabled','setting','obs'])
def test_gate_refuses_recipe_drift_before_training(monkeypatch,live_metadata,fault):
    m=api();r=runner();calls=[];n=100
    if fault=='updates':n=99
    elif fault=='counter':r.env.unwrapped.common_step_counter=1
    elif fault=='optimizer':r.alg.optimizer.state[0]={}
    elif fault=='seed':r.cfg['seed']+=1
    elif fault=='worlds':r.env.unwrapped.num_envs=1023
    elif fault=='steps':r.cfg['num_steps_per_env']=23
    elif fault=='velocity':r.env.unwrapped.metrics_manager.cfg['cat_soft'].func._max_p=.4
    elif fault=='impulse':r.env.unwrapped.metrics_manager.cfg['cat_soft'].func._imp_max_p=.1
    elif fault=='disabled':r.env.unwrapped.metrics_manager.cfg['cat_soft'].func._use_vel=False
    elif fault=='setting':r.alg.actor=models('fixed_nominal')[0]
    else:r.env.unwrapped.obs_buf['actor']=torch.zeros(1024,42)
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',lambda *a,**k:calls.append(True))
    with pytest.raises(ValueError):
        with m.matched_pilot_initial_state_gate(0,enable_trace=False):HammerOnPolicyRunner.learn(r,n)
    assert not calls


def test_both_observers_attached_after_record_and_restored_on_failure(monkeypatch,tmp_path,live_metadata):
    m=api();events=[]
    from src.tasks.hammer.rl import credit_training_trace,velocity_population_trace
    def context(name):
        @contextmanager
        def attach(r,path,*,max_updates):
            assert (tmp_path/'initial-state.json').exists() and max_updates==100
            events.append(name+' enter')
            try:yield
            finally:events.append(name+' exit')
        return attach
    monkeypatch.setattr(credit_training_trace,'attach_credit_training_trace',context('credit'))
    monkeypatch.setattr(velocity_population_trace,'attach_velocity_population_trace',context('population'))
    def learn(*a,**k):events.append('learn');raise RuntimeError('learn failed')
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',learn)
    with pytest.raises(RuntimeError):
        with m.matched_pilot_initial_state_gate(0,tmp_path):HammerOnPolicyRunner.learn(runner(),100)
    assert events==['credit enter','population enter','learn','population exit','credit exit']
    assert HammerOnPolicyRunner.learn is learn


@pytest.mark.parametrize('fault',['hash','observation','seed','setting'])
def test_reference_mismatch_preserved_exclusively(monkeypatch,tmp_path,live_metadata,fault):
    m=api();calls=[];monkeypatch.setattr(HammerOnPolicyRunner,'learn',lambda *a,**k:calls.append(True))
    with m.matched_pilot_initial_state_gate(0,enable_trace=False) as rows:HammerOnPolicyRunner.learn(runner(),100)
    reference=deepcopy(rows[0]);calls.clear();target=runner(1)
    if fault=='hash':reference['hashes']['actor_shared_non_std']='0'*64
    elif fault=='observation':target.env.unwrapped.obs_buf['actor'][0,0]=.1
    elif fault=='seed':reference['cell']['seed']+=1
    else:reference['cell']['setting']='vic_position_low'
    with pytest.raises(ValueError):
        with m.matched_pilot_initial_state_gate(1,tmp_path,enable_trace=False,reference_initial_state=reference):HammerOnPolicyRunner.learn(target,100)
    assert not calls and not(tmp_path/'initial-state.json').exists()
    record=json.loads((tmp_path/'initial-state-mismatch.json').read_text())
    assert record['actual_initial_state']['cell']['setting']=='vic_position_low'
    with pytest.raises(FileExistsError):
        with m.matched_pilot_initial_state_gate(1,tmp_path,enable_trace=False):pass


def test_save_labels_live_counters_full_marker_and_final_dedup(monkeypatch,tmp_path,live_metadata):
    m=api();r=runner(3);calls=[];(tmp_path/'train').mkdir()
    def save(r,path,infos=None):
        assert infos['impact_speed_contract']['spline_exploration_pilot']['setting']=='fixed_nominal'
        calls.append(Path(path).name);Path(path).write_bytes(b'checkpoint'+str(r.current_learning_iteration).encode())
    monkeypatch.setattr(HammerOnPolicyRunner,'save',save)
    with m.selected_pilot_checkpoint_saves(3,tmp_path) as rows:
        for label in (0,1,24,49,99,99):
            r.current_learning_iteration=label;r.env.unwrapped.common_step_counter=(label+1)*24
            HammerOnPolicyRunner.save(r,str(tmp_path/'train'/f'model_{label}.pt'))
    assert calls==['model_0.pt','model_24.pt','model_49.pt','model_99.pt']
    assert rows[-1]['deduplicated_final_save'] and len(rows)==5


@pytest.mark.parametrize('fault',['counter','old_output','wrong_marker','different_owner','changed_final'])
def test_checkpoint_refuses_drift_and_restores(monkeypatch,tmp_path,live_metadata,fault):
    m=api();r=runner();(tmp_path/'train').mkdir();calls=[]
    def save(r,path,infos=None):calls.append(True);Path(path).write_bytes(b'save')
    monkeypatch.setattr(HammerOnPolicyRunner,'save',save)
    path=tmp_path/'train'/'model_99.pt';r.current_learning_iteration=99;r.env.unwrapped.common_step_counter=2400
    if fault=='counter':r.env.unwrapped.common_step_counter=2399
    if fault=='old_output':path.write_bytes(b'old')
    infos={'impact_speed_contract':{'wrong':True}} if fault=='wrong_marker' else None
    with pytest.raises((ValueError,FileExistsError)):
        with m.selected_pilot_checkpoint_saves(0,tmp_path):
            HammerOnPolicyRunner.save(r,str(path),infos)
            if fault=='different_owner':r=runner();r.current_learning_iteration=99;r.env.unwrapped.common_step_counter=2400
            elif fault=='changed_final':next(r.alg.actor.parameters()).data.add_(.1)
            HammerOnPolicyRunner.save(r,str(path))
    assert HammerOnPolicyRunner.save is save


@pytest.mark.parametrize('field',['hashes','initial_observations'])
def test_malformed_reference_fields_fail_with_preserved_mismatch(monkeypatch,tmp_path,live_metadata,field):
    m=api();monkeypatch.setattr(HammerOnPolicyRunner,'learn',lambda *a,**k:None)
    with m.matched_pilot_initial_state_gate(0,enable_trace=False) as rows:HammerOnPolicyRunner.learn(runner(),100)
    reference=deepcopy(rows[0]);reference[field]=[]
    with pytest.raises(ValueError,match='initial'):
        with m.matched_pilot_initial_state_gate(1,tmp_path,enable_trace=False,reference_initial_state=reference):
            HammerOnPolicyRunner.learn(runner(1),100)
    assert (tmp_path/'initial-state-mismatch.json').exists()


def test_two_world_two_update_smoke_is_local_to_pilot_adapter(monkeypatch,tmp_path,live_metadata):
    m=api();assert (m.UPDATES,m.WORLDS,m.SAVE_INDICES)==(100,1024,(0,24,49,99))
    monkeypatch.setattr(m,'UPDATES',2);monkeypatch.setattr(m,'WORLDS',2);monkeypatch.setattr(m,'SAVE_INDICES',(0,1))
    r=runner();env=r.env.unwrapped;env.num_envs=env.cfg.scene.num_envs=2
    env._sampled_pd500.startup_age=torch.zeros(2)
    env.obs_buf={k:v[:2] for k,v in env.obs_buf.items()}
    calls=[]
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',lambda *a,**k:None)
    with m.matched_pilot_initial_state_gate(0,enable_trace=False) as rows:HammerOnPolicyRunner.learn(r,2)
    assert rows[0]['cell']['worlds']==2 and rows[0]['initial_observations']['actor_shape']==[2,43]
    (tmp_path/'train').mkdir()
    def save(r,path,infos=None):calls.append(Path(path).name);Path(path).write_bytes(b'smoke')
    monkeypatch.setattr(HammerOnPolicyRunner,'save',save)
    with m.selected_pilot_checkpoint_saves(0,tmp_path) as rows:
        for label in (0,1,1):
            r.current_learning_iteration=label;env.common_step_counter=(label+1)*24
            HammerOnPolicyRunner.save(r,str(tmp_path/'train'/f'model_{label}.pt'))
    assert calls==['model_0.pt','model_1.pt'] and rows[-1]['deduplicated_final_save']


def test_once_only_gate_and_exceptions_restore_exact_learn(monkeypatch,live_metadata):
    m=api();original=lambda *a,**k:None;monkeypatch.setattr(HammerOnPolicyRunner,'learn',original)
    with pytest.raises(ValueError,match='once'):
        with m.matched_pilot_initial_state_gate(0,enable_trace=False):pass
    with pytest.raises(ValueError,match='once'):
        with m.matched_pilot_initial_state_gate(0,enable_trace=False):
            HammerOnPolicyRunner.learn(runner(),100);HammerOnPolicyRunner.learn(runner(),100)
    assert HammerOnPolicyRunner.learn is original


@pytest.mark.parametrize('enabled',[True,1,None,'yes'])
def test_traces_require_output_and_explicit_boolean(enabled):
    with pytest.raises(ValueError):
        with api().matched_pilot_initial_state_gate(0,enable_trace=enabled):pass
