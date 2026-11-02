"""CPU-only distribution/configuration checks; never constructs a simulator."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict


def api():
    from src.tasks.hammer.rl import reward_exploration032
    return reward_exploration032


def actor(mode, seed=71):
    torch.manual_seed(seed)
    x=torch.zeros(4,43);x[:,-1]=1
    obs=TensorDict({'actor':x,'critic':x.clone()},batch_size=[4])
    m=api().RewardExplorationActor(obs,{'actor':['actor'],'critic':['critic']},'actor',13,
        hidden_dims=(16,16),obs_normalization=True,
        distribution_cfg={'class_name':'src.tasks.hammer.rl.reward_exploration032:RewardExplorationGaussian','mode':mode})
    return m,obs


@pytest.mark.parametrize('mode',['independent','dependent','gsde'])
def test_initial_noise_mean_and_pending_mask(mode):
    m,obs=actor(mode);baseline,_=actor('independent')
    assert torch.equal(m(obs),baseline(obs))
    for p,q in zip(m.mlp.parameters(),baseline.mlp.parameters()):assert torch.equal(p,q)
    m(obs,stochastic_output=True);d=m.distribution
    assert torch.allclose(d.std,torch.tensor([.3]*12+[1.]).expand(4,13),atol=1e-6)
    obs['actor'][:,-1]=0;m(obs,stochastic_output=True)
    a=d.mean.clone();b=a.clone();b[:,-1]+=99
    assert torch.equal(d.log_prob(a),d.log_prob(b))
    assert torch.isfinite(d.entropy).all()
    old=tuple(v.clone() for v in d.params)
    assert torch.allclose(d.kl_divergence(old,d.params),torch.zeros(4),atol=1e-6)


def test_state_std_depends_on_observation_and_learns_only_position_scale():
    m,obs=actor('dependent');obs['actor'][:,0]=torch.arange(4.)
    with torch.no_grad():m.std_head.weight[0,0]=.2
    m(obs,stochastic_output=True);d=m.distribution
    assert d.std[3,0]>d.std[0,0]
    assert torch.allclose(d.std[:,6:],torch.tensor([.3]*6+[1.]).expand(4,7))
    d.entropy.sum().backward()
    assert m.std_head.weight.grad.abs().sum()>0
    assert d.std_param.grad[6:].abs().sum()>0
    assert d.std_param.grad[:6].abs().sum()==0


def test_gsde_uses_independent_world_matrices_and_resamples_every_four_steps():
    m,obs=actor('gsde');d=m.distribution;draws=[]
    for step in range(5):
        d.begin_collection(step,4)
        m(obs,stochastic_output=True);draws.append(d.last_position_noise.clone())
        d.end_collection()
    assert torch.equal(draws[0],draws[1]) and torch.equal(draws[0],draws[3])
    assert not torch.equal(draws[0],draws[4])
    assert not torch.equal(draws[0][0],draws[0][1])
    before=d.noise_matrices.clone();m.reset(torch.tensor([False,True,False,False]))
    assert torch.equal(before[0],d.noise_matrices[0])
    assert not torch.equal(before[1],d.noise_matrices[1])
    # Minibatch probability evaluation cannot corrupt rollout noise storage.
    before=d.noise_matrices.clone();m(obs,stochastic_output=True)
    assert torch.equal(before,d.noise_matrices)


def test_gsde_marginal_variance_and_std_gradients():
    m,obs=actor('gsde');d=m.distribution
    x=obs['actor'][0:1].expand(12000,43).clone();x[:,0]=2
    obs=TensorDict({'actor':x},batch_size=[12000]);d.begin_collection(0,12000)
    m(obs,stochastic_output=True)
    noise=d.last_position_noise
    assert torch.allclose(noise.std(0),torch.full((6,),.3),atol=.012)
    d.entropy.mean().backward()
    assert d.sde_log_std.grad.abs().sum()>0
    d.end_collection()


@pytest.mark.parametrize('mode',['independent','dependent','gsde'])
def test_policy_state_roundtrip_and_likelihood(mode):
    m,obs=actor(mode);m(obs,stochastic_output=True)
    actions=m.distribution.mean+0.1;expected=m.get_output_log_prob(actions)
    restored,_=actor(mode,seed=82);restored.load_state_dict(deepcopy(m.state_dict()))
    restored(obs,stochastic_output=True)
    assert torch.equal(m(obs),restored(obs))
    assert torch.allclose(expected,restored.get_output_log_prob(actions))


def test_nine_configs_enable_exact_rewards_and_preserve_constraints():
    from scripts.z1_reward_exploration032 import build_train_config,cell_for_index
    from scripts.z1_depth_reward_training031 import build_train_config as old_config
    _,old=old_config(3)
    rows=[]
    for i in range(9):
        task,cfg=build_train_config(i);cell=cell_for_index(i);rows.append((cell['seed'],cell['mode']))
        assert cfg.env.rewards['delivered_impulse'].weight==4.
        assert cfg.env.rewards['delivered_impulse'].func is old.env.rewards['delivered_impulse'].func
        assert cfg.env.rewards['delivered_impulse'].params==old.env.rewards['delivered_impulse'].params
        assert cfg.env.rewards['delivered_impulse'].params['saturate'] is False
        assert cfg.env.rewards['native_depth_progress031'].weight==1.
        assert cfg.env.rewards['native_depth_completion031'].weight==1.
        assert cfg.env.rewards['completion'].weight==0.
        assert cfg.env.metrics['cat_soft']==old.env.metrics['cat_soft']
        assert cfg.env.metrics['substep_impulse']==old.env.metrics['substep_impulse']
        assert cfg.env.actions==old.env.actions and cfg.env.terminations==old.env.terminations
        # Terrain factories create fresh equivalent spec lambdas on each call.
        from dataclasses import asdict
        def semantic(value):
            if callable(value):
                return (value.__module__,value.__qualname__,getattr(getattr(value,'__code__',None),'co_code',None))
            if isinstance(value,dict):return {k:semantic(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [semantic(v) for v in value]
            return value
        assert semantic(asdict(cfg.env.scene))==semantic(asdict(old.env.scene))
        assert cfg.env.sim==old.env.sim
        assert cfg.agent.max_iterations==500 and cfg.agent.num_steps_per_env==24
        assert cfg.sampled_pd500_gain_mode=='variable' and cfg.enable_nan_guard
    assert len(set(rows))==9
    assert old_config(3)[1].env.rewards['delivered_impulse'].weight==0.
    assert (cfg.speed_bounds,cfg.speed_cost,cfg.speed_mode,cfg.qualification_checkpoint)==(old.speed_bounds,old.speed_cost,old.speed_mode,old.qualification_checkpoint)
    with pytest.raises(ValueError):cell_for_index(9)


def test_restore_historical_impulse_coefficient_instead_of_rescaling_event_units():
    from src.tasks.hammer.config.z1 import _free_time_quintic_env_cfg
    cfg=_free_time_quintic_env_cfg(play=False,route_mode='three_starts')
    assert cfg.rewards['delivered_impulse'].weight==4.


@pytest.mark.parametrize('mode',['independent','dependent','gsde'])
def test_metadata_dispatch_refuses_mismatched_mode_and_old_identity(mode):
    from src.tasks.hammer.rl.reward_exploration032_contract import identity,validate_identity
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models,validate_impact_speed_metadata
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
    m,obs=actor(mode)
    critic=ImpactSpeedAgeModel(obs,{'actor':['actor'],'critic':['critic']},'critic',1,
        hidden_dims=(16,16),obs_normalization=True,distribution_cfg=None)
    metadata={'schema_version':'impact_speed_v1','spline_reward_exploration032':identity(mode)}
    validate_impact_speed_models(m,critic,metadata)
    validate_impact_speed_metadata(deepcopy(metadata),metadata)
    wrong=deepcopy(metadata);wrong['spline_reward_exploration032']=identity('dependent' if mode!='dependent' else 'gsde')
    with pytest.raises(ValueError):validate_impact_speed_metadata(wrong,metadata)
    wrong=deepcopy(metadata);wrong['spline_reward_exploration032']['impulse']['weight']=1.
    with pytest.raises(ValueError):validate_identity(wrong['spline_reward_exploration032'])


def test_entry_delegates_to_existing_train_without_allocating_environment(tmp_path,monkeypatch):
    import scripts.train
    from scripts.z1_reward_exploration032 import run
    calls=[]
    monkeypatch.setattr(scripts.train,'run_train',lambda task,cfg,output:calls.append((task,cfg,output)))
    run(8,tmp_path/'fresh')
    assert len(calls)==1 and calls[0][1].agent.max_iterations==500
    assert calls[0][2]==tmp_path/'fresh/train'
    with pytest.raises(FileExistsError):run(8,tmp_path/'fresh')


def test_initialization_matches_shared_models_across_all_modes():
    from src.tasks.hammer.rl.reward_exploration032_runner import initialize_shared_models
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
    rows=[]
    for mode in api().MODES:
        a,obs=actor(mode,seed=982)
        c=ImpactSpeedAgeModel(obs,{'actor':['actor'],'critic':['critic']},'critic',1,
            hidden_dims=(16,16),obs_normalization=True,distribution_cfg=None)
        cfg={'seed':172,'actor':dict(hidden_dims=(16,16),activation='elu',obs_normalization=True),
             'critic':dict(hidden_dims=(16,16),activation='elu',obs_normalization=True)}
        runner=SimpleNamespace(cfg=cfg,device='cpu',alg=SimpleNamespace(actor=a,critic=c))
        rows.append(initialize_shared_models(runner))
    assert rows[0]==rows[1]==rows[2]


def test_opt_in_tasks_visible_to_existing_train_and_evaluation_registry():
    import subprocess,sys
    subprocess.run([sys.executable,'-B','-c',
        "import src.tasks;from mjlab.tasks.registry import list_tasks;assert 'Unitree-Z1-Hammer-RewardExploration032-gsde' in list_tasks()"],
        check=True,capture_output=True,text=True,timeout=40)
    import src.tasks
    from mjlab.tasks.registry import list_tasks,load_runner_cls
    from src.tasks.hammer.config.z1.spline_reward_exploration032_cfg import task_id
    from src.tasks.hammer.rl.reward_exploration032_runner import RewardExplorationRunner
    for mode in api().MODES:
        assert task_id(mode) in list_tasks()
        assert load_runner_cls(task_id(mode)) is RewardExplorationRunner


def test_real_manager_metadata_detects_reward_and_owner_drift():
    from tests.test_depth_reward031 import live_env
    from mjlab.managers.metrics_manager import MetricsManager
    from src.tasks.hammer.config.z1.spline_reward_exploration032_cfg import reward_env_cfg
    from src.tasks.hammer.mdp.rewards import FirstStrikeDeliveredRewardTerm
    from src.tasks.hammer.rl.reward_exploration032_contract import metadata032
    env=live_env();cfg=reward_env_cfg(play=False,mode='gsde')
    del env._z1_depth_credit031
    for key in ('native_depth_owner031','native_depth_collect031'):env.cfg.metrics[key]=cfg.metrics[key]
    env.metrics_manager=MetricsManager({k:v for k,v in env.cfg.metrics.items() if k.startswith('native_depth_')},env)
    from src.tasks.hammer.cat.hook import CatSoftHook
    hook=object.__new__(CatSoftHook)
    hook._max_p=.5;hook._imp_max_p=.2;hook._velocity_pressure=None
    hook._use_vel=True;hook._use_impulse=True
    env.metrics_manager.cfg['cat_soft']=SimpleNamespace(func=hook)
    term=deepcopy(cfg.rewards['delivered_impulse']);term.func=FirstStrikeDeliveredRewardTerm(term,env)
    old=env.reward_manager.get_term_cfg
    env.reward_manager.get_term_cfg=lambda name:term if name=='delivered_impulse' else old(name)
    assert metadata032(env)['mode']=='gsde'
    hook._max_p=.9
    with pytest.raises(ValueError):metadata032(env)
    hook._max_p=.5
    term.weight=0.
    with pytest.raises(ValueError):metadata032(env)
    term.weight=4.;env._z1_depth_credit031.ledger.completion_enabled=False
    with pytest.raises(ValueError):metadata032(env)


@pytest.mark.parametrize('mode',['independent','dependent','gsde'])
def test_existing_jit_export_keeps_same_deterministic_policy(mode):
    m,obs=actor(mode)
    exported=torch.jit.script(m.as_jit())
    assert torch.allclose(m(obs),exported(obs['actor']))


def test_algorithm_resamples_only_during_collection_without_touching_cat_semantics():
    from src.tasks.hammer.rl.cat_ppo import CatPPO
    m,obs=actor('gsde')
    # A tiny probability-only fixture, no environment or learner update.
    alg=object.__new__(api().RewardExplorationPPO)
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
    alg.actor=m;alg.critic=ImpactSpeedAgeModel(obs,{'actor':['actor'],'critic':['critic']},'critic',1,
        hidden_dims=(16,16),obs_normalization=True,distribution_cfg=None)
    alg.collection_step=0;alg.transition=SimpleNamespace()
    alg.act(obs)
    assert alg.collection_step==1 and not m.distribution.collecting
    assert torch.isfinite(alg.transition.actions_log_prob).all()
    assert api().RewardExplorationPPO.process_env_step is CatPPO.process_env_step
    assert api().RewardExplorationPPO.compute_returns is CatPPO.compute_returns


@pytest.mark.parametrize('index',[0,3,6])
def test_existing_algorithm_factory_constructs_all_modes_without_a_simulator(index):
    from dataclasses import asdict
    from scripts.z1_reward_exploration032 import build_train_config
    from src.tasks.hammer.rl.cat_storage import CatRolloutStorage
    from src.tasks.hammer.rl.reward_exploration032_runner import initialize_shared_models
    _,cfg=build_train_config(index)
    config=asdict(cfg.agent)
    x=torch.zeros(4,43);x[:,-1]=1
    obs=TensorDict({'actor':x,'critic':x.clone()},batch_size=[4])
    # Only dimension fields are read; there is no environment/reset/step method.
    dimensions=SimpleNamespace(num_envs=4,num_actions=13)
    from mjlab.rl.runner import MjlabOnPolicyRunner
    dimensions.cfg=SimpleNamespace();dimensions.get_observations=lambda:obs
    # Exercise the real runner's optional-config cleanup, not a bypass of it.
    built=MjlabOnPolicyRunner(dimensions,config,log_dir=None,device='cpu')
    alg=built.alg
    assert isinstance(alg,api().RewardExplorationPPO)
    assert isinstance(alg.storage,CatRolloutStorage)
    row=initialize_shared_models(SimpleNamespace(cfg=config,alg=alg,device='cpu'))
    assert row['seed']==cfg.agent.seed
    actions=alg.act(obs)
    assert actions.shape==(4,13) and torch.isfinite(alg.transition.actions_log_prob).all()
    assert not alg.optimizer.state
    before=(alg.actor.distribution.noise_matrices.clone() if index==6 else None)
    # Synthetic bookkeeping only: no environment step and no optimizer update.
    alg.process_env_step(obs,torch.zeros(4),torch.tensor([False,True,False,False]),
                         {'cat_delta':torch.zeros(4),'cat_r_pos':torch.zeros(4)})
    if before is not None:
        assert torch.equal(before[0],alg.actor.distribution.noise_matrices[0])
        assert not torch.equal(before[1],alg.actor.distribution.noise_matrices[1])
    assert alg.storage.step==1 and not alg.optimizer.state



def test_collected_gsde_likelihood_matches_minibatch_recomputation():
    m,obs=actor('gsde');d=m.distribution
    d.begin_collection(0,4);actions=m(obs,stochastic_output=True).detach()
    old=m.get_output_log_prob(actions).detach().clone();d.end_collection()
    m(obs,stochastic_output=True)
    assert torch.equal(old,m.get_output_log_prob(actions))


def test_entry_exercises_existing_save_filter_at_actual_study_labels(tmp_path,monkeypatch):
    import scripts.train
    from src.tasks.hammer.rl import impact_speed_contract
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
    from src.tasks.hammer.rl.reward_exploration032_contract import identity
    from scripts.z1_reward_exploration032 import run
    m,obs=actor('independent')
    critic=ImpactSpeedAgeModel(obs,{'actor':['actor'],'critic':['critic']},'critic',1,
        hidden_dims=(16,16),obs_normalization=True,distribution_cfg=None)
    metadata={'schema_version':'impact_speed_v1','spline_reward_exploration032':identity('independent')}
    monkeypatch.setattr(impact_speed_contract,'build_impact_speed_metadata',lambda env,clip:deepcopy(metadata))
    calls=[]
    def save(runner,path,infos=None):
        assert infos['impact_speed_contract']==metadata
        path=__import__('pathlib').Path(path);path.parent.mkdir(exist_ok=True)
        path.write_bytes(path.name.encode());calls.append(path.name)
    monkeypatch.setattr(HammerOnPolicyRunner,'save',save)
    def train(task,cfg,output):
        assert cfg.agent.save_interval==1
        runner=object.__new__(HammerOnPolicyRunner)
        runner.alg=SimpleNamespace(actor=m,critic=critic)
        runner.env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0),clip_actions=1.)
        for label in (0,24,49,98,99,249,499,499):
            runner.current_learning_iteration=label;runner.env.unwrapped.common_step_counter=(label+1)*24
            runner.save(str(output/f'model_{label}.pt'))
    monkeypatch.setattr(scripts.train,'run_train',train)
    run(0,tmp_path/'study')
    assert calls==['model_0.pt','model_24.pt','model_49.pt','model_99.pt','model_249.pt','model_499.pt']
    assert (tmp_path/'study/checkpoint-499-final-save-deduplicated.json').is_file()


def test_runner_initial_state_record_without_calling_parent_learning(tmp_path,monkeypatch):
    from src.tasks.hammer.rl.reward_exploration032_runner import RewardExplorationRunner
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
    m,obs=actor('dependent')
    c=ImpactSpeedAgeModel(obs,{'actor':['actor'],'critic':['critic']},'critic',1,
        hidden_dims=(16,16),obs_normalization=True,distribution_cfg=None)
    runner=object.__new__(RewardExplorationRunner)
    runner.cfg={'seed':17,'actor':dict(hidden_dims=(16,16),activation='elu',obs_normalization=True),
                'critic':dict(hidden_dims=(16,16),activation='elu',obs_normalization=True),'resume':False}
    runner.device='cpu';runner.current_learning_iteration=0
    runner.alg=SimpleNamespace(actor=m,critic=c,optimizer=SimpleNamespace(state={}))
    runner.logger=SimpleNamespace(log_dir=str(tmp_path))
    calls=[];monkeypatch.setattr(HammerOnPolicyRunner,'learn',lambda self,n,b:calls.append((n,b)))
    runner.learn(500)
    assert calls==[(500,False)] and (tmp_path/'initial-state032.json').is_file()
