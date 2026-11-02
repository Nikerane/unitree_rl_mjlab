"""Pure metadata/model/config qualification, no environment execution."""
from copy import deepcopy
import importlib
from dataclasses import asdict
from tests.test_spline_speed_variants import config_value
import pytest
from tests.test_spline_exploration_pilot_contract import pilot_env,models


def api():
    name='src.tasks.hammer.rl.spline_split_credit_contract'
    assert importlib.util.find_spec(name),'split contract missing'
    return importlib.import_module(name)


def convert_env(env,setting='vic_baseline',alpha=.5):
    from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_env_cfg
    from src.tasks.hammer.mdp.spline_split_credit import SplitSplineStrikeReward
    cfg=split_env_cfg(play=False,setting=setting,early_fraction=alpha)
    term=cfg.rewards['accepted_spline_strike'];owner=SplitSplineStrikeReward(term,env)
    env.cfg.rewards['accepted_spline_strike']=deepcopy(term)
    live=env.reward_manager.get_term_cfg('accepted_spline_strike')
    live.func=owner;live.params=deepcopy(term.params)
    del env._spline_deadline_credit
    return env


@pytest.mark.parametrize('setting',['vic_baseline','vic_position_low'])
@pytest.mark.parametrize('alpha',[.5,1.])
def test_exact_identity_models_config_difference_and_no_legacy_marker(setting,alpha):
    m=api();env=convert_env(pilot_env(setting),setting,alpha)
    marker=m.split_metadata(env)
    assert marker['early_fraction']==alpha and marker['critic_width']==43 and marker['owner'].endswith(':SplitSplineStrikeReward')
    a,c=models(setting);m.validate_split_models(a,c,{'spline_split_credit':marker})
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models,_spline_reward_profile,_observation_metadata
    validate_impact_speed_models(a,c,{'spline_split_credit':marker})
    profile=_spline_reward_profile(env);assert profile['credit_timing']['schema_version']=='spline_split_credit_v1'
    from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_env_cfg
    from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_env_cfg
    actual=asdict(split_env_cfg(play=False,setting=setting,early_fraction=alpha))
    want=asdict(pilot_env_cfg(play=False,setting=setting,credit_deadline_ms=50,critic_mode='current'))
    actual['rewards']['accepted_spline_strike']=want['rewards']['accepted_spline_strike'];assert config_value(actual)==config_value(want)


@pytest.mark.parametrize('fault',['event','fixed','entropy','width','std_frozen','bounds','mixed','extra','normalizer'])
def test_model_or_marker_drift_rejected(fault):
    m=api();env=convert_env(pilot_env());marker=m.split_metadata(env);a,c=models();metadata={'spline_split_credit':marker}
    if fault=='event':_,c=models(mode='event')
    elif fault=='fixed':a,_=models('fixed_nominal')
    elif fault=='entropy':a.distribution.gain_entropy_weight=1/6
    elif fault=='width':a.obs_dim=47
    elif fault=='std_frozen':a.distribution.std_param.requires_grad_(False)
    elif fault=='bounds':a.distribution.std_range=(.01,2.)
    elif fault=='mixed':metadata['spline_exploration_pilot']={}
    elif fault=='extra':marker['extraneous']=1
    elif fault=='normalizer':a.obs_normalizer.physical.eps=.1
    with pytest.raises(ValueError):m.validate_split_models(a,c,metadata)


@pytest.mark.parametrize('fault',['owner','alpha','width','weight','params','horizon','controller','age','gain','extra_critic'])
def test_live_drift_rejected(fault):
    m=api();env=convert_env(pilot_env());term=env.reward_manager.get_term_cfg('accepted_spline_strike')
    if fault=='owner':env._spline_split_credit=object()
    elif fault=='alpha':term.params['early_fraction']=1.
    elif fault=='width':term.params['width_m']=.02
    elif fault=='weight':term.weight=2.
    elif fault=='params':term.params['unexpected']=1
    elif fault=='horizon':env.termination_manager.get_term_cfg('nail_driven').params['window_s']=.05
    elif fault=='controller':env._sampled_pd500.gain_mode='fixed'
    elif fault=='age':env._sampled_pd500.startup_age[0]=2
    elif fault=='gain':env._sampled_pd500.arm_kp[0]+=1
    elif fault=='extra_critic':env.observation_manager.active_terms['critic'].append('extra')
    with pytest.raises(ValueError):m.split_metadata(env)


def initial_record():
    from scripts.z1_split_credit_training021 import cell_for_index
    from src.tasks.hammer.rl.impact_speed_contract import _spline_reward_profile,_observation_metadata
    from scripts.z1_exploration_pilot_training import COMMON_INITIAL_HASHES
    env=convert_env(pilot_env());marker=api().split_metadata(env)
    physical={'reference':{'path_shape':'quintic','overshoot_m':.02,'followthrough_mode':'vertical','horizontal_detour_m':0.,'fixed_route_sign':None},
        'guidance':{'weight':0.,'width_m':.05},'load':{'body_mass_kg':[1.]}}
    contract=dict(schema_version='impact_speed_v1',spline_split_credit=marker,split_physical_treatment=physical,sampled_vic_treatment=deepcopy(physical),
        reward_profile=_spline_reward_profile(env),sampled_pd=env._sampled_pd500.metadata(),
        raw_policy_clip=1.,action_dim=13,action_terms=['joint_position','joint_stiffness','strike_speed'],action_term_dims=[6,6,1],
        action_type='joint_position_variable_impedance_sampled_pd',controller_identity='sampled_vic500_v1',
        physics_dt_s=.00025,control_decimation=80,speed_bounds_mps=[1.,1.5],speed_mode='policy',
        speed_axis_world=[0.,0.,-1.],speed_cost={'coefficient':.8493676,'units':'dimensionless_per_event','reward_term_units':'cost_per_second_before_manager_dt','timeout_semantics':'reset_boundary_shaping'},
        safety={'cat_soft':{k:v for k,v in env.cfg.metrics['cat_soft'].params.items() if k!='robot_cfg'}},**_observation_metadata(env))
    return dict(cell=cell_for_index(0),phase='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT',actual_counter=0,
        critic_width=43,contract=contract,hashes={k:'a'*64 for k in (*COMMON_INITIAL_HASHES,'actor','critic')},
        initial_std=marker['initial_exploration_std'],initial_observations={'source':'existing_env_obs_buf_no_recompute',
            'actor_shape':[1024,43],'critic_shape':[1024,43],'actor_sha256':'b'*64,'critic_shared43_sha256':'b'*64,
            'scope':'saved observations only; not full physical-state identity'},
        model_initialization_seed=2026103101,collection_seed=2026103101,live_cat={'velocity_pmax':.5,'impulse_pmax':.2,'scheduler':False})


def test_initial_stdlib_reader_accepts_truthful_marker_and_full_execution_projection():
    from scripts.z1_timing_noise020 import split_contract
    row=initial_record()
    assert hasattr(split_contract,'validate_initial'),'pure initial validator missing'
    assert split_contract.validate_initial(row,row['cell'])==row['contract']['spline_split_credit']


@pytest.mark.parametrize('fault',['old','mixed','alpha','std','phase','counter','hash','seed','obs','clip','width','horizon','reward','controller','physical','safety','missing'])
def test_initial_reader_fails_closed_on_identity_or_execution_drift(fault):
    from scripts.z1_timing_noise020 import split_contract
    row=initial_record();cell=deepcopy(row['cell']);c=row['contract']
    assert hasattr(split_contract,'validate_initial'),'pure initial validator missing'
    if fault=='old':del c['spline_split_credit']
    elif fault=='mixed':c['credit_critic_experiment']={}
    elif fault=='alpha':c['spline_split_credit']['early_fraction']=1.
    elif fault=='std':row['initial_std'][0]=.7
    elif fault=='phase':row['phase']='AFTER_UPDATE'
    elif fault=='counter':row['actual_counter']=True
    elif fault=='hash':row['hashes']['actor']='bad'
    elif fault=='seed':row['collection_seed']+=1
    elif fault=='obs':c['observation_names'][-1]='credit_unpaid'
    elif fault=='clip':c['raw_policy_clip']=2.
    elif fault=='width':c['observation_widths']['critic']=47
    elif fault=='horizon':c['reward_profile']['termination']['params']['window_s']=.05
    elif fault=='reward':c['reward_profile']['terms']['accepted_spline_strike']['weight']=2.
    elif fault=='controller':c['sampled_pd']['kp'][0]+=1
    elif fault=='physical':c['split_physical_treatment']['load']['body_mass_kg'][0]+=1
    elif fault=='safety':c['safety']['cat_soft']['imp_max_p']=.1
    elif fault=='missing':del c['reward_profile']
    with pytest.raises(ValueError):split_contract.validate_initial(row,cell)


def test_legacy_reward_and_observation_metadata_remain_byte_equivalent_and_rng_unchanged():
    import json
    import torch
    from pathlib import Path
    from src.tasks.hammer.rl import impact_speed_contract as new
    path=Path('.superpowers/sdd/2026-10-30-z1-timing-noise-training020/impact_speed_contract.before-task3.py')
    namespace={'__name__':'frozen_impact_speed_contract'};exec(compile(path.read_bytes(),str(path),'exec'),namespace)
    before=torch.random.get_rng_state().clone()
    for setting in ('vic_baseline','vic_position_low'):
        for mode in ('current','event'):
            env=pilot_env(setting,mode)
            for name in ('_spline_reward_profile','_observation_metadata'):
                assert json.dumps(getattr(new,name)(env),sort_keys=True)==json.dumps(namespace[name](env),sort_keys=True)
        split=convert_env(pilot_env(setting),setting)
        assert new._observation_metadata(split)==new._observation_metadata(pilot_env(setting))
    assert torch.equal(before,torch.random.get_rng_state())


@pytest.mark.parametrize('fault',['dtype','nonfinite','critic_dtype'])
def test_model_tensor_domains_are_strict(fault):
    import torch
    m=api();a,c=models();metadata={'spline_split_credit':m.split_metadata(convert_env(pilot_env()))}
    if fault=='dtype':a.double()
    elif fault=='critic_dtype':c.double()
    else:next(a.parameters()).data.reshape(-1)[0]=float('nan')
    with pytest.raises(ValueError):m.validate_split_models(a,c,metadata)


def test_full_metadata_dispatch_matches_frozen_legacy_and_truthful_split_without_rng(monkeypatch):
    """Project fake live managers through real full builder; no plant exists."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    from src.tasks.hammer.rl import impact_speed_contract as contract,runner
    from src.tasks.hammer.mdp.impact_speed import ImpactSpeedAction,ImpactSpeedCost,ImpactSpeedState
    from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker
    path=Path('.superpowers/sdd/2026-10-30-z1-timing-noise-training020/impact_speed_contract.before-task3.py')
    frozen={'__name__':'frozen_full_contract'};exec(compile(path.read_bytes(),str(path),'exec'),frozen)
    physical=initial_record()['contract']['split_physical_treatment']
    monkeypatch.setattr(contract,'_sampled_vic_treatment_metadata',lambda env:deepcopy(physical))
    frozen['_sampled_vic_treatment_metadata']=lambda env:deepcopy(physical)
    monkeypatch.setattr(runner,'_joint_position_metadata',lambda env,path,**kw:dict(run_path=path,**kw['observation_metadata']))
    monkeypatch.setattr(runner,'_vic_controller_metadata',lambda *a,**kw:{})
    monkeypatch.setattr(runner,'_with_vic_metadata',lambda env,meta,c:dict(meta,variable_impedance={}))
    def prepare(split=False):
        env=pilot_env()
        if split:convert_env(env)
        get=env.reward_manager.get_term_cfg
        progress=deepcopy(env.cfg.rewards['spline_progress']);progress.func=progress.func(progress,env)
        cost=object.__new__(ImpactSpeedCost)
        env.reward_manager.get_term_cfg=lambda name:SimpleNamespace(func=cost,params={},weight=-.8493676) if name=='speed_goal' else progress if name=='spline_progress' else get(name)
        env.reward_manager.active_terms.append('spline_path_cost')
        # Include full real profile for configured path term.
        prior=env.reward_manager.get_term_cfg
        env.reward_manager.get_term_cfg=lambda name:env.cfg.rewards[name] if name=='spline_path_cost' else prior(name)
        tracker=object.__new__(FirstStrikeEventTracker);tracker._axis=torch.tensor([0.,0.,-1.]);env._hammer_first_strike=tracker
        speed=object.__new__(ImpactSpeedAction);speed.cfg=SimpleNamespace(bounds=(1.,1.5),mode='policy');speed.state=ImpactSpeedState(3,'cpu',(1.,1.5),'policy')
        env.action_manager=SimpleNamespace(active_terms=('joint_position','joint_stiffness','strike_speed'),action_term_dim=[6,6,1],total_action_dim=13,
            get_term=lambda name:speed if name=='strike_speed' else object())
        return env
    legacy=prepare();before=torch.random.get_rng_state().clone()
    assert json.dumps(contract.build_impact_speed_metadata(legacy,1.),sort_keys=True)==json.dumps(frozen['build_impact_speed_metadata'](legacy,1.),sort_keys=True)
    split=contract.build_impact_speed_metadata(prepare(True),1.)
    assert split['spline_split_credit']==api().split_identity('vic_baseline')
    assert not {'spline_exploration_pilot','credit_critic_experiment','spline_learning_variant','pilot_physical_treatment'}&split.keys()
    assert split['split_physical_treatment']==split['sampled_vic_treatment']
    assert torch.equal(before,torch.random.get_rng_state())


def test_file_validator_import_has_no_scientific_dependencies():
    import subprocess
    import sys
    from pathlib import Path
    code='import sys;sys.path.insert(0,'+repr(str(Path.cwd()))+');import scripts.z1_timing_noise020.split_contract as c;assert callable(c.validate_initial);assert callable(c.validate_trace_row);assert not {"torch","mujoco","numpy","mjlab"}&sys.modules.keys()'
    result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr


@pytest.fixture(autouse=True)
def forbid_environment_or_checkpoint(monkeypatch):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    def forbidden(*a,**k):raise AssertionError("physical environment/checkpoint loading forbidden in pure task3 tests")
    monkeypatch.setattr(torch,"load",forbidden)
    monkeypatch.setattr(ManagerBasedRlEnv,"__init__",forbidden)
