"""Pure pilot identity/model and legacy-dispatch checks; no physical runtime."""
from copy import deepcopy
import importlib
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from tests.test_spline_credit_critic_contract import environment
from tests.test_spline_exploration_pilot import actor
from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_env_cfg
from src.tasks.hammer.mdp.spline_learning_variants import learning_variant,learning_metadata
from src.tasks.hammer.rl.spline_exploration_pilot import PILOT_SETTINGS
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
from src.tasks.hammer.rl.spline_credit_critic import CreditEventCritic
from src.tasks.hammer.mdp.sampled_pd500 import ARM_KP,ARM_KD


def api():return importlib.import_module('src.tasks.hammer.rl.spline_exploration_pilot_contract')


def pilot_env(setting='vic_baseline',mode='current'):
    env=environment(200,mode);env.cfg=pilot_env_cfg(play=False,setting=setting,critic_mode=mode)
    env.observation_manager.get_term_cfg=lambda g,n:env.cfg.observations[g].terms[n]
    fixed=setting=='fixed_nominal'
    metadata=dict(version='fixed_pd500_v1' if fixed else 'variable_impedance_pd500_v1',kp=list(ARM_KP),kd=list(ARM_KD),pd_dt_s=.002,physics_dt_s=.00025,policy_dt_s=.02,control_ids=list(range(6)))
    if fixed:metadata['ignored_action_indices']=list(range(6,12))
    else:metadata.update(gain_mode='variable',mapping_family='author_v1_exponential',C=1.25,p_bounds=[-1.,1.])
    env._sampled_pd500=SimpleNamespace(gain_mode='fixed' if fixed else 'variable',startup_ramp_s=0.,startup_age=torch.linspace(0.,1.,env.num_envs),arm_kp=torch.tensor(ARM_KP),arm_kd=torch.tensor(ARM_KD),gains=SimpleNamespace(_kp=torch.tensor(ARM_KP).expand(env.num_envs,-1).clone(),_kd=torch.tensor(ARM_KD).expand(env.num_envs,-1).clone()),metadata=lambda:metadata)
    return env


def models(setting='vic_baseline',mode='current'):
    values=torch.zeros(3,43);values[:,-1]=1
    obs=TensorDict({'actor':values,'critic':torch.cat((values,torch.zeros(3,4)),-1) if mode=='event' else values.clone()},batch_size=[3])
    return actor(setting),(CreditEventCritic if mode=='event' else ImpactSpeedAgeModel)(obs,{'critic':['critic']},'critic',1,hidden_dims=(16,8))


@pytest.mark.parametrize('setting',PILOT_SETTINGS)
@pytest.mark.parametrize('mode',['current','event'])
def test_exact_live_identity_and_models_accept_learned_std(setting,mode):
    env=pilot_env(setting,mode);m=api();identity=m.pilot_metadata(env)
    assert identity['setting']==setting and identity['actor_width']==43
    assert identity['critic_width']==(47 if mode=='event' else 43)
    assert identity['gain_mode']==('fixed' if setting=='fixed_nominal' else 'variable')
    assert identity['gain_entropy_weight']==(1/6 if setting=='vic_gain_mean' else 1.)
    assert identity['effective_gain_entropy_weight']==(0. if setting=='fixed_nominal' else identity['gain_entropy_weight'])
    assert identity['initial_exploration_std']==([.1]*6+[.3]*6+[1.] if setting=='vic_position_low' else [.3]*12+[1.])
    a,c=models(setting,mode);a.distribution.std_param.data.fill_(.7)
    m.validate_pilot_models(a,c,{'spline_exploration_pilot':identity,'credit_critic_experiment':identity['credit_critic_experiment']})
    assert learning_variant(env.cfg) is None
    assert learning_metadata(env) is None


@pytest.mark.parametrize('bad',['mode','ramp','clock','live_marker','gain','kp','ignored'])
def test_live_metadata_refuses_misconfigured_controller_and_marker(bad):
    env=pilot_env('fixed_nominal')
    if bad=='mode':env._sampled_pd500.gain_mode='variable'
    elif bad=='ramp':env._sampled_pd500.startup_ramp_s=.1
    elif bad=='clock':env._sampled_pd500.startup_age=torch.tensor([0.,2.])
    elif bad=='live_marker':env.observation_manager.get_term_cfg=lambda *a:SimpleNamespace(func=lambda e:None,params={})
    elif bad=='gain':env._sampled_pd500.gains._kp[0,0]+=1
    elif bad=='kp':env._sampled_pd500.arm_kp[0]+=1
    else:env._sampled_pd500.metadata()['ignored_action_indices']=[]
    with pytest.raises(ValueError):api().pilot_metadata(env)


@pytest.mark.parametrize('bad',['legacy_actor','wrong_actor','critic','entropy','width','std_frozen','std_bounds','metadata','mask'])
def test_model_identity_does_not_accept_shape_only_compatibility(bad):
    m=api();identity=m.pilot_metadata(pilot_env());a,c=models()
    if bad=='legacy_actor':
        from tests.test_spline_credit_critic_contract import models as legacy
        a,_=legacy()
    elif bad=='wrong_actor':a=actor('fixed_nominal')
    elif bad=='critic':_,c=models(mode='event')
    elif bad=='entropy':a.distribution.gain_entropy_weight=1/6
    elif bad=='width':a.obs_dim=42
    elif bad=='std_frozen':a.distribution.std_param.requires_grad_(False)
    elif bad=='std_bounds':a.distribution.std_range=(.01,2.)
    elif bad=='metadata':identity['initial_exploration_std'][0]=.1
    else:a.distribution.activity_mask=torch.zeros(3,13)
    with pytest.raises(ValueError):m.validate_pilot_models(a,c,{'spline_exploration_pilot':identity})


def test_legacy_n_remains_legacy_and_malformed_pilot_cannot_fall_through():
    env=environment(200,'current');before=deepcopy(env.cfg)
    assert api().pilot_metadata(env) is None and learning_variant(env.cfg)=='N'
    assert env.cfg.observations['actor'].terms['startup_age'].func is before.observations['actor'].terms['startup_age'].func
    env=pilot_env();env.cfg.observations['critic'].terms['startup_age'].params={'setting':'bad'}
    with pytest.raises(ValueError):learning_variant(env.cfg)
