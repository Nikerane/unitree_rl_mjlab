"""Pure live-manager contract projections; no simulator or physical attempts."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from src.tasks.hammer.config.z1.spline_credit_critic_cfg import credit_critic_env_cfg
from src.tasks.hammer.rl.impact_speed_contract import (
    _observation_metadata, _spline_reward_profile, validate_impact_speed_models,
    validate_impact_speed_metadata, _OBSERVATION_WIDTHS,
)
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedLowNoiseModel, ImpactSpeedAgeModel
from src.tasks.hammer.rl.spline_credit_critic import CreditEventCritic
from tests.test_spline_credit_deadline import setup


def environment(ms=50, mode='event'):
    cfg=credit_critic_env_cfg(play=False,credit_deadline_ms=ms,critic_mode=mode)
    env,_,_,_,credit,_=setup(ms)
    env.cfg=cfg
    configs={**cfg.rewards}
    configs['accepted_spline_strike']=deepcopy(configs['accepted_spline_strike'])
    configs['accepted_spline_strike'].func=credit
    env.reward_manager=SimpleNamespace(active_terms=['r_imit','accepted_spline_strike'],get_term_cfg=lambda n:configs[n])
    env.termination_manager=SimpleNamespace(get_term_cfg=lambda n:cfg.terminations[n])
    env.metrics_manager=SimpleNamespace(active_terms=[])
    widths=[*_OBSERVATION_WIDTHS[:-2],1,1,1]
    terms={g:list(cfg.observations[g].terms)for g in ('actor','critic')}
    termdims={g:[(x,)for x in widths+([1]*4 if g=='critic'and mode=='event'else [])]for g in terms}
    env.observation_manager=SimpleNamespace(active_terms=terms,group_obs_dim={g:(sum(x[0]for x in termdims[g]),)for g in terms},group_obs_term_dim=termdims,get_term_cfg=lambda g,n:cfg.observations[g].terms[n])
    return env


def marker(ms, mode):
    import importlib
    name='src.tasks.hammer.rl.spline_credit_critic_contract'
    assert importlib.util.find_spec(name)is not None,'explicit credit critic contract is missing'
    return importlib.import_module(name).credit_critic_metadata(environment(ms,mode))


@pytest.mark.parametrize('mode',['current','event'])
@pytest.mark.parametrize('ms',[50,100,150,200])
def test_exact_new_deadline_and_critic_contract_and_reward_profile(ms,mode):
    env=environment(ms,mode);identity=marker(ms,mode)
    assert identity['credit_deadline_ms']==ms
    assert identity['critic_mode']==mode
    assert identity['actor_width']==43
    assert identity['critic_width']==(47 if mode=='event'else 43)
    obs=_observation_metadata(env)
    assert obs['observation_widths']=={'actor':43,'critic':47 if mode=='event'else 43}
    assert obs['observation_names'][-1]=='impact_speed_pending'
    if mode=='event':assert obs['critic_observation_names'][-1]=='credit_time_to_payout'
    profile=_spline_reward_profile(env)
    assert profile['credit_timing']['credit_deadline_ms']==ms
    assert profile['credit_timing']['late_safety_reversal']is False
    assert profile['credit_timing']['eligible_failure_and_timeout']=='pay_remaining_balance_once'


@pytest.mark.parametrize('mutation',['reorder','scale','noise','deadline','width','actor_extra','missing_owner'])
def test_enriched_contract_rejects_corrupt_or_mislabelled_live_inputs(mutation):
    env=environment()
    name='credit_contact_age'
    if mutation=='reorder':env.observation_manager.active_terms['critic'][-2:]=reversed(env.observation_manager.active_terms['critic'][-2:])
    elif mutation=='scale':env.cfg.observations['critic'].terms[name].scale=2.
    elif mutation=='noise':env.cfg.observations['critic'].enable_corruption=True
    elif mutation=='deadline':env.cfg.observations['critic'].terms[name].params['credit_deadline_ms']=150
    elif mutation=='width':env.observation_manager.group_obs_dim['critic']=(46,)
    elif mutation=='actor_extra':env.observation_manager.active_terms['actor'].append(name)
    else:del env._spline_deadline_credit
    with pytest.raises((ValueError,RuntimeError)):_observation_metadata(env)


def models(event=False):
    values=torch.zeros(3,43);values[:,-1]=1
    obs=TensorDict({'actor':values,'critic':torch.cat((values,torch.zeros(3,4)),-1)if event else values.clone()},batch_size=[3])
    groups={'actor':['actor'],'critic':['critic']}
    actor=ImpactSpeedLowNoiseModel(obs,groups,'actor',13,hidden_dims=(8,),distribution_cfg={'class_name':'src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedGaussian','init_std':1.,'std_type':'scalar'})
    critic=(CreditEventCritic if event else ImpactSpeedAgeModel)(obs,groups,'critic',1,hidden_dims=(8,))
    return actor,critic


def test_model_validation_requires_explicit_matching_critic_identity():
    actor,event=models(True)
    metadata={'spline_learning_variant':{'variant':'N'},'credit_critic_experiment':marker(50,'event')}
    validate_impact_speed_models(actor,event,metadata)
    with pytest.raises(ValueError):validate_impact_speed_models(actor,event,{'spline_learning_variant':{'variant':'N'}})
    with pytest.raises(ValueError):validate_impact_speed_models(actor,event,dict(metadata,credit_critic_experiment=marker(50,'current')))
    corrupt=deepcopy(metadata);corrupt['credit_critic_experiment']['actor_width']=True
    with pytest.raises(ValueError):validate_impact_speed_models(actor,event,corrupt)


def test_all_eight_contracts_reject_cross_cell_resume():
    contracts=[{'schema_version':'impact_speed_v1','credit_critic_experiment':marker(ms,mode)}for ms in [50,100,150,200]for mode in ['current','event']]
    for i,a in enumerate(contracts):
        validate_impact_speed_metadata(a,a)
        for j,b in enumerate(contracts):
            if i!=j:
                with pytest.raises(ValueError,match='metadata mismatch'):validate_impact_speed_metadata(a,b)


def test_non_spline_legacy_tasks_do_not_require_accepted_credit_term():
    from src.tasks.hammer.rl.spline_credit_critic_contract import credit_critic_metadata
    def missing(name):
        raise AssertionError('legacy non-spline task must not look up accepted strike credit')
    env=SimpleNamespace(reward_manager=SimpleNamespace(active_terms=['r_imit','speed_goal'],get_term_cfg=missing))
    assert credit_critic_metadata(env)is None


@pytest.mark.parametrize('mutation',['weight','width','parameter','owner','horizon','timeout'])
def test_reward_and_termination_identity_rejects_unapproved_treatment(mutation):
    from src.tasks.hammer.rl.spline_credit_critic_contract import credit_critic_metadata
    env=environment();cfg=env.reward_manager.get_term_cfg('accepted_spline_strike')
    if mutation=='weight':cfg.weight=.5
    elif mutation=='width':cfg.params['width_m']=.02
    elif mutation=='parameter':cfg.params['extra']=1
    elif mutation=='owner':env._spline_deadline_credit=object()
    elif mutation=='horizon':env.cfg.terminations['nail_driven'].params['window_s']=.05
    else:env.cfg.terminations['nail_driven'].time_out=True
    with pytest.raises(ValueError):credit_critic_metadata(env)


@pytest.mark.parametrize('mutation',['terms','width'])
def test_legacy_joint_metadata_retains_existing_failure_contract(mutation):
    from src.tasks.hammer.rl.runner import _joint_observation_metadata
    from src.tasks.hammer.mdp.spline_learning_variants import AGE_TAIL
    env=environment(200,'current')
    if mutation=='terms':
        env.observation_manager.active_terms['critic'][-1]='wrong_pending'
        match='actor and critic observation terms'
    else:
        env.observation_manager.group_obs_dim['critic']=(42,)
        match='actor and critic observation widths'
    with pytest.raises(ValueError,match=match):_joint_observation_metadata(env,scalar_tail=AGE_TAIL)
