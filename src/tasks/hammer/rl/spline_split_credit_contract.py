"""Truthful opt-in split owner and exact current43 pilot model contract."""
import math
import torch
from torch import nn
from scripts.z1_timing_noise020.split_contract import split_identity,validate_identity,credit_timing,same as _same,need as _need
from src.tasks.hammer.mdp.spline_split_credit import SplitSplineStrikeReward,validate_split_parameters
from src.tasks.hammer.mdp.spline_exploration_pilot import pilot_setting,pilot_startup_age
from src.tasks.hammer.mdp.sampled_pd500 import ARM_KP,ARM_KD
from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
from src.tasks.hammer.rl.spline_exploration_pilot import PILOT_ACTOR_CLASSES,PilotGaussian
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel,ImpactSpeedNormalization

validate_split_identity=validate_identity


def validate_split_reward_controller(env):
    """Validate the shared live ledger, horizon and controller without an observation-width claim."""
    if 'accepted_spline_strike' not in env.reward_manager.active_terms:return None
    cfg=env.reward_manager.get_term_cfg('accepted_spline_strike')
    if not isinstance(cfg.func,SplitSplineStrikeReward):
        _need(not hasattr(env,'_spline_split_credit'),'split owner differs from active reward')
        return None
    owner=getattr(env,'_spline_split_credit',None)
    _need(type(owner) is SplitSplineStrikeReward and cfg.func is owner,'split requires exact live owner')
    _need(not hasattr(env,'_spline_deadline_credit'),'split must not carry legacy deadline owner')
    p=cfg.params
    _need(set(p)=={'robot_cfg','width_m','early_fraction','early_deadline_ms','reserve_deadline_ms'},'unexpected split parameters')
    validate_split_parameters(p['early_fraction'],p['early_deadline_ms'],p['reserve_deadline_ms'])
    _need(cfg.weight==1. and p['width_m']==.01 and all(p[k]==getattr(owner,k) for k in ('early_fraction','early_deadline_ms','reserve_deadline_ms')),'split unit/width/timing differs')
    terminal=env.termination_manager.get_term_cfg('nail_driven')
    _need(terminal.func is horizon_window_complete and terminal.params=={'window_s':.2} and terminal.time_out is False,'split requires unchanged200ms true termination')
    manager=env.observation_manager
    setting=pilot_setting(env.cfg)
    split_identity(setting,owner.early_fraction)  # Preserve the admitted setting/fraction domain.
    controller=getattr(env,'_sampled_pd500',None)
    _need(controller is not None and controller.gain_mode=='variable' and controller.startup_ramp_s==0.,'split requires variable gains and zero ramp')
    age=controller.startup_age
    _need(isinstance(age,torch.Tensor) and age.is_floating_point() and age.shape==(env.num_envs,)
          and bool(torch.isfinite(age).all()) and bool(((age>=0)&(age<=1)).all()),'split startup age differs')
    for group in ('actor','critic'):
        term=manager.get_term_cfg(group,'startup_age')
        _need(term.func is pilot_startup_age and term.params=={'setting':setting}
              and term.noise is None and term.clip is None and term.scale is None and term.history_length==0
              and term.delay_min_lag==0 and term.delay_max_lag==0 and term.delay_hold_prob==0 and term.delay_update_period==0,'split startup observation semantics differ')
    metadata=controller.metadata()
    for name,expected in (('kp',list(ARM_KP)),('kd',list(ARM_KD)),('pd_dt_s',.002),('physics_dt_s',.00025),('policy_dt_s',.02)):
        _need(_same(metadata.get(name),expected),'split nominal controller/timing differs: '+name)
    for name,expected in (('arm_kp',ARM_KP),('arm_kd',ARM_KD)):
        value=getattr(controller,name,None)
        _need(isinstance(value,torch.Tensor) and value.shape==(6,) and torch.equal(value,value.new_tensor(expected)),'split nominal gains differ')
    _need(metadata.get('version')=='variable_impedance_pd500_v1' and metadata.get('gain_mode')=='variable'
          and metadata.get('mapping_family')=='author_v1_exponential' and metadata.get('C')==1.25
          and _same(metadata.get('p_bounds'),[-1.,1.]) and 'ignored_action_indices' not in metadata,'split variable gain mapping differs')
    return setting,owner.early_fraction


def split_metadata(env):
    shared=validate_split_reward_controller(env)
    if shared is None:return None
    manager=env.observation_manager
    _need(tuple(manager.active_terms['actor'])==tuple(manager.active_terms['critic'])
          and tuple(manager.group_obs_dim['actor'])==(43,) and tuple(manager.group_obs_dim['critic'])==(43,)
          and env.cfg.observations['critic'].enable_corruption is False,'split requires unchanged current43 observations')
    return split_identity(*shared)


def validate_split_models(actor,critic,expected_metadata):
    _need(type(expected_metadata) is dict,'split requires full metadata')
    expected=validate_identity(expected_metadata.get('spline_split_credit'))
    _need(not any(k in expected_metadata for k in ('spline_exploration_pilot','credit_critic_experiment','spline_learning_variant')),'split cannot claim mixed legacy markers')
    setting=expected['setting']
    for group,model,cls,out in (('actor',actor,PILOT_ACTOR_CLASSES[setting],13),('critic',critic,ImpactSpeedAgeModel,1)):
        _need(type(model) is cls and model.obs_dim==43 and tuple(model.obs_groups)==(group,),'split model class/observation signature differs')
        for value in model.state_dict().values():
            _need(isinstance(value,torch.Tensor) and bool(torch.isfinite(value).all())
                  and (not value.is_floating_point() or value.dtype == torch.float32),
                  'split model state requires finite float32 tensors')
        layers=[layer for layer in model.mlp if isinstance(layer,nn.Linear)]
        _need(bool(layers) and layers[0].in_features==43 and layers[-1].out_features==out,'split layer signature differs')
        normalizer=model.obs_normalizer
        _need(type(normalizer) is ImpactSpeedNormalization and normalizer.physical.mean.shape==(40,)
              and normalizer.physical.eps==.01 and normalizer.physical.until is None,'split normalizer differs')
    d=actor.distribution
    _need(actor.pilot_setting==setting and type(d) is PilotGaussian and d.output_dim==13
          and d.std_type=='scalar' and isinstance(d.std_param,nn.Parameter) and d.std_param.requires_grad
          and d.std_param.shape==(13,) and bool(torch.isfinite(d.std_param).all())
          and tuple(d.std_range)==(1e-6,1e6) and tuple(d.log_std_range)==(math.log(1e-6),math.log(1e6)), 'split Gaussian semantics differ')
    _need(d.gain_entropy_weight==1.,'split entropy treatment differs')
    _need(critic.distribution is None,'split critic must be deterministic')
    mask=getattr(d,'activity_mask',None)
    if mask is not None:
        _need(isinstance(mask,torch.Tensor) and mask.ndim==2 and mask.shape[-1]==13
              and bool((mask[:,:12]==1).all()) and bool(((mask[:,12]==0)|(mask[:,12]==1)).all()),'split live activity mask differs')
